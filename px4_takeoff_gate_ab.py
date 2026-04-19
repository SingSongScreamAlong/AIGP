# Takeoff gate A/B:
#   A: baseline  -> sleep(4) after action.takeoff(), then offboard.start()
#   B: gated     -> poll telemetry until abs(pos_d) >= 0.95 * target_alt AND
#                   abs(vel_d) < 0.3, with a hard safety timeout of 10 s.
#
# Paired/interleaved 5 pairs = 10 trials. Drift neutralized.
# Archives raw per-sample leg-0 trace for pair 3 (one representative A, one B).
import asyncio, time, math, json, os, sys, statistics
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_takeoff_gate_ab.json'
TRACE_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_takeoff_gate_ab_traces.json'
COURSE = 'technical'
PAIRS = 5
PER_TRIAL_TIMEOUT = 150.0
LAP_METRIC_S = 15.0
GATE_THRESH = 2.5
LOG_DT = 0.02  # 50 Hz trace sampling for archived pair
ARCHIVE_PAIR_IDX = 2  # zero-indexed (third pair)

# Altitude-settle gate params (B)
GATE_ALT_FRAC = 0.95
GATE_VZ_MAX = 0.3
GATE_TIMEOUT = 10.0

# Locked tune
def lock_tune():
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0
    B.RACE_PARAMS['MPC_ACC_HOR'] = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX'] = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO'] = 30.0


async def run_trial(label, archive_trace, gates):
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    for n, v in B.RACE_PARAMS.items():
        try:
            await drone.param.set_param_float(n, v)
        except Exception:
            pass

    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]

    async def tel():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]

    asyncio.ensure_future(tel())
    await asyncio.sleep(0.5)

    for _ in range(20):
        try:
            async for h in drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok:
                    break
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    armed = False
    for _ in range(10):
        try:
            await drone.action.arm()
            armed = True
            break
        except Exception:
            await asyncio.sleep(1.0)
    if not armed:
        raise RuntimeError('arm failed after retries')

    arm_ts = time.time()
    target_alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(target_alt)
    await drone.action.takeoff()

    # --------- A/B branch ---------
    if label == 'A':
        await asyncio.sleep(4.0)
        t_offboard_wait = 4.0
        alt_reached = abs(pos[2])
        vz_at_offboard = vel[2]
    else:
        wait_start = time.time()
        while True:
            elapsed = time.time() - wait_start
            if elapsed >= GATE_TIMEOUT:
                break
            if abs(pos[2]) >= GATE_ALT_FRAC * target_alt and abs(vel[2]) < GATE_VZ_MAX:
                break
            await asyncio.sleep(0.05)
        t_offboard_wait = time.time() - wait_start
        alt_reached = abs(pos[2])
        vz_at_offboard = vel[2]

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()
    t0 = time.time()

    planner = B.V51Planner(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

    # Optional trace archive
    samples = []
    trace_on = archive_trace
    stop_flag = {'v': False}
    last_cmd = {'vn': 0.0, 've': 0.0, 'vd': 0.0}

    async def logger():
        while not stop_flag['v']:
            ach_h = math.sqrt(vel[0]**2 + vel[1]**2)
            cmd_h = math.sqrt(last_cmd['vn']**2 + last_cmd['ve']**2)
            samples.append({
                't': round(time.time() - t0, 4),
                'pos_n': round(pos[0], 4),
                'pos_e': round(pos[1], 4),
                'pos_d': round(pos[2], 4),
                'vel_n': round(vel[0], 4),
                'vel_e': round(vel[1], 4),
                'vel_d': round(vel[2], 4),
                'ach_h': round(ach_h, 4),
                'cmd_vn': round(last_cmd['vn'], 4),
                'cmd_ve': round(last_cmd['ve'], 4),
                'cmd_vd': round(last_cmd['vd'], 4),
                'cmd_h': round(cmd_h, 4),
            })
            await asyncio.sleep(LOG_DT)

    logger_task = None
    if trace_on:
        logger_task = asyncio.ensure_future(logger())

    gi = 0
    leg_acc = {}
    prev_gate_ts = t0
    prev_gate_pos = list(pos)
    t_gate0 = None
    lap_complete = False
    dt = 1 / 50

    def leg_add(leg, cmd_h, ach_h):
        d = leg_acc.setdefault(leg, {'n': 0, 'cmd_sum': 0.0, 'ach_sum': 0.0, 'cmd_max': 0.0, 'ach_max': 0.0, 'ach_last': 0.0})
        d['n'] += 1
        d['cmd_sum'] += cmd_h
        d['ach_sum'] += ach_h
        if cmd_h > d['cmd_max']:
            d['cmd_max'] = cmd_h
        if ach_h > d['ach_max']:
            d['ach_max'] = ach_h
        d['ach_last'] = ach_h

    while True:
        now = time.time()
        if (now - t0) > LAP_METRIC_S:
            break
        if gi >= len(gates):
            lap_complete = True
            break
        g = gates[gi]
        d = math.sqrt((g[0]-pos[0])**2 + (g[1]-pos[1])**2 + (g[2]-pos[2])**2)
        if d < GATE_THRESH:
            dur = now - prev_gate_ts
            leg_acc.setdefault(gi, {'n': 0, 'cmd_sum': 0.0, 'ach_sum': 0.0, 'cmd_max': 0.0, 'ach_max': 0.0, 'ach_last': 0.0})['dur'] = dur
            if gi == 0:
                t_gate0 = now - t0
            gsp = math.sqrt(vel[0]**2 + vel[1]**2)
            planner.on_gate_passed(gsp)
            gi += 1
            prev_gate_ts = now
            prev_gate_pos = list(pos)
            continue
        ng = gates[gi+1] if gi+1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        last_cmd['vn'] = cmd.north_m_s
        last_cmd['ve'] = cmd.east_m_s
        last_cmd['vd'] = cmd.down_m_s
        ach_h = math.sqrt(vel[0]**2 + vel[1]**2)
        cmd_h = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        leg_add(gi, cmd_h, ach_h)
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    lap_time = time.time() - t0
    stop_flag['v'] = True
    if logger_task:
        try:
            await logger_task
        except Exception:
            pass

    try:
        await drone.offboard.stop()
    except Exception:
        pass
    try:
        await drone.action.land()
    except Exception:
        pass
    await asyncio.sleep(2)
    try:
        drone._stop_mavsdk_server()
    except Exception:
        pass
    await asyncio.sleep(1)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(1)

    # Post-process leg stats
    legs_out = {}
    for k, d in leg_acc.items():
        if d.get('n', 0) == 0:
            continue
        legs_out[k] = {
            'n': d['n'],
            'cmd_avg': d['cmd_sum']/d['n'],
            'ach_avg': d['ach_sum']/d['n'],
            'cmd_max': d['cmd_max'],
            'ach_max': d['ach_max'],
            'ach_last': d['ach_last'],
            'dur': d.get('dur'),
        }

    result = {
        'label': label,
        'lap_time': round(lap_time, 3),
        'complete': lap_complete,
        'gates_hit': gi,
        't_gate0': round(t_gate0, 3) if t_gate0 is not None else None,
        't_offboard_wait': round(t_offboard_wait, 3),
        'alt_at_offboard': round(alt_reached, 3),
        'vz_at_offboard': round(vz_at_offboard, 3),
        'legs': legs_out,
    }
    if trace_on:
        result['samples'] = samples
    return result


async def run_one(label, archive, gates):
    try:
        return await asyncio.wait_for(run_trial(label, archive, gates), timeout=PER_TRIAL_TIMEOUT)
    except Exception as e:
        print(f'[{label}] exception: {type(e).__name__}: {e}', flush=True)
        os.system('pkill -9 -f mavsdk_server 2>/dev/null')
        os.system('pkill -9 -f "bin/px4" 2>/dev/null')
        await asyncio.sleep(2)
        return {'label': label, 'error': f'{type(e).__name__}: {e}'}


async def main():
    lock_tune()
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    time.sleep(1)

    results = []
    traces = {'A': None, 'B': None}
    gates = B.COURSES[COURSE]

    for pair in range(PAIRS):
        order = ['A', 'B'] if pair % 2 == 0 else ['B', 'A']
        for label in order:
            archive = (pair == ARCHIVE_PAIR_IDX)
            print(f'\n[pair {pair+1}/{PAIRS}] trial {label} (archive={archive})', flush=True)
            B.restart_px4()
            await asyncio.sleep(2)
            r = await run_one(label, archive, gates)
            r['pair'] = pair + 1
            results.append(r)
            if archive and 'samples' in r and traces[label] is None:
                traces[label] = {
                    'pair': pair + 1,
                    'label': label,
                    'lap_time': r['lap_time'],
                    't_gate0': r['t_gate0'],
                    't_offboard_wait': r['t_offboard_wait'],
                    'alt_at_offboard': r['alt_at_offboard'],
                    'vz_at_offboard': r['vz_at_offboard'],
                    'samples': r['samples'],
                }
                # Remove samples from per-trial result to keep main log small
                del r['samples']
            print(f'[pair {pair+1}] {label} lap={r.get("lap_time")} gate0={r.get("t_gate0")} wait={r.get("t_offboard_wait")} alt@ob={r.get("alt_at_offboard")} vz@ob={r.get("vz_at_offboard")}', flush=True)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump({'pairs': PAIRS, 'results': results}, f, indent=2)
    with open(TRACE_PATH, 'w') as f:
        json.dump(traces, f)
    print(f'\n[write] {LOG_PATH}', flush=True)
    print(f'[write] {TRACE_PATH}', flush=True)

    # Summary
    def stat(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return {'n': len(xs), 'med': round(statistics.median(xs), 3), 'mean': round(statistics.mean(xs), 3)}

    A = [r for r in results if r.get('label') == 'A' and 'error' not in r]
    Bres = [r for r in results if r.get('label') == 'B' and 'error' not in r]
    print('\n=== SUMMARY ===')
    print(f'A n={len(A)}  B n={len(Bres)}')
    print(f'lap_time     A {stat([r["lap_time"] for r in A])}   B {stat([r["lap_time"] for r in Bres])}')
    print(f't_gate0      A {stat([r["t_gate0"] for r in A])}   B {stat([r["t_gate0"] for r in Bres])}')
    print(f'alt@offboard A {stat([r["alt_at_offboard"] for r in A])}   B {stat([r["alt_at_offboard"] for r in Bres])}')
    print(f'vz@offboard  A {stat([r["vz_at_offboard"] for r in A])}   B {stat([r["vz_at_offboard"] for r in Bres])}')
    print(f'offboard wait A {stat([r["t_offboard_wait"] for r in A])}   B {stat([r["t_offboard_wait"] for r in Bres])}')

    # Leg0 stats
    def leg_stat(rs, key, field):
        vs = [r['legs'].get(key, {}).get(field) for r in rs if r.get('legs')]
        return stat(vs)
    print(f'leg0 dur     A {leg_stat(A,0,"dur")}   B {leg_stat(Bres,0,"dur")}')
    print(f'leg0 ach_avg A {leg_stat(A,0,"ach_avg")}   B {leg_stat(Bres,0,"ach_avg")}')
    print(f'leg0 ach_last A {leg_stat(A,0,"ach_last")}   B {leg_stat(Bres,0,"ach_last")}')
    print(f'leg1 dur     A {leg_stat(A,1,"dur")}   B {leg_stat(Bres,1,"dur")}')

    # Paired delta
    pairs_by_id = {}
    for r in results:
        if 'error' in r:
            continue
        pairs_by_id.setdefault(r['pair'], {})[r['label']] = r
    deltas = []
    for pid, d in pairs_by_id.items():
        if 'A' in d and 'B' in d:
            deltas.append(d['B']['lap_time'] - d['A']['lap_time'])
    if deltas:
        print(f'paired delta B-A lap: n={len(deltas)} med={round(statistics.median(deltas),3)} mean={round(statistics.mean(deltas),3)}')


if __name__ == '__main__':
    asyncio.run(main())
