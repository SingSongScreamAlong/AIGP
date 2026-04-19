# Leg-0 diagnostic probe.
# One fresh technical trial. Baseline V5.1 planner, locked tune.
# Captures a 50 Hz time series covering the window from arm through ~1 s
# after gate 0 pass, so we can clearly see:
#   - whether vz is still active during leg 0 (Case 1: vertical settling theft)
#   - whether horizontal acceleration is tiny despite cmd being large
#     (Case 2: PX4 state-transition lockout)
#   - whether cmd setpoints fail to reach the controller in early offboard
#     (Case 3: offboard handoff glitch)
#
# Fields per sample:
#   t_arm:  seconds since arm
#   phase:  'pre_offboard' | planner phase label ('LAUNCH','SUSTAIN',...)
#   pos_n, pos_e, pos_d
#   vel_n, vel_e, vel_d
#   ach_h = sqrt(vel_n^2 + vel_e^2)
#   cmd_vn, cmd_ve, cmd_vd (last cmd issued; 0 before offboard)
#   cmd_h = sqrt(cmd_vn^2 + cmd_ve^2)
#   err_h = cmd_h - ach_h
#
# Plus event timestamps (all seconds from arm):
#   t_offboard_start
#   t_gate0_pass
#   t_probe_end
import asyncio, time, math, json, os, sys
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_leg0_probe.json'
COURSE = 'technical'
LOG_DT = 0.02  # 50 Hz
POST_GATE0_LOG_S = 1.0  # keep logging this long after gate 0 pass


def lock_tune():
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0
    B.RACE_PARAMS['MPC_ACC_HOR'] = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX'] = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO'] = 30.0


async def probe_trial(gates, threshold=2.5):
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

    # Health / arm retries
    for _ in range(20):
        try:
            async for h in drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok:
                    break
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    armed_ok = False
    for _ in range(10):
        try:
            await drone.action.arm()
            armed_ok = True
            break
        except Exception:
            await asyncio.sleep(1.0)
    if not armed_ok:
        raise RuntimeError('arm failed after retries')

    arm_ts = time.time()

    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    # Start a background logger that samples at 50 Hz immediately after arm.
    samples = []
    stop_logger = {'flag': False}
    last_cmd = {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'phase': 'pre_offboard'}
    events = {
        'arm_ts': 0.0,
        't_offboard_start': None,
        't_gate0_pass': None,
        't_probe_end': None,
    }

    async def logger():
        while not stop_logger['flag']:
            t_arm = time.time() - arm_ts
            ach_h = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            cmd_h = math.sqrt(last_cmd['vn'] ** 2 + last_cmd['ve'] ** 2)
            samples.append({
                't_arm': round(t_arm, 4),
                'phase': last_cmd['phase'],
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
                'err_h': round(cmd_h - ach_h, 4),
            })
            await asyncio.sleep(LOG_DT)

    logger_task = asyncio.ensure_future(logger())

    # Native PX4 takeoff settle time
    await asyncio.sleep(4)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()
    events['t_offboard_start'] = round(time.time() - arm_ts, 4)

    planner = B.V51Planner(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

    gi = 0
    gate0_pass_wall = None
    dt = 1 / 50
    while True:
        now = time.time()
        # Terminate logging criterion: gate 0 crossed + POST_GATE0_LOG_S
        if gate0_pass_wall is not None and (now - gate0_pass_wall) >= POST_GATE0_LOG_S:
            break
        # Safety timeout
        if (now - arm_ts) > 20.0:
            break
        g = gates[gi]
        d = math.sqrt((g[0] - pos[0]) ** 2 + (g[1] - pos[1]) ** 2 + (g[2] - pos[2]) ** 2)
        if d < threshold:
            if gi == 0 and gate0_pass_wall is None:
                gate0_pass_wall = now
                events['t_gate0_pass'] = round(now - arm_ts, 4)
            gspd = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            planner.on_gate_passed(gspd)
            gi += 1
            if gi >= len(gates):
                break
            continue
        ng = gates[gi + 1] if gi + 1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        last_cmd['vn'] = cmd.north_m_s
        last_cmd['ve'] = cmd.east_m_s
        last_cmd['vd'] = cmd.down_m_s
        # Determine phase tag. V5.1 computes phase internally; replicate quickly.
        dxy = math.sqrt((g[0] - pos[0]) ** 2 + (g[1] - pos[1]) ** 2)
        cs = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        ta = 0.0
        if ng:
            nx, ny = ng[0] - g[0], ng[1] - g[1]
            nd = math.sqrt(nx * nx + ny * ny)
            if nd > 0.1 and dxy > 0.1:
                ax, ay = (g[0] - pos[0]) / dxy, (g[1] - pos[1]) / dxy
                bx, by = nx / nd, ny / nd
                dp = max(-1, min(1, ax * bx + ay * by))
                ta = math.acos(dp)
        db = planner.base_blend + 0.25 * cs + (ta / math.pi) * 2.0
        last_cmd['phase'] = planner._phase(dxy, cs, ta, db)
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    events['t_probe_end'] = round(time.time() - arm_ts, 4)
    stop_logger['flag'] = True
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
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    return {'events': events, 'samples': samples, 'gate0': list(gates[0])}


async def main():
    lock_tune()
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    time.sleep(1)
    print('[init] starting fresh PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    # Broad exception wrapper — harness hygiene: don't lose the whole run.
    try:
        result = await asyncio.wait_for(probe_trial(B.COURSES[COURSE]), timeout=120.0)
    except Exception as e:
        print(f'[probe] exception: {type(e).__name__}: {e}', flush=True)
        os.system('pkill -9 -f mavsdk_server 2>/dev/null')
        os.system('pkill -9 -f "bin/px4" 2>/dev/null')
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, 'w') as f:
            json.dump({'error': f'{type(e).__name__}: {e}'}, f)
        return

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump(result, f)
    print(f'[probe] wrote {LOG_PATH}', flush=True)
    ev = result['events']
    print(f'[probe] events: {ev}', flush=True)
    print(f'[probe] samples: {len(result["samples"])}', flush=True)

    # Quick in-process summary
    samples = result['samples']
    ob = ev.get('t_offboard_start')
    g0 = ev.get('t_gate0_pass')
    if ob and g0:
        pre = [s for s in samples if s['t_arm'] < ob]
        leg0 = [s for s in samples if ob <= s['t_arm'] < g0]
        post = [s for s in samples if g0 <= s['t_arm']]
        def avg(xs, k):
            vs = [x[k] for x in xs]
            return sum(vs) / len(vs) if vs else 0.0
        def mx(xs, k):
            vs = [x[k] for x in xs]
            return max(vs) if vs else 0.0
        def mn(xs, k):
            vs = [x[k] for x in xs]
            return min(vs) if vs else 0.0
        print('\n--- PRE-OFFBOARD (native takeoff) ---')
        print(f'  n={len(pre)}  dur={ob:.2f}s')
        print(f'  pos_d range: [{mn(pre,"pos_d"):.2f}, {mx(pre,"pos_d"):.2f}]  (target={samples and result["gate0"][2]})')
        print(f'  vel_d avg={avg(pre,"vel_d"):.3f}  max_abs={max(abs(mn(pre,"vel_d")),abs(mx(pre,"vel_d"))):.3f}')
        print(f'  ach_h  avg={avg(pre,"ach_h"):.3f}  max={mx(pre,"ach_h"):.3f}')
        print('\n--- LEG 0 (offboard → gate0) ---')
        print(f'  n={len(leg0)}  dur={g0-ob:.3f}s')
        print(f'  pos_d first/last: {leg0[0]["pos_d"] if leg0 else None} / {leg0[-1]["pos_d"] if leg0 else None}')
        print(f'  vel_d avg={avg(leg0,"vel_d"):.3f}  max_abs={max(abs(mn(leg0,"vel_d")),abs(mx(leg0,"vel_d"))):.3f}')
        print(f'  ach_h  avg={avg(leg0,"ach_h"):.3f}  max={mx(leg0,"ach_h"):.3f}  final={leg0[-1]["ach_h"] if leg0 else None}')
        print(f'  cmd_h  avg={avg(leg0,"cmd_h"):.3f}  max={mx(leg0,"cmd_h"):.3f}')
        print(f'  err_h  avg={avg(leg0,"err_h"):.3f}  max={mx(leg0,"err_h"):.3f}')
        # Horizontal accel estimate over leg 0: (ach_h final - ach_h first) / dur
        if len(leg0) >= 2:
            da = leg0[-1]['ach_h'] - leg0[0]['ach_h']
            dt_ = leg0[-1]['t_arm'] - leg0[0]['t_arm']
            print(f'  horiz accel estimate (end-start/dur): {da/dt_ if dt_>0 else 0:.3f} m/s^2')
        # Instantaneous sample of peak horizontal accel using finite diff
        peak_a = 0.0
        for i in range(1, len(leg0)):
            dt_i = leg0[i]['t_arm'] - leg0[i-1]['t_arm']
            if dt_i > 0:
                a = (leg0[i]['ach_h'] - leg0[i-1]['ach_h']) / dt_i
                if a > peak_a:
                    peak_a = a
        print(f'  horiz accel peak (finite diff): {peak_a:.3f} m/s^2')


if __name__ == '__main__':
    asyncio.run(main())
