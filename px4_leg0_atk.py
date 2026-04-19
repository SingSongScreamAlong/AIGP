# Prototype: leg-0 takeoff attack.
# Hypothesis: the V5.1 baseline's LAUNCH phase is artificially throttling cmd
# during leg 0 via (a) a quadratic ramp that holds `desired` at 2.5 m/s for
# most of the first second, and (b) a LAUNCH-specific _smooth rate of 6.0 m/s^2
# (vs SUSTAIN's 12). Both are below what the drone's acceleration envelope can
# absorb, mirroring the (a)/recovery-clamp failure in reverse.
#
# Minimal one-line intervention: set px4_hover_ramp_time to a tiny value so
# _phase() essentially skips 'LAUNCH' and treats leg 0 as SUSTAIN from frame 2,
# which gives desired=cruise_speed immediately and _smooth rate=12.
#
# Success metric (decision rule):
#   - leg 0 duration drops
#   - leg 0 achieved speed rises materially
#   - total lap time improves by > 0.1s
#   - no regression on legs 1, 3, 4, 6, 7, 8, 11 (util drop > 5pp)
# If all true => LOCK; else KEEP baseline.
import asyncio, time, math, json, os, sys, statistics
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_leg0_atk.json'
COURSE = 'technical'
TRIAL_TIMEOUT = 150.0
N_PAIRS = 5

ATK_PARAMS = dict(
    hover_ramp_time=0.01,  # effectively skip LAUNCH phase
)


def lock_tune():
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0
    B.RACE_PARAMS['MPC_ACC_HOR'] = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX'] = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO'] = 30.0


def mk_planner(label):
    p = B.V51Planner(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
    if label == 'ATK':
        p.px4_hover_ramp_time = ATK_PARAMS['hover_ramp_time']
    return p


async def run_trial_with_legs(planner, gates, threshold=2.5):
    # Mirror of B.run_trial but with per-leg aggregation for cmd/ach/err,
    # and health/arm retry loop.
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break
    for n, v in B.RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0, 0, 0]; vel = [0, 0, 0]

    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)
    # Wait for health / armable with retries
    for attempt in range(20):
        try:
            async for h in drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok:
                    break
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    armed_ok = False
    for attempt in range(10):
        try:
            await drone.action.arm()
            armed_ok = True
            break
        except Exception:
            await asyncio.sleep(1.0)
    if not armed_ok:
        raise RuntimeError('arm failed after retries')

    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()
    await asyncio.sleep(4)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    gi = 0; gt = []; t0 = time.time(); dt = 1 / 50
    leg_acc = {}

    while gi < len(gates):
        if time.time() - t0 > 90:
            print(f'    TIMEOUT@gate{gi + 1}')
            break
        g = gates[gi]
        d = math.sqrt((g[0] - pos[0]) ** 2 + (g[1] - pos[1]) ** 2 + (g[2] - pos[2]) ** 2)
        if d < threshold:
            gt.append(time.time() - t0)
            gspd = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            planner.on_gate_passed(gspd)
            gi += 1
            continue
        ng = gates[gi + 1] if gi + 1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        cspd = math.sqrt(cmd.north_m_s ** 2 + cmd.east_m_s ** 2)
        aspd = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        now = time.time() - t0
        la = leg_acc.setdefault(gi, {'n': 0, 'cmd': 0.0, 'ach': 0.0, 'err': 0.0, 't_start': now, 't_end': now})
        la['n'] += 1
        la['cmd'] += cspd
        la['ach'] += aspd
        la['err'] += (cspd - aspd)
        la['t_end'] = now
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land(); await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    splits = [round(gt[i] - (gt[i - 1] if i > 0 else 0), 3) for i in range(len(gt))]
    legs = []
    for L in sorted(leg_acc.keys()):
        v = leg_acc[L]
        n = v['n']
        cmd_avg = v['cmd'] / n if n else 0
        ach_avg = v['ach'] / n if n else 0
        err_avg = v['err'] / n if n else 0
        util = ach_avg / cmd_avg if cmd_avg > 0 else 0
        dur = v['t_end'] - v['t_start']
        legs.append(dict(leg=L, n=n, dur=round(dur, 3), cmd=round(cmd_avg, 3),
                         ach=round(ach_avg, 3), err=round(err_avg, 3), util=round(util, 4)))

    return {
        'completed': gi >= len(gates),
        'time': round(total, 3),
        'gates_passed': gi,
        'splits': splits,
        'legs': legs,
    }


async def run_one(label):
    p = mk_planner(label)
    return await run_trial_with_legs(p, B.COURSES[COURSE])


async def main():
    lock_tune()
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ATK_PARAMS -> {ATK_PARAMS}', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    time.sleep(1)
    print('[init] starting fresh PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    PAIR = ['BASE', 'ATK']
    seq = []
    for _ in range(N_PAIRS):
        seq.extend(PAIR)

    results = []
    for idx, label in enumerate(seq):
        print(f'[trial {idx+1}/{len(seq)}  {label}]... ', end='', flush=True)
        try:
            r = await asyncio.wait_for(run_one(label), timeout=TRIAL_TIMEOUT)
        except asyncio.TimeoutError:
            print('HARNESS TIMEOUT', flush=True)
            os.system('pkill -9 -f mavsdk_server 2>/dev/null')
            os.system('pkill -9 -f ' + chr(39) + 'bin/px4' + chr(39) + ' 2>/dev/null')
            r = {'completed': False, 'time': 90.0, 'gates_passed': 0, 'splits': [], 'legs': [], 'harness_timeout': True}
        r['label'] = label
        r['trial_idx'] = idx
        results.append(r)
        gp = r['gates_passed']; tt = r['time']
        st = 'OK' if r['completed'] else f'FAIL@{gp}'
        ls = r.get('legs', [])
        leg0 = next((L for L in ls if L['leg'] == 0), None)
        if leg0:
            l0d = leg0['dur']; l0u = leg0['util']; l0c = leg0['cmd']; l0a = leg0['ach']
            print(f'{st} {tt}s | leg0 dur={l0d}s util={l0u:.1%} cmd={l0c} ach={l0a}', flush=True)
        else:
            print(f'{st} {tt}s', flush=True)
        B.restart_px4()
        await asyncio.sleep(2)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump({'atk_params': ATK_PARAMS, 'results': results}, f)
    print(f'\n[wrote] {LOG_PATH}', flush=True)

    def ok(lbl):
        return [r for r in results if r['label'] == lbl and r['completed']]

    base_ok = ok('BASE')
    atk_ok = ok('ATK')

    def stat(vals):
        if not vals:
            return None
        return dict(n=len(vals), median=round(statistics.median(vals), 3),
                    mean=round(statistics.mean(vals), 3),
                    sd=round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                    lo=round(min(vals), 3), hi=round(max(vals), 3))

    print('\n' + '=' * 72)
    print('PAIRED LEG-0 ATTACK SUMMARY')
    print('=' * 72)

    def leg_field(rs, leg_idx, field):
        out = []
        for r in rs:
            for L in r['legs']:
                if L['leg'] == leg_idx:
                    out.append(L[field])
                    break
        return out

    base_times = [r['time'] for r in base_ok]
    atk_times = [r['time'] for r in atk_ok]
    bs = stat(base_times); at = stat(atk_times)
    print(f'BASE time: {bs}')
    print(f'ATK  time: {at}')

    if bs and at:
        bs_med = bs['median']
        d_med = at['median'] - bs_med
        pct = d_med / bs_med * 100
        print(f'DELTA median time (ATK-BASE): {d_med:+.3f}s  ({pct:+.1f}%)')

    print('\nPer-leg dur/util/cmd/ach (median over OK trials):')
    all_leg_idx = sorted(set([L['leg'] for r in base_ok + atk_ok for L in r['legs']]))
    print('leg   BASE_dur    ATK_dur     d_dur  BASE_util   ATK_util   d_util  BASE_ach ATK_ach')
    for L in all_leg_idx:
        b_dur = leg_field(base_ok, L, 'dur')
        r_dur = leg_field(atk_ok, L, 'dur')
        b_util = leg_field(base_ok, L, 'util')
        r_util = leg_field(atk_ok, L, 'util')
        b_ach = leg_field(base_ok, L, 'ach')
        r_ach = leg_field(atk_ok, L, 'ach')
        if not (b_dur and r_dur):
            continue
        bd = statistics.median(b_dur); rd = statistics.median(r_dur)
        bu = statistics.median(b_util); ru = statistics.median(r_util)
        ba = statistics.median(b_ach); ra = statistics.median(r_ach)
        print(f'{L:>3} {bd:>9.3f}s {rd:>9.3f}s {rd - bd:>+7.3f}s {bu:>9.1%} {ru:>9.1%} {(ru - bu) * 100:>+6.1f}pp  {ba:>6.2f}  {ra:>6.2f}')

    if bs and at:
        d_med = at['median'] - bs['median']
        b_leg0_dur = statistics.median(leg_field(base_ok, 0, 'dur')) if leg_field(base_ok, 0, 'dur') else None
        r_leg0_dur = statistics.median(leg_field(atk_ok, 0, 'dur')) if leg_field(atk_ok, 0, 'dur') else None
        b_leg0_ach = statistics.median(leg_field(base_ok, 0, 'ach')) if leg_field(base_ok, 0, 'ach') else None
        r_leg0_ach = statistics.median(leg_field(atk_ok, 0, 'ach')) if leg_field(atk_ok, 0, 'ach') else None
        healthy = [1, 3, 4, 6, 7, 8, 11]
        regression = False
        for L in healthy:
            bu = leg_field(base_ok, L, 'util')
            ru = leg_field(atk_ok, L, 'util')
            if bu and ru and (statistics.median(ru) - statistics.median(bu)) < -0.05:
                regression = True
                print(f'  REGRESSION on leg {L}: util dropped >5pp')
        print('\nDECISION:')
        print(f'  leg0 dur drop: {(b_leg0_dur - r_leg0_dur) if b_leg0_dur and r_leg0_dur else None}s')
        print(f'  leg0 ach rise: {(r_leg0_ach - b_leg0_ach) if b_leg0_ach and r_leg0_ach else None} m/s')
        print(f'  total delta: {d_med:+.3f}s')
        print(f'  healthy regression: {regression}')
        if (b_leg0_dur and r_leg0_dur and r_leg0_dur < b_leg0_dur) and (d_med < -0.1) and (not regression):
            print('  ==> LOCK leg-0 attack')
        else:
            print('  ==> KEEP baseline, iterate attack')


if __name__ == '__main__':
    asyncio.run(main())
