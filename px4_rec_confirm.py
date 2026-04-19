# Prototype (a): post-launch smooth-ramp recovery planner.
# Paired interleaved confirmation on technical: baseline V5.1 vs V51Rec.
# Locked tune: MPC_XY_VEL_P_ACC=6.0, ACC_HOR=10, JERK_AUTO=30.
# Locked common config: max_speed=11, cruise_speed=9, base_blend=1.5.
#
# V51Rec idea:
#   - After leg 0 (LAUNCH) completes, enter RECOVERY state that CAPS the
#     horizontal-speed component of the cmd to (ach_speed + headroom).
#   - Direction / turn / blend logic from baseline V5.1 is preserved (we
#     rescale the horizontal magnitude, not recompute direction).
#   - Recovery auto-exits as soon as ach_speed crosses rec_threshold_frac *
#     cruise_speed, OR after rec_timeout seconds, whichever comes first.
#   - rec_headroom is chosen large enough that PX4 is still accel-saturated
#     (err > cruise/VEL_P_ACC = 1.5) but small enough to meaningfully reshape
#     the command near the drone's actual velocity.
import asyncio, time, math, json, os, sys, statistics
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_rec_confirm.json'
COURSE = 'technical'
TRIAL_TIMEOUT = 150.0
N_PAIRS = 5

REC_PARAMS = dict(
    rec_headroom=2.0,
    rec_threshold_frac=0.85,
    rec_timeout=2.5,
)


def lock_tune():
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0
    B.RACE_PARAMS['MPC_ACC_HOR'] = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX'] = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO'] = 30.0


class V51Rec(B.V51Planner):
    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5,
                 rec_headroom=4.0, rec_threshold_frac=0.85, rec_timeout=3.0):
        super().__init__(max_speed=max_speed, cruise_speed=cruise_speed, base_blend=base_blend)
        self.rec_active = True
        self.rec_headroom = rec_headroom
        self.rec_threshold = cruise_speed * rec_threshold_frac
        self.rec_timeout = rec_timeout
        self.rec_start_time = None

    def plan(self, pos, vel, target, next_gate=None):
        cmd = super().plan(pos, vel, target, next_gate)

        # Recovery only applies after leg 0 (LAUNCH) has completed.
        if (not self.rec_active) or self.is_first_leg:
            return cmd

        # Start the recovery timer at the first plan() call of leg 1
        if self.rec_start_time is None:
            self.rec_start_time = time.time()

        cs = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        rec_elapsed = time.time() - self.rec_start_time

        # Exit conditions: ach speed crosses threshold OR timeout
        if cs >= self.rec_threshold or rec_elapsed > self.rec_timeout:
            self.rec_active = False
            return cmd

        # Clamp horizontal magnitude only; preserve direction and vz
        cap = min(cs + self.rec_headroom, self.cruise_speed)
        if cap < 3.0:
            cap = 3.0
        cmd_h = math.sqrt(cmd.north_m_s ** 2 + cmd.east_m_s ** 2)
        if cmd_h > cap and cmd_h > 1e-3:
            s = cap / cmd_h
            return VelocityNedYaw(cmd.north_m_s * s, cmd.east_m_s * s, cmd.down_m_s, cmd.yaw_deg)
        return cmd


async def run_trial_with_legs(planner, gates, threshold=2.5):
    # Mirror of B.run_trial but with per-leg aggregation for cmd/ach/err.
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
        except Exception as e:
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
    leg_acc = {}  # leg_idx -> {n, cmd, ach, err, t_start, t_end}

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


def mk_planner(label):
    if label == 'BASE':
        return B.V51Planner(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
    elif label == 'REC':
        return V51Rec(max_speed=11.0, cruise_speed=9.0, base_blend=1.5, **REC_PARAMS)
    raise ValueError(label)


async def run_one(label):
    p = mk_planner(label)
    return await run_trial_with_legs(p, B.COURSES[COURSE])


async def main():
    lock_tune()
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] REC_PARAMS -> {REC_PARAMS}', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    time.sleep(1)
    print('[init] starting fresh PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    PAIR = ['BASE', 'REC']
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
        leg1 = next((L for L in ls if L['leg'] == 1), None)
        if leg1:
            l1d = leg1['dur']; l1u = leg1['util']; l1c = leg1['cmd']; l1a = leg1['ach']
            print(f'{st} {tt}s | leg1 dur={l1d}s util={l1u:.1%} cmd={l1c} ach={l1a}', flush=True)
        else:
            print(f'{st} {tt}s', flush=True)
        B.restart_px4()
        await asyncio.sleep(2)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump({'rec_params': REC_PARAMS, 'results': results}, f)
    print(f'\n[wrote] {LOG_PATH}', flush=True)

    # Summary
    def ok(lbl):
        return [r for r in results if r['label'] == lbl and r['completed']]

    base_ok = ok('BASE')
    rec_ok = ok('REC')

    def stat(vals):
        if not vals:
            return None
        return dict(n=len(vals), median=round(statistics.median(vals), 3),
                    mean=round(statistics.mean(vals), 3),
                    sd=round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                    lo=round(min(vals), 3), hi=round(max(vals), 3))

    print('\n' + '=' * 72)
    print('PAIRED RECOVERY PROTOTYPE SUMMARY')
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
    rec_times = [r['time'] for r in rec_ok]
    bs = stat(base_times); rs_ = stat(rec_times)
    print(f'BASE time: {bs}')
    print(f'REC  time: {rs_}')

    if bs and rs_:
        bs_med = bs['median']
        d_med = rs_['median'] - bs_med
        pct = d_med / bs_med * 100
        print(f'DELTA median time (REC-BASE): {d_med:+.3f}s  ({pct:+.1f}%)')

    print('\nPer-leg dur/util (median over OK trials):')
    all_leg_idx = sorted(set([L['leg'] for r in base_ok + rec_ok for L in r['legs']]))
    print('leg   BASE_dur    REC_dur     d_dur  BASE_util   REC_util   d_util')
    for L in all_leg_idx:
        b_dur = leg_field(base_ok, L, 'dur')
        r_dur = leg_field(rec_ok, L, 'dur')
        b_util = leg_field(base_ok, L, 'util')
        r_util = leg_field(rec_ok, L, 'util')
        if not (b_dur and r_dur):
            continue
        bd = statistics.median(b_dur); rd = statistics.median(r_dur)
        bu = statistics.median(b_util); ru = statistics.median(r_util)
        print(f'{L:>3} {bd:>9.3f}s {rd:>9.3f}s {rd - bd:>+7.3f}s {bu:>9.1%} {ru:>9.1%} {(ru - bu) * 100:>+6.1f}pp')

    # Decision rule: leg1 dur drops AND time improves AND no healthy leg regresses by >5% util
    if bs and rs_:
        d_med = rs_['median'] - bs['median']
        b_leg1 = statistics.median(leg_field(base_ok, 1, 'dur')) if leg_field(base_ok, 1, 'dur') else None
        r_leg1 = statistics.median(leg_field(rec_ok, 1, 'dur')) if leg_field(rec_ok, 1, 'dur') else None
        healthy = [3, 4, 6, 7, 8, 11]
        regression = False
        for L in healthy:
            bu = leg_field(base_ok, L, 'util')
            ru = leg_field(rec_ok, L, 'util')
            if bu and ru and (statistics.median(ru) - statistics.median(bu)) < -0.05:
                regression = True
                print(f'  REGRESSION on leg {L}: util dropped >5pp')
        print('\nDECISION:')
        print(f'  leg1 dur drop: {b_leg1 - r_leg1 if b_leg1 and r_leg1 else None}s')
        print(f'  total delta: {d_med:+.3f}s')
        print(f'  healthy regression: {regression}')
        if (b_leg1 and r_leg1 and r_leg1 < b_leg1) and (d_med < -0.1) and (not regression):
            print('  ==> LOCK recovery planner')
        else:
            print('  ==> KEEP baseline, refine recovery headroom / threshold')


if __name__ == '__main__':
    asyncio.run(main())
