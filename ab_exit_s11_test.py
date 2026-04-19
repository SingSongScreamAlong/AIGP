# ab_exit_s11.py â€” Session 11 EXIT-phase A/B: graduated post-turn cmd ramp.
# Control: current baseline (no EXIT phase).
# Treatment: EXIT phase after turns > 30deg, 0.3s hold with trackable cmd.
# Technical first (10 pairs), mixed conditional (10 pairs).
# Uses v2 decision rule.
import asyncio, os, sys, time, math, json
from statistics import median, stdev

sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
import bench

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
RESULT = os.path.join(LOGS, 'ab_exit_s11.json')
N_TECH = 10
N_MIXED = 10
TRIAL_TIMEOUT = 200.0
POST_RESTART_WAIT = 2.0

# EXIT phase parameters
EXIT_TA_THRESH = 0.524   # 30deg
EXIT_DURATION = 0.3      # seconds
EXIT_FLOOR_FRAC = 0.85   # min desired = cruise * this
EXIT_CMD_MARKUP = 1.15   # cmd = desired * this (vs util-based ~1.28)

ARMS = [
    {'name': 'control_baseline', 'exit_enabled': False},
    {'name': 'treatment_exit', 'exit_enabled': True},
]

class V51WithExit(B.V51Planner):
    """V51Planner subclass that adds an EXIT phase after significant turns."""
    def __init__(self, *args, exit_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_enabled = exit_enabled
        self.exit_active = False
        self.exit_start_time = 0.0
        self.last_turn_angle = 0.0
        self._pre_gate_ta = 0.0  # turn angle computed before gate pass

    def on_gate_passed(self, speed):
        super().on_gate_passed(speed)
        if self.exit_enabled and self._pre_gate_ta > EXIT_TA_THRESH:
            self.exit_active = True
            self.exit_start_time = time.time()
        else:
            self.exit_active = False

    def _phase(self, dxy, cs, ta, db):
        if self.exit_enabled and self.exit_active:
            if time.time() - self.exit_start_time < EXIT_DURATION:
                return 'EXIT'
            else:
                self.exit_active = False
        return super()._phase(dxy, cs, ta, db)

    def _smooth(self, target, phase, dt=0.02):
        if phase == 'EXIT':
            mr = 8.0; md = mr * dt
            if target > self.last_cmd_speed:
                s = min(target, self.last_cmd_speed + md)
            else:
                s = max(target, self.last_cmd_speed - md * 1.5)
            self.last_cmd_speed = s
            return s
        return super()._smooth(target, phase, dt)

    def plan(self, pos, vel, target, next_gate=None):
        # Compute turn angle for current approach (needed to trigger EXIT on gate pass)
        dx, dy = target[0]-pos[0], target[1]-pos[1]
        dxy = math.sqrt(dx*dx+dy*dy)
        if next_gate and dxy > 0.1:
            nx, ny = next_gate[0]-target[0], next_gate[1]-target[1]
            nd = math.sqrt(nx*nx+ny*ny)
            if nd > 0.1:
                ax, ay = dx/dxy, dy/dxy
                bx, by = nx/nd, ny/nd
                dot = max(-1, min(1, ax*bx+ay*by))
                self._pre_gate_ta = math.acos(dot)
            else:
                self._pre_gate_ta = 0.0
        else:
            self._pre_gate_ta = 0.0

        # Call parent plan for normal behavior
        result = super().plan(pos, vel, target, next_gate)

        # If EXIT phase is active, override the speed magnitude.
        # super().plan() already called _smooth('EXIT') which used EXIT rate (8 m/sÂ²)
        # on the util-inflated cmd (~11.5). We want to cap the smoothed output to a
        # trackable level: exit_desired * EXIT_CMD_MARKUP (~10.0 instead of ~11.5).
        if self.exit_enabled and self.exit_active and (time.time() - self.exit_start_time < EXIT_DURATION):
            exit_desired = max(self.prev_gate_speed, self.cruise_speed * EXIT_FLOOR_FRAC)
            exit_desired = min(exit_desired, self.px4_speed_ceiling)
            exit_cmd_cap = exit_desired * EXIT_CMD_MARKUP
            exit_cmd_cap = min(exit_cmd_cap, self.max_speed)
            # Parent already smoothed to some value stored in last_cmd_speed.
            # Clamp it down to our cap if it exceeds it.
            sp = min(self.last_cmd_speed, exit_cmd_cap)
            self.last_cmd_speed = sp  # keep smooth state consistent
            # Rescale parent's velocity to our capped speed
            vn, ve, vd = result.north_m_s, result.east_m_s, result.down_m_s
            h_spd = math.sqrt(vn*vn + ve*ve)
            if h_spd > 0.01:
                scale = sp / h_spd
                vn *= scale; ve *= scale
            result = VelocityNedYaw(vn, ve, vd, result.yaw_deg)

        return result


async def run_one(course, arm):
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')

    diag = await bench.wait_healthy(drone, timeout=15.0)
    if not all([diag['connected'], diag['telemetry'], diag['armable']]):
        return {'completed': False, 'error': 'unhealthy', 'health': diag}

    for n, v in B.RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0,0,0]; vel = [0,0,0]
    stop_pl = asyncio.Event()
    async def pl():
        nonlocal pos, vel
        try:
            async for pv in drone.telemetry.position_velocity_ned():
                pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
                vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
                if stop_pl.is_set(): return
        except asyncio.CancelledError:
            return
    pl_task = asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    gates = B.COURSES[course]
    planner = V51WithExit(max_speed=12.0, cruise_speed=9.0, base_blend=1.5,
                          exit_enabled=arm['exit_enabled'])
    planner.cold_ramp_seed = 0.0
    planner.z_gate_alt_frac = 0.0
    planner.tb_enabled = True
    planner.tb_vz_thresh = 1.0
    planner.px4_speed_ceiling = 9.5

    try:
        await drone.action.arm()
        alt = abs(gates[0][2])
        await drone.action.set_takeoff_altitude(alt)
        await drone.action.takeoff()
        ws = time.time()
        while True:
            if time.time()-ws >= 10.0: break
            if abs(pos[2]) >= 0.95*alt and abs(vel[2]) < 0.3: break
            await asyncio.sleep(0.05)
        t_offboard = time.time() - ws

        await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
        await drone.offboard.start()

        leg0_t = None
        max_ach = 0.0
        sustain_speeds = []
        gi = 0; t0 = time.time(); dt = 1/50
        while gi < len(gates):
            if time.time()-t0 > 90: break
            g = gates[gi]
            d = math.sqrt((g[0]-pos[0])**2+(g[1]-pos[1])**2+(g[2]-pos[2])**2)
            if d < 2.5:
                if gi == 0 and leg0_t is None:
                    leg0_t = time.time()-t0
                gspd = math.sqrt(vel[0]**2 + vel[1]**2)
                planner.on_gate_passed(gspd)
                gi += 1
                continue
            ng = gates[gi+1] if gi+1 < len(gates) else None
            cmd = planner.plan(pos, vel, g, ng)
            ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)
            if ach_spd > max_ach: max_ach = ach_spd
            if gi >= 2 and gi < len(gates)-1:
                sustain_speeds.append(ach_spd)
            await drone.offboard.set_velocity_ned(cmd)
            await asyncio.sleep(dt)
        total = time.time() - t0
        completed = gi >= len(gates)

        try: await drone.offboard.stop()
        except: pass
        try: await drone.action.land()
        except: pass

        sustain_med = round(median(sustain_speeds), 3) if sustain_speeds else None

        return {
            'arm': arm['name'], 'course': course,
            'completed': completed,
            'lap_time': round(total, 3),
            'leg0_time': round(leg0_t, 3) if leg0_t is not None else None,
            't_offboard': round(t_offboard, 3),
            'max_spd': round(max_ach, 3),
            'sustain_med_spd': sustain_med,
            'health': diag,
        }
    finally:
        stop_pl.set()
        pl_task.cancel()
        try: await pl_task
        except (asyncio.CancelledError, Exception): pass

async def trial(course, trial_idx, arm, results):
    tag = f'{course} t{trial_idx} {arm["name"]}'
    print(f'[{tag}] hardened restart...', flush=True)
    restart_ok = False
    for attempt in range(2):
        try:
            bench.hardened_restart(B)
            restart_ok = True
            break
        except Exception as e:
            if attempt == 0:
                print(f'  -> RESTART FAILED (attempt 1), retrying: {e}', flush=True)
                bench.kill_stack()
                await asyncio.sleep(3.0)
            else:
                r = {'trial': trial_idx, 'arm': arm['name'], 'course': course,
                     'completed': False, 'error': f'restart_failed_2x: {e}',
                     'restart_retried': True, 'bench_failure': True}
                results.append(r)
                bench.atomic_write_json(RESULT, results)
                print(f'  -> RESTART FAILED (attempt 2): {e}', flush=True)
                return
    await asyncio.sleep(POST_RESTART_WAIT)

    try:
        r = await asyncio.wait_for(run_one(course, arm), timeout=TRIAL_TIMEOUT)
    except asyncio.TimeoutError:
        r = {'completed': False, 'error': 'timeout', 'arm': arm['name'], 'course': course}
        bench.kill_stack()
    except Exception as e:
        r = {'completed': False, 'error': f'exception: {type(e).__name__}: {e}',
             'arm': arm['name'], 'course': course}
        bench.kill_stack()

    if r.get('error') in ('unhealthy',):
        r['bench_failure'] = True

    r['trial'] = trial_idx
    results.append(r)
    bench.atomic_write_json(RESULT, results)
    print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
          f'sus={r.get("sustain_med_spd")} done={r.get("completed")} err={r.get("error")}', flush=True)

def evaluate_v2(results, course, n_pairs):
    ctl_name = 'control_baseline'
    tre_name = 'treatment_exit'
    ctl_all = [r for r in results if r['course']==course and r['arm']==ctl_name and r.get('trial',0) >= 1]
    tre_all = [r for r in results if r['course']==course and r['arm']==tre_name and r.get('trial',0) >= 1]

    bench_fails = [r for r in ctl_all + tre_all if r.get('bench_failure')]
    total_trials = len(ctl_all) + len(tre_all)
    bench_pct = len(bench_fails) / max(total_trials, 1)
    r1 = bench_pct <= 0.20

    bf_ctl = [r for r in ctl_all if r.get('bench_failure')]
    bf_tre = [r for r in tre_all if r.get('bench_failure')]
    skew_flag = len(bench_fails) > 0 and (len(bf_ctl) == 0 or len(bf_tre) == 0)

    ctl_flown = [r for r in ctl_all if not r.get('bench_failure')]
    tre_flown = [r for r in tre_all if not r.get('bench_failure')]
    ctl_done = [r for r in ctl_flown if r.get('completed')]
    tre_done = [r for r in tre_flown if r.get('completed')]

    min_done = 8
    r3 = (len(tre_done) >= len(ctl_done)) and (len(tre_done) >= min_done)

    cl = [r['lap_time'] for r in ctl_done if r.get('lap_time') is not None]
    tl = [r['lap_time'] for r in tre_done if r.get('lap_time') is not None]
    d_lap = (median(tl) - median(cl)) if (cl and tl) else None
    r4 = d_lap is not None and d_lap <= 0.0

    cs = [r['sustain_med_spd'] for r in ctl_done if r.get('sustain_med_spd') is not None]
    ts = [r['sustain_med_spd'] for r in tre_done if r.get('sustain_med_spd') is not None]

    lines = []
    lines.append(f'  bench failures: {len(bench_fails)}/{total_trials} ({bench_pct:.0%})')
    if skew_flag:
        lines.append(f'  ARM SKEW FLAG: ctl={len(bf_ctl)} tre={len(bf_tre)} bench failures')
    lines.append(f'  Rule 1 (bench <=20%): {"PASS" if r1 else "FAIL (INCONCLUSIVE)"}')
    lines.append(f'  flown: ctl {len(ctl_done)}/{len(ctl_flown)}  tre {len(tre_done)}/{len(tre_flown'}')
    if cl and tl:
        lines.append(f'  lap median: ctl {median(cl):.3f}  tre {median(tl):.3f}  d={d_lap:+.3f}')
        if len(cl) > 1: lines.append(f'  lap stdev:  ctl {stdev(cl):.3f}  tre {stdev(tl):.3f}')
    if cs and ts:
        lines.append(f'  sustain_spd: ctl {median(cs):.3f}  tre {median(ts):.3f}  d={median(ts)-median(cs):+.3f}')
    c0 = [r['leg0_time'] for r in ctl_done if r.get('leg0_time') is not None]
    t0 = [r['leg0_time'] for r in tre_done if r.get('leg0_time') is not None]
    if c0 and t0:
        lines.append(f'  leg0 median: ctl {median(c0):.3f}  tre {median(t0):.3f}  d={median(t0)-median(c0):+.3f}')
    lines.append(f'  Rule 3 (flown completion): {"PASS" if r3 else "FAIL"}')
    lines.append(f'  Rule 4 (lap delta <= 0): {"PASS" if r4 else "FAIL"}')

    if not r1:
        verdict = 'INCONCLUSIVE'
    elif r3 and r4:
        verdict = 'ADOPT'
    else:
        verdict = 'REJECT'
    lines.append(f'  --> {verdict}')
    return verdict, '\n'.join(lines)

async def main():
    bench.acquire_singleton('exit11')
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] Session 11 EXIT-phase A/B', flush=True)
    print(f'[init] EXIT params: ta_thresh={EXIT_TA_THRESH:.3f}rad({math.degrees(EXIT_TA_THRESH):.0f}deg) '
          f'dur={EXIT_DURATION}s floor={EXIT_FLOOR_FRAC} markup={EXIT_CMD_MARKUP}', flush=True)
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ARMS = {[a["name"] for a in ARMS]}', flush=True)
    print(f'[init] result file: {RESULT}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    print('[preflight] 1 control trial', flush=True)
    await trial('technical', 0, ARMS[0], results)
    if not results[0].get('completed'):
        print(f'[preflight] FAILED: {results[0].get("error")}; aborting', flush=True)
        return
    print(f'[preflight] OK lap={results[0].get("lap_time"¹}', flush=True)

    print(f'\n=== TECHNICAL ({N_TECH} pairs) ===', flush=True)
    for t in range(1, N_TECH+1):
        for arm in ARMS:
            await trial('technical', t, arm, results)
    tech_verdict, tech_detail = evaluate_v2(results, 'technical', N_TECH)
    print(f'\n--- TECHNICAL DECISION ---\n{tech_detail}', flush=True)

    if tech_verdict == 'REJECT':
        print('\nTech¸ical REJECTED. Skipping mixed.', flush=True)
        bench.atomic_write_json(RESULT, results)
        return
    if tech_verdict == 'INCONCLUSIVE':
        print('\nTechnical INCONCLUSIVE (bench issue). Skipping mixed.', flush=True)
        bench.atomic_write_json(RESULT, results)
        return

    print(f'\n=== MIXED ({N_MIXED} pairs) ===', flush=True)
    for t in range(1, N_MIXED+1):
        for arm in ARMS:
            await trial('mixed', t, arm, results)
    mix_verdict, mix_detail = evaluate_v2(results, 'mixed', N_MIXED)
    print(f'\n--- MIXED DECISION ---\n{mix_detail}', flush=True)

    final = 'ADOPT' if (tech_verdict == 'ADOPT' and mix_verdict == 'ADOPT') else mix_verdict
    print(f'\n=== VERDICT: {final} EXIT phase ===', flush=True)

    print('Yî=== PERTQ T RIAEˆ!===', flush=True)
    for course in ('technical', 'mixed'):
        trials = sorted({r['trial'] for r in results if r['course']==course and r['trial'^>1})
        if not trials: continue
        print(f'\n--- {course.upper()} ---')
        for t in trials:
            rc = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='control_baseline']
            rt = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='treatment_exit']
            c = rc[0] if rc else {}; tr = rt[0] if rt else {}
            cl = c.get('lap_time'); tl = tr.get('lap_time')
            delta = f'{tl-cl:+.3f}' if (cl is not None and tl is not None) else 'n/a'
            print(f'  t{t}  ctl={cl}({c.get("completed")})  tre={tl}({tr.get("completed")})  d={delta}  '
                  f'sus_c={c.get("sustain_med_spd")} sus_t={tr.get("sustain_med_spd")}')
    bench.atomic_write_json(RESULT, results)

if __name__ == '__main__':
    asyncio.run(main()