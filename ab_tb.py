# ab_tb.py — Session 9 transition_blend A/B.
# Technical first (10 pairs), mixed second (10 pairs, only if technical passes).
# Uses bench.py hardening. All other interventions OFF.
import asyncio, os, sys, time, math, json
from statistics import median, stdev

sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
import bench

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
RESULT = os.path.join(LOGS, 'ab_tb_s9.json')
N_TECH = 10
N_MIXED = 10
TRIAL_TIMEOUT = 200.0
POST_RESTART_WAIT = 2.0

ARMS = [
    {'name': 'control_tb_off', 'tb': False},
    {'name': 'treatment_tb',   'tb': True},
]
TB_VZ_THRESH = 1.0

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
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
    planner.cold_ramp_seed = 0.0
    planner.z_gate_alt_frac = 0.0
    planner.tb_enabled = arm['tb']
    planner.tb_vz_thresh = TB_VZ_THRESH

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
        launch_samples = []  # (t_rel, readiness_or_ramp, cmd_spd, ach_spd, vz)
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
            cmd_spd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
            ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)
            if ach_spd > max_ach: max_ach = ach_spd
            # log launch-phase samples for diagnostics
            if gi == 0 and len(launch_samples) < 200:
                launch_samples.append((
                    round(time.time()-t0, 3),
                    round(cmd_spd, 3),
                    round(ach_spd, 3),
                    round(abs(vel[2]), 3),
                ))
            await drone.offboard.set_velocity_ned(cmd)
            await asyncio.sleep(dt)
        total = time.time() - t0
        completed = gi >= len(gates)

        try: await drone.offboard.stop()
        except: pass
        try: await drone.action.land()
        except: pass

        # first-1s grid for diagnostics
        grid = []
        if launch_samples:
            ts = launch_samples[0][0]
            for k in range(21):  # 0.0 to 2.0s in 0.1s steps
                t_target = ts + k * 0.1
                best = min(launch_samples, key=lambda r: abs(r[0]-t_target))
                grid.append({'t': round(best[0]-ts,3), 'cmd': best[1], 'ach': best[2], 'vz': best[3]})

        return {
            'arm': arm['name'], 'course': course,
            'completed': completed,
            'lap_time': round(total, 3),
            'leg0_time': round(leg0_t, 3) if leg0_t is not None else None,
            't_offboard': round(t_offboard, 3),
            'max_spd': round(max_ach, 3),
            'health': diag,
            'launch_grid': grid,
        }
    finally:
        stop_pl.set()
        pl_task.cancel()
        try: await pl_task
        except (asyncio.CancelledError, Exception): pass

async def trial(course, trial_idx, arm, results):
    tag = f'{course} t{trial_idx} {arm["name"]}'
    print(f'[{tag}] hardened restart...', flush=True)
    try:
        bench.hardened_restart(B)
    except Exception as e:
        r = {'trial': trial_idx, 'arm': arm['name'], 'course': course,
             'completed': False, 'error': f'restart_failed: {e}'}
        results.append(r)
        bench.atomic_write_json(RESULT, results)
        print(f'  -> RESTART FAILED: {e}', flush=True)
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

    r['trial'] = trial_idx
    results.append(r)
    bench.atomic_write_json(RESULT, results)
    print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
          f'done={r.get("completed")} err={r.get("error")}', flush=True)

def evaluate(results, course, n_pairs, min_done):
    """Return (pass, detail_str) for one course phase."""
    ctl = [r for r in results if r['course']==course and r['arm']=='control_tb_off']
    tre = [r for r in results if r['course']==course and r['arm']=='treatment_tb']
    cd = [r for r in ctl if r.get('completed')]
    td = [r for r in tre if r.get('completed')]
    cl = [r['lap_time'] for r in cd if r.get('lap_time') is not None]
    tl = [r['lap_time'] for r in td if r.get('lap_time') is not None]
    c0 = [r['leg0_time'] for r in cd if r.get('leg0_time') is not None]
    t0 = [r['leg0_time'] for r in td if r.get('leg0_time') is not None]
    lines = []
    lines.append(f'  completion: ctl {len(cd)}/{n_pairs}  tre {len(td)}/{n_pairs}')
    if cl and tl:
        d_lap = median(tl) - median(cl)
        lines.append(f'  lap median: ctl {median(cl):.3f}  tre {median(tl):.3f}  d={d_lap:+.3f}')
    else:
        d_lap = None
    if c0 and t0:
        d_leg0 = median(t0) - median(c0)
        lines.append(f'  [watch] leg0: ctl {median(c0):.3f}  tre {median(t0):.3f}  d={d_leg0:+.3f}')
    ra = len(td) >= len(cd) and len(td) >= min_done
    rb = d_lap is not None and d_lap <= 0.0
    lines.append(f'  rule_a (done >= ctl AND >= {min_done}/{n_pairs}): {ra}')
    lines.append(f'  rule_b (lap delta <= 0): {rb}')
    passed = ra and rb
    lines.append(f'  --> {"PASS" if passed else "FAIL"}')
    return passed, '\n'.join(lines)

async def main():
    bench.acquire_singleton('tb')
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ARMS = {ARMS}', flush=True)
    print(f'[init] tb_vz_thresh={TB_VZ_THRESH}', flush=True)
    print(f'[init] plan: {N_TECH} tech pairs first, then {N_MIXED} mixed pairs (conditional)', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    # PREFLIGHT
    print('[preflight] 1 control trial', flush=True)
    await trial('technical', 0, ARMS[0], results)
    if not results[0].get('completed'):
        print(f'[preflight] FAILED: {results[0].get("error")}; aborting', flush=True)
        return
    print(f'[preflight] OK lap={results[0].get("lap_time")}', flush=True)

    # TECHNICAL (10 pairs)
    print(f'\n=== TECHNICAL ({N_TECH} pairs) ===', flush=True)
    for t in range(1, N_TECH+1):
        for arm in ARMS:
            await trial('technical', t, arm, results)

    tech_pass, tech_detail = evaluate(results, 'technical', N_TECH, 9)
    print(f'\n--- TECHNICAL DECISION ---\n{tech_detail}', flush=True)

    if not tech_pass:
        print('\nTechnical did not pass. Skipping mixed. REJECT transition_blend.', flush=True)
        bench.atomic_write_json(RESULT, results)
        return

    # MIXED (10 pairs, conditional on technical pass)
    print(f'\n=== MIXED ({N_MIXED} pairs) ===', flush=True)
    for t in range(1, N_MIXED+1):
        for arm in ARMS:
            await trial('mixed', t, arm, results)

    mix_pass, mix_detail = evaluate(results, 'mixed', N_MIXED, 8)
    print(f'\n--- MIXED DECISION ---\n{mix_detail}', flush=True)

    verdict = 'ADOPT' if (tech_pass and mix_pass) else 'REJECT'
    print(f'\n=== VERDICT: {verdict} transition_blend (vz_thresh={TB_VZ_THRESH}) ===', flush=True)

    # per-trial dump
    print('\n=== PER-TRIAL ===', flush=True)
    for course in ('technical', 'mixed'):
        trials = sorted({r['trial'] for r in results if r['course']==course and r['trial']>=1})
        if not trials: continue
        print(f'\n--- {course.upper()} ---')
        for t in trials:
            rc = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='control_tb_off']
            rt = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='treatment_tb']
            c = rc[0] if rc else {}; tr = rt[0] if rt else {}
            cl = c.get('lap_time'); tl = tr.get('lap_time')
            delta = f'{tl-cl:+.3f}' if (cl is not None and tl is not None) else 'n/a'
            print(f'  t{t}  ctl={cl}({c.get("completed")})  tre={tl}({tr.get("completed")})  d={delta}  '
                  f'leg0_c={c.get("leg0_time")} leg0_t={tr.get("leg0_time")}')

    bench.atomic_write_json(RESULT, results)

if __name__ == '__main__':
    asyncio.run(main())
