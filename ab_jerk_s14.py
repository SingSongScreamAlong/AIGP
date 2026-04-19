# ab_jerk_s14.py — Session 14: PX4 Jerk A/B
# Control: current locked tune (MPC_JERK_AUTO=30, MPC_JERK_MAX=50)
# Treatment: MPC_JERK_AUTO=45, MPC_JERK_MAX=75
# No planner changes. Same V5.1 baseline throughout.

import asyncio, time, math, json, os, sys
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
import bench
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from statistics import median, mean, stdev

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
RESULT = os.path.join(LOGS, 'ab_jerk_s14.json')

N_TECH  = 10
N_MIXED = 10
TRIAL_TIMEOUT = 90.0
POST_RESTART_WAIT = 5.0

BASELINE_PARAMS = {
    'MPC_XY_VEL_P_ACC': 6.0,
    'MPC_ACC_HOR': 10.0,
    'MPC_ACC_HOR_MAX': 10.0,
    'MPC_JERK_AUTO': 30.0,
    'MPC_JERK_MAX': 50.0,
    'MPC_TILTMAX_AIR': 70.0,
    'MPC_XY_VEL_MAX': 15.0,
    'MPC_Z_VEL_MAX_UP': 5.0,
}

ARMS = [
    {'name': 'control_jerk30', 'jerk_auto': 30.0, 'jerk_max': 50.0},
    {'name': 'treatment_jerk45', 'jerk_auto': 45.0, 'jerk_max': 75.0},
]

async def trial(course, trial_num, arm, results):
    arm_name = arm['name']
    print(f'[{course} t{trial_num} {arm_name}] hardened restart...', flush=True)

    try:
        bench.hardened_restart(B)
    except Exception as e:
        r = {'course': course, 'trial': trial_num, 'arm': arm_name,
             'completed': False, 'error': f'restart: {e}', 'bench_failure': True}
        results.append(r)
        bench.atomic_write_json(RESULT, results)
        print(f'  -> BENCH FAILURE: {e}', flush=True)
        return

    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    # Set PX4 params — baseline + arm-specific jerk
    params = dict(BASELINE_PARAMS)
    params['MPC_JERK_AUTO'] = arm['jerk_auto']
    params['MPC_JERK_MAX'] = arm['jerk_max']

    for n, v in params.items():
        try:
            await drone.param.set_param_float(n, v)
        except Exception as e:
            print(f'    param {n}={v} failed: {e}', flush=True)

    await asyncio.sleep(POST_RESTART_WAIT)

    pos = [0, 0, 0]; vel = [0, 0, 0]
    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    gates = B.COURSES[course]
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    gate_alt_frac = 0.95; gate_vz_max = 0.3; gate_timeout = 10.0
    wait_start = time.time()
    while True:
        if (time.time() - wait_start) >= gate_timeout: break
        if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break
        await asyncio.sleep(0.05)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    gi = 0; t0 = time.time(); dt = 1/50; threshold = 2.5
    gate_events = []
    sustain_spds = []

    while gi < len(gates):
        if time.time() - t0 > TRIAL_TIMEOUT:
            break
        g = gates[gi]
        dx = g[0]-pos[0]; dy = g[1]-pos[1]; dz = g[2]-pos[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < threshold:
            tg = time.time() - t0
            gspd = math.sqrt(vel[0]**2 + vel[1]**2)
            gate_events.append({'gate': gi, 'time': round(tg, 4), 'spd': round(gspd, 3)})
            planner.on_gate_passed(gspd)
            gi += 1
            continue
        ng = gates[gi+1] if gi+1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)
        # Track sustain speed (simple proxy: samples where cmd > 10)
        cmd_spd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        if cmd_spd > 10.0 and gi > 0:
            sustain_spds.append(ach_spd)
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0
    completed = gi >= len(gates)

    try: await drone.offboard.stop()
    except: pass
    await drone.action.land()
    await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    leg0_time = gate_events[0]['time'] if gate_events else None
    sus_med = round(median(sustain_spds), 3) if sustain_spds else None

    r = {
        'course': course, 'trial': trial_num, 'arm': arm_name,
        'completed': completed,
        'lap_time': round(total, 3),
        'leg0_time': round(leg0_time, 3) if leg0_time else None,
        'sustain_med_spd': sus_med,
        'gates_passed': gi,
        'gate_events': gate_events,
        'bench_failure': False,
        'error': None if completed else 'timeout',
    }
    results.append(r)
    bench.atomic_write_json(RESULT, results)
    print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
          f'sus={r.get("sustain_med_spd")} done={r.get("completed")} err={r.get("error")}', flush=True)

def evaluate_v2(results, course, n_pairs):
    ctl_name = 'control_jerk30'
    tre_name = 'treatment_jerk45'
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
    lines.append(f'  flown: ctl {len(ctl_done)}/{len(ctl_flown)}  tre {len(tre_done)}/{len(tre_flown)}')
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
    bench.acquire_singleton('jerk14')
    print(f'[init] Session 14 — PX4 Jerk A/B', flush=True)
    print(f'[init] Control: JERK_AUTO=30, JERK_MAX=50', flush=True)
    print(f'[init] Treatment: JERK_AUTO=45, JERK_MAX=75', flush=True)
    print(f'[init] ARMS = {[a["name"] for a in ARMS]}', flush=True)
    print(f'[init] result file: {RESULT}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    print('[preflight] 1 control trial', flush=True)
    await trial('technical', 0, ARMS[0], results)
    if not results[0].get('completed'):
        print(f'[preflight] FAILED: {results[0].get("error")}; aborting', flush=True)
        return
    print(f'[preflight] OK lap={results[0].get("lap_time")}', flush=True)

    print(f'\n=== TECHNICAL ({N_TECH} pairs) ===', flush=True)
    for t in range(1, N_TECH+1):
        for arm in ARMS:
            await trial('technical', t, arm, results)
    tech_verdict, tech_detail = evaluate_v2(results, 'technical', N_TECH)
    print(f'\n--- TECHNICAL DECISION ---\n{tech_detail}', flush=True)

    if tech_verdict == 'REJECT':
        print('\nTechnical REJECTED. Skipping mixed.', flush=True)
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
    print(f'\n=== VERDICT: {final} PX4 Jerk 45/75 ===', flush=True)

    print('\n=== PER-TRIAL ===', flush=True)
    for course in ('technical', 'mixed'):
        trials = sorted({r['trial'] for r in results if r['course']==course and r['trial']>=1})
        if not trials: continue
        print(f'\n--- {course.upper()} ---')
        for t in trials:
            rc = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='control_jerk30']
            rt = [r for r in results if r['course']==course and r['trial']==t and r['arm']=='treatment_jerk45']
            c = rc[0] if rc else {}; tr = rt[0] if rt else {}
            cl = c.get('lap_time'); tl = tr.get('lap_time')
            delta = f'{tl-cl:+.3f}' if (cl is not None and tl is not None) else 'n/a'
            print(f'  t{t}  ctl={cl}({c.get("completed")})  tre={tl}({tr.get("completed")})  d={delta}  '
                  f'sus_c={c.get("sustain_med_spd")} sus_t={tr.get("sustain_med_spd")}')
    bench.atomic_write_json(RESULT, results)

if __name__ == '__main__':
    asyncio.run(main())
