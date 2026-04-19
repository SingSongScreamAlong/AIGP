"""PX4 MPC_XY_VEL_P_ACC mini-sweep on technical course.

Single-axis sweep of outer velocity-loop P gain into accel feed-forward.
Everything else locked to baseline V5.1 config.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

VEL_P_VALS = [6.0, 8.0, 10.0, 12.0]
TRIALS_PER_CELL = 3
COURSE = 'technical'
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_vel_p_sweep_ext_technical.json'
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

def set_cell(vp):
    # keep ACC_HOR / JERK at baseline - we already know they don't matter
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    # the knob under test
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = vp

async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] starting PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    gates = B.COURSES[COURSE]
    results = []
    print(f'\nVEL_P SWEEP: {len(VEL_P_VALS)} cells x {TRIALS_PER_CELL} on {COURSE} ({len(gates)} gates)')
    print('='*72)

    for ci, vp in enumerate(VEL_P_VALS, 1):
        set_cell(vp)
        print(f'\n[cell {ci}/{len(VEL_P_VALS)}] MPC_XY_VEL_P_ACC={vp}')
        for trial in range(TRIALS_PER_CELL):
            print(f'  [trial {trial+1}/{TRIALS_PER_CELL}]...', end=' ', flush=True)
            p = B.V51Planner(**PLANNER_KW)
            r = await B.run_trial(p, gates)
            r['course']    = COURSE
            r['trial']     = trial
            r['vel_p_acc'] = vp
            results.append(r)
            st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | Max:{r["max_spd"]} Err:{r["avg_err"]} Util:{r["util"]:.1%}')
            B.restart_px4(); await asyncio.sleep(2)

    # Aggregate
    print(f'\n\n{"="*72}')
    print(f'PX4 MPC_XY_VEL_P_ACC sweep on {COURSE}  (V5.1 locked, median over {TRIALS_PER_CELL})')
    print('='*72)
    print(f'{"VEL_P":>8} {"N":>3} {"Time p50":>10} {"MaxSpd":>7} {"AvgErr":>7} {"Util":>7}')
    print('-'*72)
    summary = {}
    for vp in VEL_P_VALS:
        rs = [r for r in results if r['vel_p_acc']==vp and r['completed']]
        if not rs:
            print(f'{vp:>8.2f} {0:>3}  no completions'); continue
        rs_sorted = sorted(rs, key=lambda x: x['time'])
        med = rs_sorted[len(rs_sorted)//2]
        avg_max = sum(r['max_spd'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[str(vp)] = dict(
            vel_p_acc=vp, n=len(rs), time_p50=med['time'],
            max_spd=round(avg_max,2), avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'{vp:>8.2f} {len(rs):>3} {med["time"]:>9.2f}s {avg_max:>6.2f} {avg_err:>6.2f} {avg_util:>6.1%}')

    if summary:
        best = min(summary.values(), key=lambda s: s['time_p50'])
        base = summary.get('1.8')
        print('-'*72)
        print(f'BEST:  VEL_P={best["vel_p_acc"]}  p50={best["time_p50"]}s  util={best["util"]:.1%}')
        if base:
            d = best['time_p50'] - base['time_p50']
            print(f'BASELINE (1.8): p50={base["time_p50"]}s  delta={d:+.2f}s')

    out = {'trials': results, 'summary': summary, 'grid': VEL_P_VALS,
           'course': COURSE, 'trials_per_cell': TRIALS_PER_CELL,
           'planner_kw': PLANNER_KW}
    with open(LOG_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
