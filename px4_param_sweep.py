"""PX4 param sweep on technical course.

Grid: MPC_ACC_HOR x MPC_JERK_AUTO (3x3).
Planner locked to V5.1 baseline config.
Uses px4_v51_baseline.py's run_trial / restart_px4 / COURSES / V51Planner.
"""
import asyncio, json, os, sys, time, math
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

# Sweep grid
ACC_HOR_VALS  = [6.0, 10.0, 16.0]       # low / baseline / high
JERK_VALS     = [15.0, 30.0, 60.0]      # low / baseline / high
TRIALS_PER_CELL = 3
COURSE = 'technical'
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_param_sweep_technical.json'

PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

def set_cell(acc_hor, jerk):
    """Mutate the shared RACE_PARAMS dict so run_trial picks up new values on next param.set."""
    B.RACE_PARAMS['MPC_ACC_HOR']     = acc_hor
    B.RACE_PARAMS['MPC_ACC_HOR_MAX'] = max(acc_hor, 10.0)
    B.RACE_PARAMS['MPC_JERK_AUTO']   = jerk
    B.RACE_PARAMS['MPC_JERK_MAX']    = max(jerk * 1.67, 50.0)

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
    cells = [(a, j) for a in ACC_HOR_VALS for j in JERK_VALS]
    print(f'\nSWEEP: {len(cells)} cells x {TRIALS_PER_CELL} trials on {COURSE} ({len(gates)} gates)')
    print('='*72)

    for ci, (acc_hor, jerk) in enumerate(cells, 1):
        set_cell(acc_hor, jerk)
        print(f'\n[cell {ci}/{len(cells)}] MPC_ACC_HOR={acc_hor}  MPC_JERK_AUTO={jerk}')
        for trial in range(TRIALS_PER_CELL):
            print(f'  [trial {trial+1}/{TRIALS_PER_CELL}]...', end=' ', flush=True)
            p = B.V51Planner(**PLANNER_KW)
            r = await B.run_trial(p, gates)
            r['course']       = COURSE
            r['trial']        = trial
            r['acc_hor']      = acc_hor
            r['jerk_auto']    = jerk
            r['acc_hor_max']  = B.RACE_PARAMS['MPC_ACC_HOR_MAX']
            r['jerk_max']     = B.RACE_PARAMS['MPC_JERK_MAX']
            results.append(r)
            st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | Max:{r["max_spd"]} Err:{r["avg_err"]} Util:{r["util"]:.1%}')
            B.restart_px4(); await asyncio.sleep(2)

    # Aggregate per cell (median)
    print(f'\n\n{"="*78}')
    print(f'PX4 PARAM SWEEP on {COURSE}  (V5.1 planner locked, median over {TRIALS_PER_CELL})')
    print('='*78)
    print(f'{"ACC_HOR":>8} {"JERK":>6} {"N":>3} {"Time p50":>10} {"MaxSpd":>7} {"AvgErr":>7} {"Util":>7}')
    print('-'*78)
    summary = {}
    for acc_hor, jerk in cells:
        rs = [r for r in results if r['acc_hor']==acc_hor and r['jerk_auto']==jerk and r['completed']]
        if not rs:
            print(f'{acc_hor:>8.1f} {jerk:>6.1f} {0:>3}  no completions')
            continue
        rs_sorted = sorted(rs, key=lambda x: x['time'])
        med = rs_sorted[len(rs_sorted)//2]
        avg_max = sum(r['max_spd'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[f'{acc_hor}_{jerk}'] = dict(
            acc_hor=acc_hor, jerk_auto=jerk, n=len(rs),
            time_p50=med['time'], max_spd=round(avg_max,2),
            avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'{acc_hor:>8.1f} {jerk:>6.1f} {len(rs):>3} {med["time"]:>9.2f}s {avg_max:>6.2f} {avg_err:>6.2f} {avg_util:>6.1%}')

    # Mark best
    if summary:
        best = min(summary.values(), key=lambda s: s['time_p50'])
        base = summary.get('10.0_30.0')
        print('-'*78)
        print(f'BEST:  ACC_HOR={best["acc_hor"]}  JERK={best["jerk_auto"]}  p50={best["time_p50"]}s  util={best["util"]:.1%}')
        if base:
            d = best['time_p50'] - base['time_p50']
            print(f'BASELINE (10/30): p50={base["time_p50"]}s  delta={d:+.2f}s')

    out = {'trials': results, 'summary': summary,
           'grid': {'acc_hor': ACC_HOR_VALS, 'jerk_auto': JERK_VALS},
           'course': COURSE, 'trials_per_cell': TRIALS_PER_CELL,
           'planner_kw': PLANNER_KW}
    with open(LOG_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
