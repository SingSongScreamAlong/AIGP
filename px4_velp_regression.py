"""PX4 MPC_XY_VEL_P_ACC regression check on mixed + sprint.

Compare default (1.8) vs winner (6.0) on both courses to make sure the
technical-course tune isn't regressing the longer-leg courses.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

VEL_P_VALS = [1.8, 6.0]
COURSES    = ['mixed', 'sprint']
TRIALS_PER_CELL = 3
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_velp_regression.json'
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

def set_cell(vp):
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = vp

async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] starting PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    results = []
    print(f'\nVEL_P REGRESSION: {len(VEL_P_VALS)} cells x {len(COURSES)} courses x {TRIALS_PER_CELL} trials')
    print('='*72)

    for course in COURSES:
        gates = B.COURSES[course]
        for vp in VEL_P_VALS:
            set_cell(vp)
            print(f'\n[{course}] MPC_XY_VEL_P_ACC={vp}')
            for trial in range(TRIALS_PER_CELL):
                print(f'  [trial {trial+1}/{TRIALS_PER_CELL}]...', end=' ', flush=True)
                p = B.V51Planner(**PLANNER_KW)
                r = await B.run_trial(p, gates)
                r['course']    = course
                r['trial']     = trial
                r['vel_p_acc'] = vp
                results.append(r)
                st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
                print(f'{st} {r["time"]}s | Max:{r["max_spd"]} Err:{r["avg_err"]} Util:{r["util"]:.1%}')
                B.restart_px4(); await asyncio.sleep(2)

    # Aggregate
    print(f'\n\n{"="*72}')
    print('PX4 VEL_P REGRESSION CHECK  (V5.1 locked)')
    print('='*72)
    print(f'{"Course":<10} {"VEL_P":>6} {"N":>3} {"Time p50":>10} {"MaxSpd":>7} {"AvgErr":>7} {"Util":>7}')
    print('-'*72)
    summary = {}
    for course in COURSES:
        for vp in VEL_P_VALS:
            rs = [r for r in results if r['course']==course and r['vel_p_acc']==vp and r['completed']]
            if not rs:
                print(f'{course:<10} {vp:>6.2f} {0:>3}  no completions'); continue
            rs_sorted = sorted(rs, key=lambda x: x['time'])
            med = rs_sorted[len(rs_sorted)//2]
            avg_max = sum(r['max_spd'] for r in rs)/len(rs)
            avg_err = sum(r['avg_err'] for r in rs)/len(rs)
            avg_util= sum(r['util']    for r in rs)/len(rs)
            summary[f'{course}_{vp}'] = dict(
                course=course, vel_p_acc=vp, n=len(rs), time_p50=med['time'],
                max_spd=round(avg_max,2), avg_err=round(avg_err,2), util=round(avg_util,3),
            )
            print(f'{course:<10} {vp:>6.2f} {len(rs):>3} {med["time"]:>9.2f}s {avg_max:>6.2f} {avg_err:>6.2f} {avg_util:>6.1%}')
        # delta row
        a = summary.get(f'{course}_1.8'); b = summary.get(f'{course}_6.0')
        if a and b:
            d = b['time_p50'] - a['time_p50']
            marker = 'improves' if d < 0 else 'regresses'
            print(f'  -> VEL_P=6.0 {marker} by {abs(d):.2f}s  ({d/a["time_p50"]*100:+.1f}%)')
            print()

    out = {'trials': results, 'summary': summary,
           'vel_p_vals': VEL_P_VALS, 'courses': COURSES,
           'trials_per_cell': TRIALS_PER_CELL, 'planner_kw': PLANNER_KW}
    with open(LOG_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
