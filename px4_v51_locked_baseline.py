"""Post-tune PX4 baseline: VEL_P_ACC=6.0 locked, 3 courses x 3 trials.

Uses per-trial timeout to bypass harness hangs.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

COURSES = ['technical', 'mixed', 'sprint']
TRIALS = 3
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_v51_locked_baseline.json'
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
TRIAL_TIMEOUT = 180.0

def lock_tune():
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0

async def run_one(course):
    p = B.V51Planner(**PLANNER_KW)
    return await B.run_trial(p, B.COURSES[course])

async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] starting PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    lock_tune()
    print(f'\nLOCKED TUNE: VEL_P_ACC=6.0  ACC_HOR=10  JERK_AUTO=30')
    print(f'{len(COURSES)} courses x {TRIALS} trials  (per-trial timeout {TRIAL_TIMEOUT}s)')
    print('='*72)

    results = []
    for course in COURSES:
        print(f'\n[{course}] {len(B.COURSES[course])} gates')
        for trial in range(TRIALS):
            print(f'  [trial {trial+1}/{TRIALS}]...', end=' ', flush=True)
            try:
                r = await asyncio.wait_for(run_one(course), timeout=TRIAL_TIMEOUT)
            except asyncio.TimeoutError:
                print('HARNESS_TIMEOUT — force-killing')
                os.system('pkill -9 -f mavsdk_server 2>/dev/null')
                os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
                await asyncio.sleep(3)
                r = {'completed': False, 'time': TRIAL_TIMEOUT, 'max_spd': 0, 'avg_cmd': 0,
                     'avg_ach': 0, 'avg_err': 0, 'util': 0, 'splits': [], 'gates_passed': -1,
                     'harness_timeout': True}
            r['course'] = course
            r['trial'] = trial
            results.append(r)
            if r.get('harness_timeout'):
                st = 'HANG'
            else:
                st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | Max:{r.get("max_spd",0)} Err:{r.get("avg_err",0)} Util:{r.get("util",0):.1%}')
            B.restart_px4()
            await asyncio.sleep(2)

    # Summary
    print(f'\n\n{"="*72}')
    print(f'PX4 POST-TUNE BASELINE  (V5.1 planner + VEL_P_ACC=6.0)')
    print('='*72)
    print(f'{"Course":<12} {"N":>3} {"p50":>10} {"MaxSpd":>8} {"AvgErr":>8} {"Util":>8}')
    print('-'*72)
    summary = {}
    for course in COURSES:
        rs = [r for r in results if r['course']==course and r['completed']]
        if not rs:
            print(f'{course:<12} {0:>3}  no completions'); continue
        rs_sorted = sorted(rs, key=lambda x: x['time'])
        med = rs_sorted[len(rs_sorted)//2]
        avg_max = sum(r['max_spd'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[course] = dict(
            course=course, n=len(rs), time_p50=med['time'],
            max_spd=round(avg_max,2), avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'{course:<12} {len(rs):>3} {med["time"]:>9.2f}s {avg_max:>7.2f} {avg_err:>7.2f} {avg_util:>7.1%}')

    # Compare vs pre-tune baseline
    pre = {'technical': 16.57, 'mixed': 28.79, 'sprint': 45.24}
    print(f'\n{"Course":<12} {"Pre-tune":>10} {"Post-tune":>10} {"Delta":>10}')
    print('-'*72)
    for course in COURSES:
        if course in summary:
            post = summary[course]['time_p50']
            d = post - pre[course]
            pct = d / pre[course] * 100
            print(f'{course:<12} {pre[course]:>9.2f}s {post:>9.2f}s {d:>+8.2f}s ({pct:+.1f}%)')

    with open(LOG_PATH, 'w') as f:
        json.dump({'trials': results, 'summary': summary, 'planner_kw': PLANNER_KW,
                   'tune': {'MPC_XY_VEL_P_ACC': 6.0, 'MPC_ACC_HOR': 10.0, 'MPC_JERK_AUTO': 30.0}}, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
