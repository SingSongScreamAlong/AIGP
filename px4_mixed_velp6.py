"""Isolated mixed@VEL_P=6.0 run with per-trial timeout to bypass harness hang bug."""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

TRIALS = 3
COURSE = 'mixed'
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_mixed_velp6.json'
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
TRIAL_TIMEOUT = 150.0   # seconds — wraps run_trial so harness can't hang indefinitely

def set_cell_6():
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0

async def run_one():
    p = B.V51Planner(**PLANNER_KW)
    return await B.run_trial(p, B.COURSES[COURSE])

async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] starting PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    set_cell_6()
    print(f'\n[{COURSE}] MPC_XY_VEL_P_ACC=6.0  (per-trial timeout {TRIAL_TIMEOUT}s)')
    print('='*72)

    results = []
    for trial in range(TRIALS):
        print(f'  [trial {trial+1}/{TRIALS}]...', end=' ', flush=True)
        try:
            r = await asyncio.wait_for(run_one(), timeout=TRIAL_TIMEOUT)
        except asyncio.TimeoutError:
            print('HARNESS_TIMEOUT — force-killing px4+mavsdk')
            os.system('pkill -9 -f mavsdk_server 2>/dev/null')
            os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
            await asyncio.sleep(3)
            r = {'completed': False, 'time': TRIAL_TIMEOUT, 'max_spd': 0, 'avg_cmd': 0,
                 'avg_ach': 0, 'avg_err': 0, 'util': 0, 'splits': [], 'gates_passed': -1,
                 'harness_timeout': True}
        r['course'] = COURSE
        r['trial'] = trial
        r['vel_p_acc'] = 6.0
        results.append(r)
        if r.get('harness_timeout'):
            st = 'HANG'
        else:
            st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
        print(f'{st} {r["time"]}s | Max:{r.get("max_spd",0)} Err:{r.get("avg_err",0)} Util:{r.get("util",0):.1%}')
        B.restart_px4()
        await asyncio.sleep(2)

    # Summary
    print(f'\n{"="*72}')
    print(f'{COURSE}  VEL_P=6.0  summary')
    print('='*72)
    ok = [r for r in results if r['completed']]
    if ok:
        ok_sorted = sorted(ok, key=lambda x: x['time'])
        med = ok_sorted[len(ok_sorted)//2]
        print(f'completed: {len(ok)}/{TRIALS}')
        print(f'p50 time  : {med["time"]}s')
        print(f'max_spd   : {sum(r["max_spd"] for r in ok)/len(ok):.2f}')
        print(f'avg_err   : {sum(r["avg_err"] for r in ok)/len(ok):.2f}')
        print(f'util      : {sum(r["util"] for r in ok)/len(ok):.1%}')
    else:
        print(f'completed: 0/{TRIALS}  — every trial failed or hung')
    print(f'\ntrial details:')
    for i,r in enumerate(results):
        tag = 'HANG' if r.get('harness_timeout') else ('OK' if r['completed'] else f'FAIL@{r.get("gates_passed",-1)}')
        print(f'  t{i+1}: {tag} {r["time"]}s gates={r.get("gates_passed",-1)}')

    with open(LOG_PATH, 'w') as f:
        json.dump({'trials': results, 'course': COURSE, 'vel_p_acc': 6.0}, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
