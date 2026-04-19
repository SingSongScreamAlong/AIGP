"""Technical-only cruise_speed sweep on top of locked VEL_P_ACC=6.0.

Grid: cruise_speed in [9, 10, 11, 12]
Hold: max_speed=14.0 (ceiling above highest cruise), base_blend=1.5, PX4 VEL_P_ACC=6.0
Trials: 3 per cell. Per-trial timeout: 150s.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

COURSE = 'technical'
CRUISE_SPEEDS = [9.0, 10.0, 11.0, 12.0]
MAX_SPEED_CEIL = 14.0
BASE_BLEND = 1.5
TRIALS = 3
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_cruise_sweep_technical.json'
TRIAL_TIMEOUT = 150.0

def lock_tune():
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0

async def run_one(cs):
    p = B.V51Planner(max_speed=MAX_SPEED_CEIL, cruise_speed=cs, base_blend=BASE_BLEND)
    return await B.run_trial(p, B.COURSES[COURSE])

async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] starting PX4 SITL...', flush=True)
    B.restart_px4()
    await asyncio.sleep(2)

    lock_tune()
    print(f'\nLOCKED PX4 TUNE: VEL_P_ACC=6.0')
    print(f'cruise_speed sweep on {COURSE}: {CRUISE_SPEEDS}')
    print(f'max_speed={MAX_SPEED_CEIL} (ceiling), base_blend={BASE_BLEND} held')
    print('='*72)

    results = []
    for cs in CRUISE_SPEEDS:
        print(f'\n[cruise_speed={cs}]')
        for trial in range(TRIALS):
            print(f'  [trial {trial+1}/{TRIALS}]...', end=' ', flush=True)
            try:
                r = await asyncio.wait_for(run_one(cs), timeout=TRIAL_TIMEOUT)
            except asyncio.TimeoutError:
                print('HARNESS_TIMEOUT — force-killing')
                os.system('pkill -9 -f mavsdk_server 2>/dev/null')
                os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
                await asyncio.sleep(3)
                r = {'completed': False, 'time': TRIAL_TIMEOUT, 'max_spd': 0, 'avg_cmd': 0,
                     'avg_ach': 0, 'avg_err': 0, 'util': 0, 'splits': [], 'gates_passed': -1,
                     'harness_timeout': True}
            r['cruise_speed'] = cs
            r['trial'] = trial
            results.append(r)
            if r.get('harness_timeout'):
                st = 'HANG'
            else:
                st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | Max:{r.get("max_spd",0)} Cmd:{r.get("avg_cmd",0):.2f} Err:{r.get("avg_err",0)} Util:{r.get("util",0):.1%}')
            B.restart_px4()
            await asyncio.sleep(2)

    # Summary
    print(f'\n\n{"="*72}')
    print(f'cruise_speed sweep summary  (technical, VEL_P_ACC=6.0, max_speed=14)')
    print('='*72)
    print(f'{"cruise":>8} {"N":>3} {"p50":>10} {"AvgCmd":>8} {"AvgAch":>8} {"MaxSpd":>8} {"AvgErr":>8} {"Util":>8}')
    print('-'*72)
    summary = {}
    for cs in CRUISE_SPEEDS:
        rs = [r for r in results if r['cruise_speed']==cs and r['completed']]
        if not rs:
            print(f'{cs:>8} {0:>3}  no completions'); continue
        rs_sorted = sorted(rs, key=lambda x: x['time'])
        med = rs_sorted[len(rs_sorted)//2]
        avg_cmd = sum(r['avg_cmd'] for r in rs)/len(rs)
        avg_ach = sum(r['avg_ach'] for r in rs)/len(rs)
        avg_max = sum(r['max_spd'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[cs] = dict(
            cruise_speed=cs, n=len(rs), time_p50=med['time'],
            avg_cmd=round(avg_cmd,2), avg_ach=round(avg_ach,2), max_spd=round(avg_max,2),
            avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'{cs:>8} {len(rs):>3} {med["time"]:>9.2f}s {avg_cmd:>7.2f} {avg_ach:>7.2f} {avg_max:>7.2f} {avg_err:>7.2f} {avg_util:>7.1%}')

    base = 14.046
    print(f'\n{"cruise":>8} {"p50":>10} {"Delta":>14}')
    print('-'*72)
    for cs in CRUISE_SPEEDS:
        if cs in summary:
            t = summary[cs]['time_p50']
            d = t - base
            pct = d / base * 100
            print(f'{cs:>8} {t:>9.2f}s {d:>+8.2f}s ({pct:+.1f}%)')

    with open(LOG_PATH, 'w') as f:
        json.dump({'trials': results, 'summary': {str(k): v for k,v in summary.items()},
                   'course': COURSE, 'locked_px4_tune': {'MPC_XY_VEL_P_ACC': 6.0},
                   'held_planner_kw': {'max_speed': MAX_SPEED_CEIL, 'base_blend': BASE_BLEND}},
                  f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
