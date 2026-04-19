"""Technical-only base_blend sweep on top of locked VEL_P_ACC=6.0.

Grid: base_blend in [1.0, 1.5, 2.0, 2.5, 3.0]
Hold: cruise_speed=9.0, max_speed=11.0, PX4 VEL_P_ACC=6.0
Trials: 3 per cell. Per-trial timeout: 150s.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

COURSE = 'technical'
BLENDS = [1.0, 1.5, 2.0, 2.5, 3.0]
CRUISE = 9.0
MAX_SPEED = 11.0
TRIALS = 3
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_blend_sweep_technical.json'
TRIAL_TIMEOUT = 150.0

def lock_tune():
    B.RACE_PARAMS['MPC_ACC_HOR']      = 10.0
    B.RACE_PARAMS['MPC_ACC_HOR_MAX']  = 10.0
    B.RACE_PARAMS['MPC_JERK_AUTO']    = 30.0
    B.RACE_PARAMS['MPC_JERK_MAX']     = 50.0
    B.RACE_PARAMS['MPC_XY_VEL_P_ACC'] = 6.0

async def run_one(bl):
    p = B.V51Planner(max_speed=MAX_SPEED, cruise_speed=CRUISE, base_blend=bl)
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
    print(f'base_blend sweep on {COURSE}: {BLENDS}')
    print(f'cruise_speed={CRUISE}, max_speed={MAX_SPEED} held')
    print('='*72)

    results = []
    for bl in BLENDS:
        print(f'\n[base_blend={bl}]')
        for trial in range(TRIALS):
            print(f'  [trial {trial+1}/{TRIALS}]...', end=' ', flush=True)
            try:
                r = await asyncio.wait_for(run_one(bl), timeout=TRIAL_TIMEOUT)
            except asyncio.TimeoutError:
                print('HARNESS_TIMEOUT — force-killing')
                os.system('pkill -9 -f mavsdk_server 2>/dev/null')
                os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
                await asyncio.sleep(3)
                r = {'completed': False, 'time': TRIAL_TIMEOUT, 'max_spd': 0, 'avg_cmd': 0,
                     'avg_ach': 0, 'avg_err': 0, 'util': 0, 'splits': [], 'gates_passed': -1,
                     'harness_timeout': True}
            r['base_blend'] = bl
            r['trial'] = trial
            results.append(r)
            if r.get('harness_timeout'):
                st = 'HANG'
            else:
                st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | Max:{r.get("max_spd",0)} Cmd:{r.get("avg_cmd",0):.2f} Ach:{r.get("avg_ach",0):.2f} Err:{r.get("avg_err",0)} Util:{r.get("util",0):.1%}')
            B.restart_px4()
            await asyncio.sleep(2)

    # Summary
    print(f'\n\n{"="*72}')
    print(f'base_blend sweep summary  (technical, VEL_P_ACC=6.0, cruise=9, max=11)')
    print('='*72)
    print(f'{"blend":>6} {"N":>3} {"p50":>10} {"AvgCmd":>8} {"AvgAch":>8} {"MaxSpd":>8} {"AvgErr":>8} {"Util":>8}')
    print('-'*72)
    summary = {}
    for bl in BLENDS:
        rs = [r for r in results if r['base_blend']==bl and r['completed']]
        n_total = sum(1 for r in results if r['base_blend']==bl)
        if not rs:
            print(f'{bl:>6} {0:>3}  no completions ({n_total} trials total)'); continue
        rs_sorted = sorted(rs, key=lambda x: x['time'])
        med = rs_sorted[len(rs_sorted)//2]
        avg_cmd = sum(r['avg_cmd'] for r in rs)/len(rs)
        avg_ach = sum(r['avg_ach'] for r in rs)/len(rs)
        avg_max = sum(r['max_spd'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[bl] = dict(
            base_blend=bl, n=len(rs), n_total=n_total, time_p50=med['time'],
            avg_cmd=round(avg_cmd,2), avg_ach=round(avg_ach,2), max_spd=round(avg_max,2),
            avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'{bl:>6} {len(rs):>3}/{n_total} {med["time"]:>9.2f}s {avg_cmd:>7.2f} {avg_ach:>7.2f} {avg_max:>7.2f} {avg_err:>7.2f} {avg_util:>7.1%}')

    base = 14.046  # locked-tune technical p50
    print(f'\n{"blend":>6} {"p50":>10} {"Delta":>14}')
    print('-'*72)
    for bl in BLENDS:
        if bl in summary:
            t = summary[bl]['time_p50']
            d = t - base
            pct = d / base * 100
            print(f'{bl:>6} {t:>9.2f}s {d:>+8.2f}s ({pct:+.1f}%)')

    with open(LOG_PATH, 'w') as f:
        json.dump({'trials': results, 'summary': {str(k): v for k,v in summary.items()},
                   'course': COURSE, 'locked_px4_tune': {'MPC_XY_VEL_P_ACC': 6.0},
                   'held_planner_kw': {'cruise_speed': CRUISE, 'max_speed': MAX_SPEED}},
                  f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
