"""Paired interleaved confirmation: base_blend 1.5 vs 3.0 on technical.

Order: 1.5, 3.0, 1.5, 3.0, ... (10 trials total, 5 each).
Fresh PX4 restart between every trial.
Same PX4 tune (VEL_P_ACC=6.0), same cruise=9/max=11.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B

COURSE = 'technical'
PAIR = [1.5, 3.0]
N_PAIRS = 5
LOG_PATH = '/Users/conradweeden/ai-grand-prix/logs/px4_blend_confirm.json'
TRIAL_TIMEOUT = 150.0
CRUISE = 9.0
MAX_SPEED = 11.0

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
    print(f'\nPAIRED CONFIRMATION: base_blend {PAIR[0]} vs {PAIR[1]}')
    print(f'Interleaved order, {N_PAIRS} pairs = {N_PAIRS*2} trials')
    print(f'cruise={CRUISE}, max={MAX_SPEED}, VEL_P_ACC=6.0')
    print('='*72)

    results = []
    seq = []
    for i in range(N_PAIRS):
        seq.extend(PAIR)  # [1.5, 3.0, 1.5, 3.0, ...]

    for idx, bl in enumerate(seq):
        print(f'\n[trial {idx+1}/{len(seq)}  blend={bl}]...', end=' ', flush=True)
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
        r['seq_idx'] = idx
        results.append(r)
        if r.get('harness_timeout'):
            st = 'HANG'
        else:
            st = 'OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
        print(f'{st} {r["time"]}s | Cmd:{r.get("avg_cmd",0):.2f} Ach:{r.get("avg_ach",0):.2f} Err:{r.get("avg_err",0)} Util:{r.get("util",0):.1%}')
        B.restart_px4()
        await asyncio.sleep(2)

    # Summary
    print(f'\n\n{"="*72}')
    print(f'PAIRED CONFIRMATION SUMMARY')
    print('='*72)
    summary = {}
    for bl in PAIR:
        rs = [r for r in results if r['base_blend']==bl and r['completed']]
        times = sorted([r['time'] for r in rs])
        if not times:
            print(f'blend={bl}: no completions'); continue
        med = times[len(times)//2]
        mean = sum(times)/len(times)
        # sample std dev
        if len(times) > 1:
            m = mean
            sd = (sum((t-m)**2 for t in times) / (len(times)-1)) ** 0.5
        else:
            sd = 0.0
        mn, mx = min(times), max(times)
        avg_cmd = sum(r['avg_cmd'] for r in rs)/len(rs)
        avg_ach = sum(r['avg_ach'] for r in rs)/len(rs)
        avg_err = sum(r['avg_err'] for r in rs)/len(rs)
        avg_util= sum(r['util']    for r in rs)/len(rs)
        summary[bl] = dict(
            base_blend=bl, n=len(rs), times=times,
            median=med, mean=round(mean,3), sd=round(sd,3),
            min=mn, max=mx,
            avg_cmd=round(avg_cmd,2), avg_ach=round(avg_ach,2),
            avg_err=round(avg_err,2), util=round(avg_util,3),
        )
        print(f'blend={bl}:  n={len(rs)}  median={med:.3f}  mean={mean:.3f}  sd={sd:.3f}  min={mn:.3f}  max={mx:.3f}')
        print(f'           Cmd={avg_cmd:.2f}  Ach={avg_ach:.2f}  Err={avg_err:.2f}  Util={avg_util:.1%}')
        print(f'           times: {[round(t,3) for t in times]}')

    # Paired delta via sequential pairing (trial_i of 1.5 vs trial_i of 3.0)
    if PAIR[0] in summary and PAIR[1] in summary:
        med_a = summary[PAIR[0]]['median']
        med_b = summary[PAIR[1]]['median']
        delta_med = med_b - med_a
        sd_a = summary[PAIR[0]]['sd']
        sd_b = summary[PAIR[1]]['sd']
        print(f'\nDELTA (blend={PAIR[1]} - blend={PAIR[0]}):')
        print(f'  median:   {delta_med:+.3f}s  ({delta_med/med_a*100:+.1f}%)')
        print(f'  sd 1.5:   {sd_a:.3f}s')
        print(f'  sd 3.0:   {sd_b:.3f}s')
        # Decision rule
        passes_speed = delta_med <= -0.25  # 3.0 at least 0.25s faster
        passes_var = sd_b <= sd_a * 1.5 + 0.05  # var not materially worse
        print(f'\nDECISION RULE:')
        print(f'  blend=3.0 faster by >=0.25s on median:  {"PASS" if passes_speed else "FAIL"} ({delta_med:+.3f}s)')
        print(f'  blend=3.0 variance not worse than 1.5x(1.5): {"PASS" if passes_var else "FAIL"} ({sd_b:.3f} vs {sd_a:.3f})')
        print(f'  ==> {"LOCK blend=3.0" if (passes_speed and passes_var) else "STAY at blend=1.5 (drift-confounded)"}')

    with open(LOG_PATH, 'w') as f:
        # convert int keys for JSON
        safe_summary = {str(k): v for k,v in summary.items()}
        json.dump({'trials': results, 'summary': safe_summary,
                   'course': COURSE, 'locked_px4_tune': {'MPC_XY_VEL_P_ACC': 6.0},
                   'held_planner_kw': {'cruise_speed': CRUISE, 'max_speed': MAX_SPEED},
                   'pair': PAIR, 'n_pairs': N_PAIRS}, f, indent=2)
    print(f'\nSaved to {LOG_PATH}')

if __name__ == '__main__':
    asyncio.run(main())
