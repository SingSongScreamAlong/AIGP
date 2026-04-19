"""Session 16 — Vision navigation test harness.

Runs the technical course under:
  1. Ground truth (existing V5.1 baseline)
  2. Vision nav — clean noise
  3. Vision nav — mild noise
  4. Vision nav — harsh noise

3 trials per condition. Measures: completion, lap time, search/coast frames.
Uses bench.py for restart/singleton management.
"""

import asyncio, json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench

# Import planner for ground truth arm
from px4_v51_baseline import V51Planner, COURSES

# Import vision stack
from vision_nav import (
    run_trial_vision, run_trial_groundtruth,
    VirtualCamera, GateTracker, VisionNav,
)

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 's16_vision_test.json')
COURSE = 'technical'
GATES = COURSES[COURSE]
N_TRIALS = 3
CONDITIONS = ['groundtruth', 'clean', 'mild', 'harsh']

B = os.path.expanduser('~/PX4-Autopilot')


async def run_one(condition, trial_num, results):
    tag = f'[{condition} t{trial_num}]'
    print(f'{tag} hardened restart...', flush=True)
    try:
        bench.hardened_restart(B)
    except Exception as e:
        print(f'{tag} BENCH FAIL: {e}', flush=True)
        results.append({
            'condition': condition, 'trial': trial_num,
            'completed': False, 'error': f'restart: {e}',
            'bench_failure': True,
        })
        bench.atomic_write_json(RESULT, results)
        return

    print(f'{tag} flying...', flush=True)
    try:
        if condition == 'groundtruth':
            planner = V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
            planner.px4_speed_ceiling = 9.5
            r = await run_trial_groundtruth(planner, GATES)
        else:
            r = await run_trial_vision(GATES, noise_profile=condition)

        r['condition'] = condition
        r['trial'] = trial_num
        r['bench_failure'] = False
        print(f'{tag} -> lap={r["time"]} gates={r["gates_passed"]}/{len(GATES)} '
              f'done={r["completed"]} search={r.get("search_pct",0)}% '
              f'coast={r.get("coast_pct",0)}%', flush=True)

    except Exception as e:
        r = {
            'condition': condition, 'trial': trial_num,
            'completed': False, 'error': str(e),
            'bench_failure': False,
        }
        print(f'{tag} ERROR: {e}', flush=True)

    results.append(r)
    bench.atomic_write_json(RESULT, results)


async def main():
    bench.acquire_singleton('vis16')
    print(f'[init] Session 16 — Vision Navigation Test', flush=True)
    print(f'[init] Course: {COURSE} ({len(GATES)} gates)', flush=True)
    print(f'[init] Conditions: {CONDITIONS}', flush=True)
    print(f'[init] Trials per condition: {N_TRIALS}', flush=True)
    print(f'[init] Result file: {RESULT}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    # Run all conditions
    for condition in CONDITIONS:
        print(f'\n=== {condition.upper()} ({N_TRIALS} trials) ===', flush=True)
        for t in range(1, N_TRIALS + 1):
            await run_one(condition, t, results)

    # Summary
    print(f'\n=== SUMMARY ===', flush=True)
    from statistics import median, mean
    for cond in CONDITIONS:
        trials = [r for r in results if r['condition'] == cond and r.get('completed')]
        fails = [r for r in results if r['condition'] == cond and not r.get('completed')]
        if trials:
            times = [r['time'] for r in trials]
            print(f'  {cond:12s}  completed={len(trials)}/{len(trials)+len(fails)}  '
                  f'median={median(times):.3f}  mean={mean(times):.3f}  '
                  f'search={mean([r.get("search_pct",0) for r in trials]):.1f}%  '
                  f'coast={mean([r.get("coast_pct",0) for r in trials]):.1f}%')
        else:
            print(f'  {cond:12s}  completed=0/{len(fails)}  NO DATA')

    bench.atomic_write_json(RESULT, results)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
