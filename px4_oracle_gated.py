"""
px4_oracle_gated.py

Freeze the new official PX4 baseline under:
  - gated takeoff (use_takeoff_gate=True, default thresholds)
  - VEL_P_ACC = 6.0 (assumed already set on the PX4 params side)
  - V5.1 planner (max_speed=11.0, cruise_speed=9.0, base_blend=1.5)

5 trials technical + 5 trials mixed.
Fresh PX4 restart per trial. Broad try/except. asyncio.wait_for per trial.
Incremental JSON dump. Prints median/mean/stddev/min/max per course.

This snapshot becomes the reference every future SHORT-phase experiment is
compared against. Do NOT tweak gate_timeout / gate_alt_frac here — those are
queued for later, after SHORT work stabilizes.
"""
import asyncio, json, os, statistics, time, traceback

from px4_v51_baseline import V51Planner, COURSES as _COURSES_DICT, run_trial, restart_px4

TRIALS_PER_COURSE = 5
PER_TRIAL_TIMEOUT = 180.0
COURSES = [
    ('technical', _COURSES_DICT['technical']),
    ('mixed',     _COURSES_DICT['mixed']),
]
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
OUT_JSON = '/Users/conradweeden/ai-grand-prix/logs/px4_oracle_gated.json'


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    return {
        'n':      len(xs),
        'median': round(statistics.median(xs), 4),
        'mean':   round(statistics.mean(xs),   4),
        'stdev':  round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        'min':    round(min(xs), 4),
        'max':    round(max(xs), 4),
    }


async def run_one(course_name, gates, trial_idx):
    planner = V51Planner(**PLANNER_KW)
    try:
        restart_px4()
    except Exception as e:
        return {'course': course_name, 'trial': trial_idx,
                'error': f'restart_px4: {e}', 'trace': traceback.format_exc()}
    try:
        result = await asyncio.wait_for(
            run_trial(planner, gates, use_takeoff_gate=True),
            timeout=PER_TRIAL_TIMEOUT,
        )
        result['course'] = course_name
        result['trial']  = trial_idx
        return result
    except asyncio.TimeoutError:
        return {'course': course_name, 'trial': trial_idx,
                'error': f'timeout>{PER_TRIAL_TIMEOUT}s'}
    except Exception as e:
        return {'course': course_name, 'trial': trial_idx,
                'error': str(e), 'trace': traceback.format_exc()}


def _dump(results):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'results': results, 'ts': time.time()}, f, indent=2, default=str)


async def main():
    results = []
    for course_name, gates in COURSES:
        print(f'\n===== {course_name.upper()} =====')
        for i in range(TRIALS_PER_COURSE):
            print(f'[{course_name} trial {i+1}/{TRIALS_PER_COURSE}] start')
            r = await run_one(course_name, gates, i)
            results.append(r)
            _dump(results)
            if 'error' in r:
                print(f'  ERR {r["error"]}')
            else:
                print(f'  lap={r.get("lap")} leg0={r.get("leg0_dur")} '
                      f'gate0={r.get("gate0_time")} t_off={r.get("t_offboard_wait")} '
                      f'alt={r.get("alt_at_offboard")} vz={r.get("vz_at_offboard")} '
                      f'completed={r.get("completed")}')
    # summary
    print('\n' + '=' * 70)
    print('ORACLE SUMMARY (gated takeoff, VEL_P_ACC=6.0, V5.1 planner)')
    print('=' * 70)
    for course_name, _ in COURSES:
        cs = [r for r in results if r.get('course') == course_name and 'error' not in r]
        lap   = [r.get('lap')      for r in cs]
        leg0  = [r.get('leg0_dur') for r in cs]
        leg1  = [r.get('leg1_dur') for r in cs]
        gate0 = [r.get('gate0_time') for r in cs]
        comp  = sum(1 for r in cs if r.get('completed'))
        errs  = sum(1 for r in results if r.get('course') == course_name and 'error' in r)
        print(f'\n--- {course_name.upper()} ---   completion {comp}/{TRIALS_PER_COURSE}   errors {errs}')
        print('  lap      :', _stats(lap))
        print('  leg0_dur :', _stats(leg0))
        print('  leg1_dur :', _stats(leg1))
        print('  gate0    :', _stats(gate0))
        print('  t_off    :', [r.get('t_offboard_wait') for r in cs])
        print('  alt@off  :', [r.get('alt_at_offboard') for r in cs])
        print('  vz@off   :', [r.get('vz_at_offboard')  for r in cs])
    print('\nSaved to', OUT_JSON)


if __name__ == '__main__':
    asyncio.run(main())
