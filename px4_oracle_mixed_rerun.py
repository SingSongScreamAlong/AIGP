"""
px4_oracle_mixed_rerun.py

Option-1 falsification: mixed-only, locked gated-takeoff config, with an extra
5s post-restart cooldown before arm/takeoff/gate. Everything else identical to
px4_oracle_gated.py.

Decision rule:
  5/5 clean  -> lock mixed oracle, move on
  still flaky -> escalate to option 2 (interleaved cadence test)
"""
import asyncio, json, os, statistics, time, traceback

from px4_v51_baseline import V51Planner, COURSES as _COURSES, run_trial, restart_px4

TRIALS            = 5
PER_TRIAL_TIMEOUT = 200.0
POST_RESTART_WAIT = 5.0   # <-- the only change vs px4_oracle_gated.py
PLANNER_KW        = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
GATES             = _COURSES['mixed']
OUT_JSON          = '/Users/conradweeden/ai-grand-prix/logs/px4_oracle_mixed_rerun.json'


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    return {
        'n':      len(xs),
        'median': round(statistics.median(xs), 4),
        'mean':   round(statistics.mean(xs), 4),
        'stdev':  round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        'min':    round(min(xs), 4),
        'max':    round(max(xs), 4),
    }


async def run_one(trial_idx):
    planner = V51Planner(**PLANNER_KW)
    try:
        restart_px4()
    except Exception as e:
        return {'trial': trial_idx, 'error': f'restart_px4: {e}',
                'trace': traceback.format_exc()}
    # the one change: give PX4 + mavsdk_server + UDP port a 5s grace period
    await asyncio.sleep(POST_RESTART_WAIT)
    try:
        result = await asyncio.wait_for(
            run_trial(planner, GATES, use_takeoff_gate=True),
            timeout=PER_TRIAL_TIMEOUT,
        )
        result['trial'] = trial_idx
        return result
    except asyncio.TimeoutError:
        return {'trial': trial_idx, 'error': f'timeout>{PER_TRIAL_TIMEOUT}s'}
    except Exception as e:
        return {'trial': trial_idx, 'error': str(e),
                'trace': traceback.format_exc()}


def _dump(results):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'results': results, 'config': {
            'trials': TRIALS,
            'per_trial_timeout': PER_TRIAL_TIMEOUT,
            'post_restart_wait': POST_RESTART_WAIT,
            'planner_kw': PLANNER_KW,
        }, 'ts': time.time()}, f, indent=2, default=str)


async def main():
    results = []
    print(f'===== MIXED RERUN (post_restart_wait={POST_RESTART_WAIT}s) =====')
    for i in range(TRIALS):
        print(f'[mixed trial {i+1}/{TRIALS}] start')
        r = await run_one(i)
        results.append(r)
        _dump(results)
        if 'error' in r:
            print(f'  ERR {r["error"]}')
        else:
            print(f'  lap={r.get("time")} leg0={(r.get("splits") or [None])[0]} '
                  f't_off={r.get("t_offboard_wait")} alt={r.get("alt_at_offboard")} '
                  f'vz={r.get("vz_at_offboard")} completed={r.get("completed")}')
    print('\n' + '=' * 70)
    print('MIXED RERUN SUMMARY (gated takeoff + 5s post-restart cooldown)')
    print('=' * 70)
    ok  = [r for r in results if r.get('completed') and 'error' not in r]
    err = [r for r in results if 'error' in r]
    bad = [r for r in results if 'error' not in r and not r.get('completed')]
    print(f'completion {len(ok)}/{TRIALS}  errors={len(err)}  non-complete={len(bad)}')
    if ok:
        lap  = [r['time'] for r in ok]
        leg0 = [r['splits'][0] for r in ok]
        print('  lap      :', _stats(lap))
        print('  leg0_dur :', _stats(leg0))
        print('  t_off    :', [r.get('t_offboard_wait') for r in ok])
        print('  alt@off  :', [r.get('alt_at_offboard') for r in ok])
        print('  vz@off   :', [r.get('vz_at_offboard')  for r in ok])
    print('\nSaved to', OUT_JSON)


if __name__ == '__main__':
    asyncio.run(main())
