#!/usr/bin/env python3
"""
px4_gated_confirm.py

Confirmation A/B for the gated-takeoff promotion.
A = OLD (use_takeoff_gate=False -> blind sleep(4))
B = NEW (use_takeoff_gate=True  -> altitude-settle gate)

Runs paired/interleaved 5 pairs on technical, then 5 pairs on mixed.
Fresh PX4 restart per trial. Broad try/except per trial.
Alternating order per pair to remove first/second-trial bias.

Decision rule (external):
  promote globally if
    technical still wins (median Δ = NEW - OLD < 0)
    mixed neutral-or-better (median Δ <= ~0, or ≥3/5 pairs NEW≤OLD)
    no completion regression
    no new startup weirdness
"""

import asyncio, json, os, sys, time, traceback, statistics
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')

from px4_v51_baseline import (
    V51Planner, COURSES, run_trial, restart_px4,
)

PAIRS = 5
COURSES_TO_RUN = ['technical', 'mixed']
PER_TRIAL_TIMEOUT = 180.0  # seconds, hard hang guard
PLANNER_KW = dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)
OUT_JSON = '/Users/conradweeden/ai-grand-prix/logs/px4_gated_confirm.json'


def summarize_trial(r, label, course, pair_idx, order_idx):
    splits = r.get('splits') or []
    leg0 = splits[0] if len(splits) >= 1 else None
    leg1 = (splits[1] - splits[0]) if len(splits) >= 2 else None
    return dict(
        label=label, course=course, pair=pair_idx, order=order_idx,
        completed=bool(r.get('completed')),
        gates_passed=r.get('gates_passed'),
        lap=r.get('time'),
        t_offboard_wait=r.get('t_offboard_wait'),
        alt_at_offboard=r.get('alt_at_offboard'),
        vz_at_offboard=r.get('vz_at_offboard'),
        t_gate0=leg0,
        leg0_dur=leg0,
        leg1_dur=leg1,
        max_spd=r.get('max_spd'),
        avg_ach=r.get('avg_ach'),
        avg_err=r.get('avg_err'),
        util=r.get('util'),
        splits=splits,
    )


def err_result(label, course, pair_idx, order_idx, exc):
    return dict(
        label=label, course=course, pair=pair_idx, order=order_idx,
        completed=False, error=repr(exc),
        tb=traceback.format_exc()[-2000:],
    )


async def run_one(label, course, pair_idx, order_idx):
    use_gate = (label == 'NEW')
    gates = COURSES[course]
    planner = V51Planner(**PLANNER_KW)
    try:
        r = await asyncio.wait_for(
            run_trial(planner, gates, use_takeoff_gate=use_gate),
            timeout=PER_TRIAL_TIMEOUT,
        )
        return summarize_trial(r, label, course, pair_idx, order_idx)
    except Exception as e:
        print(f'    !! TRIAL EXC {label} pair{pair_idx} order{order_idx}: {e!r}', flush=True)
        return err_result(label, course, pair_idx, order_idx, e)
    finally:
        os.system('pkill -9 -f mavsdk_server 2>/dev/null')
        os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
        os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")


async def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print('[init] cleaning stale processes...', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('pkill -9 -f px4 2>/dev/null')
    os.system('pkill -9 -f sihsim 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    print('[init] first PX4 restart...', flush=True)
    restart_px4()
    await asyncio.sleep(2)

    all_results = []

    for course in COURSES_TO_RUN:
        print(f'\n{"="*66}')
        print(f'COURSE: {course.upper()}   pairs={PAIRS}   labels: OLD (sleep4) vs NEW (gate)')
        print(f'{"="*66}', flush=True)

        for pair_idx in range(PAIRS):
            # alternate order per pair to defuse first/second bias
            order = ('OLD', 'NEW') if (pair_idx % 2 == 0) else ('NEW', 'OLD')
            print(f'\n-- pair {pair_idx+1}/{PAIRS} on {course}: order={order} --', flush=True)

            for oi, lab in enumerate(order):
                print(f'  [{course} pair{pair_idx+1} {lab}] starting...', flush=True)
                # fresh PX4 each trial
                restart_px4()
                await asyncio.sleep(2)
                res = await run_one(lab, course, pair_idx, oi)
                all_results.append(res)
                ok = res.get('completed')
                lap = res.get('lap')
                tw = res.get('t_offboard_wait')
                a = res.get('alt_at_offboard')
                v = res.get('vz_at_offboard')
                leg0 = res.get('leg0_dur')
                leg1 = res.get('leg1_dur')
                print(
                    f'  [{course} pair{pair_idx+1} {lab}] '
                    f'done: ok={ok} lap={lap} t_off={tw} '
                    f'alt@off={a} vz@off={v} leg0={leg0} leg1={leg1}',
                    flush=True,
                )
                # incremental dump so a later crash cannot lose prior trials
                with open(OUT_JSON, 'w') as f:
                    json.dump({'results': all_results}, f, indent=2)

    # ---------- summaries ----------
    def per_course_summary(course):
        rs = [r for r in all_results if r.get('course') == course]
        old = [r for r in rs if r.get('label') == 'OLD']
        new = [r for r in rs if r.get('label') == 'NEW']
        old_by_pair = {r['pair']: r for r in old}
        new_by_pair = {r['pair']: r for r in new}

        def med(xs):
            xs = [x for x in xs if x is not None]
            return statistics.median(xs) if xs else None

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs)/len(xs) if xs else None

        pairs = sorted(set(old_by_pair) & set(new_by_pair))
        delta_lap, delta_g0, delta_leg0, delta_leg1 = [], [], [], []
        for p in pairs:
            a, b = old_by_pair[p], new_by_pair[p]
            if a.get('completed') and b.get('completed'):
                if a.get('lap') is not None and b.get('lap') is not None:
                    delta_lap.append(b['lap'] - a['lap'])
                if a.get('t_gate0') is not None and b.get('t_gate0') is not None:
                    delta_g0.append(b['t_gate0'] - a['t_gate0'])
                if a.get('leg0_dur') is not None and b.get('leg0_dur') is not None:
                    delta_leg0.append(b['leg0_dur'] - a['leg0_dur'])
                if a.get('leg1_dur') is not None and b.get('leg1_dur') is not None:
                    delta_leg1.append(b['leg1_dur'] - a['leg1_dur'])

        comp_old = sum(1 for r in old if r.get('completed'))
        comp_new = sum(1 for r in new if r.get('completed'))
        err_old = [r for r in old if 'error' in r]
        err_new = [r for r in new if 'error' in r]

        s = {
            'course': course,
            'n_pairs': len(pairs),
            'completion_old': f'{comp_old}/{len(old)}',
            'completion_new': f'{comp_new}/{len(new)}',
            'errors_old': len(err_old),
            'errors_new': len(err_new),
            'delta_lap_median':  round(med(delta_lap), 3)  if delta_lap else None,
            'delta_lap_mean':    round(mean(delta_lap), 3) if delta_lap else None,
            'delta_gate0_median':round(med(delta_g0), 3)   if delta_g0 else None,
            'delta_leg0_median': round(med(delta_leg0),3)  if delta_leg0 else None,
            'delta_leg1_median': round(med(delta_leg1),3)  if delta_leg1 else None,
            'old_laps': [r.get('lap') for r in sorted(old, key=lambda r:r['pair'])],
            'new_laps': [r.get('lap') for r in sorted(new, key=lambda r:r['pair'])],
            'new_t_offboard_wait': [r.get('t_offboard_wait') for r in sorted(new, key=lambda r:r['pair'])],
            'new_alt_at_offboard': [r.get('alt_at_offboard') for r in sorted(new, key=lambda r:r['pair'])],
            'new_vz_at_offboard': [r.get('vz_at_offboard') for r in sorted(new, key=lambda r:r['pair'])],
            'old_t_offboard_wait': [r.get('t_offboard_wait') for r in sorted(old, key=lambda r:r['pair'])],
        }
        return s

    summaries = [per_course_summary(c) for c in COURSES_TO_RUN]
    out = {'results': all_results, 'summaries': summaries}
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n\n{"#"**70}')
    print('GATED TAKEOFF CONFIRMATION — SUMMARY (paired, NEW - OLD)')
    print(f'{"#"**70}')
    for s in summaries:
        print(f'\n[{s["course"].upper()}] pairs={s["n_pairs"]}  '
              f'completion OLD={s["completion_old"]}  NEW={s["completion_new"]}  '
              f'errors OLD={s["errors_old"]} NEW={s["errors_new"]}')
        print(f'  Δlap   median={s["delta_lap_median"]}  mean={s["delta_lap_mean"]}')
        print(f'  Δgate0 median={s["delta_gate0_median"]}')
        print(f'  Δleg0  median={s["delta_leg0_median"]}')
        print(f'  Δleg1  median={s["delta_leg1_median"]}')
        print(f'  NEW t_offboard_wait : {s["new_t_offboard_wait"]}')
        print(f'  NEW alt_at_offboard : {s["new_alt_at_offboard"]}')
        print(f'  NEW vz_at_offboard  : {s["new_vz_at_offboard"]}')
        print(f'  OLD laps : {s["old_laps"]}')
        print(f'  NEW laps : {s["new_laps"]}')
    print(f'\n[done] wrote {OUT_JSON}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
