#!/usr/bin/env python3
"""Loss-bucket analyzer on the new locked baseline traces.

Buckets:
  leg0_residual      - everything in leg 0 (LAUNCH + ramp to cruise)
  leg1_residual      - everything in leg 1 (first steady cruise leg)
  SHORT_gap          - phase=='SHORT', legs >= 2
  PRE_TURN_gap       - phase=='PRE_TURN', legs >= 2
  TURN_gap           - phase=='TURN', legs >= 2
  SUSTAIN_gap        - phase=='SUSTAIN', legs >= 2 and not last leg
  exit_rebuild       - everything in the LAST leg (usually util collapses)

Per-sample loss proxy = dt * (1 - ach/cmd) when cmd > 0, else 0.
Sum per bucket gives seconds of recoverable time if cmd-ach gap was closed.

Also reports per-leg absolute err*dt budget and ach/cmd utilization.
"""
import csv, json, os, sys
from collections import defaultdict

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
DT = 1/50

def load(course):
    path = os.path.join(LOGS, f'trace_{course}.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('t','cmd_spd','ach_spd','err_spd','dxy_to_gate','dist_to_gate',
                  'turn_angle_rad','blend_radius','desired_pre_ceiling',
                  'cmd_pre_smooth','pos_n','pos_e','pos_d','vel_n','vel_e','vel_d'):
            if k in r:
                try: r[k] = float(r[k])
                except: pass
        r['leg_idx'] = int(r['leg_idx'])
    return rows

def sample_loss(r):
    c = r['cmd_spd']; a = r['ach_spd']
    if c <= 0.01: return 0.0
    return DT * max(0.0, 1.0 - a/c)

def bucketize(rows):
    last_leg = max(r['leg_idx'] for r in rows)
    buckets = defaultdict(lambda: {'dur': 0.0, 'loss': 0.0, 'n': 0,
                                   'cmd_sum': 0.0, 'ach_sum': 0.0})
    for r in rows:
        L = r['leg_idx']; p = r['phase']
        if   L == 0:        b = 'leg0_residual'
        elif L == 1:        b = 'leg1_residual'
        elif L == last_leg: b = 'exit_rebuild'
        elif p == 'SHORT':    b = 'SHORT_gap'
        elif p == 'PRE_TURN': b = 'PRE_TURN_gap'
        elif p == 'TURN':     b = 'TURN_gap'
        elif p == 'SUSTAIN':  b = 'SUSTAIN_gap'
        elif p == 'LAUNCH':   b = 'leg0_residual'  # shouldn't happen past leg 0
        else: b = 'OTHER'
        s = buckets[b]
        s['dur']  += DT
        s['loss'] += sample_loss(r)
        s['n']    += 1
        s['cmd_sum'] += r['cmd_spd']
        s['ach_sum'] += r['ach_spd']
    return buckets, last_leg

def per_leg_table(rows):
    last_leg = max(r['leg_idx'] for r in rows)
    stats = defaultdict(lambda: {'n':0,'cmd':0,'ach':0,'loss':0,'t0':None,'t1':None,'ta':0})
    for r in rows:
        L = r['leg_idx']; s = stats[L]
        s['n'] += 1
        s['cmd'] += r['cmd_spd']
        s['ach'] += r['ach_spd']
        s['loss'] += sample_loss(r)
        if s['t0'] is None: s['t0'] = r['t']
        s['t1'] = r['t']
        if r['turn_angle_rad'] > s['ta']: s['ta'] = r['turn_angle_rad']
    return stats, last_leg

def report(course):
    rows = load(course)
    buckets, last_leg = bucketize(rows)
    stats, _ = per_leg_table(rows)
    total_loss = sum(b['loss'] for b in buckets.values())
    total_dur  = sum(b['dur']  for b in buckets.values())

    print(f'\n=== {course.upper()} LOSS BUCKETS (last_leg={last_leg}) ===')
    print(f'  captured duration: {total_dur:6.2f}s   total est. loss: {total_loss:6.2f}s')
    print(f'  {"bucket":<16} {"dur":>7} {"frac":>7} {"loss":>7} {"%tot":>7} {"util":>7}')
    ordered = sorted(buckets.items(), key=lambda kv: -kv[1]['loss'])
    for name, s in ordered:
        util = s['ach_sum']/s['cmd_sum'] if s['cmd_sum']>0 else 0
        pct  = 100*s['loss']/max(total_loss, 1e-9)
        frac = s['dur']/max(total_dur, 1e-9)
        print(f'  {name:<16} {s["dur"]:6.2f}s {frac:6.1%} {s["loss"]:6.2f}s {pct:6.1f}% {util:6.1%}')

    print(f'\n  PER-LEG (loss in seconds, util=ach/cmd)')
    print(f'  {"leg":>3} {"dur":>6} {"util":>6} {"loss":>6} {"ta_max":>7}')
    for L in sorted(stats.keys()):
        s = stats[L]
        util = s['ach']/s['cmd'] if s['cmd']>0 else 0
        dur = (s['t1'] or 0) - (s['t0'] or 0)
        print(f'  {L:>3} {dur:5.2f}s {util:6.1%} {s["loss"]:5.2f}s {s["ta"]:6.2f}')

    return {'course': course, 'buckets': {k: dict(v) for k, v in buckets.items()},
            'per_leg': {k: dict(v) for k, v in stats.items()}, 'total_loss': total_loss}

def main():
    out = {}
    for c in ('technical', 'mixed'):
        out[c] = report(c)
    with open(os.path.join(LOGS, 'loss_buckets.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {LOGS}/loss_buckets.json')

    # ranking across both
    print('\n=== CROSS-COURSE RANK (loss seconds) ===')
    print(f'  {"bucket":<16} {"technical":>10} {"mixed":>10} {"sum":>10}')
    keys = set()
    for c in out: keys.update(out[c]['buckets'].keys())
    combined = []
    for k in keys:
        t = out['technical']['buckets'].get(k, {}).get('loss', 0)
        m = out['mixed']['buckets'].get(k, {}).get('loss', 0)
        combined.append((k, t, m, t+m))
    for k, t, m, s in sorted(combined, key=lambda x: -x[3]):
        print(f'  {k:<16} {t:9.2f}s {m:9.2f}s {s:9.2f}s')

if __name__ == '__main__':
    main()
