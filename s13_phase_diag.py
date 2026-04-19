#!/usr/bin/env python3
"""Session 13 — Phase-transition diagnostic analysis.

Reads trace_technical.json (locked baseline).
Outputs:
  1. Per-leg phase-transition timeline (when each phase starts/ends, duration)
  2. Per-leg cmd→ach speed gap evolution (where gap opens, peak gap, time to close)
  3. Per-leg heading alignment time (time from gate pass until cmd/ach heading within 15deg)
  4. Per-phase aggregate stats (total time in phase, mean speed gap, etc.)
  5. Loss-bucket ranking: which legs/phases contribute most lap time
"""
import json, math, os, sys
from statistics import median, mean, stdev

TRACE = '/Users/conradweeden/ai-grand-prix/logs/trace_technical.json'
HDG_ALIGN_THRESH = 15.0  # degrees — threshold for "heading aligned"

with open(TRACE) as f:
    data = json.load(f)

samples = data['samples']
gate_events = data['gate_events']
gates = data['gates']
n_gates = len(gates)

def angle_deg(ax, ay, bx, by):
    ma = math.sqrt(ax*ax + ay*ay)
    mb = math.sqrt(bx*bx + by*by)
    if ma < 0.01 or mb < 0.01:
        return 0.0
    dot = (ax*bx + ay*by) / (ma * mb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def turn_angle_deg(gates, gi):
    """Turn angle at gate gi (angle between incoming and outgoing legs)."""
    if gi == 0 or gi >= len(gates) - 1:
        return 0.0
    prev = gates[gi - 1]
    curr = gates[gi]
    nxt = gates[gi + 1]
    ax, ay = curr[0] - prev[0], curr[1] - prev[1]
    bx, by = nxt[0] - curr[0], nxt[1] - curr[1]
    return angle_deg(ax, ay, bx, by)

# Build gate pass time lookup
gate_pass_time = {}
gate_pass_speed = {}
for ge in gate_events:
    gate_pass_time[ge['gate_idx']] = ge['time']
    gate_pass_speed[ge['gate_idx']] = ge.get('ach_speed', 0)

# ---- 1. Per-leg analysis ----
print("=" * 90)
print("PER-LEG PHASE TIMELINE & LOSS ANALYSIS")
print("=" * 90)

for leg in range(n_gates):
    leg_samples = [s for s in samples if s['leg_idx'] == leg]
    if not leg_samples:
        continue

    t_start = leg_samples[0]['t']
    t_end = leg_samples[-1]['t']
    leg_dur = t_end - t_start

    # Gate pass time for this leg (= when we passed gate[leg])
    gp_time = gate_pass_time.get(leg)

    # Turn angle at entry (gate we just passed = leg-1 for leg>0)
    ta = turn_angle_deg(gates, leg) if leg > 0 else 0.0

    print(f"\n--- Leg {leg} (dur={leg_dur:.3f}s, turn_in={ta:.0f}deg) ---")

    # Phase transitions within this leg
    phases_seen = []
    prev_phase = None
    for s in leg_samples:
        ph = s.get('phase')
        if ph != prev_phase:
            phases_seen.append({'phase': ph, 't_start': s['t'], 't_end': s['t'], 'n': 1})
            prev_phase = ph
        else:
            phases_seen[-1]['t_end'] = s['t']
            phases_seen[-1]['n'] += 1

    for ps in phases_seen:
        dur = ps['t_end'] - ps['t_start']
        print(f"  phase={ps['phase']:>10s}  t=[{ps['t_start']:.3f} → {ps['t_end']:.3f}]  dur={dur:.3f}s  samples={ps['n']}")

    # Speed gap analysis
    spd_gaps = [(s['t'], s['cmd_spd'], s['ach_spd'], s['cmd_spd'] - s['ach_spd']) for s in leg_samples]
    if spd_gaps:
        peak_gap = max(spd_gaps, key=lambda x: abs(x[3]))
        mean_gap = mean([abs(g[3]) for g in spd_gaps])
        # Time to close: first sample where gap is < 0.5 m/s after peak
        peak_t = peak_gap[0]
        close_t = None
        for t, cs, acs, gap in spd_gaps:
            if t > peak_t and abs(gap) < 0.5:
                close_t = t
                break
        print(f"  speed_gap: mean_abs={mean_gap:.2f} peak={peak_gap[3]:+.2f}@t={peak_gap[0]:.3f} "
              f"close_t={'%.3f' % close_t if close_t else 'never'} "
              f"(dt={'%.3f' % (close_t - peak_t) if close_t else 'n/a'})")

    # Heading alignment analysis (post-gate-pass)
    if leg > 0:
        # gate_pass_time[leg-1] is when we entered this leg
        entry_t = gate_pass_time.get(leg - 1)
        if entry_t is not None:
            post_gate = [(s['t'], angle_deg(s['cmd_vn'], s['cmd_ve'], s['vel_n'], s['vel_e']))
                         for s in leg_samples if s['t'] >= entry_t]
            if post_gate:
                # Initial heading error
                init_hdg_err = post_gate[0][1] if post_gate else 0
                peak_hdg = max(post_gate, key=lambda x: x[1])
                # Time to align within threshold
                align_t = None
                for t, herr in post_gate:
                    if herr <= HDG_ALIGN_THRESH:
                        align_t = t
                        break
                dt_align = (align_t - entry_t) if align_t else None
                print(f"  heading: init_err={init_hdg_err:.1f}deg peak={peak_hdg[1]:.1f}deg@t={peak_hdg[0]:.3f} "
                      f"align_within_{HDG_ALIGN_THRESH:.0f}deg={'%.3fs' % dt_align if dt_align else 'never'}")

# ---- 2. Per-phase aggregate stats ----
print("\n" + "=" * 90)
print("PER-PHASE AGGREGATE STATS")
print("=" * 90)

phase_buckets = {}
for s in samples:
    ph = s.get('phase', 'unknown')
    if ph not in phase_buckets:
        phase_buckets[ph] = {'samples': [], 'spd_gaps': [], 'hdg_errs': []}
    phase_buckets[ph]['samples'].append(s)
    phase_buckets[ph]['spd_gaps'].append(abs(s['cmd_spd'] - s['ach_spd']))
    herr = angle_deg(s['cmd_vn'], s['cmd_ve'], s['vel_n'], s['vel_e'])
    phase_buckets[ph]['hdg_errs'].append(herr)

total_samples = len(samples)
total_time = samples[-1]['t'] - samples[0]['t']

for ph, bucket in sorted(phase_buckets.items(), key=lambda x: -len(x[1]['samples'])):
    n = len(bucket['samples'])
    pct = n / total_samples * 100
    t_span = bucket['samples'][-1]['t'] - bucket['samples'][0]['t']
    mg = mean(bucket['spd_gaps'])
    mh = mean(bucket['hdg_errs'])
    p90h = sorted(bucket['hdg_errs'])[int(0.9 * len(bucket['hdg_errs']))]
    ach_spds = [s['ach_spd'] for s in bucket['samples']]
    mean_ach = mean(ach_spds)
    cmd_spds = [s['cmd_spd'] for s in bucket['samples']]
    mean_cmd = mean(cmd_spds)
    print(f"  {ph:>10s}: {n:4d} samples ({pct:4.1f}%)  "
          f"mean_cmd_spd={mean_cmd:.2f}  mean_ach_spd={mean_ach:.2f}  "
          f"mean_spd_gap={mg:.2f}  mean_hdg_err={mh:.1f}deg  p90_hdg={p90h:.1f}deg")

# ---- 3. Loss-bucket ranking ----
print("\n" + "=" * 90)
print("LOSS-BUCKET RANKING — Where is time being lost?")
print("=" * 90)
print("\nApproach: compare each leg duration to a theoretical minimum")
print("(distance / max_achievable_speed). The excess is the loss.\n")

max_speed = 12.0  # planner max

for leg in range(n_gates):
    leg_samples = [s for s in samples if s['leg_idx'] == leg]
    if not leg_samples:
        continue
    t_start = leg_samples[0]['t']
    t_end = leg_samples[-1]['t']
    leg_dur = t_end - t_start

    # Leg distance (gate-to-gate)
    if leg == 0:
        # first leg: from starting position (roughly at gate 0 alt) to gate 0
        # use first sample position
        sx, sy = leg_samples[0]['pos_n'], leg_samples[0]['pos_e']
        gx, gy = gates[leg][0], gates[leg][1]
        leg_dist = math.sqrt((gx - sx)**2 + (gy - sy)**2)
    else:
        prev_g = gates[leg - 1]
        curr_g = gates[leg]
        leg_dist = math.sqrt((curr_g[0] - prev_g[0])**2 + (curr_g[1] - prev_g[1])**2)

    theoretical_min = leg_dist / max_speed if max_speed > 0 else 999
    excess = leg_dur - theoretical_min

    ta = turn_angle_deg(gates, leg) if leg > 0 else 0.0

    # What phases were active and their time share
    phase_time = {}
    prev_ph = None
    prev_t = None
    for s in leg_samples:
        ph = s.get('phase', '?')
        if prev_ph is not None and prev_t is not None:
            dt = s['t'] - prev_t
            phase_time[prev_ph] = phase_time.get(prev_ph, 0) + dt
        prev_ph = ph
        prev_t = s['t']
    phase_str = '  '.join(f'{k}={v:.3f}s' for k, v in sorted(phase_time.items(), key=lambda x: -x[1]))

    # Mean achieved speed
    mean_ach = mean([s['ach_spd'] for s in leg_samples])
    mean_cmd = mean([s['cmd_spd'] for s in leg_samples])

    print(f"  Leg {leg:2d}: dur={leg_dur:.3f}s  dist={leg_dist:.1f}m  turn={ta:.0f}deg  "
          f"excess={excess:+.3f}s  mean_ach={mean_ach:.1f}  mean_cmd={mean_cmd:.1f}")
    print(f"          phases: {phase_str}")

# ---- 4. Key question: is remaining loss planner-addressable? ----
print("\n" + "=" * 90)
print("ACHIEVABILITY ANALYSIS")
print("=" * 90)

# Compare cmd_spd to ach_spd across entire lap
all_cmd = [s['cmd_spd'] for s in samples]
all_ach = [s['ach_spd'] for s in samples]
all_gap = [c - a for c, a in zip(all_cmd, all_ach)]

print(f"  Whole-lap cmd_spd: mean={mean(all_cmd):.2f}  median={median(all_cmd):.2f}")
print(f"  Whole-lap ach_spd: mean={mean(all_ach):.2f}  median={median(all_ach):.2f}")
print(f"  Whole-lap gap:     mean={mean(all_gap):+.2f}  median={median(all_gap):+.2f}  "
      f"p90={sorted(all_gap)[int(0.9*len(all_gap)):int(0.9*len(all_gap))+1][0]:+.2f}")

# Fraction of time PX4 is within 1 m/s of cmd
within_1 = sum(1 for g in all_gap if abs(g) < 1.0) / len(all_gap) * 100
within_2 = sum(1 for g in all_gap if abs(g) < 2.0) / len(all_gap) * 100
print(f"  PX4 within 1m/s of cmd: {within_1:.0f}%")
print(f"  PX4 within 2m/s of cmd: {within_2:.0f}%")

# Heading alignment stats
all_hdg = [angle_deg(s['cmd_vn'], s['cmd_ve'], s['vel_n'], s['vel_e']) for s in samples]
within_15 = sum(1 for h in all_hdg if h < 15) / len(all_hdg) * 100
within_30 = sum(1 for h in all_hdg if h < 30) / len(all_hdg) * 100
print(f"  Heading within 15deg: {within_15:.0f}%")
print(f"  Heading within 30deg: {within_30:.0f}%")
print(f"  Heading err: mean={mean(all_hdg):.1f}deg  median={median(all_hdg):.1f}deg  "
      f"p90={sorted(all_hdg)[int(0.9*len(all_hdg))]:.1f}deg  max={max(all_hdg):.1f}deg")

print("\n--- END DIAGNOSTIC ---")
