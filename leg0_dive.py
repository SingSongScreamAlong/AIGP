# Leg-0 deep-dive analyzer.
# Reads trace_{course}.json and decomposes leg 0 into sub-buckets:
#   1. vertical_settle:    |vel_d| > VZ_SETTLE  (still settling vertically)
#   2. horiz_undercommand: cmd_spd < UNDER_CMD  (planner not yet pushing)
#   3. ramp_gap:           vz settled, cmd_spd > UNDER_CMD, ach_spd < cmd_spd*0.85
#   4. tracking_ok:        the rest (planner asking, vehicle delivering)
#
# Also reports:
#   - t_offboard_wait, alt_at_offboard, vz_at_offboard
#   - time-to-first-useful-motion (ach_spd > 1.0)
#   - cmd_spd / ach_spd / vel_d trajectories at 50ms grid for the first 2s
import json, os, sys
from collections import defaultdict

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
DT = 1/50

VZ_SETTLE = 0.3      # m/s — above this, vertical is still settling
UNDER_CMD = 5.0      # m/s — below this, planner is in cold ramp
TRACK_TOL = 0.85     # ach >= cmd*TRACK_TOL counts as tracking_ok

def analyze(course):
    p = os.path.join(LOGS, f'trace_{course}.json')
    d = json.load(open(p))
    samples = d['samples']
    leg0 = [s for s in samples if s['leg_idx'] == 0]
    n = len(leg0)
    print(f'\n=== {course.upper()} LEG-0 DIVE ===')
    print(f'  total_lap         {d["total_time"]}s   completion={d["completed"]}')
    print(f'  t_offboard_wait   {d["t_offboard_wait"]}s   alt@off={d["alt_at_offboard"]}m   vz@off={d["vz_at_offboard"]}m/s')
    print(f'  leg0_n={n}  leg0_dur≈{round(n*DT,3)}s')

    if not leg0:
        print('  EMPTY')
        return

    # Sub-bucket classification
    buckets = defaultdict(lambda: {'n': 0, 'cmd_sum': 0.0, 'ach_sum': 0.0, 'vz_sum': 0.0})
    t_first_motion = None
    for s in leg0:
        cmd = s['cmd_spd']; ach = s['ach_spd']; vz = abs(s['vel_d'])
        if t_first_motion is None and ach > 1.0:
            t_first_motion = s['t']
        if vz > VZ_SETTLE:
            b = 'vertical_settle'
        elif cmd < UNDER_CMD:
            b = 'horiz_undercommand'
        elif ach < cmd * TRACK_TOL:
            b = 'ramp_gap'
        else:
            b = 'tracking_ok'
        buckets[b]['n']       += 1
        buckets[b]['cmd_sum'] += cmd
        buckets[b]['ach_sum'] += ach
        buckets[b]['vz_sum']  += vz

    print(f'  t_to_first_motion (ach>1)  {t_first_motion}')
    print(f'  {"bucket":<22} {"n":>5} {"dur":>7} {"frac":>6} {"avg_cmd":>8} {"avg_ach":>8} {"avg_vz":>7}')
    order = ('vertical_settle','horiz_undercommand','ramp_gap','tracking_ok')
    for b in order:
        v = buckets[b]
        nn = v['n']; dur = nn*DT; frac = nn/n
        ac = v['cmd_sum']/max(nn,1); aa = v['ach_sum']/max(nn,1); az = v['vz_sum']/max(nn,1)
        print(f'  {b:<22} {nn:>5} {dur:>6.3f}s {frac:>5.1%} {ac:>8.3f} {aa:>8.3f} {az:>7.3f}')

    # Time-grid trajectory: every 50ms across leg 0
    print(f'  TRAJECTORY (every 100ms, t=time-since-leg0-start):')
    print(f'    {"t":>5} {"cmd":>6} {"ach":>6} {"vz":>6} {"vd":>6}')
    t0 = leg0[0]['t']
    last_print = -1.0
    for s in leg0:
        rel = s['t'] - t0
        if rel - last_print >= 0.10 - 1e-6:
            print(f'    {rel:>5.2f} {s["cmd_spd"]:>6.2f} {s["ach_spd"]:>6.2f} {s["vel_d"]:>6.2f} {s["pos_d"]:>6.2f}')
            last_print = rel

if __name__ == '__main__':
    for course in ('technical','mixed'):
        analyze(course)
