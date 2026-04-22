"""Offline belief-model replay — Session 19.

Single targeted scenario that isolates the yaw-propagation bug signature:
a sustained 2-second dropout during which the drone yaws 90°. This is
the regime where the bug expressed (mild-dropout through-turn) in the
S18 A/B results. Compares FIXED propagate vs an in-memory reproduction
of the BUGGY propagate so we can read the delta directly.

The definitive A/B is still s18_belief_test.py on PX4. This is a smoke
test to increase confidence before that expensive run.

Run standalone:
    python test_belief_replay.py
"""

import math
import sys
import types

try:
    import mavsdk.offboard  # noqa: F401
except ImportError:
    mavsdk_stub = types.ModuleType("mavsdk")
    offboard_stub = types.ModuleType("mavsdk.offboard")
    class _VNY:
        def __init__(self, vn, ve, vd, yd): pass
    offboard_stub.VelocityNedYaw = _VNY
    mavsdk_stub.offboard = offboard_stub
    sys.modules["mavsdk"] = mavsdk_stub
    sys.modules["mavsdk.offboard"] = offboard_stub

from gate_belief import GateBelief  # noqa: E402


def _buggy_propagate(belief, vel, yaw_rad, dt):
    """Pre-fix propagate — uses current yaw for both body↔NED conversions."""
    if belief.confidence < belief.MIN_CONFIDENCE:
        return
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    body_x = belief.range_est * math.cos(belief.bearing_h) * math.cos(belief.bearing_v)
    body_y = belief.range_est * math.sin(belief.bearing_h) * math.cos(belief.bearing_v)
    body_z = belief.range_est * math.sin(belief.bearing_v)
    dn = body_x * cos_y - body_y * sin_y
    de = body_x * sin_y + body_y * cos_y
    dd = body_z
    dn -= vel[0] * dt
    de -= vel[1] * dt
    dd -= vel[2] * dt
    new_bx = dn * cos_y + de * sin_y
    new_by = -dn * sin_y + de * cos_y
    new_bz = dd
    r = math.sqrt(new_bx**2 + new_by**2 + new_bz**2)
    if r > 0.1:
        belief.bearing_h = math.atan2(new_by, new_bx)
        horiz = math.sqrt(new_bx**2 + new_by**2)
        belief.bearing_v = math.atan2(new_bz, horiz) if horiz > 0.1 else 0.0
        belief.range_est = r
    belief.confidence *= belief.CONFIDENCE_DECAY
    belief.ticks_since_detection += 1
    belief._prev_yaw_rad = yaw_rad


DT = 1.0 / 50.0


def gt_observe(gate_ned, drone_pos, yaw_rad):
    dn = gate_ned[0] - drone_pos[0]
    de = gate_ned[1] - drone_pos[1]
    dd = gate_ned[2] - drone_pos[2]
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    bx = dn * cy + de * sy
    by = -dn * sy + de * cy
    r = math.sqrt(bx * bx + by * by + dd * dd)
    bh = math.atan2(by, bx) if r > 1e-6 else 0.0
    horiz = math.sqrt(bx * bx + by * by)
    bv = math.atan2(dd, horiz) if horiz > 0.1 else 0.0
    return bh, bv, r


def bel_to_ned(b, yaw, pos):
    bx = b.range_est * math.cos(b.bearing_h) * math.cos(b.bearing_v)
    by = b.range_est * math.sin(b.bearing_h) * math.cos(b.bearing_v)
    bz = b.range_est * math.sin(b.bearing_v)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (pos[0] + bx*cy - by*sy, pos[1] + bx*sy + by*cy, pos[2] + bz)


def run(use_fix, yaw_rate_deg_s=45.0):
    """Run the targeted bug-expression scenario.

    - Gate stationary at NED (20, 0, -2).
    - Drone starts at origin, yaw=0, flying forward at 4 m/s.
    - t ∈ [0.0, 0.5):  detections flow normally → belief anchored.
    - t ∈ [0.5, 2.5):  SUSTAINED DROPOUT. Drone yaws right at yaw_rate_deg_s
                        with no detections. This is where the bug lives.
    - t ∈ [2.5, 3.0):  detections resume, belief should recover.

    Returns:
        list of (t, error_m, phase) tuples, and summary numbers.
    """
    gate = (20.0, 0.0, -2.0)
    pos = [0.0, 0.0, 0.0]
    yaw = 0.0
    yaw_rate = math.radians(yaw_rate_deg_s)
    speed = 4.0

    b = GateBelief()
    total = int(3.0 / DT)
    errors = []
    dropout_peak_err = 0.0

    for k in range(total):
        t = k * DT
        vn = speed * math.cos(yaw)
        ve = speed * math.sin(yaw)
        pos[0] += vn * DT
        pos[1] += ve * DT

        # Phase-gated yaw (drone only yaws during dropout window)
        in_dropout = 0.5 <= t < 2.5
        if in_dropout:
            yaw += yaw_rate * DT

        bh_t, bv_t, r_t = gt_observe(gate, pos, yaw)

        if in_dropout:
            if use_fix:
                b.propagate([vn, ve, 0.0], yaw, DT)
            else:
                _buggy_propagate(b, [vn, ve, 0.0], yaw, DT)
            phase = "DROP"
        else:
            b.update_detected(bh_t, bv_t, r_t, 0.9, yaw)
            phase = "TRK"

        if b.is_alive:
            bned = bel_to_ned(b, yaw, pos)
            err = math.sqrt(sum((bned[i] - gate[i]) ** 2 for i in range(3)))
            errors.append((t, err, phase))
            if phase == "DROP" and err > dropout_peak_err:
                dropout_peak_err = err
    return errors, dropout_peak_err


def peak_dropout_err(errs):
    drops = [e for _, e, p in errs if p == "DROP"]
    return max(drops) if drops else float("nan")


def end_dropout_err(errs):
    drops = [(t, e) for t, e, p in errs if p == "DROP"]
    if not drops:
        return float("nan")
    return drops[-1][1]  # last tick before detection resumes


def main():
    print("Belief replay — sustained 2 s dropout during 90° yaw rotation\n")
    print("Scenario: gate at NED(20,0,-2), drone flies forward 4 m/s,")
    print("          detections anchor belief in [0, 0.5), then dropout")
    print("          [0.5, 2.5) while drone yaws right at 45 deg/s.\n")

    fix_errs, _ = run(use_fix=True)
    bug_errs, _ = run(use_fix=False)

    fix_peak = peak_dropout_err(fix_errs)
    bug_peak = peak_dropout_err(bug_errs)
    fix_end = end_dropout_err(fix_errs)
    bug_end = end_dropout_err(bug_errs)

    print(f"{'variant':<8}  {'peak_during_drop':>18}  {'end_of_drop':>13}")
    print(f"{'fixed':<8}  {fix_peak:>17.3f} m  {fix_end:>12.3f} m")
    print(f"{'buggy':<8}  {bug_peak:>17.3f} m  {bug_end:>12.3f} m")
    print(f"{'delta':<8}  {bug_peak - fix_peak:>+17.3f} m  "
          f"{bug_end - fix_end:>+12.3f} m  (positive = fix wins)")

    print()
    ok = True
    # Fix must keep peak error reasonable (drone translates ~8 m during
    # dropout, so the absolute bound is ~10 m even with perfect belief).
    if fix_peak > 5.0:
        print(f"FAIL: fix peak {fix_peak:.3f} m exceeds 5.0 m")
        ok = False
    # Fix must strictly beat bug by a meaningful margin on the yaw-heavy run.
    if (bug_peak - fix_peak) < 3.0:
        print(f"FAIL: fix doesn't beat bug by ≥3 m (delta={bug_peak-fix_peak:.3f})")
        ok = False
    if ok:
        print(f"PASS: fix holds belief within {fix_peak:.2f} m of truth "
              f"through 90° dropout-yaw; bug drifts to {bug_peak:.2f} m.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
