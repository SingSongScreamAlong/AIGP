"""Unit tests for BeliefNav gate-aware fallback — Session 19o.

The behaviour under test: when the navigator knows NED gate positions
(via `set_gates_ned`) and the current `tracker_state.target_idx` points
to a valid gate, `plan()` should steer toward that gate in NED geometry
— even if the tracker reports `detected=False`.

This plugs the post-pass out-of-FOV gap:
  * Gate N just got passed → tracker.target_idx advances to N+1.
  * Belief is reset (confidence 0) by `on_gate_passed`.
  * For the next several ticks, the detector doesn't see N+1 yet
    (it's behind the drone or sharply off-axis).
  * Pre-S19o: belief-coast has nothing to coast on, belief-search
    does a blind yaw sweep. Drone drifts off-course.
  * Post-S19o: navigator knows gate[N+1]'s NED position, computes
    direct bearing from its pose estimate, emits NED velocity
    pointing at it. Drone converges even without a detection.

Run standalone:
    python test_belief_nav_gate_aware.py
"""

import math
import sys
import types
from dataclasses import dataclass


# gate_belief imports VelocityNedYaw from mavsdk.offboard; stub it for
# test environments that don't have the SDK installed.
try:
    import mavsdk.offboard  # noqa: F401
except ImportError:
    mavsdk_stub = types.ModuleType("mavsdk")
    offboard_stub = types.ModuleType("mavsdk.offboard")

    class _VelocityNedYaw:
        def __init__(self, vn, ve, vd, yaw_deg):
            self.north_m_s = vn
            self.east_m_s = ve
            self.down_m_s = vd
            self.yaw_deg = yaw_deg

    offboard_stub.VelocityNedYaw = _VelocityNedYaw
    mavsdk_stub.offboard = offboard_stub
    sys.modules["mavsdk"] = mavsdk_stub
    sys.modules["mavsdk.offboard"] = offboard_stub

from gate_belief import BeliefNav  # noqa: E402


@dataclass
class _Tracker:
    target_idx: int
    detected: bool
    bearing_h: float = 0.0
    bearing_v: float = 0.0
    range_est: float = 20.0
    confidence: float = 0.0
    frames_since_seen: int = 0
    search_mode: bool = False


def _make_nav():
    nav = BeliefNav()
    # Avoid the time-gated behaviour at mission start.
    nav.mission_start_time = 0.0
    return nav


def test_gate_aware_steers_toward_target_without_detection():
    """No detection, but gates_ned set → velocity points at target."""
    nav = _make_nav()
    gates = [
        (10.0, 0.0, -1.5),   # gate 0 — just passed
        (20.0, 10.0, -1.5),  # gate 1 — current target, NE of drone
        (30.0, 0.0, -1.5),
    ]
    nav.set_gates_ned(gates)

    pos = [10.0, 0.0, -1.5]  # drone is at gate 0
    vel = [0.0, 0.0, 0.0]
    yaw_deg = 0.0

    state = _Tracker(target_idx=1, detected=False)
    # Call plan() several times to let the speed smoother ramp off zero
    # (max_rate = 12 m/s² × dt = 0.02 ⇒ ~0.24 m/s per tick). One call
    # would give a sub-m/s command, which hides sign/ratio errors in
    # the assertions.
    for _ in range(30):
        cmd = nav.plan(state, pos, vel, yaw_deg, dt=0.02)

    # dN=+10, dE=+10 → yaw_to_gate = 45°.
    assert abs(cmd.yaw_deg - 45.0) < 1.0, f"yaw_deg={cmd.yaw_deg} not near 45°"
    # Velocity components should both be positive and roughly equal.
    assert cmd.north_m_s > 0.5, f"vn={cmd.north_m_s} not pushing north"
    assert cmd.east_m_s > 0.5, f"ve={cmd.east_m_s} not pushing east"
    assert abs(cmd.north_m_s - cmd.east_m_s) < 0.5, (
        f"vn={cmd.north_m_s}, ve={cmd.east_m_s} not roughly equal on 45° heading"
    )
    print("  ✓ no detection → velocity steers toward target gate")


def test_gate_aware_prefers_detection_when_present():
    """When a detection is present, tracking takes over and the gate-
    aware fallback is bypassed — `_plan_tracking` uses bearing_h directly,
    which should dominate even if gates_ned is also set."""
    nav = _make_nav()
    gates = [(10.0, 0.0, -1.5), (20.0, 10.0, -1.5)]
    nav.set_gates_ned(gates)

    pos = [0.0, 0.0, -1.5]
    vel = [0.0, 0.0, 0.0]
    yaw_deg = 0.0

    # Detection says target is straight ahead (body bearing 0), even
    # though gates_ned says it's NE. Tracking path should win.
    state = _Tracker(
        target_idx=1, detected=True,
        bearing_h=0.0, bearing_v=0.0, range_est=10.0, confidence=0.9,
    )
    cmd = nav.plan(state, pos, vel, yaw_deg, dt=0.02)

    # yaw_deg should reflect bearing_h (0 rad → 0 deg from yaw=0), not
    # the 45° that gate_aware would produce.
    assert abs(cmd.yaw_deg) < 5.0, (
        f"yaw_deg={cmd.yaw_deg} — detection should override gate_aware"
    )
    print("  ✓ detection present → tracking path wins, not gate-aware")


def test_gate_aware_noop_when_gates_ned_unset():
    """If set_gates_ned was never called, plan() must fall back to the
    legacy coast/search path — no crash, no attribute error."""
    nav = _make_nav()
    # Deliberately do NOT call set_gates_ned.

    pos = [0.0, 0.0, -1.5]
    vel = [0.0, 0.0, 0.0]
    yaw_deg = 0.0
    state = _Tracker(target_idx=0, detected=False)

    try:
        cmd = nav.plan(state, pos, vel, yaw_deg, dt=0.02)
    except Exception as e:
        raise AssertionError(f"legacy path crashed with gates_ned=None: {e}")
    # We don't assert on the command values — only that we didn't crash
    # and the return type is sensible.
    assert hasattr(cmd, "north_m_s") and hasattr(cmd, "yaw_deg")
    print("  ✓ gates_ned unset → legacy coast/search path still works")


def test_gate_aware_out_of_range_target_falls_back():
    """If target_idx is past the end of gates_ned (e.g., final gate
    already passed), the gate-aware branch should skip and the legacy
    path should handle it."""
    nav = _make_nav()
    gates = [(10.0, 0.0, -1.5), (20.0, 0.0, -1.5)]
    nav.set_gates_ned(gates)

    pos = [0.0, 0.0, -1.5]
    vel = [0.0, 0.0, 0.0]
    yaw_deg = 0.0
    state = _Tracker(target_idx=5, detected=False)  # past the end

    try:
        cmd = nav.plan(state, pos, vel, yaw_deg, dt=0.02)
    except Exception as e:
        raise AssertionError(f"crashed on target_idx past end: {e}")
    assert hasattr(cmd, "yaw_deg")
    print("  ✓ target_idx past end of gates_ned → safe fallback")


def test_gate_aware_negative_target_idx_falls_back():
    """Defensive: negative target_idx (shouldn't happen, but guard
    against underflow/bad input) must not index gates_ned."""
    nav = _make_nav()
    gates = [(10.0, 0.0, -1.5)]
    nav.set_gates_ned(gates)

    pos = [0.0, 0.0, -1.5]
    state = _Tracker(target_idx=-1, detected=False)

    try:
        cmd = nav.plan(state, pos, [0.0, 0.0, 0.0], 0.0, dt=0.02)
    except Exception as e:
        raise AssertionError(f"crashed on negative target_idx: {e}")
    assert hasattr(cmd, "yaw_deg")
    print("  ✓ negative target_idx → safe fallback")


def test_pose_trusted_false_skips_gate_aware():
    """S19p: when pose_trusted=False, gate-aware fallback must be
    bypassed even with gates_ned set and a valid target_idx. The
    drone should fall through to belief-coast/search instead of
    committing to a (likely-wrong) world-frame heading.

    With no prior detection (belief.confidence=0), the navigator
    drops into search mode — which yaw-sweeps but does NOT command
    a gate-direction velocity. So vn/ve should not be a strong
    push toward the (wrong) world target.
    """
    nav = _make_nav()
    gates = [(0.0, 0.0, -1.5), (20.0, 10.0, -1.5)]
    nav.set_gates_ned(gates)
    nav.pose_trusted = False  # ESKF is rejecting most fixes

    pos = [0.0, 0.0, -1.5]
    state = _Tracker(target_idx=1, detected=False)
    # Prime smoother — make sure speed ramps would have time to fire.
    for _ in range(30):
        cmd = nav.plan(state, pos, [0.0, 0.0, 0.0], 0.0, dt=0.02)

    # Gate-aware would yield yaw ≈ 45° + sizeable vn/ve toward (20, 10).
    # Search/coast (with belief.confidence=0) produces yaw sweep,
    # near-zero translation. Assert we are NOT in the gate-aware regime.
    not_gate_aware_yaw = abs(cmd.yaw_deg - 45.0) > 5.0
    weak_translation = (cmd.north_m_s ** 2 + cmd.east_m_s ** 2) ** 0.5 < 1.5
    assert not_gate_aware_yaw or weak_translation, (
        f"pose_trusted=False but cmd looks gate-aware: "
        f"yaw={cmd.yaw_deg}, vn={cmd.north_m_s}, ve={cmd.east_m_s}"
    )
    print("  ✓ pose_trusted=False → gate-aware bypassed, fell through")


def test_pose_trusted_default_is_true():
    """Backward-compat: navigators created without explicit pose-trust
    setup must default to trusted (so legacy callers without pose
    fusion don't lose the gate-aware fallback)."""
    nav = _make_nav()
    assert nav.pose_trusted is True, (
        f"pose_trusted default should be True for legacy compat, got {nav.pose_trusted}"
    )
    print("  ✓ pose_trusted defaults to True for legacy callers")


def main():
    tests = [
        ("gate-aware steers toward target w/o detection",
         test_gate_aware_steers_toward_target_without_detection),
        ("detection overrides gate-aware",
         test_gate_aware_prefers_detection_when_present),
        ("gates_ned unset → legacy path",
         test_gate_aware_noop_when_gates_ned_unset),
        ("target_idx past end → fallback",
         test_gate_aware_out_of_range_target_falls_back),
        ("negative target_idx → fallback",
         test_gate_aware_negative_target_idx_falls_back),
        ("pose_trusted=False skips gate-aware",
         test_pose_trusted_false_skips_gate_aware),
        ("pose_trusted defaults to True",
         test_pose_trusted_default_is_true),
    ]
    failures = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} PASSED")


if __name__ == "__main__":
    main()
