"""Unit tests for GateBelief.propagate — yaw invariance and translation math.

Session 19 regression guard for the yaw-propagation bug fixed in
gate_belief.py. The canonical failure case: pure yaw with zero translation
should leave the NED position of the believed gate unchanged, which means
the bearing_h in the body frame must rotate by exactly -dyaw.

Run standalone:
    python test_gate_belief.py
"""

import math
import sys
import types

# gate_belief imports VelocityNedYaw from mavsdk.offboard at module level,
# but the class under test (GateBelief) does not use it. Stub the import
# so these tests can run in any environment.
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

from gate_belief import GateBelief  # noqa: E402


EPS_POS = 0.05     # 5 cm tolerance on reconstructed NED position
EPS_ANG = math.radians(1.0)  # 1 degree tolerance on bearing


def body_to_ned(bearing_h, bearing_v, range_est, yaw_rad, drone_pos=(0.0, 0.0, 0.0)):
    """Convert a belief in body frame + drone yaw/pos to an absolute NED point."""
    body_x = range_est * math.cos(bearing_h) * math.cos(bearing_v)
    body_y = range_est * math.sin(bearing_h) * math.cos(bearing_v)
    body_z = range_est * math.sin(bearing_v)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    dn = body_x * cos_y - body_y * sin_y
    de = body_x * sin_y + body_y * cos_y
    dd = body_z
    return (drone_pos[0] + dn, drone_pos[1] + de, drone_pos[2] + dd)


def seed_belief(bearing_h=0.0, bearing_v=0.0, range_est=10.0, yaw_rad=0.0):
    """Seed a GateBelief with a single detection at the given body-frame bearing."""
    b = GateBelief()
    b.update_detected(
        bearing_h_rad=bearing_h,
        bearing_v_rad=bearing_v,
        range_est=range_est,
        det_confidence=1.0,
        yaw_rad=yaw_rad,
    )
    return b


def assert_close(actual, expected, tol, label):
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: |{actual:.4f} - {expected:.4f}| = "
            f"{abs(actual - expected):.4f} > tol={tol}"
        )


def test_pure_yaw_leaves_gate_stationary_in_ned():
    """Drone yaws 90° with zero translation. The believed gate's NED
    position must not drift; only the body-frame bearing should rotate."""
    # Drone at origin, facing East (yaw=π/2). Gate detected dead ahead at 10 m.
    # That places the gate at NED = (0, 10, 0).
    initial_yaw = math.pi / 2
    b = seed_belief(bearing_h=0.0, bearing_v=0.0, range_est=10.0, yaw_rad=initial_yaw)
    gate_ned_truth = body_to_ned(0.0, 0.0, 10.0, initial_yaw)
    assert abs(gate_ned_truth[0]) < 1e-9 and abs(gate_ned_truth[1] - 10.0) < 1e-9

    # Drone yaws to North (yaw=0) over 20 ticks at 50 Hz with zero translation.
    dt = 0.02
    steps = 20
    vel = [0.0, 0.0, 0.0]
    for i in range(1, steps + 1):
        frac = i / steps
        yaw_rad = initial_yaw * (1.0 - frac)  # π/2 → 0 linearly
        b.propagate(vel, yaw_rad, dt)

    # At yaw=0 (facing North), the gate that's at NED (0, 10, 0) is 10 m to
    # the East of the drone, which is now on the drone's right → bearing +π/2.
    assert_close(b.bearing_h, math.pi / 2, EPS_ANG, "final bearing_h after yaw")
    assert_close(b.range_est, 10.0, EPS_POS, "range after pure yaw")

    # Reconstruct absolute NED position using current yaw and compare to truth.
    final_ned = body_to_ned(b.bearing_h, b.bearing_v, b.range_est, 0.0)
    assert_close(final_ned[0], gate_ned_truth[0], EPS_POS, "NED.N stability")
    assert_close(final_ned[1], gate_ned_truth[1], EPS_POS, "NED.E stability")
    assert_close(final_ned[2], gate_ned_truth[2], EPS_POS, "NED.D stability")
    print("  ✓ pure yaw: gate NED drift <5 cm, bearing rotated to +π/2")


def test_pure_forward_translation():
    """Drone flies forward 2 m with no yaw change. Gate 10 m ahead should
    now appear 8 m ahead with bearing still ≈ 0."""
    yaw_rad = 0.0
    b = seed_belief(bearing_h=0.0, bearing_v=0.0, range_est=10.0, yaw_rad=yaw_rad)

    # 1 m/s forward (North, since yaw=0) for 2 s at 50 Hz
    dt = 0.02
    steps = 100
    vel = [1.0, 0.0, 0.0]
    for _ in range(steps):
        b.propagate(vel, yaw_rad, dt)

    assert_close(b.range_est, 8.0, 0.3, "range after 2 m forward")
    assert_close(b.bearing_h, 0.0, EPS_ANG, "bearing after pure forward")
    print(f"  ✓ pure forward: range 10→{b.range_est:.2f} m, bearing held at 0")


def test_translation_and_yaw_combined():
    """Drone both moves and yaws. NED-reconstructed gate position must
    remain near its stationary truth location."""
    initial_yaw = 0.0
    b = seed_belief(bearing_h=0.0, bearing_v=0.0, range_est=15.0, yaw_rad=initial_yaw)
    gate_ned_truth = body_to_ned(0.0, 0.0, 15.0, initial_yaw)  # (15, 0, 0)

    dt = 0.02
    steps = 50  # 1 second of flight
    drone_pos = [0.0, 0.0, 0.0]
    # Drone flies North at 3 m/s AND yaws right at ~30 deg/s
    vel_n, vel_e = 3.0, 0.0
    yaw_rate = math.radians(30.0)

    for _ in range(steps):
        drone_pos[0] += vel_n * dt
        drone_pos[1] += vel_e * dt
        new_yaw = initial_yaw + yaw_rate * dt
        b.propagate([vel_n, vel_e, 0.0], new_yaw, dt)
        initial_yaw = new_yaw

    # Reconstruct gate NED from current belief + current yaw + current drone pos.
    final_yaw = initial_yaw
    final_ned = body_to_ned(b.bearing_h, b.bearing_v, b.range_est, final_yaw,
                             drone_pos=tuple(drone_pos))
    assert_close(final_ned[0], gate_ned_truth[0], 0.3, "combined: NED.N stability")
    assert_close(final_ned[1], gate_ned_truth[1], 0.3, "combined: NED.E stability")
    print(f"  ✓ combined motion+yaw: NED drift "
          f"N={abs(final_ned[0]-gate_ned_truth[0]):.3f} m, "
          f"E={abs(final_ned[1]-gate_ned_truth[1]):.3f} m")


def test_dead_belief_propagate_is_noop():
    """If confidence is below MIN_CONFIDENCE, propagate is a no-op."""
    b = GateBelief()
    b.confidence = 0.001  # below MIN_CONFIDENCE
    b.range_est = 42.0
    b.propagate([10.0, 0.0, 0.0], 0.0, 1.0)
    assert b.range_est == 42.0, "dead belief propagate mutated state"
    print("  ✓ dead belief: propagate is no-op")


def test_unseeded_prev_yaw_falls_back_safely():
    """If _prev_yaw_rad is None but confidence > threshold (shouldn't happen
    in normal flow, but guard against it), propagate should not crash."""
    b = GateBelief()
    b.bearing_h = 0.0
    b.bearing_v = 0.0
    b.range_est = 10.0
    b.confidence = 0.9
    b._prev_yaw_rad = None  # not seeded
    try:
        b.propagate([0.0, 0.0, 0.0], 0.5, 0.02)
    except Exception as e:
        raise AssertionError(f"propagate crashed with prev_yaw=None: {e}")
    print("  ✓ unseeded prev_yaw: propagate does not crash")


def main():
    tests = [
        ("pure yaw leaves gate stationary in NED", test_pure_yaw_leaves_gate_stationary_in_ned),
        ("pure forward translation", test_pure_forward_translation),
        ("translation + yaw combined", test_translation_and_yaw_combined),
        ("dead belief propagate is noop", test_dead_belief_propagate_is_noop),
        ("unseeded prev_yaw fallback", test_unseeded_prev_yaw_falls_back_safely),
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
