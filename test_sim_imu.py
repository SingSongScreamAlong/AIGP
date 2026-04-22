"""Tests for the MockKinematicAdapter synthetic IMU + the full chain
mock → PoseFusion → ESKF — Session 19g.

Covers:
  1. At rest: accel_body ≈ [0, 0, -g], gyro ≈ 0. Sanity-check the
     gravity convention so a future refactor can't silently flip a sign.
  2. Constant-velocity steady state: specific force stays at -g once the
     first-order filter has converged (world accel → 0).
  3. Horizontal acceleration: during a vel ramp, body-x accel shows the
     expected a_world_N value (yaw=0 ⇒ body-x == world-N).
  4. Yaw rate in gyro-z: yaw-step command produces positive gyro_z while
     the first-order filter slews, converges to 0 after settling.
  5. Full fusion chain: drive the mock through a short S-curve, feed
     IMU → PoseFusion + periodic vision, fused pose tracks truth.
  6. Bias learning: inject a constant accel_bias on the mock, feed
     through PoseFusion with vision updates, filter's learned b_a
     converges toward the true bias.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from sim.mock import MockKinematicAdapter, GRAVITY_NED
from sim.adapter import SimCapability, IMUReading
from estimation import PoseFusion, IMUSample


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _imu_to_sample(r: IMUReading) -> IMUSample:
    """Duck-type bridge: adapter.IMUReading → pose_fusion.IMUSample."""
    return IMUSample(
        accel_body=r.accel_body,
        gyro_body=r.gyro_body,
        timestamp=r.timestamp,
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_capability_and_construction():
    print("[1] Adapter advertises IMU capability and constructs cleanly")
    a = MockKinematicAdapter()
    assert bool(a.capabilities & SimCapability.IMU), "IMU flag missing"
    assert bool(a.capabilities & SimCapability.VELOCITY_NED)
    info = a.info()
    assert info.backend == "mock_kinematic"
    assert info.tick_rate_hz == 200.0  # default dt=0.005
    print(f"  backend={info.backend}, tick_rate={info.tick_rate_hz} Hz")
    print("  ✓ capability + info correct")


def test_at_rest_specific_force():
    print("[2] At rest: accel_body = [0, 0, -g], gyro = 0")
    a = MockKinematicAdapter(dt=0.01)
    # No command, no truth motion. Integrate a few ticks.
    for _ in range(20):
        imu = a.step()
    # a_world = 0 (no motion), specific_force = -g_NED in world,
    # yaw=0 ⇒ body = world. Expect accel_body ≈ [0, 0, -9.81].
    assert np.allclose(imu.accel_body, -GRAVITY_NED, atol=1e-12), \
        f"accel_body wrong: {imu.accel_body}"
    assert np.allclose(imu.gyro_body, 0.0, atol=1e-12)
    print(f"  accel_body={imu.accel_body}, gyro_body={imu.gyro_body}")
    print("  ✓ gravity sign + magnitude correct")


def test_constant_vel_steady_state():
    print("[3] Constant velocity steady state: specific force → -g")
    a = MockKinematicAdapter(dt=0.005, vel_tau=0.05)
    # Set truth directly to matched vel+cmd so filter is already at steady.
    a.set_truth(pos=np.zeros(3), vel=np.array([3.0, 0.0, 0.0]), yaw_rad=0.0)
    _run(a.send_velocity_ned(3.0, 0.0, 0.0, 0.0))  # yaw_deg=0
    for _ in range(5):
        imu = a.step()
    assert np.allclose(imu.accel_body, -GRAVITY_NED, atol=1e-9), \
        f"specific force under const vel should be -g, got {imu.accel_body}"
    print(f"  v=3 N, imu_accel_body={imu.accel_body}")
    print("  ✓ no phantom acceleration under constant velocity")


def test_horizontal_acceleration_transient():
    print("[4] Velocity ramp: body-x accel tracks world-N accel (yaw=0)")
    dt = 0.001
    vel_tau = 0.1
    a = MockKinematicAdapter(dt=dt, vel_tau=vel_tau)
    # Step-command 5 m/s in N. At the FIRST tick after the command,
    # Δv_N = α·(v_cmd - 0) where α = 1 - exp(-dt/tau). Expected world
    # accel for this tick: a_N = α·5 / dt. Body-x = world-N (yaw=0).
    _run(a.send_velocity_ned(5.0, 0.0, 0.0, 0.0))
    imu = a.step()
    alpha = 1.0 - np.exp(-dt / vel_tau)
    expected_a_N = alpha * 5.0 / dt
    # Body-x accel = a_world_N - 0 (no gravity in x), should ≈ expected.
    assert abs(imu.accel_body[0] - expected_a_N) < 1e-6, \
        f"body-x accel {imu.accel_body[0]:.4f} vs expected {expected_a_N:.4f}"
    # Body-z still reads -g (no vertical acceleration commanded).
    assert abs(imu.accel_body[2] + 9.81) < 1e-9
    print(f"  α={alpha:.4f}, expected a_N={expected_a_N:.4f}, "
          f"body-x={imu.accel_body[0]:.4f}")
    print("  ✓ horizontal accel correctly projected into body-x")


def test_yaw_rate_in_gyro_z():
    print("[5] Yaw step → positive gyro_z during slew, → 0 after settle")
    dt = 0.001
    yaw_tau = 0.1
    a = MockKinematicAdapter(dt=dt, yaw_tau=yaw_tau)
    # Command yaw = 90°.
    _run(a.send_velocity_ned(0.0, 0.0, 0.0, 90.0))
    imu0 = a.step()
    # Expected gyro_z on first tick: α·(π/2) / dt.
    alpha = 1.0 - np.exp(-dt / yaw_tau)
    expected_wz = alpha * (np.pi / 2.0) / dt
    assert imu0.gyro_body[2] > 0, "gyro_z should be positive for yaw increase"
    assert abs(imu0.gyro_body[2] - expected_wz) < 1e-6, \
        f"gyro_z {imu0.gyro_body[2]:.4f} vs expected {expected_wz:.4f}"
    # Let it converge (many time constants). After N·τ, yaw_err decays
    # as exp(-N); gyro_z is proportional. Need ~15τ for a 1e-4 residual.
    for _ in range(int(15 * yaw_tau / dt)):
        imu = a.step()
    assert abs(imu.gyro_body[2]) < 1e-4, f"gyro_z didn't settle: {imu.gyro_body[2]}"
    print(f"  first-tick gyro_z={imu0.gyro_body[2]:.3f}, settled={imu.gyro_body[2]:.3e}")
    print("  ✓ yaw rate correctly shows up in gyro z and settles")


def test_full_fusion_chain_tracks_truth():
    print("[6] Full chain: mock → PoseFusion + vision, fused pos ≈ truth")
    dt_imu = 0.005         # 200 Hz IMU
    vision_every = 20      # 10 Hz vision
    T = 4.0
    n = int(T / dt_imu)

    a = MockKinematicAdapter(dt=dt_imu, vel_tau=0.15, yaw_tau=0.15)
    pf = PoseFusion()

    # Start at a known non-zero state so we're not testing the zero case.
    a.set_truth(pos=np.array([1.0, 2.0, -1.5]), vel=np.zeros(3), yaw_rad=0.0)
    pf.seed(p=np.array([1.0, 2.0, -1.5]), v=np.zeros(3), yaw_rad=0.0)

    # Drive a gentle S-curve: ramp N, then turn, then ramp E.
    _run(a.send_velocity_ned(2.0, 0.0, 0.0, 0.0))

    max_err = 0.0
    for i in range(n):
        t = (i + 1) * dt_imu
        # Simple schedule: change yaw at 1 s, change heading at 2 s.
        if abs(t - 1.0) < dt_imu / 2:
            _run(a.send_velocity_ned(2.0, 0.0, 0.0, 45.0))
        if abs(t - 2.5) < dt_imu / 2:
            _run(a.send_velocity_ned(0.0, 2.0, 0.0, 90.0))

        imu_reading = a.step()
        pf.on_imu(_imu_to_sample(imu_reading))

        if i > 0 and i % vision_every == 0:
            truth_pos = a._pos.copy()   # truth access for the test
            truth_yaw = a._yaw
            pf.on_vision_pose(truth_pos, truth_yaw)

        # Compare fused position with truth
        p_est, _, _ = pf.pose()
        truth_pos = a._pos
        err = float(np.linalg.norm(p_est - truth_pos))
        max_err = max(max_err, err)

    tel = pf.telemetry
    final_err = float(np.linalg.norm(pf.pose()[0] - a._pos))
    print(f"  fixes: {tel.vision_fixes_accepted}/{tel.vision_fixes_rejected}  "
          f"imu applied: {tel.imu_samples_seen - tel.imu_samples_dropped}")
    print(f"  max tracking err: {max_err:.3f} m  |  final err: {final_err:.3f} m")
    assert max_err < 0.5, f"tracking error too high: {max_err}"
    assert final_err < 0.10, f"final error too high: {final_err}"
    print("  ✓ fused pose tracks S-curve truth through IMU+vision")


def test_accel_bias_observable_via_vision():
    print("[7] Constant accel bias: filter learns it from vision corrections")
    dt = 0.005
    T = 8.0
    n = int(T / dt)
    # 0.3 m/s² constant bias on body-x accel channel.
    bias_true = np.array([0.3, 0.0, 0.0])

    a = MockKinematicAdapter(dt=dt, vel_tau=0.1,
                             accel_bias=bias_true)
    pf = PoseFusion()
    a.set_truth(pos=np.zeros(3), vel=np.zeros(3), yaw_rad=0.0)
    # Use a loose bias prior so the Kalman gain on b_a is nontrivial.
    # The default 1e-2 is near-pinning; 0.5 m/s² matches MEMS IMU spec.
    pf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0,
            bias_sigma=0.5)

    # Hover (zero cmd). Truth stays put; mock still reports biased accel.
    for i in range(n):
        imu_reading = a.step()
        pf.on_imu(_imu_to_sample(imu_reading))
        if i > 0 and i % 20 == 0:  # 10 Hz vision
            pf.on_vision_pose(a._pos.copy(), a._yaw)

    b_a, b_g = pf.biases()
    # ESKF body-x accel bias should approach bias_true[0]. Be generous
    # on tolerance — bias observability depends on motion excitation,
    # and we're hovering. But even in hover the Kalman update sees the
    # drift and attributes part of it to bias. We expect LARGER-than-zero
    # bias recovery but not perfect.
    print(f"  true bias     = {bias_true}")
    print(f"  learned b_a   = {b_a}")
    print(f"  |learned|/true[0] = {b_a[0] / bias_true[0]:.3f}")
    # Sign should match and magnitude should be nontrivial.
    assert b_a[0] > 0.05, f"learned bias too small: {b_a}"
    assert b_a[0] < bias_true[0] + 0.2, f"learned bias overshot wildly: {b_a}"
    print("  ✓ bias partially observable through vision-corrected hover")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_capability_and_construction,
        test_at_rest_specific_force,
        test_constant_vel_steady_state,
        test_horizontal_acceleration_transient,
        test_yaw_rate_in_gyro_z,
        test_full_fusion_chain_tracks_truth,
        test_accel_bias_observable_via_vision,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        return 1
    print(f"{len(tests)}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
