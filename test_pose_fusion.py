"""Tests for PoseFusion — the race-facing wrapper around ESKF.

Coverage:
  1. End-to-end plumbing: synthetic IMU from truth kinematics + periodic
     vision fixes. Fused pose tracks truth within tolerance.
  2. Bad dt handling: first sample and too-large dt both get dropped,
     telemetry records the drop count.
  3. Distractor rejection via gating: telemetry records it, state
     doesn't move.
  4. Auto-seed from first vision fix when no explicit seed was given.
  5. Accepted-fix counter increments correctly; `pose()` snapshot
     shape and types are sane for callers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from estimation import PoseFusion, IMUSample, EskfConfig, ESKF
from estimation.eskf import GRAVITY_NED


# ─────────────────────────────────────────────────────────────────────
# Helper: synthetic IMU from a constant-velocity truth trajectory
# ─────────────────────────────────────────────────────────────────────

def _constant_vel_imu(t: float, vel: np.ndarray, yaw: float = 0.0) -> IMUSample:
    """Drone moving at constant velocity: zero world-frame acceleration,
    so a_m_body = R^T · (-g). For level (yaw-only rotation), gravity
    stays on the body z-axis regardless of yaw."""
    a_m = -GRAVITY_NED.copy()
    w_m = np.zeros(3)
    return IMUSample(accel_body=a_m, gyro_body=w_m, timestamp=t)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_end_to_end_constant_velocity():
    print("[1] Constant-velocity tracking: IMU + vision through PoseFusion")
    pf = PoseFusion()
    pf.seed(p=np.zeros(3), v=np.array([2.0, 0.0, 0.0]), yaw_rad=0.0)

    vel_truth = np.array([2.0, 0.0, 0.0])
    dt = 0.005
    vision_every = 20
    T = 10.0
    n = int(T / dt)

    max_err = 0.0
    for i in range(n):
        t = (i + 1) * dt
        applied = pf.on_imu(_constant_vel_imu(t, vel_truth))
        # Every 20 steps, a vision fix at truth position.
        if i > 0 and i % vision_every == 0:
            p_truth = vel_truth * t
            pf.on_vision_pose(p_truth, 0.0)
        p_est, _, _ = pf.pose()
        p_truth = vel_truth * t
        err = np.linalg.norm(p_est - p_truth)
        max_err = max(max_err, err)

    final_err = np.linalg.norm(pf.pose()[0] - vel_truth * T)
    tel = pf.telemetry
    print(f"  Fixes accepted/rejected: {tel.vision_fixes_accepted}/{tel.vision_fixes_rejected}")
    print(f"  IMU applied/dropped    : {tel.imu_samples_seen - tel.imu_samples_dropped}/{tel.imu_samples_dropped}")
    print(f"  Max error along run    : {max_err:.4f} m")
    print(f"  Final position error   : {final_err:.4f} m")
    assert max_err < 0.2, f"tracking error too high: {max_err}"
    assert final_err < 0.05, f"final error too high: {final_err}"
    print("  ✓ fused pose tracks constant-velocity truth")


def test_bad_dt_dropped():
    print("[2] Bad dt samples are dropped and counted")
    pf = PoseFusion()
    pf.seed(p=np.zeros(3))

    # First sample: no prior dt, should drop.
    first = IMUSample(-GRAVITY_NED.copy(), np.zeros(3), 0.0)
    applied = pf.on_imu(first)
    assert not applied, "first sample with no prior dt should not be applied"
    assert pf.telemetry.imu_samples_seen == 1

    # Normal sample: dt = 0.005, should apply.
    normal = IMUSample(-GRAVITY_NED.copy(), np.zeros(3), 0.005)
    applied = pf.on_imu(normal)
    assert applied

    # Negative dt (clock jump backward): drop.
    backward = IMUSample(-GRAVITY_NED.copy(), np.zeros(3), 0.004)
    applied = pf.on_imu(backward)
    assert not applied
    assert pf.telemetry.imu_samples_dropped == 1

    # Huge dt (e.g., telemetry hiccup): drop.
    huge = IMUSample(-GRAVITY_NED.copy(), np.zeros(3), 100.0)
    applied = pf.on_imu(huge)
    assert not applied
    assert pf.telemetry.imu_samples_dropped == 2

    print(f"  seen={pf.telemetry.imu_samples_seen}, dropped={pf.telemetry.imu_samples_dropped}")
    print("  ✓ bad-dt samples dropped with correct counter")


def test_distractor_rejected_telemetry():
    print("[3] Distractor vision fix rejected; telemetry + state correct")
    pf = PoseFusion()
    pf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    # Let a tight covariance settle.
    dt = 0.005
    t = 0.0
    for i in range(200):
        t += dt
        pf.on_imu(IMUSample(-GRAVITY_NED.copy(), np.zeros(3), t))
        if i > 0 and i % 20 == 0:
            pf.on_vision_pose(np.zeros(3), 0.0)
    p_before = pf.pose()[0].copy()

    # Distractor: 50 m offset, should be rejected at default threshold.
    inno, mahal, accepted = pf.on_vision_pose(
        np.array([50.0, 0.0, 0.0]), 0.0
    )
    assert not accepted, "distractor should be rejected"
    assert mahal > PoseFusion.DEFAULT_MAHALANOBIS_THRESHOLD
    assert pf.telemetry.vision_fixes_rejected == 1
    assert np.allclose(pf.pose()[0], p_before), "state moved on rejected fix"
    print(f"  mahal={mahal:.1f}, rejected; telemetry records 1 rejection")
    print("  ✓ gating + telemetry wired correctly")


def test_auto_seed_from_first_vision():
    print("[4] First vision fix auto-seeds an un-seeded filter")
    pf = PoseFusion()
    assert not pf.is_seeded

    p0 = np.array([10.0, -5.0, -2.0])
    yaw0 = 0.5
    inno, mahal, accepted = pf.on_vision_pose(p0, yaw0)
    assert accepted
    assert pf.is_seeded
    p_fused, _, yaw_fused = pf.pose()
    # Auto-seed places state exactly at the measurement (before any IMU).
    assert np.allclose(p_fused, p0), f"auto-seed position wrong: {p_fused}"
    assert abs(yaw_fused - yaw0) < 1e-9, f"auto-seed yaw wrong: {yaw_fused}"
    print(f"  Seed from vision: p={p_fused}, yaw={np.degrees(yaw_fused):.2f}°")
    print("  ✓ auto-seed works")


def test_recent_reject_rate_rolling_window():
    print("[6] recent_reject_rate() tracks rolling vision outcome window")
    pf = PoseFusion(reject_window=10)

    # Under the min_samples floor → returns 0.0 even with rejections.
    pf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)
    assert pf.recent_reject_rate() == 0.0, "should be 0 before any fixes"

    # Tight covariance so distractors get rejected reliably.
    dt = 0.005
    t = 0.0
    for _ in range(400):
        t += dt
        pf.on_imu(IMUSample(-GRAVITY_NED.copy(), np.zeros(3), t))

    # Feed 10 distractors (50m off) — should all be rejected.
    for _ in range(10):
        pf.on_vision_pose(np.array([50.0, 0.0, 0.0]), 0.0)
    rate_all_rej = pf.recent_reject_rate()
    assert rate_all_rej > 0.9, f"expected ~1.0 rejection rate, got {rate_all_rej}"

    # Feed 10 clean fixes at the current filter mean — should all pass
    # and evict the distractor record from the rolling window.
    p_cur = pf.pose()[0]
    for _ in range(10):
        pf.on_vision_pose(p_cur.copy(), 0.0)
    rate_after_recovery = pf.recent_reject_rate()
    assert rate_after_recovery < 0.1, (
        f"expected near-0 after recovery, got {rate_after_recovery}"
    )

    # min_samples floor: explicitly ask for 100 → not enough samples.
    assert pf.recent_reject_rate(min_samples=100) == 0.0
    print(f"  all-reject={rate_all_rej:.2f}, post-recovery={rate_after_recovery:.2f}")
    print("  ✓ rolling window + min_samples floor correct")


def test_pose_snapshot_shape():
    print("[5] pose() and biases() return numpy arrays with expected shapes")
    pf = PoseFusion()
    pf.seed(p=np.ones(3), v=np.ones(3), yaw_rad=0.0)
    p, v, yaw = pf.pose()
    b_a, b_g = pf.biases()
    assert p.shape == (3,) and p.dtype == np.float64, f"p shape/dtype wrong: {p}"
    assert v.shape == (3,) and v.dtype == np.float64
    assert isinstance(yaw, float)
    assert b_a.shape == (3,) and b_g.shape == (3,)
    # Snapshot should be a copy, not a view on internal state.
    p[0] = 999.0
    assert pf.pose()[0][0] == 1.0, "pose() returned view instead of copy"
    print("  ✓ shapes and copy-semantics correct")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_end_to_end_constant_velocity,
        test_bad_dt_dropped,
        test_distractor_rejected_telemetry,
        test_auto_seed_from_first_vision,
        test_recent_reject_rate_rolling_window,
        test_pose_snapshot_shape,
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
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        return 1
    print(f"{len(tests)}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
