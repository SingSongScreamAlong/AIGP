"""Tests for the ESKF (Error-State Kalman Filter) — Session 19.

Each test probes a specific property of the filter. Failures should point
at a specific math bug, not a vague "it doesn't work":
  1. Quaternion/rotation helpers round-trip cleanly.
  2. At rest with zero bias and no noise, integrated position does not drift.
     Catches gravity-compensation sign errors.
  3. Constant forward specific force produces correct position & velocity.
     Catches body→world rotation direction errors.
  4. Pure yaw rate leaves position invariant but rotates attitude.
     Catches quaternion-integration sign errors.
  5. Vision correction pulls position back to truth under IMU bias drift.
     Catches error-injection sign errors.
  6. Gyro bias estimation converges when driven with constant bias.
     Catches bias-coupling Jacobian errors.
  7. Covariance stays positive-definite over thousands of steps.
     Catches accumulating float drift / Joseph-form errors.

Tests are NumPy-only — no scipy, no torch — matching the module itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from estimation.eskf import (
    ESKF, EskfConfig, GRAVITY_NED,
    quat_mul, quat_to_rotmat, quat_to_yaw, exp_quat, exp_so3, skew,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Helper correctness
# ─────────────────────────────────────────────────────────────────────

def test_quat_helpers():
    print("[1] Quaternion & rotation helpers")

    # Identity quaternion yields identity rotation
    q_id = np.array([1., 0., 0., 0.])
    R = quat_to_rotmat(q_id)
    assert np.allclose(R, np.eye(3)), f"identity R wrong: {R}"

    # 90° yaw about Down (+Z in NED): forward → right
    q_yaw90 = exp_quat(np.array([0, 0, np.pi/2]))
    R = quat_to_rotmat(q_yaw90)
    forward_body = np.array([1., 0., 0.])
    forward_world = R @ forward_body
    assert np.allclose(forward_world, [0, 1, 0], atol=1e-9), \
        f"+90° yaw: body forward should map to world east; got {forward_world}"
    assert abs(quat_to_yaw(q_yaw90) - np.pi/2) < 1e-9, \
        f"quat_to_yaw(+90°) = {np.degrees(quat_to_yaw(q_yaw90)):.3f}°"
    print(f"  ✓ +90° yaw: body-forward → world-east, yaw extraction={np.degrees(quat_to_yaw(q_yaw90)):.2f}°")

    # Quaternion multiplication: yaw90 ⊗ yaw90 == yaw180
    q2 = quat_mul(q_yaw90, q_yaw90)
    yaw2 = quat_to_yaw(q2)
    assert abs(abs(yaw2) - np.pi) < 1e-9, f"90°∘90° yaw = {np.degrees(yaw2):.3f}° (want ±180°)"
    print(f"  ✓ 90°∘90° composition = ±180° (got {np.degrees(yaw2):.2f}°)")

    # Rodrigues exp_so3 ↔ exp_quat consistency
    phi = np.array([0.1, -0.2, 0.3])
    R_from_so3 = exp_so3(phi)
    R_from_quat = quat_to_rotmat(exp_quat(phi))
    assert np.allclose(R_from_so3, R_from_quat, atol=1e-12), \
        "exp_so3 and exp_quat disagree"
    print("  ✓ exp_so3 ↔ exp_quat consistent")

    # Skew symmetry: [v]_× w == v × w
    v = np.array([1., 2., 3.]); w = np.array([-1., 0.5, 0.25])
    assert np.allclose(skew(v) @ w, np.cross(v, w)), "skew() wrong"
    print("  ✓ skew() matches numpy cross product")


# ─────────────────────────────────────────────────────────────────────
# 2. At-rest integration
# ─────────────────────────────────────────────────────────────────────

def test_at_rest_no_drift():
    print("[2] Drone at rest: no drift over 10 s of noiseless IMU")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    # Accelerometer reads specific force of stationary drone: −g in body
    # (level: R=I, a_world=0 ⇒ a_m = R^T(a-g) = -g).
    a_m = -GRAVITY_NED.copy()
    w_m = np.zeros(3)
    dt = 0.005  # 200 Hz
    for _ in range(2000):  # 10 s
        eskf.predict(a_m, w_m, dt)

    p, v = eskf.state.p, eskf.state.v
    assert np.linalg.norm(p) < 1e-6, f"position drifted to {p}"
    assert np.linalg.norm(v) < 1e-6, f"velocity drifted to {v}"
    print(f"  ✓ 10 s @ 200 Hz: |p|={np.linalg.norm(p):.2e} m, |v|={np.linalg.norm(v):.2e} m/s")


# ─────────────────────────────────────────────────────────────────────
# 3. Constant forward acceleration
# ─────────────────────────────────────────────────────────────────────

def test_constant_forward_accel():
    print("[3] Constant 1 m/s² forward specific force")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    # a_world = 1 m/s² North. Body = world (level, yaw=0).
    # a_m = R^T (a_world - g) = 1·e_N - g = [1, 0, -9.81]
    a_m = np.array([1.0, 0.0, -9.81])
    w_m = np.zeros(3)
    dt = 0.005
    T = 2.0
    n = int(T / dt)
    for _ in range(n):
        eskf.predict(a_m, w_m, dt)

    # Expected: v = 2 m/s North, p = 0.5·1·4 = 2 m North.
    p, v = eskf.state.p, eskf.state.v
    assert abs(p[0] - 2.0) < 5e-3, f"p_N expected 2.0, got {p[0]:.4f}"
    assert abs(v[0] - 2.0) < 1e-3, f"v_N expected 2.0, got {v[0]:.4f}"
    assert np.linalg.norm(p[1:]) < 1e-3, f"lateral drift {p[1:]}"
    print(f"  ✓ After 2 s: p_N={p[0]:.4f} m (want 2.0), v_N={v[0]:.4f} m/s (want 2.0)")


# ─────────────────────────────────────────────────────────────────────
# 4. Pure yaw rotation
# ─────────────────────────────────────────────────────────────────────

def test_pure_yaw_rotation():
    print("[4] Pure yaw rate: π/2 yaw over π seconds, no translation")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    # Level drone yawing at 0.5 rad/s. Because the drone rotates in place,
    # the measured specific force must also rotate: a_m(t) = R(t)^T · (-g).
    # For the level case this stays [0, 0, -9.81] regardless of yaw, so we
    # can keep a_m constant.
    dt = 0.005
    yaw_rate = 0.5
    a_m = -GRAVITY_NED.copy()
    w_m = np.array([0.0, 0.0, yaw_rate])
    T = np.pi          # → π · 0.5 = π/2 rad yaw
    n = int(T / dt)
    for _ in range(n):
        eskf.predict(a_m, w_m, dt)

    yaw_est = eskf.state.yaw_rad
    p = eskf.state.p
    assert abs(abs(yaw_est) - np.pi/2) < 1e-3, \
        f"yaw_est={np.degrees(yaw_est):.4f}° (want ±90°)"
    assert np.linalg.norm(p) < 5e-3, f"position drifted to {p}"
    print(f"  ✓ yaw={np.degrees(yaw_est):.3f}° (want 90°), |p|={np.linalg.norm(p):.2e} m")


# ─────────────────────────────────────────────────────────────────────
# 5. Vision correction pulls drift back
# ─────────────────────────────────────────────────────────────────────

def test_vision_correction_bounds_drift():
    print("[5] Vision fixes at 10 Hz keep drift bounded under constant accel bias")

    # True accelerometer bias — filter doesn't know it; must learn from vision.
    true_bias = np.array([0.02, -0.015, 0.01])
    dt_imu = 0.005
    a_m = -GRAVITY_NED + true_bias  # drone truly at rest; bias corrupts reading
    w_m = np.zeros(3)

    # Phase A — 2 s of pure IMU integration, no vision. Isolates the drift.
    eskf_nocorr = ESKF()
    eskf_nocorr.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)
    for _ in range(int(2.0 / dt_imu)):
        eskf_nocorr.predict(a_m, w_m, dt_imu)
    free_drift = np.linalg.norm(eskf_nocorr.state.p)

    # Phase B — 10 s of IMU + 10 Hz vision. Vision is at truth (0,0,0).
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0,
              p_sigma=0.05, v_sigma=0.05, att_sigma=0.02, bias_sigma=0.05)
    vision_every = 20  # 10 Hz
    for i in range(int(10.0 / dt_imu)):
        eskf.predict(a_m, w_m, dt_imu)
        if i > 0 and i % vision_every == 0:
            eskf.update_vision(p_vis=np.zeros(3), yaw_vis=0.0)
    final_err = np.linalg.norm(eskf.state.p)

    print(f"  Free-run drift after 2 s (no vision) : {free_drift:.3f} m")
    print(f"  Final error after 10 s (10 Hz vision): {final_err:.3f} m")

    assert free_drift > 0.02, f"expected clear IMU drift without vision; got {free_drift}"
    assert final_err < 0.15, f"vision-corrected error too large: {final_err:.3f} m"
    assert final_err < free_drift, "vision didn't actually help"
    print("  ✓ vision correction keeps drift bounded")


# ─────────────────────────────────────────────────────────────────────
# 6. Gyro bias estimation
# ─────────────────────────────────────────────────────────────────────

def test_gyro_bias_convergence():
    print("[6] Gyro bias convergence with periodic yaw corrections")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0,
              p_sigma=0.02, v_sigma=0.02, att_sigma=0.02, bias_sigma=0.05)

    true_gyro_bias = 0.01   # rad/s on z-axis (yaw)
    dt_imu = 0.005
    vision_every = 10       # 20 Hz
    T = 30.0
    n = int(T / dt_imu)

    a_m = -GRAVITY_NED.copy()
    # Measured gyro = truth (0) + bias
    w_m = np.array([0.0, 0.0, true_gyro_bias])

    for i in range(n):
        eskf.predict(a_m, w_m, dt_imu)
        if i > 0 and i % vision_every == 0:
            # Truth: not actually yawing (bias is a measurement error).
            eskf.update_vision(p_vis=np.zeros(3), yaw_vis=0.0)

    estimated = eskf.state.b_g[2]
    err = abs(estimated - true_gyro_bias)
    print(f"  True bias   = {true_gyro_bias:.6f} rad/s")
    print(f"  Estimated   = {estimated:.6f} rad/s (err = {err*1000:.3f} mrad/s)")
    assert err < 2e-3, f"bias estimate not converged: err={err:.6f}"
    print("  ✓ gyro bias converges within 2 mrad/s")


# ─────────────────────────────────────────────────────────────────────
# 7. Numerical stability
# ─────────────────────────────────────────────────────────────────────

def test_covariance_stays_psd():
    print("[7] Covariance stays positive-definite over 10,000 steps with noise")
    rng = np.random.default_rng(42)
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    dt = 0.005
    a_base = -GRAVITY_NED.copy()
    for i in range(10000):
        # Small random IMU perturbation to exercise covariance propagation.
        a_m = a_base + rng.normal(0, 0.05, size=3)
        w_m = rng.normal(0, 0.01, size=3)
        eskf.predict(a_m, w_m, dt)
        if i > 0 and i % 100 == 0:
            eskf.update_vision(
                p_vis=eskf.state.p + rng.normal(0, 0.1, size=3),
                yaw_vis=eskf.state.yaw_rad + rng.normal(0, 0.05),
            )

    P = eskf.state.P
    eigs = np.linalg.eigvalsh(P)
    min_eig = eigs.min()
    max_eig = eigs.max()
    assert min_eig > -1e-10, f"Covariance lost PSD: min eig = {min_eig}"
    assert np.all(np.isfinite(P)), "Covariance has NaN/inf"
    print(f"  Final cov eigs: min={min_eig:.3e}, max={max_eig:.3e}")
    print("  ✓ covariance remains PSD and finite after 10,000 steps")


# ─────────────────────────────────────────────────────────────────────
# 8. Yaw tracking with vision through a turn
# ─────────────────────────────────────────────────────────────────────

def test_yaw_tracking_through_turn():
    print("[8] Yaw tracks through 90° turn with 10 Hz vision + noisy IMU")
    rng = np.random.default_rng(7)
    cfg = EskfConfig()
    eskf = ESKF(cfg)
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0)

    true_yaw = 0.0
    yaw_rate = 1.0        # rad/s → 90° in ~1.57 s
    dt = 0.005
    T = 2.0
    n = int(T / dt)

    max_err = 0.0
    for i in range(n):
        true_yaw += yaw_rate * dt
        a_m = -GRAVITY_NED + rng.normal(0, 0.01, size=3)
        w_m = np.array([0.0, 0.0, yaw_rate]) + rng.normal(0, 0.002, size=3)
        eskf.predict(a_m, w_m, dt)
        if i > 0 and i % 20 == 0:   # 10 Hz vision
            eskf.update_vision(
                p_vis=np.zeros(3) + rng.normal(0, 0.05, size=3),
                yaw_vis=true_yaw + rng.normal(0, 0.02),
            )
        err = abs((eskf.state.yaw_rad - true_yaw + np.pi) % (2*np.pi) - np.pi)
        if i > 100:  # give filter a moment to warm up
            max_err = max(max_err, err)

    final_err = abs((eskf.state.yaw_rad - true_yaw + np.pi) % (2*np.pi) - np.pi)
    print(f"  Final true yaw = {np.degrees(true_yaw):.2f}°")
    print(f"  Final est  yaw = {np.degrees(eskf.state.yaw_rad):.2f}°")
    print(f"  Max  err after warmup = {np.degrees(max_err):.2f}°")
    assert final_err < np.radians(3.0), f"yaw tracking error {np.degrees(final_err):.2f}° too large"
    assert max_err < np.radians(6.0), f"peak tracking error {np.degrees(max_err):.2f}° too large"
    print("  ✓ yaw tracks within 3° final / 6° peak through 90° turn")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 9-10. Chi-squared gating
# ─────────────────────────────────────────────────────────────────────

def test_gating_rejects_distractor():
    print("[9] Chi-squared gating rejects an outlier fix")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0,
              p_sigma=0.05, v_sigma=0.05, att_sigma=0.02, bias_sigma=0.01)

    # Let a tight covariance settle with a handful of clean fixes.
    a_m = -GRAVITY_NED.copy()
    dt = 0.005
    for i in range(400):  # 2 s
        eskf.predict(a_m, np.zeros(3), dt)
        if i % 20 == 0:
            eskf.update_vision(np.zeros(3), 0.0)

    p_before = eskf.state.p.copy()
    P_before = eskf.state.P.copy()

    # Distractor: vision claims the drone is 20 m off, which is 200σ away
    # from the current position estimate with σ=0.1 m. Must reject.
    y, mahal, accepted = eskf.update_vision(
        np.array([20.0, 0.0, 0.0]), 0.0, max_mahalanobis=3.64
    )
    print(f"  Distractor at +20 m North: mahal={mahal:.2f}, accepted={accepted}")
    assert not accepted, "distractor should have been rejected"
    assert mahal > 3.64, f"mahal={mahal} should exceed threshold 3.64"
    assert np.allclose(eskf.state.p, p_before), "state changed on rejected fix"
    assert np.allclose(eskf.state.P, P_before), "covariance changed on rejected fix"
    print("  ✓ rejected fix leaves state and covariance untouched")


def test_gating_accepts_valid_fix():
    print("[10] Chi-squared gating accepts a valid fix")
    eskf = ESKF()
    eskf.seed(p=np.zeros(3), v=np.zeros(3), yaw_rad=0.0,
              p_sigma=0.1, v_sigma=0.1, att_sigma=0.05, bias_sigma=0.01)

    # A tiny 5 cm offset — well inside 1σ of the vision noise (0.1 m default).
    y, mahal, accepted = eskf.update_vision(
        np.array([0.05, 0.0, 0.0]), 0.0, max_mahalanobis=3.64
    )
    print(f"  5 cm offset: mahal={mahal:.3f}, accepted={accepted}")
    assert accepted, "valid fix should have been accepted"
    assert mahal < 3.64
    # State should have moved partway toward the measurement.
    moved = eskf.state.p[0]
    assert 0.0 < moved < 0.05, f"state didn't update correctly; p_N={moved}"
    print(f"  ✓ state moved to p_N={moved:.4f} m toward the measurement")


def main() -> int:
    tests = [
        test_quat_helpers,
        test_at_rest_no_drift,
        test_constant_forward_accel,
        test_pure_yaw_rotation,
        test_vision_correction_bounds_drift,
        test_gyro_bias_convergence,
        test_covariance_stays_psd,
        test_yaw_tracking_through_turn,
        test_gating_rejects_distractor,
        test_gating_accepts_valid_fix,
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
