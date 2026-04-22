"""Error-State Kalman Filter for visual-inertial pose fusion — Session 19.

Purpose
-------
Round 2 of the AI Grand Prix uses a real 3D-scanned environment with
obstacle geometry and visual distractors. That breaks a pure-perception
pose pipeline in two ways:
  1. A YOLO false-positive on a distractor can produce a high-confidence
     but geometrically-wrong pose fix that poisons the belief model.
  2. A transient vision dropout during a yaw turn leaves the belief model
     coasting on a yaw-rate guess that drifts fast (the S18 bug root cause).

The ESKF fuses IMU (accelerometer + gyroscope, ~200 Hz on the Neros)
with occasional vision pose fixes (~30 Hz best-case). Between fixes, the
filter integrates the IMU; each fix pulls the estimate back and refines
the learned IMU biases. This is load-bearing for:
  * surviving multi-second vision dropouts through turns,
  * rejecting distractor pose fixes via chi-squared gating (future),
  * ESKF-smoothed pose is what feeds the planner instead of raw YOLO.

Convention
----------
  NED world frame. Gravity g = [0, 0, +9.81] m/s^2 (positive Down).
  Body frame: x-forward, y-right, z-down (FRD, matches PX4).
  Quaternion: scalar-first [w, x, y, z], body-to-world.
  Specific force measurement: a_m = R^T (a_world - g) + b_a + noise_a,
  so v_dot_world = R(q) (a_m - b_a) + g.

Nominal state (16D): [p(3), v(3), q(4), b_a(3), b_g(3)]
Error state    (15D): [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]
  Attitude error uses body-frame right-multiplicative convention:
      q_true = q_nominal ⊗ Exp(δθ)
  So δθ is a small rotation vector expressed in the body frame.

References
----------
  Solà, "Quaternion kinematics for the error-state Kalman filter"
  (arXiv:1711.02508). We follow his Section 6 derivation for the
  right-multiplicative body-frame error convention.

Keep this module NumPy-only — no scipy, no torch — so it ports
cleanly to the Neros onboard compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

GRAVITY_NED = np.array([0.0, 0.0, 9.81])  # NED: Down is +Z


# ─────────────────────────────────────────────────────────────────────
# Quaternion and rotation helpers
# ─────────────────────────────────────────────────────────────────────

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert body-to-world quaternion (scalar-first) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation about NED z-axis / Down) in radians."""
    w, x, y, z = q
    return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def skew(v: np.ndarray) -> np.ndarray:
    """[v]_× skew-symmetric matrix such that [v]_× w = v × w."""
    x, y, z = v
    return np.array([
        [0,  -z,  y],
        [z,   0, -x],
        [-y,  x,  0],
    ])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector → rotation matrix. phi in radians."""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        # Taylor series near identity; good to ~1e-16.
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta)) * (K @ K)


def exp_quat(phi: np.ndarray) -> np.ndarray:
    """Rotation vector → scalar-first quaternion."""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        # Approximation: q ≈ [1, phi/2]
        return quat_normalize(np.array([1.0, phi[0]/2, phi[1]/2, phi[2]/2]))
    half = theta / 2
    s = np.sin(half) / theta
    return np.array([np.cos(half), phi[0]*s, phi[1]*s, phi[2]*s])


# ─────────────────────────────────────────────────────────────────────
# State and config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EskfState:
    """Nominal state + 15x15 error-state covariance."""
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))       # NED position (m)
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))       # NED velocity (m/s)
    q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))  # body→world
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))     # accel bias (m/s²)
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))     # gyro bias (rad/s)
    P: np.ndarray = field(default_factory=lambda: np.eye(15) * 1e-4) # error-state cov

    @property
    def yaw_rad(self) -> float:
        return quat_to_yaw(self.q)

    @property
    def R_body_to_world(self) -> np.ndarray:
        return quat_to_rotmat(self.q)


@dataclass
class EskfConfig:
    """IMU and vision noise parameters.

    Defaults are tuned for a Neros-class 8-inch quadrotor with a
    consumer-grade MEMS IMU sampled at ~200 Hz. Override with
    calibrated values when the real Neros IMU sheet is available.
    """
    # IMU continuous-time noise densities
    accel_noise_density: float = 0.02         # m/s² / √Hz
    gyro_noise_density: float = 0.0017        # rad/s / √Hz
    accel_bias_rw: float = 1e-4               # accel bias random walk
    gyro_bias_rw: float = 1e-5                # gyro bias random walk

    # Vision measurement noise (1-sigma)
    vision_pos_sigma: float = 0.10            # m
    vision_yaw_sigma: float = 0.05            # rad (~2.9°)


# ─────────────────────────────────────────────────────────────────────
# ESKF core
# ─────────────────────────────────────────────────────────────────────

class ESKF:
    """Error-state Kalman filter for visual-inertial drone pose.

    Usage:
        cfg = EskfConfig()
        eskf = ESKF(cfg)
        eskf.state.p = p0            # optional — seed from takeoff pose
        eskf.state.v = np.zeros(3)
        ...
        for each IMU sample (a_m, w_m, dt):
            eskf.predict(a_m, w_m, dt)
        when a vision fix (p_vis, yaw_vis) arrives:
            eskf.update_vision(p_vis, yaw_vis)
    """

    def __init__(self, config: Optional[EskfConfig] = None):
        self.cfg = config or EskfConfig()
        self.state = EskfState()

    # ── Prediction (IMU) ────────────────────────────────────────────

    def predict(self, a_m: np.ndarray, w_m: np.ndarray, dt: float) -> None:
        """Propagate nominal state + covariance over dt using IMU sample.

        a_m : specific force measurement (body frame, m/s²)
        w_m : angular rate measurement  (body frame, rad/s)
        dt  : time step since previous IMU sample (s)
        """
        s = self.state
        cfg = self.cfg

        # Bias-compensated inputs
        a_body = a_m - s.b_a
        w_body = w_m - s.b_g

        # Rotation at start of interval (used for covariance Jacobians)
        R = quat_to_rotmat(s.q)

        # ── Nominal state propagation (simple Euler is fine for 200 Hz) ──
        a_world = R @ a_body + GRAVITY_NED
        s.p = s.p + s.v * dt + 0.5 * a_world * dt * dt
        s.v = s.v + a_world * dt
        # Attitude: q_new = q ⊗ Exp(w * dt) (right-multiplicative body frame)
        dq = exp_quat(w_body * dt)
        s.q = quat_normalize(quat_mul(s.q, dq))
        # Biases modeled as constant in mean (random-walk is zero-mean)

        # ── Error-state Jacobian F (15x15) ──
        # Per Solà Eq. (263): discretized via Euler.
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt                            # δp_dot ← δv
        F[3:6, 6:9] = -R @ skew(a_body) * dt                    # δv_dot ← δθ (body)
        F[3:6, 9:12] = -R * dt                                   # δv_dot ← δb_a
        # δθ_dot: use Exp(-w dt) — first-order: (I - [w]_× dt)
        F[6:9, 6:9] = exp_so3(-w_body * dt)                     # δθ_dot ← δθ
        F[6:9, 12:15] = -np.eye(3) * dt                          # δθ_dot ← δb_g
        # biases: identity (random-walk mean is zero)

        # ── Process noise Q (15x15) ──
        # Convert continuous densities to discrete variances (integrate over dt).
        Q = np.zeros((15, 15))
        sig_a = cfg.accel_noise_density
        sig_g = cfg.gyro_noise_density
        Q[3:6, 3:6] = np.eye(3) * (sig_a * sig_a * dt)          # velocity ← accel noise
        Q[6:9, 6:9] = np.eye(3) * (sig_g * sig_g * dt)          # attitude ← gyro noise
        Q[9:12, 9:12] = np.eye(3) * (cfg.accel_bias_rw**2 * dt) # accel bias RW
        Q[12:15, 12:15] = np.eye(3) * (cfg.gyro_bias_rw**2 * dt)

        # ── Covariance update ──
        s.P = F @ s.P @ F.T + Q
        # Symmetrize against float drift
        s.P = 0.5 * (s.P + s.P.T)

    # ── Correction (vision pose) ────────────────────────────────────

    def update_vision(
        self,
        p_vis: np.ndarray,
        yaw_vis: float,
        pos_sigma: Optional[float] = None,
        yaw_sigma: Optional[float] = None,
        max_mahalanobis: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """Fuse a vision pose measurement: NED position (3) + yaw (1).

        Parameters
        ----------
        p_vis, yaw_vis     : the measurement
        pos_sigma, yaw_sigma : per-fix overrides of the default vision noise
        max_mahalanobis    : chi-squared gating threshold (NOT squared). If
            the innovation's Mahalanobis distance exceeds this, the fix is
            rejected — no state change, no covariance shrink — and the
            returned `accepted` flag is False. For 4-DOF innovation at 99%
            confidence the threshold is ≈ √13.28 ≈ 3.64.

        Returns
        -------
        (innovation_4, mahalanobis_distance, accepted)
        """
        s = self.state
        cfg = self.cfg
        pos_sigma = pos_sigma if pos_sigma is not None else cfg.vision_pos_sigma
        yaw_sigma = yaw_sigma if yaw_sigma is not None else cfg.vision_yaw_sigma

        # Innovation: measurement minus prediction.
        yaw_hat = quat_to_yaw(s.q)
        y_pos = p_vis - s.p
        y_yaw = _wrap_pi(yaw_vis - yaw_hat)
        y = np.array([y_pos[0], y_pos[1], y_pos[2], y_yaw])

        # Measurement Jacobian H (4x15).
        # Position measurement: dz_p/dδp = I, rest zero.
        # Yaw measurement: small-angle body-frame δθ projected onto world-z.
        # Using world-z column of R handles non-level attitude correctly.
        H = np.zeros((4, 15))
        H[0:3, 0:3] = np.eye(3)                    # position rows
        R = quat_to_rotmat(s.q)
        H[3, 6:9] = R[2, :]

        # Measurement noise R.
        Rm = np.diag([pos_sigma**2, pos_sigma**2, pos_sigma**2, yaw_sigma**2])

        # Innovation covariance + Mahalanobis distance for gating.
        S = H @ s.P @ H.T + Rm
        # solve is more numerically stable than explicit inverse.
        S_inv_y = np.linalg.solve(S, y)
        mahal_sq = float(y @ S_inv_y)
        mahal = float(np.sqrt(max(mahal_sq, 0.0)))

        if max_mahalanobis is not None and mahal > max_mahalanobis:
            # Reject: no state/cov change. Caller can log and continue.
            return y, mahal, False

        # Kalman gain + update.
        K = s.P @ H.T @ np.linalg.inv(S)
        dx = K @ y  # 15-vector

        # Inject error into nominal state.
        s.p = s.p + dx[0:3]
        s.v = s.v + dx[3:6]
        s.q = quat_normalize(quat_mul(s.q, exp_quat(dx[6:9])))
        s.b_a = s.b_a + dx[9:12]
        s.b_g = s.b_g + dx[12:15]

        # Joseph form covariance update (numerically stable).
        I_KH = np.eye(15) - K @ H
        s.P = I_KH @ s.P @ I_KH.T + K @ Rm @ K.T
        s.P = 0.5 * (s.P + s.P.T)

        # ESKF reset: since error has been injected into nominal, the linearization
        # point of the attitude error shifts. Solà Eq. (287) gives the exact reset
        # Jacobian G; to first order in δθ it's identity, which is what we use.

        return y, mahal, True

    # ── Convenience ─────────────────────────────────────────────────

    def seed(
        self,
        p: np.ndarray,
        v: Optional[np.ndarray] = None,
        yaw_rad: Optional[float] = None,
        p_sigma: float = 0.1,
        v_sigma: float = 0.1,
        att_sigma: float = 0.05,
        bias_sigma: float = 0.01,
    ) -> None:
        """Initialize state with a known position (+ optional velocity/yaw)
        and reasonable starting covariance. Call once before the first
        predict() to avoid the covariance blowing up from the 1e-4 default."""
        s = self.state
        s.p = np.asarray(p, dtype=float)
        if v is not None:
            s.v = np.asarray(v, dtype=float)
        if yaw_rad is not None:
            half = yaw_rad / 2.0
            s.q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        # Reset covariance: well-known position/yaw, unknown bias.
        P = np.zeros((15, 15))
        P[0:3, 0:3] = np.eye(3) * (p_sigma ** 2)
        P[3:6, 3:6] = np.eye(3) * (v_sigma ** 2)
        P[6:9, 6:9] = np.eye(3) * (att_sigma ** 2)
        P[9:12, 9:12] = np.eye(3) * (bias_sigma ** 2)
        P[12:15, 12:15] = np.eye(3) * (bias_sigma ** 2)
        s.P = P


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _wrap_pi(a: float) -> float:
    """Wrap angle into [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
