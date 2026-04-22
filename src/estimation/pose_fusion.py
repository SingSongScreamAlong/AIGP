"""PoseFusion — thin race-facing wrapper around ESKF — Session 19.

What this adds on top of raw `ESKF`:
  * Timestamped IMU ingestion (computes dt, sanity-gates on bad samples).
  * Timestamped vision pose ingestion with optional chi-squared gating.
  * Bootstrap from the first vision fix — avoids the caller having to seed
    the filter manually when the vehicle pose is unknown at start.
  * An accumulated telemetry record (n accepted/rejected fixes, last
    innovation Mahalanobis distance) for logging and diagnostics.
  * A read-only `pose()` snapshot returning `(pos_ned, vel_ned, yaw_rad)`
    for plug-in compatibility with `SimState`.

Keep this layer *thin* — the math lives in `eskf.py`. This is only
glue. Rule of thumb: if you're reaching for np.einsum in here, the
logic belongs in ESKF instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .eskf import ESKF, EskfConfig


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IMUSample:
    """One IMU reading in the body frame.

    accel_body: specific force (m/s²); gravity-loaded — a drone at rest
                reads ≈ −g on the body z-axis (body z is Down).
    gyro_body : angular rate (rad/s)
    timestamp : monotonic seconds (any consistent epoch; dt is what matters)
    """
    accel_body: np.ndarray
    gyro_body: np.ndarray
    timestamp: float


@dataclass
class FusionTelemetry:
    imu_samples_seen: int = 0
    imu_samples_dropped: int = 0       # bad dt (negative or > 1 s)
    vision_fixes_accepted: int = 0
    vision_fixes_rejected: int = 0
    last_innovation: np.ndarray = field(default_factory=lambda: np.zeros(4))
    last_mahalanobis: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# PoseFusion
# ─────────────────────────────────────────────────────────────────────

class PoseFusion:
    """Wraps ESKF for race-loop consumption.

    Typical lifecycle:
        pf = PoseFusion(EskfConfig(...))
        pf.seed(initial_pos)                      # optional
        while racing:
            pf.on_imu(IMUSample(a, w, t))         # fast loop, ~200 Hz
            if vision_pose_available:
                pf.on_vision_pose(p_world, yaw,
                                  max_mahalanobis=3.64)  # gated
            p, v, yaw = pf.pose()                 # feed planner
    """

    # Default gating threshold: 99% chi-squared at 4 DOF ⇒ √13.28 ≈ 3.644.
    # Callers can tighten or loosen per-fix.
    DEFAULT_MAHALANOBIS_THRESHOLD = 3.64

    # S19p: rolling window for pose-trust heuristic. Sized to cover
    # ~1 s of vision at 20 Hz — short enough to react when the filter
    # starts diverging, long enough to average out single-fix outliers.
    DEFAULT_REJECT_WINDOW = 20

    def __init__(
        self,
        config: Optional[EskfConfig] = None,
        max_imu_dt: float = 1.0,
        reject_window: int = DEFAULT_REJECT_WINDOW,
    ):
        self.eskf = ESKF(config)
        self._last_imu_t: Optional[float] = None
        self._seeded: bool = False
        self.telemetry = FusionTelemetry()
        self._max_imu_dt = max_imu_dt
        # Window of recent vision outcomes — 1 for accepted, 0 for
        # rejected. Used by `recent_reject_rate()` to drive the pose-
        # trust gate on the gate-aware BeliefNav fallback (S19o-A).
        self._recent_outcomes: deque[int] = deque(maxlen=int(reject_window))

    # ── Bootstrap ───────────────────────────────────────────────────

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
        """Initialize filter with a known pose. Call once before the
        first `on_imu` to avoid covariance under-initialization.

        The sigma parameters set the initial 1-σ uncertainty on each
        error-state block. Raise `bias_sigma` (e.g. 0.1–0.5) when the
        filter needs to learn a nontrivial IMU bias from data — the
        default 0.01 is tight enough that the Kalman gain on the bias
        block is small."""
        self.eskf.seed(
            p=np.asarray(p, dtype=float),
            v=None if v is None else np.asarray(v, dtype=float),
            yaw_rad=yaw_rad,
            p_sigma=p_sigma,
            v_sigma=v_sigma,
            att_sigma=att_sigma,
            bias_sigma=bias_sigma,
        )
        self._seeded = True

    # ── Inputs ──────────────────────────────────────────────────────

    def on_imu(self, sample: IMUSample) -> bool:
        """Predict the filter forward to this IMU sample's timestamp.

        Returns True if the sample was applied, False if it was dropped
        (first sample with no prior dt, or dt outside sanity range)."""
        self.telemetry.imu_samples_seen += 1
        if self._last_imu_t is None:
            self._last_imu_t = sample.timestamp
            return False
        dt = sample.timestamp - self._last_imu_t
        if dt <= 0.0 or dt > self._max_imu_dt:
            self.telemetry.imu_samples_dropped += 1
            # Reset clock to current sample so we can recover on the next one.
            self._last_imu_t = sample.timestamp
            return False
        if not self._seeded:
            # First sample received without explicit seed — fall back to
            # identity pose + default covariance. Caller should prefer
            # explicit seed() when initial pose is known.
            self.eskf.seed(p=np.zeros(3))
            self._seeded = True
        self.eskf.predict(
            np.asarray(sample.accel_body, dtype=float),
            np.asarray(sample.gyro_body, dtype=float),
            dt,
        )
        self._last_imu_t = sample.timestamp
        return True

    def on_vision_pose(
        self,
        p_world: np.ndarray,
        yaw_rad: float,
        pos_sigma: Optional[float] = None,
        yaw_sigma: Optional[float] = None,
        max_mahalanobis: Optional[float] = DEFAULT_MAHALANOBIS_THRESHOLD,
    ) -> Tuple[np.ndarray, float, bool]:
        """Fuse a vision-derived world-frame pose.

        Returns the (innovation, mahalanobis_distance, accepted) triple
        from the underlying ESKF update. A rejected fix leaves state
        and covariance untouched.
        """
        if not self._seeded:
            # First vision fix seeds the filter — useful when IMU hasn't
            # arrived yet or when the vehicle pose is unknown at startup.
            self.seed(np.asarray(p_world, dtype=float), yaw_rad=yaw_rad)
            self.telemetry.vision_fixes_accepted += 1
            inno = np.zeros(4)
            self.telemetry.last_innovation = inno
            self.telemetry.last_mahalanobis = 0.0
            self._recent_outcomes.append(1)
            return inno, 0.0, True

        inno, mahal, accepted = self.eskf.update_vision(
            p_vis=np.asarray(p_world, dtype=float),
            yaw_vis=float(yaw_rad),
            pos_sigma=pos_sigma,
            yaw_sigma=yaw_sigma,
            max_mahalanobis=max_mahalanobis,
        )
        self.telemetry.last_innovation = inno
        self.telemetry.last_mahalanobis = mahal
        if accepted:
            self.telemetry.vision_fixes_accepted += 1
            self._recent_outcomes.append(1)
        else:
            self.telemetry.vision_fixes_rejected += 1
            self._recent_outcomes.append(0)
        return inno, mahal, accepted

    # ── Diagnostics ─────────────────────────────────────────────────

    def recent_reject_rate(self, min_samples: int = 5) -> float:
        """Fraction of recent vision fixes that were chi-squared rejected.

        Returns the rejection rate over the rolling window. Returns 0.0
        when fewer than `min_samples` outcomes have been observed —
        callers should treat that as "trust the pose by default" rather
        than guessing from sparse data.

        Used by the gate-aware BeliefNav fallback to decide whether the
        fused pose is healthy enough to navigate on. When the chi-
        squared gate is rejecting most fixes, the ESKF is in a self-
        destructive divergence (covariance shrinks faster than truth
        error, so valid fixes look like outliers, drift compounds);
        steering on that pose actively makes things worse.
        """
        if len(self._recent_outcomes) < int(min_samples):
            return 0.0
        accepted = sum(self._recent_outcomes)
        rejected = len(self._recent_outcomes) - accepted
        return rejected / len(self._recent_outcomes)

    # ── Outputs ─────────────────────────────────────────────────────

    def pose(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fused pose snapshot: (pos_ned, vel_ned, yaw_rad)."""
        s = self.eskf.state
        return s.p.copy(), s.v.copy(), s.yaw_rad

    def biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Learned IMU biases: (accel_bias, gyro_bias)."""
        s = self.eskf.state
        return s.b_a.copy(), s.b_g.copy()

    @property
    def is_seeded(self) -> bool:
        return self._seeded
