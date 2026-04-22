"""MockKinematicAdapter — kinematic sim with synthetic IMU for ESKF testing.

This is a second mock adapter (distinct from the CLI-only `_MockFlightAdapter`
in `run_race.py`). Purpose: exercise the full IMU → ESKF → fused-pose chain
in sandbox tests, without hardware or a real sim backend.

Design:
  * First-order velocity tracking: `v̇ = (v_cmd - v) / tau`. Produces smooth,
    physically-integrable acceleration suitable for IMU synthesis.
  * Yaw tracked the same way. Level flight only (roll = pitch = 0) — that's
    all the belief/planner layer emits, so it's all we need here.
  * `step(dt)` advances truth and returns a synthesized `IMUReading`.
  * `get_imu()` returns the last synthesized sample (duck-compatible with
    `estimation.pose_fusion.IMUSample`).
  * Noise + static bias injection for realistic filter stress tests.

NOT intended for:
  * Running the race-loop CLI (use `_MockFlightAdapter` in run_race.py for
    that — it's simpler and doesn't need IMU).
  * Modeling real drone dynamics (no drag, no thrust saturation, no
    aerodynamic coupling). Good enough for validating fusion plumbing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .adapter import SimState, SimCapability, SimInfo, IMUReading


# Gravity in NED (positive Down) — matches ESKF convention.
GRAVITY_NED = np.array([0.0, 0.0, 9.81])


def _wrap_pi(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class MockKinematicAdapter:
    """Kinematic sim with synthetic IMU output.

    Sensor model:
      accel_body = R(yaw)ᵀ · (a_world - g_NED) + bias + N(0, σ_a²)
                   where a_world is the world-frame linear acceleration
                   computed from the first-order vel-tracking model, and
                   g_NED = [0, 0, +9.81].
      gyro_body  = [0, 0, ẏ] + bias + N(0, σ_g²)   (level flight)

    A drone at rest with zero command reads accel_body ≈ [0, 0, -9.81],
    which is the accelerometer measuring the *reaction* to gravity along
    body-z-down.
    """

    capabilities = (
        SimCapability.VELOCITY_NED
        | SimCapability.POSITION_NED
        | SimCapability.IMU
        | SimCapability.ARM_ACTION
        | SimCapability.RESET
        | SimCapability.WALLCLOCK_PACED
    )

    def __init__(
        self,
        dt: float = 0.005,              # 200 Hz default — matches flight IMU rate
        vel_tau: float = 0.15,          # velocity tracking time constant (s)
        yaw_tau: float = 0.15,          # yaw tracking time constant (s)
        accel_noise_sigma: float = 0.0, # m/s² per axis
        gyro_noise_sigma: float = 0.0,  # rad/s per axis
        accel_bias: Optional[np.ndarray] = None,  # 3-vec, m/s²
        gyro_bias: Optional[np.ndarray] = None,   # 3-vec, rad/s
        seed: int = 0,
        initial_altitude_m: float = 0.0,
        auto_step: bool = False,        # step() once per send_velocity_ned call
    ):
        self._dt_default = float(dt)
        self._vel_tau = float(vel_tau)
        self._yaw_tau = float(yaw_tau)
        self._accel_noise = float(accel_noise_sigma)
        self._gyro_noise = float(gyro_noise_sigma)
        self._accel_bias = (
            np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float).copy()
        )
        self._gyro_bias = (
            np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float).copy()
        )
        self._rng = np.random.default_rng(seed)
        self._auto_step = bool(auto_step)
        # Remember the seed altitude so reset() can restore it. Without
        # this, a reset() drops the drone to the ground and (worse)
        # clears auto_step — MockDCLAdapter.connect() hit exactly this
        # footgun during its first smoke test.
        self._initial_altitude_m = float(initial_altitude_m)
        self._seed = int(seed)

        # Truth state (NED world). Body att is level (roll=pitch=0), yaw-only.
        self._pos = np.array([0.0, 0.0, -float(initial_altitude_m)])
        self._vel = np.zeros(3)
        self._yaw = 0.0

        # Setpoints from send_velocity_ned.
        self._vel_cmd = np.zeros(3)
        self._yaw_cmd = 0.0

        # Elapsed sim time + last-sample snapshot.
        self._t = 0.0
        self._last_imu: Optional[IMUReading] = None
        self._armed = False
        self._connected = False

    # ── Test-facing helpers ────────────────────────────────────────────

    def set_truth(
        self,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        yaw_rad: Optional[float] = None,
    ) -> None:
        """Hard-set the truth state. Tests use this to place the drone
        at a known condition before integrating. Also resyncs the
        command setpoint so the first-order filter doesn't immediately
        snap back to zero."""
        if pos is not None:
            self._pos = np.asarray(pos, dtype=float).copy()
        if vel is not None:
            self._vel = np.asarray(vel, dtype=float).copy()
            self._vel_cmd = self._vel.copy()
        if yaw_rad is not None:
            self._yaw = float(yaw_rad)
            self._yaw_cmd = self._yaw

    def step(self, dt: Optional[float] = None) -> IMUReading:
        """Advance truth by `dt` (seconds), synthesize and return an IMU
        reading for this tick. Also caches it for `get_imu()`."""
        dt = float(dt) if dt is not None else self._dt_default
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # First-order tracking toward commanded velocity and yaw.
        # alpha = 1 - exp(-dt/tau) is the stable discretization.
        alpha_v = 1.0 - np.exp(-dt / max(self._vel_tau, 1e-9))
        alpha_y = 1.0 - np.exp(-dt / max(self._yaw_tau, 1e-9))

        new_vel = self._vel + alpha_v * (self._vel_cmd - self._vel)
        yaw_err = _wrap_pi(self._yaw_cmd - self._yaw)
        new_yaw = _wrap_pi(self._yaw + alpha_y * yaw_err)

        # World-frame acceleration and yaw rate over this tick.
        a_world = (new_vel - self._vel) / dt
        yaw_rate = _wrap_pi(new_yaw - self._yaw) / dt

        # Integrate position (trapezoidal for slightly better truth).
        self._pos = self._pos + 0.5 * (self._vel + new_vel) * dt

        # Advance state.
        self._vel = new_vel
        self._yaw = new_yaw
        self._t += dt

        # Synthesize IMU. Level flight ⇒ R = Rz(yaw); R^T rotates world
        # vectors into the body frame.
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        R_T = np.array([[ cy, sy, 0.0],
                        [-sy, cy, 0.0],
                        [0.0, 0.0, 1.0]])

        # Specific force measured by accelerometer: a_world - g_NED,
        # rotated into body. Matches the ESKF predict() convention.
        specific_force_world = a_world - GRAVITY_NED
        accel_body = R_T @ specific_force_world
        gyro_body = np.array([0.0, 0.0, yaw_rate])

        # Add bias + white noise.
        if self._accel_noise > 0.0:
            accel_body = accel_body + self._rng.normal(0.0, self._accel_noise, size=3)
        accel_body = accel_body + self._accel_bias

        if self._gyro_noise > 0.0:
            gyro_body = gyro_body + self._rng.normal(0.0, self._gyro_noise, size=3)
        gyro_body = gyro_body + self._gyro_bias

        imu = IMUReading(
            accel_body=accel_body,
            gyro_body=gyro_body,
            timestamp=self._t,
        )
        self._last_imu = imu
        return imu

    # ── SimAdapter surface ─────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def reset(self) -> None:
        # Re-seed via __init__ so every invariant gets zeroed in one
        # place. All configuration kwargs must be forwarded — a missing
        # kwarg silently reverts to its default and that's how
        # MockDCLAdapter.connect() ended up with a ground-fixed drone
        # and auto_step=False after S19t.
        self.__init__(
            dt=self._dt_default,
            vel_tau=self._vel_tau,
            yaw_tau=self._yaw_tau,
            accel_noise_sigma=self._accel_noise,
            gyro_noise_sigma=self._gyro_noise,
            accel_bias=self._accel_bias,
            gyro_bias=self._gyro_bias,
            seed=self._seed,
            initial_altitude_m=self._initial_altitude_m,
            auto_step=self._auto_step,
        )

    async def get_state(self) -> SimState:
        return SimState(
            pos_ned=tuple(float(x) for x in self._pos),
            vel_ned=tuple(float(x) for x in self._vel),
            att_rad=(0.0, 0.0, float(self._yaw)),
            timestamp=float(self._t),
            armed=self._armed,
            connected=self._connected,
        )

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        return None

    async def get_imu(self) -> Optional[IMUReading]:
        return self._last_imu

    async def send_velocity_ned(self, vn, ve, vd, yaw_deg) -> None:
        self._vel_cmd = np.array([float(vn), float(ve), float(vd)])
        self._yaw_cmd = _wrap_pi(np.radians(float(yaw_deg)))
        if self._auto_step:
            # Race-loop-compatible mode: each command advances the sim
            # by the default tick, so the next get_state()/get_imu()
            # reflects motion under this command.
            self.step(self._dt_default)

    async def send_position_ned(self, n, e, d, yaw_deg) -> None:
        # Not modeled — position-hold is outside this mock's scope.
        pass

    async def send_attitude(self, roll_deg, pitch_deg, yaw_deg, thrust) -> None:
        # Attitude control not modeled (level flight only).
        pass

    async def arm(self) -> None:
        self._armed = True

    async def disarm(self) -> None:
        self._armed = False

    async def takeoff(self, altitude_m: float) -> None:
        # Snap to altitude; tests that care about takeoff dynamics should
        # drive step() manually.
        self._pos[2] = -float(altitude_m)

    async def land(self) -> None:
        self._pos[2] = 0.0
        self._vel[:] = 0.0
        self._vel_cmd[:] = 0.0

    async def start_offboard(self, initial_mode: str = "velocity") -> None:
        pass

    async def stop_offboard(self) -> None:
        pass

    def info(self) -> SimInfo:
        return SimInfo(
            backend="mock_kinematic",
            capabilities=self.capabilities,
            tick_rate_hz=1.0 / self._dt_default,
            notes=(
                "Kinematic sim with first-order vel/yaw tracking and "
                "synthetic IMU. Level flight only. For ESKF integration tests."
            ),
        )
