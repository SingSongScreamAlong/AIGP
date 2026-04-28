"""MockDCLAdapter — pre-landing stand-in for DCLSimAdapter.

Purpose: smoke-test the full `--backend dcl`-ish CLI chain BEFORE the
real DCL AI Race League SDK ships (~May 2026). The goal is to catch
PX4-assumption leaks in RaceRunner / RaceLoop now, rather than on
day-1 when the real SDK lands.

Shape: matches `DCLSimAdapter`'s *expected* capability set — no
ARM_ACTION (gym-style drones are airborne at reset), no WALLCLOCK_PACED
(gym envs advance per step() call).

Physics: a simplified attitude dynamics model where roll/pitch tilt
the thrust vector to produce lateral/forward acceleration, throttle
controls vertical thrust, and yaw tracks a heading setpoint. This
mirrors the DCL tech spec control interface (Throttle, Roll, Pitch,
Yaw) and exercises the AttitudeController → adapter pipeline.

Not a model of DCL. It is a scaffold for validating our adapter seam.
When the real SDK ships, fill in `DCLSimAdapter` and either delete
this file or keep it around as an offline-testing fallback — do NOT
extend it to try to mimic DCL's behaviour. See
`docs/DCL_INTEGRATION_CHECKLIST.md` for the day-1 punch list.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .adapter import IMUReading, SimCapability, SimInfo, SimState


# Gravity constant (m/s²)
_G = 9.81


def _wrap_pi(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MockDCLAdapter:
    """DCL-shaped adapter with simplified attitude dynamics.

    Capabilities mirror `DCLSimAdapter` (ATTITUDE | CAMERA_RGB | IMU |
    RESET) — notably NO VELOCITY_NED (DCL uses T/R/P/Y stick commands),
    NO ARM_ACTION, and NO WALLCLOCK_PACED, so RaceRunner's lifecycle
    branch skips arm/takeoff and the loop runs as fast as asyncio lets it.

    Physics model (simplified):
      - Thrust vector points body-up; roll/pitch tilt it to produce
        lateral and forward acceleration.
      - throttle ∈ [0, 1] → total_thrust = throttle * max_thrust
      - Roll φ, Pitch θ → horizontal accel components via tilt.
      - Yaw tracks a heading setpoint via first-order filter.
      - Drag proportional to velocity (simple damping).

    Delegation map:
      send_attitude              → primary control path (attitude dynamics)
      send_velocity_ned          → raise NotImplementedError (DCL doesn't support)
      send_position_ned          → raise NotImplementedError (DCL doesn't support)
      get_camera_frame           → zeros frame (placeholder)
      arm/disarm/takeoff/land    → no-op (gym-style)
      start/stop_offboard        → no-op (gym-style)
    """

    capabilities = (
        SimCapability.ATTITUDE
        | SimCapability.CAMERA_RGB
        | SimCapability.IMU
        | SimCapability.RESET
    )

    DEFAULT_FRAME_H = 480
    DEFAULT_FRAME_W = 640

    def __init__(
        self,
        dt: float = 1.0 / 50,
        max_thrust_accel: float = 20.0,  # m/s², ~2g max
        drag_coeff: float = 0.3,         # velocity damping
        yaw_tau: float = 0.10,           # yaw tracking time constant
        initial_altitude_m: float = 1.0,
        frame_h: int = DEFAULT_FRAME_H,
        frame_w: int = DEFAULT_FRAME_W,
        seed: int = 0,
    ):
        self._dt = float(dt)
        self._max_thrust = float(max_thrust_accel)
        self._drag = float(drag_coeff)
        self._yaw_tau = float(yaw_tau)
        self._initial_alt = float(initial_altitude_m)
        self._frame_h = int(frame_h)
        self._frame_w = int(frame_w)
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

        # Truth state (NED world)
        self._pos = np.array([0.0, 0.0, -float(initial_altitude_m)])
        self._vel = np.zeros(3)
        self._yaw = 0.0                # rad
        self._roll = 0.0               # rad (truth, from commands)
        self._pitch = 0.0              # rad (truth, from commands)
        self._t = 0.0

        # Command setpoints (from send_attitude)
        self._cmd_throttle = 0.5       # hover-ish default
        self._cmd_roll_rad = 0.0
        self._cmd_pitch_rad = 0.0
        self._cmd_yaw_rad = 0.0

        self._last_imu: Optional[IMUReading] = None
        self._connected = False

        # Placeholder camera frame
        self._frame = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)

    # ── Physics step ──────────────────────────────────────
    def _step(self, dt: Optional[float] = None) -> IMUReading:
        """Advance attitude dynamics by one tick. Returns IMU reading."""
        dt = float(dt) if dt is not None else self._dt

        # Apply attitude commands immediately (stick → attitude).
        # In reality there's a rate controller; we model instant response
        # for now, which is conservative (real drone is slower).
        self._roll = self._cmd_roll_rad
        self._pitch = self._cmd_pitch_rad

        # Yaw tracks setpoint via first-order filter
        alpha_y = 1.0 - math.exp(-dt / max(self._yaw_tau, 1e-9))
        yaw_err = _wrap_pi(self._cmd_yaw_rad - self._yaw)
        new_yaw = _wrap_pi(self._yaw + alpha_y * yaw_err)
        yaw_rate = _wrap_pi(new_yaw - self._yaw) / dt if dt > 0 else 0.0
        self._yaw = new_yaw

        # Thrust model: total_thrust along body-up (negative body-z in FRD)
        total_accel = self._cmd_throttle * self._max_thrust

        # Decompose thrust vector into NED acceleration.
        # Body-up in NED when rolled by φ and pitched by θ:
        #   a_north = total_accel * (-sin(θ) * cos(ψ) + sin(φ)*cos(θ)*sin(ψ)) ... simplified:
        # For small-to-moderate angles, standard decomposition:
        cr = math.cos(self._roll)
        sr = math.sin(self._roll)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)

        # Thrust in body frame points up (negative z in FRD) = [0, 0, -total_accel]
        # Rotate to NED: R_body_to_ned @ [0, 0, -total_accel]
        # Using ZYX Euler rotation (yaw, pitch, roll):
        # R_body_to_ned @ [0, 0, -T] with ZYX Euler convention:
        #   North = -T*(cy*sp*cr + sy*sr)
        #   East  =  T*(cy*sr - sy*sp*cr)
        #   Down  = -T*cp*cr
        thrust_ned = np.array([
            -total_accel * (cy * sp * cr + sy * sr),   # North
            total_accel * (cy * sr - sy * sp * cr),    # East
            -total_accel * cp * cr,                     # Down (thrust opposes gravity)
        ])

        # Net acceleration = thrust + gravity - drag
        a_ned = thrust_ned + np.array([0.0, 0.0, _G]) - self._drag * self._vel

        # Integrate (semi-implicit Euler)
        new_vel = self._vel + a_ned * dt
        self._pos = self._pos + 0.5 * (self._vel + new_vel) * dt
        old_vel = self._vel.copy()
        self._vel = new_vel
        self._t += dt

        # Synthesize IMU
        a_world = (self._vel - old_vel) / dt if dt > 0 else np.zeros(3)
        # Body rotation matrix transpose (NED → body)
        R_T = np.array([
            [ cy*cp, sy*cp, -sp],
            [ cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
            [ cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr],
        ])
        specific_force = a_world - np.array([0.0, 0.0, _G])
        accel_body = R_T @ specific_force
        gyro_body = np.array([0.0, 0.0, yaw_rate])  # simplified: level-ish

        imu = IMUReading(
            accel_body=accel_body,
            gyro_body=gyro_body,
            timestamp=self._t,
        )
        self._last_imu = imu
        return imu

    # ── Lifecycle ────────────────────────────────────────
    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def reset(self) -> None:
        self.__init__(
            dt=self._dt,
            max_thrust_accel=self._max_thrust,
            drag_coeff=self._drag,
            yaw_tau=self._yaw_tau,
            initial_altitude_m=self._initial_alt,
            frame_h=self._frame_h,
            frame_w=self._frame_w,
            seed=self._seed,
        )

    # ── Telemetry ────────────────────────────────────────
    async def get_state(self) -> SimState:
        return SimState(
            pos_ned=tuple(float(x) for x in self._pos),
            vel_ned=tuple(float(x) for x in self._vel),
            att_rad=(float(self._roll), float(self._pitch), float(self._yaw)),
            timestamp=float(self._t),
            armed=True,  # gym-style: always armed
            connected=self._connected,
        )

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        return self._frame

    async def get_imu(self) -> Optional[IMUReading]:
        return self._last_imu

    # ── Commands ─────────────────────────────────────────
    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_deg: float
    ) -> None:
        raise NotImplementedError(
            "MockDCLAdapter.send_velocity_ned: DCL uses attitude commands "
            "(Throttle/Roll/Pitch/Yaw), not velocity NED. Use "
            "send_attitude() via the AttitudeController."
        )

    async def send_position_ned(
        self, n: float, e: float, d: float, yaw_deg: float
    ) -> None:
        raise NotImplementedError(
            "MockDCLAdapter.send_position_ned: DCL uses attitude commands."
        )

    async def send_attitude(
        self, roll_deg: float, pitch_deg: float, yaw_deg: float, thrust: float
    ) -> None:
        self._cmd_roll_rad = math.radians(float(roll_deg))
        self._cmd_pitch_rad = math.radians(float(pitch_deg))
        self._cmd_yaw_rad = _wrap_pi(math.radians(float(yaw_deg)))
        self._cmd_throttle = max(0.0, min(1.0, float(thrust)))
        # Each command advances the sim by one tick (gym-style).
        self._step(self._dt)

    # ── Action / flight mode ─────────────────────────────
    # Gym-style: drone is airborne at reset, no arm/takeoff/land model.
    async def arm(self) -> None:
        pass

    async def disarm(self) -> None:
        pass

    async def takeoff(self, altitude_m: float) -> None:
        pass

    async def land(self) -> None:
        pass

    async def start_offboard(self, initial_mode: str = "velocity") -> None:
        pass

    async def stop_offboard(self) -> None:
        pass

    # ── Introspection ────────────────────────────────────
    def info(self) -> SimInfo:
        return SimInfo(
            backend="mock_dcl",
            capabilities=self.capabilities,
            tick_rate_hz=1.0 / self._dt if self._dt > 0 else None,
            notes=(
                "STAND-IN for DCLSimAdapter pre-landing smoke tests. "
                "Attitude dynamics model (T/R/P/Y → forces). "
                "Zeros camera frames. Deletable once real DCL SDK lands."
            ),
        )
