"""MockDCLAdapter — pre-landing stand-in for DCLSimAdapter.

Purpose: smoke-test the full `--backend dcl`-ish CLI chain BEFORE the
real DCL AI Race League SDK ships (~May 2026). The goal is to catch
PX4-assumption leaks in RaceRunner / RaceLoop now, rather than on
day-1 when the real SDK lands.

Shape: matches `DCLSimAdapter`'s *expected* capability set — no
ARM_ACTION (gym-style drones are airborne at reset), no WALLCLOCK_PACED
(gym envs advance per step() call). Physics is delegated to
`MockKinematicAdapter`; the only thing added on top is a placeholder
camera frame (HxWx3 uint8 zeros) so `get_camera_frame()` actually
returns bytes, exercising any downstream code path that assumes a
live camera.

Not a model of DCL. It is a scaffold for validating our adapter seam.
When the real SDK ships, fill in `DCLSimAdapter` and either delete
this file or keep it around as an offline-testing fallback — do NOT
extend it to try to mimic DCL's behaviour. See
`docs/DCL_INTEGRATION_CHECKLIST.md` for the day-1 punch list.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .adapter import IMUReading, SimCapability, SimInfo, SimState
from .mock import MockKinematicAdapter


class MockDCLAdapter:
    """DCL-shaped adapter backed by MockKinematicAdapter physics.

    Capabilities mirror `DCLSimAdapter` (VELOCITY_NED | POSITION_NED |
    CAMERA_RGB | IMU | RESET) — notably NO ARM_ACTION and NO
    WALLCLOCK_PACED, so RaceRunner's lifecycle branch skips arm/takeoff
    and the loop runs as fast as asyncio lets it.

    Delegation map:
      connect/disconnect/reset  → MockKinematicAdapter
      get_state / get_imu       → MockKinematicAdapter
      send_velocity/position_ned → MockKinematicAdapter
      get_camera_frame           → zeros frame (self)
      send_attitude              → raise NotImplementedError (match worst-case DCL)
      arm/disarm/takeoff/land    → no-op (gym-style)
      start/stop_offboard        → no-op (gym-style)
    """

    capabilities = (
        SimCapability.VELOCITY_NED
        | SimCapability.POSITION_NED
        | SimCapability.CAMERA_RGB
        | SimCapability.IMU
        | SimCapability.RESET
    )

    DEFAULT_FRAME_H = 480
    DEFAULT_FRAME_W = 640

    def __init__(
        self,
        dt: float = 1.0 / 50,
        vel_tau: float = 0.05,
        yaw_tau: float = 0.10,
        auto_step: bool = True,
        initial_altitude_m: float = 1.0,
        frame_h: int = DEFAULT_FRAME_H,
        frame_w: int = DEFAULT_FRAME_W,
        seed: int = 0,
    ):
        # Physics delegate — same kinematic model MockKinematicAdapter uses
        # for fusion tests. auto_step=True means each send_velocity_ned
        # advances the truth state + emits a fresh IMU sample.
        self._kin = MockKinematicAdapter(
            dt=dt,
            vel_tau=vel_tau,
            yaw_tau=yaw_tau,
            auto_step=auto_step,
            initial_altitude_m=initial_altitude_m,
            seed=seed,
        )
        # Placeholder camera frame. Real DCL will return meaningful bytes;
        # until then, any test that wants actual image content should use
        # VirtualDetector (which renders from VirtualCamera, not adapter
        # frames) and will ignore this.
        self._frame = np.zeros((int(frame_h), int(frame_w), 3), dtype=np.uint8)

    # ── Lifecycle ────────────────────────────────────────
    async def connect(self) -> None:
        await self._kin.connect()
        # Gym-style: a "connect" is really instantiate + first reset.
        await self._kin.reset()

    async def disconnect(self) -> None:
        await self._kin.disconnect()

    async def reset(self) -> None:
        await self._kin.reset()

    # ── Telemetry ────────────────────────────────────────
    async def get_state(self) -> SimState:
        return await self._kin.get_state()

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        # Return a *new view* of the zeros frame. Intentionally the same
        # underlying bytes each call — the frame is a placeholder, so
        # copying per call would waste cycles. Downstream code should
        # treat frames as read-only anyway.
        return self._frame

    async def get_imu(self) -> Optional[IMUReading]:
        return await self._kin.get_imu()

    # ── Commands ─────────────────────────────────────────
    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_deg: float
    ) -> None:
        await self._kin.send_velocity_ned(vn, ve, vd, yaw_deg)

    async def send_position_ned(
        self, n: float, e: float, d: float, yaw_deg: float
    ) -> None:
        await self._kin.send_position_ned(n, e, d, yaw_deg)

    async def send_attitude(
        self, roll_deg: float, pitch_deg: float, yaw_deg: float, thrust: float
    ) -> None:
        # DCL is not expected to support attitude control. Matching the
        # worst-case assumption here means code paths that branch on
        # `SimCapability.ATTITUDE` will stay honest during smoke tests.
        raise NotImplementedError(
            "MockDCLAdapter.send_attitude: DCL is not expected to support "
            "attitude commands. If the real SDK does, add SimCapability."
            "ATTITUDE to both DCLSimAdapter.capabilities AND the real "
            "implementation, not here."
        )

    # ── Action / flight mode ─────────────────────────────
    # Gym-style: drone is airborne at reset, no arm/takeoff/land model.
    # RaceRunner checks SimCapability.ARM_ACTION before calling these,
    # but they stay as no-ops in case something else in the stack calls
    # them directly.
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
            tick_rate_hz=1.0 / self._kin._dt_default if self._kin._dt_default > 0 else None,
            notes=(
                "STAND-IN for DCLSimAdapter pre-landing smoke tests. "
                "Zeros camera frames; physics from MockKinematicAdapter. "
                "Deletable once real DCL SDK lands."
            ),
        )
