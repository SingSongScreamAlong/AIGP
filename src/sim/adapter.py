"""Sim adapter abstraction — Session 19 scaffold.

Lets the control/planning stack talk to more than one sim backend
without changing call sites. Today's backends:

    PX4SITLAdapter  — wraps mavsdk-python connected to PX4 SITL (or
                      mock_sim.py). This is what s18/control_skeleton
                      currently use.
    DCLSimAdapter   — stub. The DCL AI Race League simulator is
                      expected to drop ~May 2026. The surface here
                      documents the methods the rest of the stack
                      already needs, so when the real API shows up we
                      touch ONE file instead of every call site.

Design rules:
  * NED coordinates everywhere (north, east, down). Yaws are degrees
    on the wire because that's what mavsdk uses and what the existing
    planner emits; internal math elsewhere stays radians.
  * Async throughout — mavsdk is async, and a gym-sync DCL API can
    be trivially awaited via `await asyncio.to_thread(...)`.
  * Capability flags on each adapter so callers can gracefully
    degrade (e.g. skip vision loop when the backend has no camera).
  * No global state. Controllers construct an adapter, hold it,
    close it. No module-level singletons.

This file intentionally does NOT rewrite control_skeleton_v5 to use
the adapter — that migration is a follow-up. The adapter exists first
so the DCL stub has a spec to match.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Optional, Protocol, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SimState:
    """Single-tick snapshot of drone state in NED frame.

    Units:
      pos_ned   — meters
      vel_ned   — m/s
      att_rad   — (roll, pitch, yaw) radians
      timestamp — seconds (monotonic-ish; each adapter says what it is)
      armed     — bool, best-effort
      connected — bool
    """
    pos_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    att_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: float = 0.0
    armed: bool = False
    connected: bool = False


class SimCapability(Flag):
    """What a backend can actually do.

    Consumers check (adapter.capabilities & SimCapability.X) before
    calling methods that only some backends support. Any method whose
    capability is missing must raise NotImplementedError, never
    silently no-op.
    """
    NONE            = 0
    VELOCITY_NED    = auto()   # send_velocity_ned
    POSITION_NED    = auto()   # send_position_ned
    ATTITUDE        = auto()   # send_attitude
    CAMERA_RGB      = auto()   # get_camera_frame returns HxWx3 uint8
    ARM_ACTION      = auto()   # arm / disarm / takeoff / land
    RESET           = auto()   # reset() actually restores t=0 state
    WALLCLOCK_PACED = auto()   # sim runs in real time (PX4 SITL does)
    IMU             = auto()   # get_imu() returns raw accelerometer + gyro readings


@dataclass
class SimInfo:
    """Static metadata about a backend. Returned by info()."""
    backend: str                          # "px4_sitl", "dcl", "mock", ...
    capabilities: SimCapability
    tick_rate_hz: Optional[float] = None  # None = wallclock / variable
    notes: str = ""


@dataclass
class IMUReading:
    """One IMU sample from a backend that advertises SimCapability.IMU.

    Duck-type compatible with `estimation.pose_fusion.IMUSample` — same
    three fields, same units and conventions. Backends may produce this
    synthetically (MockKinematicAdapter) or pass through real hardware
    readings (PX4 raw_imu, DCL obs['imu']).

    Conventions:
      accel_body — specific force in body FRD (m/s²), gravity-loaded.
                   A level drone at rest reads ≈ [0, 0, -9.81] (body z is
                   Down, gravity points Down in NED, so the accelerometer
                   measures the reaction force pointing UP in the body z
                   axis ⇒ negative z).
      gyro_body  — angular rate about body FRD axes (rad/s).
      timestamp  — monotonic-ish seconds (the backend says what epoch).
    """
    accel_body: np.ndarray
    gyro_body: np.ndarray
    timestamp: float


# ─────────────────────────────────────────────────────────────────────
# Protocol — the contract every adapter honors
# ─────────────────────────────────────────────────────────────────────

class SimAdapter(Protocol):
    """Contract for any sim backend the race stack can run against.

    All methods are async. Implementations are free to do heavy lifting
    in threads (see DCLSimAdapter for that pattern) — the protocol is
    async because the PX4 path is already async and we want one shape.
    """

    capabilities: SimCapability

    # Lifecycle ────────────────────────────────────────
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def reset(self) -> None: ...

    # Telemetry ────────────────────────────────────────
    async def get_state(self) -> SimState: ...
    async def get_camera_frame(self) -> Optional[np.ndarray]: ...
    async def get_imu(self) -> Optional[IMUReading]: ...

    # Commands ─────────────────────────────────────────
    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_deg: float
    ) -> None: ...
    async def send_position_ned(
        self, n: float, e: float, d: float, yaw_deg: float
    ) -> None: ...
    async def send_attitude(
        self, roll_deg: float, pitch_deg: float, yaw_deg: float, thrust: float
    ) -> None: ...

    # Action / flight mode ─────────────────────────────
    async def arm(self) -> None: ...
    async def disarm(self) -> None: ...
    async def takeoff(self, altitude_m: float) -> None: ...
    async def land(self) -> None: ...
    async def start_offboard(self, initial_mode: str = "velocity") -> None: ...
    async def stop_offboard(self) -> None: ...

    # Introspection ────────────────────────────────────
    def info(self) -> SimInfo: ...


# ─────────────────────────────────────────────────────────────────────
# PX4 SITL adapter — wraps mavsdk-python
# ─────────────────────────────────────────────────────────────────────

class PX4SITLAdapter:
    """Adapter over mavsdk-python talking to PX4 SITL (or mock_sim.py).

    This backs the current race stack. It does not own the PX4 process —
    the existing `bench.py` start/kill harness still does that. This
    class is just the per-flight client.

    Reset is best-effort: mavsdk has no scenario-reset, so reset()
    lands + disarms. Full scenario reset still requires bench.py killing
    and relaunching the SITL process; that's why RESET is not in
    capabilities.
    """

    capabilities = (
        SimCapability.VELOCITY_NED
        | SimCapability.POSITION_NED
        | SimCapability.ATTITUDE
        | SimCapability.ARM_ACTION
        | SimCapability.WALLCLOCK_PACED
    )

    def __init__(self, connection_string: str = "udpin://0.0.0.0:14540"):
        self.conn_str = connection_string

        # Import mavsdk lazily. Keeps the module importable on machines
        # that only have the DCL sim installed.
        from mavsdk import System
        self._System = System

        self._drone = None
        self._state = SimState()
        self._telem_tasks: list[asyncio.Task] = []

    # Lifecycle ────────────────────────────────────────
    async def connect(self) -> None:
        self._drone = self._System()
        await self._drone.connect(system_address=self.conn_str)
        async for cs in self._drone.core.connection_state():
            if cs.is_connected:
                self._state = SimState(connected=True)
                break
        await self._start_telemetry()

    async def disconnect(self) -> None:
        for t in self._telem_tasks:
            t.cancel()
        self._telem_tasks.clear()
        self._drone = None
        self._state = SimState()

    async def reset(self) -> None:
        # mavsdk has no scenario reset; best effort is land+disarm.
        # Full reset must come from bench.py killing PX4 SITL.
        if self._drone is None:
            return
        try:
            await self._drone.action.land()
        except Exception:
            pass
        try:
            await self._drone.action.disarm()
        except Exception:
            pass

    # Telemetry ────────────────────────────────────────
    async def _start_telemetry(self) -> None:
        import math

        async def pos_loop():
            async for p in self._drone.telemetry.position_velocity_ned():
                self._state = SimState(
                    pos_ned=(p.position.north_m, p.position.east_m, p.position.down_m),
                    vel_ned=(p.velocity.north_m_s, p.velocity.east_m_s, p.velocity.down_m_s),
                    att_rad=self._state.att_rad,
                    timestamp=asyncio.get_event_loop().time(),
                    armed=self._state.armed,
                    connected=True,
                )

        async def att_loop():
            async for a in self._drone.telemetry.attitude_euler():
                self._state = SimState(
                    pos_ned=self._state.pos_ned,
                    vel_ned=self._state.vel_ned,
                    att_rad=(math.radians(a.roll_deg),
                             math.radians(a.pitch_deg),
                             math.radians(a.yaw_deg)),
                    timestamp=self._state.timestamp,
                    armed=self._state.armed,
                    connected=True,
                )

        async def armed_loop():
            async for armed in self._drone.telemetry.armed():
                self._state = SimState(
                    pos_ned=self._state.pos_ned,
                    vel_ned=self._state.vel_ned,
                    att_rad=self._state.att_rad,
                    timestamp=self._state.timestamp,
                    armed=bool(armed),
                    connected=True,
                )

        self._telem_tasks = [
            asyncio.ensure_future(pos_loop()),
            asyncio.ensure_future(att_loop()),
            asyncio.ensure_future(armed_loop()),
        ]

    async def get_state(self) -> SimState:
        return self._state

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        # PX4 SITL via mavsdk-python doesn't deliver camera frames.
        # Vision for this backend comes from VirtualCamera (synthetic
        # projection) or a separate Gazebo camera bridge. Callers that
        # need CAMERA_RGB must pick a different adapter.
        return None

    async def get_imu(self) -> Optional[IMUReading]:
        # mavsdk exposes `telemetry.imu()` (scaled_imu) and `raw_imu()`
        # but the streams are noisy and not all PX4 builds publish them
        # at useful rates. When we're on real Neros hardware the IMU
        # comes straight from the flight-controller microros bridge, not
        # mavsdk. Leaving this as a hook: callers that need IMU fusion
        # in PX4 SITL should either subscribe via mavsdk.telemetry.imu
        # in a follow-up patch or inject synthetic IMU via a wrapper.
        return None

    # Commands ─────────────────────────────────────────
    async def send_velocity_ned(self, vn, ve, vd, yaw_deg) -> None:
        from mavsdk.offboard import VelocityNedYaw
        await self._drone.offboard.set_velocity_ned(
            VelocityNedYaw(vn, ve, vd, yaw_deg)
        )

    async def send_position_ned(self, n, e, d, yaw_deg) -> None:
        from mavsdk.offboard import PositionNedYaw
        await self._drone.offboard.set_position_ned(
            PositionNedYaw(n, e, d, yaw_deg)
        )

    async def send_attitude(self, roll_deg, pitch_deg, yaw_deg, thrust) -> None:
        from mavsdk.offboard import Attitude
        await self._drone.offboard.set_attitude(
            Attitude(roll_deg, pitch_deg, yaw_deg, thrust)
        )

    # Action / flight mode ─────────────────────────────
    async def arm(self) -> None:
        await self._drone.action.arm()

    async def disarm(self) -> None:
        await self._drone.action.disarm()

    async def takeoff(self, altitude_m: float) -> None:
        try:
            await self._drone.action.set_takeoff_altitude(altitude_m)
        except Exception:
            pass  # not all mavsdk versions expose this
        await self._drone.action.takeoff()

    async def land(self) -> None:
        await self._drone.action.land()

    async def start_offboard(self, initial_mode: str = "velocity") -> None:
        # mavsdk requires a setpoint before start()
        if initial_mode == "velocity":
            await self.send_velocity_ned(0.0, 0.0, 0.0, 0.0)
        else:
            s = await self.get_state()
            await self.send_position_ned(*s.pos_ned, 0.0)
        await self._drone.offboard.start()

    async def stop_offboard(self) -> None:
        await self._drone.offboard.stop()

    # Introspection ────────────────────────────────────
    def info(self) -> SimInfo:
        return SimInfo(
            backend="px4_sitl",
            capabilities=self.capabilities,
            tick_rate_hz=None,  # wallclock
            notes="mavsdk-python over UDP; no camera, no scenario reset.",
        )


# ─────────────────────────────────────────────────────────────────────
# DCL sim adapter — stub, fill in when the sim drops
# ─────────────────────────────────────────────────────────────────────

class DCLSimAdapter:
    """Adapter for the DCL AI Race League simulator. STUB.

    The DCL sim is expected to ship around May 2026 as a Python-
    accessible race simulator with the FPV camera rendered in-sim.
    The public API has not been published. This stub exists so:

      1. The rest of the stack can be written against SimAdapter.
      2. We have a single file to edit once DCL lands (swap in the
         real imports + bodies; keep the method signatures).
      3. Expected capabilities are documented now so consumers like
         YOLO/PnP know they can ask for CAMERA_RGB here.

    Every method raises NotImplementedError with a specific hint about
    what we expect to map it to. Do NOT silently no-op — that would let
    the race stack ship with broken control in a way that only shows up
    in-sim.
    """

    # Best guess at capabilities; adjust when the real API is published.
    capabilities = (
        SimCapability.VELOCITY_NED
        | SimCapability.POSITION_NED
        | SimCapability.CAMERA_RGB
        | SimCapability.IMU
        | SimCapability.RESET
    )

    def __init__(self, scenario: str = "round1_simple", seed: Optional[int] = None):
        self.scenario = scenario
        self.seed = seed
        self._env = None  # will hold the DCL env/client handle

    # Lifecycle ────────────────────────────────────────
    async def connect(self) -> None:
        # EXPECTED: `self._env = dcl.make(scenario=self.scenario, seed=self.seed)`
        # or similar gym-style constructor. Wrap sync calls via
        # asyncio.to_thread so the race loop stays non-blocking.
        raise NotImplementedError(
            "DCLSimAdapter.connect: waiting on DCL Python API (~May 2026). "
            "Expected shape: dcl.make(scenario=...) returning a gym-like env."
        )

    async def disconnect(self) -> None:
        raise NotImplementedError("DCLSimAdapter.disconnect: expected env.close().")

    async def reset(self) -> None:
        # EXPECTED: obs = env.reset(seed=self.seed); store obs → self._state
        raise NotImplementedError(
            "DCLSimAdapter.reset: gym-style reset() returning initial observation."
        )

    # Telemetry ────────────────────────────────────────
    async def get_state(self) -> SimState:
        # EXPECTED: pull pos/vel/att out of the most recent obs dict.
        raise NotImplementedError(
            "DCLSimAdapter.get_state: map obs['pose']/obs['twist'] → SimState."
        )

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        # EXPECTED: obs['camera'] as HxWx3 uint8 BGR or RGB; document
        # whichever DCL ships and convert here so downstream vision is
        # invariant.
        raise NotImplementedError(
            "DCLSimAdapter.get_camera_frame: expected HxWx3 uint8 from obs."
        )

    async def get_imu(self) -> Optional[IMUReading]:
        # EXPECTED: obs['imu'] → {'accel': [ax, ay, az], 'gyro': [wx, wy, wz]}
        # in body FRD with accel in m/s² (specific force, gravity-loaded)
        # and gyro in rad/s. If DCL publishes gyro in deg/s or accel in g's,
        # convert here so the fusion stack sees consistent SI units.
        raise NotImplementedError(
            "DCLSimAdapter.get_imu: map obs['imu'] → IMUReading (body FRD, "
            "specific force in m/s², gyro in rad/s)."
        )

    # Commands ─────────────────────────────────────────
    async def send_velocity_ned(self, vn, ve, vd, yaw_deg) -> None:
        # EXPECTED: env.step({'cmd': 'vel_ned', 'vn': vn, ...}) or
        # similar. The race stack prefers velocity commands (V5.1
        # planner emits these), so this is the hot path.
        raise NotImplementedError(
            "DCLSimAdapter.send_velocity_ned: hot path — wire first."
        )

    async def send_position_ned(self, n, e, d, yaw_deg) -> None:
        raise NotImplementedError("DCLSimAdapter.send_position_ned")

    async def send_attitude(self, roll_deg, pitch_deg, yaw_deg, thrust) -> None:
        # Attitude control may or may not be supported by DCL. If not,
        # strip the ATTITUDE capability flag on publish.
        raise NotImplementedError("DCLSimAdapter.send_attitude")

    # Action / flight mode ─────────────────────────────
    # DCL sim likely starts the drone already flying — gym-style envs
    # usually don't model arm/takeoff. These are no-ops by default;
    # override if DCL exposes them.
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

    # Introspection ────────────────────────────────────
    def info(self) -> SimInfo:
        return SimInfo(
            backend="dcl",
            capabilities=self.capabilities,
            tick_rate_hz=None,  # unknown until API ships
            notes="STUB — awaiting DCL Python API drop (~May 2026).",
        )


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────

def make_adapter(backend: str, **kwargs) -> SimAdapter:
    """Construct an adapter by name. Keeps call sites decoupled.

    Known backends:
      "px4_sitl" — PX4SITLAdapter(connection_string=...)
      "dcl"     — DCLSimAdapter(scenario=..., seed=...)

    Raises ValueError on unknown backend so a typo fails loudly.
    """
    if backend == "px4_sitl":
        return PX4SITLAdapter(**kwargs)  # type: ignore[return-value]
    if backend == "dcl":
        return DCLSimAdapter(**kwargs)   # type: ignore[return-value]
    if backend == "mock_kinematic":
        # Lazy import to avoid circular dep on estimation-facing users
        # that don't need the kinematic mock.
        from .mock import MockKinematicAdapter
        return MockKinematicAdapter(**kwargs)  # type: ignore[return-value]
    if backend == "mock_dcl":
        # Pre-landing stand-in for DCLSimAdapter. See src/sim/mock_dcl.py
        # and docs/DCL_INTEGRATION_CHECKLIST.md. Deletable once the real
        # DCL SDK is wired into DCLSimAdapter.
        from .mock_dcl import MockDCLAdapter
        return MockDCLAdapter(**kwargs)  # type: ignore[return-value]
    raise ValueError(
        f"Unknown sim backend: {backend!r}. "
        f"Known: 'px4_sitl', 'dcl', 'mock_kinematic', 'mock_dcl'."
    )
