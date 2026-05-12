"""DCL Spec SimAdapter — VADR-TS-002 §4 over MAVLink + §4.6 over UDP.

Bridges the spec's wire protocols to the existing `SimAdapter` contract
in `sim.adapter`, so `RaceRunner` / `RaceLoop` / `BeliefNav` can fly
against a real (or mock) DCL simulator without any changes to the
race-loop side of the stack.

Inbound (sim → us, MAVLink):
    HEARTBEAT       — liveness
    ATTITUDE        — roll/pitch/yaw + body rates
    HIGHRES_IMU     — accel + gyro, fed to PoseFusion
    TIMESYNC        — clock-sync echoes (handled by MAVLinkClient)

Inbound (sim → us, UDP 5600):
    JPEG frames     — assembled by VisionStreamReceiver

Outbound (us → sim, MAVLink):
    HEARTBEAT (≥2 Hz, automatic in MAVLinkClient)
    SET_ATTITUDE_TARGET
    SET_POSITION_TARGET_LOCAL_NED (used for velocity setpoints, the
    spec's preferred control surface alongside attitude)

Capabilities advertised:
    ATTITUDE | VELOCITY_NED | CAMERA_RGB | IMU | WALLCLOCK_PACED

Capabilities deliberately NOT advertised:
    ARM_ACTION  — spec §4 says the simulator handles arming and starts
                  the drone airborne; arm()/takeoff()/land() are no-ops
    RESET       — no scenario-reset message in spec; reset is at the
                  sim's discretion
    POSITION_NED — SET_POSITION_TARGET_LOCAL_NED supports position
                  setpoints, but the existing race stack only ever
                  drives velocity, so we skip it for now to keep the
                  capability surface honest

Pose convention:
    The spec does not provide LOCAL_POSITION_NED. Position and ground-
    frame velocity must come from PoseFusion (ESKF over HIGHRES_IMU +
    vision fixes). Until the ESKF is seeded, `get_state()` returns
    pos_ned=(0,0,0) and vel_ned=(0,0,0); only `att_rad` is populated
    from the ATTITUDE message. RaceLoop already supports this: when a
    `pose_fusion` is supplied, it ignores adapter pos/vel and uses the
    filter output instead.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

import numpy as np

from sim.adapter import (
    IMUReading,
    SimCapability,
    SimInfo,
    SimState,
)
from sim.mavlink_client import MAVLinkClient
from vision.stream_receiver import VisionStreamReceiver

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class DCLSpecAdapter:
    """Live VADR-TS-002 adapter."""

    # Deliberately absent: SimCapability.RELIABLE_POSE — VADR-TS-002 §4.3
    # ships only ATTITUDE + HIGHRES_IMU. Position must be fused from IMU
    # + vision; get_state().pos_ned is a stub (0,0,0). RaceLoop honors
    # this by deferring its first-tick PoseFusion seed to the first
    # vision fix instead of seeding from adapter truth.
    capabilities = (
        SimCapability.ATTITUDE
        | SimCapability.VELOCITY_NED
        | SimCapability.CAMERA_RGB
        | SimCapability.IMU
        | SimCapability.WALLCLOCK_PACED
    )

    def __init__(
        self,
        mavlink_conn: str = "udpin://0.0.0.0:14540",
        vision_port: int = 5600,
        vision_bind: str = "0.0.0.0",
        wait_for_sim_s: float = 10.0,
        target_system: int = 1,
        target_component: int = 1,
    ):
        self._mav = MAVLinkClient(connection_string=mavlink_conn)
        self._vision = VisionStreamReceiver(port=vision_port, bind=vision_bind)
        self._wait_for_sim_s = wait_for_sim_s
        self._target_system = target_system
        self._target_component = target_component

        self._connected = False
        self._last_imu_t_received: float = 0.0
        # Cached frame: we decode JPEG → numpy each tick, but only when
        # a new frame_id has arrived. Avoids decoding the same JPEG twice
        # if the race loop ticks faster than 30 Hz.
        self._last_frame_id: Optional[int] = None
        self._last_decoded: Optional[np.ndarray] = None

    # ──────────────────────── lifecycle ────────────────────────

    async def connect(self) -> None:
        # Start the UDP socket FIRST so we don't drop early packets while
        # MAVLink negotiation is in progress.
        self._vision.start()
        self._mav.start()
        # Wait for the simulator's first HEARTBEAT to be sure we're talking.
        got = await asyncio.to_thread(
            self._mav.wait_for_simulator, self._wait_for_sim_s
        )
        if not got:
            raise TimeoutError(
                f"DCLSpecAdapter: no MAVLink HEARTBEAT received within "
                f"{self._wait_for_sim_s}s on {self._mav._conn_str}"
            )
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        # Stop in reverse order: MAVLink first so we don't keep sending
        # heartbeats after the vision UDP socket is gone.
        try:
            self._mav.stop()
        finally:
            self._vision.stop()

    async def reset(self) -> None:
        # No scenario-reset message in the spec. Best we can do is
        # tear-down + bring-up; the sim controls the world state.
        await self.disconnect()
        await self.connect()

    # ──────────────────────── telemetry ────────────────────────

    async def get_state(self) -> SimState:
        att = self._mav.latest_attitude()
        return SimState(
            pos_ned=(0.0, 0.0, 0.0),         # not in spec; fuse from IMU + vision
            vel_ned=(0.0, 0.0, 0.0),
            att_rad=(att.roll, att.pitch, att.yaw),
            timestamp=time.monotonic(),
            armed=True,                       # DCL drones start armed (spec §4)
            connected=self._connected,
        )

    async def get_imu(self) -> Optional[IMUReading]:
        imu = self._mav.latest_imu()
        # If we've never received an IMU sample, time_usec is zero.
        if imu.time_usec == 0:
            return None
        # De-duplicate: only return a fresh sample, not the same one twice.
        if imu.received_monotonic_s <= self._last_imu_t_received:
            return None
        self._last_imu_t_received = imu.received_monotonic_s
        accel = np.array([imu.xacc, imu.yacc, imu.zacc], dtype=float)
        gyro = np.array([imu.xgyro, imu.ygyro, imu.zgyro], dtype=float)
        # HIGHRES_IMU.time_usec is sim-side microseconds since boot.
        return IMUReading(
            accel_body=accel,
            gyro_body=gyro,
            timestamp=imu.time_usec * 1e-6,
        )

    async def get_camera_frame(self) -> Optional[np.ndarray]:
        if not HAS_CV2:
            return None
        item = self._vision.latest_frame(timeout_s=0.0)
        if item is None:
            return self._last_decoded
        jpeg, _sim_time_ns, frame_id = item
        if frame_id == self._last_frame_id:
            return self._last_decoded
        # Decode JPEG → BGR numpy. cv2.imdecode runs in the calling
        # thread; the JPEG is small enough that this is cheap (~1 ms).
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._last_decoded
        self._last_frame_id = frame_id
        self._last_decoded = img
        return img

    # ──────────────────────── commands ────────────────────────

    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_deg: float
    ) -> None:
        # Spec §4.3 / §4.4: SET_POSITION_TARGET_LOCAL_NED with the
        # position bits ignored and velocity bits active is the wire
        # representation of a velocity-NED setpoint.
        yaw_rad = float(np.radians(yaw_deg))
        await asyncio.to_thread(
            self._mav.send_position_target_local_ned,
            None, None, None,              # ignore position
            float(vn), float(ve), float(vd),
            yaw_rad, None,                  # yaw, yaw_rate (ignored)
            self._target_system,
            self._target_component,
        )

    async def send_position_ned(
        self, n: float, e: float, d: float, yaw_deg: float
    ) -> None:
        raise NotImplementedError(
            "DCLSpecAdapter does not advertise POSITION_NED."
        )

    async def send_attitude(
        self,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        thrust: float,
    ) -> None:
        # MAVLinkClient takes radians.
        await asyncio.to_thread(
            self._mav.send_attitude_target,
            float(np.radians(roll_deg)),
            float(np.radians(pitch_deg)),
            float(np.radians(yaw_deg)),
            float(thrust),
            self._target_system,
            self._target_component,
        )

    # ──────────────────────── action / mode ────────────────────────
    # DCL starts the drone airborne in offboard control; these are no-ops.

    async def arm(self) -> None:
        return

    async def disarm(self) -> None:
        return

    async def takeoff(self, altitude_m: float) -> None:
        return

    async def land(self) -> None:
        return

    async def start_offboard(self, initial_mode: str = "velocity") -> None:
        return

    async def stop_offboard(self) -> None:
        return

    # ──────────────────────── introspection ────────────────────────

    def info(self) -> SimInfo:
        return SimInfo(
            backend="dcl_spec",
            capabilities=self.capabilities,
            tick_rate_hz=None,  # wallclock-paced
            notes=(
                "VADR-TS-002 §4 MAVLink (HEARTBEAT/ATTITUDE/HIGHRES_IMU/"
                "TIMESYNC ← sim; SET_ATTITUDE_TARGET / "
                "SET_POSITION_TARGET_LOCAL_NED / HEARTBEAT → sim) plus "
                "§4.6 UDP:5600 JPEG vision stream."
            ),
        )
