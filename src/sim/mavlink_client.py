"""VADR-TS-002 §4 MAVLink Client.

Thin pymavlink wrapper that implements exactly the message set the spec
calls out, so contestant code can talk to the DCL race simulator over
the UDP SITL bridge without dragging in the full mavsdk stack.

Per VADR-TS-002:

    §4.2  Transport       UDP
    §4.4  Timing          physics 120 Hz, command rate < 100 Hz,
                          minimum heartbeat 2 Hz
    §4.3  Messages
       Simulator → Client : HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC
       Client    → Sim    : SET_POSITION_TARGET_LOCAL_NED,
                            SET_ATTITUDE_TARGET, HEARTBEAT (≥2 Hz)

The client maintains the heartbeat loop on its own thread, parses
incoming telemetry into snapshot dataclasses, and exposes thread-safe
accessors for the latest state. Outbound command helpers honour the
MAV_FRAME_LOCAL_NED / MAV_FRAME_BODY_NED conventions documented in §3.8.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from pymavlink import mavutil


# ─────────────────────────── data classes ───────────────────────────

@dataclass
class AttitudeSnapshot:
    """Latest ATTITUDE telemetry (radians, rad/s) plus wallclock receipt."""
    time_boot_ms: int = 0
    roll: float = 0.0     # rad, +Y axis up
    pitch: float = 0.0    # rad
    yaw: float = 0.0      # rad, NED
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0
    received_monotonic_s: float = 0.0


@dataclass
class ImuSnapshot:
    """Latest HIGHRES_IMU telemetry. Units per MAVLink common.xml."""
    time_usec: int = 0
    xacc: float = 0.0     # m/s² body X
    yacc: float = 0.0     # m/s² body Y
    zacc: float = 0.0     # m/s² body Z
    xgyro: float = 0.0    # rad/s
    ygyro: float = 0.0
    zgyro: float = 0.0
    abs_pressure: float = 0.0
    temperature: float = 0.0
    received_monotonic_s: float = 0.0


@dataclass
class TimeSyncSnapshot:
    """Latest TIMESYNC pair (sim time + our local time at receipt)."""
    sim_time_ns: int = 0
    received_monotonic_s: float = 0.0


@dataclass
class ClientStats:
    msgs_received: int = 0
    msgs_unknown: int = 0
    heartbeats_sent: int = 0
    attitude_targets_sent: int = 0
    position_targets_sent: int = 0
    last_sim_heartbeat_s: float = 0.0


# ─────────────────────────── MAVLink helpers ───────────────────────

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX (yaw → pitch → roll) Tait-Bryan to quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


# Bits of POSITION_TARGET_TYPEMASK to ignore individual fields.
# (Per common.xml POSITION_TARGET_TYPEMASK_*_IGNORE.)
_PT_IGNORE_PX = 1 << 0
_PT_IGNORE_PY = 1 << 1
_PT_IGNORE_PZ = 1 << 2
_PT_IGNORE_VX = 1 << 3
_PT_IGNORE_VY = 1 << 4
_PT_IGNORE_VZ = 1 << 5
_PT_IGNORE_AFX = 1 << 6
_PT_IGNORE_AFY = 1 << 7
_PT_IGNORE_AFZ = 1 << 8
_PT_IGNORE_YAW = 1 << 10
_PT_IGNORE_YAW_RATE = 1 << 11
_PT_IGNORE_ACCEL = _PT_IGNORE_AFX | _PT_IGNORE_AFY | _PT_IGNORE_AFZ
_PT_IGNORE_POS = _PT_IGNORE_PX | _PT_IGNORE_PY | _PT_IGNORE_PZ
_PT_IGNORE_VEL = _PT_IGNORE_VX | _PT_IGNORE_VY | _PT_IGNORE_VZ

# ATTITUDE_TARGET_TYPEMASK bits.
_AT_IGNORE_BODY_ROLL_RATE = 1 << 0
_AT_IGNORE_BODY_PITCH_RATE = 1 << 1
_AT_IGNORE_BODY_YAW_RATE = 1 << 2
_AT_IGNORE_THROTTLE = 1 << 6
_AT_IGNORE_ATTITUDE = 1 << 7
_AT_IGNORE_BODY_RATES = (
    _AT_IGNORE_BODY_ROLL_RATE
    | _AT_IGNORE_BODY_PITCH_RATE
    | _AT_IGNORE_BODY_YAW_RATE
)


# ─────────────────────────── client ───────────────────────────

class MAVLinkClient:
    """Spec-compliant MAVLink client for VADR-TS-002.

    Typical use::

        client = MAVLinkClient("udpin://0.0.0.0:14540")
        client.start()
        client.wait_for_simulator(timeout_s=10.0)
        client.send_attitude_target(roll=0.0, pitch=0.05, yaw=1.57, thrust=0.55)
        att = client.latest_attitude()
        client.stop()

    Connection-string examples (pymavlink-style):
      * ``udpin://0.0.0.0:14540`` — listen for the sim to come to us
      * ``udpout://127.0.0.1:14540`` — push to a sim already listening
    """

    HEARTBEAT_HZ_DEFAULT = 2.0  # spec §4.4 minimum

    def __init__(
        self,
        connection_string: str = "udpin://0.0.0.0:14540",
        source_system: int = 245,
        source_component: int = 190,  # MAV_COMP_ID_ONBOARD_COMPUTER
        heartbeat_hz: float = HEARTBEAT_HZ_DEFAULT,
    ):
        self._conn_str = self._translate_conn(connection_string)
        self._source_system = source_system
        self._source_component = source_component
        self._heartbeat_period_s = 1.0 / max(heartbeat_hz, 0.5)

        self._mav: Optional[mavutil.mavfile] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._lock = threading.Lock()
        self._attitude = AttitudeSnapshot()
        self._imu = ImuSnapshot()
        self._timesync = TimeSyncSnapshot()
        self._got_sim_heartbeat = threading.Event()
        self.stats = ClientStats()

    # ─────────────── lifecycle ───────────────

    def start(self):
        if self._mav is not None:
            return
        self._mav = mavutil.mavlink_connection(
            self._conn_str,
            source_system=self._source_system,
            source_component=self._source_component,
            dialect="common",
            input=False,  # not stdin
        )
        self._stop_flag.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="MAVLinkClient.rx", daemon=True
        )
        self._hb_thread = threading.Thread(
            target=self._hb_loop, name="MAVLinkClient.hb", daemon=True
        )
        self._rx_thread.start()
        self._hb_thread.start()

    def stop(self):
        self._stop_flag.set()
        for t in (self._rx_thread, self._hb_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._rx_thread = None
        self._hb_thread = None
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def wait_for_simulator(self, timeout_s: float = 10.0) -> bool:
        """Block until at least one sim HEARTBEAT has been received."""
        return self._got_sim_heartbeat.wait(timeout=timeout_s)

    # ─────────────── telemetry accessors ───────────────

    def latest_attitude(self) -> AttitudeSnapshot:
        with self._lock:
            return AttitudeSnapshot(**self._attitude.__dict__)

    def latest_imu(self) -> ImuSnapshot:
        with self._lock:
            return ImuSnapshot(**self._imu.__dict__)

    def latest_timesync(self) -> TimeSyncSnapshot:
        with self._lock:
            return TimeSyncSnapshot(**self._timesync.__dict__)

    # ─────────────── command senders ───────────────

    def send_attitude_target(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        thrust: float = 0.5,
        target_system: int = 1,
        target_component: int = 1,
    ):
        """SET_ATTITUDE_TARGET with body rates ignored, attitude + thrust active."""
        if self._mav is None:
            raise RuntimeError("start() the client first")
        q = _euler_to_quat(roll, pitch, yaw)
        self._mav.mav.set_attitude_target_send(
            int((time.monotonic() * 1000) % 0xFFFFFFFF),  # time_boot_ms
            target_system,
            target_component,
            _AT_IGNORE_BODY_RATES,  # type_mask
            q,
            0.0, 0.0, 0.0,         # body rates (ignored)
            float(max(0.0, min(1.0, thrust))),
        )
        self.stats.attitude_targets_sent += 1

    def send_position_target_local_ned(
        self,
        x_n: Optional[float] = None,
        y_e: Optional[float] = None,
        z_d: Optional[float] = None,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
        vz: Optional[float] = None,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        target_system: int = 1,
        target_component: int = 1,
    ):
        """SET_POSITION_TARGET_LOCAL_NED in MAV_FRAME_LOCAL_NED (§3.8)."""
        if self._mav is None:
            raise RuntimeError("start() the client first")

        mask = _PT_IGNORE_ACCEL
        # Build mask: any None field is ignored by the autopilot.
        if x_n is None: mask |= _PT_IGNORE_PX
        if y_e is None: mask |= _PT_IGNORE_PY
        if z_d is None: mask |= _PT_IGNORE_PZ
        if vx is None: mask |= _PT_IGNORE_VX
        if vy is None: mask |= _PT_IGNORE_VY
        if vz is None: mask |= _PT_IGNORE_VZ
        if yaw is None: mask |= _PT_IGNORE_YAW
        if yaw_rate is None: mask |= _PT_IGNORE_YAW_RATE

        self._mav.mav.set_position_target_local_ned_send(
            int((time.monotonic() * 1000) % 0xFFFFFFFF),  # time_boot_ms
            target_system,
            target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            float(x_n or 0.0), float(y_e or 0.0), float(z_d or 0.0),
            float(vx or 0.0),  float(vy or 0.0),  float(vz or 0.0),
            0.0, 0.0, 0.0,                       # accel (ignored)
            float(yaw or 0.0), float(yaw_rate or 0.0),
        )
        self.stats.position_targets_sent += 1

    # ─────────────── internal loops ───────────────

    def _hb_loop(self):
        assert self._mav is not None
        while not self._stop_flag.is_set():
            try:
                self._mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,  # base_mode
                    0,  # custom_mode
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                self.stats.heartbeats_sent += 1
            except Exception:
                # Don't kill the loop on transient socket errors.
                pass
            self._stop_flag.wait(self._heartbeat_period_s)

    def _rx_loop(self):
        assert self._mav is not None
        while not self._stop_flag.is_set():
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.25)
            except Exception:
                continue
            if msg is None:
                continue
            self.stats.msgs_received += 1
            t_now = time.monotonic()
            mtype = msg.get_type()

            if mtype == "HEARTBEAT":
                self.stats.last_sim_heartbeat_s = t_now
                self._got_sim_heartbeat.set()

            elif mtype == "ATTITUDE":
                with self._lock:
                    self._attitude = AttitudeSnapshot(
                        time_boot_ms=msg.time_boot_ms,
                        roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw,
                        rollspeed=msg.rollspeed,
                        pitchspeed=msg.pitchspeed,
                        yawspeed=msg.yawspeed,
                        received_monotonic_s=t_now,
                    )

            elif mtype == "HIGHRES_IMU":
                with self._lock:
                    self._imu = ImuSnapshot(
                        time_usec=msg.time_usec,
                        xacc=msg.xacc, yacc=msg.yacc, zacc=msg.zacc,
                        xgyro=msg.xgyro, ygyro=msg.ygyro, zgyro=msg.zgyro,
                        abs_pressure=msg.abs_pressure,
                        temperature=msg.temperature,
                        received_monotonic_s=t_now,
                    )

            elif mtype == "TIMESYNC":
                # If tc1 == 0 the peer is asking; echo with our monotonic ns.
                # Otherwise it's a reply / unsolicited timestamp.
                try:
                    if msg.tc1 == 0:
                        self._mav.mav.timesync_send(time.monotonic_ns(), msg.ts1)
                    with self._lock:
                        self._timesync = TimeSyncSnapshot(
                            sim_time_ns=int(msg.tc1) if msg.tc1 != 0 else int(msg.ts1),
                            received_monotonic_s=t_now,
                        )
                except Exception:
                    pass

            else:
                self.stats.msgs_unknown += 1

    # ─────────────── helpers ───────────────

    @staticmethod
    def _translate_conn(conn: str) -> str:
        """Accept mavsdk-style ``udpin://host:port`` and translate to
        pymavlink's ``udpin:host:port`` form."""
        for prefix, replacement in (
            ("udpin://", "udpin:"),
            ("udpout://", "udpout:"),
            ("udp://", "udpin:"),  # mavsdk's ``udp://`` listens by default
            ("tcp://", "tcp:"),
        ):
            if conn.startswith(prefix):
                return replacement + conn[len(prefix):]
        return conn
