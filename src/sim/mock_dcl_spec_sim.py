"""Mock VADR-TS-002 Simulator — Mac-side integration target.

Promoted from the /tmp test harness so contestants can run the full
race stack against a spec-compliant simulator without DCL.

What it does:
  * Speaks VADR-TS-002 §4 MAVLink to the client:
      - HEARTBEAT @ 2 Hz   (MAV_TYPE_QUADROTOR / MAV_AUTOPILOT_GENERIC)
      - ATTITUDE @ 50 Hz   (driven by simple kinematic model)
      - HIGHRES_IMU @ 100 Hz (gravity-loaded body-frame accel + gyro)
      - TIMESYNC @ 1 Hz    (sim-time-ns echoes per spec)
  * Streams JPEG frames per §4.6 on UDP:5600 @ 30 Hz
      - Default: synthetic color-bar frames (no gates rendered)
      - Custom: pass a `frame_renderer` callable for actual gates
  * Receives the contestant's commands and applies them to a kinematic
    drone:
      - SET_ATTITUDE_TARGET    → drives roll/pitch/yaw/thrust
      - SET_POSITION_TARGET_LOCAL_NED → drives velocity (vx/vy/vz)
    Position is integrated from velocity each tick.

What it does NOT do:
  * Render gates with the same geometry as the real DCL sim
    (we don't have the DCL course definitions yet)
  * Simulate aerodynamics, drag, or motor dynamics
  * Detect gate-pass events (the contestant's race loop does that)

Usage as a standalone process::

    # Terminal 1: start the mock sim
    python -m sim.mock_dcl_spec_sim

    # Terminal 2: run the race stack against it
    python run_race.py --backend dcl_spec --connection udpin://0.0.0.0:14540

Usage as a library in a test::

    sim = MockDCLSpecSim()
    sim.start()
    try:
        adapter = make_adapter("dcl_spec",
                               mavlink_conn="udpin://0.0.0.0:14540")
        # ... drive the adapter ...
    finally:
        sim.stop()
"""
from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from pymavlink import mavutil

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from .stream_sender import _synthetic_frame  # type: ignore
    from .stream_receiver import build_packets    # type: ignore
except Exception:
    # When run as a script, the package-relative imports fail. Fall
    # back to absolute imports — `vision/` is a sibling package.
    import sys
    from pathlib import Path
    _SRC = str(Path(__file__).resolve().parent.parent)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from vision.stream_sender import _synthetic_frame
    from vision.stream_receiver import build_packets


# ─────────────────────────── kinematic drone ───────────────────────────

@dataclass
class _DroneState:
    """Tiny kinematic model. NED frame, attitude in radians."""
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -2.0]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0       # NED yaw in radians
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # Latest setpoints
    cmd_mode: str = "idle"  # "attitude" | "velocity" | "idle"
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_yaw: float = 0.0
    target_thrust: float = 0.5
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def step(self, dt: float):
        """Crude first-order tracking of the latest command."""
        if self.cmd_mode == "velocity":
            # Pin velocity to setpoint with first-order lag.
            tau = 0.15
            alpha = min(1.0, dt / tau)
            self.vel = (1 - alpha) * self.vel + alpha * self.target_vel
        elif self.cmd_mode == "attitude":
            # First-order attitude lag plus a crude thrust-to-vertical-vel mapping.
            tau = 0.10
            alpha = min(1.0, dt / tau)
            self.roll = (1 - alpha) * self.roll + alpha * self.target_roll
            self.pitch = (1 - alpha) * self.pitch + alpha * self.target_pitch
            # Yaw uses shortest-angle wrap.
            yaw_err = ((self.target_yaw - self.yaw + math.pi)
                       % (2 * math.pi)) - math.pi
            self.yaw += alpha * yaw_err
            # Thrust above hover ⇒ climb (vd negative).
            self.vel[2] = (self.target_thrust - 0.5) * -8.0
            # Tilted attitude ⇒ horizontal accel in body frame.
            ax_body = math.tan(self.pitch) * 9.81
            ay_body = math.tan(self.roll) * 9.81
            cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
            self.vel[0] = (1 - alpha) * self.vel[0] + alpha * (
                cos_y * ax_body - sin_y * ay_body
            )
            self.vel[1] = (1 - alpha) * self.vel[1] + alpha * (
                sin_y * ax_body + cos_y * ay_body
            )
        # Always update yaw towards the target (so attitude messages stay sane
        # even in idle/velocity modes).
        if self.cmd_mode == "velocity" and abs(self.target_yaw) > 1e-6:
            yaw_err = ((self.target_yaw - self.yaw + math.pi)
                       % (2 * math.pi)) - math.pi
            self.yaw += min(1.0, dt / 0.20) * yaw_err

        # Integrate position.
        self.pos = self.pos + self.vel * dt
        # Don't let the kinematic drone fall through the ground.
        if self.pos[2] > 0.0:
            self.pos[2] = 0.0
            self.vel[2] = min(self.vel[2], 0.0)


# ─────────────────────────── mock simulator ───────────────────────────

# Type for the optional frame renderer hook.
# Args: drone_state, frame_index, image_size_wh
# Returns: BGR numpy image at the requested size.
FrameRenderer = Callable[[_DroneState, int, tuple], np.ndarray]


class MockDCLSpecSim:
    """Spec-compliant mock simulator. Run `start()` then `stop()`."""

    HEARTBEAT_HZ = 2.0
    ATTITUDE_HZ = 50.0
    IMU_HZ = 100.0
    TIMESYNC_HZ = 1.0
    VIDEO_HZ = 30.0
    PHYSICS_HZ = 120.0  # spec §4.4

    IMG_W = 640
    IMG_H = 360

    def __init__(
        self,
        mavlink_port: int = 14540,
        mavlink_host: str = "127.0.0.1",
        vision_port: int = 5600,
        vision_host: str = "127.0.0.1",
        jpeg_quality: int = 80,
        frame_renderer: Optional[FrameRenderer] = None,
        verbose: bool = True,
    ):
        if not HAS_CV2:
            raise RuntimeError("cv2 required for MockDCLSpecSim")
        self._mav_port = mavlink_port
        self._mav_host = mavlink_host
        self._vid_port = vision_port
        self._vid_host = vision_host
        self._jpeg_q = jpeg_quality
        self._renderer = frame_renderer or self._default_renderer
        self._verbose = verbose

        self._drone = _DroneState()
        self._mav: Optional[mavutil.mavfile] = None
        self._vid_sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._t0 = 0.0
        self._frame_idx = 0
        self.cmds_received: dict = {}

    # ─────────────────────────── lifecycle ───────────────────────────

    def start(self):
        if self._mav is not None:
            return
        self._mav = mavutil.mavlink_connection(
            f"udpout:{self._mav_host}:{self._mav_port}",
            source_system=1, source_component=1, dialect="common",
        )
        self._vid_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop.clear()
        self._t0 = time.monotonic()

        for target, name in (
            (self._physics_loop, "physics"),
            (self._mavlink_tx_loop, "mavlink_tx"),
            (self._mavlink_rx_loop, "mavlink_rx"),
            (self._video_loop, "video"),
        ):
            t = threading.Thread(target=target, name=f"MockDCL.{name}", daemon=True)
            t.start()
            self._threads.append(t)

        if self._verbose:
            print(f"[mock_dcl_spec] MAVLink → udp:{self._mav_host}:{self._mav_port}")
            print(f"[mock_dcl_spec] Vision  → udp:{self._vid_host}:{self._vid_port}")
            print(f"[mock_dcl_spec] {self.IMG_W}×{self.IMG_H} @ {self.VIDEO_HZ:g} Hz")

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._vid_sock:
            try: self._vid_sock.close()
            except OSError: pass
            self._vid_sock = None
        if self._mav:
            try: self._mav.close()
            except Exception: pass
            self._mav = None
        if self._verbose:
            print(f"[mock_dcl_spec] received: {self.cmds_received}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ─────────────────────────── loops ───────────────────────────

    def _physics_loop(self):
        period = 1.0 / self.PHYSICS_HZ
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - last
            last = now
            self._drone.step(dt)
            self._stop.wait(max(0.0, period - (time.monotonic() - now)))

    def _mavlink_tx_loop(self):
        last_hb = 0.0
        last_att = 0.0
        last_imu = 0.0
        last_ts = 0.0
        hb_p = 1.0 / self.HEARTBEAT_HZ
        att_p = 1.0 / self.ATTITUDE_HZ
        imu_p = 1.0 / self.IMU_HZ
        ts_p = 1.0 / self.TIMESYNC_HZ
        while not self._stop.is_set() and self._mav is not None:
            now = time.monotonic()
            boot_ms = int((now - self._t0) * 1000) & 0xFFFFFFFF

            if now - last_hb >= hb_p:
                try:
                    self._mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_QUADROTOR,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                        0, 0, mavutil.mavlink.MAV_STATE_ACTIVE,
                    )
                except Exception:
                    pass
                last_hb = now

            if now - last_att >= att_p:
                try:
                    self._mav.mav.attitude_send(
                        boot_ms,
                        float(self._drone.roll),
                        float(self._drone.pitch),
                        float(self._drone.yaw),
                        float(self._drone.rollspeed),
                        float(self._drone.pitchspeed),
                        float(self._drone.yawspeed),
                    )
                except Exception:
                    pass
                last_att = now

            if now - last_imu >= imu_p:
                # Body-frame accel: rotate gravity into body. For a level
                # drone gravity reads ≈ (0, 0, -9.81) in body FRD.
                g_world = np.array([0.0, 0.0, 9.81])  # NED, gravity is +Z (down)
                # World→body rotation by -roll, -pitch, -yaw (ZYX).
                cy, sy = math.cos(self._drone.yaw), math.sin(self._drone.yaw)
                cp, sp = math.cos(self._drone.pitch), math.sin(self._drone.pitch)
                cr, sr = math.cos(self._drone.roll), math.sin(self._drone.roll)
                R_b_w = np.array([
                    [cy * cp,                cp * sy,               -sp     ],
                    [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
                    [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr],
                ])
                g_body = R_b_w @ g_world
                # Accelerometer reads specific force = a - g, with sign convention
                # so that a level drone at rest reads (0, 0, -g) on body z. We
                # approximate body acceleration as zero (kinematic model),
                # so accel = -g_body.
                accel_body = -g_body
                try:
                    self._mav.mav.highres_imu_send(
                        int(now * 1e6),
                        float(accel_body[0]), float(accel_body[1]),
                        float(accel_body[2]),
                        float(self._drone.rollspeed),
                        float(self._drone.pitchspeed),
                        float(self._drone.yawspeed),
                        0.0, 0.0, 0.0,        # mag
                        1013.25, 0.0, 0.0,     # pressure / diff_pressure / pressure_alt
                        25.0, 0, 0,            # temp / fields_updated / id
                    )
                except Exception:
                    pass
                last_imu = now

            if now - last_ts >= ts_p:
                try:
                    self._mav.mav.timesync_send(0, time.monotonic_ns())
                except Exception:
                    pass
                last_ts = now

            # Sleep until the next event boundary (small slice — IMU is fastest).
            self._stop.wait(0.002)

    def _mavlink_rx_loop(self):
        while not self._stop.is_set() and self._mav is not None:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.25)
            except Exception:
                continue
            if msg is None:
                continue
            mt = msg.get_type()
            self.cmds_received[mt] = self.cmds_received.get(mt, 0) + 1

            if mt == "SET_ATTITUDE_TARGET":
                q = msg.q
                # Quaternion → ZYX Euler.
                w, x, y, z = q[0], q[1], q[2], q[3]
                self._drone.target_roll = math.atan2(
                    2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
                self._drone.target_pitch = math.asin(sp)
                self._drone.target_yaw = math.atan2(
                    2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                self._drone.target_thrust = float(msg.thrust)
                self._drone.cmd_mode = "attitude"

            elif mt == "SET_POSITION_TARGET_LOCAL_NED":
                # Per spec, ignore-position bits are typically set when sending
                # velocity. We just take the velocity fields.
                self._drone.target_vel = np.array(
                    [float(msg.vx), float(msg.vy), float(msg.vz)]
                )
                self._drone.target_yaw = float(msg.yaw)
                self._drone.cmd_mode = "velocity"

    def _video_loop(self):
        period = 1.0 / self.VIDEO_HZ
        next_t = time.monotonic()
        while not self._stop.is_set() and self._vid_sock is not None:
            now = time.monotonic()
            if now < next_t:
                self._stop.wait(next_t - now)
                continue
            next_t += period
            img = self._renderer(self._drone, self._frame_idx,
                                 (self.IMG_W, self.IMG_H))
            ok, buf = cv2.imencode(
                ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q]
            )
            if not ok:
                continue
            packets = build_packets(
                frame_id=self._frame_idx & 0xFFFFFFFF,
                jpeg_bytes=buf.tobytes(),
                sim_time_ns=int(now * 1e9),
                max_payload_size=1400,
            )
            for p in packets:
                try:
                    self._vid_sock.sendto(p, (self._vid_host, self._vid_port))
                except OSError:
                    break
            self._frame_idx += 1

    # ─────────────────────────── default renderer ───────────────────────────

    def _default_renderer(self, drone: _DroneState, idx: int,
                          size: tuple) -> np.ndarray:
        """Color-bar frame with a HUD. No gates rendered — for protocol
        smoke testing only. Pass a `frame_renderer` to inject your own
        gate-rendering logic (e.g. the synthetic data generator)."""
        w, h = size
        img = _synthetic_frame(idx, w, h)
        hud = (
            f"pos=({drone.pos[0]:+.1f},{drone.pos[1]:+.1f},{drone.pos[2]:+.1f}) "
            f"yaw={math.degrees(drone.yaw):+.0f}deg "
            f"mode={drone.cmd_mode}"
        )
        cv2.putText(img, hud, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img


# ─────────────────────────── CLI ───────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="VADR-TS-002 mock simulator (no gate rendering).")
    p.add_argument("--mavlink-port", type=int, default=14540)
    p.add_argument("--mavlink-host", default="127.0.0.1")
    p.add_argument("--vision-port", type=int, default=5600)
    p.add_argument("--vision-host", default="127.0.0.1")
    p.add_argument("--quality", type=int, default=80)
    args = p.parse_args()

    sim = MockDCLSpecSim(
        mavlink_port=args.mavlink_port, mavlink_host=args.mavlink_host,
        vision_port=args.vision_port, vision_host=args.vision_host,
        jpeg_quality=args.quality,
    )
    sim.start()
    try:
        print("[mock_dcl_spec] running. Ctrl-C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
