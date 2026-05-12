"""Drone-eye gate renderer for MockDCLSpecSim.

Builds a MuJoCo scene from a list of NED gate positions and renders
the camera view for the drone's current pose. Spec-compliant intrinsics
(VADR-TS-002 §3.8) and the 20° upward camera tilt are applied so the
output matches what the YOLO model is trained on.

This is intentionally a thin renderer — no domain randomization, no
multi-gate decoys, no label generation. The synthetic data generator
(`vision/gate_yolo/generate_data_keypoints.py`) does that for training;
this one only needs to produce the *frame the drone would see right now*
during a race-loop tick.

Coordinate frames:
    NED      — North (X+), East (Y+), Down (Z+).  Used by everything
               in the race stack.
    World    — MuJoCo's default world frame: X+, Y+, Z up.
               Conversion: world.xyz = (n, e, -d).
    Camera   — Body-FRD with the spec's 20° upward tilt applied last.
               Body forward = (cos(yaw), sin(yaw), 0) in world.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# VADR-TS-002 §3.7 gate dimensions.
_OUTER_HALF = 1.35   # outer frame half-extent (2.7 m total)
_INNER_HALF = 0.75   # inner opening half-extent (1.5 m total)
_RIM_DEPTH_HALF = 0.13  # gate depth: 0.26 m → half = 0.13
_RIM_THICKNESS_HALF = (_OUTER_HALF - _INNER_HALF) / 2.0  # 0.30 m

# VADR-TS-002 §3.8 camera.
_FX = 320.0
_FY = 320.0
_CX = 320.0
_CY = 180.0
_IMG_W = 640
_IMG_H = 360
_VFOV_DEG = 2 * math.degrees(math.atan(_CY / _FY))  # ≈ 58.36°
_TILT_RAD = math.radians(20.0)

# Gate rim colors — cycle through so adjacent gates look different and
# YOLO has to depend on geometry, not color, like in training data.
_GATE_COLORS = [
    "1.0 0.5 0.0",   # orange
    "0.0 0.8 0.2",   # green
    "0.2 0.3 1.0",   # blue
    "0.9 0.9 0.0",   # yellow
    "1.0 0.2 0.7",   # magenta
    "0.0 0.9 0.9",   # cyan
]


def _gate_xml(idx: int, n: float, e: float, d: float, yaw_deg: float) -> str:
    """Build the MuJoCo `<body>` for one gate. Hollow rim around a 1.5 m opening.

    NED → world: x=n, y=e, z=-d (so altitude becomes positive Z up).
    """
    color = _GATE_COLORS[idx % len(_GATE_COLORS)]
    x, y, z = float(n), float(e), float(-d)
    return f"""
        <body name="gate_{idx}" pos="{x} {y} {z}" euler="0 0 {yaw_deg}">
          <!-- Right post -->
          <geom type="box" pos=" {_INNER_HALF + _RIM_THICKNESS_HALF} 0 0"
                size="{_RIM_THICKNESS_HALF} {_RIM_DEPTH_HALF} {_OUTER_HALF}"
                rgba="{color} 1"/>
          <!-- Left post -->
          <geom type="box" pos="-{_INNER_HALF + _RIM_THICKNESS_HALF} 0 0"
                size="{_RIM_THICKNESS_HALF} {_RIM_DEPTH_HALF} {_OUTER_HALF}"
                rgba="{color} 1"/>
          <!-- Top bar -->
          <geom type="box" pos="0 0 {_INNER_HALF + _RIM_THICKNESS_HALF}"
                size="{_OUTER_HALF} {_RIM_DEPTH_HALF} {_RIM_THICKNESS_HALF}"
                rgba="{color} 1"/>
          <!-- Bottom bar -->
          <geom type="box" pos="0 0 -{_INNER_HALF + _RIM_THICKNESS_HALF}"
                size="{_OUTER_HALF} {_RIM_DEPTH_HALF} {_RIM_THICKNESS_HALF}"
                rgba="{color} 1"/>
        </body>"""


class CourseRenderer:
    """Renders the drone's-eye view of a course of NED-positioned gates.

    Args:
        gates_ned: Iterable of (N, E, D) tuples.
        gate_yaws_rad: Optional same-length iterable of gate yaws in
            radians (about world Z, same convention as drone yaw).
            If None, each gate is yawed to face the previous gate
            (so the drone naturally flies *through* it).
        img_w / img_h: Render size. Defaults to spec §3.8.
    """

    def __init__(
        self,
        gates_ned: Sequence[Tuple[float, float, float]],
        gate_yaws_rad: Optional[Sequence[float]] = None,
        img_w: int = _IMG_W,
        img_h: int = _IMG_H,
    ):
        if not HAS_MUJOCO:
            raise RuntimeError("CourseRenderer requires mujoco")
        if not gates_ned:
            raise ValueError("gates_ned must not be empty")
        self._gates = [tuple(map(float, g)) for g in gates_ned]
        self._img_w = img_w
        self._img_h = img_h

        if gate_yaws_rad is None:
            yaws = self._auto_orient(self._gates)
        else:
            yaws = list(gate_yaws_rad)
        self._yaws = yaws

        xml = self._build_xml()
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._renderer = mujoco.Renderer(self._model, height=img_h, width=img_w)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # ─────────────────────── geometry helpers ───────────────────────

    @staticmethod
    def _auto_orient(gates) -> list[float]:
        """Yaw each gate so its plane is perpendicular to the
        previous→current direction (the drone's natural approach).
        First gate is oriented like the second."""
        yaws = []
        n_gates = len(gates)
        for i, g in enumerate(gates):
            if i == 0 and n_gates > 1:
                ref_prev = (0.0, 0.0, g[2])  # treat origin as virtual "prev gate"
                ref_next = gates[1]
                d_n = ref_next[0] - g[0]
                d_e = ref_next[1] - g[1]
                # Use the bisector of (origin → g) and (g → next).
                a_n = g[0] - ref_prev[0]
                a_e = g[1] - ref_prev[1]
                yaw = math.atan2((a_e + d_e) / 2.0, (a_n + d_n) / 2.0)
            elif i == n_gates - 1:
                # Last gate: face the previous-to-current direction.
                prev = gates[i - 1]
                yaw = math.atan2(g[1] - prev[1], g[0] - prev[0])
            else:
                # Bisector of (prev→curr) and (curr→next).
                prev = gates[i - 1]
                nxt = gates[i + 1]
                a_n = g[0] - prev[0]; a_e = g[1] - prev[1]
                d_n = nxt[0] - g[0];  d_e = nxt[1] - g[1]
                yaw = math.atan2((a_e + d_e) / 2.0, (a_n + d_n) / 2.0)
            # Convert "facing direction" to "gate plane normal yaw".
            # The gate's local Y axis is the axis the drone passes through;
            # in the XML we made the rim sit in the X-Z plane, so euler yaw
            # rotates it about Z. yaw=0 → drone passes along +Y direction.
            # We want the drone passing along its approach vector, so yaw
            # equals the approach-direction yaw + 90° (so +Y becomes that
            # direction).
            yaws.append(yaw - math.pi / 2.0)
        return yaws

    def _build_xml(self) -> str:
        bodies = "".join(
            _gate_xml(i, n, e, d, math.degrees(self._yaws[i]))
            for i, (n, e, d) in enumerate(self._gates)
        )
        return f"""
        <mujoco model="dcl_course">
          <option gravity="0 0 -9.81"/>
          <visual>
            <global offwidth="{self._img_w}" offheight="{self._img_h}"
                    fovy="{_VFOV_DEG:.4f}"/>
          </visual>
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.30 0.32 0.35" rgb2="0.22 0.24 0.27"/>
            <material name="floor_mat" texture="grid" texrepeat="20 20" reflectance="0.05"/>
            <texture name="sky" type="skybox" builtin="gradient"
                     rgb1="0.55 0.65 0.85" rgb2="0.85 0.90 0.95"
                     width="256" height="256"/>
          </asset>
          <worldbody>
            <light pos="20 20 30" dir="-1 -1 -2" diffuse="0.95 0.95 0.95"
                   specular="0.2 0.2 0.2" castshadow="true"/>
            <light pos="-15 0 25" dir="1 0 -1" diffuse="0.5 0.5 0.55"
                   castshadow="false"/>
            <geom type="plane" size="200 200 0.1" material="floor_mat"/>
            {bodies}
          </worldbody>
        </mujoco>
        """

    # ─────────────────────── camera math ───────────────────────

    @staticmethod
    def _drone_pose_to_cam(
        pos_ned: Sequence[float],
        yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert (drone NED pos, yaw) to (cam_pos, cam_forward, cam_up)
        in MuJoCo world frame, with the spec's 20° upward camera tilt
        applied."""
        n, e, d = float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])
        cam_pos = np.array([n, e, -d])  # NED → MuJoCo (Z up)

        body_forward = np.array(
            [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
        )
        world_up = np.array([0.0, 0.0, 1.0])
        body_right = np.cross(body_forward, world_up)
        body_right /= max(np.linalg.norm(body_right), 1e-9)

        # Rodrigues rotation: rotate body_forward about body_right by +tilt.
        c, s = math.cos(_TILT_RAD), math.sin(_TILT_RAD)
        cam_forward = (
            body_forward * c
            + np.cross(body_right, body_forward) * s
            + body_right * float(np.dot(body_right, body_forward)) * (1 - c)
        )
        cam_forward /= np.linalg.norm(cam_forward)
        cam_up = np.cross(body_right, cam_forward)
        cam_up /= np.linalg.norm(cam_up)
        return cam_pos, cam_forward, cam_up

    # ─────────────────────── render ───────────────────────

    def render(
        self,
        drone_pos_ned: Sequence[float],
        drone_yaw_rad: float,
    ) -> np.ndarray:
        """Return the BGR image the drone's camera would see."""
        cam_pos, cam_forward, _cam_up = self._drone_pose_to_cam(
            drone_pos_ned, drone_yaw_rad
        )
        target = cam_pos + cam_forward * 5.0
        self._cam.lookat[:] = target
        self._cam.distance = float(np.linalg.norm(target - cam_pos))
        self._cam.azimuth = float(
            math.degrees(math.atan2(cam_forward[1], cam_forward[0]))
        )
        self._cam.elevation = float(
            math.degrees(math.asin(max(-1.0, min(1.0, cam_forward[2]))))
        )
        self._renderer.update_scene(self._data, self._cam)
        rgb = self._renderer.render()  # (H, W, 3) RGB uint8
        # Convert to BGR for cv2 / downstream JPEG encoding.
        return rgb[..., ::-1].copy()
