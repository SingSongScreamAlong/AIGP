"""
Phase 2: Generate synthetic data with KEYPOINT labels for YOLOv8-pose.
======================================================================
Extends the Phase 1 data generator to output 4 gate corner keypoints
alongside bounding boxes. This is the foundation for PnP pose estimation.

YOLOv8-pose label format (per line):
    class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v kp4_x kp4_y kp4_v

Where:
    - cx, cy, w, h = normalized bounding box (same as detection)
    - kpN_x, kpN_y = normalized keypoint pixel coordinates
    - kpN_v = visibility (0=not labeled, 1=labeled but occluded, 2=labeled and visible)

Corner ordering (consistent for PnP):
    kp1 = top-left      (-x, +y in gate frame)
    kp2 = top-right     (+x, +y in gate frame)
    kp3 = bottom-right  (+x, -y in gate frame)
    kp4 = bottom-left   (-x, -y in gate frame)

This is the OPENING corners (0.2m half-size), not the outer frame.

Prerequisites:
    pip install mujoco numpy opencv-python pyyaml

Usage:
    python generate_data_keypoints.py --num_samples 500 --output datasets/gates_keypoints
    python generate_data_keypoints.py --num_samples 2000 --output datasets/gates_kp_large

Author: Conrad Weeden
Date: April 2026
"""

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("WARNING: mujoco not installed. Install with: pip install mujoco")


# ============================================================
# Gate Geometry Constants
# ============================================================
# Gate opening half-size (the hole the drone flies through) — VADR-TS-002 §3.7
GATE_HALF_SIZE = 0.75  # 1.5m x 1.5m inner opening per spec

# Gate outer frame half-size — VADR-TS-002 §3.7
FRAME_HALF_SIZE = 1.35  # 2.7m x 2.7m outer frame per spec

# Gate depth half-size — VADR-TS-002 §3.7 (depth 260mm)
GATE_DEPTH_HALF = 0.13

# 4 corners of the gate OPENING in the gate's local coordinate frame
# Order: top-left, top-right, bottom-right, bottom-left (clockwise from top-left)
# In gate local frame: X=right, Y=up, Z=through gate
GATE_CORNERS_LOCAL = np.array([
    [-GATE_HALF_SIZE,  GATE_HALF_SIZE, 0.0],  # top-left
    [ GATE_HALF_SIZE,  GATE_HALF_SIZE, 0.0],  # top-right
    [ GATE_HALF_SIZE, -GATE_HALF_SIZE, 0.0],  # bottom-right
    [-GATE_HALF_SIZE, -GATE_HALF_SIZE, 0.0],  # bottom-left
], dtype=np.float64)

# 4 corners of the gate OUTER FRAME (for bounding box computation)
GATE_FRAME_CORNERS_LOCAL = np.array([
    [-FRAME_HALF_SIZE,  FRAME_HALF_SIZE, 0.0],
    [ FRAME_HALF_SIZE,  FRAME_HALF_SIZE, 0.0],
    [ FRAME_HALF_SIZE, -FRAME_HALF_SIZE, 0.0],
    [-FRAME_HALF_SIZE, -FRAME_HALF_SIZE, 0.0],
], dtype=np.float64)

# Default image dimensions — VADR-TS-002 §3.8 / §4.6
IMG_WIDTH = 640
IMG_HEIGHT = 360

# Default pinhole intrinsics — VADR-TS-002 §3.8
DEFAULT_FX = 320.0
DEFAULT_FY = 320.0
DEFAULT_CX = 320.0
DEFAULT_CY = 180.0

# Body-to-camera tilt — VADR-TS-002 §3.8 (camera tilted 20° upward from body)
CAMERA_TILT_DEG = 20.0


# ============================================================
# 3D → 2D Projection (Pinhole Camera Model)
# ============================================================
def project_points_to_image(points_3d, cam_pos, cam_forward, cam_up,
                             fov_deg=90.0, img_w=IMG_WIDTH, img_h=IMG_HEIGHT,
                             fx=None, fy=None, cx=None, cy=None):
    """
    Project 3D world points to 2D pixel coordinates using a pinhole camera model.
    Uses explicit fx/fy/cx/cy if provided (spec-exact), otherwise derives from fov.

    Args:
        points_3d: (N, 3) array of 3D points in world frame
        cam_pos: (3,) camera position in world frame
        cam_forward: (3,) camera forward direction (unit vector)
        cam_up: (3,) camera up direction (unit vector)
        fov_deg: vertical field of view in degrees
        img_w, img_h: image dimensions in pixels

    Returns:
        pixels: (N, 2) array of pixel coordinates (x, y)
        in_front: (N,) boolean array — True if point is in front of camera
    """
    # Build camera coordinate frame
    cam_forward = cam_forward / np.linalg.norm(cam_forward)
    cam_right = np.cross(cam_forward, cam_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up_ortho = np.cross(cam_right, cam_forward)

    # Transform points to camera frame
    rel = points_3d - cam_pos  # (N, 3)
    x_cam = rel @ cam_right      # right in camera
    y_cam = rel @ cam_up_ortho   # up in camera
    z_cam = rel @ cam_forward    # forward (depth)

    # Intrinsics: use explicit values if provided, else derive from VFoV
    if fx is None or fy is None:
        _fy = (img_h / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        _fx = _fy  # square pixels
    else:
        _fx, _fy = fx, fy
    _cx = cx if cx is not None else img_w / 2.0
    _cy = cy if cy is not None else img_h / 2.0

    # Project (pinhole)
    in_front = z_cam > 0.01  # avoid division by zero
    z_safe = np.where(in_front, z_cam, 1.0)

    px = (_fx * x_cam / z_safe) + _cx
    py = _cy - (_fy * y_cam / z_safe)  # flip Y (image Y goes down)

    pixels = np.stack([px, py], axis=-1)
    return pixels, in_front


def transform_corners_to_world(corners_local, gate_pos, gate_yaw):
    """
    Transform gate corners from local frame to world frame.

    Args:
        corners_local: (N, 3) corners in gate local frame
        gate_pos: (3,) gate position in world frame
        gate_yaw: yaw angle in radians

    Returns:
        corners_world: (N, 3) corners in world frame
    """
    cos_y = np.cos(gate_yaw)
    sin_y = np.sin(gate_yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])
    return (R @ corners_local.T).T + gate_pos


# ============================================================
# Keypoint Label Generation
# ============================================================
def corners_to_yolo_keypoints(opening_pixels, opening_visible, frame_pixels, frame_visible,
                                img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    """
    Convert projected gate corners to YOLOv8-pose format label.

    Args:
        opening_pixels: (4, 2) pixel coords of opening corners (for keypoints)
        opening_visible: (4,) boolean — is each opening corner visible
        frame_pixels: (4, 2) pixel coords of outer frame corners (for bbox)
        frame_visible: (4,) boolean — is each frame corner visible
        img_w, img_h: image dimensions

    Returns:
        label_str: YOLOv8-pose format string, or None if gate not visible enough
    """
    # Need at least 2 opening corners visible for useful keypoints
    n_visible = np.sum(opening_visible)
    if n_visible < 2:
        return None

    # Need at least 2 frame corners visible for a reasonable bbox
    if np.sum(frame_visible) < 2:
        return None

    # Compute bounding box from ALL visible frame corners (and opening corners)
    all_pixels = np.vstack([
        frame_pixels[frame_visible],
        opening_pixels[opening_visible]
    ])

    x_min = np.clip(np.min(all_pixels[:, 0]), 0, img_w)
    x_max = np.clip(np.max(all_pixels[:, 0]), 0, img_w)
    y_min = np.clip(np.min(all_pixels[:, 1]), 0, img_h)
    y_max = np.clip(np.max(all_pixels[:, 1]), 0, img_h)

    # Skip tiny boxes
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 10 or box_h < 10:
        return None

    # Normalized bbox (YOLO format)
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    w = box_w / img_w
    h = box_h / img_h

    # Clamp to valid range
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)

    # Build keypoint string: kp_x kp_y visibility for each of 4 corners
    kp_parts = []
    for i in range(4):
        if opening_visible[i]:
            kp_x = np.clip(opening_pixels[i, 0] / img_w, 0, 1)
            kp_y = np.clip(opening_pixels[i, 1] / img_h, 0, 1)
            # Check if keypoint is within image bounds
            in_image = (0 <= opening_pixels[i, 0] <= img_w and
                       0 <= opening_pixels[i, 1] <= img_h)
            vis = 2 if in_image else 1  # 2=visible, 1=occluded/outside
            kp_parts.extend([f"{kp_x:.6f}", f"{kp_y:.6f}", str(vis)])
        else:
            kp_parts.extend(["0.000000", "0.000000", "0"])

    # class_id cx cy w h kp1_x kp1_y kp1_v ... kp4_x kp4_y kp4_v
    label = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kp_parts)
    return label


# ============================================================
# Synthetic Data Generator (Standalone — no sim dependency)
# ============================================================
class SyntheticKeypointGenerator:
    """
    Generates synthetic training images with gate corner keypoint labels.

    Uses MuJoCo for 3D rendering with domain randomization.
    Falls back to a minimal scene if the full LSY sim isn't available.
    """

    def __init__(self, sim_xml_path=None, img_w=IMG_WIDTH, img_h=IMG_HEIGHT, fov=90.0,
                 fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY,
                 camera_tilt_deg=CAMERA_TILT_DEG):
        self.img_w = img_w
        self.img_h = img_h
        self.fov = fov
        # VADR-TS-002 §3.8 pinhole intrinsics
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        # VADR-TS-002 §3.8 body→camera tilt (radians)
        self.camera_tilt_rad = np.radians(camera_tilt_deg)

        if not HAS_MUJOCO:
            raise RuntimeError("MuJoCo is required. Install with: pip install mujoco")

        # Try to load the full sim, fall back to standalone
        if sim_xml_path and os.path.exists(sim_xml_path):
            self._setup_from_sim(sim_xml_path)
        else:
            self._setup_standalone()

    def _setup_standalone(self):
        """Create a minimal MuJoCo scene with gates for rendering."""
        # Minimal XML with a gate and ground plane
        xml = """
        <mujoco model="gate_scene">
          <option gravity="0 0 -9.81"/>
          <visual>
            <!-- VADR-TS-002 §3.8: VFoV = 2*atan(cy/fy) = 2*atan(180/320) ≈ 58.36° -->
            <global offwidth="{w}" offheight="{h}" fovy="58.36"/>
          </visual>

          <worldbody>
            <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
            <light pos="5 5 8" dir="-0.5 -0.5 -1" diffuse="0.6 0.6 0.6"/>

            <!-- Ground -->
            <geom type="plane" size="30 30 0.1" rgba="0.3 0.3 0.3 1"/>

            <!-- Gate 1 — VADR-TS-002 §3.7: outer 2.7m, inner 1.5m opening, depth 0.26m
                 Hollow rim: 4 bars (each 0.6m thick) framing the 1.5m hole -->
            <body name="gate1" pos="4.0 0.0 1.5">
              <geom type="box" pos=" 1.05 0  0   " size="0.3  0.13 1.35" rgba="1 0.5 0 1"/>
              <geom type="box" pos="-1.05 0  0   " size="0.3  0.13 1.35" rgba="1 0.5 0 1"/>
              <geom type="box" pos=" 0    0  1.05" size="0.75 0.13 0.3 " rgba="1 0.5 0 1"/>
              <geom type="box" pos=" 0    0 -1.05" size="0.75 0.13 0.3 " rgba="1 0.5 0 1"/>
            </body>

            <!-- Gate 2 -->
            <body name="gate2" pos="-2.0 6.0 2.0" euler="0 0 45">
              <geom type="box" pos=" 1.05 0  0   " size="0.3  0.13 1.35" rgba="0 0.8 0.2 1"/>
              <geom type="box" pos="-1.05 0  0   " size="0.3  0.13 1.35" rgba="0 0.8 0.2 1"/>
              <geom type="box" pos=" 0    0  1.05" size="0.75 0.13 0.3 " rgba="0 0.8 0.2 1"/>
              <geom type="box" pos=" 0    0 -1.05" size="0.75 0.13 0.3 " rgba="0 0.8 0.2 1"/>
            </body>

            <!-- Gate 3 -->
            <body name="gate3" pos="8.0 -4.0 2.5" euler="0 0 -30">
              <geom type="box" pos=" 1.05 0  0   " size="0.3  0.13 1.35" rgba="0.2 0.3 1 1"/>
              <geom type="box" pos="-1.05 0  0   " size="0.3  0.13 1.35" rgba="0.2 0.3 1 1"/>
              <geom type="box" pos=" 0    0  1.05" size="0.75 0.13 0.3 " rgba="0.2 0.3 1 1"/>
              <geom type="box" pos=" 0    0 -1.05" size="0.75 0.13 0.3 " rgba="0.2 0.3 1 1"/>
            </body>

            <!-- Gate 4 -->
            <body name="gate4" pos="0.0 -6.0 1.5" euler="0 0 90">
              <geom type="box" pos=" 1.05 0  0   " size="0.3  0.13 1.35" rgba="0.9 0.9 0 1"/>
              <geom type="box" pos="-1.05 0  0   " size="0.3  0.13 1.35" rgba="0.9 0.9 0 1"/>
              <geom type="box" pos=" 0    0  1.05" size="0.75 0.13 0.3 " rgba="0.9 0.9 0 1"/>
              <geom type="box" pos=" 0    0 -1.05" size="0.75 0.13 0.3 " rgba="0.9 0.9 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """.format(w=self.img_w, h=self.img_h)

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Gate info: (body_name, position, yaw)
        self.gates = []
        for name in ["gate1", "gate2", "gate3", "gate4"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = self.data.xpos[body_id].copy()
            # Extract yaw from rotation matrix
            rotmat = self.data.xmat[body_id].reshape(3, 3)
            yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
            self.gates.append({
                "name": name,
                "pos": pos,
                "yaw": yaw,
                "body_id": body_id,
            })

        self.renderer = mujoco.Renderer(self.model, height=self.img_h, width=self.img_w)
        print(f"Standalone scene: {len(self.gates)} gates loaded")

    def _setup_from_sim(self, xml_path):
        """Load the full LSY drone racing sim environment."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Find all gate bodies
        self.gates = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "gate" in name.lower():
                pos = self.data.xpos[i].copy()
                rotmat = self.data.xmat[i].reshape(3, 3)
                yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                self.gates.append({
                    "name": name,
                    "pos": pos,
                    "yaw": yaw,
                    "body_id": i,
                })

        self.renderer = mujoco.Renderer(self.model, height=self.img_h, width=self.img_w)
        print(f"Sim loaded from {xml_path}: {len(self.gates)} gates found")

    def _randomize_camera_pose(self):
        """
        Place camera at a random position looking at a random gate.
        Simulates what the drone camera would see during flight.
        """
        gate = random.choice(self.gates)
        gate_pos = gate["pos"]
        gate_yaw = gate["yaw"]

        # Random distance from gate (scaled for 2.7m gates per spec §3.7)
        dist = random.uniform(2.0, 15.0)

        # Random lateral and vertical offset (also scaled for larger gates)
        lateral = random.uniform(-3.0, 3.0)
        vertical = random.uniform(-1.5, 2.0)

        # Camera position: in front of gate + offsets
        # Gate faces along its local Y axis after yaw rotation
        cam_pos = np.array([
            gate_pos[0] - dist * np.cos(gate_yaw) + lateral * np.sin(gate_yaw),
            gate_pos[1] - dist * np.sin(gate_yaw) - lateral * np.cos(gate_yaw),
            gate_pos[2] + vertical,
        ])

        # Look toward the gate center (with slight random jitter)
        jitter = np.array([
            random.uniform(-0.15, 0.15),
            random.uniform(-0.15, 0.15),
            random.uniform(-0.1, 0.1),
        ])
        target = gate_pos + jitter

        # Pre-compensate body aim: point body so that AFTER 20° upward tilt,
        # the camera lands somewhere on/near the gate. This avoids gates
        # falling out of frame entirely when the tilt is applied.
        # Aim body at a point ~dist*tan(tilt) BELOW the gate (with jitter for diversity).
        dist_to_gate = np.linalg.norm(target - cam_pos)
        precompensate = dist_to_gate * np.tan(self.camera_tilt_rad)
        # Add randomized vertical offset so gate appears at varied frame heights
        vertical_aim_offset = -precompensate + random.uniform(-precompensate * 0.4,
                                                                precompensate * 0.6)
        target_aimed = target.copy()
        target_aimed[2] += vertical_aim_offset

        body_forward = target_aimed - cam_pos
        body_forward = body_forward / np.linalg.norm(body_forward)
        world_up = np.array([0.0, 0.0, 1.0])

        # Apply 20° upward camera tilt (VADR-TS-002 §3.8)
        # Rotate body_forward about the body's right axis by +tilt to pitch camera up
        body_right = np.cross(body_forward, world_up)
        body_right = body_right / max(np.linalg.norm(body_right), 1e-9)
        tilt = self.camera_tilt_rad
        # Rodrigues' rotation: rotate body_forward about body_right by +tilt
        cam_forward = (body_forward * np.cos(tilt)
                       + np.cross(body_right, body_forward) * np.sin(tilt)
                       + body_right * np.dot(body_right, body_forward) * (1 - np.cos(tilt)))
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        # Camera up is also rotated (perpendicular to forward, in the vertical plane)
        cam_up = np.cross(body_right, cam_forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        return cam_pos, cam_forward, cam_up, gate

    def _randomize_lighting(self):
        """Randomize scene lighting for domain randomization."""
        for i in range(self.model.nlight):
            # Random light position (scaled for larger scene)
            self.model.light_pos[i] = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(3, 12),
            ]
            # Random light color/intensity
            intensity = random.uniform(0.3, 1.0)
            self.model.light_diffuse[i] = [
                intensity * random.uniform(0.7, 1.0),
                intensity * random.uniform(0.7, 1.0),
                intensity * random.uniform(0.7, 1.0),
            ]

    def _randomize_gate_colors(self):
        """Randomize gate colors for domain randomization."""
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "gate" in str(name).lower():
                self.model.geom_rgba[i] = [
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0),
                    1.0,
                ]

        # Also randomize ground color occasionally
        if random.random() < 0.3:
            ground_id = 0  # Usually the first geom
            self.model.geom_rgba[ground_id] = [
                random.uniform(0.1, 0.5),
                random.uniform(0.1, 0.5),
                random.uniform(0.1, 0.5),
                1.0,
            ]

    def render_from_camera(self, cam_pos, cam_forward, cam_up):
        """Render the scene from a given camera pose."""
        target = cam_pos + cam_forward * 2.0
        dist = 0.01  # Very small — we position the camera directly

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = np.linalg.norm(target - cam_pos)
        cam.azimuth = np.degrees(np.arctan2(cam_forward[1], cam_forward[0]))
        cam.elevation = np.degrees(np.arcsin(np.clip(cam_forward[2], -1, 1)))

        self.renderer.update_scene(self.data, cam)
        frame = self.renderer.render()
        return frame  # RGB uint8 (H, W, 3)

    def generate_sample(self, randomize=True):
        """
        Generate one training sample: rendered image + keypoint labels.

        Returns:
            frame: (H, W, 3) RGB image
            labels: list of label strings in YOLOv8-pose format
        """
        if randomize:
            self._randomize_lighting()
            self._randomize_gate_colors()

        cam_pos, cam_forward, cam_up, target_gate = self._randomize_camera_pose()
        frame = self.render_from_camera(cam_pos, cam_forward, cam_up)

        # Generate labels for ALL visible gates
        labels = []
        for gate in self.gates:
            gate_pos = gate["pos"]
            gate_yaw = gate["yaw"]

            # Transform opening corners to world frame
            opening_world = transform_corners_to_world(
                GATE_CORNERS_LOCAL, gate_pos, gate_yaw
            )
            # Transform frame corners to world frame
            frame_world = transform_corners_to_world(
                GATE_FRAME_CORNERS_LOCAL, gate_pos, gate_yaw
            )

            # Project to image using spec-exact pinhole intrinsics
            opening_pixels, opening_in_front = project_points_to_image(
                opening_world, cam_pos, cam_forward, cam_up,
                fov_deg=self.fov, img_w=self.img_w, img_h=self.img_h,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            )
            frame_pixels, frame_in_front = project_points_to_image(
                frame_world, cam_pos, cam_forward, cam_up,
                fov_deg=self.fov, img_w=self.img_w, img_h=self.img_h,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            )

            # Generate label
            label = corners_to_yolo_keypoints(
                opening_pixels, opening_in_front,
                frame_pixels, frame_in_front,
                img_w=self.img_w, img_h=self.img_h
            )

            if label is not None:
                labels.append(label)

        return frame, labels

    def generate_dataset(self, output_dir, num_samples=500, train_split=0.8, prefix=""):
        """
        Generate a complete YOLOv8-pose format dataset.

        Args:
            output_dir: Root directory for the dataset.
            num_samples: Number of images to generate.
            train_split: Fraction of images for training.
            prefix: Optional filename prefix for accumulation across cycles.

        Creates:
            output_dir/
            ├── data.yaml          (dataset config for ultralytics)
            ├── train/
            │   ├── images/
            │   └── labels/
            └── valid/
                ├── images/
                └── labels/
        """
        output = Path(output_dir)
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train

        # Create directory structure
        for split in ["train", "valid"]:
            (output / split / "images").mkdir(parents=True, exist_ok=True)
            (output / split / "labels").mkdir(parents=True, exist_ok=True)

        # Write data.yaml for YOLOv8-pose
        data_yaml = {
            "path": str(output.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "nc": 1,
            "names": ["gate"],
            # Keypoint configuration
            "kpt_shape": [4, 3],  # 4 keypoints, 3 values each (x, y, visibility)
        }

        with open(output / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"\nGenerating {num_samples} samples ({num_train} train, {num_val} val)")
        print(f"Output: {output}")
        print(f"Format: YOLOv8-pose (bbox + 4 keypoints per gate)")
        print()

        total_gates = 0
        empty_frames = 0
        t_start = time.time()

        for i in range(num_samples):
            split = "train" if i < num_train else "valid"
            idx = i if i < num_train else (i - num_train)

            frame, labels = self.generate_sample(randomize=True)

            if len(labels) == 0:
                empty_frames += 1
                continue

            total_gates += len(labels)

            # Save image (convert RGB to BGR for OpenCV)
            img_path = output / split / "images" / f"{prefix}gate_kp_{idx:05d}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Save labels
            label_path = output / split / "labels" / f"{prefix}gate_kp_{idx:05d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(labels) + "\n")

            # Progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (num_samples - i - 1) / rate
                print(f"  [{i+1}/{num_samples}] {rate:.1f} img/s, "
                      f"ETA {eta:.0f}s, {total_gates} gates, "
                      f"{empty_frames} empty frames skipped")

        elapsed = time.time() - t_start
        print(f"\nDone! Generated {num_samples - empty_frames} images in {elapsed:.1f}s")
        print(f"  Total gate instances: {total_gates}")
        print(f"  Empty frames skipped: {empty_frames}")
        print(f"  Avg gates per image: {total_gates / max(1, num_samples - empty_frames):.1f}")
        print(f"\nDataset ready: {output / 'data.yaml'}")
        return str(output / "data.yaml")


# ============================================================
# Visualization / Debug
# ============================================================
def visualize_keypoints(image_path, label_path, img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    """
    Draw keypoint labels on an image for visual verification.
    Useful for debugging the data generator.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        return None

    with open(label_path, "r") as f:
        lines = f.read().strip().split("\n")

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
    corner_names = ["TL", "TR", "BR", "BL"]

    for line in lines:
        parts = line.split()
        if len(parts) < 17:  # 5 (bbox) + 12 (4 keypoints * 3)
            continue

        # Parse bbox
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Parse and draw keypoints
        for k in range(4):
            kp_x = float(parts[5 + k*3])
            kp_y = float(parts[6 + k*3])
            kp_v = int(parts[7 + k*3])

            if kp_v > 0:
                px = int(kp_x * img_w)
                py = int(kp_y * img_h)
                color = colors[k]
                cv2.circle(img, (px, py), 6, color, -1)
                cv2.putText(img, corner_names[k], (px + 8, py - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw lines connecting corners (gate outline)
        visible_pts = []
        for k in range(4):
            kp_v = int(parts[7 + k*3])
            if kp_v > 0:
                px = int(float(parts[5 + k*3]) * img_w)
                py = int(float(parts[6 + k*3]) * img_h)
                visible_pts.append((px, py))

        for j in range(len(visible_pts)):
            cv2.line(img, visible_pts[j], visible_pts[(j+1) % len(visible_pts)],
                    (0, 200, 200), 1)

    return img


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLOv8-pose keypoint training data")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of images to generate")
    parser.add_argument("--output", type=str, default="datasets/gates_keypoints",
                       help="Output directory")
    parser.add_argument("--sim_xml", type=str, default=None,
                       help="Path to MuJoCo sim XML (optional, uses standalone scene if not provided)")
    parser.add_argument("--img_w", type=int, default=640, help="Image width")
    parser.add_argument("--img_h", type=int, default=480, help="Image height")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV (degrees)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization images for debugging")
    args = parser.parse_args()

    print("=" * 60)
    print("AI Grand Prix - Phase 2: Keypoint Data Generator")
    print("=" * 60)

    gen = SyntheticKeypointGenerator(
        sim_xml_path=args.sim_xml,
        img_w=args.img_w,
        img_h=args.img_h,
        fov=args.fov,
    )

    data_yaml = gen.generate_dataset(
        output_dir=args.output,
        num_samples=args.num_samples,
    )

    # Optional: visualize a few samples for debugging
    if args.visualize:
        print("\nGenerating visualization samples...")
        viz_dir = Path(args.output) / "viz"
        viz_dir.mkdir(exist_ok=True)

        img_dir = Path(args.output) / "train" / "images"
        lbl_dir = Path(args.output) / "train" / "labels"

        for img_file in sorted(img_dir.glob("*.jpg"))[:10]:
            lbl_file = lbl_dir / img_file.with_suffix(".txt").name
            if lbl_file.exists():
                viz = visualize_keypoints(str(img_file), str(lbl_file),
                                         args.img_w, args.img_h)
                if viz is not None:
                    cv2.imwrite(str(viz_dir / f"viz_{img_file.name}"), viz)

        print(f"Visualizations saved to: {viz_dir}")
