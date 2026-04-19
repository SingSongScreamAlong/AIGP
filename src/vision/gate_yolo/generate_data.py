"""
Synthetic Gate Detection Data Generator
========================================
Generates labeled training images from the LSY Drone Racing MuJoCo sim.

This gives us full control over our training data:
  - Unlimited images with perfect bounding box labels
  - Domain randomization for sim-to-real transfer
  - Camera views from positions the drone will actually fly

The sim has 4 gates with known 3D positions. We:
  1. Place a virtual camera at various positions around the track
  2. Render each view with MuJoCo's offscreen renderer
  3. Project gate 3D corners → 2D pixel coordinates
  4. Convert to YOLO format (normalized center + width/height)
  5. Optionally randomize lighting, colors, textures

Output: YOLOv8-format dataset ready for 02_train.py

Usage:
    cd ~/ai-grand-prix/src/vision/gate_yolo
    python generate_data.py --num_images 500 --randomize
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Gate geometry from gate.xml — the 4 inner corners of the gate opening
# In the gate's local frame, the opening is 0.4m wide (±0.2) and 0.4m tall (±0.2)
# Frame outer edge is 0.7m (±0.35), textured panels sit between frame and opening
# ---------------------------------------------------------------------------
GATE_HALF_SIZE = 0.2  # half-width of the gate opening
GATE_CORNERS_LOCAL = np.array([
    [0.0, -GATE_HALF_SIZE,  GATE_HALF_SIZE],   # top-left
    [0.0,  GATE_HALF_SIZE,  GATE_HALF_SIZE],   # top-right
    [0.0,  GATE_HALF_SIZE, -GATE_HALF_SIZE],   # bottom-right
    [0.0, -GATE_HALF_SIZE, -GATE_HALF_SIZE],   # bottom-left
])

# Full frame corners (outer boundary) for bounding box
FRAME_HALF_SIZE = 0.35
GATE_FRAME_CORNERS_LOCAL = np.array([
    [0.0, -FRAME_HALF_SIZE,  FRAME_HALF_SIZE],
    [0.0,  FRAME_HALF_SIZE,  FRAME_HALF_SIZE],
    [0.0,  FRAME_HALF_SIZE, -FRAME_HALF_SIZE],
    [0.0, -FRAME_HALF_SIZE, -FRAME_HALF_SIZE],
])


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (radians) to a 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ])
    return R


def quat_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def gate_corners_world(gate_pos, gate_rpy):
    """Get gate frame corners in world coordinates."""
    R = euler_to_rotation_matrix(*gate_rpy)
    corners = (R @ GATE_FRAME_CORNERS_LOCAL.T).T + gate_pos
    return corners


def project_points_to_image(points_3d, cam_pos, cam_forward, cam_up, fovy_deg, width, height):
    """
    Project 3D world points to 2D pixel coordinates using a pinhole camera model.

    Args:
        points_3d: (N, 3) array of 3D points
        cam_pos: (3,) camera position in world
        cam_forward: (3,) camera forward direction (normalized)
        cam_up: (3,) camera up direction (normalized)
        fovy_deg: vertical field of view in degrees
        width, height: image dimensions in pixels

    Returns:
        pixels: (N, 2) array of (x, y) pixel coordinates
        visible: (N,) boolean array — True if point is in front of camera
    """
    # Build camera coordinate frame
    cam_right = np.cross(cam_forward, cam_up)
    cam_right /= np.linalg.norm(cam_right)
    cam_up_fixed = np.cross(cam_right, cam_forward)
    cam_up_fixed /= np.linalg.norm(cam_up_fixed)

    # Points in camera frame
    diff = points_3d - cam_pos  # (N, 3)
    x_cam = diff @ cam_right
    y_cam = diff @ cam_up_fixed
    z_cam = diff @ cam_forward  # depth

    visible = z_cam > 0.01  # in front of camera

    # Focal length from FOV
    fovy_rad = np.radians(fovy_deg)
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy  # square pixels

    # Project
    z_safe = np.where(visible, z_cam, 1.0)
    px = (fx * x_cam / z_safe) + width / 2.0
    py = height / 2.0 - (fy * y_cam / z_safe)  # flip y (image y goes down)

    pixels = np.stack([px, py], axis=-1)
    return pixels, visible


def corners_to_yolo(corners_2d, visible, img_width, img_height):
    """
    Convert projected 2D corners to YOLO bounding box format.

    Returns:
        (x_center, y_center, width, height) normalized 0-1, or None if not visible
    """
    if not np.all(visible):
        return None

    x_min = np.clip(corners_2d[:, 0].min(), 0, img_width)
    x_max = np.clip(corners_2d[:, 0].max(), 0, img_width)
    y_min = np.clip(corners_2d[:, 1].min(), 0, img_height)
    y_max = np.clip(corners_2d[:, 1].max(), 0, img_height)

    # Skip if box is too small or entirely off-screen
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 5 or box_h < 5:
        return None
    if x_max <= 0 or x_min >= img_width or y_max <= 0 or y_min >= img_height:
        return None

    # YOLO format: normalized center + size
    cx = (x_min + x_max) / 2.0 / img_width
    cy = (y_min + y_max) / 2.0 / img_height
    w = box_w / img_width
    h = box_h / img_height

    return (cx, cy, w, h)


class SyntheticDataGenerator:
    """Generate labeled gate detection data from the MuJoCo sim."""

    def __init__(self, config_path: str = None, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.fovy = 60.0  # degrees, typical drone camera FOV

        # Gate positions from the level0 config
        self.gates = [
            {"pos": np.array([0.5, 0.25, 0.7]), "rpy": np.array([0.0, 0.0, -0.78])},
            {"pos": np.array([1.05, 0.75, 1.2]), "rpy": np.array([0.0, 0.0, 2.35])},
            {"pos": np.array([-1.0, -0.25, 0.7]), "rpy": np.array([0.0, 0.0, 3.14])},
            {"pos": np.array([0.0, -0.75, 1.2]), "rpy": np.array([0.0, 0.0, 0.0])},
        ]

        # Pre-compute gate corners in world frame
        self.gate_corners = []
        for g in self.gates:
            corners = gate_corners_world(g["pos"], g["rpy"])
            self.gate_corners.append(corners)

        # Set up MuJoCo for rendering
        self._setup_mujoco()

    def _setup_mujoco(self):
        """Load the LSY sim's MuJoCo model for rendering."""
        # Try to load the sim environment and use its model
        try:
            self._setup_from_sim()
        except Exception as e:
            print(f"Could not load full sim ({e}), building standalone MuJoCo scene...")
            self._setup_standalone()

    def _setup_from_sim(self):
        """Load the full sim environment for accurate rendering."""
        import gymnasium
        from lsy_drone_racing.utils import load_config

        # Find the config
        lsy_path = Path.home() / "ai-grand-prix" / "sims" / "lsy_drone_racing"
        config = load_config(lsy_path / "config" / "level0.toml")
        config.sim.render = False  # We'll render manually

        env = gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=42,
        )

        obs, info = env.reset()

        # Get the underlying MuJoCo model and data
        core_env = env.unwrapped
        self.mj_model = core_env.sim.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)

        # Sync the state
        import jax.numpy as jnp
        self.mj_data.qpos[:] = np.array(core_env.sim.mjx_data.qpos[0, :])
        self.mj_data.mocap_pos[:] = np.array(core_env.sim.mjx_data.mocap_pos[0, :])
        self.mj_data.mocap_quat[:] = np.array(core_env.sim.mjx_data.mocap_quat[0, :])
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Create renderer
        self.renderer = mujoco.Renderer(self.mj_model, height=self.height, width=self.width)
        self.fovy = self.mj_model.vis.global_.fovy

        self.env = env
        self._has_sim = True
        print(f"Loaded full sim. Model has {self.mj_model.nbody} bodies, fovy={self.fovy:.1f}")

    def _setup_standalone(self):
        """Build a minimal MuJoCo scene with just the gates for rendering."""
        xml = self._build_scene_xml()
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer = mujoco.Renderer(self.mj_model, height=self.height, width=self.width)
        self._has_sim = False
        print("Built standalone MuJoCo scene with gates.")

    def _build_scene_xml(self):
        """Build a MuJoCo XML with the 4 gates positioned in the scene."""
        gate_bodies = ""
        for i, g in enumerate(self.gates):
            x, y, z = g["pos"]
            r, p, yaw = np.degrees(g["rpy"])
            gate_bodies += f"""
        <body name="gate_{i}" pos="{x} {y} {z}" euler="{r} {p} {yaw}">
            <geom type="box" size="0.01 0.35 0.35" rgba="0.5 0.5 0.5 1"/>
            <geom type="box" size="0.005 0.2 0.05" pos="0 0 0.25" rgba="0.8 0.2 0.2 1"/>
            <geom type="box" size="0.005 0.2 0.05" pos="0 0 -0.25" rgba="0.8 0.2 0.2 1"/>
            <geom type="box" size="0.005 0.05 0.2" pos="0 -0.25 0" rgba="0.2 0.2 0.8 1"/>
            <geom type="box" size="0.005 0.05 0.2" pos="0 0.25 0" rgba="0.2 0.2 0.8 1"/>
        </body>"""

        xml = f"""
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <visual>
            <global fovy="60" offwidth="{self.width}" offheight="{self.height}"/>
          </visual>
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.4 0.4 0.4" rgb2="0.35 0.35 0.35"/>
            <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
          </asset>
          <worldbody>
            <light pos="0 0 4" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3" castshadow="true"/>
            <light pos="2 2 3" dir="-1 -1 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
            <geom type="plane" size="5 5 0.01" material="floor_mat"/>
            {gate_bodies}
          </worldbody>
        </mujoco>
        """
        return xml

    def _randomize_camera_pos(self):
        """Generate a random camera position that looks toward a gate."""
        # Pick a random gate to look at
        target_idx = random.randint(0, len(self.gates) - 1)
        target_pos = self.gates[target_idx]["pos"].copy()

        # Add slight jitter to target
        target_pos += np.random.uniform(-0.1, 0.1, 3)

        # Camera distance from gate: 0.5m to 3.0m (typical racing distances)
        dist = np.random.uniform(0.5, 3.0)

        # Random direction to place camera (hemisphere facing the gate front)
        gate_yaw = self.gates[target_idx]["rpy"][2]
        gate_forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])

        # Camera should generally be in front of the gate (where drone approaches from)
        # Add random lateral and vertical offset
        lateral_offset = np.random.uniform(-1.5, 1.5)
        vertical_offset = np.random.uniform(-0.3, 0.5)

        # Position camera in front of gate with some randomness
        cam_pos = target_pos - gate_forward * dist
        cam_pos += np.array([0, lateral_offset * 0.3, vertical_offset])

        # Look direction: toward the gate with some noise
        look_dir = target_pos - cam_pos
        look_dir /= np.linalg.norm(look_dir)

        # Camera up (roughly world up with some tilt)
        tilt = np.random.uniform(-0.1, 0.1)
        cam_up = np.array([tilt, tilt, 1.0])
        cam_up /= np.linalg.norm(cam_up)

        return cam_pos, look_dir, cam_up

    def _randomize_lighting(self):
        """Randomize scene lighting for domain randomization."""
        if not hasattr(self, "mj_model"):
            return

        n_lights = self.mj_model.nlight
        for i in range(n_lights):
            # Random light intensity
            intensity = np.random.uniform(0.3, 1.5)
            self.mj_model.light_diffuse[i] = np.array([intensity, intensity, intensity])

            # Slight color tint
            tint = np.random.uniform(0.8, 1.2, 3)
            self.mj_model.light_diffuse[i] *= tint
            self.mj_model.light_diffuse[i] = np.clip(self.mj_model.light_diffuse[i], 0, 1)

            # Random position jitter
            self.mj_model.light_pos[i] += np.random.uniform(-0.5, 0.5, 3)

    def _randomize_gate_colors(self):
        """Randomize gate material colors for domain randomization."""
        if not self._has_sim:
            # For standalone scene, randomize geom colors
            for i in range(self.mj_model.ngeom):
                name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and "gate" in str(name):
                    # Random but gate-like colors
                    self.mj_model.geom_rgba[i, :3] = np.random.uniform(0.2, 1.0, 3)

    def render_from_camera(self, cam_pos, cam_forward, cam_up):
        """Render a frame from a specific camera position and orientation."""
        # Set up the virtual camera using a lookat-style configuration
        target = cam_pos + cam_forward * 2.0  # look-at point

        # Build camera struct
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = float(np.linalg.norm(target - cam_pos))
        cam.azimuth = float(np.degrees(np.arctan2(cam_forward[1], cam_forward[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(cam_forward[2], -1, 1))))

        self.renderer.update_scene(self.mj_data, camera=cam)

        frame = self.renderer.render()
        return frame  # RGB numpy array

    def generate_sample(self, randomize=False):
        """
        Generate one labeled training sample.

        Returns:
            frame: (H, W, 3) RGB image
            labels: list of YOLO format strings "class_id cx cy w h"
        """
        # Randomize if requested
        if randomize:
            self._randomize_lighting()
            self._randomize_gate_colors()

        # Random camera position
        cam_pos, cam_forward, cam_up = self._randomize_camera_pos()

        # Render
        frame = self.render_from_camera(cam_pos, cam_forward, cam_up)

        # Project gates to 2D and create labels
        labels = []
        for i, corners_3d in enumerate(self.gate_corners):
            pixels, visible = project_points_to_image(
                corners_3d, cam_pos, cam_forward, cam_up,
                self.fovy, self.width, self.height
            )
            yolo_box = corners_to_yolo(pixels, visible, self.width, self.height)
            if yolo_box is not None:
                cx, cy, w, h = yolo_box
                # Class 0 = gate
                labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return frame, labels

    def generate_dataset(self, output_dir: str, num_images: int = 500,
                         train_split: float = 0.8, randomize: bool = True):
        """
        Generate a full YOLOv8-format dataset.

        Args:
            output_dir: Where to save the dataset
            num_images: Total number of images to generate
            train_split: Fraction for training (rest goes to validation)
            randomize: Apply domain randomization
        """
        # Create directory structure
        for split in ["train", "valid"]:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

        num_train = int(num_images * train_split)
        total_labels = 0
        empty_frames = 0

        print(f"Generating {num_images} images ({num_train} train, {num_images - num_train} val)...")
        print(f"Domain randomization: {'ON' if randomize else 'OFF'}")
        print()

        for i in range(num_images):
            split = "train" if i < num_train else "valid"
            frame, labels = self.generate_sample(randomize=randomize)

            # Save image (convert RGB to BGR for OpenCV)
            img_path = os.path.join(output_dir, split, "images", f"frame_{i:05d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Save labels
            label_path = os.path.join(output_dir, split, "labels", f"frame_{i:05d}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            total_labels += len(labels)
            if len(labels) == 0:
                empty_frames += 1

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] Generated. Gates in this frame: {len(labels)}")

        # Write data.yaml
        yaml_content = f"""# Synthetic Gate Detection Dataset
# Generated from LSY Drone Racing MuJoCo sim
# {num_images} images with domain randomization={'ON' if randomize else 'OFF'}

train: {os.path.abspath(os.path.join(output_dir, 'train', 'images'))}
val: {os.path.abspath(os.path.join(output_dir, 'valid', 'images'))}

nc: 1
names: ['gate']
"""
        yaml_path = os.path.join(output_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"\n{'='*50}")
        print(f"Dataset generation complete!")
        print(f"  Total images:  {num_images}")
        print(f"  Total labels:  {total_labels} gate detections")
        print(f"  Empty frames:  {empty_frames} (no visible gates)")
        print(f"  Avg gates/img: {total_labels / max(num_images - empty_frames, 1):.1f}")
        print(f"  Output dir:    {output_dir}")
        print(f"  data.yaml:     {yaml_path}")
        print(f"{'='*50}")

        return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gate detection training data")
    parser.add_argument("--output", default="datasets/gates",
                        help="Output directory for the dataset")
    parser.add_argument("--num_images", type=int, default=500,
                        help="Number of images to generate")
    parser.add_argument("--width", type=int, default=640,
                        help="Image width in pixels")
    parser.add_argument("--height", type=int, default=480,
                        help="Image height in pixels")
    parser.add_argument("--randomize", action="store_true", default=True,
                        help="Apply domain randomization")
    parser.add_argument("--no-randomize", dest="randomize", action="store_false",
                        help="Disable domain randomization")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Train/val split ratio")
    args = parser.parse_args()

    print("Gate Detection Synthetic Data Generator")
    print("=" * 40)
    print()

    gen = SyntheticDataGenerator(width=args.width, height=args.height)
    gen.generate_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        train_split=args.train_split,
        randomize=args.randomize,
    )

    print(f"\nNext step: train your model with:")
    print(f"  python 02_train.py")


if __name__ == "__main__":
    main()
