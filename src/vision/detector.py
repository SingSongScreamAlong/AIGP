"""Detector abstraction — Session 19.

Bridges the camera→planner gap. Every detector emits the same
GateDetection shape the existing belief model and navigator already
consume, so swapping vision backends is a 1-line change at the call
site.

Backends today:
    VirtualDetector — wraps the existing VirtualCamera that projects
                      known 3D gates with synthetic noise. Used for
                      regression tests, ablations, and when the sim
                      doesn't render a camera frame.
    YoloPnpDetector — wraps GateKeypointDetector (YOLOv8-pose + PnP).
                      Takes an HxWx3 image and produces body-frame
                      bearings the belief model understands.

The critical invariant: both detectors emit identical GateDetection
shapes. The only semantic difference is gate_idx — VirtualDetector
tags with sim-truth indices for bookkeeping; YoloPnpDetector tags
gate_idx=-1 ("unknown, caller associates"), because a real image tells
you "there is a gate" but not "this is gate #3".

Imports are lazy. This module must import cleanly without
ultralytics / torch / the current vision_nav.py being present, so it
can be unit-tested in isolation and deployed to minimal targets.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Re-usable GateDetection shape
# ─────────────────────────────────────────────────────────────────────
#
# We mirror the fields vision_nav.GateDetection already defines, so the
# belief model can consume either. If vision_nav is importable we re-
# export its class; otherwise we define a local equivalent. Duplicating
# a dataclass is cheaper than forcing every consumer to depend on the
# full vision_nav module (which imports mavsdk at top).

try:
    # Add repo root if called from inside src/
    _REPO = Path(__file__).resolve().parents[2]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from vision_nav import GateDetection as _GateDetectionImported  # type: ignore
    GateDetection = _GateDetectionImported
    _FROM_VISION_NAV = True
except Exception:
    _FROM_VISION_NAV = False

    @dataclass
    class GateDetection:  # type: ignore[no-redef]
        """Body-frame detection of a single gate.

        Matches the shape expected by GateBelief.update_detected and the
        existing tracker: bearings in DEGREES (converted to radians at
        the consumer), range in meters, angular_size in degrees,
        confidence in [0, 1], in_fov bool.
        """
        gate_idx: int
        bearing_h_deg: float
        bearing_v_deg: float
        range_est: float
        angular_size_deg: float
        confidence: float
        in_fov: bool


# ─────────────────────────────────────────────────────────────────────
# Detector protocol
# ─────────────────────────────────────────────────────────────────────

class DroneStateView(Protocol):
    """Minimal view of drone state a detector might need.

    This is a structural Protocol — any object with pos_ned/vel_ned/
    att_rad tuples satisfies it, including sim.adapter.SimState.
    """
    pos_ned: tuple
    vel_ned: tuple
    att_rad: tuple


class Detector(Protocol):
    """Consumes a camera frame + drone state; returns body-frame
    gate detections.

    The frame may be None if the backend doesn't need one (e.g.
    VirtualDetector projects known-gates from state alone). The
    contract: return an empty list when no gate is visible; never
    None; never raise on a well-formed call.
    """
    def detect(
        self,
        frame: Optional[np.ndarray],
        state: DroneStateView,
    ) -> List[GateDetection]: ...

    def name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────
# VirtualDetector — projection-based, for sim + tests
# ─────────────────────────────────────────────────────────────────────

class VirtualDetector:
    """Projects known 3D gates through a synthetic camera model.

    Wraps the existing vision_nav.VirtualCamera so regression tests
    and the S18 A/B harness keep working unchanged. Consumes drone
    state via the DroneStateView protocol; frame arg is ignored.
    """

    def __init__(self, gates, noise_profile: str = "clean", seed: int = 42):
        """
        Args:
            gates: list of (N, E, D) tuples — sim-truth gate positions.
            noise_profile: key into vision_nav.NOISE_PROFILES.
            seed: RNG seed forwarded to VirtualCamera. Each seed produces
                a different noise/miss_prob trajectory — used by
                `soak.py` to surface variance-driven failure modes.
                Default 42 preserves prior deterministic behaviour for
                test fixtures that assume it.
        """
        from vision_nav import VirtualCamera
        self._camera = VirtualCamera(gates, noise_profile, seed=seed)
        self._profile = noise_profile
        self._seed = int(seed)

    def detect(self, frame, state) -> List[GateDetection]:
        pos = list(state.pos_ned)
        vel = list(state.vel_ned)
        yaw_deg = math.degrees(state.att_rad[2])
        return self._camera.observe(pos, vel, yaw_deg)

    def name(self) -> str:
        return f"virtual[{self._profile},seed={self._seed}]"


# ─────────────────────────────────────────────────────────────────────
# YoloPnpDetector — YOLOv8-pose + PnP over a real camera frame
# ─────────────────────────────────────────────────────────────────────

class YoloPnpDetector:
    """Real vision path: YOLO keypoints → PnP → body-frame bearings.

    Imports ultralytics lazily so this module is still importable on
    machines without torch/YOLO installed (e.g. the test sandbox).

    Coordinate convention:
      * OpenCV / PnP camera frame: X right, Y down, Z forward.
      * Our body frame:            X forward, Y right, Z down.
    The axis remap below converts tvec → body_forward/right/down
    before computing bearings. This is the same convention the belief
    model and VirtualCamera already assume, so downstream math is
    unchanged.

    Confidence is the YOLO detection confidence attenuated by
    reprojection error: a geometrically-bad PnP shouldn't be trusted
    even with a high YOLO score.
    """

    # Default intrinsics for Neros 12MP wide-angle camera. Override
    # via constructor if you calibrate.
    DEFAULT_FOV_DEG = 90.0
    DEFAULT_IMG_W = 640
    DEFAULT_IMG_H = 480
    GATE_PHYSICAL_SIZE = 2.0  # meters, consistent with VirtualCamera

    def __init__(
        self,
        model_path: str,
        img_w: int = DEFAULT_IMG_W,
        img_h: int = DEFAULT_IMG_H,
        fov_deg: float = DEFAULT_FOV_DEG,
        conf_threshold: float = 0.5,
        keypoint_conf_threshold: float = 0.3,
        camera_matrix=None,
        dist_coeffs=None,
        max_reproj_err_px: float = 8.0,
    ):
        # Lazy imports so bare `import detector` works without YOLO.
        try:
            from vision.gate_yolo.detect_keypoints import GateKeypointDetector
        except Exception:
            # Fallback path if called from repo root
            sys.path.insert(0, str(Path(__file__).parent / "gate_yolo"))
            from detect_keypoints import GateKeypointDetector  # type: ignore

        self._impl = GateKeypointDetector(
            model_path=model_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            img_w=img_w,
            img_h=img_h,
            fov_deg=fov_deg,
            conf_threshold=conf_threshold,
            keypoint_conf_threshold=keypoint_conf_threshold,
        )
        self._img_w = img_w
        self._img_h = img_h
        self._fov_deg = fov_deg
        self._max_reproj = max_reproj_err_px
        self._model_path = model_path

    def detect(self, frame: Optional[np.ndarray], state) -> List[GateDetection]:
        if frame is None:
            return []
        dets = self._impl.detect(frame, estimate_pose=True)
        out: List[GateDetection] = []
        for d in dets:
            pose = d.get("pose")
            if not pose or not pose.get("success"):
                continue

            # Reject geometrically-bad PnPs. A 2 m gate at 10 m typical
            # range shouldn't reproject with errors above ~8 px.
            reproj = float(pose.get("reprojection_error", 0.0))
            if reproj > self._max_reproj:
                continue

            tvec = pose["tvec"]  # camera frame: (X right, Y down, Z forward)
            # Axis remap → body frame (X forward, Y right, Z down)
            body_x = float(tvec[2])  # forward
            body_y = float(tvec[0])  # right
            body_z = float(tvec[1])  # down

            # Behind the camera or nonsense depth
            if body_x <= 0.1:
                continue

            horiz = math.sqrt(body_x * body_x + body_y * body_y)
            bearing_h = math.atan2(body_y, body_x)
            bearing_v = math.atan2(body_z, horiz) if horiz > 0.01 else 0.0
            range_est = float(np.linalg.norm(tvec))

            angular_size = math.degrees(
                2.0 * math.atan(self.GATE_PHYSICAL_SIZE / (2.0 * range_est))
            ) if range_est > 0.1 else 0.0

            # FOV check is redundant (YOLO wouldn't have detected outside
            # frame) but we set it for consistency with VirtualCamera.
            fov_h_lim = math.radians(self._fov_deg / 2.0)
            fov_v_lim = math.radians(self._fov_deg * self._img_h / self._img_w / 2.0)
            in_fov = abs(bearing_h) <= fov_h_lim and abs(bearing_v) <= fov_v_lim

            # Attenuate YOLO conf by reprojection quality.
            det_conf = float(d.get("confidence", 0.0))
            geom_conf = max(0.0, 1.0 - reproj / self._max_reproj)
            confidence = det_conf * geom_conf

            out.append(GateDetection(
                gate_idx=-1,                     # unknown — caller associates
                bearing_h_deg=math.degrees(bearing_h),
                bearing_v_deg=math.degrees(bearing_v),
                range_est=range_est,
                angular_size_deg=angular_size,
                confidence=confidence,
                in_fov=in_fov,
            ))
        # Sort nearest first, same convention as GateKeypointDetector
        out.sort(key=lambda g: g.range_est)
        return out

    def name(self) -> str:
        return f"yolo_pnp[{Path(self._model_path).name}]"


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────

def make_detector(kind: str, **kwargs) -> Detector:
    """Construct a detector by name.

    Known kinds:
      "virtual"    — VirtualDetector(gates=..., noise_profile=...)
      "yolo_pnp"   — YoloPnpDetector(model_path=..., ...)
      "classical"  — ClassicalGateDetector(color_profiles=..., ...)

    Raises ValueError on unknown kind so typos fail loudly.
    """
    if kind == "virtual":
        return VirtualDetector(**kwargs)  # type: ignore[return-value]
    if kind == "yolo_pnp":
        return YoloPnpDetector(**kwargs)  # type: ignore[return-value]
    if kind == "classical":
        from vision.classical_detector import ClassicalGateDetector
        return ClassicalGateDetector(**kwargs)  # type: ignore[return-value]
    raise ValueError(
        f"Unknown detector kind: {kind!r}. Known: 'virtual', 'yolo_pnp', 'classical'."
    )
