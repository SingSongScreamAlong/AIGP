"""Classical CV gate detector for VQ1 highlighted gates.

VQ1 gates are brightly colored (neon/LED) against a desaturated
environment. This detector uses HSV color thresholding + contour
analysis to find gates without any ML model.

Pipeline:
    Frame → HSV → color mask → morphology → contours → gate candidates
    → bounding rect → PnP (if 4 corners found) or pinhole range estimate
    → GateDetection (body-frame bearing + range)

Advantages over YOLO for VQ1:
    - Zero training data required
    - No GPU / ultralytics dependency
    - Sub-millisecond inference on CPU
    - Robust to the specific VQ1 visual design (highlighted gates)

Limitations:
    - Fails on VQ2/physical where gates are not color-highlighted
    - Tuning needed for actual VQ1 gate colors (adjust HSV ranges)
    - No keypoint regression — range estimated from apparent size

Coordinate conventions:
    - Image: origin top-left, X right, Y down
    - Body frame: X forward, Y right, Z down (FRD)
    - Bearings in degrees for GateDetection compatibility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Import the shared GateDetection type
try:
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "src"))
    from vision.detector import GateDetection
except ImportError:
    from dataclasses import dataclass as _dc

    @_dc
    class GateDetection:  # type: ignore[no-redef]
        gate_idx: int
        bearing_h_deg: float
        bearing_v_deg: float
        range_est: float
        angular_size_deg: float
        confidence: float
        in_fov: bool


# ─────────────────────────────────────────────────────────────────
# HSV color profiles for different gate highlight colors
# ─────────────────────────────────────────────────────────────────

@dataclass
class ColorProfile:
    """HSV range for a gate highlight color.

    H: 0-179 in OpenCV (half of 0-360 degrees)
    S: 0-255
    V: 0-255
    """
    name: str
    h_low: int
    h_high: int
    s_low: int
    s_high: int
    v_low: int
    v_high: int


# Common racing gate highlight colors. The actual VQ1 colors will
# be tuned once the sim drops. These are reasonable starting points.
PROFILES = {
    "neon_green": ColorProfile("neon_green", 35, 85, 100, 255, 150, 255),
    "neon_orange": ColorProfile("neon_orange", 5, 25, 150, 255, 180, 255),
    "neon_blue": ColorProfile("neon_blue", 100, 130, 100, 255, 150, 255),
    "neon_red": ColorProfile("neon_red", 0, 10, 150, 255, 150, 255),
    "neon_yellow": ColorProfile("neon_yellow", 20, 35, 150, 255, 180, 255),
    "neon_magenta": ColorProfile("neon_magenta", 140, 170, 100, 255, 150, 255),
    # Bright/white LED gates (high V, low S)
    "bright_white": ColorProfile("bright_white", 0, 179, 0, 60, 220, 255),
    # Catch-all: any highly saturated bright color
    "any_bright": ColorProfile("any_bright", 0, 179, 120, 255, 160, 255),
}


# ─────────────────────────────────────────────────────────────────
# Classical Detector
# ─────────────────────────────────────────────────────────────────

class ClassicalGateDetector:
    """HSV color + contour gate detector for highlighted gates.

    Implements the Detector protocol so it's a drop-in replacement
    for VirtualDetector or YoloPnpDetector in the race loop.
    """

    # Gate physical size — used to estimate range from apparent pixel size
    GATE_PHYSICAL_SIZE_M: float = 2.0  # meters (match VirtualCamera)

    # Minimum contour area (pixels) to consider as a gate candidate
    MIN_CONTOUR_AREA: int = 200

    # Maximum number of detections to return per frame
    MAX_DETECTIONS: int = 5

    # Morphology kernel size for cleaning up the mask
    MORPH_KERNEL_SIZE: int = 5

    def __init__(
        self,
        color_profiles: Optional[List[str]] = None,
        img_w: int = 640,
        img_h: int = 480,
        fov_deg: float = 90.0,
        min_contour_area: int = MIN_CONTOUR_AREA,
        gate_size_m: float = GATE_PHYSICAL_SIZE_M,
    ):
        """
        Args:
            color_profiles: List of profile names from PROFILES to use.
                If None, uses ["any_bright"] for maximum coverage.
            img_w: Expected image width.
            img_h: Expected image height.
            fov_deg: Camera vertical FOV in degrees.
            min_contour_area: Minimum contour area in pixels.
            gate_size_m: Physical gate size for range estimation.
        """
        if color_profiles is None:
            color_profiles = ["any_bright"]

        self._profiles: List[ColorProfile] = []
        for name in color_profiles:
            if name not in PROFILES:
                raise ValueError(
                    f"Unknown color profile {name!r}. "
                    f"Known: {sorted(PROFILES.keys())}"
                )
            self._profiles.append(PROFILES[name])

        self._img_w = img_w
        self._img_h = img_h
        self._fov_deg = fov_deg
        self._min_area = min_contour_area
        self._gate_size = gate_size_m

        # Precompute focal length from FOV
        self._fy = (img_h / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        self._fx = self._fy  # square pixels
        self._cx = img_w / 2.0
        self._cy = img_h / 2.0

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE),
        )

        self._profile_names = color_profiles

    def detect(self, frame, state) -> List[GateDetection]:
        """Detect highlighted gates in a camera frame.

        Args:
            frame: HxWx3 BGR uint8 image (OpenCV convention).
                Returns [] if None.
            state: DroneStateView (not used by classical detector,
                but required by Detector protocol).

        Returns:
            List of GateDetection in body frame, sorted nearest first.
        """
        if frame is None:
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build combined mask from all color profiles
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for prof in self._profiles:
            lo = np.array([prof.h_low, prof.s_low, prof.v_low])
            hi = np.array([prof.h_high, prof.s_high, prof.v_high])
            m = cv2.inRange(hsv, lo, hi)

            # Handle hue wraparound for red (H near 0 and 179)
            if prof.name == "neon_red":
                lo2 = np.array([170, prof.s_low, prof.v_low])
                hi2 = np.array([179, prof.s_high, prof.v_high])
                m = cv2.bitwise_or(m, cv2.inRange(hsv, lo2, hi2))

            mask = cv2.bitwise_or(mask, m)

        # Morphological cleanup: close gaps, remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and score candidates
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # Gate aspect ratio filter: gates are roughly square
            # Allow 0.3 - 3.0 aspect ratio to be generous
            aspect = w / max(h, 1)
            if aspect < 0.3 or aspect > 3.0:
                continue

            # Center of the bounding rect in pixel coordinates
            cx_px = x + w / 2.0
            cy_px = y + h / 2.0

            # Estimate range from apparent size
            # apparent_size_px ≈ gate_physical_size * focal_length / range
            apparent_size = max(w, h)
            if apparent_size < 5:
                continue
            range_est = (self._gate_size * self._fx) / apparent_size

            # Body-frame bearings from pixel coordinates
            # Image center → bearing = 0
            dx_px = cx_px - self._cx
            dy_px = cy_px - self._cy

            bearing_h_rad = math.atan2(dx_px, self._fx)
            bearing_v_rad = math.atan2(dy_px, self._fy)

            # Angular size
            angular_size = math.degrees(
                2.0 * math.atan(self._gate_size / (2.0 * max(range_est, 0.1)))
            )

            # Confidence: based on contour area relative to bounding box
            # (how "filled" the detection is — gates are hollow so ratio is
            # moderate, but noise blobs tend to be very sparse or very dense)
            fill_ratio = area / max(w * h, 1)
            # Sweet spot: 0.15 - 0.85 fill ratio for a hollow rectangle
            if 0.1 < fill_ratio < 0.9:
                conf = 0.6 + 0.4 * min(area / 2000.0, 1.0)
            else:
                conf = 0.3 + 0.2 * min(area / 2000.0, 1.0)

            # FOV check
            fov_h_lim = math.radians(self._fov_deg / 2.0)
            fov_v_lim = math.radians(
                self._fov_deg * self._img_h / self._img_w / 2.0
            )
            in_fov = (
                abs(bearing_h_rad) <= fov_h_lim
                and abs(bearing_v_rad) <= fov_v_lim
            )

            candidates.append(GateDetection(
                gate_idx=-1,  # unknown identity
                bearing_h_deg=math.degrees(bearing_h_rad),
                bearing_v_deg=math.degrees(bearing_v_rad),
                range_est=range_est,
                angular_size_deg=angular_size,
                confidence=conf,
                in_fov=in_fov,
            ))

        # Sort nearest first, limit count
        candidates.sort(key=lambda d: d.range_est)
        return candidates[:self.MAX_DETECTIONS]

    def name(self) -> str:
        return f"classical[{','.join(self._profile_names)}]"

    def detect_with_mask(
        self, frame
    ) -> Tuple[List[GateDetection], np.ndarray]:
        """Detect and return the debug mask for visualization.

        Returns:
            (detections, mask) where mask is the combined HSV threshold
            image. Useful for tuning color profiles.
        """
        if frame is None:
            return [], np.zeros((1, 1), dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for prof in self._profiles:
            lo = np.array([prof.h_low, prof.s_low, prof.v_low])
            hi = np.array([prof.h_high, prof.s_high, prof.v_high])
            m = cv2.inRange(hsv, lo, hi)
            if prof.name == "neon_red":
                lo2 = np.array([170, prof.s_low, prof.v_low])
                hi2 = np.array([179, prof.s_high, prof.v_high])
                m = cv2.bitwise_or(m, cv2.inRange(hsv, lo2, hi2))
            mask = cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # Reuse the standard detect() for the detections — the mask
        # was computed identically there. Slightly wasteful but keeps
        # the debug path simple.
        detections = self.detect(frame, None)
        return detections, mask
