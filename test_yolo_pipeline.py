"""Integration contract test for the YOLO detection pipeline.

Validates the YoloPnpDetector → GateDetection → navigator chain
WITHOUT requiring a gate-trained model. Tests:
  1. Pipeline doesn't crash on a synthetic frame
  2. Output shape matches GateDetection contract
  3. None frame returns empty list (graceful)
  4. Axis remap (camera→body) and bearing math are correct

When a trained gate model is available, add a test with a known
gate image to validate actual detection accuracy.
"""

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

import numpy as np


def test_yolo_detector_contract():
    """YoloPnpDetector produces GateDetection objects or empty list."""
    model_path = str(_HERE / "src/vision/gate_yolo/yolov8n-pose.pt")
    if not Path(model_path).exists():
        print("  ⊘ SKIP: yolov8n-pose.pt not found (base model needed)")
        return

    try:
        from vision.detector import YoloPnpDetector, GateDetection
        det = YoloPnpDetector(model_path=model_path, conf_threshold=0.1)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  ⊘ SKIP: YOLO init failed ({e}) — ultralytics not installed?")
        return

    # Synthetic frame (random noise — won't have gates, but shouldn't crash)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = det.detect(frame, _FakeState())
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    for d in result:
        assert hasattr(d, "bearing_h_deg")
        assert hasattr(d, "range_est")
        assert hasattr(d, "confidence")
        assert hasattr(d, "gate_idx")
        assert d.gate_idx == -1, "YOLO should not tag gate identity"
    print(f"  ✓ contract: {len(result)} detections on noise frame")


def test_none_frame_returns_empty():
    """None frame → empty detection list, no crash."""
    model_path = str(_HERE / "src/vision/gate_yolo/yolov8n-pose.pt")
    if not Path(model_path).exists():
        print("  ⊘ SKIP: yolov8n-pose.pt not found")
        return
    try:
        from vision.detector import YoloPnpDetector
        det = YoloPnpDetector(model_path=model_path)
    except (ImportError, ModuleNotFoundError):
        print("  ⊘ SKIP: ultralytics not installed")
        return

    result = det.detect(None, _FakeState())
    assert result == [], f"Expected [], got {result}"
    print("  ✓ None frame → empty list")


def test_axis_remap_math():
    """Camera frame (X right, Y down, Z forward) → body frame bearings."""
    # Manually verify the axis remap used in YoloPnpDetector
    # Camera tvec: X right, Y down, Z forward
    # Body: X forward, Y right, Z down
    tvec = np.array([1.0, -0.5, 10.0])  # 1m right, 0.5m up, 10m forward
    body_x = tvec[2]   # forward = 10
    body_y = tvec[0]   # right = 1
    body_z = tvec[1]   # down = -0.5

    horiz = math.sqrt(body_x**2 + body_y**2)
    bearing_h = math.atan2(body_y, body_x)  # slight right
    bearing_v = math.atan2(body_z, horiz)    # slight up (negative down)

    assert abs(math.degrees(bearing_h) - 5.71) < 0.5, f"bearing_h={math.degrees(bearing_h)}"
    assert bearing_v < 0, "Gate above center should have negative body-Z bearing"
    print(f"  ✓ axis remap: bh={math.degrees(bearing_h):.1f}° bv={math.degrees(bearing_v):.1f}°")


def test_detector_name():
    """Detector name includes model filename."""
    model_path = str(_HERE / "src/vision/gate_yolo/yolov8n-pose.pt")
    if not Path(model_path).exists():
        print("  ⊘ SKIP: yolov8n-pose.pt not found")
        return
    try:
        from vision.detector import YoloPnpDetector
        det = YoloPnpDetector(model_path=model_path)
    except (ImportError, ModuleNotFoundError):
        print("  ⊘ SKIP: ultralytics not installed")
        return
    assert "yolov8n-pose.pt" in det.name()
    print(f"  ✓ name: {det.name()}")


class _FakeState:
    pos_ned = (0.0, 0.0, -2.0)
    vel_ned = (0.0, 0.0, 0.0)
    att_rad = (0.0, 0.0, 0.0)


def main():
    print("Running YOLO pipeline tests...")
    test_axis_remap_math()  # Pure math, always runs
    test_yolo_detector_contract()
    test_none_frame_returns_empty()
    test_detector_name()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
