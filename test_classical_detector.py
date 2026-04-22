"""Tests for the classical CV gate detector.

Validates:
  1. Detection of a bright-colored gate on a dark background
  2. No false positives on a blank/dark frame
  3. Range estimation from apparent size
  4. Bearing estimation from pixel offset
  5. Multiple gate detection
  6. None frame handling
  7. Color profile selection
"""

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

import cv2
import numpy as np

from vision.classical_detector import ClassicalGateDetector, PROFILES


def _make_dark_frame(w=640, h=480):
    """Create a dark desaturated background (VQ1-like)."""
    frame = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    # Add slight noise
    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def _draw_gate(frame, cx, cy, size, color_bgr):
    """Draw a bright hollow rectangle (gate) on the frame."""
    half = size // 2
    thickness = max(size // 8, 3)
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)
    return frame


def test_detect_bright_green_gate():
    """Detect a neon green gate on a dark background."""
    frame = _make_dark_frame()
    # Draw a bright green rectangle in the center
    _draw_gate(frame, 320, 240, 80, (0, 255, 0))

    det = ClassicalGateDetector(color_profiles=["neon_green"])
    result = det.detect(frame, None)

    assert len(result) >= 1, f"Expected ≥1 detection, got {len(result)}"
    d = result[0]
    # Should be roughly centered (bearing near 0)
    assert abs(d.bearing_h_deg) < 10.0, f"bearing_h={d.bearing_h_deg}° (expected near 0)"
    assert abs(d.bearing_v_deg) < 10.0, f"bearing_v={d.bearing_v_deg}° (expected near 0)"
    assert d.range_est > 0, f"range_est should be positive"
    assert d.confidence > 0.3, f"confidence={d.confidence} too low"
    assert d.gate_idx == -1, "gate_idx should be -1 (unknown)"
    print(f"  ✓ neon green gate: range={d.range_est:.1f}m, "
          f"bh={d.bearing_h_deg:.1f}°, conf={d.confidence:.2f}")


def test_no_false_positives_on_dark():
    """Dark frame with no gates → no detections."""
    frame = _make_dark_frame()
    det = ClassicalGateDetector(color_profiles=["neon_green"])
    result = det.detect(frame, None)
    assert len(result) == 0, f"Expected 0 detections on dark frame, got {len(result)}"
    print("  ✓ no false positives on dark frame")


def test_range_from_apparent_size():
    """Larger apparent gate → closer range estimate."""
    det = ClassicalGateDetector(color_profiles=["any_bright"])

    # Small gate (far away)
    frame_far = _make_dark_frame()
    _draw_gate(frame_far, 320, 240, 30, (0, 255, 255))
    res_far = det.detect(frame_far, None)

    # Large gate (close)
    frame_close = _make_dark_frame()
    _draw_gate(frame_close, 320, 240, 120, (0, 255, 255))
    res_close = det.detect(frame_close, None)

    assert len(res_far) >= 1 and len(res_close) >= 1
    assert res_close[0].range_est < res_far[0].range_est, (
        f"Close gate range {res_close[0].range_est:.1f} should be < "
        f"far gate range {res_far[0].range_est:.1f}"
    )
    print(f"  ✓ range estimation: close={res_close[0].range_est:.1f}m, "
          f"far={res_far[0].range_est:.1f}m")


def test_bearing_from_offset():
    """Gate offset to the right → positive bearing_h."""
    det = ClassicalGateDetector(color_profiles=["any_bright"])

    # Gate offset to the right
    frame = _make_dark_frame()
    _draw_gate(frame, 500, 240, 60, (255, 255, 0))
    result = det.detect(frame, None)

    assert len(result) >= 1
    assert result[0].bearing_h_deg > 5.0, (
        f"Right-offset gate should have positive bearing, got {result[0].bearing_h_deg}°"
    )

    # Gate offset to the left
    frame2 = _make_dark_frame()
    _draw_gate(frame2, 100, 240, 60, (255, 255, 0))
    result2 = det.detect(frame2, None)

    assert len(result2) >= 1
    assert result2[0].bearing_h_deg < -5.0, (
        f"Left-offset gate should have negative bearing, got {result2[0].bearing_h_deg}°"
    )
    print(f"  ✓ bearing: right={result[0].bearing_h_deg:.1f}°, "
          f"left={result2[0].bearing_h_deg:.1f}°")


def test_multiple_gates():
    """Detect multiple gates in one frame."""
    det = ClassicalGateDetector(color_profiles=["any_bright"])
    frame = _make_dark_frame()
    _draw_gate(frame, 160, 240, 50, (0, 0, 255))
    _draw_gate(frame, 480, 240, 70, (0, 255, 0))
    result = det.detect(frame, None)

    assert len(result) >= 2, f"Expected ≥2 detections, got {len(result)}"
    # Sorted nearest first — larger gate should be closer
    print(f"  ✓ multiple gates: {len(result)} detected")


def test_none_frame():
    """None frame returns empty list."""
    det = ClassicalGateDetector()
    result = det.detect(None, None)
    assert result == []
    print("  ✓ None frame → []")


def test_detector_name():
    """Name includes color profile."""
    det = ClassicalGateDetector(color_profiles=["neon_green", "neon_blue"])
    assert "neon_green" in det.name()
    assert "neon_blue" in det.name()
    print(f"  ✓ name: {det.name()}")


def test_any_bright_catches_various_colors():
    """'any_bright' profile detects various bright colors."""
    det = ClassicalGateDetector(color_profiles=["any_bright"])

    colors_bgr = [
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
    ]

    detected = 0
    for color in colors_bgr:
        frame = _make_dark_frame()
        _draw_gate(frame, 320, 240, 60, color)
        result = det.detect(frame, None)
        if len(result) >= 1:
            detected += 1

    assert detected >= 3, f"any_bright detected only {detected}/5 colors"
    print(f"  ✓ any_bright: detected {detected}/5 colors")


def test_factory_registration():
    """ClassicalGateDetector is available via make_detector('classical')."""
    from vision.detector import make_detector
    det = make_detector("classical", color_profiles=["neon_green"])
    assert "classical" in det.name()
    print(f"  ✓ factory: {det.name()}")


def main():
    print("Running ClassicalGateDetector tests...")
    test_detect_bright_green_gate()
    test_no_false_positives_on_dark()
    test_range_from_apparent_size()
    test_bearing_from_offset()
    test_multiple_gates()
    test_none_frame()
    test_detector_name()
    test_any_bright_catches_various_colors()
    test_factory_registration()
    print(f"\nAll tests passed! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
