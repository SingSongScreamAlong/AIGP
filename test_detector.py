"""Tests for src/vision/detector.py — Session 19.

Two separable layers to exercise:
  * VirtualDetector correctness: with one gate at NED (10, 0, 0) and
    the drone at origin facing North, the detector must emit exactly
    one GateDetection with bearing_h ≈ 0° and range ≈ 10 m.
  * YoloPnpDetector adapter correctness: the axis remap from OpenCV
    camera frame (X right, Y down, Z forward) to body frame (X
    forward, Y right, Z down), the behind-camera reject, the
    reprojection-error reject, and the confidence attenuation.

The YOLO model itself is not exercised — we inject a fake
GateKeypointDetector that returns prepared pose dicts, so these tests
run in the sandbox without torch/ultralytics and without a .pt file.

Run standalone:
    python test_detector.py
"""

from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass


# Stub mavsdk so vision_nav can be imported (it imports mavsdk at top).
if "mavsdk" not in sys.modules:
    m = types.ModuleType("mavsdk")
    o = types.ModuleType("mavsdk.offboard")

    class _S:
        def __init__(self, *a, **k): pass
    m.System = _S
    for n in ("VelocityNedYaw", "PositionNedYaw", "Attitude"):
        setattr(o, n, type(n, (), {"__init__": lambda self, *a, **k: None}))
    o.OffboardError = type("OffboardError", (Exception,), {})
    sys.modules["mavsdk"] = m
    sys.modules["mavsdk.offboard"] = o


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                             # for vision_nav
sys.path.insert(0, os.path.join(_HERE, "src"))        # for vision.detector
sys.path.insert(0, os.path.join(_HERE, "src", "vision"))  # for detector if needed


@dataclass
class _FakeState:
    """Mimics SimState for the Detector protocol."""
    pos_ned: tuple = (0.0, 0.0, 0.0)
    vel_ned: tuple = (0.0, 0.0, 0.0)
    att_rad: tuple = (0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────
# VirtualDetector tests
# ─────────────────────────────────────────────────────────────────────

def test_virtual_detector_sees_gate_ahead():
    from vision.detector import VirtualDetector, GateDetection
    # Single gate 10 m north of origin, drone at origin facing North
    det = VirtualDetector(gates=[(10.0, 0.0, 0.0)], noise_profile="clean")
    dets = det.detect(frame=None, state=_FakeState())
    assert isinstance(dets, list)
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    d = dets[0]
    assert isinstance(d, GateDetection)
    assert d.gate_idx == 0
    assert abs(d.bearing_h_deg) < 2.0, f"bearing_h {d.bearing_h_deg} should be ~0"
    assert abs(d.range_est - 10.0) < 0.5, f"range {d.range_est} should be ~10"
    assert d.in_fov is True
    print(f"  ✓ gate-ahead: bearing={d.bearing_h_deg:.2f}°, range={d.range_est:.2f} m")


def test_virtual_detector_yaw_rotates_bearing():
    from vision.detector import VirtualDetector
    # Gate north, drone yaws east (90°); gate should now be on the LEFT → bearing ~ -90°
    det = VirtualDetector(gates=[(10.0, 0.0, 0.0)], noise_profile="clean")
    state = _FakeState(att_rad=(0.0, 0.0, math.radians(90.0)))
    dets = det.detect(frame=None, state=state)
    # FOV 120° only covers ±60° — gate should be out of FOV
    assert len(dets) == 0, "gate should be behind the 120° FOV at yaw=90°"
    print("  ✓ yaw carries bearing correctly; gate drops out of FOV")


def test_virtual_detector_frame_arg_ignored():
    from vision.detector import VirtualDetector
    import numpy as np
    det = VirtualDetector(gates=[(10.0, 0.0, 0.0)], noise_profile="clean")
    # Passing a frame doesn't change behaviour for a virtual detector
    d1 = det.detect(frame=None, state=_FakeState())
    d2 = det.detect(frame=np.zeros((240, 320, 3), dtype=np.uint8), state=_FakeState())
    assert len(d1) == len(d2) == 1
    print("  ✓ frame arg ignored by VirtualDetector")


# ─────────────────────────────────────────────────────────────────────
# YoloPnpDetector tests — inject a fake GateKeypointDetector
# ─────────────────────────────────────────────────────────────────────

def _install_fake_keypoint_detector(monkey_dets):
    """Install a fake `detect_keypoints.GateKeypointDetector` module.

    monkey_dets is the raw `detect()` return value we want the fake
    to emit.
    """
    # Build fake modules so the lazy import inside YoloPnpDetector finds them.
    gate_yolo_pkg = types.ModuleType("vision.gate_yolo")
    gate_yolo_pkg.__path__ = []  # mark as package
    detect_mod = types.ModuleType("vision.gate_yolo.detect_keypoints")

    class FakeGateKeypointDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._dets = monkey_dets

        def detect(self, frame, estimate_pose=True):
            return list(self._dets)

    detect_mod.GateKeypointDetector = FakeGateKeypointDetector
    sys.modules["vision.gate_yolo"] = gate_yolo_pkg
    sys.modules["vision.gate_yolo.detect_keypoints"] = detect_mod


def _pose_dict(tvec, reproj=1.0):
    import numpy as np
    return {
        "success": True,
        "tvec": np.array(tvec, dtype=float),
        "rvec": np.zeros(3),
        "distance": float(np.linalg.norm(tvec)),
        "rotation_matrix": np.eye(3),
        "euler_angles": np.zeros(3),
        "gate_normal": np.array([0.0, 0.0, 1.0]),
        "reprojection_error": reproj,
        "num_points_used": 4,
    }


def test_yolo_pnp_axis_remap_gate_dead_ahead():
    """OpenCV tvec=(0,0,5) → gate 5 m directly in front → bearing_h=0°,
    bearing_v=0°, range=5."""
    _install_fake_keypoint_detector([{
        "bbox": (100, 100, 200, 200),
        "confidence": 0.9,
        "keypoints": None,
        "keypoint_confs": None,
        "visibility": None,
        "pose": _pose_dict((0.0, 0.0, 5.0), reproj=0.5),
        "distance": 5.0,
    }])
    from vision.detector import YoloPnpDetector
    det = YoloPnpDetector(model_path="FAKE.pt")
    dets = det.detect(frame=object(), state=_FakeState())
    assert len(dets) == 1
    d = dets[0]
    assert d.gate_idx == -1  # unknown, caller associates
    assert abs(d.bearing_h_deg) < 0.1, f"bearing_h {d.bearing_h_deg} should be 0"
    assert abs(d.bearing_v_deg) < 0.1, f"bearing_v {d.bearing_v_deg} should be 0"
    assert abs(d.range_est - 5.0) < 0.01
    # Confidence = 0.9 * (1 - 0.5/8) ≈ 0.844
    assert 0.8 < d.confidence < 0.9
    print(f"  ✓ dead-ahead: bearing=(0,0), range=5, conf={d.confidence:.3f}")


def test_yolo_pnp_axis_remap_gate_to_right():
    """tvec=(2,0,5) → gate 5 m forward, 2 m to the right → bearing_h ≈ +21.8°"""
    _install_fake_keypoint_detector([{
        "bbox": (300, 100, 400, 200),
        "confidence": 0.8,
        "pose": _pose_dict((2.0, 0.0, 5.0), reproj=1.0),
        "distance": math.sqrt(2**2 + 5**2),
    }])
    # Reimport so fake is picked up freshly
    import importlib
    import vision.detector as d_mod
    importlib.reload(d_mod)
    det = d_mod.YoloPnpDetector(model_path="FAKE.pt")
    dets = det.detect(frame=object(), state=_FakeState())
    assert len(dets) == 1
    expected_bearing = math.degrees(math.atan2(2.0, 5.0))
    assert abs(dets[0].bearing_h_deg - expected_bearing) < 0.1
    print(f"  ✓ gate-right: bearing_h={dets[0].bearing_h_deg:.2f}° "
          f"(expect {expected_bearing:.2f}°)")


def test_yolo_pnp_rejects_behind_camera():
    """tvec=(0,0,-3) → gate behind → dropped."""
    _install_fake_keypoint_detector([{
        "bbox": (0, 0, 10, 10),
        "confidence": 0.9,
        "pose": _pose_dict((0.0, 0.0, -3.0), reproj=0.5),
        "distance": 3.0,
    }])
    import importlib, vision.detector as d_mod
    importlib.reload(d_mod)
    det = d_mod.YoloPnpDetector(model_path="FAKE.pt")
    dets = det.detect(frame=object(), state=_FakeState())
    assert dets == [], f"behind-camera gate should be rejected, got {dets}"
    print("  ✓ behind-camera gate rejected")


def test_yolo_pnp_rejects_high_reproj_error():
    """Geometrically-bad PnP (reproj > max_reproj_err_px) must be dropped."""
    _install_fake_keypoint_detector([{
        "bbox": (0, 0, 10, 10),
        "confidence": 0.9,
        "pose": _pose_dict((0.0, 0.0, 5.0), reproj=50.0),  # huge error
        "distance": 5.0,
    }])
    import importlib, vision.detector as d_mod
    importlib.reload(d_mod)
    det = d_mod.YoloPnpDetector(model_path="FAKE.pt", max_reproj_err_px=8.0)
    dets = det.detect(frame=object(), state=_FakeState())
    assert dets == [], "high-reproj detection should be dropped"
    print("  ✓ high-reproj detection rejected")


def test_yolo_pnp_sorts_nearest_first():
    """Multiple detections sorted by increasing range."""
    _install_fake_keypoint_detector([
        {"bbox": (0, 0, 10, 10), "confidence": 0.9,
         "pose": _pose_dict((0.0, 0.0, 20.0), reproj=1.0), "distance": 20.0},
        {"bbox": (0, 0, 10, 10), "confidence": 0.9,
         "pose": _pose_dict((0.0, 0.0, 5.0), reproj=1.0), "distance": 5.0},
        {"bbox": (0, 0, 10, 10), "confidence": 0.9,
         "pose": _pose_dict((0.0, 0.0, 12.0), reproj=1.0), "distance": 12.0},
    ])
    import importlib, vision.detector as d_mod
    importlib.reload(d_mod)
    det = d_mod.YoloPnpDetector(model_path="FAKE.pt")
    dets = det.detect(frame=object(), state=_FakeState())
    assert len(dets) == 3
    ranges = [d.range_est for d in dets]
    assert ranges == sorted(ranges), f"not sorted: {ranges}"
    print(f"  ✓ sorted nearest-first: ranges={['%.1f' % r for r in ranges]}")


def test_yolo_pnp_frame_none_returns_empty():
    _install_fake_keypoint_detector([{
        "bbox": (0, 0, 10, 10), "confidence": 0.9,
        "pose": _pose_dict((0.0, 0.0, 5.0)), "distance": 5.0}])
    import importlib, vision.detector as d_mod
    importlib.reload(d_mod)
    det = d_mod.YoloPnpDetector(model_path="FAKE.pt")
    assert det.detect(frame=None, state=_FakeState()) == []
    print("  ✓ None frame → empty list (no crash)")


def test_factory_dispatch():
    import importlib, vision.detector as d_mod
    importlib.reload(d_mod)
    # make_detector("virtual")
    d = d_mod.make_detector("virtual", gates=[(5.0, 0.0, 0.0)], noise_profile="clean")
    assert isinstance(d, d_mod.VirtualDetector)
    assert d.name().startswith("virtual")
    try:
        d_mod.make_detector("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("factory accepted unknown kind")
    print("  ✓ factory dispatches and rejects unknowns")


# ─────────────────────────────────────────────────────────────────────
# Round-trip: consistency between VirtualDetector and YoloPnpDetector
# ─────────────────────────────────────────────────────────────────────

def test_both_detectors_agree_on_ahead_gate():
    """Given the same physical layout — a gate 8 m directly in front —
    VirtualDetector and YoloPnpDetector should produce detections with
    bearings within 1° and ranges within 0.5 m. This validates that the
    axis remap in YoloPnpDetector is consistent with VirtualCamera's
    body-frame convention."""
    import importlib, vision.detector as d_mod

    _install_fake_keypoint_detector([{
        "bbox": (100, 100, 200, 200), "confidence": 0.9,
        "pose": _pose_dict((0.0, 0.0, 8.0), reproj=0.5), "distance": 8.0,
    }])
    importlib.reload(d_mod)

    v = d_mod.VirtualDetector(gates=[(8.0, 0.0, 0.0)], noise_profile="clean")
    y = d_mod.YoloPnpDetector(model_path="FAKE.pt")
    vd = v.detect(frame=None, state=_FakeState())[0]
    yd = y.detect(frame=object(), state=_FakeState())[0]

    assert abs(vd.bearing_h_deg - yd.bearing_h_deg) < 1.0, \
        f"bearing_h mismatch: virtual={vd.bearing_h_deg}, yolo={yd.bearing_h_deg}"
    assert abs(vd.bearing_v_deg - yd.bearing_v_deg) < 1.0
    assert abs(vd.range_est - yd.range_est) < 0.5, \
        f"range mismatch: virtual={vd.range_est}, yolo={yd.range_est}"
    print(f"  ✓ both detectors agree: Δbearing_h={abs(vd.bearing_h_deg - yd.bearing_h_deg):.2f}°, "
          f"Δrange={abs(vd.range_est - yd.range_est):.2f} m")


def main():
    tests = [
        ("virtual detector: gate ahead",             test_virtual_detector_sees_gate_ahead),
        ("virtual detector: yaw drops out of FOV",   test_virtual_detector_yaw_rotates_bearing),
        ("virtual detector: frame arg ignored",      test_virtual_detector_frame_arg_ignored),
        ("yolo-pnp: axis remap dead ahead",          test_yolo_pnp_axis_remap_gate_dead_ahead),
        ("yolo-pnp: axis remap gate to right",       test_yolo_pnp_axis_remap_gate_to_right),
        ("yolo-pnp: behind-camera rejected",         test_yolo_pnp_rejects_behind_camera),
        ("yolo-pnp: high reproj rejected",           test_yolo_pnp_rejects_high_reproj_error),
        ("yolo-pnp: sorts nearest first",            test_yolo_pnp_sorts_nearest_first),
        ("yolo-pnp: None frame → empty list",        test_yolo_pnp_frame_none_returns_empty),
        ("factory dispatch",                         test_factory_dispatch),
        ("virtual and yolo agree on ahead gate",     test_both_detectors_agree_on_ahead_gate),
    ]
    failures = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failures += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} PASSED")


if __name__ == "__main__":
    main()
