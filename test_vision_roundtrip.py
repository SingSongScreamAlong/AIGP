"""End-to-end vision-path round-trip tests — Session 19f.

Problem this file closes: `YoloPnpDetector` has a half-dozen subtle
assumptions baked in — the OpenCV PnP coordinate convention, the
axis remap cv X/Y/Z → body forward/right/down, the reprojection-error
rejection, and the behind-camera clamp. Each piece has its own
unit test in `test_detector.py`, but the full chain — synthetic
camera → PnP → detector → belief → navigator — has never been
exercised as one piece in the sandbox. A bug along that chain would
only show up on hardware with a trained YOLO model.

What this test does: it stands in a `SyntheticGateKeypointDetector`
that has the same interface as the real `GateKeypointDetector`, but
generates 4 keypoints by projecting known gate corners through a
pinhole camera using OpenCV's own `projectPoints`. Those keypoints
go through the real `GatePoseEstimator` and the real
`YoloPnpDetector.detect()`. If axis conventions are wrong, if PnP
returns tvec in an unexpected frame, if reprojection error
explodes — this test will fail before your GPU comes back online.

Layers tested:
  1. OpenCV project→solve round-trip: tvec round-trips exactly.
  2. Pinhole projection → GatePoseEstimator tvec matches our assumed
     (X_right, Y_down, Z_forward) convention.
  3. YoloPnpDetector w/ synthetic _impl returns GateDetection whose
     bearing/range match the ground-truth drone-to-gate geometry.
  4. Full RaceLoop runs to completion with this detector through a
     simple 2-gate course.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

try:
    import cv2
except ImportError:
    print("OpenCV not installed — skipping vision round-trip tests")
    sys.exit(0)

# Stub mavsdk so gate_belief (which currently imports mavsdk.offboard at
# module scope) can load on a machine without PX4 tooling installed. Same
# trick run_race.py uses. Only applied when mavsdk isn't already present.
try:
    import mavsdk  # noqa: F401
except Exception:
    import types
    _m = types.ModuleType("mavsdk")
    _o = types.ModuleType("mavsdk.offboard")

    class _Stub:
        def __init__(self, *a, **k): pass
    _m.System = _Stub

    class _VNY:
        def __init__(self, vn, ve, vd, yd):
            self.north_m_s, self.east_m_s = vn, ve
            self.down_m_s, self.yaw_deg = vd, yd
    _o.VelocityNedYaw = _VNY
    for _n in ("PositionNedYaw", "Attitude"):
        setattr(_o, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    _o.OffboardError = type("OffboardError", (Exception,), {})
    sys.modules["mavsdk"] = _m
    sys.modules["mavsdk.offboard"] = _o

from vision.pnp_pose import GatePoseEstimator


# ─────────────────────────────────────────────────────────────────────
# Camera intrinsics used across all tests (matches YoloPnpDetector defaults)
# ─────────────────────────────────────────────────────────────────────

IMG_W = 640
IMG_H = 480
FOV_DEG = 90.0
# Same formula GatePoseEstimator uses: fy = (img_h/2) / tan(fov/2), fx=fy.
FY = (IMG_H / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))
FX = FY
CX = IMG_W / 2.0
CY = IMG_H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros(5)

# Gate geometry — matches pnp_pose.GATE_CORNERS_3D (0.4 m edge)
GATE_HALF = 0.2
# Local-frame corners of a 0.4 m square gate, TL/TR/BR/BL order
GATE_CORNERS_LOCAL = np.array([
    [-GATE_HALF, +GATE_HALF, 0.0],   # TL
    [+GATE_HALF, +GATE_HALF, 0.0],   # TR
    [+GATE_HALF, -GATE_HALF, 0.0],   # BR
    [-GATE_HALF, -GATE_HALF, 0.0],   # BL
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# Helpers: body-frame and axis-swap math
# ─────────────────────────────────────────────────────────────────────

def world_to_body_xy(dn: float, de: float, dd: float, yaw_rad: float) -> Tuple[float, float, float]:
    """NED offset (drone → gate) expressed in body frame (X fwd, Y right, Z down)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    body_x = dn * cy + de * sy
    body_y = -dn * sy + de * cy
    body_z = dd
    return body_x, body_y, body_z


def body_to_cam(body_x: float, body_y: float, body_z: float) -> np.ndarray:
    """Body (X fwd, Y right, Z down) → OpenCV camera (X right, Y down, Z fwd)."""
    return np.array([body_y, body_z, body_x], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# SyntheticGateKeypointDetector
# ─────────────────────────────────────────────────────────────────────

class SyntheticGateKeypointDetector:
    """Stand-in for the real GateKeypointDetector used in tests.

    Projects each in-FOV gate's 4 corners to pixel coords via a
    pinhole camera, then runs the SAME GatePoseEstimator the real
    pipeline uses. This exercises the full PnP codepath without
    needing a trained YOLO model.

    Gate orientation: each gate is assumed to face the drone's
    current position (a simplification that's valid for testing the
    pipeline; real course gates have fixed orientations but our
    detector doesn't know or care).
    """

    def __init__(
        self,
        gates_ned: List[tuple],
        state_fn: Callable[[], Tuple[np.ndarray, float]],
        gate_size_m: float = GATE_HALF * 2,
        pixel_noise_sigma: float = 0.0,
    ):
        self.gates_ned = [np.asarray(g, dtype=float) for g in gates_ned]
        self.state_fn = state_fn
        # Match the real GatePoseEstimator
        self.pose_estimator = GatePoseEstimator(
            img_w=IMG_W, img_h=IMG_H, fov_deg=FOV_DEG
        )
        self.gate_size = gate_size_m
        self.pixel_noise = pixel_noise_sigma
        self._rng = np.random.default_rng(0)

    def detect(self, frame, estimate_pose=True):
        pos_ned, yaw_rad = self.state_fn()
        dets = []
        for gi, g in enumerate(self.gates_ned):
            dn = g[0] - pos_ned[0]
            de = g[1] - pos_ned[1]
            dd = g[2] - pos_ned[2]
            bx, by, bz = world_to_body_xy(dn, de, dd, yaw_rad)
            if bx < 0.3:
                continue  # behind camera or too close
            rng_m = math.sqrt(bx*bx + by*by + bz*bz)
            if rng_m > 30.0:
                continue

            # Project 4 gate corners. The gate faces the drone, so its
            # local frame's Z axis points from gate center back to camera
            # (= −gate_center_in_camera_coords, normalized).
            gate_cam = body_to_cam(bx, by, bz)  # gate center in OpenCV cam
            z_axis = -gate_cam / np.linalg.norm(gate_cam)  # gate→camera
            # world-up in camera = body-up (body -z) = (0, 0, -1)_body
            # → cam: body_to_cam(0, 0, -1) = (0, -1, 0). So up_in_cam = (0, -1, 0).
            up_cam = np.array([0, -1, 0], dtype=np.float64)
            # Gate local Y axis: project up_cam perpendicular to z_axis
            y_axis = up_cam - np.dot(up_cam, z_axis) * z_axis
            y_axis /= max(np.linalg.norm(y_axis), 1e-9)
            # Gate local X = Y × Z (right-handed)
            x_axis = np.cross(y_axis, z_axis)
            R_local_to_cam = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3x3

            # Transform corners to cam frame
            corners_cam = (R_local_to_cam @ GATE_CORNERS_LOCAL.T).T + gate_cam

            # Project via cv2 (identity extrinsic — already in cam frame)
            pixels, _ = cv2.projectPoints(
                corners_cam.reshape(-1, 1, 3),
                rvec=np.zeros(3), tvec=np.zeros(3),
                cameraMatrix=K, distCoeffs=DIST,
            )
            pixels = pixels.reshape(-1, 2)
            # Optional pixel noise
            if self.pixel_noise > 0:
                pixels = pixels + self._rng.normal(0, self.pixel_noise, pixels.shape)
            # FOV check — all 4 corners must be inside image
            if not np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < IMG_W)
                          & (pixels[:, 1] >= 0) & (pixels[:, 1] < IMG_H)):
                continue

            visibility = np.array([2, 2, 2, 2])
            pose = self.pose_estimator.estimate_pose(pixels, visibility=visibility)
            if not pose.get("success"):
                continue
            dets.append({
                "bbox": (0, 0, IMG_W, IMG_H),
                "confidence": 0.95,
                "keypoints": pixels,
                "keypoint_confs": np.array([0.9, 0.9, 0.9, 0.9]),
                "visibility": visibility,
                "pose": pose,
                "distance": pose["distance"],
            })
        dets.sort(key=lambda d: d["distance"])
        return dets


# ─────────────────────────────────────────────────────────────────────
# Test 1 — OpenCV projectPoints/solvePnP round-trip sanity
# ─────────────────────────────────────────────────────────────────────

def test_opencv_roundtrip():
    print("[1] OpenCV projectPoints→solvePnP round-trip identity")
    # Gate 5 m ahead, slightly right and above (cam frame: X right, Y down, Z fwd)
    # Place gate center at camera (1, -0.5, 5): 1 m right, 0.5 m up, 5 m forward.
    true_tvec = np.array([1.0, -0.5, 5.0])
    true_rvec = np.zeros(3)  # gate plane aligned with cam plane

    # Project corners
    pixels, _ = cv2.projectPoints(
        GATE_CORNERS_LOCAL.reshape(-1, 1, 3),
        rvec=true_rvec, tvec=true_tvec,
        cameraMatrix=K, distCoeffs=DIST,
    )
    pixels = pixels.reshape(-1, 2)

    # Solve PnP using SOLVEPNP_IPPE_SQUARE (4-corner planar)
    ok, rvec, tvec = cv2.solvePnP(
        GATE_CORNERS_LOCAL.reshape(-1, 1, 3),
        pixels.reshape(-1, 1, 2),
        K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    assert ok
    tvec = tvec.flatten()
    err = np.linalg.norm(tvec - true_tvec)
    print(f"  true tvec   = {true_tvec}")
    print(f"  recovered   = {tvec}")
    print(f"  err         = {err:.2e} m")
    assert err < 1e-6, f"round-trip error {err} too large"
    print("  ✓ OpenCV project/solve round-trips to <1 µm")


# ─────────────────────────────────────────────────────────────────────
# Test 2 — GatePoseEstimator tvec lives in OpenCV convention
# ─────────────────────────────────────────────────────────────────────

def test_posenet_tvec_convention():
    print("[2] GatePoseEstimator tvec follows (X_right, Y_down, Z_forward)")
    # Put the drone at origin with yaw 0, gate 5 m ahead.
    # body_x=5, body_y=0, body_z=0 → cam=(0, 0, 5)
    drone_state = (np.zeros(3), 0.0)
    syn = SyntheticGateKeypointDetector(
        gates_ned=[(5.0, 0.0, 0.0)],
        state_fn=lambda: drone_state,
    )
    dets = syn.detect(frame=None)
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    tvec = dets[0]["pose"]["tvec"]
    print(f"  recovered tvec = {tvec}")
    # Expected (X_right, Y_down, Z_forward): (0, 0, 5)
    assert abs(tvec[0]) < 0.01, f"X should be ~0, got {tvec[0]}"
    assert abs(tvec[1]) < 0.01, f"Y should be ~0, got {tvec[1]}"
    assert abs(tvec[2] - 5.0) < 0.01, f"Z should be ~5, got {tvec[2]}"
    print("  ✓ dead-ahead gate 5 m gives tvec≈(0, 0, 5)")

    # Now shift drone right: gate appears to the LEFT of the drone (-Y body).
    # body_y = -3 → cam X = -3.
    drone_state2 = (np.array([0.0, 3.0, 0.0]), 0.0)  # drone at +3E
    syn.state_fn = lambda: drone_state2
    dets = syn.detect(frame=None)
    assert len(dets) == 1
    tvec = dets[0]["pose"]["tvec"]
    print(f"  drone at +3E → tvec = {tvec}")
    assert abs(tvec[0] - (-3.0)) < 0.05, f"cam_X expected -3, got {tvec[0]}"
    assert abs(tvec[1]) < 0.05, f"cam_Y expected 0, got {tvec[1]}"
    assert abs(tvec[2] - 5.0) < 0.05, f"cam_Z expected 5, got {tvec[2]}"
    print("  ✓ drone +3 E: tvec cam-X = -3 (gate appears to the left)")


# ─────────────────────────────────────────────────────────────────────
# Test 3 — YoloPnpDetector with synthetic _impl returns correct bearings
# ─────────────────────────────────────────────────────────────────────

def test_yolopnp_with_synthetic_impl():
    print("[3] YoloPnpDetector axis remap through synthetic pipeline")
    # Place gate 5 m north, 3 m east. Drone at origin, yaw=0.
    gate_ned = (5.0, 3.0, 0.0)
    # body: bx=5, by=3, bz=0 → expected bearing = atan2(3, 5) ≈ 30.96°
    expected_bearing_deg = math.degrees(math.atan2(3.0, 5.0))
    expected_range = math.sqrt(5*5 + 3*3)
    drone_state = (np.zeros(3), 0.0)

    # Build a YoloPnpDetector, replace its _impl with synthetic.
    from vision.detector import YoloPnpDetector
    det = object.__new__(YoloPnpDetector)  # skip __init__ (avoids YOLO import)
    det._img_w = IMG_W
    det._img_h = IMG_H
    det._fov_deg = FOV_DEG
    det._max_reproj = 8.0
    det._model_path = "synthetic://test"
    det._impl = SyntheticGateKeypointDetector(
        gates_ned=[gate_ned],
        state_fn=lambda: drone_state,
    )

    out = det.detect(frame=np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), state=None)
    assert len(out) == 1, f"expected 1 GateDetection, got {len(out)}"
    gd = out[0]
    print(f"  bearing_h = {gd.bearing_h_deg:.2f}° (expected {expected_bearing_deg:.2f}°)")
    print(f"  range     = {gd.range_est:.2f} m (expected {expected_range:.2f} m)")
    print(f"  conf      = {gd.confidence:.3f}")
    assert abs(gd.bearing_h_deg - expected_bearing_deg) < 0.5
    assert abs(gd.range_est - expected_range) < 0.05
    assert gd.confidence > 0.8  # geometrically clean → high conf
    print("  ✓ YoloPnpDetector.detect() produces correct bearing/range end-to-end")


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Full RaceLoop completes through synthetic vision path
# ─────────────────────────────────────────────────────────────────────

def test_raceloop_completes_via_synthetic_vision():
    print("[4] RaceLoop completes a 2-gate course through the synthetic vision path")
    from race_loop import RaceLoop
    from vision.detector import YoloPnpDetector
    from gate_belief import BeliefNav

    # Tiny 2-gate course straight ahead
    gates = [(5.0, 0.0, -1.0), (10.0, 0.0, -1.0)]

    # MockAdapter (same shape as the one in test_race_loop.py)
    from sim.adapter import SimState, SimCapability, SimInfo

    class MockAdapter:
        capabilities = (SimCapability.VELOCITY_NED | SimCapability.POSITION_NED
                        | SimCapability.ARM_ACTION | SimCapability.CAMERA_RGB)

        def __init__(self, dt=0.02):
            self._pos = [0.0, 0.0, -1.0]
            self._vel = [0.0, 0.0, 0.0]
            self._yaw = 0.0
            self._dt = dt

        async def connect(self): pass
        async def disconnect(self): pass
        async def reset(self): pass

        async def get_state(self):
            return SimState(
                pos_ned=tuple(self._pos), vel_ned=tuple(self._vel),
                att_rad=(0.0, 0.0, self._yaw), timestamp=time.time(),
                armed=True, connected=True,
            )

        async def get_camera_frame(self):
            # Synthetic detector doesn't read the frame pixels, but returning
            # something with the right shape keeps the contract honest.
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        async def send_velocity_ned(self, vn, ve, vd, yaw_deg):
            self._vel = [vn, ve, vd]
            self._pos[0] += vn * self._dt
            self._pos[1] += ve * self._dt
            self._pos[2] += vd * self._dt
            desired = math.radians(yaw_deg)
            delta = (desired - self._yaw + math.pi) % (2*math.pi) - math.pi
            self._yaw += delta * min(1.0, self._dt / 0.15)

        async def send_position_ned(self, *a, **k): pass
        async def send_attitude(self, *a, **k): pass
        async def arm(self): pass
        async def disarm(self): pass
        async def takeoff(self, alt): self._pos[2] = -alt
        async def land(self): pass
        async def start_offboard(self, initial_mode="velocity"): pass
        async def stop_offboard(self): pass

        def info(self):
            return SimInfo(backend="mock+synth", capabilities=self.capabilities,
                           notes="kinematic-only; synthetic-vision test")

    adapter = MockAdapter()

    # YoloPnpDetector with synthetic _impl tied to the mock's state
    def state_fn():
        return np.array(adapter._pos), adapter._yaw

    det = object.__new__(YoloPnpDetector)
    det._img_w = IMG_W; det._img_h = IMG_H; det._fov_deg = FOV_DEG
    det._max_reproj = 8.0; det._model_path = "synthetic://test"
    det._impl = SyntheticGateKeypointDetector(gates_ned=gates, state_fn=state_fn)

    navigator = BeliefNav(max_speed=6.0, cruise_speed=4.0)

    loop = RaceLoop(
        adapter=adapter, detector=det, navigator=navigator,
        gate_count=len(gates), command_hz=50,
        associate_mode="nearest",
    )

    result = asyncio.run(loop.run(timeout_s=30.0))
    print(f"  Gates passed : {result.gates_passed}/{result.gate_count}")
    print(f"  Total time   : {result.total_time_s:.2f} s")
    print(f"  Completed    : {result.completed}")
    assert result.completed, f"race did not complete: {result}"
    assert result.gates_passed == 2
    print("  ✓ 2-gate course completes through YoloPnpDetector(synthetic)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_opencv_roundtrip,
        test_posenet_tvec_convention,
        test_yolopnp_with_synthetic_impl,
        test_raceloop_completes_via_synthetic_vision,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        return 1
    print(f"{len(tests)}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
