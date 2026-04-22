"""Session 19h: End-to-end test — RaceLoop with PoseFusion as pose source.

Integration test across the full stack:

    MockKinematicAdapter (IMU + first-order vel tracking)
      → VirtualDetector (projects truth gates)
      → PoseFusion (IMU + backprojected vision)
      → BeliefNav (consumes FUSED pose, not adapter truth)
      → velocity commands back to the mock

Coverage:
  1. Legacy path unchanged: RaceLoop without pose_fusion runs a 2-gate
     course with MockKinematicAdapter, completes both gates.
  2. Fusion path completes the same course: PoseFusion fuses IMU +
     detection-backprojections, navigator drives from the fused pose,
     drone still passes both gates.
  3. Fused pose tracks truth throughout the run within a tight envelope.
  4. Constructor rejects pose_fusion without gates_ned (fail-loud).
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import types

import numpy as np


# Stub mavsdk so gate_belief + vision_nav can import.
if "mavsdk" not in sys.modules:
    m = types.ModuleType("mavsdk")
    o = types.ModuleType("mavsdk.offboard")

    class _System:
        def __init__(self, *a, **k): pass
    m.System = _System

    class _VNY:
        def __init__(self, vn, ve, vd, yd):
            self.north_m_s = vn
            self.east_m_s = ve
            self.down_m_s = vd
            self.yaw_deg = yd
    o.VelocityNedYaw = _VNY
    for n in ("PositionNedYaw", "Attitude"):
        setattr(o, n, type(n, (), {"__init__": lambda self, *a, **k: None}))
    o.OffboardError = type("OffboardError", (Exception,), {})
    sys.modules["mavsdk"] = m
    sys.modules["mavsdk.offboard"] = o


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src"))


from sim.mock import MockKinematicAdapter
from vision.detector import VirtualDetector
from gate_belief import BeliefNav
from race_loop import RaceLoop
from estimation import PoseFusion


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _build_stack(gates, command_hz=50, with_fusion=False, capture_truth=False):
    """Construct adapter + detector + navigator + loop; optionally with
    PoseFusion wired in. Returns (loop, adapter, pose_fusion_or_None).
    Can also return a list that the caller appends truth-vs-fused to
    when capture_truth=True."""
    dt = 1.0 / command_hz
    # vel_tau small so the mock accelerates quickly enough to complete
    # the course; still nonzero so IMU integration has meaningful accel.
    adapter = MockKinematicAdapter(
        dt=dt,
        vel_tau=0.05,
        yaw_tau=0.10,
        auto_step=True,
        initial_altitude_m=1.0,
    )
    detector = VirtualDetector(gates=gates, noise_profile="clean")
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)

    pf = None
    if with_fusion:
        pf = PoseFusion()
        pf.seed(
            p=np.array([0.0, 0.0, -1.0]),
            v=np.zeros(3),
            yaw_rad=0.0,
            bias_sigma=0.05,
        )

    loop = RaceLoop(
        adapter=adapter,
        detector=detector,
        navigator=navigator,
        gate_count=len(gates),
        command_hz=command_hz,
        associate_mode="target_idx",
        pose_fusion=pf,
        gates_ned=gates if with_fusion else None,
        vision_pos_sigma=0.20,
    )
    return loop, adapter, pf


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_legacy_path_unchanged():
    print("[1] Legacy path: RaceLoop without fusion completes a 2-gate course")
    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    loop, adapter, pf = _build_stack(gates, with_fusion=False)
    assert pf is None
    result = _run(loop.run(timeout_s=15.0, log_steps=True))
    print(f"  gates passed: {result.gates_passed}/{result.gate_count}  "
          f"time: {result.total_time_s:.2f}s")
    assert result.gates_passed == 2, \
        f"expected 2 gates, got {result.gates_passed}"
    assert not result.timeout
    print("  ✓ legacy path still completes")


def test_fusion_path_completes_course():
    print("[2] Fusion path: PoseFusion-driven RaceLoop completes same course")
    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    loop, adapter, pf = _build_stack(gates, with_fusion=True)
    assert pf is not None
    result = _run(loop.run(timeout_s=15.0, log_steps=True))
    tel = pf.telemetry
    print(f"  gates passed: {result.gates_passed}/{result.gate_count}  "
          f"time: {result.total_time_s:.2f}s")
    print(f"  fusion telemetry: imu_seen={tel.imu_samples_seen}  "
          f"vision_ok={tel.vision_fixes_accepted}  "
          f"vision_rej={tel.vision_fixes_rejected}")
    assert result.gates_passed == 2, \
        f"fusion path failed to complete: {result.gates_passed}/2"
    assert not result.timeout
    assert tel.imu_samples_seen > 50, "IMU samples weren't being fed"
    assert tel.vision_fixes_accepted > 0, "no vision fixes were accepted"
    print("  ✓ fusion-driven RaceLoop completes the course")


def test_fused_pose_tracks_truth():
    print("[3] Fused pose tracks truth throughout the run")
    gates = [(12.0, 0.0, -1.0)]
    loop, adapter, pf = _build_stack(gates, with_fusion=True)

    # Drive the loop tick-by-tick so we can sample fused vs truth.
    max_err = 0.0
    n_samples = 0
    for _ in range(400):  # up to 8 s at 50 Hz
        _run(loop.step())
        truth = np.array(adapter._pos, dtype=float)
        fused = pf.pose()[0]
        err = float(np.linalg.norm(fused - truth))
        max_err = max(max_err, err)
        n_samples += 1
        if loop.target_idx >= loop.gate_count:
            break
    final_truth = np.array(adapter._pos, dtype=float)
    final_fused = pf.pose()[0]
    final_err = float(np.linalg.norm(final_fused - final_truth))
    print(f"  samples: {n_samples}  max err: {max_err:.3f} m  "
          f"final err: {final_err:.3f} m")
    print(f"  final truth: {final_truth}")
    print(f"  final fused: {final_fused}")
    assert max_err < 0.75, f"fused pose drifted too far from truth: {max_err}"
    assert final_err < 0.25, f"final err too high: {final_err}"
    print("  ✓ fused pose stays close to truth through the run")


def test_constructor_rejects_fusion_without_gates():
    print("[4] Constructor rejects pose_fusion without gates_ned")
    gates = [(10.0, 0.0, -1.0)]
    adapter = MockKinematicAdapter(dt=0.02, auto_step=True)
    detector = VirtualDetector(gates=gates, noise_profile="clean")
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    pf = PoseFusion()
    try:
        RaceLoop(
            adapter=adapter, detector=detector, navigator=navigator,
            gate_count=1, pose_fusion=pf, gates_ned=None,
        )
    except ValueError as e:
        assert "gates_ned" in str(e), f"wrong error message: {e}"
        print(f"  ✓ raised ValueError as expected: '{e}'")
        return
    raise AssertionError("RaceLoop accepted pose_fusion with gates_ned=None")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_legacy_path_unchanged,
        test_fusion_path_completes_course,
        test_fused_pose_tracks_truth,
        test_constructor_rejects_fusion_without_gates,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        return 1
    print(f"{len(tests)}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
