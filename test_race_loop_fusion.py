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
    # Python 3.12+ removed implicit event-loop creation in
    # asyncio.get_event_loop(). Create a fresh loop per call to keep
    # tests isolated.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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

def test_unreliable_pose_defers_seed_to_first_vision_fix():
    """Backends without SimCapability.RELIABLE_POSE (e.g. DCLSpecAdapter
    where VADR-TS-002 ships no LOCAL_POSITION_NED) must NOT have the
    filter pre-seeded from the adapter's stub pos_ned. Seeding is
    deferred to the first vision fix's auto-bootstrap path.

    Setup:
      * Wrap MockKinematicAdapter (which DOES have RELIABLE_POSE) in a
        fake adapter whose capabilities mask drops RELIABLE_POSE and
        whose get_state() returns a deliberately-wrong pos_ned to
        guarantee that any pre-seed would corrupt the filter.
      * After first tick (no vision yet), filter must remain unseeded.
      * After first vision fix, filter is seeded near truth.
    """
    print("test_unreliable_pose_defers_seed_to_first_vision_fix ...", flush=True)
    from sim.adapter import SimCapability, SimState

    gates = [(8.0, 0.0, -1.5), (16.0, 0.0, -1.5)]

    # Inner adapter — proper kinematic, drone starts at (5, 0, -1.5).
    inner = MockKinematicAdapter(
        dt=0.02, vel_tau=0.05, yaw_tau=0.10,
        auto_step=True, initial_altitude_m=1.5,
    )
    # Pre-position at NED (5, 0, -1.5) so it's NOT at the inner adapter's
    # default origin. Real DCL drones don't start at (0,0,0) either.
    inner._pos = np.array([5.0, 0.0, -1.5])

    class _UnreliablePoseAdapter:
        """Wraps `inner` but reports a confidently-wrong pos_ned and
        clears the RELIABLE_POSE capability bit. Everything else is
        delegated to the wrapped adapter."""
        capabilities = inner.capabilities & ~SimCapability.RELIABLE_POSE

        def __init__(self, inner): self._inner = inner

        def __getattr__(self, name): return getattr(self._inner, name)

        async def get_state(self):
            true = await self._inner.get_state()
            # Lie about pos/vel — return values that, if used to seed
            # the filter, would put it 99 m away from truth.
            return SimState(
                pos_ned=(99.0, 99.0, 99.0),
                vel_ned=(0.0, 0.0, 0.0),
                att_rad=true.att_rad,
                timestamp=true.timestamp,
                armed=true.armed,
                connected=true.connected,
            )

    adapter = _UnreliablePoseAdapter(inner)
    assert SimCapability.RELIABLE_POSE not in adapter.capabilities, \
        "test setup wrong: RELIABLE_POSE should be cleared"

    # Use a stub detector instead of VirtualDetector — VirtualDetector
    # would project gates from the lied (99,99,99) adapter pose and find
    # them behind the camera, returning []. We need a detector that
    # produces a controllable detection so we can verify the seed path.
    from vision.detector import GateDetection
    class _StubDetector:
        """Returns either no detection or a single detection at a
        configurable body-frame range/bearing — toggled via `enabled`."""
        def __init__(self):
            self.enabled = False
        def name(self): return "stub"
        def detect(self, frame, state):
            if not self.enabled:
                return []
            # Drone at NED (5, 0, -1.5), gate 0 at (8, 0, -1.5):
            # body-frame vector to gate is (3, 0, 0) → range 3 m, bearings 0.
            return [GateDetection(
                gate_idx=0,
                bearing_h_deg=0.0,
                bearing_v_deg=0.0,
                range_est=3.0,
                confidence=0.9,
                angular_size_deg=15.0,
                in_fov=True,
            )]

    detector = _StubDetector()
    navigator = BeliefNav(max_speed=8.0, cruise_speed=6.0)
    pf = PoseFusion()  # NOT pre-seeded

    loop = RaceLoop(
        adapter=adapter, detector=detector, navigator=navigator,
        gate_count=len(gates), command_hz=50,
        pose_fusion=pf, gates_ned=gates, vision_pos_sigma=0.20,
    )

    async def drive():
        await adapter.connect()
        await adapter.start_offboard("velocity")

        # Tick 1: detector disabled → no vision fix → filter must NOT
        # have been pre-seeded from adapter truth. (This is the whole
        # reason RELIABLE_POSE exists.)
        detector.enabled = False
        await loop.step()
        assert not pf.is_seeded, \
            f"filter was pre-seeded from adapter truth even though " \
            f"adapter lacks RELIABLE_POSE. is_seeded={pf.is_seeded}"

        # Tick 2: enable detector → first vision fix bootstraps the filter.
        detector.enabled = True
        await loop.step()
        assert pf.is_seeded, \
            "filter should have been seeded by first vision fix's " \
            "auto-bootstrap path"

        # Sanity: filter pose should be near the back-projected drone-NED
        # of the detection (drone at (5, 0, -1.5), gate at (8, 0, -1.5),
        # range 3 → drone-NED = (8 - 3, 0, -1.5) = (5, 0, -1.5)).
        p, _, _ = pf.pose()
        err = float(np.linalg.norm(p - np.array([5.0, 0.0, -1.5])))
        assert err < 0.5, \
            f"filter seeded at wrong location: pose={p}, err={err:.3f}"

        await adapter.stop_offboard()
        await adapter.disconnect()

    _run(drive())
    print("  ✓ filter unseeded after no-detection tick on unreliable backend")
    print("  ✓ filter seeded near truth on first vision fix")


def test_reliable_pose_still_seeds_from_adapter():
    """Sanity check: the RELIABLE_POSE-gated change must NOT regress
    the legacy behavior on backends that DO have reliable pose. The
    filter should be seeded from adapter truth on tick 0 as before."""
    print("test_reliable_pose_still_seeds_from_adapter ...", flush=True)
    from sim.adapter import SimCapability

    gates = [(8.0, 0.0, -1.5)]
    adapter = MockKinematicAdapter(
        dt=0.02, vel_tau=0.05, yaw_tau=0.10,
        auto_step=True, initial_altitude_m=1.5,
    )
    assert SimCapability.RELIABLE_POSE in adapter.capabilities

    detector = VirtualDetector(gates=gates, noise_profile="clean")
    navigator = BeliefNav(max_speed=8.0, cruise_speed=6.0)
    pf = PoseFusion()  # NOT pre-seeded

    loop = RaceLoop(
        adapter=adapter, detector=detector, navigator=navigator,
        gate_count=len(gates), command_hz=50,
        pose_fusion=pf, gates_ned=gates, vision_pos_sigma=0.20,
    )

    async def drive():
        await adapter.connect()
        await adapter.start_offboard("velocity")
        # Suppress detection on first tick so the only thing that could
        # seed the filter is the adapter-truth path.
        detector_orig = detector.detect
        detector.detect = lambda frame, state: []
        await loop.step()
        # On a RELIABLE_POSE adapter we expect the filter to have been
        # seeded from adapter truth even with no vision available.
        assert pf.is_seeded, \
            "filter should be seeded from adapter truth on first tick " \
            "when RELIABLE_POSE is advertised"
        detector.detect = detector_orig
        await adapter.stop_offboard()
        await adapter.disconnect()

    _run(drive())
    print("  ✓ filter seeded from adapter truth on RELIABLE_POSE backend")


def main() -> int:
    tests = [
        test_legacy_path_unchanged,
        test_fusion_path_completes_course,
        test_fused_pose_tracks_truth,
        test_constructor_rejects_fusion_without_gates,
        test_unreliable_pose_defers_seed_to_first_vision_fix,
        test_reliable_pose_still_seeds_from_adapter,
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
