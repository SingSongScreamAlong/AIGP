"""Tests for src/vision/identity.py — Session 19x.

Covers backprojection math, the IdentityTagger accept / ambiguity gates,
and an end-to-end reproducer of the S19m cascade against the RaceLoop
picker — showing that wrapping a ``gate_idx=-1``-emitting detector in
``TaggedDetector`` eliminates the failure.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

# mavsdk stub is set up by conftest.py at repo root.

from vision.detector import GateDetection  # noqa: E402
from vision.identity import (               # noqa: E402
    IdentityTagger,
    TaggedDetector,
    backproject_ned,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _det(
    gate_idx=-1,
    bearing_h_deg=0.0,
    bearing_v_deg=0.0,
    range_est=10.0,
    angular_size_deg=15.0,
    confidence=0.8,
    in_fov=True,
):
    return GateDetection(
        gate_idx=gate_idx,
        bearing_h_deg=bearing_h_deg,
        bearing_v_deg=bearing_v_deg,
        range_est=range_est,
        angular_size_deg=angular_size_deg,
        confidence=confidence,
        in_fov=in_fov,
    )


@dataclass
class _FakeState:
    pos_ned: tuple = (0.0, 0.0, 0.0)
    vel_ned: tuple = (0.0, 0.0, 0.0)
    att_rad: tuple = (0.0, 0.0, 0.0)  # roll, pitch, yaw


class _FakeInnerDetector:
    """Returns a pre-specified list of detections regardless of frame."""

    def __init__(self, detections, called_with=None):
        self._detections = detections
        self.calls = 0

    def detect(self, frame, state):
        self.calls += 1
        return list(self._detections)

    def name(self) -> str:
        return "fake_inner"


# ─────────────────────────────────────────────────────────────────────
# 1. Backprojection math
# ─────────────────────────────────────────────────────────────────────

def test_backproject_straight_ahead_at_origin():
    """Zero yaw, zero bearing, range 10 → 10 m north of drone."""
    p = backproject_ned(0.0, 0.0, 10.0, (0.0, 0.0, 0.0), 0.0)
    assert p == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)


def test_backproject_yaw_90_east():
    """Drone facing east (yaw=+90°), detection straight ahead, range 10
    → point is 10 m east of drone, not north."""
    p = backproject_ned(0.0, 0.0, 10.0, (0.0, 0.0, 0.0), math.radians(90.0))
    assert p == pytest.approx((0.0, 10.0, 0.0), abs=1e-6)


def test_backproject_translated():
    """Drone at (3, 4, -2), yaw 0, detection forward 5 m →
    world point is (3+5, 4, -2) = (8, 4, -2)."""
    p = backproject_ned(0.0, 0.0, 5.0, (3.0, 4.0, -2.0), 0.0)
    assert p == pytest.approx((8.0, 4.0, -2.0), abs=1e-6)


def test_backproject_bearing_right_yaw_zero():
    """Drone facing north, detection bearing_h=+90° (to the right) at
    range 5 → world point is 5 m east of drone."""
    p = backproject_ned(90.0, 0.0, 5.0, (0.0, 0.0, 0.0), 0.0)
    assert p == pytest.approx((0.0, 5.0, 0.0), abs=1e-6)


def test_backproject_vertical():
    """Bearing_v=+30° below horizon, 10 m range, drone at origin →
    descends 10 sin(30°) = 5 m (positive D)."""
    p = backproject_ned(0.0, 30.0, 10.0, (0.0, 0.0, -2.0), 0.0)
    horiz = 10.0 * math.cos(math.radians(30.0))
    assert p == pytest.approx((horiz, 0.0, -2.0 + 5.0), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────
# 2. IdentityTagger — accept / reject / passthrough
# ─────────────────────────────────────────────────────────────────────

def test_tag_clean_assigns_nearest_gate():
    """Gate at (10, 0, 0); detection from origin facing N backprojects
    to ~(10, 0, 0) → tagged with idx 0."""
    gates = [(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    tagger = IdentityTagger(gates)
    det = _det(bearing_h_deg=0.0, range_est=10.0)
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == 0


def test_tag_existing_idx_passes_through_unchanged():
    """A detection already carrying gate_idx≥0 must be left alone. This
    lets the tagger be wrapped around VirtualDetector defensively
    without clobbering sim-truth tags."""
    gates = [(10.0, 0.0, 0.0)]
    tagger = IdentityTagger(gates)
    det = _det(gate_idx=7, bearing_h_deg=45.0, range_est=50.0)  # way off gate 0
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == 7


def test_tag_out_of_radius_returns_minus_one():
    """Detection back-projects 10 m away from any gate → unmatched."""
    gates = [(100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
    tagger = IdentityTagger(gates, accept_radius_m=2.5)
    det = _det(bearing_h_deg=0.0, range_est=10.0)  # lands at (10, 0, 0)
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == -1


def test_tag_ambiguous_returns_minus_one():
    """Two gates within accept_radius of back-projected point, and
    within ambiguity_ratio of each other → leave as -1."""
    # Gates 0.5 m apart; detection lands between them.
    gates = [(10.0, 0.0, 0.0), (10.5, 0.0, 0.0)]
    tagger = IdentityTagger(
        gates, accept_radius_m=2.0, ambiguity_ratio=1.3,
    )
    det = _det(range_est=10.25)  # midpoint
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == -1


def test_tag_ambiguity_disabled_takes_nearest():
    """With ambiguity_ratio=0, the ratio gate is off — always take
    nearest if within radius."""
    gates = [(10.0, 0.0, 0.0), (10.5, 0.0, 0.0)]
    tagger = IdentityTagger(
        gates, accept_radius_m=2.0, ambiguity_ratio=0.0,
    )
    det = _det(range_est=10.1)  # slightly closer to gate 0
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == 0


def test_tag_empty_gate_list_returns_minus_one():
    """Degenerate but possible — no gates configured. Should not raise."""
    tagger = IdentityTagger([])
    det = _det(range_est=10.0)
    tagged = tagger.tag(det, (0.0, 0.0, 0.0), 0.0)
    assert tagged.gate_idx == -1


def test_tag_all_batch_preserves_order():
    """tag_all processes each detection independently and preserves
    input order."""
    gates = [(10.0, 0.0, 0.0), (10.0, 5.0, 0.0), (10.0, -5.0, 0.0)]
    tagger = IdentityTagger(gates, accept_radius_m=2.0)
    dets = [
        _det(bearing_h_deg=0.0, range_est=10.0),         # → gate 0
        _det(bearing_h_deg=math.degrees(math.atan2(5, 10)), range_est=math.hypot(10, 5)),  # → gate 1
        _det(bearing_h_deg=math.degrees(math.atan2(-5, 10)), range_est=math.hypot(10, 5)), # → gate 2
    ]
    tagged = tagger.tag_all(dets, (0.0, 0.0, 0.0), 0.0)
    assert [t.gate_idx for t in tagged] == [0, 1, 2]


# ─────────────────────────────────────────────────────────────────────
# 3. TaggedDetector wrapper
# ─────────────────────────────────────────────────────────────────────

def test_tagged_detector_wraps_inner_and_tags():
    """TaggedDetector.detect calls inner.detect and retags output."""
    gates = [(10.0, 0.0, 0.0)]
    inner = _FakeInnerDetector([_det(bearing_h_deg=0.0, range_est=10.0)])
    wrapped = TaggedDetector(inner, IdentityTagger(gates))
    state = _FakeState(pos_ned=(0.0, 0.0, 0.0), att_rad=(0.0, 0.0, 0.0))

    dets = wrapped.detect(frame=None, state=state)
    assert inner.calls == 1
    assert len(dets) == 1
    assert dets[0].gate_idx == 0
    # Other fields preserved
    assert dets[0].range_est == pytest.approx(10.0)


def test_tagged_detector_handles_empty():
    """Empty list in → empty list out (no tagger calls needed)."""
    gates = [(10.0, 0.0, 0.0)]
    inner = _FakeInnerDetector([])
    wrapped = TaggedDetector(inner, IdentityTagger(gates))
    state = _FakeState()
    assert wrapped.detect(None, state) == []


def test_tagged_detector_name_reports_composition():
    """name() surfaces that the detector is tagged so log scrapers can
    see it."""
    inner = _FakeInnerDetector([])
    wrapped = TaggedDetector(inner, IdentityTagger([]))
    assert "tagged" in wrapped.name()
    assert "fake_inner" in wrapped.name()


# ─────────────────────────────────────────────────────────────────────
# 4. Cascade reproducer — the whole point of this module
# ─────────────────────────────────────────────────────────────────────
#
# S19m scenario: just-passed gate still in FOV, plus the real target.
# Real YOLO emits gate_idx=-1 for both. ``_pick_detection`` in
# ``target_idx`` mode falls into its permissive branch (``gi < 0``) and
# takes the nearest-first detection — which is the just-passed gate.
# The belief model then re-anchors on a gate already behind the drone.
#
# With the tagger, gate_idx is populated, exact-match wins, and the
# picker returns the right detection.

def test_picker_cascade_with_untagged_detections():
    """Untagged gate_idx=-1 detections + picker target_idx=1 + nearest
    is gate 0 (behind) → picker returns gate 0's detection (the bug)."""
    from race_loop import RaceLoop  # noqa: E402

    # Two gates: passed (behind the drone, NE=(5,0)) and target
    # (ahead, NE=(20,0)). Drone at (8, 0, -2) facing N.
    gates = [(5.0, 0.0, -2.0), (20.0, 0.0, -2.0)]

    # Stand-in for the race-loop slice we're exercising. We don't
    # construct a full RaceLoop — just instantiate and reach in to
    # _pick_detection. RaceLoop needs adapter/detector/navigator but
    # for _pick_detection only target_idx/associate_mode/_gates_ned
    # matter. Easier to build it inline.
    picker = _make_bare_picker(
        target_idx=1, associate_mode="target_idx", gates_ned=gates,
    )

    # Detections: gate 0 at 3 m behind (still in FOV, nearest first),
    # gate 1 at 12 m ahead. Both ``gate_idx=-1`` as real YOLO would
    # emit. Nearest-first ordering puts gate 0 first.
    detections_untagged = [
        _det(gate_idx=-1, bearing_h_deg=180.0, range_est=3.0),   # gate 0 (behind)
        _det(gate_idx=-1, bearing_h_deg=0.0,   range_est=12.0),  # gate 1 (target)
    ]

    picked = picker._pick_detection(detections_untagged)
    # Bug: picker returns gate 0's detection (the just-passed one) —
    # belief would anchor on the wrong gate.
    assert picked is detections_untagged[0]


def test_picker_resolves_after_tagging():
    """Same scenario but detections run through the tagger first →
    picker's exact-match branch finds the target."""
    from race_loop import RaceLoop  # noqa: E402

    gates = [(5.0, 0.0, -2.0), (20.0, 0.0, -2.0)]
    picker = _make_bare_picker(
        target_idx=1, associate_mode="target_idx", gates_ned=gates,
    )

    # Detections back-project to near their true gates. Drone at (8, 0, -2)
    # facing N (yaw=0):
    #   - gate 0 at (5, 0, -2): relative = (-3, 0, 0) → body (forward=-3).
    #     That's "behind". We encode as bearing_h=180°, range=3. Back-
    #     projection: forward axis with yaw=0 and bearing 180° goes
    #     to negative-N direction → (8-3, 0, -2) = (5, 0, -2) ✓
    #   - gate 1 at (20, 0, -2): relative = (+12, 0, 0). bearing_h=0,
    #     range=12 → (20, 0, -2) ✓
    tagger = IdentityTagger(gates, accept_radius_m=2.0)
    drone_pos = (8.0, 0.0, -2.0)
    drone_yaw = 0.0

    raw = [
        _det(gate_idx=-1, bearing_h_deg=180.0, range_est=3.0),
        _det(gate_idx=-1, bearing_h_deg=0.0,   range_est=12.0),
    ]
    tagged = tagger.tag_all(raw, drone_pos, drone_yaw)
    assert [t.gate_idx for t in tagged] == [0, 1]

    picked = picker._pick_detection(tagged)
    assert picked is tagged[1]  # the real target — cascade eliminated


# ─────────────────────────────────────────────────────────────────────
# Test helper — bare RaceLoop-like object exposing _pick_detection
# ─────────────────────────────────────────────────────────────────────

def _make_bare_picker(target_idx, associate_mode, gates_ned):
    """Minimal shim that exposes the parts of RaceLoop needed by
    _pick_detection: target_idx, associate_mode, _gates_ned, navigator
    with a truthy gates_ned attribute.

    Avoids the full RaceLoop constructor (which needs adapter/detector/
    navigator + camera warnings) — the picker is pure logic.
    """
    from race_loop import RaceLoop  # noqa: E402

    class _NavWithGates:
        def __init__(self, gates):
            self.gates_ned = gates

    shim = RaceLoop.__new__(RaceLoop)
    shim.target_idx = target_idx
    shim.associate_mode = associate_mode
    shim._gates_ned = gates_ned
    shim.navigator = _NavWithGates(gates_ned)
    return shim


# ─────────────────────────────────────────────────────────────────────
# 5. End-to-end integration — does the tagger actually change outcomes?
# ─────────────────────────────────────────────────────────────────────
#
# The picker unit tests above prove the logic is correct in isolation.
# They don't prove:
#   (a) that the cascade pattern actually fires during a *running* race
#       (topology-dependent — the drone has to see the just-passed gate
#       within FOV simultaneously with the real target), and
#   (b) that the tagger's completion rate matches or beats the untagged
#       path across a spread of seeds.
#
# This test runs a small N-trial integration comparison: stub-untagged
# detector (VirtualDetector output with all gate_idx forced to -1) vs
# TaggedDetector(stub, IdentityTagger(gates)). It's end-to-end — full
# RaceRunner lifecycle, mock_kinematic physics, fast-time. The
# assertion is weak on purpose: tagged must not regress. If it matches
# or beats, either the cascade doesn't fire in practice (V5.1
# robustness swallows it) OR the tagger closed it. Both are useful
# data points; the print summary surfaces which.

def _stub_yolo_of(inner):
    """Wrap a Detector so every returned detection has gate_idx=-1.

    Models what ``YoloPnpDetector`` does today — real vision knows
    "there's a gate", not "this is gate #3".
    """
    from dataclasses import replace as _replace

    class _StubYolo:
        def detect(self, frame, state):
            dets = inner.detect(frame, state)
            return [_replace(d, gate_idx=-1) for d in dets]

        def name(self):
            return f"stub_yolo[{inner.name()}]"

    return _StubYolo()


def _run_race_once(detector, gates, timeout_s=45.0, command_hz=50):
    """Sync wrapper around RaceRunner.fly — returns (completed, gates_passed, time_s).

    Uses a freshly-created event loop scoped to this call, rather than
    ``asyncio.run`` which permanently closes the default loop and
    breaks every subsequent test that does
    ``asyncio.get_event_loop().run_until_complete``.
    """
    import asyncio as _asyncio
    from race.runner import RaceRunner
    from sim.mock import MockKinematicAdapter
    from gate_belief import BeliefNav

    adapter = MockKinematicAdapter(
        dt=1.0 / command_hz,
        vel_tau=0.05, yaw_tau=0.10,
        auto_step=True, initial_altitude_m=1.0,
    )
    navigator = BeliefNav(max_speed=12.0, cruise_speed=9.0)
    runner = RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=gates, command_hz=command_hz, takeoff_altitude_m=2.0,
    )
    loop = _asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            runner.fly(timeout_s=timeout_s, realtime=False, log_steps=False)
        )
    finally:
        loop.close()
    return bool(result.completed), int(result.run.gates_passed), float(result.run.total_time_s)


def test_tagger_integration_does_not_regress_completion():
    """Run N trials of technical/mild with stub-untagged vs tagged;
    assert tagged ≥ untagged completion. Captures which regime we're in
    via printed output.

    Design choice: technical + mild (not sprint + harsh) because
    sprint/harsh is the regime where V5.1 has the thinnest timing
    margin; any detector-side regression there gets blamed on the
    planner. Technical/mild is well within the control margin, so any
    failures can be attributed to the perception path under test.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).resolve().parent
    for _p in (str(_HERE), str(_HERE / "src")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from courses import get_course
    from vision.detector import VirtualDetector
    from vision.identity import IdentityTagger, TaggedDetector

    gates = get_course("technical")
    N = 5  # per condition; with fast-time each trial is ~0.02s wall

    untagged_done = 0
    tagged_done = 0
    for i in range(N):
        seed = 4000 + i  # fresh seed range, separate from soak runs

        inner_u = VirtualDetector(
            gates=gates, noise_profile="mild", seed=seed,
        )
        stub_u = _stub_yolo_of(inner_u)

        inner_t = VirtualDetector(
            gates=gates, noise_profile="mild", seed=seed,
        )
        stub_t = _stub_yolo_of(inner_t)
        tagged = TaggedDetector(stub_t, IdentityTagger(gates))

        u_ok, _, _ = _run_race_once(stub_u, gates)
        t_ok, _, _ = _run_race_once(tagged, gates)

        untagged_done += int(u_ok)
        tagged_done += int(t_ok)

    # Weak assertion: the fix must not regress.
    assert tagged_done >= untagged_done, (
        f"tagged ({tagged_done}/{N}) regressed below untagged ({untagged_done}/{N})"
    )
    # Informative: print the regime so the test log captures which
    # case we're in. A delta > 0 means the tagger actually rescues
    # something; a tie means V5.1 handles the cascade implicitly.
    print(
        f"\n  stub-untagged: {untagged_done}/{N}  "
        f"tagged: {tagged_done}/{N}  "
        f"delta: {tagged_done - untagged_done}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
