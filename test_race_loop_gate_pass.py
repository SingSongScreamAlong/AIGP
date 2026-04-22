"""Gate-pass detector tests — Session 19k + 19l.

Exercises the position-based + range-based combined gate-pass detector
in src/race_loop.py. Four classes of test:

  1. **Legacy compatibility** — when gates_ned is None, the detector
     must fall through to the old range-only behaviour so callers
     that didn't opt in still work.
  2. **Position-based backstop** — when detection drops at close
     range (the common YOLO failure mode where the gate fills frame
     and the detector can't localise it), the position signal must
     still fire the pass.
  3. **False-pass immunity** — the detector must NOT fire when the
     drone approaches the gate but turns away before crossing, and
     must NOT double-fire on the same gate after one pass.
  4. **Distractor-spoofing contract (S19l)** — honest docs for what
     this layer does and does not defend against. The range signal
     is vulnerable to decoys at short camera range on the drone's
     path; the position signal rejects decoys far from the drone's
     physical path. Full Round-2 defense is detector-layer (YOLO)
     work, not race-loop work.

These are unit-level tests on `_check_gate_pass` / `_check_gate_pass_position`
directly; integration coverage (full race loop with fusion) lives in
test_race_loop_fusion.py.

Run standalone:
    python test_race_loop_gate_pass.py
"""

from __future__ import annotations

import os
import sys
import types


# Stub mavsdk so gate_belief's imports don't blow up.
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


from race_loop import RaceLoop, TrackerState  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Test doubles — enough shape for RaceLoop to construct, but we
# poke _check_gate_pass directly so we don't need async.
# ─────────────────────────────────────────────────────────────────────

class _StubAdapter:
    capabilities = 0

class _StubDetector:
    def detect(self, *a, **k): return []
    def name(self): return "stub"

class _StubNavigator:
    def __init__(self):
        self.gate_passes = 0
    def on_gate_passed(self, *a, **k):
        self.gate_passes += 1


def _build(gates_ned=None, gate_count: int = 2) -> RaceLoop:
    return RaceLoop(
        adapter=_StubAdapter(),
        detector=_StubDetector(),
        navigator=_StubNavigator(),
        gate_count=gate_count,
        command_hz=50,
        associate_mode="nearest",
        pose_fusion=None,
        gates_ned=gates_ned,
    )


def _tracker(detected: bool, range_est: float = 99.0) -> TrackerState:
    return TrackerState(
        target_idx=0,
        detected=detected,
        bearing_h=0.0,
        bearing_v=0.0,
        range_est=range_est,
        confidence=0.9 if detected else 0.0,
        frames_since_seen=0 if detected else 99,
        search_mode=not detected,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Legacy compatibility: gates_ned=None uses range-only dispatch
# ─────────────────────────────────────────────────────────────────────

def test_legacy_range_based_path_when_gates_ned_is_none():
    """With gates_ned=None, the loop must use the old range-crossing
    heuristic. Position-based doesn't apply — there are no NED gates
    to compare against — and the detector should behave exactly as
    it did pre-S19k.
    """
    loop = _build(gates_ned=None)
    assert loop._gates_ned is None, "setup guard"

    # No detection → no pass.
    assert loop._check_gate_pass(_tracker(detected=False)) is False
    assert loop.target_idx == 0

    # Detected but above PASSAGE_RANGE → no pass.
    assert loop._check_gate_pass(_tracker(detected=True, range_est=3.0)) is False
    assert loop.target_idx == 0

    # Detected, range below PASSAGE_RANGE → fires.
    assert loop._check_gate_pass(_tracker(detected=True, range_est=1.0)) is True
    assert loop.target_idx == 1
    assert loop.navigator.gate_passes == 1

    print("  ✓ legacy range-based path behaves as pre-S19k when gates_ned is None")


def test_gates_ned_allowed_without_pose_fusion():
    """Post-S19k: supplying gates_ned alone (no pose_fusion) should
    enable the position-based detector, not raise. Pre-S19k this
    combination raised ValueError because gates_ned was coupled to
    fusion-only backprojection.
    """
    # Shouldn't raise.
    loop = _build(gates_ned=[(10.0, 0.0, -1.0), (20.0, 0.0, -1.0)])
    assert loop._gates_ned is not None
    assert len(loop._gates_ned) == 2
    print("  ✓ gates_ned can now be provided without pose_fusion")


# ─────────────────────────────────────────────────────────────────────
# 2. Position signal fires on flyby (even without a detection)
# ─────────────────────────────────────────────────────────────────────

def test_position_signal_fires_on_close_approach_without_detection():
    """Drone passes within PASSAGE_RADIUS of target gate. Detection is
    dropped (range_est=99, detected=False). Position-based must fire
    on the local minimum — this is the YOLO close-range failure mode
    the position detector exists for.
    """
    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates)

    # Drone approaches gate 0 along +N axis, passes, recedes.
    # No detections the whole time (simulates YOLO collapse at close range).
    path = [
        [8.0, 0.0, -1.0],   # dist ≈ 2.00 → inside radius
        [9.2, 0.0, -1.0],   # dist ≈ 0.80 → getting closer
        [10.1, 0.0, -1.0],  # dist ≈ 0.10 → closest approach
        [11.0, 0.0, -1.0],  # dist ≈ 1.00 → receding, local min detected
    ]
    fired_at = None
    for i, pos in enumerate(path):
        fired = loop._check_gate_pass_position(_tracker(detected=False), pos)
        if fired:
            fired_at = i
            break
    assert fired_at is not None, \
        f"position detector never fired on a clean flyby, target_idx={loop.target_idx}"
    # Must fire after closest approach (tick 3, where distance grew).
    assert fired_at == 3, f"expected fire at tick 3 (receding), got {fired_at}"
    assert loop.target_idx == 1
    assert loop.navigator.gate_passes == 1
    print(f"  ✓ position signal fired on flyby (tick {fired_at}, no detection)")


def test_range_signal_still_fires_in_normal_pass():
    """In the common case where detection is healthy AND the drone is
    close, the range-based signal must fire the pass on the first tick
    that crosses PASSAGE_RANGE — same timing as pre-S19k, so fusion
    races have the same behaviour they used to.
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)
    # Drone near gate, detection healthy, range < 2.5 m.
    fired = loop._check_gate_pass_position(
        _tracker(detected=True, range_est=2.0),
        [8.0, 0.0, -1.0],   # fused dist ≈ 2.0 m — inside sanity radius
    )
    assert fired is True
    assert loop.target_idx == 1
    print("  ✓ range signal still fires when detection is healthy")


# ─────────────────────────────────────────────────────────────────────
# 3. No false positives
# ─────────────────────────────────────────────────────────────────────

def test_no_fire_on_approach_without_crossing():
    """Drone approaches the gate, gets within PASSAGE_RADIUS, then
    turns away (distance starts growing) but detection was never
    below PASSAGE_RANGE. The position signal WILL fire on this (it's
    a local minimum within radius) — which is the intended behaviour
    for the missed-detection case. This test instead verifies that
    distances that stay OUTSIDE the radius never fire, even if the
    drone turns around.
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)
    # Drone stays 3 m+ away from the gate throughout.
    path = [
        [5.0, 0.0, -1.0],   # dist 5.0
        [6.0, 0.0, -1.0],   # dist 4.0
        [7.0, 0.0, -1.0],   # dist 3.0 → closest, outside radius
        [6.0, 0.0, -1.0],   # dist 4.0 → receding
        [5.0, 0.0, -1.0],   # dist 5.0
    ]
    for pos in path:
        fired = loop._check_gate_pass_position(_tracker(detected=False), pos)
        assert fired is False, \
            f"position detector fired at pos={pos} even though drone " \
            f"never entered PASSAGE_RADIUS={RaceLoop.PASSAGE_RADIUS}"
    assert loop.target_idx == 0
    print("  ✓ no fire when drone stays outside PASSAGE_RADIUS")


def test_no_double_fire_on_same_gate():
    """Once gate 0 passes, subsequent ticks near gate 0 must not
    advance target_idx again. The detector should be latched onto
    gate 1 now, even if the drone lingers near gate 0's NED.
    """
    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=2)
    # Fire gate 0 pass.
    loop._check_gate_pass_position(_tracker(detected=False), [9.0, 0.0, -1.0])
    loop._check_gate_pass_position(_tracker(detected=False), [9.9, 0.0, -1.0])
    fired = loop._check_gate_pass_position(_tracker(detected=False), [10.5, 0.0, -1.0])
    assert fired, "gate 0 pass must fire on receding"
    assert loop.target_idx == 1, "target should advance after first pass"

    # Now linger near gate 0 — must NOT advance to gate 2. Distance
    # measurements are now versus gate 1 at N=25, so we're 15 m away —
    # well outside PASSAGE_RADIUS, no fire.
    loop._check_gate_pass_position(_tracker(detected=False), [11.0, 0.0, -1.0])
    fired2 = loop._check_gate_pass_position(_tracker(detected=False), [10.0, 0.0, -1.0])
    assert fired2 is False, "must not re-fire on gate 0's vicinity"
    assert loop.target_idx == 1, "target_idx must stay at 1"
    print("  ✓ no double-fire; detector latches onto next gate after pass")


def test_target_idx_clamped_past_last_gate():
    """Once all gates are passed, the detector must be idempotent
    — no crashes, no out-of-range gate lookups, target_idx stays
    at gate_count.
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)
    loop._check_gate_pass_position(_tracker(detected=False), [9.0, 0.0, -1.0])
    loop._check_gate_pass_position(_tracker(detected=False), [9.9, 0.0, -1.0])
    loop._check_gate_pass_position(_tracker(detected=False), [10.5, 0.0, -1.0])
    assert loop.target_idx == 1

    # Several more ticks — must not IndexError on gates_ned[1].
    for _ in range(5):
        fired = loop._check_gate_pass_position(
            _tracker(detected=True, range_est=1.0),
            [10.2, 0.0, -1.0],
        )
        assert fired is False
        assert loop.target_idx == 1
    print("  ✓ idempotent after all gates passed")


# ─────────────────────────────────────────────────────────────────────
# 4. Distractor/decoy behaviour
# ─────────────────────────────────────────────────────────────────────

def test_range_signal_still_passes_when_pose_drifts():
    """Once fused pose drifts far from truth, the range signal should
    keep firing (we don't want a drifted pose to permanently stall
    the race). S19l ruled out sanity-gating the range signal on
    position (the AND-gate experiment): it caused a stall cascade
    where a drone with drifted fused-pose would orbit the target
    gate, starve the fusion of vision updates, drift further, and
    never pass. The un-gated OR-path is the lesser evil — the
    distractor defense moves to the detector (YOLO) layer instead.
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)
    # Fused pose claims we're 50 m away, but detection says range=1 m.
    # If we hard-gated the range signal on fused position sanity, we'd
    # block this — and in-flight that blockage compounds (see S19l log).
    fired = loop._check_gate_pass_position(
        _tracker(detected=True, range_est=1.0),
        [60.0, 0.0, -1.0],
    )
    assert fired is True
    print("  ✓ range signal fires even when fused pose has drifted "
          "(prevents permanent stall)")


# ─────────────────────────────────────────────────────────────────────
# 5. Honest distractor-spoofing boundary documentation (S19l)
#
# These two tests encode the actual scope of the position-based
# gate-pass detector's distractor defense. The S19k project log
# over-claimed — see the S19l correction entry.
# ─────────────────────────────────────────────────────────────────────

def test_range_signal_vulnerable_to_near_distractor():
    """Failure-mode contract: the range signal is SPOOFABLE by a
    decoy gate that YOLO detects at short camera range, even when
    the drone is physically far from the real target gate's NED.

    Scenario: drone at (-10, 0, -1), target gate at (10, 0, -1) —
    20 m away. Detector returns a detection with range_est=1.0 m
    (a decoy in the drone's camera cone). S19l verified this fires
    the pass. This is NOT the behaviour we'd want in Round 2, but
    it's the behaviour we accept at this layer — sanity-gating on
    pose caused stall cascades. The defense moves to the detector
    (distractor-augmented YOLO training).
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)
    fired = loop._check_gate_pass_position(
        _tracker(detected=True, range_est=1.0),
        [-10.0, 0.0, -1.0],   # 20 m from target gate — clearly not there
    )
    assert fired is True, (
        "range signal is intentionally ungated; if this ever starts "
        "failing, check race_loop.py _check_gate_pass_position and "
        "re-read the S19l project-log entry before 'fixing' it"
    )
    assert loop.target_idx == 1
    print("  ✓ range signal fires on near-distractor spoof "
          "(documented limitation — see S19l)")


def test_position_signal_rejects_distant_distractor():
    """Protection contract: the position signal DOES reject decoys
    the drone never physically approaches. Drone loops far from the
    target gate (no NED position ever within PASSAGE_RADIUS). Even
    if a decoy detector fires healthy detections the whole time,
    the position signal must not fire — we never physically hit the
    local minimum at the real target gate.

    This is the defense the position detector DOES provide: decoy
    gates placed well off the drone's physical path can't sneak in
    a pass. Decoys placed ON the drone's path near the target can
    (that's range-signal territory — see the vulnerability test
    above).
    """
    gates = [(10.0, 0.0, -1.0)]
    loop = _build(gates_ned=gates, gate_count=1)

    # Drone circles at |N|=30 m — target gate is at N=10, so drone
    # is ≥ 20 m away throughout. Decoy "detections" report short
    # range but never cross PASSAGE_RANGE (simulate the range
    # signal being quiescent so we can observe position-only).
    path = [
        [-30.0, 0.0, -1.0],
        [-20.0, 10.0, -1.0],
        [0.0,  20.0, -1.0],
        [20.0, 10.0, -1.0],
        [30.0, 0.0, -1.0],
        [20.0, -10.0, -1.0],
        [0.0, -20.0, -1.0],
        [-20.0, -10.0, -1.0],
    ]
    for pos in path:
        # Decoy detection above PASSAGE_RANGE, so range signal is off.
        # Tests the position path in isolation.
        fired = loop._check_gate_pass_position(
            _tracker(detected=True, range_est=8.0),
            pos,
        )
        assert fired is False, (
            f"position signal fired at {pos} (dist "
            f"{((pos[0]-10)**2 + pos[1]**2) ** 0.5:.1f}m from target) — "
            "position detector must NOT fire when drone never entered radius"
        )
    assert loop.target_idx == 0
    print("  ✓ position signal rejects decoys the drone never physically "
          "approached (S19l contract)")


def test_legacy_range_path_refractory_blocks_cascade():
    """S19r — legacy path (gates_ned=None) with drone_ned supplied
    must refuse to re-fire on a stuck close-range detection until the
    drone has moved PASSAGE_REFRACTORY from the previous pass.

    This is the legacy-path equivalent of the S19m baseline cascade
    that S19n closed on the position path. Without the refractory, a
    detector that keeps reporting range < PASSAGE_RANGE (because the
    just-passed gate's detection stays nearest for several ticks while
    target_idx advances) races target_idx through the remaining gates
    in a handful of ticks.
    """
    loop = _build(gates_ned=None)
    pos0 = [0.0, 0.0, -1.0]
    # First pass: drone at origin, detection at 1 m → fires.
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0), pos0
    ) is True
    assert loop.target_idx == 1

    # Next tick: drone has barely moved (0.5 m), detection still
    # reporting close range because nearest-fallback is holding on to
    # the just-passed gate. Range signal MUST be blocked.
    pos1 = [0.5, 0.0, -1.0]
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0), pos1
    ) is False, "refractory should block re-fire with drone barely moved"
    assert loop.target_idx == 1, "target_idx should NOT have advanced"

    # Drone keeps moving but still inside refractory radius (< 5 m).
    pos2 = [3.0, 0.0, -1.0]
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.5), pos2
    ) is False

    # Past PASSAGE_REFRACTORY (5.0 m) → next legit pass fires.
    pos3 = [6.0, 0.0, -1.0]
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0), pos3
    ) is True
    assert loop.target_idx == 2
    print("  ✓ legacy-path refractory blocks cascade; clears after drone moves")


def test_legacy_range_path_without_drone_ned_still_cascades():
    """Contract: `_check_gate_pass(tracker)` without drone_ned must
    behave exactly like pre-S19r — no refractory. This preserves the
    minimal-unit-test signature and documents that callers that want
    the refractory must supply a pose.
    """
    loop = _build(gates_ned=None, gate_count=3)
    # Burn through three gates with a stuck close-range detection and
    # no drone_ned. Without the refractory, target_idx advances on every
    # firing.
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0)
    ) is True
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0)
    ) is True
    assert loop._check_gate_pass(
        _tracker(detected=True, range_est=1.0)
    ) is True
    assert loop.target_idx == 3
    print("  ✓ legacy-path without drone_ned preserves pre-S19r cascade-prone behaviour")


def main():
    tests = [
        ("legacy range path when gates_ned is None",
            test_legacy_range_based_path_when_gates_ned_is_none),
        ("S19r legacy-path refractory blocks cascade",
            test_legacy_range_path_refractory_blocks_cascade),
        ("legacy-path w/o drone_ned preserves pre-S19r behaviour",
            test_legacy_range_path_without_drone_ned_still_cascades),
        ("gates_ned allowed without pose_fusion",
            test_gates_ned_allowed_without_pose_fusion),
        ("position signal fires on flyby without detection",
            test_position_signal_fires_on_close_approach_without_detection),
        ("range signal still fires in normal pass",
            test_range_signal_still_fires_in_normal_pass),
        ("no fire when approach never enters radius",
            test_no_fire_on_approach_without_crossing),
        ("no double-fire on same gate",
            test_no_double_fire_on_same_gate),
        ("idempotent after all gates passed",
            test_target_idx_clamped_past_last_gate),
        ("range signal fires even on pose drift",
            test_range_signal_still_passes_when_pose_drifts),
        ("range signal vulnerable to near-distractor (S19l contract)",
            test_range_signal_vulnerable_to_near_distractor),
        ("position signal rejects distant distractor (S19l contract)",
            test_position_signal_rejects_distant_distractor),
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
            import traceback; traceback.print_exc()
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} PASSED")


if __name__ == "__main__":
    main()
