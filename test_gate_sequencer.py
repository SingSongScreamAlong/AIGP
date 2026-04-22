"""Tests for the track-agnostic GateSequencer.

Validates:
  1. Basic latch-track-pass cycle
  2. Suppression of recently-passed gates
  3. Forward-bias filtering (rear detections rejected)
  4. Refractory period (no double-pass)
  5. Multi-gate sequence on a simulated course
  6. Unknown gate_count (no early termination)
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Make src importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE))

from race.gate_sequencer import GateSequencer, PassedGate


# ── Minimal detection stub ──────────────────────────────────────────

@dataclass
class FakeDet:
    """Minimal GateDetection-like object for testing."""
    bearing_h_deg: float
    bearing_v_deg: float
    range_est: float
    confidence: float
    gate_idx: int = -1
    angular_size_deg: float = 5.0
    in_fov: bool = True


def make_det(bearing_h_deg=0.0, range_est=10.0, confidence=0.9,
             bearing_v_deg=0.0) -> FakeDet:
    return FakeDet(
        bearing_h_deg=bearing_h_deg,
        bearing_v_deg=bearing_v_deg,
        range_est=range_est,
        confidence=confidence,
    )


# ── Tests ───────────────────────────────────────────────────────────

def test_basic_pass():
    """Single gate: approach → pass → count increments."""
    seq = GateSequencer(gate_count=1)
    drone_pos = [0.0, 0.0, -2.0]
    yaw = 0.0  # facing north

    # Approach: detections at decreasing range
    for r in [15.0, 10.0, 7.0, 5.0, 3.5]:
        det = make_det(range_est=r)
        target, passed = seq.update([det], drone_pos, yaw)
        assert target is not None, f"Should have a target at range {r}"
        assert not passed, f"Should NOT pass at range {r}"

    # Pass: range drops below threshold
    det = make_det(range_est=2.0)
    target, passed = seq.update([det], drone_pos, yaw)
    assert passed, "Gate should be passed at range 2.0"
    assert seq.gates_passed == 1
    assert seq.completed
    print("  ✓ test_basic_pass")


def test_suppression():
    """After passing a gate, detections near it are suppressed."""
    seq = GateSequencer(gate_count=3)
    drone_pos = [0.0, 0.0, -2.0]
    yaw = 0.0

    # Approach and pass first gate
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone_pos, yaw)
    _, passed = seq.update([make_det(range_est=2.0)], drone_pos, yaw)
    assert passed

    # Move drone forward past refractory
    drone_pos = [6.0, 0.0, -2.0]

    # Detection from the PASSED gate (behind us, but close to its NED)
    # The passed gate was at roughly (0, 0, -2) + forward offset
    # Now we're at (6, 0, -2). A detection bearing 180° (behind) at
    # range 6m would place at roughly (0, 0, -2) — near the passed gate
    passed_gate_det = make_det(bearing_h_deg=180.0, range_est=6.0)
    target, _ = seq.update([passed_gate_det], drone_pos, yaw)
    # Should be filtered by forward-bias (180° > 75°)
    assert target is None, "Detection behind drone should be filtered"

    # Detection in front — new gate
    new_gate_det = make_det(bearing_h_deg=5.0, range_est=12.0)
    target, _ = seq.update([new_gate_det], drone_pos, yaw)
    assert target is not None, "Forward detection should be accepted"
    assert target.range_est == 12.0
    print("  ✓ test_suppression")


def test_forward_bias():
    """Detections outside MAX_TARGET_BEARING are rejected."""
    seq = GateSequencer()

    # Forward detection — accepted
    target, _ = seq.update(
        [make_det(bearing_h_deg=10.0, range_est=8.0)],
        [0.0, 0.0, -2.0], 0.0,
    )
    assert target is not None

    # Side detection at 80° — rejected (>75°)
    target, _ = seq.update(
        [make_det(bearing_h_deg=80.0, range_est=8.0)],
        [0.0, 0.0, -2.0], 0.0,
    )
    assert target is None, "80° bearing should be filtered"

    # Side detection at -70° — accepted
    target, _ = seq.update(
        [make_det(bearing_h_deg=-70.0, range_est=8.0)],
        [0.0, 0.0, -2.0], 0.0,
    )
    assert target is not None
    print("  ✓ test_forward_bias")


def test_refractory():
    """After a pass, the next pass is blocked until drone moves."""
    seq = GateSequencer(gate_count=5)
    drone_pos = [0.0, 0.0, -2.0]
    yaw = 0.0

    # Pass first gate
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone_pos, yaw)
    _, passed = seq.update([make_det(range_est=2.0)], drone_pos, yaw)
    assert passed
    assert seq.gates_passed == 1

    # Immediately present another close detection — should NOT pass
    # (drone hasn't moved past refractory)
    _, passed = seq.update([make_det(range_est=1.5)], drone_pos, yaw)
    assert not passed, "Should be in refractory"

    # Move drone slightly (not enough)
    drone_pos = [2.0, 0.0, -2.0]
    for r in [8.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone_pos, yaw)
    _, passed = seq.update([make_det(range_est=2.0)], drone_pos, yaw)
    assert not passed, "Should still be in refractory (moved 2m < 5m)"

    # Move drone far enough
    drone_pos = [8.0, 0.0, -2.0]
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone_pos, yaw)
    _, passed = seq.update([make_det(range_est=2.0)], drone_pos, yaw)
    assert passed, "Refractory should be cleared after 8m displacement"
    assert seq.gates_passed == 2
    print("  ✓ test_refractory")


def test_multi_gate_sequence():
    """Simulate flying through a 4-gate course."""
    seq = GateSequencer(gate_count=4)

    # Gate positions (unknown to sequencer; we use them to generate detections)
    gates = [(10, 0, -2), (20, 5, -2), (30, 5, -2), (40, 0, -2)]
    drone = [0.0, 0.0, -2.0]
    yaw = 0.0

    for gi, gate in enumerate(gates):
        # Approach
        steps = 20
        start = list(drone)
        for s in range(steps):
            frac = (s + 1) / steps
            drone[0] = start[0] + (gate[0] - start[0]) * frac
            drone[1] = start[1] + (gate[1] - start[1]) * frac
            drone[2] = start[2] + (gate[2] - start[2]) * frac

            # Compute body-frame detection
            dn = gate[0] - drone[0]
            de = gate[1] - drone[1]
            dd = gate[2] - drone[2]
            r = math.sqrt(dn * dn + de * de + dd * dd)
            if r < 0.1:
                r = 0.1
            bh = math.degrees(math.atan2(de * math.cos(yaw) - dn * math.sin(yaw),
                                          dn * math.cos(yaw) + de * math.sin(yaw)))
            # Simplify — use heading=0 so bearing is just atan2(de, dn)
            bh = math.degrees(math.atan2(de - drone[1] + drone[1], dn))

            dets = [make_det(bearing_h_deg=bh, range_est=r)]

            # Also add a detection for the NEXT gate if visible
            if gi + 1 < len(gates):
                ng = gates[gi + 1]
                ndn = ng[0] - drone[0]
                nde = ng[1] - drone[1]
                nr = math.sqrt(ndn * ndn + nde * nde)
                nbh = math.degrees(math.atan2(nde, ndn))
                dets.append(make_det(bearing_h_deg=nbh, range_est=nr))

            target, passed = seq.update(dets, drone, yaw, elapsed_time=float(s))

            if passed:
                assert seq.gates_passed == gi + 1, (
                    f"Expected {gi + 1} gates passed, got {seq.gates_passed}"
                )
                break

        assert seq.gates_passed >= gi + 1, (
            f"Gate {gi} was never passed (gates_passed={seq.gates_passed})"
        )

    assert seq.gates_passed == 4
    assert seq.completed
    print("  ✓ test_multi_gate_sequence")


def test_unknown_gate_count():
    """When gate_count is None, completed is always False."""
    seq = GateSequencer(gate_count=None)
    assert not seq.completed

    drone_pos = [0.0, 0.0, -2.0]
    yaw = 0.0
    # Pass a gate
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone_pos, yaw)
    _, passed = seq.update([make_det(range_est=2.0)], drone_pos, yaw)
    assert passed
    assert seq.gates_passed == 1
    assert not seq.completed  # Still not completed — no known gate_count
    print("  ✓ test_unknown_gate_count")


def test_low_confidence_filtered():
    """Detections below MIN_CONFIDENCE are ignored."""
    seq = GateSequencer()
    det = make_det(range_est=5.0, confidence=0.05)
    target, _ = seq.update([det], [0, 0, -2], 0.0)
    assert target is None, "Low-confidence detection should be filtered"
    print("  ✓ test_low_confidence_filtered")


def test_predict_next_heading():
    """1-gate prediction projects along drone heading, not north."""
    seq = GateSequencer(gate_count=5)
    drone = [0.0, 0.0, -2.0]
    yaw = math.radians(45.0)  # NE heading

    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone, yaw)
    seq.update([make_det(range_est=2.0)], drone, yaw)
    assert seq.gates_passed == 1

    pred = seq.predict_next_ned()
    assert pred is not None
    # Gate was ~2m ahead in NE direction from (0,0,-2)
    # Prediction should be ~15m further in NE from that
    g = seq.passed_gates[0].ned_est
    dn = pred[0] - g[0]
    de = pred[1] - g[1]
    heading = math.atan2(de, dn)
    assert abs(heading - yaw) < 0.01, f"Prediction heading {math.degrees(heading):.1f}° != drone yaw 45°"
    print("  ✓ test_predict_next_heading")


def test_predict_two_gate_extrapolation():
    """2-gate prediction extrapolates from last two passed gates."""
    seq = GateSequencer(gate_count=5)
    drone = [0.0, 0.0, -2.0]
    yaw = 0.0

    # Pass gate 0
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone, yaw)
    seq.update([make_det(range_est=2.0)], drone, yaw)
    g0 = seq.passed_gates[0].ned_est

    # Move past refractory, pass gate 1
    drone = [10.0, 5.0, -2.0]
    for r in [8.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r, bearing_h_deg=10.0)], drone, yaw)
    seq.update([make_det(range_est=2.0, bearing_h_deg=10.0)], drone, yaw)
    assert seq.gates_passed == 2
    g1 = seq.passed_gates[1].ned_est

    pred = seq.predict_next_ned()
    # Should extrapolate: g1 + (g1 - g0)
    expected_n = g1[0] + (g1[0] - g0[0])
    expected_e = g1[1] + (g1[1] - g0[1])
    assert abs(pred[0] - expected_n) < 0.5, f"N: {pred[0]:.1f} != {expected_n:.1f}"
    assert abs(pred[1] - expected_e) < 0.5, f"E: {pred[1]:.1f} != {expected_e:.1f}"
    print("  ✓ test_predict_two_gate_extrapolation")


def test_nav_gate_list_length():
    """Nav gate list has passed_count + 1 entries (includes prediction)."""
    seq = GateSequencer(gate_count=5)
    drone = [0.0, 0.0, -2.0]
    assert len(seq.get_nav_gate_list()) == 0  # no gates passed yet

    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r)], drone, 0.0)
    seq.update([make_det(range_est=2.0)], drone, 0.0)
    nav = seq.get_nav_gate_list()
    assert len(nav) == 2, f"Expected 2 (passed + predicted), got {len(nav)}"
    print("  ✓ test_nav_gate_list_length")


def test_ned_suppression_geometry():
    """Verify NED backprojection for suppression is geometrically correct."""
    seq = GateSequencer(gate_count=5, suppression_radius=3.0)

    # Gate is 10m north. Drone is at origin, facing north (yaw=0).
    drone = [0.0, 0.0, -2.0]
    yaw = 0.0

    # Approach and pass
    for r in [10.0, 5.0, 3.0]:
        seq.update([make_det(range_est=r, bearing_h_deg=0.0)], drone, yaw)
    _, passed = seq.update(
        [make_det(range_est=2.0, bearing_h_deg=0.0)], drone, yaw
    )
    assert passed

    # The passed gate's estimated NED should be ~(2.0, 0, -2)
    # (drone at origin + 2m forward in body frame = 2m north)
    pg = seq.passed_gates[0]
    assert abs(pg.ned_est[0] - 2.0) < 0.5, f"Expected N≈2.0, got {pg.ned_est[0]}"
    assert abs(pg.ned_est[1]) < 0.5, f"Expected E≈0, got {pg.ned_est[1]}"
    print("  ✓ test_ned_suppression_geometry")


def main():
    print("Running GateSequencer tests...")
    test_basic_pass()
    test_suppression()
    test_forward_bias()
    test_refractory()
    test_multi_gate_sequence()
    test_unknown_gate_count()
    test_low_confidence_filtered()
    test_predict_next_heading()
    test_predict_two_gate_extrapolation()
    test_nav_gate_list_length()
    test_ned_suppression_geometry()
    print(f"\nAll tests passed! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
