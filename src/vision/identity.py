"""IdentityTagger — Session 19x.

Closes the gap between "real vision produces detections" and "race loop
wants gate-identity-tagged detections". Concretely:

  * ``VirtualDetector`` emits ``gate_idx=N`` because it has sim-truth
    access to the gate list being projected.
  * ``YoloPnpDetector`` emits ``gate_idx=-1`` because a real image tells
    you "there is a gate" but not "this is gate #3".

``RaceLoop._pick_detection(associate_mode="target_idx")`` prefers a
detection matching the current target's gate_idx, and its fallback
branches were tuned against tagged VirtualDetector output. Feeding it
``gate_idx=-1`` collapses every target_idx branch to nearest-first —
which is exactly the condition that produced the S19m cascade (picker
anchors on just-passed gate still in FOV, belief model mis-anchors,
drone drifts). S19r's drone-displacement refractory blocks the *range*
cascade, but the belief anchor still misbehaves.

The fix is data association, not a picker patch: give real detections a
``gate_idx`` before they enter the race loop. The tagger back-projects
each detection's body-frame bearing + range through the drone pose into
NED, then nearest-neighbors against the known gate list. Unambiguous
matches get tagged; ambiguous or out-of-radius detections stay at -1
(safe default — the picker's ``gi < 0`` permissive branch still handles
them, with the existing refractory preventing cascades).

Design notes:

  * **Why accept-radius + ambiguity-ratio, not max-likelihood.**
    A physically-sized gate (2 m square) at typical range has
    back-projection noise well under 1 m on clean bearings; the failure
    we're guarding against is gross mis-association, not the fine-
    grained multi-hypothesis tracking problem. A two-sigma radius
    around the gate center + a ratio check against second-nearest is
    enough to trade a handful of "ambiguous → -1" over-rejections for
    zero cross-tags. Upgrade only if real YOLO data says we need to.

  * **Why pull pose from the adapter state, not the fused estimate.**
    Fusion can diverge; adapter truth cannot. If the ESKF blows up, a
    tagger using fused pose would stamp detections with wrong identity
    and lock the belief onto the wrong gate. Using adapter state lets
    the tagger stay correct even when fusion is in the weeds — the
    cost is that the tag is only as good as the adapter's own pose
    estimate (sim truth in mock/DCL, EKF2 output in PX4/real).

  * **Why a wrapper Detector, not a picker patch.**
    Keeps the Detector protocol unchanged and makes the swap a 1-line
    change at the construction site. No race-loop API drift, no
    coupling of association to the loop's target_idx state. The tagger
    is stateless; wrapping and unwrapping is cheap.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Sequence, Tuple

# GateDetection lives alongside in detector.py. Keep the import local
# and lazy so this module is usable as a mixin without dragging the
# whole vision import chain.


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def backproject_ned(
    bearing_h_deg: float,
    bearing_v_deg: float,
    range_est: float,
    drone_pos_ned: Sequence[float],
    drone_yaw_rad: float,
) -> Tuple[float, float, float]:
    """Convert a body-frame detection to an absolute NED point.

    Body frame: X forward, Y right, Z down (matches the repo's
    convention). NED: N, E, D. Yaw rotates body→NED around the down
    axis.
    """
    bh = math.radians(bearing_h_deg)
    bv = math.radians(bearing_v_deg)
    # Body cartesian
    bx = range_est * math.cos(bh) * math.cos(bv)   # forward
    by = range_est * math.sin(bh) * math.cos(bv)   # right
    bz = range_est * math.sin(bv)                  # down
    # Yaw rotation body → NED
    cy, sy = math.cos(drone_yaw_rad), math.sin(drone_yaw_rad)
    dn = bx * cy - by * sy
    de = bx * sy + by * cy
    dd = bz
    return (
        float(drone_pos_ned[0] + dn),
        float(drone_pos_ned[1] + de),
        float(drone_pos_ned[2] + dd),
    )


class IdentityTagger:
    """Assigns gate_idx to bearing/range detections via NED matching.

    Params:
        gates_ned: list of (N, E, D) gate centroids in world frame.
        accept_radius_m: a detection's back-projected point must lie
            within this distance of a gate centroid to be considered a
            match at all. Default 2.5 m — gates are 2 m square, this
            handles ~1 m bearing-induced position noise at 10 m range.
        ambiguity_ratio: a match is accepted only if the second-nearest
            gate is at least this much farther than the nearest. 1.3 =
            nearest must be ≥30% closer. Defends against courses where
            two gates fall inside accept_radius simultaneously. Set to
            0 to disable the ratio gate and trust nearest-only.
    """

    DEFAULT_ACCEPT_RADIUS_M = 2.5
    DEFAULT_AMBIGUITY_RATIO = 1.3

    def __init__(
        self,
        gates_ned: Sequence[Tuple[float, float, float]],
        accept_radius_m: float = DEFAULT_ACCEPT_RADIUS_M,
        ambiguity_ratio: float = DEFAULT_AMBIGUITY_RATIO,
    ):
        self.gates = [tuple(g) for g in gates_ned]
        self.accept_radius_m = float(accept_radius_m)
        self.ambiguity_ratio = float(ambiguity_ratio)

    def tag(self, detection, drone_pos_ned, drone_yaw_rad):
        """Return a copy of ``detection`` with ``gate_idx`` set.

        Returns the original ``gate_idx`` untouched if it's already a
        valid non-negative index (so VirtualDetector output passes
        through unchanged — lets users wrap every detector
        defensively).

        Sets ``gate_idx = -1`` when no gate falls within
        ``accept_radius_m`` of the back-projected point, or when two
        gates are closer together than ``ambiguity_ratio`` allows.
        """
        existing = int(getattr(detection, "gate_idx", -1))
        if existing >= 0:
            return detection

        if not self.gates:
            return replace(detection, gate_idx=-1)

        world = backproject_ned(
            detection.bearing_h_deg,
            detection.bearing_v_deg,
            detection.range_est,
            drone_pos_ned,
            drone_yaw_rad,
        )

        dists = [_euclidean(world, g) for g in self.gates]
        order = sorted(range(len(dists)), key=lambda i: dists[i])
        nearest_i = order[0]
        nearest_d = dists[nearest_i]

        if nearest_d > self.accept_radius_m:
            return replace(detection, gate_idx=-1)

        if self.ambiguity_ratio > 0 and len(order) > 1:
            second_d = dists[order[1]]
            # Guard div-by-zero — if nearest_d is tiny, ratio is huge →
            # unambiguous, keep.
            if nearest_d > 1e-6 and second_d / nearest_d < self.ambiguity_ratio:
                return replace(detection, gate_idx=-1)

        return replace(detection, gate_idx=nearest_i)

    def tag_all(self, detections, drone_pos_ned, drone_yaw_rad) -> List:
        """Batch convenience — returns a new list of tagged detections."""
        return [
            self.tag(d, drone_pos_ned, drone_yaw_rad) for d in detections
        ]


class TaggedDetector:
    """Composes a Detector with an IdentityTagger.

    Drop-in replacement for any ``Detector`` — satisfies the same
    Protocol. Pulls drone pos + yaw from the ``state`` argument that
    the race loop already passes to ``detect()``.

    Usage:
        inner = YoloPnpDetector(model_path=..., ...)
        tagger = IdentityTagger(gates_ned=gates)
        detector = TaggedDetector(inner, tagger)
        # detector.detect(frame, state) now returns gate_idx-tagged
        # detections, and ``_pick_detection(associate_mode="target_idx")``
        # will find exact-match matches.
    """

    def __init__(self, inner, tagger: IdentityTagger):
        self._inner = inner
        self._tagger = tagger

    def detect(self, frame, state):
        raw = self._inner.detect(frame, state)
        if not raw:
            return raw
        yaw_rad = float(state.att_rad[2])
        pos = state.pos_ned
        return self._tagger.tag_all(raw, pos, yaw_rad)

    def name(self) -> str:
        inner_name = self._inner.name() if hasattr(self._inner, "name") else "?"
        return f"tagged[{inner_name}]"
