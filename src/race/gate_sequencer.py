"""Track-agnostic gate sequencing — discovers gates from vision alone.

Replaces the hardcoded gate list for unknown courses (VQ1/VQ2). When
`gates_ned` is not supplied, the race stack needs a way to:
  1. Latch onto the nearest visible forward gate
  2. Track it until passed
  3. Suppress re-detection of recently-passed gates
  4. Advance to the next gate

The sequencer is stateful per race. Construct, call update() per tick.

Gate suppression works by estimating the NED position of each passed
gate (from drone pose + body-frame bearing/range at pass time) and
rejecting future detections whose backprojected NED falls within a
suppression radius of any passed gate. This is cheap to compute and
good enough for courses where gates are ≥5 m apart.

Coordinate conventions match the rest of the stack:
  - NED world frame
  - Body frame: X forward, Y right, Z down (FRD)
  - Bearings: horizontal (rad, 0=forward, +right), vertical (rad, +down)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class PassedGate:
    """Record of a gate that has been passed."""
    index: int              # sequential pass number (0, 1, 2, ...)
    ned_est: Tuple[float, float, float]  # estimated NED of the gate center
    pass_time: float        # race elapsed time at pass
    drone_yaw_rad: float = 0.0  # drone heading at pass time


class GateSequencer:
    """Vision-only gate sequencing for unknown courses.

    Usage per tick::

        target, passed = seq.update(detections, drone_pos_ned, drone_yaw_rad, t)
        # target: best detection to navigate toward (or None)
        # passed: True if the current gate was just passed this tick

    Design rules:
      * Never requires a priori gate positions. All knowledge is
        built from observations.
      * Gate pass = range crossing below PASS_RANGE while the
        detection was being tracked, AND the drone has moved
        past the refractory distance since the last pass.
      * Suppression is purely geometric (NED distance to passed-gate
        estimates). False-suppression risk is bounded by the
        suppression radius; set conservatively for VQ1 gate spacing.
      * Forward-bias: prefer detections within ±MAX_TARGET_BEARING
        of the drone's heading. This rejects gates behind the drone
        that YOLO might still detect at the frame edge.
    """

    # --- Tuning ---
    PASS_RANGE: float = 2.5            # m, range threshold for pass
    REFRACTORY_DISPLACEMENT: float = 5.0  # m, drone must move this far after a pass
    SUPPRESSION_RADIUS: float = 4.0    # m, NED distance to suppress passed gates
    MAX_TARGET_BEARING: float = math.radians(75.0)  # ±75° from forward
    MIN_CONFIDENCE: float = 0.15       # ignore low-confidence detections
    DEFAULT_SPACING: float = 15.0      # m, assumed gate spacing for 1-gate prediction

    def __init__(
        self,
        gate_count: Optional[int] = None,
        pass_range: float = PASS_RANGE,
        suppression_radius: float = SUPPRESSION_RADIUS,
    ):
        """
        Args:
            gate_count: If known, the sequencer reports completion after
                this many passes. If None, races until timeout.
            pass_range: Range (m) below which a tracked gate is considered
                passed.
            suppression_radius: NED distance (m) within which a detection
                is suppressed (matched to a previously-passed gate).
        """
        self.gate_count = gate_count
        self.PASS_RANGE = pass_range
        self.SUPPRESSION_RADIUS = suppression_radius

        # State
        self.gates_passed: int = 0
        self.passed_gates: List[PassedGate] = []
        self._tracking_range: Optional[float] = None
        self._frames_tracking: int = 0  # consecutive frames with a target
        self._frames_without: int = 0   # consecutive frames without a target
        self._drone_ned_at_pass: Optional[Tuple[float, float, float]] = None

    @property
    def completed(self) -> bool:
        """True when all gates have been passed (only meaningful when
        gate_count was provided)."""
        if self.gate_count is None:
            return False
        return self.gates_passed >= self.gate_count

    def update(
        self,
        detections: Sequence,
        drone_pos_ned: Sequence[float],
        drone_yaw_rad: float,
        elapsed_time: float = 0.0,
    ) -> Tuple[Optional[object], bool]:
        """Process one tick of detections.

        Args:
            detections: List of GateDetection objects (body-frame).
            drone_pos_ned: Current drone NED position (truth or fused).
            drone_yaw_rad: Current drone yaw in radians.
            elapsed_time: Race elapsed time for logging.

        Returns:
            (target_detection, gate_passed_this_tick)
        """
        if self.completed:
            return None, False

        # 1. Filter: remove low-confidence and suppressed detections
        candidates = self._filter(detections, drone_pos_ned, drone_yaw_rad)

        # 2. Pick target: nearest forward detection
        target = self._pick_target(candidates)

        # 3. Check refractory: has drone moved far enough since last pass?
        in_refractory = self._check_refractory(drone_pos_ned)

        # 4. Check gate pass
        passed = False
        if target is not None:
            self._frames_tracking += 1
            self._frames_without = 0
            current_range = target.range_est

            if (
                not in_refractory
                and self._tracking_range is not None
                and self._frames_tracking >= 3  # need a few frames of tracking
                and current_range < self.PASS_RANGE
            ):
                # Pass detected
                passed = True
                gate_ned = self._estimate_gate_ned(
                    drone_pos_ned, drone_yaw_rad, target
                )
                self.passed_gates.append(PassedGate(
                    index=self.gates_passed,
                    ned_est=gate_ned,
                    pass_time=elapsed_time,
                    drone_yaw_rad=drone_yaw_rad,
                ))
                self.gates_passed += 1
                self._drone_ned_at_pass = tuple(drone_pos_ned)
                self._tracking_range = None
                self._frames_tracking = 0
            else:
                self._tracking_range = current_range
        else:
            self._frames_without += 1
            self._frames_tracking = 0
            # Don't reset _tracking_range on a single missed frame —
            # keep it alive for a few ticks so transient dropouts don't
            # break pass detection.
            if self._frames_without > 10:
                self._tracking_range = None

        return target, passed

    def get_passed_positions(self) -> List[Tuple[float, float, float]]:
        """Return NED estimates of all passed gates. Useful for building
        a running gate_list for the navigator's gate-aware fallback."""
        return [pg.ned_est for pg in self.passed_gates]

    def predict_next_ned(self) -> Optional[Tuple[float, float, float]]:
        """Extrapolate the likely position of the NEXT gate from the
        trajectory of passed gates.

        Uses the last two passed gates to project a heading + spacing.
        Falls back to the last passed gate + a forward offset when only
        one gate has been passed. Returns None before any gate is passed.

        The prediction is intentionally rough — it's used for the
        navigator's fallback heading, not precision planning. Even a
        30° error is far better than blind search.
        """
        n = len(self.passed_gates)
        if n == 0:
            return None

        if n == 1:
            # Only one gate passed — project forward in the drone's
            # heading at pass time by DEFAULT_SPACING. Much better than
            # assuming north on courses with early turns.
            g = self.passed_gates[0].ned_est
            yaw = self.passed_gates[0].drone_yaw_rad
            return (
                g[0] + self.DEFAULT_SPACING * math.cos(yaw),
                g[1] + self.DEFAULT_SPACING * math.sin(yaw),
                g[2],
            )

        # Two or more gates: extrapolate from the last two.
        g_prev = self.passed_gates[-2].ned_est
        g_last = self.passed_gates[-1].ned_est
        dn = g_last[0] - g_prev[0]
        de = g_last[1] - g_prev[1]
        dd = g_last[2] - g_prev[2]
        # Project the same vector forward
        return (
            g_last[0] + dn,
            g_last[1] + de,
            g_last[2] + dd,
        )

    def get_nav_gate_list(self) -> List[Tuple[float, float, float]]:
        """Return the discovered gate list PLUS the predicted next gate.

        This gives the navigator a world-frame target to steer toward
        when no detection is available. The predicted gate is appended
        at index `gates_passed` — exactly where `target_idx` points.
        """
        positions = self.get_passed_positions()
        predicted = self.predict_next_ned()
        if predicted is not None:
            positions.append(predicted)
        return positions

    # --- Internal methods ---

    def _filter(self, detections, drone_pos_ned, drone_yaw_rad):
        """Remove low-confidence detections and those matching passed gates."""
        out = []
        for det in detections:
            # Confidence filter
            if det.confidence < self.MIN_CONFIDENCE:
                continue
            # Forward filter: reject detections behind the drone
            bearing_h = getattr(det, "bearing_h_deg", None)
            if bearing_h is not None:
                bearing_h_rad = math.radians(bearing_h)
            else:
                # Some detection types might have bearing_h directly in rad
                bearing_h_rad = getattr(det, "bearing_h", 0.0)
            if abs(bearing_h_rad) > self.MAX_TARGET_BEARING:
                continue
            # Suppression: check if this detection matches a passed gate
            if self._is_suppressed(det, drone_pos_ned, drone_yaw_rad):
                continue
            out.append(det)
        return out

    def _is_suppressed(self, det, drone_pos_ned, drone_yaw_rad) -> bool:
        """Check if a detection's estimated NED matches any passed gate."""
        if not self.passed_gates:
            return False
        det_ned = self._estimate_gate_ned(drone_pos_ned, drone_yaw_rad, det)
        for pg in self.passed_gates:
            dn = det_ned[0] - pg.ned_est[0]
            de = det_ned[1] - pg.ned_est[1]
            dd = det_ned[2] - pg.ned_est[2]
            dist = math.sqrt(dn * dn + de * de + dd * dd)
            if dist < self.SUPPRESSION_RADIUS:
                return True
        return False

    def _estimate_gate_ned(
        self, drone_pos_ned, drone_yaw_rad, det
    ) -> Tuple[float, float, float]:
        """Backproject a body-frame detection into NED world coordinates."""
        # Get bearing in radians
        bearing_h = getattr(det, "bearing_h_deg", None)
        if bearing_h is not None:
            bh = math.radians(bearing_h)
        else:
            bh = getattr(det, "bearing_h", 0.0)
        bearing_v = getattr(det, "bearing_v_deg", None)
        if bearing_v is not None:
            bv = math.radians(bearing_v)
        else:
            bv = getattr(det, "bearing_v", 0.0)
        r = det.range_est

        # Body-frame offset (FRD: X forward, Y right, Z down)
        body_x = r * math.cos(bv) * math.cos(bh)
        body_y = r * math.cos(bv) * math.sin(bh)
        body_z = r * math.sin(bv)

        # Rotate body→NED using yaw (assume level flight)
        cy = math.cos(drone_yaw_rad)
        sy = math.sin(drone_yaw_rad)
        dn = body_x * cy - body_y * sy
        de = body_x * sy + body_y * cy
        dd = body_z

        return (
            drone_pos_ned[0] + dn,
            drone_pos_ned[1] + de,
            drone_pos_ned[2] + dd,
        )

    def _pick_target(self, candidates):
        """Pick the best target from filtered candidates.

        Strategy: nearest detection. In the common case (1-2 gates visible),
        nearest-first is correct. For dense gate fields, this could be
        enhanced with heading-consistency scoring, but VQ1 (<10 gates,
        simple layout) doesn't need it.
        """
        if not candidates:
            return None
        # Already sorted nearest-first by detector convention;
        # but sort explicitly to be safe.
        return min(candidates, key=lambda d: d.range_est)

    def _check_refractory(self, drone_pos_ned) -> bool:
        """True if the drone hasn't moved far enough since the last pass."""
        if self._drone_ned_at_pass is None:
            return False
        dn = drone_pos_ned[0] - self._drone_ned_at_pass[0]
        de = drone_pos_ned[1] - self._drone_ned_at_pass[1]
        dd = drone_pos_ned[2] - self._drone_ned_at_pass[2]
        disp = math.sqrt(dn * dn + de * de + dd * dd)
        if disp >= self.REFRACTORY_DISPLACEMENT:
            self._drone_ned_at_pass = None
            return False
        return True
