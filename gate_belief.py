"""Gate Belief Model — Session 17 step 3.

Maintains a probabilistic estimate of the next gate's position relative
to the drone, using only bearing/range observations from the camera.

Key properties:
- Fuses noisy detections via exponential moving average
- Propagates belief forward using drone velocity when no detection
- Confidence decays over time without updates
- Provides a "best guess" bearing/range even during dropouts
- Structured search: fly toward belief peak instead of panic sweep

Replaces the binary detected/not-detected with continuous belief.
"""

import math
import time
from mavsdk.offboard import VelocityNedYaw


class GateBelief:
    """Maintains a belief about the current target gate position.

    State: bearing (rad from drone forward), range (m), confidence (0..1).

    On detection: update state with exponential smoothing.
    On no detection: propagate using drone motion, decay confidence.

    The belief is in a BODY-RELATIVE frame that gets updated each tick
    to account for drone motion (velocity and yaw rate).
    """

    # Tuning constants
    ALPHA_BEARING = 0.4      # EMA weight for new bearing observation
    ALPHA_RANGE = 0.3        # EMA weight for new range observation
    CONFIDENCE_DECAY = 0.97  # per-tick confidence decay without detection (at 50Hz)
    MIN_CONFIDENCE = 0.02    # below this, belief is considered dead
    INITIAL_RANGE = 20.0     # default range assumption for a new gate
    RANGE_APPROACH_RATE = 0.0  # updated from velocity projection each tick

    def __init__(self):
        self.bearing_h = 0.0       # rad, horizontal bearing from drone forward
        self.bearing_v = 0.0       # rad, vertical bearing
        self.range_est = self.INITIAL_RANGE
        self.confidence = 0.0
        self.last_update_time = None
        self.ticks_since_detection = 0
        self.total_detections = 0
        self._prev_yaw_rad = None

    def reset(self):
        """Reset belief for a new gate target."""
        self.bearing_h = 0.0
        self.bearing_v = 0.0
        self.range_est = self.INITIAL_RANGE
        self.confidence = 0.0
        self.last_update_time = None
        self.ticks_since_detection = 0
        self.total_detections = 0
        self._prev_yaw_rad = None

    def update_detected(self, bearing_h_rad, bearing_v_rad, range_est,
                         det_confidence, yaw_rad):
        """Update belief with a new detection.

        Args:
            bearing_h_rad: horizontal bearing from drone forward axis (rad)
            bearing_v_rad: vertical bearing (rad)
            range_est: estimated range in meters
            det_confidence: detection confidence (0..1)
            yaw_rad: current drone yaw (for yaw-rate tracking)
        """
        if self.total_detections == 0:
            # First detection — snap to it
            self.bearing_h = bearing_h_rad
            self.bearing_v = bearing_v_rad
            self.range_est = range_est
            self.confidence = det_confidence
        else:
            # EMA fusion — weight by detection confidence
            alpha_b = self.ALPHA_BEARING * det_confidence
            alpha_r = self.ALPHA_RANGE * det_confidence
            self.bearing_h = self.bearing_h * (1 - alpha_b) + bearing_h_rad * alpha_b
            self.bearing_v = self.bearing_v * (1 - alpha_b) + bearing_v_rad * alpha_b
            self.range_est = self.range_est * (1 - alpha_r) + range_est * alpha_r
            # Confidence jumps toward detection confidence
            self.confidence = max(self.confidence, det_confidence * 0.9)

        self.ticks_since_detection = 0
        self.total_detections += 1
        self.last_update_time = time.time()
        self._prev_yaw_rad = yaw_rad

    def propagate(self, vel, yaw_rad, dt):
        """Propagate belief forward when no detection.

        Uses drone velocity to update the body-frame bearing/range estimate.
        The gate is stationary in world frame; the drone moves.

        Args:
            vel: [vN, vE, vD] drone velocity in NED
            yaw_rad: current drone yaw
            dt: time step
        """
        if self.confidence < self.MIN_CONFIDENCE:
            return  # belief is dead, nothing to propagate

        # 1. Convert belief from body frame to world-relative offset
        #    body_x = forward, body_y = right
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Gate offset in body frame
        body_x = self.range_est * math.cos(self.bearing_h) * math.cos(self.bearing_v)
        body_y = self.range_est * math.sin(self.bearing_h) * math.cos(self.bearing_v)
        body_z = self.range_est * math.sin(self.bearing_v)

        # Convert to NED offset
        dn = body_x * cos_y - body_y * sin_y
        de = body_x * sin_y + body_y * cos_y
        dd = body_z

        # 2. Subtract drone motion (gate is stationary, drone moved)
        dn -= vel[0] * dt
        de -= vel[1] * dt
        dd -= vel[2] * dt

        # 3. Convert back to body frame (using CURRENT yaw, which may have changed)
        new_body_x = dn * cos_y + de * sin_y
        new_body_y = -dn * sin_y + de * cos_y
        new_body_z = dd

        # 4. Extract new bearing and range
        new_range = math.sqrt(new_body_x**2 + new_body_y**2 + new_body_z**2)
        if new_range > 0.1:
            self.bearing_h = math.atan2(new_body_y, new_body_x)
            horiz_dist = math.sqrt(new_body_x**2 + new_body_y**2)
            self.bearing_v = math.atan2(new_body_z, horiz_dist) if horiz_dist > 0.1 else 0.0
            self.range_est = new_range

        # 5. Decay confidence
        self.confidence *= self.CONFIDENCE_DECAY
        self.ticks_since_detection += 1
        self._prev_yaw_rad = yaw_rad

    @property
    def is_alive(self):
        """Whether the belief has enough confidence to be useful."""
        return self.confidence >= self.MIN_CONFIDENCE

    @property
    def is_stale(self):
        """Whether belief is getting old but still usable."""
        return self.ticks_since_detection > 25  # 0.5s at 50Hz

    @property
    def search_bearing(self):
        """Best-guess bearing for structured search.

        Returns bearing in rad. If belief is alive, this is the propagated
        estimate. If dead, returns 0.0 (forward).
        """
        if self.is_alive:
            return self.bearing_h
        return 0.0

    @property
    def search_range(self):
        """Best-guess range for speed management during search."""
        if self.is_alive:
            return self.range_est
        return self.INITIAL_RANGE


class BeliefNav:
    """Vision navigator enhanced with gate belief model.

    Replaces VisionNav's binary search/coast with belief-driven navigation:
    - TRACKING: detection available, belief updated, full speed toward gate
    - COAST: no detection, belief alive, fly toward belief with reduced speed
    - SEARCH: belief dead, structured spiral search

    The key improvement: coast uses propagated belief instead of freezing
    the last bearing, and search flies toward the belief peak instead of
    panic-sweeping.
    """

    def __init__(self, max_speed=12.0, cruise_speed=9.0):
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.px4_speed_ceiling = 9.5
        self.last_cmd_speed = 0.0
        self.mission_start_time = None

        # PX4 utilization (from V5.1)
        self.px4_util_straight = 0.78
        self.px4_util_turn = 0.55

        # Speed management
        self.base_blend = 1.5

        # Belief-driven coast
        self.coast_speed_frac = 0.7   # higher than old 0.5 — belief is better than frozen bearing
        self.coast_min_speed = 3.0

        # Structured search
        self.search_speed = 3.0       # m/s — faster than old 2.0
        self.search_yaw_rate = 25.0   # deg/s — slower, more deliberate than old 30
        self.search_yaw_direction = 1.0  # +1 or -1, flips on timeout
        self.search_phase_ticks = 0
        self.SEARCH_PHASE_DURATION = 75  # 1.5s per sweep direction at 50Hz

        # Gate belief
        self.belief = GateBelief()
        self.current_gate = 0

    def on_gate_passed(self):
        """Called when tracker detects gate passage."""
        self.belief.reset()
        self.current_gate += 1
        self.search_yaw_direction = 1.0
        self.search_phase_ticks = 0

    def _speed_for_range(self, range_est, confidence=1.0):
        """Speed based on range, scaled by confidence."""
        if range_est < 4.0:
            base = max(3.0, self.cruise_speed * 0.5)
        elif range_est < 8.0:
            base = self.cruise_speed * 0.8
        else:
            base = self.cruise_speed
        return base

    def _px4_cmd(self, desired, turn_angle=0.0):
        """PX4 utilization correction."""
        tr = min(abs(turn_angle) / math.pi, 1.0)
        util = self.px4_util_straight * (1 - tr) + self.px4_util_turn * tr
        return min(desired / max(util, 0.3), self.max_speed)

    def _smooth(self, target, dt=0.02):
        """Smooth speed transitions."""
        max_rate = 12.0
        max_delta = max_rate * dt
        if target > self.last_cmd_speed:
            s = min(target, self.last_cmd_speed + max_delta)
        else:
            s = max(target, self.last_cmd_speed - max_delta * 1.5)
        self.last_cmd_speed = s
        return s

    def plan(self, tracker_state, pos, vel, yaw_deg, dt=0.02):
        """Produce velocity command using belief model.

        Args:
            tracker_state: TrackerState from GateTracker
            pos: [N, E, D]
            vel: [vN, vE, vD]
            yaw_deg: current heading
            dt: timestep
        Returns:
            VelocityNedYaw
        """
        if self.mission_start_time is None:
            self.mission_start_time = time.time()

        yaw_rad = math.radians(yaw_deg)

        # Handle gate advancement
        if tracker_state.target_idx > self.current_gate:
            for _ in range(tracker_state.target_idx - self.current_gate):
                self.on_gate_passed()

        # ── Update belief from detection ──
        if tracker_state.detected:
            self.belief.update_detected(
                tracker_state.bearing_h,
                tracker_state.bearing_v,
                tracker_state.range_est,
                tracker_state.confidence,
                yaw_rad,
            )
            self.search_phase_ticks = 0
            return self._plan_tracking(tracker_state, yaw_rad, dt)

        # ── No detection: propagate belief ──
        self.belief.propagate(vel, yaw_rad, dt)

        if self.belief.is_alive:
            # Coast toward belief
            return self._plan_coast(vel, yaw_rad, dt)
        else:
            # Belief dead — structured search
            return self._plan_search(yaw_deg, yaw_rad, dt)

    def _plan_tracking(self, state, yaw_rad, dt):
        """Full speed toward detected gate."""
        desired = self._speed_for_range(state.range_est, state.confidence)
        cmd_speed = self._px4_cmd(desired, abs(state.bearing_h))
        sp = self._smooth(cmd_speed, dt)

        target_yaw_rad = yaw_rad + state.bearing_h
        vn = sp * math.cos(target_yaw_rad)
        ve = sp * math.sin(target_yaw_rad)
        vd = sp * math.sin(state.bearing_v) * 2.0

        return VelocityNedYaw(vn, ve, vd, math.degrees(target_yaw_rad))

    def _plan_coast(self, vel, yaw_rad, dt):
        """Fly toward belief estimate with reduced speed."""
        # Use belief bearing/range
        b_h = self.belief.bearing_h
        b_range = self.belief.range_est
        b_v = self.belief.bearing_v

        # Speed: scale by confidence and ticks since detection
        conf_scale = max(0.5, self.belief.confidence)
        range_speed = self._speed_for_range(b_range)
        coast_speed = range_speed * self.coast_speed_frac * conf_scale
        coast_speed = max(coast_speed, self.coast_min_speed)

        cmd_speed = self._px4_cmd(coast_speed, abs(b_h))
        sp = self._smooth(cmd_speed, dt)

        target_yaw_rad = yaw_rad + b_h
        vn = sp * math.cos(target_yaw_rad)
        ve = sp * math.sin(target_yaw_rad)
        vd = sp * math.sin(b_v) * 1.5  # slightly less aggressive vertical

        return VelocityNedYaw(vn, ve, vd, math.degrees(target_yaw_rad))

    def _plan_search(self, yaw_deg, yaw_rad, dt):
        """Structured search: alternating yaw sweeps.

        Instead of continuous rotation (panic sweep), does:
        - Sweep right for 1.5s
        - Sweep left for 1.5s
        - Repeat

        While moving slowly forward in the last known direction.
        """
        self.search_phase_ticks += 1

        # Flip direction every SEARCH_PHASE_DURATION ticks
        if self.search_phase_ticks >= self.SEARCH_PHASE_DURATION:
            self.search_yaw_direction *= -1.0
            self.search_phase_ticks = 0

        # Yaw command: current heading + search rotation
        search_yaw = yaw_deg + self.search_yaw_rate * self.search_yaw_direction * dt

        # Speed: slow but not stopped
        sp = self._smooth(self.search_speed, dt)

        # Direction: use belief's last known bearing if available, else forward
        fly_bearing = self.belief.search_bearing
        fly_yaw_rad = yaw_rad + fly_bearing

        vn = sp * math.cos(fly_yaw_rad)
        ve = sp * math.sin(fly_yaw_rad)
        vd = 0.0

        return VelocityNedYaw(vn, ve, vd, search_yaw)
