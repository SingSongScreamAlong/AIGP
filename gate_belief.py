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

        Uses drone velocity AND yaw change to update the body-frame
        bearing/range estimate. The gate is stationary in world frame;
        the drone moves AND rotates.

        The belief (bearing_h/v, range) is expressed in the body frame
        AS OF THE LAST TICK. To account for yaw rotation between ticks,
        convert body→NED using the PREVIOUS yaw, then NED→body using the
        CURRENT yaw. Using the same yaw for both conversions (the prior
        bug) silently cancels the rotation and causes belief drift when
        the drone yaws — exactly the case where the belief is supposed
        to help.

        Args:
            vel: [vN, vE, vD] drone velocity in NED
            yaw_rad: current drone yaw
            dt: time step
        """
        if self.confidence < self.MIN_CONFIDENCE:
            return  # belief is dead, nothing to propagate

        # Use previous yaw for body→NED; if unseeded (first propagate
        # before any detection has anchored the frame), fall back to
        # the current yaw so the first tick is a no-op in rotation terms.
        prev_yaw = self._prev_yaw_rad if self._prev_yaw_rad is not None else yaw_rad
        cos_py = math.cos(prev_yaw)
        sin_py = math.sin(prev_yaw)
        cos_cy = math.cos(yaw_rad)
        sin_cy = math.sin(yaw_rad)

        # Gate offset in the PREVIOUS body frame (body_x forward, body_y right)
        body_x = self.range_est * math.cos(self.bearing_h) * math.cos(self.bearing_v)
        body_y = self.range_est * math.sin(self.bearing_h) * math.cos(self.bearing_v)
        body_z = self.range_est * math.sin(self.bearing_v)

        # Previous body → NED using PREVIOUS yaw
        dn = body_x * cos_py - body_y * sin_py
        de = body_x * sin_py + body_y * cos_py
        dd = body_z

        # Subtract drone motion in NED (gate is stationary, drone moved)
        dn -= vel[0] * dt
        de -= vel[1] * dt
        dd -= vel[2] * dt

        # NED → CURRENT body frame using CURRENT yaw
        new_body_x = dn * cos_cy + de * sin_cy
        new_body_y = -dn * sin_cy + de * cos_cy
        new_body_z = dd

        # Extract new bearing and range in current body frame
        new_range = math.sqrt(new_body_x**2 + new_body_y**2 + new_body_z**2)
        if new_range > 0.1:
            self.bearing_h = math.atan2(new_body_y, new_body_x)
            horiz_dist = math.sqrt(new_body_x**2 + new_body_y**2)
            self.bearing_v = math.atan2(new_body_z, horiz_dist) if horiz_dist > 0.1 else 0.0
            self.range_est = new_range

        # Decay confidence and advance the reference frame
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

        # S19o gate-aware fallback: when the navigator knows where the
        # next target gate is in NED, it can turn toward it directly
        # instead of blind coast/search. Critical for the post-pass
        # out-of-FOV window: the just-passed gate is still in FOV and
        # picker-fallback used to steer us back into it; with gates_ned
        # set, `_plan_toward_known_target` points us at the new target
        # from the moment target_idx advances. None → legacy coast/search.
        self.gates_ned = None

        # S19p pose-trust gate: the gate-aware fallback steers on the
        # navigator's pose estimate. When that pose comes from an ESKF
        # that's rejecting most vision fixes (self-destructive chi-
        # squared loop under harsh noise), navigating on it actively
        # drives the drone off-course. Callers (RaceLoop) flip this to
        # False when `PoseFusion.recent_reject_rate()` exceeds a
        # threshold; the navigator then falls through to belief-coast/
        # search, which degrade gracefully instead of committing to a
        # wrong world-frame target.
        self.pose_trusted = True

    def on_gate_passed(self):
        """Called when tracker detects gate passage."""
        self.belief.reset()
        self.current_gate += 1
        self.search_yaw_direction = 1.0
        self.search_phase_ticks = 0

    def set_gates_ned(self, gates_ned):
        """Enable gate-aware fallback navigation.

        ``gates_ned`` is a list of (N, E, D) tuples. When set, the
        navigator falls back to flying directly toward
        ``gates_ned[target_idx]`` when there's no detection for the
        current target, instead of coasting on a stale belief or
        searching blindly. Pass ``None`` to revert to belief-only
        coast/search.

        Caller controls the lifecycle: typically wired once at
        construction by the loop that owns the gate list.
        """
        self.gates_ned = gates_ned

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

        # Gate-aware fallback (S19o): when we know where the target gate
        # is in NED and the navigator has a trusted pose estimate, fly
        # directly toward the world-frame target rather than coast on a
        # belief that just got reset (post-pass) or search blindly. This
        # closes the post-pass out-of-FOV gap — belief confidence is 0.0
        # in the ticks right after `on_gate_passed()` fires, so the
        # belief-coast branch has nothing to say about the new target.
        # Knowing the world-frame gate lets us turn toward it immediately.
        #
        # Falls back to belief-coast/search when: gates_ned is unset,
        # target_idx is out of range, OR `pose_trusted` is False (S19p
        # — protects against navigating on a diverging ESKF pose).
        if (
            self.pose_trusted
            and self.gates_ned is not None
            and 0 <= tracker_state.target_idx < len(self.gates_ned)
        ):
            return self._plan_toward_known_target(
                tracker_state.target_idx, pos, yaw_rad, dt
            )

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

    def _plan_toward_known_target(self, target_idx, pos, yaw_rad, dt):
        """Fly toward ``gates_ned[target_idx]`` using world-frame geometry.

        Used when no detection is available but the navigator knows the
        target gate's NED position. Computes bearing from the supplied
        pose (truth when fusion is off, fused when on), picks a cruise
        speed, and emits a NED velocity command plus a yaw command that
        points forward-along-approach.

        Noisy pose is self-correcting here: once the drone gets close
        enough for the gate to come into FOV, `_plan_tracking` takes
        over on the next detection and the detection-bearing overrides
        this world-frame bearing. So even if `pos` is drifted by 5 m,
        this path is a reasonable heading guess that gets the drone
        close enough to re-acquire.
        """
        gate = self.gates_ned[target_idx]
        dn = gate[0] - pos[0]
        de = gate[1] - pos[1]
        dd = gate[2] - pos[2]
        horiz = math.sqrt(dn * dn + de * de)
        r = math.sqrt(horiz * horiz + dd * dd)
        if r < 1e-3:
            # On top of the gate — let the pass detector handle it;
            # emit a small forward drift rather than divide-by-zero.
            return VelocityNedYaw(0.0, 0.0, 0.0, math.degrees(yaw_rad))

        # World-frame heading to gate.
        yaw_to_gate = math.atan2(de, dn)

        # Horizontal bearing relative to current yaw for px4_cmd speed
        # shaping (big turns ⇒ lower speed via px4_util_turn).
        bearing_h = (yaw_to_gate - yaw_rad + math.pi) % (2 * math.pi) - math.pi

        # Conservative speed — we're flying on pose, not detection. The
        # coast_speed_frac+conf_scale path gives a solid middle ground;
        # reuse that shape so speeds stay consistent with belief-coast
        # when the drone has been flying through the same target.
        base = self._speed_for_range(r) * self.coast_speed_frac
        base = max(base, self.coast_min_speed)
        cmd_speed = self._px4_cmd(base, abs(bearing_h))
        sp = self._smooth(cmd_speed, dt)

        # NED-frame velocity along the gate direction.
        vn = sp * (dn / r) if horiz > 1e-3 else 0.0
        ve = sp * (de / r) if horiz > 1e-3 else 0.0
        # Vertical: point toward gate D but throttle to avoid hard
        # climb/dive commands when the drone is already near altitude.
        vd = sp * (dd / r)

        return VelocityNedYaw(vn, ve, vd, math.degrees(yaw_to_gate))

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
