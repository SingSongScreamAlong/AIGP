"""Session 16 — Vision-first gate navigation for PX4 SITL.

Replaces hardcoded gate-coordinate chasing with a simulated FPV camera
+ vision navigation loop. The planner never sees absolute gate coords;
it receives bearing/range estimates from the virtual camera with
configurable noise, missed detections, and FOV limits.

Architecture:
  VirtualCamera  — projects gates into camera frame, adds noise
  GateTracker    — maintains detection state, gate sequencing, search mode
  VisionNav      — consumes tracker output, produces velocity commands
  run_trial_vision — drop-in replacement for run_trial() in baseline
"""

import asyncio, time, math, random, json, os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

# ── Noise profiles ────────────────────────────────────────────────
NOISE_PROFILES = {
    'clean': {
        'bearing_sigma_deg': 0.5,    # very low bearing noise
        'range_sigma_frac': 0.02,    # 2% range noise
        'miss_prob': 0.01,           # 1% missed detection
        'fov_h_deg': 120.0,          # wide FOV
        'fov_v_deg': 90.0,
        'max_detect_range': 60.0,    # detect far away
        'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
    },
    'mild': {
        'bearing_sigma_deg': 2.0,
        'range_sigma_frac': 0.08,
        'miss_prob': 0.05,
        'fov_h_deg': 100.0,
        'fov_v_deg': 75.0,
        'max_detect_range': 40.0,
        'min_detect_range': 0.8,
        'bearing_bias_deg': 0.0,
    },
    'harsh': {
        'bearing_sigma_deg': 5.0,
        'range_sigma_frac': 0.15,
        'miss_prob': 0.15,
        'fov_h_deg': 80.0,
        'fov_v_deg': 60.0,
        'max_detect_range': 25.0,
        'min_detect_range': 1.0,
        'bearing_bias_deg': 1.5,     # systematic bias
    },
    # S19w — pessimistic "brutal" profile added to push past the
    # clean/mild/harsh envelope in soak. The triad clean→harsh
    # bottomed out with 100% completion; brutal is the tightening
    # screw: double bearing_sigma, double the miss rate, tighter FOV,
    # shorter detect range, larger systematic bearing bias. Used by
    # `soak.py` to find where V5.1+BeliefNav starts losing races and
    # where ESKF-fusion becomes necessary for the sensor budget we
    # actually expect on hardware (Neros 12MP with partial-obscuration
    # events). Not recommended for run_race.py production flights;
    # expect completion rates < 100% even on small courses.
    'brutal': {
        'bearing_sigma_deg': 10.0,
        'range_sigma_frac': 0.30,
        'miss_prob': 0.35,
        'fov_h_deg': 60.0,
        'fov_v_deg': 45.0,
        'max_detect_range': 15.0,
        'min_detect_range': 1.2,
        'bearing_bias_deg': 3.0,
    },
}


# ── Data structures ───────────────────────────────────────────────
@dataclass
class GateDetection:
    """What the camera reports about a single gate."""
    gate_idx: int               # which gate (sim ground truth, for bookkeeping only)
    bearing_h_deg: float        # horizontal bearing from drone forward axis (+ = right)
    bearing_v_deg: float        # vertical bearing (+ = down)
    range_est: float            # estimated range in meters
    angular_size_deg: float     # apparent gate size (proxy for range)
    confidence: float           # 0..1
    in_fov: bool                # whether gate is within camera FOV


@dataclass
class TrackerState:
    """What the gate tracker believes about the current target."""
    target_idx: int             # which gate we're trying to reach
    detected: bool              # do we currently see it?
    bearing_h: float            # horizontal bearing to target (rad)
    bearing_v: float            # vertical bearing to target (rad)
    range_est: float            # estimated range
    confidence: float
    frames_since_seen: int      # consecutive frames without detection
    search_mode: bool           # lost the gate, searching


# ── Virtual FPV Camera ────────────────────────────────────────────
class VirtualCamera:
    """Simulates an FPV camera mounted on the drone.

    Takes drone pose (position + heading from telemetry) and gate positions
    (known to sim only), produces noisy gate detections as if from a real
    vision pipeline.

    The navigator ONLY sees GateDetection objects, never raw coordinates.
    """

    GATE_PHYSICAL_SIZE = 2.0  # meters, approximate gate opening diameter

    def __init__(self, gates: list, noise_profile: str = 'clean', seed: int = 42):
        """
        Args:
            seed: RNG seed for noise + miss_prob sampling. Hardcoded to 42
                for over a year prior to S19w — soak variance came from
                wall-clock-paced PX4 SITL jitter, which masked a latent
                sensitivity to noise-sample trajectories. Modern sandbox
                soak (`soak.py`) varies this across trials to surface
                detector-driven failure modes that the single-seed bench
                missed.
        """
        self.gates = gates  # [(N, E, D), ...] — sim ground truth, not exposed to nav
        self.np = NOISE_PROFILES[noise_profile]
        self.rng = random.Random(int(seed))

    def observe(self, pos: list, vel: list, yaw_deg: float) -> List[GateDetection]:
        """Produce camera observations for all gates.

        Args:
            pos: [N, E, D] drone position from telemetry
            vel: [vN, vE, vD] drone velocity
            yaw_deg: drone heading in degrees (0=North, 90=East)

        Returns:
            List of GateDetection for each gate visible in FOV
        """
        detections = []
        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        fov_h_rad = math.radians(self.np['fov_h_deg'] / 2)
        fov_v_rad = math.radians(self.np['fov_v_deg'] / 2)

        for gi, g in enumerate(self.gates):
            # Vector from drone to gate in NED
            dn = g[0] - pos[0]
            de = g[1] - pos[1]
            dd = g[2] - pos[2]

            true_range = math.sqrt(dn*dn + de*de + dd*dd)
            if true_range < 0.01:
                continue

            # Range check
            if true_range > self.np['max_detect_range']:
                continue
            if true_range < self.np['min_detect_range']:
                continue

            # Transform to body frame (forward = yaw direction)
            # Body X = forward, Body Y = right, Body Z = down
            body_x = dn * cos_y + de * sin_y    # forward
            body_y = -dn * sin_y + de * cos_y   # right
            body_z = dd                          # down (NED convention)

            # Horizontal bearing (+ = right of center)
            bearing_h = math.atan2(body_y, body_x)
            # Vertical bearing (+ = below center)
            horiz_dist = math.sqrt(body_x*body_x + body_y*body_y)
            bearing_v = math.atan2(body_z, horiz_dist) if horiz_dist > 0.01 else 0.0

            # FOV check
            in_fov = abs(bearing_h) <= fov_h_rad and abs(bearing_v) <= fov_v_rad
            if not in_fov:
                continue

            # Behind the drone
            if body_x < 0:
                continue

            # Missed detection
            if self.rng.random() < self.np['miss_prob']:
                continue

            # Add noise
            noisy_bearing_h = bearing_h + math.radians(
                self.rng.gauss(self.np['bearing_bias_deg'], self.np['bearing_sigma_deg'])
            )
            noisy_bearing_v = bearing_v + math.radians(
                self.rng.gauss(0, self.np['bearing_sigma_deg'] * 0.7)
            )

            range_noise = 1.0 + self.rng.gauss(0, self.np['range_sigma_frac'])
            noisy_range = true_range * range_noise

            # Angular size (inversely proportional to range)
            true_angular = math.degrees(2 * math.atan(self.GATE_PHYSICAL_SIZE / (2 * true_range)))
            noisy_angular = true_angular * (1.0 + self.rng.gauss(0, self.np['range_sigma_frac']))

            # Confidence: drops with range and off-axis angle
            range_conf = max(0, 1.0 - true_range / self.np['max_detect_range'])
            angle_conf = max(0, 1.0 - abs(bearing_h) / fov_h_rad)
            confidence = range_conf * angle_conf * (1.0 - self.np['miss_prob'])

            detections.append(GateDetection(
                gate_idx=gi,
                bearing_h_deg=math.degrees(noisy_bearing_h),
                bearing_v_deg=math.degrees(noisy_bearing_v),
                range_est=noisy_range,
                angular_size_deg=noisy_angular,
                confidence=confidence,
                in_fov=True,
            ))

        return detections


# ── Gate Tracker ──────────────────────────────────────────────────
class GateTracker:
    """Tracks which gate we're targeting and manages sequencing.

    Does NOT know gate coordinates. Works entirely from camera detections.
    Determines gate passage from range crossing a threshold.
    """

    PASSAGE_RANGE = 2.5        # gate passed when range < this
    SEARCH_TIMEOUT_FRAMES = 50 # ~1s at 50Hz — enter search after this many missed frames

    def __init__(self, num_gates: int):
        self.num_gates = num_gates
        self.current_target = 0
        self.frames_since_seen = 0
        self.last_range = None
        self.last_bearing_h = 0.0
        self.last_bearing_v = 0.0
        self.last_confidence = 0.0
        self.gates_passed = 0
        self.search_mode = False
        self.passage_log = []

    def update(self, detections: List[GateDetection], dt: float) -> TrackerState:
        """Process new camera frame detections.

        Returns TrackerState describing what we believe about the current target.
        """
        if self.current_target >= self.num_gates:
            # All gates passed
            return TrackerState(
                target_idx=self.current_target, detected=False,
                bearing_h=0, bearing_v=0, range_est=0, confidence=0,
                frames_since_seen=self.frames_since_seen, search_mode=False,
            )

        # Find detection matching current target
        target_det = None
        for d in detections:
            if d.gate_idx == self.current_target:
                target_det = d
                break

        # If not found by index, look for closest detection in roughly
        # the expected direction (handles cases where gate_idx isn't known
        # to the tracker in a real system — here we use it as a proxy)
        # In a real vision system, the tracker would use appearance matching.

        if target_det is not None:
            self.frames_since_seen = 0
            self.search_mode = False
            self.last_bearing_h = math.radians(target_det.bearing_h_deg)
            self.last_bearing_v = math.radians(target_det.bearing_v_deg)
            self.last_range = target_det.range_est
            self.last_confidence = target_det.confidence

            # Check gate passage
            if target_det.range_est < self.PASSAGE_RANGE:
                self.passage_log.append({
                    'gate': self.current_target,
                    'time': time.time(),
                    'range': target_det.range_est,
                })
                self.gates_passed += 1
                self.current_target += 1
                self.frames_since_seen = 0
                self.last_range = None
                # Immediately look for next gate
                return self.update(detections, dt)
        else:
            self.frames_since_seen += 1
            if self.frames_since_seen > self.SEARCH_TIMEOUT_FRAMES:
                self.search_mode = True

        return TrackerState(
            target_idx=self.current_target,
            detected=(target_det is not None),
            bearing_h=self.last_bearing_h,
            bearing_v=self.last_bearing_v,
            range_est=self.last_range if self.last_range else 20.0,
            confidence=self.last_confidence,
            frames_since_seen=self.frames_since_seen,
            search_mode=self.search_mode,
        )


# ── Vision Navigator ──────────────────────────────────────────────
class VisionNav:
    """Converts camera-derived gate observations into velocity commands.

    No access to gate coordinates. Steers based on:
    - bearing to target gate (from camera)
    - estimated range (from camera angular size)
    - speed management from V5.1 policy (adapted for bearing-based input)

    On lost detection: holds last known bearing, reduces speed.
    On search mode: enters a slow yaw scan to re-acquire.
    """

    def __init__(self, max_speed=12.0, cruise_speed=9.0):
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.px4_speed_ceiling = 9.5
        self.last_cmd_speed = 0.0
        self.mission_start_time = None

        # Speed management from V5.1
        self.px4_util_straight = 0.78
        self.px4_util_turn = 0.55
        self.px4_max_decel = 4.0
        self.base_blend = 1.5

        # Search behavior
        self.search_yaw_rate = 30.0  # deg/s yaw scan when lost
        self.search_speed = 2.0       # slow forward during search
        self.coast_speed_frac = 0.5   # fraction of speed to hold when coasting (no detection)
        self.coast_frames_max = 25    # coast this many frames before slowing further

    def _speed_for_range(self, range_est: float, confidence: float) -> float:
        """Determine desired speed based on estimated range to gate."""
        if range_est < 4.0:
            # Close to gate — slow for passage
            return max(3.0, self.cruise_speed * 0.5)
        elif range_est < 8.0:
            # Medium range — moderate speed
            return self.cruise_speed * 0.8
        else:
            # Far — cruise
            return self.cruise_speed

    def _px4_cmd(self, desired: float, turn_angle: float = 0.0) -> float:
        """Apply PX4 utilization correction (from V5.1)."""
        tr = min(abs(turn_angle) / math.pi, 1.0)
        util = self.px4_util_straight * (1 - tr) + self.px4_util_turn * tr
        return min(desired / max(util, 0.3), self.max_speed)

    def _smooth(self, target: float, dt: float = 0.02) -> float:
        """Smooth speed transitions."""
        max_rate = 12.0  # m/s per second
        max_delta = max_rate * dt
        if target > self.last_cmd_speed:
            s = min(target, self.last_cmd_speed + max_delta)
        else:
            s = max(target, self.last_cmd_speed - max_delta * 1.5)
        self.last_cmd_speed = s
        return s

    def plan(self, tracker_state: TrackerState, pos: list, vel: list,
             yaw_deg: float, dt: float = 0.02) -> VelocityNedYaw:
        """Produce velocity command from tracker state.

        Args:
            tracker_state: current belief about target gate
            pos: [N, E, D] from IMU/estimator (we use altitude)
            vel: [vN, vE, vD] from IMU
            yaw_deg: current heading
            dt: control timestep

        Returns:
            VelocityNedYaw command for PX4 offboard
        """
        if self.mission_start_time is None:
            self.mission_start_time = time.time()

        yaw_rad = math.radians(yaw_deg)

        # ── Search mode: slow yaw scan ──
        if tracker_state.search_mode:
            # Rotate to find the gate
            search_yaw = yaw_deg + self.search_yaw_rate * dt
            sp = self._smooth(self.search_speed, dt)
            # Fly slowly forward while rotating
            vn = sp * math.cos(math.radians(search_yaw))
            ve = sp * math.sin(math.radians(search_yaw))
            vd = 0.0
            return VelocityNedYaw(vn, ve, vd, search_yaw)

        # ── No detection but not yet in search mode: coast ──
        if not tracker_state.detected:
            # Hold last bearing, reduce speed
            coast_frac = max(0.3, 1.0 - tracker_state.frames_since_seen / self.coast_frames_max)
            sp = self._smooth(self.last_cmd_speed * coast_frac, dt)
            # Fly toward last known bearing
            fly_yaw_rad = yaw_rad + tracker_state.bearing_h
            vn = sp * math.cos(fly_yaw_rad)
            ve = sp * math.sin(fly_yaw_rad)
            vd = 0.0
            return VelocityNedYaw(vn, ve, vd, math.degrees(fly_yaw_rad))

        # ── Active detection: steer toward gate ──
        desired_speed = self._speed_for_range(tracker_state.range_est, tracker_state.confidence)
        cmd_speed = self._px4_cmd(desired_speed, abs(tracker_state.bearing_h))
        sp = self._smooth(cmd_speed, dt)

        # Target direction: current heading + bearing offset
        target_yaw_rad = yaw_rad + tracker_state.bearing_h

        # NED velocity
        vn = sp * math.cos(target_yaw_rad)
        ve = sp * math.sin(target_yaw_rad)

        # Vertical: use bearing_v to maintain altitude toward gate
        # Proportional control on vertical bearing
        vd = sp * math.sin(tracker_state.bearing_v) * 2.0

        target_yaw_deg = math.degrees(target_yaw_rad)

        return VelocityNedYaw(vn, ve, vd, target_yaw_deg)


# ── Run trial (vision mode) ──────────────────────────────────────
RACE_PARAMS = {
    'MPC_ACC_HOR': 10.0, 'MPC_ACC_HOR_MAX': 10.0, 'MPC_JERK_AUTO': 30.0,
    'MPC_JERK_MAX': 50.0, 'MPC_TILTMAX_AIR': 70.0, 'MPC_XY_VEL_MAX': 15.0,
    'MPC_Z_VEL_MAX_UP': 5.0,
}

async def run_trial_vision(gates, noise_profile='clean', threshold=2.5,
                           gate_alt_frac=0.95, gate_vz_max=0.3, gate_timeout=10.0):
    """Run one lap using vision navigation (no direct gate coord access).

    The VisionNav receives only camera observations, never gate coordinates.
    Gate coordinates are known only to the VirtualCamera (simulating the real world).
    """
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    for n, v in RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    # Telemetry state
    pos = [0, 0, 0]
    vel = [0, 0, 0]
    yaw_deg = 0.0

    async def telem_loop():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]

    async def heading_loop():
        nonlocal yaw_deg
        async for h in drone.telemetry.heading():
            yaw_deg = h.heading_deg

    asyncio.ensure_future(telem_loop())
    asyncio.ensure_future(heading_loop())
    await asyncio.sleep(0.5)

    # Takeoff
    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    _wait_start = time.time()
    while True:
        if (time.time() - _wait_start) >= gate_timeout: break
        if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break
        await asyncio.sleep(0.05)

    t_offboard_wait = round(time.time() - _wait_start, 3)

    # Start offboard
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    # Build vision stack — camera gets gate coords (sim truth), nav does NOT
    camera = VirtualCamera(gates, noise_profile)
    tracker = GateTracker(len(gates))
    nav = VisionNav()

    t0 = time.time()
    dt = 1 / 50
    max_spd = 0
    cmds = []
    achs = []
    det_log = []    # log detection stats per frame
    search_frames = 0
    coast_frames = 0

    while tracker.current_target < len(gates):
        if time.time() - t0 > 90:
            print(f'      TIMEOUT@gate{tracker.current_target}')
            break

        # Camera observes
        detections = camera.observe(pos, vel, yaw_deg)

        # Tracker updates
        state = tracker.update(detections, dt)

        # Nav produces command
        cmd = nav.plan(state, pos, vel, yaw_deg, dt)

        # Stats
        cspd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        aspd = math.sqrt(vel[0]**2 + vel[1]**2)
        cmds.append(cspd)
        achs.append(aspd)
        if aspd > max_spd:
            max_spd = aspd

        if state.search_mode:
            search_frames += 1
        elif not state.detected:
            coast_frames += 1

        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0

    # Cleanup
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land()
    await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    await asyncio.sleep(2)
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    await asyncio.sleep(2)

    # Results
    avg_cmd = sum(cmds) / len(cmds) if cmds else 0
    avg_ach = sum(achs) / len(achs) if achs else 0
    total_frames = len(cmds)

    return {
        'completed': tracker.gates_passed >= len(gates),
        'time': round(total, 3),
        'gates_passed': tracker.gates_passed,
        'max_spd': round(max_spd, 2),
        'avg_cmd': round(avg_cmd, 2),
        'avg_ach': round(avg_ach, 2),
        'noise_profile': noise_profile,
        'search_frames': search_frames,
        'coast_frames': coast_frames,
        'total_frames': total_frames,
        'search_pct': round(100 * search_frames / max(total_frames, 1), 1),
        'coast_pct': round(100 * coast_frames / max(total_frames, 1), 1),
        't_offboard_wait': t_offboard_wait,
    }


# ── Ground truth baseline (same interface) ────────────────────────
async def run_trial_groundtruth(planner, gates, threshold=2.5,
                                gate_alt_frac=0.95, gate_vz_max=0.3, gate_timeout=10.0):
    """Run one lap using ground-truth coordinates (existing V5.1 approach).

    This is the control arm — same as the current baseline.
    """
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    for n, v in RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0, 0, 0]
    vel = [0, 0, 0]

    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]

    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    _wait_start = time.time()
    while True:
        if (time.time() - _wait_start) >= gate_timeout: break
        if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break
        await asyncio.sleep(0.05)

    t_offboard_wait = round(time.time() - _wait_start, 3)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    gi = 0
    t0 = time.time()
    dt = 1 / 50
    max_spd = 0
    cmds = []
    achs = []

    while gi < len(gates):
        if time.time() - t0 > 90:
            break
        g = gates[gi]
        d = math.sqrt((g[0]-pos[0])**2 + (g[1]-pos[1])**2 + (g[2]-pos[2])**2)
        if d < threshold:
            gspd = math.sqrt(vel[0]**2 + vel[1]**2)
            planner.on_gate_passed(gspd)
            gi += 1
            continue
        ng = gates[gi+1] if gi+1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        cspd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        aspd = math.sqrt(vel[0]**2 + vel[1]**2)
        cmds.append(cspd)
        achs.append(aspd)
        if aspd > max_spd: max_spd = aspd
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land()
    await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    await asyncio.sleep(2)
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    await asyncio.sleep(2)

    avg_cmd = sum(cmds) / len(cmds) if cmds else 0
    avg_ach = sum(achs) / len(achs) if achs else 0

    return {
        'completed': gi >= len(gates),
        'time': round(total, 3),
        'gates_passed': gi,
        'max_spd': round(max_spd, 2),
        'avg_cmd': round(avg_cmd, 2),
        'avg_ach': round(avg_ach, 2),
        'noise_profile': 'groundtruth',
        'search_frames': 0,
        'coast_frames': 0,
        'total_frames': len(cmds),
        'search_pct': 0,
        'coast_pct': 0,
        't_offboard_wait': t_offboard_wait,
    }
