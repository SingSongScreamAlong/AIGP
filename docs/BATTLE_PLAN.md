# AI Grand Prix - Battle Plan

**Objective:** Win the 500K autonomous drone racing competition.
**Competitor:** Conrad Weeden (solo)
**Date:** April 6, 2026

---

## The Competition

| Phase | When | What |
|-------|------|------|
| Virtual Qualifier 1 | May-June 2026 | Simple course, highlighted gates, desaturated environment, easy perception |
| Virtual Qualifier 2 | June-July 2026 | Complex environment, 3D-scanned, no visual aids, harder lighting |
| Physical Qualifier | September 2026 | Real Neros drones, Southern California, indoor |
| AI Grand Prix Ohio | November 2026 | Finals, indoor, spectators/flash, 500K on the line |

**Scoring:** Time-trial. Fastest valid time (all gates passed in order) wins. No head-to-head.

---

## The Interface

From tech spec VADR-TS-001 (March 9, 2026):

**Protocol:** MAVLink v2 over UDP via MAVSDK

**What the sim sends you:**

| Message | Data |
|---------|------|
| HEARTBEAT | Connection status |
| ATTITUDE | Roll, pitch, yaw, angular rates |
| HIGHRES_IMU | Accelerometer, gyroscope, magnetometer |
| ODOMETRY | Position + velocity in local NED frame |
| TIMESYNC | Clock synchronization |

**What you send back:**

| Message | Use |
|---------|-----|
| SET_POSITION_TARGET_LOCAL_NED | Position/velocity setpoints (safer, easier) |
| SET_ATTITUDE_TARGET | Direct attitude commands: throttle, roll, pitch, yaw (aggressive, fast) |
| HEARTBEAT | Keep-alive (min 2 Hz) |

**Timing:** 120 Hz physics. 50-120 Hz command rate recommended.

**What you DONT get:** GPS, absolute position, depth, LiDAR, engine RPM, battery state.

**What you DO get that matters:** ODOMETRY gives local position + velocity. This is the spine.

**Vision:** Forward-facing FPV camera, ~12MP wide-angle, monocular. Spec TBD. No depth.

**Compute:** ~100 TOPS on the physical drone. Sim runs on mid-tier Windows PC with dedicated GPU.

**Runtime:** Python 3.14.2 confirmed. C/Cython/compiled extensions allowed.

---

## Architecture

Vision + Telemetry -> Perception -> Planning -> Control -> Pilot Commands -> Stabilized Controller

MAVSDK TELEMETRY -> STATE ESTIMATOR -> PLANNER -> COMMANDER -> MAVSDK COMMAND
VISION STREAM -> GATE DETECTOR -> GATE SEQUENCER (feeds into planner)
RECOVERY MODULE (monitors all, intervenes on failure)

---

## Module Breakdown

### 1. MAVSDK Connection Layer
Foundation. MAVLink comms with the sim.
- Connect to sim UDP endpoint
- Maintain HEARTBEAT at >=2 Hz
- Subscribe to ODOMETRY, ATTITUDE, HIGHRES_IMU
- Send SET_ATTITUDE_TARGET or SET_POSITION_TARGET_LOCAL_NED
- Start with position mode (easier), graduate to attitude mode (faster)
- MAVSDK-Python is async (asyncio)

### 2. State Estimator
- State: position (x,y,z), velocity (vx,vy,vz), orientation (roll,pitch,yaw) + rates
- Q1: ODOMETRY gives position+velocity directly. Thin wrapper.
- Q2/Physical: Fuse ODOMETRY + IMU + visual gate PnP via complementary filter or EKF

### 3. Gate Detector
- Q1 (highlighted gates): Color thresholding or contour detection. Classical CV wins.
- Q2 (realistic): YOLOv8-pose with keypoint regression for 4 corners -> PnP -> 3D pose
- Output: bbox, confidence, 4 corner keypoints, 3D position+orientation via PnP

### 4. Gate Sequencer
- Track gate order, current target, detect gate passes
- All gates must be passed in correct order (failed run otherwise)
- Multiple visible gates -> always prioritize NEXT gate

### 5. Trajectory Planner
- Phase 1: Direct-to-gate with fly-through offset
- Phase 2: Look-ahead 2-3 gates, smooth spline through centers
- Phase 3: Minimum-time trajectory using drone dynamics model

### 6. Controller
- Position mode (starter): SET_POSITION_TARGET_LOCAL_NED
- Attitude mode (speed): SET_ATTITUDE_TARGET with PID cascade
- Position error -> velocity setpoint -> attitude setpoint -> command
- Log EVERYTHING: commanded vs actual position, velocity, attitude

### 7. Recovery Module
- Lost gate: hover, yaw scan, reacquire
- Bad approach: abort, pull back, re-approach
- Oscillation: reduce gains, stabilize, re-plan
- State uncertainty: trust telemetry, reduce speed

---

## Development Phases

### Phase 0: Foundation (Now -> April 20)
Goal: MAVSDK control loop working against PX4 SITL
- Install PX4 SITL + Gazebo as stand-in sim
- Install MAVSDK-Python, connect to SITL
- Read ODOMETRY, ATTITUDE, HIGHRES_IMU at 50 Hz
- Fly to fixed waypoint with SET_POSITION_TARGET_LOCAL_NED
- Fly forward with SET_ATTITUDE_TARGET
- Fly a square pattern autonomously
- Build async main loop: telemetry -> plan -> command at 50 Hz
- Set up telemetry logging
Deliverable: Python script 

### Phase 0.5: DCL Gate Vision Training (Parallel with Phase 0)
Goal: Train gate detection on real DCL visuals before the competition sim exists

The competition sim is almost certainly built on the same engine as DCL The Game (5 dollars on Steam). The gates, lighting, and textures are likely identical or very close. This gives us a massive shortcut.

Setup:
- Buy DCL The Game on Steam (Windows machine)
- Build Python screen-capture pipeline (dxcam on Windows for low-latency)
- Capture game window at 30+ fps, feed frames as FPV camera stream
- Fly manually through courses, recording footage

Data collection + labeling:
- Record 30+ min of manual flight through various courses
- Extract frames at 5-10 fps = thousands of gate images
- Bootstrap labels with color/contour detection
- Clean up in CVAT or Roboflow
- Split train/val

Training:
- Retrain YOLOv8-pose on real DCL gate images (replace MuJoCo synthetic data)
- Train keypoint model for 4 gate corners on DCL gate geometry
- Validate PnP pose estimation with DCL gate dimensions
- Build real-time overlay: capture -> detect -> draw bbox/keypoints -> display

Why this matters:
- Eliminates sim-to-real gap for gate appearance - train on actual engine visuals
- MuJoCo synthetic data was guessing at gate geometry/colors. This is the real thing.
- By May sim drop, detector already knows what DCL gates look like
- 5 dollar game = cheapest, highest-value training data in the competition
- Also gives intuition for DCL physics model and gate layouts

Deliverable: Gate detector trained on real DCL visuals with live screen-capture demo.
that connects via MAVSDK, receives telemetry, flies a pattern.

### Phase 1: Gate Navigation (April 20 -> May sim release)
Goal: Fly through gate sequence using position commands
- Gate sequencer: hard-code gate list, fly through in order
- Gate pass detection
- Direct-to-gate trajectory with fly-through offset
- 100% completion rate, every run
- Basic recovery: hover if lost, stabilize on oscillation
Deliverable: Drone reliably completes multi-gate course. Slow is fine.

### Phase 2: Sim Integration (May sim release -> June)
Goal: Running in actual DCL sim, passing all gates
- Download competition sim + course
- Swap MAVSDK connection to DCL sim
- Integrate vision stream
- Build Q1 gate detector (simple CV for highlighted gates)
- End-to-end: start -> detect -> fly through all -> finish
- 100% clean run completion rate
- Build replay/analysis tooling
Deliverable: Complete clean runs on Q1 course. Every time.

### Phase 3: Speed (June -> early July)
Goal: Fastest valid time on Q1
- Profile time loss by sector
- Look-ahead trajectory (2-3 gates ahead)
- Racing line optimization with smooth splines
- Graduate to attitude control
- 100+ automated trials, track median/p5/p95 lap times
- Target: top 10% of leaderboard
Deliverable: Fast, consistent Q1 times with zero reliability drop.

### Phase 4: Q2 Hardening (July)
Goal: Survive harder environment
- YOLOv8-pose retrained on sim data
- Confidence thresholding, reject bad detections
- Visual dropout handling, maintain trajectory on stale data
- Perturbation testing: latency jitter, noise, dropped frames
- Fallback: dead reckoning on telemetry if vision fails
Deliverable: Stack survives Q2 with minimal time loss.

### Phase 5: Physical Prep (August -> September)
Goal: Sim-to-real transfer
- Inject realistic noise, dynamics mismatch, latency
- Test worst-case perturbation combinations
- One-command deployment to physical drone
Deliverable: Stack that just works on real hardware.

---

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| Communication | MAVSDK-Python | Official MAVLink SDK, async, battle-tested |
| Gate Detection Q1 | OpenCV classical CV | Highlighted gates = simple is fast |
| Gate Detection Q2 | YOLOv8-pose | Keypoint regression, already partially trained |
| Pose Estimation | OpenCV solvePnP | 4 corners + known gate size -> 3D pose |
| Trajectory | NumPy + custom | B-spline or minimum-snap through waypoints |
| Controller | Custom PID cascade | Position -> velocity -> attitude -> commands |
| Logging | Custom JSON/CSV | Every state + command + detection, every frame |
| Testing | Custom harness | Automated runs, perturbation injection, stats |
| Performance | Cython or C | Where Python is too slow |

---

## Solo Strategy

1. Leverage ODOMETRY hard. State estimation is mostly solved for Q1.
2. Minimal vision for Q1. Highlighted gates = classical CV. Save ML for Q2.
3. Automate everything. Harness runs sim, logs results, reports stats.
4. Steal from open-source. MAVSDK examples, PX4 controllers, traj opt libs.
5. Focus on the bottleneck. One thing loses the most time. Find it, fix it.
6. Ship early, iterate fast. Working slow > half-built fast.

---

## What We Already Have

| Asset | Status | Use |
|-------|--------|-----|
| YOLOv8 gate detector | Trained | Useful for Q2, overkill for Q1 |
| YOLOv8-pose keypoint model | Trained (0.27 mAP50) | Retrain on sim data for Q2 |
| PnP pose estimation | Built | Directly usable |
| Synthetic data generator | Built | Replace with sim data |
| Tech spec PDF | Downloaded | Reference doc |

---

## Critical Deadlines

| Date | Event | Our Target |
|------|-------|------------|
| April 20 | - | MAVSDK skeleton flying in PX4 SITL |
| May TBD | Sim release | Same-day integration, first run in 48h |
| May-June | VQ1 opens | Submit clean runs, iterate on speed |
| June-July | VQ2 opens | Hardened perception, maintain completion |
| End July | VQ2 cutoff | Top qualifying time locked |
| September | Physical qualifier | Robust tested stack |
| November | AI Grand Prix Ohio | Win |

---

## Next Action

Right now: Install MAVSDK-Python and PX4 SITL. Get a drone flying a square in simulation.
