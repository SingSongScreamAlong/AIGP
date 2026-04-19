# AI Grand Prix - Competition Strategy

Conrad Weeden | Solo Entry | April 2026

---

## 1. THE SITUATION

### What We're Up Against
- **Competition:** Anduril AI Grand Prix, $500K prize pool + Anduril job
- **Format:** Fully autonomous drone racing. Zero human control. Software only wins.
- **Hardware:** Identical Neros drones for all teams. NVIDIA Jetson Orin NX compute. Single FPV camera (12MP wide-angle). IMU (accel + gyro). No LiDAR.
- **Language:** Python-based AI algorithms submitted to DCL platform
- **Team size:** Up to 8 people. We are 1.

### Timeline (Hard Deadlines)

| Date | Event |
|------|-------|
| Now (Apr 6) | Building foundation |
| Late April | Interface specs expected (from March release) |
| May 2026 | **DCL simulator drops** - this is the starting gun |
| May-July 2026 | Virtual qualification on DCL sim |
| End of July | Round 2 cutoff |
| September 2026 | In-person physical qualifier (Southern California, 2 weeks) |
| November 2026 | AI Grand Prix Championship (Columbus, Ohio) |

### What Decides the Winner

This is purely a software competition on identical hardware. The drone that navigates the course fastest without crashing wins. That means:

1. **Fastest gate detection** - see gates, know where they are in 3D
2. **Best state estimation** - know exactly where the drone is at all times
3. **Optimal trajectory** - fly the shortest, fastest path through gates
4. **Robust control** - execute that trajectory at high speed without crashing
5. **Generalization** - work on tracks never seen before

### The Competition (Who We're Racing Against)

- University research labs with teams of 8 PhD students
- Defense/robotics companies building drone autonomy for a living
- A2RL veterans who have already done this in Abu Dhabi

### Our Advantage

- Solo means zero coordination overhead, zero politics, total focus
- We can move fast and iterate faster than committees
- We're building the full stack ourselves, so we understand every piece
- We have 7 months, which is enough if we use them right

---

## 2. THE WINNING ARCHITECTURE

### Why We're Following MonoRace (And Improving On It)

MonoRace won the A2RL 2025 championship - beating all AI teams AND three world champion human FPV pilots. They did it with exactly the same constraints we face: single monocular camera + IMU, onboard compute only, no external tracking. Their drone hit 28.23 m/s - the fastest fully onboard autonomous flight ever recorded.

Their architecture:

```
Camera (30 fps) + IMU (500 Hz)
        |                |
Gate Segmentation    IMU Integration
  (U-Net, GateSeg)       |
        |                |
Corner Extraction        |
  (QuAdGate)             |
        |                |
  PnP Pose Est. <--------+
        |
  State Estimation (gate-relative position)
        |
  Neural Controller (MLP, PPO-trained, 500Hz)
        |
  Motor Commands
```

**Why this works:** The drone doesn't need to know where it is in the world. It only needs to know where it is *relative to the next gate*. Camera sees gate, extract corners, geometry gives you 3D pose, controller flies you through it. Repeat for every gate.

**Why not pure RL end-to-end?** Pure RL (camera pixels to motor commands) is sexy but fragile. It doesn't generalize to new tracks, it's a black box you can't debug, and it takes massive compute to train. The modular approach lets us debug each piece independently and swap components.

### Our Architecture (Adapted for Solo Dev)

```
Camera Frame (640x480 RGB)
    |
[PHASE 1] Gate Detection (YOLOv8-nano)       <-- DONE, training now
    |
    Bounding boxes + confidence
    |
[PHASE 2] Corner Keypoint Extraction
    |
    4 corner pixel coordinates per gate
    |
[PHASE 3] PnP Pose Estimation
    |
    3D gate position + orientation relative to drone
    |
[PHASE 4] State Estimation (gate-relative)
    |
    Fuse with IMU for smooth, high-rate estimates
    |
[PHASE 5] Control Policy
    |
    Option A: Classical (PID/MPC) - fast to build, reliable
    Option B: Neural (PPO-trained MLP) - higher ceiling
    |
Motor Commands -> Drone
```

---

## 3. THE PLAN (Phase by Phase)

### PHASE 1: Gate Detection [IN PROGRESS]

**Status:** YOLOv8-nano trained on 1600 synthetic images, 95.1% mAP50
**Timeline:** April 6-12 (1 week)

**What:** Detect racing gates in camera frames with bounding boxes.

**What's left:**
- Complete training on merged 1600-image dataset (running now)
- Beef up domain randomization (varied backgrounds, aggressive color/lighting/blur)
- Generate 5000+ images with extreme variation
- Test on diverse unseen conditions

**Why YOLOv8-nano:** Fastest YOLO variant (84ms on CPU, under 10ms on Jetson Orin NX GPU). The Orin NX can run this at 100+ FPS, leaving compute budget for everything else. We can always scale up to YOLOv8-small later if accuracy matters more.

**Justification:** We built our own synthetic data pipeline instead of using external datasets because we control the data exactly, we can generate unlimited training images, domain randomization is built in from day one, and when the DCL sim drops, we plug in the new gate model and regenerate instantly.

---

### PHASE 2: Corner Keypoint Detection

**Timeline:** April 12-26 (2 weeks)

**What:** Upgrade from bounding boxes to precise gate corner locations (4 corners per gate).

**Approach - Two Options:**

**Option A: Segmentation + Geometric Corner Extraction (MonoRace approach)**
- Train a U-Net to produce binary gate masks
- Extract corners from mask contours using line fitting
- Pros: Proven to work at competition level
- Cons: Two-stage pipeline, more complex

**Option B: Direct Keypoint Regression (Our approach - simpler)**
- Modify YOLOv8-pose to predict 4 keypoints per detection
- Single model does detection + corners in one pass
- We already have the bounding box labels; extend to include corner coordinates
- Our synthetic data generator knows exact 3D corner positions - free labels!
- Pros: Single model, faster inference, simpler pipeline
- Cons: Less proven at competition scale

**Decision: Start with Option B.** It's simpler for a solo developer, our data pipeline already has the corner coordinates in 3D (we project them for labels), and YOLOv8-pose is a drop-in upgrade from what we already have. If it doesn't hit accuracy targets, fall back to Option A.

**Why this matters:** Bounding boxes tell you "a gate is roughly here." Corners tell you "the gate opening is exactly here." You can't do PnP pose estimation without corners. This is the step that turns detection into navigation.

---

### PHASE 3: PnP Pose Estimation

**Timeline:** April 26 - May 10 (2 weeks)

**What:** Given 4 gate corner pixel coordinates + known gate dimensions, compute 3D position and orientation of the gate relative to the drone camera.

**How:** OpenCV solvePnP with the known gate geometry (0.4m x 0.4m opening). Input: 4 pixel corners + gate physical dimensions. Output: rotation vector + translation vector (6DOF pose).

**Key details:**
- Need camera intrinsics (focal length, principal point) - will come from DCL sim specs
- RANSAC-based PnP for robustness against noisy corners
- Handle partial visibility: new ADR-VINS paper (2026) shows you can work with as few as 2 visible corners by integrating directly into a Kalman filter
- Estimate pose for the 2 nearest gates (current target + next) for trajectory planning

**Why PnP over learned depth:** PnP is deterministic - given accurate corners and known gate size, the math gives you exact 3D position. No training required. No generalization issues. Works on any gate of known dimensions, on any track, first time.

**Justification:** This is geometry, not learning. It works the same whether it's gate 1 on track A or gate 47 on track Z. This is how we solve the "courses we've never seen" problem - the geometry is universal.

---

### PHASE 4: State Estimation + IMU Fusion

**Timeline:** May 10-24 (2 weeks, overlapping with DCL sim release)

**What:** Combine visual gate observations with IMU data to maintain a smooth, high-rate estimate of the drone's state (position, velocity, orientation) at 500Hz.

**Approach:**
- Error-State Kalman Filter (ESKF)
- IMU provides high-rate prediction (500Hz): integrate accel + gyro for position/velocity/attitude
- Gate observations provide low-rate corrections (30-60Hz): PnP pose resets drift
- Gate-relative coordinate frame: always estimate position relative to the next gate, not in a global frame

**Why gate-relative, not world-relative:** No need for a map of the full track. Works on any track layout. MonoRace proved this is sufficient to beat human champions. Simpler state space means faster convergence.

**Why ESKF over full VIO:** VIO (Visual-Inertial Odometry) like VINS-Mono is powerful but complex to implement solo. ESKF with gate landmarks is simpler, and we have strong landmarks (gates are big, well-defined). If we need more, we upgrade later.

**Critical timing:** The DCL simulator drops in May. This phase is designed to coincide with that release. Once we have the sim, we calibrate everything against their specific camera model and physics.

---

### PHASE 5: Control Policy

**Timeline:** May 24 - June 2026 (ongoing through qualification)

**What:** Given the drone's state relative to upcoming gates, compute motor commands that fly the fastest safe trajectory through each gate.

**Approach (staged):**

**Stage A: PID Waypoint Controller (Week 1)**
- Simple but working immediately
- Generate waypoints at gate centers
- PID on position error, velocity, and yaw
- This is our "always have a working entry" baseline

**Stage B: Trajectory Optimization (Weeks 2-3)**
- Minimum-time trajectory through gate sequence
- Use gate poses from Phase 3 to plan curves, not just straight-line waypoints
- Account for drone dynamics (max thrust, max angular rate)
- Pre-compute optimal trajectory, then track it

**Stage C: Neural Controller via RL (Weeks 3+)**
- Train an MLP policy with PPO in the DCL sim
- Input: gate-relative state (position, velocity, angles to next 2 gates)
- Output: motor commands or thrust + attitude commands
- Train directly in the competition sim for zero sim-to-real gap
- This is the high-ceiling approach that MonoRace used to hit 28 m/s

**Justification for staged approach:** We always have something that works. Stage A enters qualification with a slow but reliable run. Stage B improves lap times. Stage C is the competitive edge. Solo developer can't afford to go all-in on RL and have nothing if it doesn't converge.

---

### PHASE 6: DCL Sim Integration + Qualification

**Timeline:** May-July 2026

**What:** When the DCL simulator drops, everything above gets retrained and integrated against the actual competition environment.

**Day 1 with DCL sim:**
1. Capture frames, retrain gate detector on DCL gate model
2. Extract camera intrinsics from sim, update PnP
3. Characterize drone dynamics, tune controller
4. Run full pipeline end-to-end

**Qualification strategy:**
- Submit a working entry ASAP (even if slow) to establish a baseline
- Iterate rapidly: run, analyze failures, fix, resubmit
- Focus on reliability first, speed second
- One crash = DNF. Finishing beats crashing.

---

## 4. WHAT WE'RE NOT DOING (And Why)

**End-to-end RL (camera to motors):** Doesn't generalize to new tracks. Black box debugging. Massive training compute. MonoRace explicitly rejected this.

**SLAM / Full mapping:** Overkill. We don't need a map of the environment. We need to know where the next gate is and fly through it.

**Stereo vision / Depth estimation:** We only have one camera. Monocular depth is noisy. PnP with known gate geometry is exact.

**Custom hardware modifications:** Not allowed. Software only.

**Building our own sim from scratch:** Waste of time. We use LSY now for practice, DCL sim when it drops for the real thing.

---

## 5. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DCL sim drops late | Medium | High | Keep training on LSY sim, architecture is sim-agnostic |
| Corner keypoints too inaccurate | Medium | High | Fall back to segmentation + geometric extraction |
| RL controller doesn't converge | Medium | Medium | Keep PID/trajectory controller as reliable fallback |
| Solo dev can't cover all phases | High | Medium | Prioritize reliability over performance ceiling |
| Domain gap (sim vs real) | High | High | Aggressive domain randomization + calibration on physical drone in Sept |
| New track layouts fail | Medium | High | Gate-relative architecture is inherently track-agnostic |
| Jetson Orin NX compute limits | Low | Medium | YOLOv8-nano + simple MLP = lightweight pipeline |

---

## 6. WEEKLY MILESTONES

| Week | Dates | Deliverable |
|------|-------|-------------|
| 1 | Apr 6-12 | Gate detector v2 trained (1600+ images), domain randomization hardened |
| 2 | Apr 13-19 | YOLOv8-pose corner keypoint model trained on synthetic data |
| 3 | Apr 20-26 | Corner detection validated, PnP pose estimation working in sim |
| 4 | Apr 27-May 3 | State estimation (ESKF) with IMU fusion, end-to-end demo in LSY sim |
| 5 | May 4-10 | PID controller flying through gates in sim autonomously |
| 6 | May 11-17 | **DCL SIM EXPECTED** - retrain everything on competition env |
| 7 | May 18-24 | Full pipeline running in DCL sim, first qualification submission |
| 8 | May 25-31 | Trajectory optimization, speed improvements |
| 9-12 | June | RL controller training, lap time optimization, iteration |
| 13-16 | July | Qualification deadline prep, robustness testing, edge cases |
| 17-20 | Aug | Prep for physical qualifier, domain randomization for real-world |
| 21-24 | Sept | **Physical qualifier in SoCal** - 2-week intensive |
| 28-32 | Nov | **Championship in Columbus, Ohio** |

---

## 7. COMPUTE STRATEGY

| Task | Hardware | Why |
|------|----------|-----|
| Development + testing | Mac (M3 Pro) | Fast iteration, MuJoCo sim, code writing |
| GPU training (YOLO, RL) | PC (NVIDIA GPU) | 10-50x faster training than CPU |
| DCL sim | TBD (check requirements) | May need PC with GPU for sim rendering |
| Competition inference | Jetson Orin NX (on drone) | ~100 TOPS, runs YOLOv8-nano at 100+ FPS |

**Priority: Set up PC with GPU for training this week.** CPU training works but is 10x slower. Every hour of training time matters when we're iterating against a deadline.

---

## 8. SUCCESS CRITERIA

**Minimum viable:** Qualify for physical round (top N virtual teams)
**Target:** Finish in top 10 at championship
**Stretch:** Podium finish

**What "qualify" likely requires:**
- Complete the virtual course without crashing
- Reasonable lap time (doesn't need to be fastest, just competitive)
- Reliable across multiple runs (consistency matters)

**Our edge:** Most teams will over-engineer. A clean, reliable pipeline that finishes every run beats a fast one that crashes half the time. Reliability first, speed second.

---

*Last updated: April 6, 2026*
*Document everything. Own every piece. Ship fast.*
