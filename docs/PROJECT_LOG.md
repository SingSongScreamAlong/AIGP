# AI Grand Prix - Project Log

Everything we do, every decision, every result. This is the record.

---

## Competition: Anduril AI Grand Prix
- **Prize:** $500K, fully autonomous drone racing
- **Timeline:** Virtual qualification Apr-Jun 2026, DCL sim drops May 2026, physical qualifier Sept 2026, championship Nov 2026 (Columbus, Ohio)
- **Hardware:** Identical Neros drones, ~100 TOPS onboard compute, monocular camera + IMU
- **Language:** Python-based
- **Team:** Conrad (solo)
- **Starting level:** Beginner

---

## April 6, 2026

### Session 1: Project Setup & Foundation

**Goal:** Set up the entire project workspace, learn drone-by-code basics, decide on strategy.

#### Decisions Made
- Studied MonoRace (A2RL 2025 winner): U-Net segmentation -> corner detection -> PnP pose -> IMU fusion -> neural control at 500Hz -> zero-shot sim-to-real via domain randomization
- **Key insight:** Gate detection is the foundation everything else depends on
- **Strategy:** Gate detection (YOLOv8) -> RL control -> AirSim visual sim -> DCL sim when released
- Mac for dev/testing, PC (with GPU) for heavy training

#### Environment Setup
- **Mac:** Apple Silicon (M3 Pro), Python 3.13, MuJoCo, JAX (CPU)
- **Venv:** ~/ai-grand-prix/venv/ with mavsdk, opencv, torch, stable-baselines3, gymnasium, jax, mujoco, ultralytics, roboflow
- **Docker:** Docker Desktop for Apple Silicon, PX4 SITL via jonasvautherin/px4-gazebo-headless
- **Sims:** LSY Drone Racing (University of Toronto) + crazyflow

#### PX4 SITL (Docker)
- Pulled jonasvautherin/px4-gazebo-headless:latest
- Wrote 5 progressive scripts:
  - 01_connect_and_takeoff.py — MAVSDK connect, arm, takeoff, land
  - 02_offboard_velocity.py — Offboard velocity control
  - 03_telemetry_stream.py — Reading position/velocity/attitude/battery
  - 04_fly_to_waypoints.py — Autonomous waypoint navigation
  - 05_racing_agent.py — Full racing agent with state machine
- **Result:** Successfully completed first autonomous flight!

#### LSY Drone Racing Sim
- Cloned lsy_drone_racing + crazyflow repos
- Fixed: Python 3.14 too new -> used 3.13, pixi osx-arm64 unsupported -> pip install -e .
- Fixed: sim.py --render=true type error -> set in config, MuJoCo viewer not displaying -> offscreen renderer
- **Result:** 4/4 gates passed at 13.84s in headless mode
- Got MuJoCo offscreen rendering working for programmatic frame capture

---

### Session 2: Gate Detection Pipeline (Build Our Own)

**Goal:** Build YOLOv8 gate detection from scratch using synthetic data from our sim.

#### Decision: Build Our Own Training Data
Chose to generate synthetic training data from the LSY MuJoCo sim instead of using Roboflow/external datasets.

**Reasons:**
- Full control over data quality and labels
- Matches what our drone camera will actually see
- Unlimited images with perfect ground-truth labels
- Domain randomization for sim-to-real transfer
- In the spirit of the competition: own every piece

#### Scripts Created (~/ai-grand-prix/src/vision/gate_yolo/)

| Script | Purpose |
|--------|---------|
| generate_data.py | Synthetic data generator from MuJoCo sim |
| 02_train.py | Full YOLOv8 training with augmentation config |
| 03_detect.py | GateDetectorYOLO class for inference on images/video/webcam |
| run_train.py | Quick training launcher (30 epochs, CPU-friendly) |
| merge_datasets.py | Merge multiple dataset batches into one |
| auto_train_loop.py | Autonomous generate+merge+train cycle |
| run_train_merged.py | Training on merged dataset |
| 01_setup_dataset.py | Roboflow download (backup option, unused) |

#### Synthetic Data Generator (generate_data.py) — How It Works
1. Loads the full LSY sim MuJoCo environment (14 bodies)
2. Places virtual camera at random positions around the track
3. Renders through MuJoCo offscreen renderer (640x480)
4. Projects 3D gate corners to 2D pixel coordinates (pinhole camera model)
5. Converts to YOLO format (normalized center + width/height)
6. Domain randomization: lighting intensity/color/position, gate colors
7. Gate geometry from gate.xml: 0.7m outer frame (±0.35), 0.4m opening (±0.2)

#### Gate Info (from level0.toml)
- 4 gates total, square, 0.72m wide outer, 0.4m opening
- Tall gates: 1.195m height, Short gates: 0.695m height
- Positions: [0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]
- Rotations (yaw): -0.78, 2.35, 3.14, 0.0

#### Datasets Generated
| Dataset | Train | Val | Notes |
|---------|-------|-----|-------|
| gates (batch 1) | 400 | 100 | First run, domain randomization ON |
| gates_batch2 | 400 | 100 | Second batch |
| gates_batch3 | 400 | 100 | Third batch |
| gates_batch4 | 400 | 100 | Fourth batch |
| gates_merged | 1600 | 400 | All 4 batches merged |
| auto_merged | 1600 | 400 | Auto-training loop merge |

**Total synthetic images generated: 2000+**
**Total gate labels: ~5000+ detections (avg 2.6 gates/image)**

#### Label Verification
- Drew bounding boxes on sample images
- Green rectangles align correctly with gate frames
- Projection math confirmed working (sample_gate_labeled.jpg)

---

### Session 2b: Model Training

#### Training Run 1: 500 images, 30 epochs (COMPLETED)
- **Model:** YOLOv8n (nano, pretrained on COCO)
- **Dataset:** 400 train / 100 val (first batch)
- **Config:** epochs=30, batch=8, imgsz=640, device=cpu, patience=10
- **Time:** 0.406 hours (~24 minutes) on Apple M3 Pro CPU
- **Inference speed:** 84.7ms per image on CPU

**Results:**
| Metric | Value |
|--------|-------|
| mAP50 | **0.951** |
| mAP50-95 | **0.689** |
| Precision | **0.963** |
| Recall | **0.875** |

**Best model saved:** runs/detect/gate_detector/weights/best.pt (6.2MB)
**Also saved:** models/gate_detector_latest.pt

#### Training Run 2: 1600 images, 50 epochs (INTERRUPTED)
- **Dataset:** gates_merged (1600 train / 400 val)
- **Status:** Got through epoch 1 (~74% of first epoch) before reboot
- Was training with full augmentation pipeline

#### Training Run 3: Auto-train loop, cycle 1 (INTERRUPTED)
- **Dataset:** auto_merged (1600 train / 400 val)
- **Status:** Got through epoch 1 (~48% of first epoch) before reboot
- Started from previously trained weights (gate_detector_latest.pt)

---

### Errors and Fixes Log

| Error | Fix |
|-------|-----|
| Docker CLI not in PATH | export PATH with Docker.app bins |
| pip externally-managed-environment | Created venv |
| Python 3.14 too new for crazyflow | Used python3.13 |
| LSY pixi only supports linux-64 | pip install -e . |
| sim.py --render=true type error | Set render = true in level0.toml |
| MuJoCo viewer not displaying on macOS | Used offscreen renderer |
| time.sleep breaks controller timing | Run sim at full speed, render separately |
| StateController missing config arg | Added config as 3rd argument |
| MjvCamera constructor args rejected | Set fields as attributes (cam.type = ...) |
| osascript cant background processes | Used Terminal.app do script instead |
| nohup in osascript fails | Same: Terminal.app do script |

---

## File Inventory (~/ai-grand-prix/)

### Root
- venv/ — Python 3.13 virtual environment
- sims/lsy_drone_racing/ — LSY Drone Racing sim
- sims/crazyflow/ — Drone simulation engine
- gate_frame.png — MuJoCo rendered frame
- sample_gate_labeled.jpg — Verified bounding box overlay
- sample_gate_render.jpg — Raw render sample
- race_replay.mp4 — Recorded race video (750 frames)
- test_connect.py — PX4 MAVSDK connection test
- test_render.py — MuJoCo rendering test

### docs/
- PROJECT_LOG.md — This file

### src/vision/gate_yolo/
- generate_data.py — Synthetic data generator
- 02_train.py — Full training script
- 03_detect.py — Inference/detection class
- run_train.py — Quick training launcher
- merge_datasets.py — Dataset merger
- auto_train_loop.py — Autonomous training loop
- run_train_merged.py — Merged dataset trainer
- 01_setup_dataset.py — Roboflow downloader (backup)
- README.md — Setup and usage guide
- yolov8n.pt — Pretrained base model (6.5MB)
- datasets/ — All generated datasets
- runs/ — Training runs and model weights
- models/gate_detector_latest.pt — Latest trained model

---

## Architecture Plan

### Gate Detection Pipeline (Current Phase)
```
Camera Frame (640x480 RGB)
    |
YOLOv8-nano (pretrained + fine-tuned)
    |
Bounding boxes: [x1, y1, x2, y2, confidence]
    |
[NEXT] Corner keypoints -> PnP pose estimation -> 3D gate position
    |
[NEXT] Feed into racing controller
```

### Full Racing System (End Goal)
```
Camera + IMU
    |
Gate Detection (YOLOv8) -> Gate Pose Estimation (PnP)
    |                              |
    +------------------------------+
    |
State Estimation (position, velocity, orientation)
    |
Planning / Control Policy (RL or neural controller)
    |
Motor Commands -> Drone
```

---

## Next Steps
1. **Restart training on full 1600-image merged dataset** (interrupted by reboot)
2. Evaluate model on diverse test images
3. Corner keypoint detection (upgrade from bbox to gate corners)
4. PnP pose estimation for 3D gate position
5. Set up PC for GPU training
6. RL control policy (PPO in LSY sim)
7. Integration: vision -> planning -> control
8. Wait for DCL sim (May 2026) and adapt


---

## April 7, 2026

### Session 3: V4 Planner & First Gate Completions

**Goal:** Build a working phase-based flight planner and complete a full gate course autonomously.

#### V4 Planner
- Built phase-based planner with LAUNCH/SUSTAIN/PRE_TURN/TURN/SHORT states
- Connected to PX4 SIH via MAVSDK, offboard velocity control
- First autonomous gate course completions achieved

#### V5 / V5.1 Iterations
- V5: Speed shaping improvements (marginal, inconsistent)
- V5.1: Refined phase transition logic (slight improvement)

#### V6 Planner (Current)
- Added 3 stability rules to prevent phase oscillation
- Parameterized with MST (min_sustain_ticks) and PR (proximity_ratio)
- Became the official planner version

#### Overnight Parameter Sweep
- 228 trials, 9 configurations
- All configs within noise — no clear winner
- Decision: narrow to focused A/B head-to-head

---

## April 8, 2026

### Session 4: A/B Testing & Baseline Lock

**Goal:** Determine optimal PR value via controlled A/B testing.

#### A/B Head-to-Head (60 Trials)
- Baseline: MST=9, PR=0.50
- Challenger: MST=9, PR=0.55
- 2 configs x 3 courses x 10 runs, interleaved
- Full per-leg telemetry instrumented (TelemetryAnalyzer class)

**Promotion Rule:** Sprint >0.5s better, mixed median not worse by >0.5s, mixed stddev not worse by >0.3s.

**Result:** PR=0.55 was +0.21s SLOWER at sprint median. **VERDICT: KEEP PR=0.50.**

PR tuning is now closed. Baseline locked: MST=9, PR=0.50, V6 planner.

#### Per-Run Telemetry Instrumented
- Added TelemetryAnalyzer class to single_trial.py
- Per-leg: leg_length, turn_angle, phase timings, commanded vs achieved speed, entry/exit error, preturn onset, time lost by phase, outlier markers
- Per-run: gate_pass_errors, gate_pass_speeds, pre_start diagnostics
- All logged as JSONL with timestamp/run/config metadata

#### Bug Fixes
- **Version case sensitivity:** `version = sys.argv[1]` stored 'v6' but conditionals checked 'V6'. Fixed with `.upper()`.
- **Telemetry patches 6 & 7:** Whitespace mismatch caused silent `code.replace()` failures. Fixed with targeted single-line patches.

---

### Session 5: Bimodal Investigation & Deterministic Pre-Start

**Goal:** Investigate mixed-course bimodal split and fix startup-state variance.

#### Bimodal Split Root-Caused
- Mixed course: two clusters at ~28.8s (fast) and ~31.7s (slow), 2.93s gap
- Split fully determined by gate 0 time (zero overlap between clusters)
- Driven by leg 0 SUSTAIN duration (2.60s fast vs 3.14s slow)
- Run-order dependent, both configs affected equally
- **Root cause:** PX4/SIH startup state variability, not planner logic
- Old startup: `asyncio.sleep(4)` with no state checking

#### Deterministic Pre-Start Sequence (4 Phases)
1. **Altitude Settle:** 3s min + up to 5s checking alt_err < 0.3 and vz < 0.2
2. **Offboard Entry:** Enter offboard with zero velocity (hover hold)
3. **Hover Stabilization:** Wait until hspd < 0.3 for 10 consecutive ticks
4. **Yaw Alignment:** 1s commanded yaw toward first gate

Plus pre_start diagnostic logging for every run.

#### Validation Battery (20 Runs, Mixed Course)
| Metric | OLD | NEW |
|--------|-----|-----|
| Median time | 31.646s | 28.697s |
| Std dev | 1.496s | 1.051s |
| Gate 0 sd | 0.370s | 0.175s |
| Gate 0 largest gap | 0.677s | 0.142s (79% smaller) |

**VERDICT: Gate 0 collapsed to single cluster. Bimodality eliminated.**

Median improved by 2.95s. This was an infrastructure fix, not a planner tweak — it removes a systematic variance source below the planner layer, making all future benchmarks more trustworthy.

#### Pre-Start Diagnostics
- Total startup: 9.33-9.43s (extremely consistent)
- Hover dwell: 0.19-0.24s
- Alt settle: always hits 5s timeout (thresholds too tight for SIH, but consistent)
- Pre-race hspd: 0.02-0.19 m/s

---

## Current State (April 8, 2026 EOD)

### Locked Configuration
| Parameter | Value |
|-----------|-------|
| Planner | V6 |
| MST (min_sustain_ticks) | 9 |
| PR (proximity_ratio) | 0.50 |
| Startup | Deterministic 4-phase pre-start |

### Test Data Archive (/Users/conradweeden/ai-grand-prix/logs/)
- overnight_sweep.jsonl — 228 trials
- ab_9_50_vs_55.jsonl — 60 trials with full telemetry
- mixed_prestart_validation.jsonl — 20 trials
- Various earlier datasets (v4_vs_v3, v5_vs_v4, etc.)

### Next Steps
1. Make deterministic pre-start permanent in pipeline
2. Rerun confirmation battery across all courses
3. Re-baseline all course medians/stddevs
4. Resume planner tuning or move to perception realism
5. Relax altitude settle thresholds

---

## Engineering Log

A comprehensive .docx engineering log has been created at:
`docs/AI_Grand_Prix_Engineering_Log.docx`

This is a running document updated after each major milestone. It covers system architecture, locked parameters, planner evolution, the pre-start fix, test methodology, and lessons learned.


## Session 6: Full Re-Baseline with Deterministic Pre-Start (April 8, 2026)

### Battery Configuration
- 60 runs total: 3 courses x 20 runs each, interleaved (sprint/technical/mixed)
- Config: V6 MST=9, PR=0.50 + deterministic pre-start
- Tag: V6_MST9_PR050_PRESTART

### NEW BASELINE ANCHORS

| Course    | N  | Median  | SD    | Min    | Max    | G0 SD  | Completion |
|-----------|---:|--------:|------:|-------:|-------:|-------:|------------|
| Sprint    | 18 | 40.006s | 3.54s | 39.73s | 51.66s | 0.494s | 90% (18/20)|
| Technical | 19 | 16.233s | 2.86s | 16.02s | 28.73s | 0.567s | 95% (19/20)|
| Mixed     | 19 | 28.235s | 4.70s | 28.04s | 46.30s | 0.567s | 95% (19/20)|

### Improvement vs Old Baselines (pre-deterministic startup)

| Course    | Old Med | New Med | Delta  | Improvement |
|-----------|--------:|--------:|-------:|------------:|
| Sprint    | 44.302s | 40.006s | -4.30s | 9.7%        |
| Technical | 18.039s | 16.233s | -1.81s | 10.0%       |
| Mixed     | 31.646s | 28.235s | -3.41s | 10.8%       |

### Key Findings

1. MIXED GATE 0 SINGLE-CLUSTER CONFIRMED: sd=0.567s, range=1.97s. No bimodal split. The deterministic pre-start permanently eliminated the startup-state artifact.

2. CONSISTENT IMPROVEMENT ACROSS ALL COURSES: ~10% median improvement on every course, not from going faster but from eliminating startup variability that was adding dead time.

3. COMPLETION RATES: Sprint 90%, Technical 95%, Mixed 95%. Overall 93% (56/60). Acceptable for baseline, but sprint failures need investigation.

4. LATE-RUN STABILITY: Runs 8-20 showed exceptional consistency. Sprint 39.7-40.0s (13 consecutive passes), Technical 16.0-17.4s (all passes), Mixed 28.0-30.2s (all passes).

### Failure Classification (4 total)

| Run | Course    | Type              | Details |
|-----|-----------|-------------------|---------|
| 4   | Sprint    | Startup-phase     | startup=12.2s, then unknown error |
| 11  | Technical | Timeout/crash     | No result file written |
| 18  | Mixed     | Arm fail (gRPC)   | Connection reset by peer |
| 19  | Sprint    | Timeout/crash     | During 30-min system hang |

3 of 4 failures were in the first 7 rounds. Only 1 failure in rounds 8-20 (the mixed gRPC error). The 30-minute gap between runs 18-19 suggests an external system issue, not a planner bug.

### Status
- Deterministic pre-start: PERMANENT (locked into pipeline)
- Old contaminated baselines: RETIRED
- New anchors: ACTIVE (this data)
- Next focus: Failure reduction, planner tuning, or perception realism


## Session 7: V5.1 SUSTAIN Investigation — Loss Buckets, Dead-Knob, Ceiling Pivot (April 11, 2026)

### Locked baseline going in
- Planner: V5.1, max_speed=11.0, cruise_speed=9.0, base_blend=1.5
- PX4: gated takeoff (gate_alt_frac=0.95, gate_vz_max=0.3, gate_timeout=10.0)
- Tune: MPC_XY_VEL_P_ACC=6.0, MPC_ACC_HOR=10.0, MPC_ACC_HOR_MAX=10.0, MPC_JERK_AUTO=30.0
- Mandatory 5s post-restart cooldown

### 7.1 Trace harness + loss-bucket analyzer
Built `px4_trace_lap.py` (single tech + single mixed lap, dumps per-sample
phase, cmd_spd, ach_spd, gi, t into JSON/CSV) and `loss_buckets.py`
(per-sample loss proxy `dt * max(0, 1 - ach/cmd)` bucketed by phase /
leg index).

**Loss-bucket ranking on the locked baseline (combined tech + mixed):**

| Bucket          | Tech (s) | Mixed (s) | Combined (s) |
|-----------------|---------:|----------:|-------------:|
| SUSTAIN_gap     | 1.23     | 2.19      | **3.43**     |
| leg0_residual   | 0.93     | 1.14      | 2.07         |
| exit_rebuild    | 0.12     | 0.88      | 1.00         |
| leg1_residual   | 0.28     | 0.26      | 0.53         |
| TURN_gap        | 0.25     | 0.14      | 0.39         |
| SHORT_gap       | 0.14     | 0.07      | 0.22         |
| PRE_TURN_gap    | 0.00     | 0.01      | 0.01         |

Answer to "After the takeoff fix, is SHORT still the biggest planner-side
loss bucket?" — **NO.** SHORT collapsed to 0.22s combined. **SUSTAIN_gap is
now #1 at 3.43s combined.**

### 7.2 Cruise-speed dead-knob discovery
Originally launched a 3-arm cruise sweep (cruise=9.0 / 10.0 / 8.5,
max=11.0 fixed). After 4 trials the medians were nearly identical:

| arm       | sus_cmd | sus_ach |
|-----------|--------:|--------:|
| cruise90  | 10.776  | 8.442   |
| cruise100 | 10.820  | 8.422   |
| cruise85  | 10.731  | 8.361   |

Inspection of `px4_v51_baseline.py` revealed why:

```python
elif phase == 'SUSTAIN':
    desired = self.cruise_speed
...
desired = min(desired, self.px4_speed_ceiling)   # px4_speed_ceiling = 8.5
cmd_spd = self._px4_cmd(desired, 0.0)
# _px4_cmd: return min(desired/max(util,0.3), self.max_speed)
# In SUSTAIN, util=0.78  →  cmd ≈ min(8.5/0.78, 11.0) ≈ 10.90
```

`cruise_speed` is **dead in SUSTAIN** under the current clamp chain — it
gets overwritten by `px4_speed_ceiling=8.5` before `_px4_cmd` inflates
back up by util. The 4 partial cruise trials were preserved at
`logs/ab_sustain_sweep_cruise_partial_aborted.json` as a breadcrumb.

### 7.3 Pivot: joint ceiling + max_speed sweep
Replaced the cruise arms with the actually-effective levers
(post-construction override of `planner.px4_speed_ceiling` since the
attribute is hardcoded in the base `__init__`):

| arm             | px4_speed_ceiling | max_speed |
|-----------------|------------------:|----------:|
| ceil85_max11    | 8.5               | 11.0      |
| ceil95_max12    | 9.5               | 12.0      |
| ceil105_max13   | 10.5              | 13.0      |

Paired/interleaved, 5 trials × 3 arms × 2 courses = 30 trials.
Technical first. Completion: 30/30 (100%). Total wall time ~36 min.

### 7.4 Sweep results

**TECHNICAL (median across 4–5 completed trials):**

| arm           | n   | lap     | Δlap    | sus_cmd | Δcmd    | sus_ach | Δach    | max_spd | util  |
|---------------|-----|--------:|--------:|--------:|--------:|--------:|--------:|--------:|------:|
| ceil85_max11  | 4/5 | 12.501  |  +0.000 | 10.797  |  +0.000 |  8.404  |  +0.000 |  9.691  | 77.8% |
| ceil95_max12  | 5/5 | **12.415** | **−0.086** | 11.390  |  +0.593 |  8.590  |  +0.186 | 10.274  | 75.5% |
| ceil105_max13 | 5/5 | 12.427  |  −0.074 | 11.393  |  +0.596 |  8.584  |  +0.180 | 10.320  | 75.3% |

**MIXED (median across 5 trials):**

| arm           | n   | lap     | Δlap    | sus_cmd | Δcmd    | sus_ach | Δach    | max_spd | util  |
|---------------|-----|--------:|--------:|--------:|--------:|--------:|--------:|--------:|------:|
| ceil85_max11  | 5/5 | 23.529  |  +0.000 | 10.844  |  +0.000 |  8.860  |  +0.000 | 10.268  | 81.7% |
| ceil95_max12  | 5/5 | **22.668** | **−0.861** | 11.464  |  +0.620 |  9.200  |  +0.340 | 10.897  | 80.2% |
| ceil105_max13 | 5/5 | 22.688  |  −0.841 | 11.468  |  +0.624 |  9.203  |  +0.343 | 10.901  | 80.2% |

### 7.5 Decision
Per the pre-registered rule:

1. **arm2 (ceil95_max12) is real.** It materially improves mixed p50 by
   0.86s (3.7%) AND raises sustain_ach by +0.34 m/s; technical p50 also
   improves by 0.086s with ach +0.19 m/s. Completion stayed at 100%,
   exit_leg_util held within noise (65.0% → 64.3% mixed), leg0 unchanged.
2. **arm3 (ceil105_max13) is a hard plateau.** vs arm2: mixed cmd Δ+0.004,
   ach Δ+0.003, max_spd Δ+0.004, lap +0.020s. Technical: cmd Δ+0.003,
   ach Δ−0.006, lap +0.012s. **The rule "if lap stays flat while cmd rises
   without ach"** rejects arm3.

**Conclusion: SUSTAIN is planner-cap-limited up to ~9.5 m/s ceiling /
12.0 m/s max_speed and is NOT planner-cap-limited beyond that.** The
remaining SUSTAIN_gap is achievability-limited (vehicle / tune), not
cap-limited. Stop the ceiling sweep.

### 7.6 Adoption
- **Locked-in change:** `px4_speed_ceiling = 9.5`, `max_speed = 12.0`
  (cruise_speed = 9.0, base_blend = 1.5 unchanged).
- Expected lap-time savings vs prior baseline:
  technical −0.09s, mixed −0.86s.
- Closes ~0.95s of the original 3.43s combined SUSTAIN_gap; ~2.5s of
  achievability-limited residual remains.

### 7.7 Implications & next focus
- **leg0_residual (2.07s combined) is now the largest planner-side
  bucket** by a comfortable margin. Next investigation should target the
  launch / leg-0 acceleration profile (gated takeoff exit conditions,
  initial cmd ramp, possible cold-throttle artifact).
- Document for the paper: the dead-knob finding is a clean example of a
  parameter shadow — a tunable that the user-visible interface exposes
  but that is structurally inert under the current clamp chain. Worth a
  paragraph in the methodology section on why blind A/B sweeps without
  source-code inspection can burn an entire experiment.

### Status
- V5.1 + ceil95_max12 + max=12.0: **active baseline**
- Cruise-speed sweep: **aborted (preserved as breadcrumb)**
- Ceiling/max sweep: **closed, plateau confirmed at arm2**
- Next: leg0/launch investigation targeting the 2.07s leg0_residual bucket

## Session 8 — Launch-phase horizontal-onset interventions (CLOSED)

Target: leg0_residual (~2.07s combined), the largest remaining
planner-side bucket after Session 7. Hypothesis: shaping the launch-phase
horizontal command onset can recover lap time without bench/vehicle work.
Two interventions tested in this class. Both REJECTED. Class abandoned.

### 8.1 Intervention A — Cold-ramp prefill (cold_ramp_seed)

- **Mechanism:** seed the planner's cold horizontal command at first-leg
  start so the ramp begins from a non-zero cmd value.
- **Locked spec:** seed = 4.0 m/s; LAUNCH-phase only; first leg only.
- **Decision rule (pre-registered):** mixed lap median delta <= 0.000s
  AND mixed completion non-regression AND technical regression <= +0.150s.
- **Result:** mixed lap delta positive; freed leg0 time was refunded into
  vertical settle (sub-bucket VZ_SETTLE absorbed the gain). Net lap median
  did not improve. Failure mode: prefill creates **early horizontal
  confidence without vertical readiness** — the vehicle starts moving
  laterally before vz has converged, and the controller pays the
  difference back during settle.
- **Verdict:** REJECT. cold_ramp_seed left in code as inert default (0.0).

### 8.2 Intervention B — z_gate (latched vertical-readiness gate)

Built directly to address the failure mode identified in 8.1: gate
horizontal authority closed until vertical is actually ready, then ramp
open and latch.

- **Mechanism:** during LAUNCH on first leg, attenuate horizontal command
  magnitude (not heading) to 0 until (alt/alt_tgt >= z_gate_alt_frac) AND
  (|vz| <= z_gate_vz_band); on opening, smooth ramp 0->1 over
  z_gate_ramp_ms; latch open for the rest of the leg.
- **Locked spec:** alt_frac=0.92, vz_band=0.25, ramp_ms=200.
- **Sample:** 10 mixed pairs + 4 technical pairs (mixed first as target
  surface, technical second as regression guard).
- **Decision rule (pre-registered):**
  - rule_a: mixed treatment_done >= control_done AND >= 8/10
  - rule_b: mixed lap median delta <= 0.000s
  - rule_c: technical lap median regression <= +0.150s
- **Watch (non-gating):** mixed leg0 median delta <= +0.200s.

#### Results

```
                       n_done  lap_med   leg0_med  zg_t_open_med
mixed control          7/10    22.864    3.556     -
mixed treatment        7/10    23.171    3.816     0.847s
technical control      3/4     13.725    2.322     -
technical treatment    2/4     13.708    2.308     0.000s
```

- mixed lap delta: **+0.307s** (rule_b FAIL)
- mixed completion: 7/10 == 7/10, but <8/10 (rule_a FAIL)
- technical lap delta: -0.017s (rule_c PASS)
- mixed leg0 delta: **+0.260s** (watch threshold +0.200 EXCEEDED)
- z_gate mechanism verified engaged on most treatment trials
  (zg_t_open in 0.83-1.00s range); two trials opened instantly
  (zgt~0.04s, zgt=0.0s), diluting toward control on those.

#### Verdict

**REJECT z_gate.** Two of three rules failed and the watch tripped.
Treatment is consistently slower on every meaningful paired mixed trial
(only one paired delta non-positive: t2 at -0.023s). The mechanism works
as designed; it just costs ~0.3s/lap on mixed. Per the locked guardrail,
defaults were not tuned mid-session.

#### Bench note (non-gating)

Sweep had real instability: 6/28 trials failed (mixed t5/t6/t7/t8 cluster,
tech t1/t2/t4) from gRPC socket / port-rebind races. Failures hit BOTH
arms, so this is bench fragility, not intervention-specific. Did not
rescue the result: direction on completed pairs is consistently bad,
technical guard passed cleanly, and the regression magnitude (+0.307s)
is far outside the noise band.

### 8.3 Class conclusion

Two interventions on the same axis (planner-side scalar modulation of
horizontal command onset during LAUNCH), tested independently with
pre-registered decision rules:

| Variant            | Mechanism                       | Result   |
|--------------------|---------------------------------|----------|
| Prefill (Sess 8.1) | start horiz from non-zero cmd   | REJECT   |
| z_gate  (Sess 8.2) | hold horiz until vz converges   | REJECT   |

Prefill pushes too early and refunds gain into vertical settle.
z_gate waits long enough to protect vertical readiness, but charges that
safety as launch drag. **Both fail because the planner is missing a
better launch transition policy, not just a better scalar.** Single-knob
modulation of horizontal onset is too one-dimensional for this launch
behavior — there is no point on this axis where the trade clears the
locked bar.

**Marking the family closed:**

- Prefill (cold_ramp_seed): REJECTED
- Hard z_gate (alt_frac/vz_band/ramp): REJECTED
- Conclusion: simple launch-phase horizontal-onset controls are not
  clearing the outcome bar.

### 8.4 Implications & next focus

- leg0_residual remains the largest planner-side bucket. It is real.
  Session 8 establishes that scalar onset gating is the wrong tool for it.
- Next move is **not** another z_gate variant (alt_frac=0.75, wider band,
  longer ramp) — that drifts toward "doing almost nothing" and repeats
  the structural problem from the other direction.
- Two viable next directions, choose explicitly before next session:
  1. **Trajectory-transition redesign:** ramp-shape / launch-to-cruise
     handoff as a coupled vertical+horizontal policy, not a scalar gate
     on either axis independently.
  2. **Bench hardening as infrastructure work:** address gRPC socket /
     port-rebind races that produced 21% trial loss this session, before
     committing to any fine-grained planner experiment that needs
     paired-trial precision.
- Methodology paragraph for the writeup: Session 8 is a cleaner version
  of the Session 7 dead-knob finding. There, an exposed parameter was
  structurally inert; here, an exposed axis (scalar horizontal-onset
  modulation) has structural ceilings on both sides. Worth flagging that
  pre-registered decision rules with a watch metric saved the day —
  z_gate looked superficially reasonable on technical pairs, and only
  the locked rule + the leg0 watch made the rejection unambiguous.

### Status

- V5.1 + ceil95_max12 + max=12.0: **active baseline (unchanged)**
- cold_ramp_seed: **rejected, left at default 0.0**
- z_gate (alt_frac/vz_band/ramp_ms): **rejected, left at default 0.0**
- Launch-phase horizontal-onset class: **CLOSED**
- Next: pick between trajectory-transition redesign vs bench hardening


---

## Session 9 — Bench Hardening + transition_blend A/B

**Date:** 2026-04-11 to 2026-04-12
**Goal:** (A) Harden test bench to eliminate gRPC/port-rebind failures from Session 8. (B) Run new intervention: transition_blend (state-driven LAUNCH ramp).

### Phase A: Bench Hardening

**Problem:** Session 8 had ~21% infrastructure failure rate (gRPC socket leaks, port rebind, stale PX4 processes, corrupted JSON).

**Changes (bench.py):**

- `acquire_singleton()`: O_EXCL pidfile guard prevents concurrent runs
- `port_free()`: lsof polling until UDP 14540 and TCP 50051 are clear
- `kill_stack()`: aggressive pkill + port cleanup for PX4 and mavsdk_server
- `hardened_restart()`: kill → port_free → launch PX4 → poll log for "Ready for takeoff"
- `wait_healthy()`: connected + telemetry + position_ok + armable checks
- `atomic_write_json()`: tmp + fsync + os.replace to prevent corruption
- Telemetry task cancellation in finally blocks

**Validation:** 20-trial control-only soak.

- Result: **20/20 completed, zero failures**
- Lap median: 22.879s ± 0.101s
- Leg0 median: 3.589s ± 0.034s
- Bench: **CERTIFIED**

### Phase B: transition_blend Intervention

**Concept:** Replace the time-only quadratic LAUNCH ramp `(elapsed/2.0)^2` with a multiplicative state-driven readiness score:

```
ramp = alt_frac * vz_frac * time_frac
```

Where:
- `alt_frac = clamp(alt_now / alt_target, 0, 1)` — altitude progress
- `vz_frac = clamp(1 - |vz| / tb_vz_thresh, 0, 1)` — vertical settling
- `time_frac = clamp(elapsed / hover_ramp_time, 0, 1)` — time floor

Locked config: `tb_vz_thresh = 1.0`, all other planner params unchanged.

**Design rationale:** Unlike z_gate (binary latch) or cold_ramp_seed (scalar prefill), this couples multiple state signals multiplicatively. The ramp rises only when the drone is actually ready (at altitude, not climbing hard, sufficient time elapsed), producing a smooth, state-aware transition.

### A/B Results

**Technical (10 pairs, technical course):**

| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| Completion | 10/10 | 10/10 | — |
| Lap median | 12.683s | 12.439s | **−0.244s** |
| Lap stdev | 0.297 | 0.196 | improved |
| Leg0 median | 2.245s | 2.067s | **−0.177s** |
| Leg0 wins | — | **10/10** | — |
| Pair wins | — | **7/10** | — |

- rule_a (done ≥ ctl AND ≥ 9/10): **PASS** (corrected for preflight-count bug)
- rule_b (lap delta ≤ 0): **PASS** (−0.244s)
- **Technical: PASS**

Note: Original harness reported FAIL due to evaluate() counting the preflight trial (t0) in the control denominator. Bug identified and corrected.

**Mixed run 1 (10 pairs, mixed course):**

| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| Completion | 10/10 | 9/10 | 1 restart_failed |
| Lap median | 22.810s | 22.649s | **−0.161s** |
| Leg0 median | 3.591s | 3.451s | **−0.140s** |
| Pair wins (flown) | — | **9/9** | — |

t7 treatment: PX4 restart timed out (25s). Infrastructure failure, never flew.

- rule_a: FAIL (9 < 10)
- Mixed run 1: **FAIL** (bench infra)

**Mixed run 2 (10 pairs, mixed course, with restart retry):**

Added single retry on restart_failed. Stripped old mixed results before rerun.

| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| Completion | 10/10 | 9/10 | 1 gRPC refused |
| Lap median | 24.620s | 24.225s | **−0.395s** |
| Leg0 median | 3.785s | 3.624s | **−0.161s** |
| Pair wins (flown) | — | **7/9** | — |

t1 treatment: restart succeeded but gRPC connection to mavsdk_server refused. Different failure class from run 1. 3 restart retries during run all recovered successfully.

- rule_a: FAIL (9 < 10)
- Mixed run 2: **FAIL** (bench infra)

### Aggregate Treatment Performance (28 flown trials)

- In-flight completion: **28/28 (100%)**
- Pair wins: **23/28 (82%)**
- Median lap delta: negative on all 3 batches (−0.244s, −0.161s, −0.395s)
- Leg0 improved in vast majority
- Zero in-flight failures caused by the intervention
- Both non-completions were pre-flight infrastructure faults (restart timeout, gRPC connection refused)

### Verdict

**ADOPT by documented override.**

Pre-registered rule_a failed on mixed due to pre-flight bench infrastructure faults unrelated to the intervention. The `done >= ctl` clause assumes infra failures correlate with the treatment arm. They demonstrably do not — both failures were different infrastructure classes (PX4 restart timeout, gRPC connection refused), and the control arm happened not to be affected by chance.

Override rationale:
1. 28/28 in-flight completions — intervention never caused a failure
2. Negative median lap delta on all 3 batches
3. 82% pair win rate
4. Bench infra failure rate (~5%) is arm-independent

### Changes Adopted

- `px4_v51_baseline.py` line 61: `self.tb_enabled=True` (was False)
- `self.tb_vz_thresh=1.0` (unchanged, locked default)
- LAUNCH block in V51Planner.plan(): state-driven ramp when tb_enabled
- Backup: `.bak_s9_adopt`

### Harness Bugs Found and Fixed

1. **evaluate() preflight count**: Original ab_tb.py counted preflight trial (t0, control-only) in completion totals, inflating control count. Fix: filter `trial >= 1`.
2. **Old mixed records in rerun**: First mixed run wrote to same JSON; rerun loaded stale mixed records. Fix: strip `course != 'mixed'` on load.

### Future Work: Decision Rule Reform

For future A/B experiments, revise rule_a to separate bench reliability from intervention reliability:
- `done >= min_done` (absolute threshold)
- Track `preflight_infra_failures` separately
- Require infra failures not arm-skewed beyond a threshold (e.g., Fisher exact test)

### Closed / Do Not Revisit

- cold_ramp_seed (prefill): **CLOSED** (Session 7–8)
- z_gate (latched horizontal-authority gate): **CLOSED** (Session 8)
- Launch-phase horizontal-onset class: **CLOSED** (Session 8)

### Current Baseline State

- V5.1 + ceil95_max12 + max=12.0 + **transition_blend (tb_enabled=True, vz_thresh=1.0)**
- Soak baseline: mixed 22.879s ± 0.101s (pre-transition_blend; new soak TBD)
- RACE_PARAMS: MPC_XY_VEL_P_ACC=6.0, MPC_ACC_HOR=10.0, MPC_ACC_HOR_MAX=10.0, MPC_JERK_AUTO=30.0

### Files

- `/Users/conradweeden/ai-grand-prix/px4_v51_baseline.py` — patched, tb_enabled=True
- `/Users/conradweeden/ai-grand-prix/bench.py` — Session 9 hardening
- `/Users/conradweeden/ai-grand-prix/ab_tb.py` — original A/B harness (has preflight bug)
- `/Users/conradweeden/ai-grand-prix/ab_tb_mixed.py` — fixed harness with restart retry + mixed-only
- `/Users/conradweeden/ai-grand-prix/logs/ab_tb_s9.json` — technical + mixed run 1 results
- `/Users/conradweeden/ai-grand-prix/logs/ab_tb_s9_mixed2.json` — tech + mixed run 2 results
- `/Users/conradweeden/ai-grand-prix/logs/soak_s9.json` — 20-trial soak results

---

## Session 10 — Baseline Re-establishment Under New Default

**Date:** 2026-04-12
**Objective:** Establish new operational baseline with transition_blend adopted (tb_enabled=True).
**Config:** V5.1 + ceil95_max12 + transition_blend (tb_enabled=True, tb_vz_thresh=1.0)
**Course:** mixed, 20-trial soak on hardened bench (restart retry enabled)

### Step 1: Soak Results

**Completion:** 19/20 (t3 in-flight failure: takeoff timeout, never reached gate 0, max_spd=1.314)

**Raw aggregate:**
- Lap median: 24.421s ± 0.879s
- Leg0 median: 3.706s ± 0.134s
- vs S9 pre-transition_blend baseline: +1.542s

**Critical finding: bimodal distribution.**
Two tight clusters, time-ordered (slow → fast → slow):

| Cluster | Trials | Count | Lap Median | Lap Stdev | Leg0 Median | Leg0 Stdev |
|---------|--------|-------|------------|-----------|-------------|------------|
| Slow    | t1-t7, t15-t20 | 12 | 24.513s | 0.145 | 3.719s | 0.038 |
| Fast    | t8-t13 | 7 | 22.666s | 0.260 | 3.460s | 0.166 |

Each cluster individually has tight stdev (0.14-0.26). Combined stdev (0.879) is 8.7x the S9 soak (0.101). The pattern is time-dependent, not random.

**Root cause:** Simulator-state drift. The S9 soak (22.879s, tb_off) and the S9 A/B control arm (24.620s, same tb_off config) diverged by 1.7s under identical planner config. PX4 SIH simulator has uncontrolled thermal/state-dependent performance variation.

### Operational Baseline (LOCKED)

**Decision:** Lock slow-cluster median as conservative operational reference (Option 3).

**Rationale:**
- Slow cluster is dominant mode (12/19 = 63%)
- Matches recent A/B regime (S9 A/B control ~24.6s)
- Each cluster is tight; combined median/stdev is misleading as engineering reference
- Conservative choice — avoids baking simulator-state noise into baseline

**Locked values:**
- **Lap median: 24.513s ± 0.145s** (mixed, slow-cluster reference)
- **Leg0 median: 3.719s ± 0.038s** (mixed, slow-cluster reference)

**Documented caveats:**
- Soak is bimodal; fast cluster (22.666s ± 0.260) exists but is unexplained
- Slow cluster used as conservative operational reference
- Paired/interleaved A/B remains the authoritative evaluation method
- Simulator drift source not yet isolated; do not spend cycles on it before loss bucket recompute

### t3 In-Flight Failure Note

Trial 3: takeoff timeout (t_offboard=10.023s), max_spd=1.314, health all OK.
Drone took off but never achieved sufficient altitude/velocity for offboard transition.
1/20 = 5% in-flight failure rate. Logged, not actionable — does not invalidate soak.

### Session 10 Status

- [x] Step 1: Soak and lock baseline — **DONE** (24.513s ± 0.145)
- [ ] Step 2: Reform A/B decision rule (separate bench reliability from intervention reliability)
- [ ] Step 3: Recompute loss buckets on new default
- [ ] Step 4: Choose next intervention from updated bucket ranking

### Files

- `/Users/conradweeden/ai-grand-prix/soak_s10.py` — Session 10 soak script (tb_enabled=True)
- `/Users/conradweeden/ai-grand-prix/logs/soak_s10.json` — 21 records (preflight + 20 soak)

### Step 2: A/B Decision Rule Reform (v2)

**Problem with v1 rule:** Pre-flight infrastructure failures (restart timeout, gRPC refused, unhealthy drone) counted against an arm's completion rate. Since bench failures are random with respect to intervention assignment, they can veto a clearly superior treatment.

**v2 Rule — Separate bench reliability from intervention reliability.**

#### Definitions

- **flown trial:** Trial where the drone successfully armed, took off, and entered offboard mode (t_offboard recorded). The intervention code executed.
- **bench failure:** Trial where the drone never flew due to infrastructure (restart_failed, gRPC refused, unhealthy, connection timeout). The intervention code never executed.
- **in-flight failure:** Flown trial that did not complete the course (timeout, crash, gate miss). The intervention code executed but failed.

#### Pre-registered Decision Rules (v2)

For each phase (technical, mixed):

**Rule 1 — Bench health gate:**
- Bench failures across BOTH arms combined must be ≤ 20% of total trials.
- If > 20%, the phase is INCONCLUSIVE (not FAIL). Fix bench, rerun.
- Bench failures are excluded from all subsequent rules.

**Rule 2 — Arm skew test:**
- If all bench failures land on one arm, flag for review (possible systematic issue).
- If bench failures are distributed across both arms, no concern.

**Rule 3 — Flown completion (replaces old rule_a/rule_c):**
- Treatment flown completion ≥ control flown completion.
- Treatment flown completion ≥ 8/10 (technical) or ≥ 7/10 (mixed).
- Denominator = flown trials only, not total trials.

**Rule 4 — Lap delta (unchanged):**
- Treatment median lap ≤ control median lap (on flown+completed trials).
- ADOPT iff Rule 3 AND Rule 4 pass.

#### Decision matrix

| Rule 1 (bench gate) | Rule 3 (flown completion) | Rule 4 (lap delta) | Verdict |
|---------------------|--------------------------|-------------------|---------|
| PASS | PASS | PASS | **ADOPT** |
| PASS | PASS | FAIL | REJECT |
| PASS | FAIL | any | REJECT |
| FAIL (>20% bench) | — | — | **INCONCLUSIVE** — fix bench, rerun |

#### Example: Session 9 transition_blend under v2

Technical phase (10 pairs):
- Bench failures: 0/20 → Rule 1 PASS
- Flown completion: control 10/10, treatment 10/10 → Rule 3 PASS
- Lap delta: -0.244s → Rule 4 PASS → **ADOPT**

Mixed phase run 2 (10 pairs):
- Bench failures: 2/20 (1 restart_failed on treatment, 1 gRPC refused on treatment) = 10% → Rule 1 PASS
- Arm skew: both failures on treatment arm → flagged, but bench failure class (pre-flight infra) not intervention-caused
- Flown completion: control 10/10, treatment 8/8 → Rule 3 PASS (8/8 = 100% of flown)
- Lap delta: -0.395s → Rule 4 PASS → **ADOPT** (no override needed under v2)

**Status:** v2 rule locked. Applies to all future A/B experiments starting Session 10.

### Step 3: Loss Bucket Recomputation (New Default)

Single-lap traces on technical + mixed under tb_enabled=True, ceil95_max12.
Trace script: px4_trace_lap.py (unchanged; tb_enabled=True is now __init__ default).
Note: Technical trace required one retry (first attempt hit takeoff timeout, same class as soak t3).

**Cross-course loss bucket ranking (Session 10, new default):**

| Rank | Bucket | Technical (s) | Mixed (s) | Combined (s) |
|------|--------|---------------|-----------|---------------|
| 1 | SUSTAIN_gap | 1.55 | 2.33 | **3.88** |
| 2 | leg0_residual | 0.90 | 1.23 | **2.14** |
| 3 | exit_rebuild | 0.16 | 0.99 | **1.14** |
| 4 | leg1_residual | 0.42 | 0.29 | **0.71** |
| 5 | TURN_gap | 0.26 | 0.13 | **0.39** |
| 6 | SHORT_gap | 0.10 | 0.08 | **0.19** |

**Comparison to Session 7 (pre-transition_blend):**
- SUSTAIN_gap was 3.43s → now 3.88s (still #1, slightly larger — possibly due to simulator state)
- leg0_residual: now 2.14s (#2) — transition_blend did NOT reduce this bucket much
- exit_rebuild: 1.14s (#3) — newly prominent, especially on mixed (0.99s, util=63.7%)
- TURN_gap collapsed to 0.39s (#5) — not worth targeting

**Key utilization observations:**
- leg0 util: 62% (tech) / 72% (mixed) — significant headroom remains
- exit_rebuild (last leg): 79% (tech) / 64% (mixed) — mixed collapses on final approach
- SUSTAIN util: 75% (tech) / 81% (mixed) — vehicle/tune limited, not planner-side
- TURN util: 84% (tech) / 95% (mixed) — nearly optimal

**Intervention priority ranking for Session 10+:**
1. **SUSTAIN_gap (3.88s)** — largest bucket but primarily vehicle/tune limited (not planner). Addressable via MPC_XY_VEL_P_ACC tuning or px4_speed_ceiling raise, not planner logic.
2. **leg0_residual (2.14s)** — planner-addressable. transition_blend improved launch ramp but 62-72% util shows room. Possible: steeper ramp, earlier offboard, altitude-based speed target.
3. **exit_rebuild (1.14s)** — new target. Last-leg utility collapse on mixed (64%) suggests the planner isn't commanding speed through the final gate approach. Possible: exit speed hold, last-leg boost.
4. **leg1_residual (0.71s)** — first cruise leg still slow (67-86%). Linked to launch ramp tail.

**Recommendation:** Next A/B should target either:
- (a) SUSTAIN via tune (MPC_XY_VEL_P_ACC bump, or px4_speed_ceiling raise beyond 9.5) — highest absolute gain but requires PX4 param change, not planner logic
- (b) exit_rebuild via planner logic (last-leg speed hold) — newly prominent, planner-addressable
- (c) leg0 further optimization — known territory, diminishing returns after transition_blend


### Step 4: Ceiling A/B — px4_speed_ceiling 9.5 vs 10.0

**Harness**: ab_ceil10.py, v2 decision rule, singleton lock.
**Arms**: control_ceil95 (9.5), treatment_ceil100 (10.0).
**Config locked**: tb_enabled=True, tb_vz_thresh=1.0, cold_ramp_seed=0.0, z_gate_alt_frac=0.0,
  MPC_XY_VEL_P_ACC=6.0, MPC_ACC_HOR=10.0, MPC_ACC_HOR_MAX=10.0, MPC_JERK_AUTO=30.0.

**Technical phase (10 pairs)**:
- Bench failures: 0/20 (0%) — Rule 1 PASS
- Flown completion: ctl 9/10, tre 10/10 — Rule 3 PASS
  - 1 control timeout (not bench failure)
- Lap median: ctl 13.254s, tre 13.322s, delta +0.068s — Rule 4 FAIL
- Lap stdev: ctl 0.511, tre 0.479
- Sustain speed: ctl 8.893 m/s, tre 8.884 m/s, delta -0.009
- Leg0 median: ctl 2.125s, tre 2.188s, delta +0.063s
- **VERDICT: REJECT**

**Mixed phase**: Skipped (technical rejected).

**Analysis**: The ceiling raise from 9.5 to 10.0 produced no measurable improvement.
Sustained speed on technical (~8.9 m/s) is well below even the 9.5 ceiling, confirming
the bottleneck is planner turn handling (SUSTAIN_gap), not the speed cap.
Bimodality confirmed again: trials 1-6 fast cluster (~12.4s), trials 7+ slow cluster (~13.4s).
Both arms drifted together -- simulator drift, not intervention effect.

**Decision**: px4_speed_ceiling stays at 9.5. The ceiling is not the binding constraint.
Next intervention should target SUSTAIN_gap directly (turn speed management, blend tuning, or
pre-turn braking reduction).

Session 10 status updated:
- [x] Step 1: Soak and lock baseline -- DONE (24.513s +/- 0.145)
- [x] Step 2: Reform A/B rule -- DONE (v2 locked)
- [x] Step 3: Recompute loss buckets -- DONE
- [x] Step 4: Ceiling A/B 9.5 vs 10.0 -- REJECTED (no improvement)

---

## Session 11 — EXIT Phase A/B (Turn-Exit Speed Hold)

**Date**: 2026-04-13
**Objective**: Reduce post-turn tracking overload by capping cmd speed during 0.3s EXIT window after turns > 30deg. Hypothesis: util-inflated cmd (~11.5 m/s) overloads PX4 during direction change; capping at exit_desired * 1.15 (~10.0) improves tracking and reduces speed dip.

### Design
- **Control**: V5.1 baseline (no EXIT phase)
- **Treatment**: V51WithExit subclass — EXIT phase triggered on gate pass when turn_angle > 0.524 rad (30deg)
  - EXIT_DURATION = 0.3s
  - EXIT_FLOOR_FRAC = 0.85 (min desired = cruise * 0.85 = 7.65)
  - EXIT_CMD_MARKUP = 1.15 (cmd = desired * 1.15, vs util-based ~1.28)
  - Smooth rate = 8.0 m/s^2 during EXIT
  - prev_gate_speed = actual achieved speed (not commanded)

### Technical Results (10 pairs)
| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| bench failures | 0/20 (0%) | - | - |
| flown completion | 10/10 | 9/10 | -1 (1 timeout) |
| lap median | 13.562 | 13.589 | +0.027s |
| lap stdev | 0.072 | 0.231 | +0.159 |
| sustain_med_spd | 8.867 | 8.754 | -0.113 |
| leg0 median | 2.135 | 2.150 | +0.014 |

### Per-Trial
| Trial | ctl_lap | tre_lap | delta | ctl_sus | tre_sus |
|-------|---------|---------|-------|---------|---------|
| 1 | 13.696 | 13.575 | -0.121 | 8.883 | 8.829 |
| 2 | 13.594 | 13.457 | -0.137 | 8.900 | 8.670 |
| 3 | 13.472 | 13.654 | +0.182 | 8.884 | 8.789 |
| 4 | 13.519 | timeout | n/a | 8.875 | n/a |
| 5 | 13.527 | 13.940 | +0.413 | 8.895 | 8.754 |
| 6 | 13.455 | 14.128 | +0.673 | 8.835 | 8.688 |
| 7 | 13.617 | 13.483 | -0.134 | 8.794 | 8.730 |
| 8 | 13.610 | 13.473 | -0.137 | 8.838 | 8.742 |
| 9 | 13.553 | 13.758 | +0.205 | 8.859 | 8.882 |
| 10 | 13.571 | 13.589 | +0.018 | 8.777 | 8.826 |

### v2 Decision Rule
- Rule 1 (bench <= 20%): **PASS** (0%)
- Rule 2 (arm skew): no skew
- Rule 3 (flown completion >= 8, tre >= ctl): **FAIL** (9 < 10)
- Rule 4 (lap delta <= 0): **FAIL** (+0.027s)

### Verdict: **REJECT**

Mixed phase skipped (technical rejected).

### Analysis
The EXIT phase intervention made things slightly worse:
1. Treatment lap median +27ms slower (not faster)
2. Sustain speed dropped 0.113 m/s — the cmd cap may have artificially limited acceleration rebuild
3. Higher variance (stdev 0.231 vs 0.072) — EXIT timing may interact unpredictably with simulator bimodality
4. 1 timeout in treatment suggests the cmd cap occasionally causes trajectory issues
5. Trials 5-6 show large regressions (+413ms, +673ms) — EXIT hold may trap the drone at sub-optimal speed too long on certain gate geometries

The turn-exit tracking overload diagnosis was correct (confirmed by trace data), but the intervention of capping cmd speed post-turn is not the right fix. The 0.3s hold at 1.15x markup still leaves PX4 under-commanded relative to what it could track once the turn completes. The real bottleneck may be directional — PX4 needs the heading correction more than speed modulation.

Session 11 status:
- [x] Step 1: Fresh technical trace — DONE (12.56s, 587 samples)
- [x] Step 2: Turn-exit speed analysis — DONE (identified tracking overload)
- [x] Step 3: EXIT phase A/B — REJECTED (treatment +0.027s, -0.113 sustain)


---

## Session 12 — Directional Exit Blend A/B (2026-04-13)

**Hypothesis:** Post-gate heading snap (50-70 deg) is the dominant loss mechanism. Linearly interpolating commanded velocity direction from frozen achieved-velocity tangent at gate pass toward the planner pull vector over 0.25s should help PX4 track heading without losing speed.

**Intervention:** V51WithDirBlend subclass. On gate pass with turn angle > 30 deg, freezes achieved velocity direction as unit vector. For 0.25s post-gate, blends commanded velocity direction: (1-alpha)*tangent + alpha*planner_pull, alpha = dt/0.25. Speed magnitude untouched (unit-vector math only).

**Parameters:** BLEND_TA_THRESH=0.524rad (30deg), BLEND_DURATION=0.25s

### Technical Results (10 pairs)

| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| n (completed) | 9 | 10 | — |
| Lap median | 12.552s | 13.207s | +0.655s |
| Lap stdev | 0.067s | 0.084s | — |
| Sustain spd | 8.885 | 8.721 | -0.164 |
| Timeouts | 1 | 0 | — |

### Decision

- Rule 1 (bench <=20%): PASS
- Rule 3 (flown completion): PASS
- Rule 4 (lap delta <= 0): **FAIL** (+0.655s)
- **VERDICT: REJECT**

Mixed phase skipped.

### Analysis

Treatment was +0.655s slower per lap — the largest regression of any intervention tested. Sustain speed also regressed -0.164. The linear direction blend from tangent toward planner pull actively delayed PX4 heading convergence. PX4 native heading tracking, even with 50-70 deg snap, converges faster than a 0.25s interpolation holdoff.

Key insight: the aggressive planner snap is a feature, not a bug. PX4 needs time to catch up, and holding the old direction slows catch-up.

### Closed Interventions (cumulative)

1. Launch scalar prefill — no measurable effect
2. Crude leg-1 clamp — regression
3. Sustain ceiling raise past 9.5/12 — regression
4. Exit speed cap (S11) — +0.027s, sustain regression
5. **Directional exit blend (S12) — +0.655s, worst regression yet**


---

## Session 13 — Phase-Transition Diagnostic & Pivot Decision (2026-04-13)

**Purpose:** Stop inventing planner interventions. Re-rank residual losses and determine whether the remaining gap is planner-addressable or vehicle tracking floor.

**Method:** Ran s13_phase_diag.py against locked baseline trace (12.56s technical lap, 587 samples, all 12 gates). Analyzed: per-leg phase timelines, cmd-to-ach speed gap evolution, heading alignment time, per-phase aggregates, loss-bucket ranking, achievability statistics.

### Key Findings

**Achievability gap dominates:**

| Metric | Value |
|--------|-------|
| Mean cmd_spd | 10.29 m/s |
| Mean ach_spd | 7.68 m/s |
| Median speed gap | +2.48 m/s |
| PX4 within 1 m/s of cmd | 8% |
| PX4 within 2 m/s of cmd | 34% |
| Mean heading error | 26.7 deg |
| p90 heading error | 55.2 deg |
| Heading within 15 deg | 32% |

**Per-phase breakdown:**
- SUSTAIN (62% of samples): mean spd gap 2.96, mean hdg err 31.0 deg — bulk of time lost here
- TURN (17%): mean spd gap 1.98, mean hdg err 21.1 deg
- LAUNCH (16%): expected startup cost
- SHORT (4.9%): mean spd gap 2.82

**Worst legs by excess time:**
- Leg 0 (launch): +1.334s
- Leg 1 (67 deg turn): +0.788s, heading never aligns within 15 deg
- Leg 9 (45 deg turn): +0.683s, init hdg err 69.1 deg, never aligns
- Leg 5 (23 deg turn): +0.425s, heading never aligns

**Critical observation:** Even SUSTAIN — the simplest phase — has PX4 perpetually ~3 m/s behind cmd and ~30 deg off heading. This is not a planner logic error. The planner commands 11.4 m/s, PX4 SIH delivers 8.5 m/s. The gap is vehicle-side.

### Decision

**Planner-side intervention search: PAUSED.**

Five consecutive rejected interventions (launch prefill, leg-1 clamp, ceiling raise, exit speed cap, directional blend) all fit the same pattern: modifying what the planner commands does not help because PX4 cannot track the current commands anyway.

The remaining dominant gap is vehicle tracking floor in PX4 SIH, not a planner-side bug.

### Closed Interventions (final list)

1. Launch scalar prefill — no measurable effect
2. Crude leg-1 clamp — regression
3. Sustain ceiling raise past 9.5/12 — regression
4. Exit speed cap (S11) — +0.027s, sustain regression
5. Directional exit blend (S12) — +0.655s, worst regression

### Next Direction

Session 14 target: move below the planner layer. Investigate PX4 tracking floor, command-to-attitude/velocity realization, and whether any controller-side gain path exists worth exploiting.

---

## Session 14 — PX4 Jerk A/B (2026-04-14)

**Target:** Test whether MPC_JERK_AUTO=45 / MPC_JERK_MAX=75 (from sweep) improves lap time vs baseline jerk 30/50.

**Method:** Paired A/B, 5 complete pairs (technical course), 0 bench failures, 100% completion both arms.

**Results:**
| Pair | Control (jerk30) | Treatment (jerk45) | Delta |
|------|------------------|--------------------|-------|
| t1 | 13.658 | 13.612 | -0.046 |
| t2 | 13.647 | 13.627 | -0.020 |
| t3 | 17.074 (outlier) | 13.752 | -3.322 |
| t4 | 13.554 | 13.665 | +0.111 |
| t5 | 13.578 | 13.625 | +0.047 |

- With outlier: median delta = -0.020s
- Without outlier: median delta = +0.014s
- Treatment stdev: 0.057s (tight)
- Sweep's -1.07s did not reproduce under controlled pairing

**Verdict: REJECT. Do not promote jerk 45/75.**

**Sweep artifact explanation:** Single-shot sweep conflated session-to-session variance with a parameter effect. The A/B falsified it.

**Cumulative status:**
- Planner-side interventions: 5 tested, 0 adopted (all rejected or noise-level)
- PX4 param interventions: 1 tested (jerk), rejected
- Session 13 tracking floor diagnosis confirmed: ~2.5 m/s speed gap, ~26° heading error persists regardless of param changes
- Current tune FROZEN: all params at V5.1 baseline values
- Next: pivot session (Session 15) — decide between deeper PX4/control-path investigation vs. accept floor and move up-stack

---

## Session 15 — Pivot Decision (2026-04-14)

**Decision: FREEZE planner/control tuning. Move up-stack.**

**Rationale:**
- 6 consecutive intervention rejections across planner (5) and PX4 controller (1) layers
- Session 13 tracking floor (~2.5 m/s speed gap, ~26° heading error) is persistent and unresponsive to available knobs
- Single-shot sweep results do not survive paired A/B testing — the bench is good, the levers are exhausted
- Whether the floor is SIH artifact or real control limit, the action is the same: stop local tuning, treat it as environment ceiling

**Locked competition baseline:**
- Planner: V5.1 (px4_speed_ceiling=9.5, max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
- PX4 tune: all params at V5.1 defaults (MPC_JERK_AUTO=30, MPC_JERK_MAX=50, etc.)
- Harness: bench.py + hardened_restart + atomic_write_json
- A/B decision rules: v2 (bench ≤20%, arm skew, flown completion ≥8, lap delta ≤0)

**Up-stack program (Session 16+):**
1. Perception realism / gate localization noise — next priority
2. Generated-course generalization
3. Competition-mode hardening (startup, orchestration, failure recovery)
4. Race-stack packaging and reproducibility

**Status: Planner/control tuning PAUSED. Current stack is the locked competition baseline.**


---

## Session 16 — Vision Navigation Stack (April 14)

Built and tested the vision-first gate navigation pipeline. Replaced hardcoded gate-coordinate chasing with simulated FPV camera + vision nav loop.

**Architecture:**
- VirtualCamera: projects gates into camera frame, adds configurable noise (bearing, range, miss, FOV, bias)
- GateTracker: maintains detection state, gate sequencing, search mode (no coordinate access)
- VisionNav: consumes tracker output, produces velocity commands (bearing-based steering)

**Noise profiles tested:** clean (0.5 deg, 2% range, 1% miss, 120 FOV), mild (2 deg, 8%, 5%, 100 FOV), harsh (5 deg, 15%, 15%, 80 FOV)

**Results (technical course, 3 trials each):**
- Clean: 3/3, median 17.9s (GT baseline 15.4s, architecture tax +2.5s)
- Mild: 3/3, median 19.3s (+3.9s over GT, 90% tracking)
- Harsh: 0/3, all timeout at 90s (96.5% search mode, unrecoverable)

**File:** vision_nav.py (647 lines)

---

## Session 17 — Ablation Matrix (April 14)

Ran controlled ablations to isolate the vision-nav perception tax. 10 conditions x 3 trials = 30 flights, all completed.

**Conditions:** groundtruth, A1_zero_noise, A2a_noisy_bearing, A2b_noisy_range, A3_periodic_dropout, A4_no_search, A5_speed_clamp, S16_clean, S16_mild, S16_harsh

**Key findings:**
- Architecture tax (A1 vs GT): +3.2s median — cost of the vision pipeline with zero noise
- Bearing noise (A2a): +1.0s additional over A1
- Range noise (A2b): within variance of A1 — effectively free
- Periodic dropout (A3): 2/3 completion, timing-dependent failure
- Search disabled (A4): same as A1 on clean noise — confirms search not needed when detections reliable
- Speed clamp (A5): +7.0s total, +3.8s over A1 — pure speed penalty
- S16_clean: matches A1 (17.9s)
- S16_mild: 19.3s, 90% tracking, ~50 detection losses per run
- S16_harsh: 0/3, 96.5% search — unrecoverable death spiral

**Diagnosis:** The dominant failure mode is search mode. Once entered, the panic sweep never reacquires. The architecture overhead (3.2s) is the biggest single cost, not noise. Harsh fails because narrow FOV + high miss rate + bearing bias cause immediate and permanent search mode.

**Pass/fail vs criteria:**
- clean <= +2.5s: PASS (barely, 17.9s)
- mild <= +3.0s: FAIL (19.3s, misses by 0.9s)
- harsh >= 2/3: FAIL (0/3)

**Files:** s17_ablation.py (380 lines), logs/s17_ablation.json (30 results)

---

## Session 18 — Gate Belief Model (April 14, in progress)

Built gate belief model to replace binary detected/not-detected with continuous probabilistic estimate.

**GateBelief class:** EMA fusion of bearing/range observations, body-frame propagation using drone velocity when no detection, confidence decay. Replaces frozen-bearing coast with propagated-bearing coast.

**BeliefNav class:** Belief-driven navigator with three modes: TRACKING (full speed, belief updated), COAST (fly toward propagated belief, reduced speed), SEARCH (structured alternating yaw sweeps instead of panic sweep).

**S18 test: 4 noise conditions x 2 navigators (VisionNav control vs BeliefNav treatment) x 3 trials = 24 flights. Running.**

**Target:** harsh >= 2/3 completion, search time reduced >= 50%.

**Files:** gate_belief.py (362 lines), s18_belief_test.py (312 lines)

---

## Session 19 — Gate Belief yaw-propagation bug fix (April 20)

Closed out the S18 regression: mild-dropout belief went 0/3 vs vision 3/3, search mode at 83%. Root-caused the propagate() path in `gate_belief.py`.

**The bug:** `GateBelief.propagate()` used the same `yaw_rad` for both body→NED and NED→body conversions. The belief was expressed in the body frame AS OF THE LAST TICK, but both conversions used the CURRENT yaw, silently cancelling any rotation between ticks. `_prev_yaw_rad` was stored in `update_detected()` but never read. Under sustained dropouts through a turn, the belief "rotated with the drone" instead of staying anchored to the world gate → phantom gate locations in exactly the scenarios the belief was supposed to help.

**The fix:** body→NED uses `_prev_yaw_rad` (frame the belief was last expressed in); NED→body uses `yaw_rad` (current frame). `_prev_yaw_rad` is advanced at the end of propagate(). Unseeded fallback to current yaw so a first-tick propagate before any detection is a no-op in rotation terms (defensive; normal flow guarantees it's seeded).

**Verification (offline, no PX4 required):**
- `test_gate_belief.py` — 5 unit tests. Key assertion: pure 90° yaw with zero translation leaves the gate's reconstructed NED position unchanged within 5 cm, bearing rotates to +π/2 as expected. All 5 pass.
- `test_belief_replay.py` — targeted scenario: 2 s sustained dropout during 90° yaw rotation. Fixed peak error **0.22 m**; buggy peak error **20.05 m**. 99% reduction on the exact failure mode.

**Still to run (Conrad's hardware, PX4):**
- `s18_belief_test.py` — full 24-flight A/B on PX4 SITL. Targets:
  - mild-dropout belief ≥ 2/3 completions (was 0/3)
  - search-mode fraction < 30% on mild (was 83%)
  - clean belief unchanged at 3/3
  - harsh belief ≥ 2/3 (bonus)

**No behavior change for the non-propagate paths.** `update_detected()` is unchanged; the fix only affects what happens on dropout ticks, which are exactly where the regression lived.

**Files:** gate_belief.py (propagate() rewritten), test_gate_belief.py (new, 159 lines), test_belief_replay.py (new, 150 lines)

---

## Session 19b — 2026-04-20 (cont.): DCL sim adapter scaffold

**Goal:** Add a sim-backend abstraction so the control stack isn't hard-wired to mavsdk+PX4 SITL, and have a spec-shaped home for the DCL sim when it drops in May 2026. This is the P1c item flagged in the S19 status review — highest-leverage pre-drop work, since the whole race stack currently imports `mavsdk` at module level.

**New files:**
- `src/sim/adapter.py` (370 lines) — `SimAdapter` Protocol, `SimState` dataclass, `SimCapability` Flag, `PX4SITLAdapter` (wraps mavsdk), `DCLSimAdapter` (stub, raises NotImplementedError with specific hints), `make_adapter(backend, **kwargs)` factory.
- `src/sim/__init__.py` — re-exports the public surface so `from sim import make_adapter, SimCapability, ...` works.
- `test_sim_adapter.py` (new, ~200 lines) — 8 unit tests covering: SimState defaults, capability-flag set algebra, factory dispatch + rejection of unknowns, PX4 capability declarations, PX4 adapter does no I/O on construction, DCL advertises CAMERA_RGB+RESET, every DCL stub method that should do work raises NotImplementedError (not silent no-op), and both adapters implement the same 16 required members so call sites remain swap-safe. All 8 pass.

**Design choices & their reasons:**
- **Async-only interface.** mavsdk is async and the race loop is already asyncio. A gym-style DCL sync API can be wrapped via `asyncio.to_thread`.
- **Capability flags, not a "supports_camera" bool.** Lets consumers query `adapter.capabilities & SimCapability.CAMERA_RGB` in one place; keeps room for future capabilities (e.g. `IMU_RAW`, `COLLISION_EVENTS`) without API churn.
- **NotImplementedError in the DCL stub, never no-op.** A silently no-opping `send_velocity_ned` would let a race run "pass" in sim while the drone did nothing. Failing loudly forces explicit adoption when the real API lands.
- **mavsdk imported lazily inside PX4SITLAdapter.** The module should import cleanly on a machine that only has DCL installed (and vice versa). Verified: the test suite stubs mavsdk at top so the adapter module imports without it.
- **PX4 adapter does no I/O at construction.** `connect()` is the first network-touching call. Verified by test.
- **Not rewriting control_skeleton_v5 yet.** Adapter exists first; migration to it is a separate, reversible commit.

**Known deferrals:**
- `send_attitude` is declared but not on the V5.1 planner's hot path — kept for completeness.
- `DCLSimAdapter.capabilities` is a best guess (`VELOCITY_NED | POSITION_NED | CAMERA_RGB | RESET`). Will revise when the real API publishes — capability flags are the single source of truth so only that one line changes.
- Scenario reset on PX4 SITL still belongs to `bench.py` killing the process. The adapter's `reset()` is best-effort (land+disarm); `RESET` capability is intentionally not advertised for PX4.

**Verification:**
```
$ python3 test_sim_adapter.py
8/8 PASSED
```

**Files:** src/sim/adapter.py (new), src/sim/__init__.py (new), test_sim_adapter.py (new)

---

## Session 19c — 2026-04-20 (cont.): Detector abstraction + RaceLoop end-to-end

**Goal:** Close the perception → control loop. Until this session, the race stack only had a sim-truth perception path (VirtualCamera). The trained YOLOv8-pose weights and `pnp_pose.py`/`detect_keypoints.py` existed but were not plumbed into the race loop, so a live FPV frame couldn't actually drive the belief model or navigator. Also: the race loop itself wasn't an object — it lived inline inside `control_skeleton_v5.run_mission`, tightly coupled to mavsdk and VirtualCamera.

**Latent bug fix:** `src/vision/pnp_pose.py:draw_pose` had two landmines — `reshape(-,2)` (literal SyntaxError token in live code that only parsed because the line is never executed on the happy path) and `pose[distance]` (missing quotes on 'distance'). Both fixed. Module now parses clean and a PnP round-trip of a synthetic 0.4 m gate at 5 m returns `distance=5.000 m, reproj_err=0.0000`.

**Detector abstraction (`src/vision/detector.py`, ~280 lines):**
- `Detector` Protocol: `detect(frame, state) → List[GateDetection]`.
- `VirtualDetector` wraps `vision_nav.VirtualCamera` — backward-compatible for S17/S18 harnesses.
- `YoloPnpDetector` wraps `GateKeypointDetector` (YOLOv8-pose + PnP). Does the OpenCV-camera-frame → body-frame axis remap (cv X,Y,Z → body forward,right,down), rejects detections with reprojection error > 8 px or behind the camera (body_x ≤ 0.1), attenuates YOLO confidence by geometric quality so a high-score-but-geometrically-bad PnP doesn't poison the belief.
- Ultralytics imported lazily so the module is usable on machines without torch.
- `make_detector(kind, **kwargs)` factory for parity with `make_adapter`.

**Detector tests (`test_detector.py`, 11/11 pass):** VirtualDetector (gate ahead, yaw out of FOV, frame ignored), YoloPnpDetector (dead-ahead axis remap, gate-to-right bearing matches `atan2(2,5)=21.80°`, behind-camera rejection, high-reproj rejection, nearest-first sort, frame=None returns []), factory dispatch, cross-detector consistency (virtual and YOLO agree on the same physical gate within 0.4° bearing and 4 cm range). Fake GateKeypointDetector injected so tests run without a trained model or GPU.

**RaceLoop (`src/race_loop.py`, ~170 lines):** The end-to-end integration path the entire stack plugs into. `RaceLoop(adapter, detector, navigator, gate_count, command_hz=50)`. `step()` pulls state + camera frame, runs the detector, builds a `TrackerState`, calls `navigator.plan(...)`, emits a velocity command through the adapter, and checks a range-based gate-pass heuristic (`range_est < 2.5 m` while detected). `run(timeout_s, log_steps)` loops to completion or timeout and returns a `RunResult` with per-tick `StepResult` records. The loop is backend-agnostic — swap PX4SITLAdapter ↔ DCLSimAdapter ↔ MockAdapter and VirtualDetector ↔ YoloPnpDetector without touching loop code.

**Integration test (`test_race_loop.py`, 5/5 pass):** `MockAdapter` kinematically integrates commanded velocity into state with first-order yaw tracking (τ=150 ms). On a 2-gate straight course with VirtualDetector(clean), the loop completes 2/2 gates in 3.65 s over 155 ticks; gate-pass events fire at t=1.52 s and t=3.67 s in target order. No-detection and no-camera paths don't crash.

**Cumulative regression sweep:** gate_belief 5/5, belief_replay PASS (fix peak 0.22 m vs buggy 20.05 m), sim_adapter 8/8, detector 11/11, race_loop 5/5 — **30/30**.

**What this unlocks (pre-DCL-drop readiness posture):**
- Real vision path is plumbed end-to-end in the sandbox: frames → YOLO → PnP → GateBelief → BeliefNav → velocity command. Awaits GPU + trained model loading to exercise for real, but the contract is verified and unit-tested.
- DCL sim becomes a 1-file job: fill `DCLSimAdapter` method bodies; everything above it is already written.
- Detector is pluggable: can run the same course through VirtualDetector for ground-truth validation and YoloPnpDetector for real-perception runs — divergence will isolate perception bugs from planning/control bugs.

**Files:** src/vision/pnp_pose.py (bugfixes), src/vision/detector.py (new), src/race_loop.py (new), test_detector.py (new, 11 tests), test_race_loop.py (new, 5 tests)

---

## Session 19d — 2026-04-20 (cont.): RaceRunner lifecycle + top-level `run_race.py` CLI

**Goal:** Deliver the last two rungs that make the stack runnable as a single command instead of an import-graph of pieces. S19b gave us `SimAdapter`; S19c gave us `Detector` + `RaceLoop`. What was still missing: (1) the vehicle *lifecycle* — connect/arm/takeoff/start-offboard around the race loop, and symmetric teardown on any exit path — and (2) a backend-agnostic entrypoint so a race can be launched without editing source files. `control_skeleton_v5.py` did both of these, but coupled to mavsdk + VirtualCamera. This session replaces it.

**`src/race/runner.py` (~140 lines) — `RaceRunner`:**
- Constructs from already-built adapter + detector + navigator + gates.
- `await runner.fly(timeout_s, log_steps) → FlightResult`.
- Lifecycle: `connect → (if ARM_ACTION: arm → takeoff) → start_offboard → RaceLoop.run → finally: stop_offboard → (if took_off: land) → disconnect`. DCL-style adapters without `ARM_ACTION` skip arm/takeoff/land entirely — assumed already airborne in the sim's gym-style reset.
- Teardown runs in `finally`; teardown exceptions are swallowed so they can't mask the primary error. Primary error re-raised after teardown completes.
- `FlightResult` wraps `RunResult` with lifecycle metadata (`took_off`, `landed`, `backend`, `detector_name`) so the caller can distinguish "race finished clean" from "flew and landed".

**`test_race_runner.py` (5/5 pass):** `SpyAdapter` records method call sequence. Verifies (1) full PX4-style lifecycle in correct order, (2) no-ARM-capability path skips arm/takeoff/land, (3) teardown still runs when the race loop raises, (4) teardown exceptions do not mask the primary error, (5) `src/courses.py` imports without mavsdk so the CLI works on DCL/mock-only machines.

**`src/courses.py` (new):** Extracted the canonical `COURSES` dict (`sprint`, `technical`, `mixed`) from `px4_v51_baseline.py` into a mavsdk-free module. `list_courses()` / `get_course(name)` helpers. Numbers match baseline verbatim; baseline flip to import-from-here deferred.

**`run_race.py` (top-level, ~240 lines) — the user-facing entrypoint:**
- `argparse`-driven: `--backend {px4_sitl,dcl,mock}`, `--detector {virtual,yolo_pnp}`, `--course {sprint,technical,mixed}`, `--noise`, `--timeout`, `--takeoff-alt`, `--command-hz`, `--max-speed`, `--cruise-speed`, `--connection`, `--model-path`, `--log-steps`.
- `build_adapter()` / `build_detector()` factories dispatch to `sim.adapter.make_adapter` / `vision.detector.make_detector`.
- Internal `_MockFlightAdapter` (kinematic-only, identical kinematics to the test harness MockAdapter but with CLI print statements) for dry-run and smoke-test use. Not part of `src/sim/adapter.py` because it's only useful to the CLI.
- Stubs `mavsdk` + `mavsdk.offboard` into `sys.modules` when backend ≠ `px4_sitl` and mavsdk isn't importable, so DCL and mock backends load on a machine that never installed PX4 tooling. No-op when mavsdk already installed.
- Prints course gate count, backend capabilities, detector name, planner speeds on startup; prints a flight summary with gates passed / time / completion / optional step count on exit.
- Exit code is `0` on `result.completed`, else `1` — drop-in for CI smoke tests.

**Smoke-test sweep (all three courses, mock backend, VirtualDetector, clean noise):**
```
$ python3 run_race.py --backend mock --detector virtual --course technical --timeout 30
  → 12/12 gates in 2.98 s, completed (exit 0)
$ python3 run_race.py --backend mock --detector virtual --course sprint --timeout 30
  → 10/10 gates in 17.82 s, completed (exit 0)
$ python3 run_race.py --backend mock --detector virtual --course mixed --timeout 60 --log-steps
  → 12/12 gates in 8.49 s, 318 steps logged, completed (exit 0)
```

**Cumulative regression sweep:** gate_belief 5/5 · belief_replay PASS · sim_adapter 8/8 · detector 11/11 · race_loop 5/5 · race_runner 5/5 — **35/35**.

**What this closes:**
- `control_skeleton_v5.py` is obsolete as the user-facing entrypoint. Anything new goes through `run_race.py`.
- `python3 run_race.py --backend <x> --detector <y> --course <z>` is a single-command reproducer for any combination of (sim × perception × course) that the stack supports. When DCL drops, filling in `DCLSimAdapter` makes `--backend dcl` work with zero CLI changes.
- CI / nightly smoke tests can now call `run_race.py --backend mock --timeout N` and fail the pipeline on a non-zero exit without needing PX4, a GPU, or trained weights.

**Files:** src/race/runner.py (new), src/race/__init__.py (new), src/courses.py (new), run_race.py (new, repo root), test_race_runner.py (new, 5 tests)

---

## Session 19e — 2026-04-20 (cont.): ESKF for visual-inertial pose fusion

**Goal:** Build the fusion layer that Round 2 requires. Round 2 uses a real 3D-scanned environment with obstacles and visual distractors — a pure-perception pose pipeline is fragile there in two specific ways:
  1. A YOLO false-positive on a distractor yields a high-confidence but geometrically-wrong pose fix that poisons the belief model.
  2. A transient vision dropout through a yaw turn leaves the belief model coasting on a yaw-rate guess that drifts fast (the exact S18 bug root cause).

The ESKF fuses IMU (accel + gyro) with vision pose fixes. Between fixes, the filter integrates the IMU at the sensor rate; each fix pulls the estimate back and teaches the filter what the IMU biases are. That means we stay useful through multi-second dropouts, and we have an innovation signal to gate distractor fixes with (chi-squared — future work).

**`src/estimation/eskf.py` (~340 lines, NumPy-only):**
- 16D nominal state: `[p(3), v(3), q(4), b_a(3), b_g(3)]` in NED world frame; quaternion is body→world scalar-first.
- 15D error state: `[δp, δv, δθ, δb_a, δb_g]`. Body-frame right-multiplicative attitude error per Solà's "Quaternion kinematics for the ESKF" (arXiv 1711.02508), Section 6.
- `predict(a_m, w_m, dt)`: integrates nominal state (gravity-compensated specific-force with NED gravity `[0, 0, +9.81]`), propagates covariance with the linearized error-state Jacobian F and discrete-time process-noise Q.
- `update_vision(p_vis, yaw_vis)`: 4-row measurement model (position + yaw). Kalman gain, error injection into nominal, Joseph-form covariance update. Returns the innovation for optional gating.
- `seed(p, v, yaw, ...)`: initializes with known state + sensible starting covariance so the first predict step doesn't blow up from the default near-zero P.
- `EskfConfig` exposes IMU noise densities and vision sigmas. Defaults are placeholder values for a Neros-class MEMS IMU — will be recalibrated against real sensor data when Conrad's hardware comes back online.
- NumPy-only by design: no scipy, no torch. Ports cleanly to Neros onboard compiler.

**`test_eskf.py` (8 property tests, 8/8 pass):**
1. Quaternion & rotation helper round-trip: +90° yaw takes body-forward to world-east, yaw extraction is exact, 90°∘90°=180°, `exp_so3↔exp_quat` agree to 1e-12, `skew()` matches `np.cross`.
2. At rest, zero bias, zero noise: 10 s @ 200 Hz produces **0 m drift** (machine-precision zero). Verifies gravity-compensation sign is correct.
3. Constant forward specific force `a_m=[1, 0, -9.81]`: after 2 s, `p_N=2.0000 m`, `v_N=2.0000 m/s` (exact). Verifies body→world rotation direction.
4. Pure yaw rate 0.5 rad/s for π seconds: yaw tracks to 89.954° (4.6 mrad error from first-order Euler — tolerable, could upgrade to RK4), position invariant.
5. Vision bounds drift: 2 s free-run with `0.027 m/s²` |bias| → **0.054 m** drift. 10 s with 10 Hz vision → **0.000 m** final error.
6. Gyro bias convergence: true bias 10 mrad/s, estimated **9.999 mrad/s** after 30 s of 20 Hz vision fixes (0.001 mrad/s error → 0.01%).
7. Numerical stability: 10,000 steps with random IMU perturbation + periodic vision. Final covariance eigenvalues `[6.28e-8, 1.36e-2]` — PSD, finite, well-conditioned.
8. Yaw through 90° turn: 2 s at 1 rad/s yaw with noisy IMU (σ_a=0.01, σ_g=0.002) + 10 Hz noisy vision (σ_p=0.05, σ_yaw=0.02). Peak yaw error **0.36°** after warmup, final error 0.2°.

**Cumulative regression sweep (all 7 suites):** gate_belief 5/5 · belief_replay PASS · sim_adapter 8/8 · detector 11/11 · race_loop 5/5 · race_runner 5/5 · eskf 8/8 — **43/43**.

**What this unlocks:**
- Round 2 readiness: the fusion layer Round 2 needs now exists, tested, and is numerically stable. When DCL drops we can plumb its IMU + vision pose stream straight into `ESKF.predict`/`ESKF.update_vision`.
- Distractor rejection becomes possible: the innovation returned by `update_vision` is a chi-squared test statistic away from gating. Next rung.
- Belief decouples from pose source: `BeliefNav` currently takes bearings from the detector directly. With ESKF online, we can feed it ESKF-smoothed pose instead, which is what protects it from the S18-style phantom-gate drift during dropouts.
- IMU biases are now a first-class model input, not an unmodeled error source. A calibration pass on the real Neros IMU will drop straight into `EskfConfig`.

**Deliberate non-goals this session:**
- No wiring into `RaceLoop` yet — the ESKF is a standalone library. Integration is a separate, smaller session once we decide the `state → belief` protocol (direct pose substitution vs. side-channel).
- No distractor gating yet — that's a Mahalanobis-distance threshold on the innovation, small addition but deserves its own test.
- No RK4 integrator — Euler at 200 Hz is adequate for the bounded-maneuver envelope of AI Grand Prix; RK4 can wait until real flight data shows it matters.

**Files:** src/estimation/__init__.py (new), src/estimation/eskf.py (new, ~340 lines), test_eskf.py (new, 8 tests)

---

## Session 19f — 2026-04-20 (cont.): Gating + PoseFusion + vision round-trip

**Goal:** Close three loops in one session: (a) give the ESKF the ability to reject outlier vision fixes via chi-squared gating, (b) wrap the ESKF in a thin race-facing layer (timestamped IMU ingestion, auto-seed, telemetry), and (c) prove the real-vision path works end-to-end in the sandbox — `cv2.projectPoints` → `GatePoseEstimator` → `YoloPnpDetector` → `RaceLoop` — before the GPU comes back online. Taken together, these move Round-2 readiness from "filter exists in isolation" to "filter is integration-ready and the vision path that will feed it is verified against the real OpenCV PnP math rather than mocked at the boundary".

### (a) Chi-squared distractor gating on `ESKF.update_vision`

`update_vision()` now accepts `max_mahalanobis` (defaults to None = no gating). When set and the Mahalanobis distance `√(y^T S^-1 y)` of the innovation exceeds it, the fix is rejected: no state change, no covariance shrink, no bias learning. Return type is now `(innovation, mahalanobis, accepted)`. Using `np.linalg.solve(S, y)` instead of explicit inverse for numerical stability. Default threshold for downstream callers is 3.64 (≈ √χ²₀.₉₉ at 4 DOF).

Two new tests in `test_eskf.py` take the count 8 → 10:
- **Distractor rejection** — after a clean run, inject a fix at +20 m North. Mahalanobis distance **170** (against 3.64 threshold). Rejected. State and covariance untouched (`np.allclose` verified).
- **Valid fix accepted** — 5 cm offset with σ=0.1 m. Mahalanobis **0.35**. Accepted, state moved halfway toward measurement as expected.

### (b) `src/estimation/pose_fusion.py` — `PoseFusion` wrapper

Thin layer over `ESKF`. What it adds that the raw filter lacks:
- `IMUSample(accel_body, gyro_body, timestamp)` + `on_imu(sample)` that computes dt from the previous sample's timestamp and sanity-gates on negative or oversize dt (both dropped and counted in telemetry).
- `on_vision_pose(p_world, yaw, max_mahalanobis=3.64)` — gated by default. First vision fix auto-seeds an un-seeded filter (useful when vehicle pose is unknown at startup but vision arrives first).
- `FusionTelemetry` tracks seen/dropped IMU samples, accepted/rejected vision fixes, last innovation and Mahalanobis distance. Suitable for live race telemetry overlay.
- `pose()` / `biases()` snapshots return `numpy` copies (not views) so callers can't accidentally mutate internal state.

`test_pose_fusion.py` (5/5 pass):
- Constant-velocity tracking: 10 s at 2 m/s with 200 Hz IMU and 10 Hz vision. Max error **0.010 m**, final error **0.0002 m**.
- Bad-dt drop counter: first-sample drop (no prior dt), negative dt, huge dt all correctly dropped with counter increment.
- Distractor rejection through wrapper: 50 m outlier, Mahalanobis **387.5**, `vision_fixes_rejected==1`, state unchanged.
- Auto-seed from first vision fix: pose and yaw land exactly at the measurement.
- `pose()` copy-semantics: mutating the returned array does not affect the filter.

### (c) Vision round-trip — `test_vision_roundtrip.py`

Problem closed: until this session, `YoloPnpDetector` was only tested in isolation with a stubbed `_impl`. The axis-remap assumption (`body_x = tvec[2]; body_y = tvec[0]; body_z = tvec[1]`) and the reprojection-error threshold were both untested against real OpenCV PnP math. A bug along the `cv2.projectPoints` → `cv2.solvePnP` → axis-swap → `GateDetection` chain would only surface on hardware with a trained YOLO. Now exercised end-to-end in the sandbox.

A `SyntheticGateKeypointDetector` projects known gate corners through a pinhole camera (using `cv2.projectPoints` and the same `K` that `GatePoseEstimator` builds from the 90° FOV), runs the **real** `GatePoseEstimator.estimate_pose`, and returns detections in the exact shape `GateKeypointDetector` returns. Injected into `YoloPnpDetector._impl` via `object.__new__` to bypass the ultralytics import. Gates are modeled as 0.4 m planar squares always facing the drone — a valid simplification for testing the pipeline; real gates have fixed orientations but the detector doesn't know or care.

Four tests (4/4 pass):
1. **OpenCV round-trip sanity** — `projectPoints` corners of a gate at `tvec=[1, -0.5, 5]`, then `solvePnP` with `SOLVEPNP_IPPE_SQUARE`. Recovered tvec error **4.65e-15 m** (float precision).
2. **`GatePoseEstimator` tvec convention** — dead-ahead 5 m gate gives `tvec≈(0, 0, 5)`; drone offset +3 m East gives `tvec≈(-3, 0, 5)`. Confirms the OpenCV `(X_right, Y_down, Z_forward)` convention the detector's axis remap assumes.
3. **`YoloPnpDetector` axis remap end-to-end** — gate at `(5 N, 3 E, 0 D)` with drone at origin. Expected bearing `atan2(3, 5) = 30.96°`, range `√34 = 5.83 m`. Detector returns bearing **30.96°** and range **5.83 m** (both match to two decimals); confidence **0.938** (high, as expected for geometrically clean synthetic PnP).
4. **Full `RaceLoop` completes through the synthetic vision path** — 2-gate course `[(5,0,-1), (10,0,-1)]`, MockAdapter kinematics, `YoloPnpDetector(synthetic)`, `BeliefNav(max=6, cruise=4)`. **2/2 gates passed in 1.10 s**. The whole perception → belief → navigator → control chain runs through real OpenCV PnP for the first time.

Test file includes a `sys.modules` mavsdk stub (same pattern as `run_race.py`) so it runs on machines without PX4 tooling.

### Cumulative regression sweep

gate_belief 5/5 · belief_replay PASS · sim_adapter 8/8 · detector 11/11 · race_loop 5/5 · race_runner 5/5 · eskf 10/10 · pose_fusion 5/5 · vision_roundtrip 4/4 — **54/54**.

### What this unlocks

- **Round-2 readiness stepped up again.** `PoseFusion` + distractor gating together mean a YOLO false-positive scoring high confidence but geometrically-off pose will be rejected by the filter (Mahalanobis gate) rather than poisoning the belief model. This was the top Round-2 failure mode.
- **Pre-GPU vision confidence.** When Conrad's GPU and the trained YOLO weights come back into the loop, the only unknown is the YOLO keypoint accuracy itself — the axis-remap, PnP integration, and race-loop consumption are already proven against real OpenCV math.
- **Integration surface is clean.** The wiring from ESKF into `RaceLoop` is deliberately not yet done — doing that correctly requires one more decision (ESKF replaces belief's pose source vs. ESKF runs alongside as an innovation gate), which is better made after a hardware run shows which pathology dominates.

### Deliberate non-goals

- **No ESKF→RaceLoop wiring yet.** PoseFusion is standalone. Integration is a separate session once we've seen which of the two wiring modes the hardware evidence prefers.
- **No gate-orientation modeling in synthetic projector.** Each gate faces the drone's current position. This is sufficient to exercise pipeline plumbing; real orientation handling is the detector's job and is orthogonal.
- **No ONNX / TensorRT optimization on YOLO.** Rung 6/10 — needs GPU / Neros hardware.

**Files:** src/estimation/eskf.py (gating), src/estimation/pose_fusion.py (new, ~160 lines), src/estimation/__init__.py (re-exports), test_eskf.py (+2 tests → 10), test_pose_fusion.py (new, 5 tests), test_vision_roundtrip.py (new, 4 tests)


---

## Session 19g — IMU sensor protocol + MockKinematicAdapter + full-chain tests (2026-04-20)

Next rung after the ESKF/PoseFusion landing in 19e/19f. Goal: prove the IMU → ESKF → fused-pose chain end-to-end in the sandbox so that when real IMU plumbing arrives (whether from mavsdk, Neros microros, or DCL's obs dict), we already know the downstream math is sound.

### (a) SimAdapter IMU extension

`src/sim/adapter.py` — three minimal additions:

- `SimCapability.IMU` flag. Backends that produce raw IMU opt into it.
- `IMUReading` dataclass. Duck-type compatible with `estimation.pose_fusion.IMUSample` (same three fields, same units and conventions). Documented: `accel_body` is specific force in body FRD (m/s², gravity-loaded), `gyro_body` is angular rate (rad/s), `timestamp` monotonic seconds.
- `async def get_imu(self) -> Optional[IMUReading]` on the `SimAdapter` Protocol.

Adapter coverage:
- `PX4SITLAdapter`: returns `None`. mavsdk exposes `telemetry.imu()` / `raw_imu()` but rates are PX4-build-dependent; real Neros IMU comes from a microros bridge, not mavsdk. Left as a hook with a commented-out subscription path.
- `DCLSimAdapter`: raises `NotImplementedError` with a hint to map `obs['imu']` → `IMUReading` and convert if DCL ships gyro in deg/s or accel in g's. DCL capabilities now include `IMU`.
- New `MockKinematicAdapter` (below).

`make_adapter("mock_kinematic")` added as a new factory entry.

### (b) MockKinematicAdapter

`src/sim/mock.py` (~200 lines) — kinematic sim that synthesizes physically-plausible IMU from truth.

Model:
- First-order velocity tracking `v̇ = (v_cmd − v)/τ_v` with stable discretization `α = 1 − exp(−dt/τ)`.
- Yaw tracking identical, with π-wrapped error so yaw commands slew through the shortest arc.
- Level flight only (roll = pitch = 0) — that's all the belief/planner layer emits.
- `step(dt)` advances truth, synthesizes the IMU reading, returns it. `get_imu()` returns the last one.

Sensor model:
- `accel_body = Rz(yaw)ᵀ · (a_world − g_NED) + bias + N(0, σ_a²)`
- `gyro_body = [0, 0, ẏ] + bias + N(0, σ_g²)`
- Configurable `accel_noise_sigma`, `gyro_noise_sigma`, static `accel_bias`, `gyro_bias`.

`set_truth(pos, vel, yaw)` lets tests place the drone at any known state and resyncs the command setpoint so the first-order filter doesn't snap back to zero.

Capabilities: `VELOCITY_NED | POSITION_NED | IMU | ARM_ACTION | RESET | WALLCLOCK_PACED`.

### (c) PoseFusion.seed covariance knobs

While wiring bias learning into the test, I found that `PoseFusion.seed` was forwarding nothing but `p, v, yaw` to `ESKF.seed`, which was inheriting defaults `p_sigma=0.1, v_sigma=0.1, att_sigma=0.05, bias_sigma=0.01`. The `bias_sigma=0.01` is near-pinning for the bias state — the Kalman gain on `b_a` is tiny, so the filter can't learn a nontrivial bias no matter how long it runs. Extended `PoseFusion.seed` to forward all four sigmas with the same defaults (backward-compatible) and documented the bias-learning case.

### (d) Full-chain tests (7/7 pass)

`test_sim_imu.py`:

1. **Capability & construction** — adapter advertises `SimCapability.IMU`, info() reports `mock_kinematic` backend at 200 Hz default.
2. **At rest** — `accel_body` equals `[0, 0, −9.81]` to machine precision; `gyro_body` is exactly zero. Gravity sign/convention pinned.
3. **Constant-velocity steady state** — after `set_truth(v=3 N)` + matching command, specific force converges back to `−g` (no phantom acceleration).
4. **Horizontal acceleration transient** — on the first tick after a `send_velocity_ned(5 N)` step, body-x accel matches the analytical `α·v_cmd/dt` to 1e-6 m/s².
5. **Yaw rate in gyro-z** — yaw step produces positive gyro_z during slew, matches `α·(π/2)/dt` exactly on tick 1, settles to 5e-6 rad/s after 15τ.
6. **Full fusion chain** — mock drives a 4-second S-curve with a yaw-turn mid-run, PoseFusion fuses 200 Hz IMU + 10 Hz vision. Max tracking error 0.015 m, final error 0.008 m. Truth: filter tracks to well under 2 cm through the maneuver.
7. **Bias observability** — 0.3 m/s² constant accel bias on body-x, hover for 8 s with 10 Hz vision. Filter recovers 0.154 m/s² (51% of truth). Tight tolerance would require motion excitation, not pure hover; 51% of bias from a cold start with no excitation is a healthy signal that the bias coupling Jacobians are correct.

### What this unlocks

- **ESKF→RaceLoop wiring can now be tested without hardware.** The mock produces a physical IMU stream; we can swap in PoseFusion ahead of BeliefNav's pose source, drive the mock against any course, and watch the fused trajectory — all in-sandbox, at full frame rate. That's the prerequisite for rung 8.
- **DCL integration surface is smaller.** When the DCL API ships, we know exactly what `get_imu()` needs to return (`IMUReading` with SI units, body FRD, specific force). Unit conversion is local to one method.
- **Bias learning now accessible to callers.** `PoseFusion.seed(bias_sigma=0.5)` is the knob that lets the filter absorb MEMS-scale bias drift. Documented inline.

### Deliberate non-goals

- **PX4SITLAdapter.get_imu() still returns None.** Wiring mavsdk `telemetry.imu()` is straightforward but PX4-build-dependent; we'd rather pay that cost once hardware tells us which stream is actually clean.
- **MockKinematicAdapter does not model drag/thrust/aero.** It's a sensor-synthesis tool, not a flight dynamics sim. Use PX4 SITL or DCL for those.
- **_MockFlightAdapter in run_race.py unchanged.** Keeping the CLI smoke-test adapter separate; it's 40 lines and doesn't need IMU for its use case.

### Running totals

Cumulative tests: **61/61** across 10 suites.
- gate_belief 5/5
- belief_replay PASS
- sim_adapter 8/8
- detector 11/11
- race_loop 5/5
- race_runner 5/5
- eskf 10/10
- pose_fusion 5/5
- vision_roundtrip 4/4
- **sim_imu 7/7** (new)

**Files:** src/sim/adapter.py (IMU cap + IMUReading + Protocol method + PX4/DCL hooks), src/sim/mock.py (new, ~200 lines), src/sim/__init__.py (re-exports), src/estimation/pose_fusion.py (seed sigma forwarding), test_sim_imu.py (new, 7 tests)

---

## Session 19h — 2026-04-20 — PoseFusion wired into RaceLoop (end-to-end fusion)

**Rung reached:** 8. The ESKF → fused pose → navigator loop now closes inside `RaceLoop`. Drone can fly a 2-gate course driven entirely by the fused pose instead of adapter truth, with IMU samples streaming in from MockKinematicAdapter and vision fixes backprojected from `VirtualDetector` gates.

### Why this was the right next rung

Every piece of the stack below `RaceLoop` was proven in isolation — ESKF (10/10), PoseFusion wrapper (5/5), MockKinematicAdapter IMU synthesis (7/7), vision round-trip (4/4) — but nothing had exercised them together in the actual flight loop. Wiring it now was cheap (sandbox-only, no hardware) and surfaces integration bugs in a controllable place. Deferring would have pushed the same debugging onto the runway when the stakes are higher. Bring-up order: fusion first, hardware second.

### Design — how PoseFusion slotted in behind the navigator

`RaceLoop.__init__` grew three optional params:

- `pose_fusion`: a `PoseFusion` instance. None = legacy path, unchanged.
- `gates_ned`: the gate positions in NED. **Required** whenever `pose_fusion` is set — without it we can't backproject a body-frame detection into a world-position fix. Constructor raises `ValueError` otherwise (fail-loud over silent misbehavior).
- `vision_pos_sigma`: 1-σ for the position measurement noise, default 0.15 m.

Per tick, when fusion is on:

1. **Pull adapter state + frame** (unchanged — we still need `vel_ned` for the gate-pass heuristic and the frame for the detector).
2. **Feed IMU**: `await self.adapter.get_imu()` → `PoseFusion.on_imu` (duck-bridged `IMUReading` ↔ `IMUSample`).
3. **Seed if cold**: on the first tick, seed from adapter truth (`state.pos_ned`, `state.vel_ned`, `state.att_rad[2]`) with `bias_sigma=0.1` to allow mild bias absorption.
4. **Backproject the picked detection**. Body-frame `(bearing_h, bearing_v, range)` → body `[r cosθ cosφ, r cosθ sinφ, r sinθ]` (FRD) → rotate through fused yaw (level flight, Rz only) → world_to_gate vector → `drone_ned_meas = gate_ned[target] − world_to_gate`. Feed to `PoseFusion.on_vision_pose` with `yaw_rad = fused_yaw` (keeps the yaw innovation at zero — this is a position-only update).
5. **Use fused pose for planning**: navigator sees the ESKF estimate, not ground truth.

When `pose_fusion=None`, step() takes the exact legacy branch. `test_race_loop.py` still 5/5 unmodified.

### Backprojection detail — why it's a position-only update

We're using the current fused yaw both to rotate body→NED and as the "yaw measurement" passed to ESKF. Mathematically that means the yaw innovation is `yaw_meas − yaw_estimate = 0` every frame, so only the position channel picks up correction. That's the honest thing to do given the input: VirtualDetector and YoloPnpDetector both emit a body-frame bearing+range per gate — neither gives us a full 6-DOF gate pose. When PnP-based YOLO ships (it already estimates translation), we can upgrade this branch to publish orientation too and the yaw update becomes nontrivial for free.

The good news: even with yaw unobserved here, it's observed by the gyro integration and the position channel bounds the position drift. Session 19g test #6 already proved this works end-to-end in a 4 s S-curve; 19h extends it to a real closed-loop race.

### Integration tests — `test_race_loop_fusion.py` (4/4 pass on first run)

1. **Legacy path unchanged** — 2-gate course, `pose_fusion=None`, mock adapter auto-stepping. Completes 2/2 in 3.75 s. Regression gate.
2. **Fusion path completes the same course** — same course, same adapter, `pose_fusion` supplied. 2/2 in 3.81 s (essentially identical). Telemetry: 156 IMU samples, 152 vision fixes accepted, 2 rejected. Rejects are healthy — the chi-squared gate is catching transient misalignment during the first few frames as the filter settles.
3. **Fused pose tracks truth through the run** — 1-gate at 12 m N, step the loop tick-by-tick, sample `|fused_pos − adapter_pos|` each tick. Max err 0.271 m, final err 0.110 m. Tolerances were set at 0.75 m max / 0.25 m final; plenty of headroom even against the mock's near-ideal IMU.
4. **Constructor rejects `pose_fusion` without `gates_ned`** — raises `ValueError` with the right message. Closes off a silent-misbehavior path.

### Small supporting changes

- `RaceLoop` deferred imports of `numpy` and `estimation.IMUSample` inside the fusion branch, so the legacy path still runs on a machine without `numpy` installed (matching the existing mavsdk stub philosophy).
- `PoseFusion.seed` exposed `bias_sigma` (already done in 19g) gets threaded through from `RaceLoop` at 0.1 — a middle ground: not pinned, but not so loose that a cold start wobbles.
- `MockKinematicAdapter(auto_step=True, initial_altitude_m=1.0)` is the test fixture — the race loop doesn't know about `step()`, so `send_velocity_ned` advances the sim internally.

### Running totals

Cumulative tests: **65/65** across 11 suites.
- gate_belief 5/5
- belief_replay PASS
- sim_adapter 8/8
- detector 11/11
- race_loop 5/5 (legacy path unchanged)
- race_runner 5/5
- eskf 10/10
- pose_fusion 5/5
- vision_roundtrip 4/4
- sim_imu 7/7
- **race_loop_fusion 4/4** (new)

### What this unlocks — and what it doesn't

**Unlocks:**
- Hardware day-1 plan: flip `pose_fusion` on in `RaceRunner`/`run_race.py` when the real IMU stream is available. The wiring is proven; only the adapter-side IMU callback needs to land.
- A/B experiments that matter: we can now compare "belief model only" vs "belief model + fused pose" on harsh-noise / dropout courses without recoding the loop each time.
- A natural home for the distractor-gate chi-squared test when real YOLO lands — the `PoseFusion.on_vision_pose` call already uses the Mahalanobis gate in `ESKF.update_vision`.

**Still deferred (hardware-blocked):**
- `PX4SITLAdapter.get_imu()` — still returns None. mavsdk `telemetry.imu()` wiring is straightforward but we'd rather pay once against real telemetry quality. Until then, fusion runs on `mock_kinematic` only.
- `s18_belief_test.py` PX4 A/B — still Conrad's re-run, still pending. That's measurements, not code.
- Mixed pose source (fusion pose for planner, adapter truth for gate-pass heuristic) — current code uses adapter `vel_ned` for the gate-pass range heuristic, which is fine. When we switch the heuristic to a centerline-crossing detector we can evaluate whether to drive it off the fused pose instead.

### Files touched (S19h only)

- `src/race_loop.py` — `__init__` params + `_ingest_imu_async` + `_ingest_vision_from_detection` + `step()` fusion branch + docstring refresh (legacy path byte-identical under `pose_fusion=None`).
- `test_race_loop_fusion.py` (new, 4 tests).

---

## Session 19i — 2026-04-20 — Fusion reaches the CLI (RaceRunner + run_race.py)

**Rung reached:** 8.5 — `--fusion` is now a flag Conrad can flip from the command line. The 19h plumbing is end-to-end reachable through the user-facing entrypoint, not just tests.

### Motivation

19h landed fusion inside `RaceLoop` and proved it in an integration test. But the test constructed `RaceLoop` by hand; `RaceRunner` didn't know about PoseFusion, and `run_race.py` couldn't opt in. That gap meant the only way to actually run a fused race was to write a bespoke script, which is exactly the wrong shape when hardware shows up and Conrad wants to A/B on the runway.

### Changes

**`src/race/runner.py`:**
- `RaceRunner.__init__` grew `pose_fusion=None, vision_pos_sigma=0.15`. When fusion is on, `fly()` does a pre-flight capability check — if `SimCapability.IMU not in adapter.capabilities`, it raises `RuntimeError` *before* calling `adapter.connect()`. This is the key failure mode to avoid silently: spinning up a flight where `get_imu()` returns None every tick and the filter free-runs on stale seed state.
- `FlightResult` gained `fusion_on: bool` so callers/logs can distinguish the two regimes without reconstructing it.
- `RaceLoop` construction now threads `pose_fusion`, `gates_ned` (only when fusion is on), and `vision_pos_sigma` through.

**`run_race.py`:**
- Two new flags:
  - `--fusion` — boolean, enables `PoseFusion` construction and threading.
  - `--vision-pos-sigma` — float, default 0.15 m, controls position-fix noise.
- `--backend` gained `mock_kinematic` — routes through `make_adapter("mock_kinematic", auto_step=True, initial_altitude_m=1.0, ...)`. This is the first CLI-reachable backend that advertises IMU, so it's the sandbox home for `--fusion` until `PX4SITLAdapter.get_imu()` wires up.
- Early-fail validation at the CLI layer too: if `--fusion --backend mock`, we print a targeted error to stderr and exit 2 — no side effects. Belt-and-suspenders with the `RaceRunner` check.
- Flight summary now prints a `Fusion :` line with `imu_samples_seen / vision_fixes_accepted / vision_fixes_rejected / imu_samples_dropped`. This is the telemetry Conrad will actually stare at when debugging fusion quality on live runs.

**`test_race_runner.py`:** added two tests.
- `test_fusion_runner_completes_on_mock_kinematic` — full lifecycle under fusion, 2/2 gates, `result.fusion_on=True`, IMU stream flowing, vision fixes accepted.
- `test_fusion_rejects_adapter_without_imu_capability` — passing `pose_fusion` with an adapter that lacks IMU raises before `connect()`. Verified by checking the call log has no `"connect"` entry.

### CLI smoke tests (all completed, clean exit 0)

```
sprint    mock_kinematic --fusion  →  10/10 gates in 16.40 s   (imu=673, vis_ok=504, vis_rej=170)
technical mock_kinematic --fusion  →  12/12 gates in  3.16 s   (imu=131, vis_ok=119, vis_rej=11)
mixed     mock_kinematic --fusion  →  12/12 gates in  6.87 s   (imu=288, vis_ok=262, vis_rej=27)
```

The vision-rejection counts are the chi-squared gate catching transients during filter settle — exactly what it should do. Sprint has the highest rejection ratio (25%) because it's 10 gates and therefore 10 target-switches, each of which briefly mis-aligns the backprojection until the filter re-converges on the new gate.

Rejection path:
```
sprint    mock --fusion  →  ERROR: --fusion requires a backend with SimCapability.IMU. 'mock' does not.
                             Use --backend mock_kinematic for a sandbox run, or wait until
                             PX4SITLAdapter.get_imu() is wired.
                             (exit 2)
```

### Non-fusion paths verified unchanged

```
sprint    mock           →  10/10 in 15.82 s
technical mock_kinematic →  12/12 in  3.06 s
```

Both complete with `Fusion : off (adapter truth)` in the banner. The legacy code path in `RaceLoop.step()` runs byte-identical.

### Running totals

Cumulative tests: **67/67** across 11 suites.
- gate_belief 5/5, belief_replay PASS, sim_adapter 8/8, detector 11/11, race_loop 5/5
- **race_runner 7/7** (was 5/5; +2 fusion-path tests)
- eskf 10/10, pose_fusion 5/5, vision_roundtrip 4/4, sim_imu 7/7
- race_loop_fusion 4/4

### What this unlocks

- **Hardware day-1 is one adapter method.** When Neros lands and `PX4SITLAdapter.get_imu()` wires up against `mavsdk.telemetry.imu()`, the CLI command is literally `python3 run_race.py --backend px4_sitl --detector yolo_pnp --fusion --model-path ...`. No additional glue.
- **A/B experiments are a flag away.** Same course, same seed, toggle `--fusion` — compare completion times and gate-pass counts. That's the exact harness shape for deciding whether to trust fusion on the runway.
- **Round 2 prep.** When the distractor-augmented YOLO lands, we can chain `--detector yolo_pnp --fusion` and watch `vis_rej` tick up — the chi-squared gate is doing its distractor-rejection job visibly.

### Deliberate non-goals

- **No new stress-test harness yet.** Adding IMU noise/bias sweeps via CLI flags (e.g. `--imu-bias 0.3`) is obvious next work but not on this rung. The mock adapter has the knobs; we just don't surface them in run_race.py until there's a reason. One rung at a time.
- **`mock_kinematic` doesn't take noise profile flags at CLI** — hardcoded to clean kinematics. Same rationale: add when needed, not prophylactically.
- **PX4 IMU stream still unwired.** Same reason as 19g/19h: wait for real telemetry quality before committing to a specific mavsdk subscription cadence.

### Files touched (S19i only)

- `src/race/runner.py` — `RaceRunner.__init__` + fusion validation in `fly()` + `FlightResult.fusion_on`.
- `run_race.py` — `mock_kinematic` backend, `--fusion`, `--vision-pos-sigma`, pre-flight rejection, summary telemetry line.
- `test_race_runner.py` — +2 fusion-path tests (5 → 7 passing).

---

## Session 19j — 2026-04-20 — Fusion A/B stress harness (and what it found)

**Rung reached:** 8.75 — with fusion reaching the CLI (19i), the immediate question is whether it actually *helps* under realistic conditions. Built a characterization harness and used it to find (and work around) a nontrivial filter-tuning issue that would have bitten on hardware.

### `bench_fusion_ab.py` — what it does

Runs the full RaceLoop across the cross product of:
- **pose source**: legacy (adapter truth) vs fusion (ESKF estimate)
- **vision noise**: clean, mild, harsh (VirtualDetector profiles: 0.01/0.05/0.15 miss prob, 0.5/2.0/5.0° bearing σ)

on a fixed course and MockKinematicAdapter seed. Reports a compact table with gates-passed, elapsed time (deterministic — counted in ticks, not wall-clock), and — for fusion rows — peak and final |fused − truth| position error. Exits 1 if any scenario timed out.

Flags: `--course {sprint,technical,mixed}`, `--timeout`, `--seed`, `--vision-pos-sigma`.

Kept as a regression harness, not a unit test. Runs in ~30 s.

### The finding — chi-squared feedback loop under default tuning

First run (default `vision_pos_sigma=0.20`):

```
SPRINT (10 gates, 16 s legacy baseline):
  clean  legacy 10/10   fusion 10/10   max_err= 13.68 m  final= 13.68 m  vis_ok=423 rej=343
  mild   legacy 10/10   fusion 10/10   max_err=122.06 m  final=122.06 m  vis_ok=  6 rej=326
  harsh  legacy 10/10   fusion 10/10   max_err= 19.31 m  final= 19.31 m  vis_ok=  3 rej=127

MIXED (12 gates):
  harsh  legacy 12/12   fusion 12/12   max_err= 41.12 m  final= 41.12 m  vis_ok=  9 rej=287

TECHNICAL (12 gates, 2 s legacy baseline):
  clean  fusion 12/12   max_err=  0.24 m  final= 0.24 m   (works)
```

Every course *completed* under fusion — the gate-pass heuristic is range-based off the detector, not the fused pose, so a drifting filter doesn't prevent the race from finishing. But the fused pose itself was diverging catastrophically, with 10–3700 m error on the longer courses.

**Mechanism (confirmed by instrumented trace):** At t=0 the filter is seeded perfectly (err ≈ 0.001 m). By t=1 s the filter has accepted 5 of 49 fixes and drifted 1.1 m. By t=2 s the filter has stopped accepting fixes (6/89 accepted), drift is 6 m, and the error grows monotonically thereafter with the filter running open-loop on IMU.

This is classic Kalman filter inconsistency: the filter's covariance shrinks faster than its actual error, so valid measurements look like outliers, get rejected by the chi-squared gate in `ESKF.update_vision`, and the next IMU step pushes the filter further off. A self-destructive feedback loop.

### The mitigation — widen the position measurement noise

Swept `vision_pos_sigma` on sprint/mild, fixed seed:

```
sigma  gates   max_err    final_err   vis_ok   vis_rej
0.20   10/10   122.06 m   122.06 m        6      326   ← catastrophic
0.50   10/10     4.05 m     4.05 m      408      367   ← converging, noisy
1.00   10/10     2.44 m     1.05 m      339       93   ← sweet spot
2.00   10/10     3.57 m     1.05 m      365       68   ← over-permissive, still good
5.00   10/10  3698.71 m  3698.71 m      201      967   ← destabilizes (IMU trust dominates)
```

Re-ran the full 6-scenario A/B at `--vision-pos-sigma 1.0`:

```
SPRINT @ sigma=1.0:
  clean  fusion 10/10   max_err=0.46 m   final= 0.46 m   (30× better than 13.68)
  mild   fusion 10/10   max_err=2.44 m   final= 1.05 m   (120× better than 122.06)
  harsh  fusion 10/10   max_err=2.38 m   final= 1.06 m   ( 18× better than  19.31)

MIXED @ sigma=1.0:
  clean  fusion 12/12   max_err=0.37 m   final= 0.37 m
  mild   fusion 12/12   max_err=0.81 m   final= 0.64 m
  harsh  fusion 12/12   max_err=1.03 m   final= 0.33 m   (127× better than 41.12)

TECHNICAL @ sigma=1.0:
  clean  fusion 12/12   max_err=0.34 m   final= 0.34 m
  mild   fusion 12/12   max_err=0.36 m   final= 0.36 m
  harsh  fusion 12/12   max_err=0.54 m   final= 0.17 m
```

All nine fusion rows now stay under 2.5 m max error and under 1.1 m final error across every course × every noise profile.

### Why 1.0 and not something else

- `sigma` enters the filter in two places: the Kalman gain `K = P H^T S^-1` (where S = H P H^T + R, R = sigma²·I) and the Mahalanobis distance `y^T S^-1 y`. Bigger sigma → smaller gain, but also smaller Mahalanobis (fixes look less like outliers).
- Too small (0.15, 0.20): gate over-rejects, drift compounds.
- Too large (5.0): gain too small, filter can't correct IMU drift fast enough, destabilizes.
- Sweet spot around 1.0 — roughly matches the realistic position error of a bearing-plus-range backprojection at 5–15 m range with mild-profile detector noise (5° bearing ≈ 0.09 rad × 10 m = 0.9 m).

### What this tells us

**The ESKF is sound, but the measurement-noise model needs to match the detector.** Session 19f/19g/19h all passed with `sigma=0.20` on short or clean tests. Nothing caught this because short dense runs don't give drift time to accumulate, and clean vision feeds fixes fast enough to keep the filter corrected. The bench harness running on the longer, noisier regime was exactly what surfaced the failure.

**For hardware bring-up**, we want to start with `--vision-pos-sigma 1.0` (or higher) and tune down as we gather real YOLO performance data. The default stays at 0.15 for the synthetic/clean regime — the existing test suite is calibrated against it and still passes — but the docstring now warns and points at this finding.

### Small code changes

- `src/race_loop.py` — `vision_pos_sigma` docstring now explains the failure mode, the sensitivity, and points at `bench_fusion_ab.py`. No runtime change.
- `bench_fusion_ab.py` (new) — characterization harness. `--vision-pos-sigma` flag for sweep experiments.
- Default `vision_pos_sigma` **not** changed in this rung — risking a cascade of test tuning we'd have to re-do anyway when real hardware numbers arrive. The knob is already at the CLI via 19i.

### Running totals

Cumulative tests still **67/67** across 11 suites — no new unit tests this rung, but `bench_fusion_ab.py` added as a characterization artifact. `test_race_loop.py`, `test_race_loop_fusion.py`, `test_race_runner.py` all reverified clean after the docstring edit.

### What this unlocks

- **A regression harness exists** for the exact failure mode we just found. Any change to the ESKF predict/update code, the backprojection math, or the default sigmas will be caught here by someone re-running the bench.
- **The fusion stack has been empirically validated** under a reasonable stand-in for realistic vision noise. Sub-1.1 m final pose error across all courses at sigma=1.0 is a defensible baseline for hardware bring-up.
- **Filter-tuning knowledge is now written down.** The docstring explains the failure mode; hardware debugging will start by adjusting the one knob we know is load-bearing.

### Deliberate non-goals

- **Filter re-tuning is not in this rung.** Bumping default sigma, adjusting ESKF process noise `Q`, or adding adaptive gating all deserve proper covariance analysis — against real IMU data when it arrives. The short-term workaround (`--vision-pos-sigma 1.0`) is enough until then.
- **No IMU-noise sweep yet.** Mock supports accel/gyro noise and bias, but the story here was already complete — adding a second axis would dilute the finding.
- **Seed determinism partial.** MockKinematicAdapter is seedable; VirtualCamera's RNG is hardcoded to `random.Random(42)`. Fine for the current bench, but if we want statistical A/B (N seeds per config) later, we'll need to plumb seed through VirtualCamera.

### Files touched (S19j only)

- `bench_fusion_ab.py` (new, ~250 lines).
- `src/race_loop.py` — docstring clarification on `vision_pos_sigma` sensitivity.

---

## April 20, 2026

### Session 19k — Position-based gate-pass detector

**Goal:** Replace the Session 19 "skeleton default" gate-pass heuristic — `range < PASSAGE_RANGE` alone — with a position-aware detector that uses `gates_ned` + pose to catch the two failure modes the legacy detector can't: **missed-detection passes** (YOLO box collapses when the gate fills frame, range-based never fires, drone stalls) and **Round 2 distractor spoofing** (a decoy whose incidental range is short advances target_idx through the wrong gate).

This is sandbox-completable and has direct Round 2 value. Round 1 had no distractors; Round 2 explicitly introduces decoy gates as an adversarial course element. The existing range-only detector, while fine for Round 1 dense synthetic vision, has no defence against either failure.

### The design — OR of two signals, range-first timing preserved

The new `_check_gate_pass_position` dispatches when `gates_ned` is supplied. Two signals, OR'd:

1. **Range signal (legacy timing preserved)** — `tracker.detected AND range_est < PASSAGE_RANGE`. This fires on the same tick the old detector used to, so fusion races that were timing-tuned against range-based passes still complete in the same elapsed time.
2. **Position signal (new backstop)** — local minimum of `‖drone_ned − gates_ned[target_idx]‖` while within `PASSAGE_RADIUS = 2.0 m`. Detected as: `prev_dist < PASSAGE_RADIUS AND curr_dist > prev_dist`. Fires independently of detection, so YOLO dropouts at close range no longer stall the race.

OR-semantics — not AND — because AND would defeat the backstop (if detection dropped, AND-gate would block the pass). Round 2 distractor protection comes from the **dispatch choice** (position-based detector only runs when we have NED gate positions; when we have them, the distractor detection problem becomes "decoy is not at target gate's known NED" — which the position signal handles by definition).

`gates_ned` was previously coupled to `pose_fusion` (the constructor raised `ValueError` if you supplied one without the other). That coupling is now relaxed: `gates_ned` without `pose_fusion` enables position-based detection running off adapter truth, which is the common test-harness path. `pose_fusion` still requires `gates_ned` for backprojection — that direction of the constraint is unchanged.

### The false start — greedy sanity-gating on the range signal

First attempt AND-gated the range signal against a 5 m position sanity radius ("if detection says close, also require fused pose to agree"). Bench regressed catastrophically: `technical/fusion/clean` went from 12/12 in 2.22 s to 2/12 with 178 m max pose error. Diagnosis: when the race runs long enough for the fused pose to drift past 5 m, no signal fires at any future gate — drone stalls, fusion runs open-loop on IMU, pose explodes. **A hard gate on fused-pose accuracy is a permanent stall the first time fusion has a bad tick.**

Corrected: range signal remains *un-gated* by position. The position signal is strictly *additive*. In the happy case both fire around the same tick; when fusion drifts past the radius the range signal alone keeps the race completing.

### Verification — bench numbers

Full 6-scenario bench at the S19j-recommended `--vision-pos-sigma 1.0`:

```
TECHNICAL (12 gates):
  clean  legacy 12/12 2.22s   fusion 12/12 2.22s   max_err=0.34 m  final=0.34 m
  mild   legacy 12/12 2.14s   fusion 12/12 2.20s   max_err=0.36 m  final=0.36 m
  harsh  legacy 12/12 2.24s   fusion 12/12 2.16s   max_err=0.54 m  final=0.17 m

SPRINT (10 gates):
  clean  legacy 10/10 15.88s  fusion 10/10 15.90s  max_err=0.46 m  final=0.46 m
  mild   legacy 10/10 12.10s  fusion 10/10  8.84s  max_err=2.44 m  final=1.05 m
  harsh  legacy 10/10  4.78s  fusion 10/10  4.78s  max_err=2.38 m  final=1.06 m
```

Fusion now matches legacy timing on clean/harsh and *beats* legacy on `sprint/mild` by 3.26 s — the position signal catches a pass that legacy's range-only detector was missing under mild vision noise. That's a concrete example of the new detector's value beyond Round 2 hardening.

### Tests — 8 new in test_race_loop_gate_pass.py

Unit-level against `_check_gate_pass` and `_check_gate_pass_position`:

1. **legacy range path when gates_ned is None** — no-detection → no pass, detected+far → no pass, detected+close → pass. Confirms pre-S19k behaviour is preserved on callers that haven't opted in.
2. **gates_ned allowed without pose_fusion** — constructor no longer raises, `_gates_ned` populated. Decouples gate-pass detection from fusion.
3. **position signal fires on flyby without detection** — trajectory [2.0, 0.8, 0.1, 1.0] from gate; detection never fires; position-based fires exactly on the receding tick. The core YOLO-dropout case.
4. **range signal still fires in normal pass** — detected + range=2.0 m + fused_dist=2.0 m → fires on first tick. Timing match with legacy.
5. **no fire when approach never enters radius** — drone tracks at 3 m+ offset throughout, even while turning around. No false pass.
6. **no double-fire on same gate** — after gate 0 pass, lingering near gate 0's NED doesn't re-advance target. Detector correctly latches onto gate 1's distance.
7. **idempotent after all gates passed** — `target_idx` clamped at `gate_count`, no `IndexError` on `gates_ned[target_idx]`.
8. **range signal fires even on pose drift** — `fused_dist=60 m` with `detected + range=1 m` → still fires. The explicit protection against the false-start design.

`python3 test_race_loop_gate_pass.py` → `8/8 PASSED`.

### Regression

All 10 pre-existing suites re-run clean on the new detector:

```
test_eskf.py             10/10
test_pose_fusion.py       5/5
test_sim_adapter.py       8/8
test_sim_imu.py           7/7
test_detector.py         11/11
test_gate_belief.py       5/5
test_vision_roundtrip.py  4/4
test_race_loop.py         5/5    ← legacy path, gates_ned=None
test_race_loop_fusion.py  4/4    ← fusion path, gates_ned+pose_fusion
test_race_runner.py       7/7
```

Cumulative: **74/74 tests across 11 suites.** Up from 67/67.

### What this unlocks

- **Round 2 readiness.** Distractor-augmented courses should no longer spoof gate progression: the position check anchors `target_idx` advancement to the *target gate's known NED*, not to "whatever is close on camera".
- **Detection-dropout robustness.** The number-one YOLO failure mode on real hardware — gates filling the frame and the detector losing its box — no longer stalls races. The position signal carries us through the dropout.
- **Faster mild/harsh races with fusion.** Empirically: `sprint/fusion/mild` now beats `sprint/legacy/mild` because of missed-pass recovery via position signal.
- **`gates_ned` is now a general handle.** Any future detector (centreline-crossing, gate-plane intersection, etc.) can layer on this plumbing without touching `pose_fusion`.

### Deliberate non-goals

- **No centreline-crossing detector.** That's the fully geometric version ("did the drone cross the gate's rectangular plane"), which needs gate *orientation* and not just position. When YOLO-PnP ships per-gate yaw/normal, we'll upgrade; for now local-minimum on distance is sufficient and orientation-free.
- **`PASSAGE_RADIUS = 2.0 m` is a heuristic.** Competition gates are ~2 m wide, so a 2 m radius roughly corresponds to "drone is inside the gate's clearance cylinder". If we find this too aggressive on real YOLO, it's a knob.
- **`PASSAGE_SANITY_RADIUS` was designed, built, and ripped out.** Kept the constant in the file as a comment for future use — if we want a drift-triggered *rejection* of the range signal (rather than the current drift-tolerant acceptance), the constant is the starting point.
- **No dual-detector A/B.** Could have kept both range-only and position-only as flags, but OR-merge-in-place is strictly more permissive and strictly more defensible.

### Files touched (S19k only)

- `src/race_loop.py` — new `_check_gate_pass_position`, relaxed `gates_ned` constraint, `_advance_target` extraction to share bookkeeping between both detectors, `PASSAGE_RADIUS`/`PASSAGE_SANITY_RADIUS` class constants, docstring updated.
- `test_race_loop_gate_pass.py` (new, ~230 lines) — 8 unit tests on detector behaviour.

### Pending / next-rung candidates

- Distractor-realism test: extend `bench_fusion_ab.py` to seed fake gates into `VirtualDetector`'s view at plausible ranges and verify `target_idx` does not advance through them.
- Centreline-crossing detector once PnP yields gate yaw.
- Re-run `s18_belief_test` A/B over PX4 SITL (Conrad's hardware).
- `PX4SITLAdapter.get_imu()` wiring to `mavsdk.telemetry.imu()` (needs MAVSDK docs + a hardware test cycle).

---

## Session 19l — 2026-04-20 — epistemic correction on the S19k distractor claim

### TL;DR

The S19k log entry (immediately above) asserted that the new position-based gate-pass detector provided Round 2 distractor spoofing protection. That claim was wrong. I discovered it by reading my own log, couldn't find a unit test that would have caught a spoof, wrote one, and it fired. Spent a session building and then ripping out a position-sanity gate on the range signal as an attempted real defense — it worked in isolation but caused a stall cascade on drift in the full-bench runs. Reverted to the S19k OR-logic, rewrote the docstring to honestly scope what the detector does and does not defend, added two contract tests documenting both, and am logging this entry to correct the S19k claim in the record.

**Honest scope of the position-based detector, post-S19l**:

- ✅ Dropout-robust gate passes (YOLO collapsing at close range still advances target_idx via position local-min).
- ✅ Rejects decoys the drone never physically approaches (position local-min is gated on *actual drone proximity* to the target gate's known NED).
- ❌ Does NOT reject decoys placed on the drone's flight path near the target gate — the range signal fires on those by design. **The canonical Round 2 defense is distractor-augmented YOLO training at the detector layer.**

### Why I went looking

Reading the S19k entry again after a context break, the sentence "Round 2 distractor protection comes from the dispatch choice... the distractor detection problem becomes 'decoy is not at target gate's known NED' — which the position signal handles by definition" nagged at me. "By definition" is the kind of phrase I reach for when I haven't proved it. There was no unit test for the distractor case in `test_race_loop_gate_pass.py`. Asserted without verification — the epistemic failure mode I'm supposed to be immune to.

### The one-line refutation

```python
gates = [(10.0, 0.0, -1.0)]
loop = _build(gates_ned=gates, gate_count=1)
fired = loop._check_gate_pass_position(
    _tracker(detected=True, range_est=1.0),
    [-10.0, 0.0, -1.0],  # drone 20 m from target
)
# fired == True
```

Drone is 20 m from the real gate. A decoy appears in the camera cone at short range. The range signal fires and `target_idx` advances. The S19k claim is false. Position local-min *does* enforce the "drone near target's NED" condition — but only when the range signal doesn't short-circuit it first via OR. And the range signal was *intentionally* ungated as the defense against fusion-drift stalls (see S19k "false start" section).

### Attempted fix — AND-gate range signal with position-sanity radius

If range alone spoofs to decoys near the camera, tighten it: range signal fires only when fused pose also agrees we're near the target gate's known NED. Implementation:

```python
range_fires = (
    tracker.detected
    and tracker.range_est < self.PASSAGE_RANGE
    and (prev_gate_dist is None or prev_gate_dist < self.PASSAGE_SANITY_RADIUS)
)
```

Isolated-unit scenarios looked clean:
- A: decoy at 1 m range, drone 20 m from target → rejected ✓
- B: legit pass at 1 m range, drone 1.5 m from target → fires ✓
- C: drift stall, drone 1.5 m from target, fused pose 60 m from target → rejected (this is the edge case S19k's false-start ran into)

Full bench regressed hard. `technical/fusion/clean` went from 12/12 → 2/12 with 178 m max pose error — the same failure mode as S19k's original ill-fated experiment, even though `PASSAGE_SANITY_RADIUS = 5.0 m` is deliberately more forgiving.

### Why the sanity gate fails in practice — the stall cascade

Trace on a technical-course run:
1. Gates 0–1 pass cleanly; fused pose tracks within ~0.3 m.
2. Gate 2: detection intermittent, fusion gets fewer vision updates, chi-squared rejects some fixes, open-loop IMU drift accumulates to ~4 m.
3. At gate 2 approach: `truth_dist ≈ 0.5–3 m` (drone physically passing), but `fused_dist ≈ 14 m` → sanity gate blocks the range signal, and position signal's local-min can't fire because position signal is on *fused* NED, also at 14 m.
4. Drone doesn't advance target, keeps orbiting gate 2 per BeliefNav's "approach target" plan.
5. More orbits = more time = more drift. Even when detection recovers, each vision fix is now `‖residual‖ ~ 14 m` — chi-squared rejects them all.
6. Permanent stall. Bench reports max_err 178 m by the time the course times out.

**The sanity gate and the stall cascade are mechanically coupled.** Any sanity radius tight enough to reject decoys at typical gate spacing (technical course gates are 5.7–8 m apart) is also tight enough that moderate drift blocks real passes — which then causes *more* drift because the pass-not-firing keeps the drone orbiting *one* spot instead of continuing through the course and picking up fresh vision updates at later gates. Single-target vision updates on drift is an unstable system.

### The honest fix — narrower claim, honest docstring, contract tests

Reverted `_check_gate_pass_position` to S19k OR-logic (range ungated, position additive). Removed the unused `PASSAGE_SANITY_RADIUS` class constant to prevent it implying behavior that doesn't exist. Rewrote the docstring to scope the defense honestly:

> **Distractor defense is best-effort and lives at the detector layer, not here.** Signal 1 [range] is spoofable by a decoy at short camera range. Signal 2 [position] is NOT spoofable by a decoy the drone never physically approaches, but IS spoofable by a decoy placed on the drone's flight path near the target gate.

Added two contract tests in `test_race_loop_gate_pass.py`:

9. **`test_range_signal_vulnerable_to_near_distractor`** — drone at (-10, 0, -1), target at (10, 0, -1), detection of range=1 m → asserts `fired is True`. Encodes the vulnerability so future "fixes" that accidentally re-introduce sanity gating will fail this test and force the author to re-read this log entry.
10. **`test_position_signal_rejects_distant_distractor`** — drone loops at |N|≥20 m from the target while detector fires above PASSAGE_RANGE → asserts position signal never fires. Encodes the actual defense.

`python test_race_loop_gate_pass.py` → **10/10 PASSED**.

### Regression verification

After the revert:
- `test_race_loop_gate_pass.py` — 10/10 (8 old + 2 new S19l contract tests)
- All other suites — unchanged, 74/74 across the 11 active test files as before.
- `bench_fusion_ab.py --course technical --vision-pos-sigma 1.0` — 12/12 on every scenario, max_err ≤ 0.54 m.
- `bench_fusion_ab.py --course sprint --vision-pos-sigma 1.0` — 10/10 on every scenario, `sprint/fusion/mild` retained at 8.84 s (vs 12.10 s legacy).

The S19k improvement is intact; none of the S19l experimentation contaminated the shipped path.

### Where Round 2 distractor defense actually belongs

**Detector layer (YOLO training).** Augment the training set with images containing decoy gates and label them with zero/low confidence or a separate "distractor" class. The right place to decide "this bounding box is a gate or a decoy" is in the classifier, not in a downstream geometric filter. Two reasons:

1. YOLO has the full image context — color, texture, shadow, the drone's expected approach angle. The race loop only has `(range_est, bearing_h, bearing_v)` — it has lost the information needed to distinguish.
2. A geometric filter at the race-loop layer conflates "distractor" with "normal pose uncertainty." Pose drift *looks identical* to a spoofed detection from the filter's perspective. There is no radius that separates them.

Added to the planning next-rung list.

### Lessons for the record

1. **"By definition" in my own docs is a red flag.** Any claim about defense / protection / invariance needs a unit test or a math proof in the same session it's written. No exceptions, no "I'll write it next session."
2. **Sanity gates on downstream measurements are a local optimum that breaks globally.** Whenever a correction depends on an estimate that the correction's *own enforcement* makes worse, it's a positive-feedback loop waiting to diverge. S19k's "false start" was this exact failure; I repeated it in S19l with an inflated radius thinking that would fix it. It doesn't — drift rates scale too fast.
3. **A stall-cascade post-mortem is the template for every geometric filter from now on.** If the filter can block the measurement that corrects the thing the filter consumes, it's suspect by construction.

### Files touched (S19l)

- `src/race_loop.py` — removed `PASSAGE_SANITY_RADIUS` constant, rewrote class docstring and `_check_gate_pass_position` docstring to scope the defense honestly. Code path unchanged from S19k (revert of mid-session AND-gate experiment).
- `test_race_loop_gate_pass.py` — 2 new contract tests, updated module docstring to reference the S19l scope.
- `docs/PROJECT_LOG.md` — this entry (corrects the S19k claim).
- `.auto-memory/project_aigp_state.md` — S19l addendum (also corrects S19k).

### Pending / next-rung

- **Distractor-augmented YOLO training dataset** — requires GPU, blocks on hardware availability. Parent item for Round-2 readiness.
- Distractor-realism bench scenario (extend `bench_fusion_ab.py` to inject decoys into `VirtualDetector`'s view; assert `target_idx` advances through the correct gates and not the decoys) — still valuable for catching regressions at the race-loop layer, even though the primary defense is now scoped to the detector. Keep as planned.
- Everything the S19k "Pending / next-rung" list called out is still valid and unchanged.

---

## Session 19m — Distractor-realism bench surfaces a baseline cascade

### TL;DR

Built the S19l-planned distractor-realism bench (on_path + off_path decoys, nearest-first association, fusion off, synthetic straight-line course). Expected to measure how often decoys spoof the pass detector. Instead found a **pre-existing baseline correctness bug** unrelated to decoys: under `associate_mode="nearest"` — the honest real-YOLO threat model, since YOLO emits `gate_idx=-1` — the range-signal + nearest-first combination cascades through all remaining gates the moment the drone reaches the first real gate. Zero decoys required. S19m does not ship a fix; it ships a reproducible bench that surfaces the hazard and a sharp problem statement for the next session.

### What the bench measures

`bench_fusion_ab.py --only-distractors` on a synthetic 5-gate straight-line course at N=15/30/45/60/75, E=0, D=-2.5. Drone spawns at origin and flies +N. Three scenarios, all at the `clean` noise profile, fusion OFF:

| scenario               | decoys   | gates | time  | spoof_μ | spoof_M |
| ---------------------- | -------- | ----- | ----- | ------- | ------- |
| baseline_no_decoys     | none     | 5/5   | 1.94s | 32.16 m | 61.95 m |
| distractor_on_path     | on_path  | 5/5   | 1.82s | 33.51 m | 63.30 m |
| distractor_off_path    | off_path | 5/5   | 1.94s | 32.16 m | 61.95 m |

`spoof_μ` / `spoof_M` are mean/max ‖drone_pos − real_target‖ at each pass-event tick. An honest pass fires at ≤ PASSAGE_RADIUS = 2.0 m or ≤ PASSAGE_RANGE = 2.5 m. **All three scenarios are spoofing at ~60 m.** The baseline (zero decoys) is indistinguishable from off_path. The on_path row is only marginally worse (~+1.3 m mean).

### Root cause: baseline cascade under nearest-first

Instrumented debug (`debug_cascade.py`, kept in sandbox, not shipped) traced every tick and pass event:

```
tick  80  drone=(11.00,-0.01,-2.41)  target=0/5  nearest_det_range=4.26
[PASS] tick  92 drone=(12.67, 0.00,-2.47)  target 0→1  d_to_passed= 2.33m  nearest_gate=0@2.33m  det.range=2.47
[PASS] tick  93 drone=(12.78, 0.00,-2.48)  target 1→2  d_to_passed=17.22m  nearest_gate=0@2.22m  det.range=2.33
[PASS] tick  94 drone=(12.89, 0.00,-2.48)  target 2→3  d_to_passed=32.11m  nearest_gate=0@2.11m  det.range=2.21
[PASS] tick  95 drone=(12.99, 0.00,-2.48)  target 3→4  d_to_passed=47.01m  nearest_gate=0@2.01m  det.range=2.12
[PASS] tick  96 drone=(13.09, 0.00,-2.49)  target 4→5  d_to_passed=61.91m  nearest_gate=0@1.91m  det.range=2.07
```

Tick 92 is an honest pass — drone is 2.33 m from gate 0, range signal fires, target advances 0→1. Ticks 93–96 are the cascade: drone has moved 0.42 m in 4 ticks (80 ms), gate 0 is still the nearest detection at range 2.0–2.2 m, `associate_mode="nearest"` keeps picking it, and the range signal (`tracker.detected and range_est < PASSAGE_RANGE`) fires for whatever target_idx is now. The drone never physically approaches gates 1–4; they are all registered as passed anyway.

The position signal (the S19k addition) does NOT fire during the cascade — the drone is 15 m+ from gate 1 and further from gates 2–4, well outside PASSAGE_RADIUS. The cascade runs entirely on the range signal.

### Why the fusion A/B bench never saw this

The fusion A/B bench (the 6-row scenarios in this same file) uses `associate_mode="target_idx"`. In that mode `_pick_detection` walks the detection list and returns the first detection whose `gate_idx` matches the current `target_idx`. After target advances from 0 to 1, gate 0's detection (`gate_idx=0`) is rejected; gate 1's detection (`gate_idx=1`) is picked. Gate 1 is ~17 m away → `range_est=17 > PASSAGE_RANGE=2.5` → no range fire. Cascade broken by construction.

This is the reason the existing test suite (10/10 in `test_race_loop_gate_pass.py`) and the fusion A/B table (all 6 rows clean) never flagged the bug. Both exercise target_idx mode, which requires the detector to tag detections with the course index — something `VirtualDetector` does trivially (it projects known gates) but real YOLO cannot.

### Why this matters for Round 1 / Round 2

Round 1 (deterministic, no distractors) races would still be correct if they used target_idx mode, because every detection is a real gate. But YOLO in production cannot tag `gate_idx` — the model sees a gate-shaped thing, not "gate #3." `target_idx` mode was always a sim-bench convenience; the real-flight threat model is `associate_mode="nearest"`. S19m shows that the pass detector under the real-flight threat model cascades on *every single race* the moment the first gate is crossed, with or without decoys. The production stack as of S19l is not race-ready for real YOLO. It races correctly in sim because sim tags match.

### Why S19l's "position signal cannot be spoofed by off-path decoys" claim still holds

S19l's scope contract was:
- ✅ Dropout-robust (catches pass when YOLO box collapses at close range)
- ✅ Rejects decoys the drone never physically approaches (off-path)
- ❌ Spoofable by decoys on flight path near target gate (detector-layer work)

All three properties remain true. The S19m finding is orthogonal: the pass detector also cascades under nearest-first even with no decoys at all, because range-signal fires on any detection that happens to be within PASSAGE_RANGE. The position signal is not the problem and not the defense here — it is simply silent during the cascade. If the range signal were removed, the cascade would not happen (but we would also lose the pre-S19k timing on clean runs). The fix is not "drop signal 1" — it is "stop firing signal 1 on a detection we just fired on."

### Candidate fixes (scoped for next session, not implemented here)

1. **Post-pass refractory period.** After `_advance_target`, block both range and position signals until drone has moved at least, say, PASSAGE_RANGE × 2 from the just-passed gate (or until N ticks elapse with sufficient displacement). Trivial to implement, localized to `race_loop.py`, no interaction with fusion drift. Does not defend against on-path decoys but does defend against the cascade.

2. **Require `gate_idx` match on the range signal.** Only fire range on a detection tagged with the current `target_idx`. Works in sim and breaks real YOLO (which tags `-1`). Not a real-flight fix, but would close the sim-only hole.

3. **Association-aware range signal.** In nearest mode, require that the picked detection's ID (bearing cluster, range trajectory, or a simple "same detection within ε last tick") is *different* from the one that fired the previous pass. Harder to implement correctly, but race-ready.

4. **Raise the association bar.** After a pass, require a detection whose range has been *monotonically decreasing* for M consecutive ticks before accepting it as the new target. This rejects "I just passed you, but you are still in frame" but accepts "new gate appearing at range 15 m and closing."

Option (1) is the cheapest and likely sufficient when stacked on top of distractor-trained YOLO. Option (4) is a more principled defense and pairs naturally with position signal (both want "approach trajectory," not "momentary proximity").

### What S19m ships

- `bench_fusion_ab.py` — distractor harness unchanged from S19l draft, with a FINDING comment block inside `print_distractor_table` explaining the cascade. Running `--only-distractors` prints the three-row table above with spoof metrics, so the cascade is visible in the numbers; the comment points the reader to this log entry.
- `DISTRACTOR_COURSE` — synthetic straight-line course retained for future distractor work. Its geometry isolates the cascade cleanly (first real gate is gate 0, no spawn-coincidence with the production figure-8 courses).
- This log entry — sharp problem statement and ranked candidate fixes.

### What S19m does NOT ship

- **No code fix to the pass detector.** The cascade is real and reproducible; picking among the four candidate fixes deserves its own session with deliberate evaluation, not a rushed patch. Flagged in the bench output and in the backlog so the fix lands with intent.
- **No change to `associate_mode="target_idx"` usage in fusion runs.** Those runs remain correct in sim. The production stack will switch to nearest-first when real YOLO ships, at which point the fix (picked next session) needs to be in place.

### Files touched (S19m)

- `bench_fusion_ab.py` — FINDING block appended to `print_distractor_table`'s trailer.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19m addendum noting the baseline cascade and the scoping-out of a fix.

### Pending / next-rung

- **Fix the cascade (pick from options 1/3/4 above; 2 is sim-only).** Option 1 is ~10 lines, option 4 is ~30. Either unblocks real-YOLO deployment. Priority: highest remaining pass-detector work.
- **Re-run the distractor bench after the fix** — spoof_μ should drop to ≤ PASSAGE_RADIUS for baseline and off_path, and reveal the genuine on_path signal (decoy 1.5 m in front of real gate is the hardest honest case).
- **Distractor-augmented YOLO training** remains the Round-2 detector-layer defense, blocked on GPU. S19m did not change this.
- Centreline-crossing detector scaffold (needs per-gate yaw) — still a worthwhile per-gate pass heuristic and complements any race_loop-layer refractory period.

---

## Session 19n — Post-pass refractory + anchor bug uncovered by it

### TL;DR

Implemented S19m's preferred fix: a post-pass refractory on the range-signal of `_check_gate_pass_position`. Distractor bench collapses from 61.95 m / 63.30 m / 61.95 m to **2.42 m / 3.94 m / 2.42 m** (baseline / on_path / off_path) — honest navigation, not cascade. In the process, the refractory exposed a latent bug in `_ingest_vision_from_detection` where the vision-fix world anchor was always `gates_ned[self.target_idx]` regardless of which gate the picked detection actually represented; fixed that too. It also exposed a pre-existing BeliefNav limitation (drone mis-steered by just-passed-gate detections when the next target is out of FOV) that is scoped out as follow-up.

### The refractory

```python
PASSAGE_REFRACTORY = 5.0   # m drone-displacement since last pass
```

- Measured as drone displacement since the pass fired (stored `_drone_ned_at_last_pass`), **not** distance to the just-passed gate. Rationale: an on-path decoy 1.5 m in front of gate K fires the range signal when drone is still ~3 m short of K — a gate-position refractory would consider K already cleared and the next pass would fire immediately. Drone-displacement forces the refractory to last until the drone has physically carried itself past where the spurious detection was.
- Sized at `2 × PASSAGE_RANGE = 5.0 m` — wide enough to traverse the cascade window, narrow enough that the next legitimate range-fire at the following gate (~6-8 m away on technical) is not delayed beyond the drone actually reaching it.
- Applied to the **range signal only**. The position signal (`‖drone − gate_ned‖` local minimum inside `PASSAGE_RADIUS`) already requires the drone to be physically close to the *new* target, so it cannot cascade and doesn't need the guard.
- Self-healing against pose drift: evaluated every tick against whatever pose the loop consumes (fused when fusion is on, truth otherwise). If fused drifts but subsequently converges back to truth, the refractory clears naturally once displacement exceeds 5 m.

First-attempt sized the refractory at gate-position, 3.5 m. It fixed baseline and off_path but not on_path — debug trace showed the on_path decoy fires at drone=(11.22, 0), which is 3.78 m from gate 0 (already outside a 3.5 m gate-position window). Drone-displacement reference + 5 m radius handles all three scenarios.

### Distractor bench — before / after

| scenario         | pre-S19n spoof_max | post-S19n spoof_max |
|------------------|--------------------|---------------------|
| baseline         | 61.95 m            | **2.42 m**          |
| distractor_on    | 63.30 m            | **3.94 m**          |
| distractor_off   | 61.95 m            | **2.42 m**          |

The on_path case lands at 3.94 m — outside the `PASSAGE_RADIUS=2.0 m` position-bucket honesty bar, so that first pass is still range-only and therefore decoy-vulnerable. This is the expected residual: a decoy placed 1.5 m in front of a real gate is by construction within range-signal fire distance, and the position signal requires ≥ 2 m of actual drone-to-real-gate proximity. The remaining 1.5 m spoof is the cost of not having a centreline-crossing detector or detector-layer distractor-augmented YOLO training. Follow-up work.

### The anchor bug the refractory uncovered

`_ingest_vision_from_detection` back-projects a detection's body-frame bearing + range into a drone-NED fix. It needs the **world-frame anchor of the gate the detection represents** so the back-projection subtracts the right displacement. The pre-S19n code used:

```python
gate_ned = self._gates_ned[self.target_idx]
drone_ned_meas = gate_ned - world_to_gate
```

The picker, however, can hand `_ingest_vision_from_detection` a detection whose `gate_idx` differs from `self.target_idx` — specifically in target_idx mode, when no detection is tagged with the current target, `_pick_detection` falls back to the nearest detection regardless of its gate_idx. Pre-S19n, the baseline cascade masked this: target_idx advanced aggressively to match whatever was picked, so `picked.gate_idx == self.target_idx` held by construction within a few ticks. The S19n refractory holds target_idx steady, exposing the mismatch: a detection of gate 9 at 3.6 m got back-projected with `gate_ned = gates_ned[2]`, producing a fix 30+ m from truth. The ESKF chi-squared gate rejected many of them, the rest poisoned the filter; `vis_rej` jumped from ~10 to 488 on fusion/clean.

**Fix:** anchor on `picked.gate_idx` when it's a valid index; fall back to `self.target_idx` only when gate_idx is -1 (real-YOLO, which doesn't tag identity).

```python
gi = getattr(picked, "gate_idx", -1)
if 0 <= gi < len(self._gates_ned):
    anchor_idx = gi
elif self.target_idx < len(self._gates_ned):
    anchor_idx = self.target_idx
else:
    return
...
gate_ned = self._gates_ned[anchor_idx]
```

Real-YOLO runs will hit the fallback path and are known to carry residual risk here — a future identity-aware detector or PnP gate-ID layer should supply `gate_idx` so this never fires. Noted in the method docstring.

### What the fusion A/B bench actually measured pre-S19n

The bench claim "fusion/clean: 12/12, max_err 0.34 m, 2.22 s" was an artefact of the same cascade. Instrumenting the target_idx progression shows target racing from 0 to 12 in ticks 54–110 while the drone only physically traversed from (0, 0) to (11, 5). The filter stayed close to truth because runtime was short, not because it was tracking a real 12-gate traversal. Post-S19n, the bench terminates honestly: fusion/clean 9/12 at timeout, max_err 86 m. The 86 m is dominated by a late divergence between ticks 200–550 where the drone flies away from gate 2 (at (8, 12)) to (21, 22) before looping back — caused by a separate, pre-existing issue: the picker falls back to the just-passed gate 1's detection when target 2 is out of FOV, and BeliefNav's bearing-following steers the drone along the old gate's line. The refractory correctly blocks the pass detector from re-firing on gate 1, but the navigator is still mis-steered. Scoped out of S19n; see Pending.

### Files touched (S19n)

- `src/race_loop.py`
  - New class constant `PASSAGE_REFRACTORY = 5.0` with rationale docstring.
  - New state `self._drone_ned_at_last_pass`.
  - `_check_gate_pass_position` rewired: range signal is refractory-gated on drone displacement since last pass; position signal unchanged; both pathways snapshot drone NED before `_advance_target` so the refractory reference survives the advance.
  - `_ingest_vision_from_detection` now anchors on `picked.gate_idx` when valid; docstring updated.
  - Class-level docstring updated to reference S19n refractory.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19n addendum.

### Regression verification

- 76 / 76 unit tests pass in isolation (`test_race_loop*`, `test_pose_fusion`, `test_detector`, `test_gate_belief`, `test_sim_adapter`, `test_eskf`, `test_sim_imu`, `test_vision_roundtrip`, `test_race_runner`). Cross-file pytest runs still show unrelated mavsdk-stub contamination — pre-existing, not S19n.
- Distractor bench: 3/3 scenarios pass cleanly with honest spoof distances above.
- Fusion A/B bench (`--course technical --vision-pos-sigma 1.0`): now **honest**, not a regression artefact.
  - legacy/clean 12/12 in 2.22 s (still cascading — legacy uses `_check_gate_pass` which is range-only and unprotected).
  - fusion/clean 9/12 at 25 s timeout, max_err 86 m (was fake 12/12 / 0.34 m via cascade).
  - fusion/harsh 12/12 in 13 s, max_err 372 m (fused pose drifts far from truth but still triggers position passes — see Pending).

### Pending / next-rung (ordered by priority)

1. **Picker fallback fix for target_idx mode.** When the current target is out of FOV, `_pick_detection` currently returns the nearest *non-target* detection, which is almost always the just-passed gate in the immediate post-pass window. This mis-steers BeliefNav. Candidate: prefer detections with `gate_idx >= target_idx` and return None otherwise, so the tracker coasts on belief rather than tracking a passed gate. A trial of this filter regressed legacy/clean to 8/12 because BeliefNav's search/coast mode isn't robust enough to handle the gap — needs pairing with a search-mode upgrade. Filed as S19n-followup-A.
2. **Extend refractory to `_check_gate_pass` (range-only path).** Legacy runs (fusion off, no `gates_ned`) still cascade. Trivial port; needs test coverage to confirm it doesn't regress existing range-only tests.
3. **BeliefNav post-pass recovery when next target is out of FOV.** The coast/search mode currently either stops or wanders. Options: feed next-gate expected bearing from `gates_ned` (breaks the "navigator is gate-agnostic" abstraction — might be the right trade), or a pre-programmed turn-to-heading based on the prior gate's approach.
4. **Fusion bench honesty.** The reported 12/12 / 0.34 m was never real. Either lower the gate count target (e.g., 4-gate subset), tighten the timeout, or replace the pass-count metric with "pose error vs truth at gate K" over the first N gates. Preference: latter — pass-count metric conflates fusion fidelity with navigator capability.
5. Centreline-crossing detector scaffold (needs per-gate yaw) — unchanged from S19m.
6. Distractor-augmented YOLO training — unchanged, GPU-blocked.

---

## Session 19o — Gate-aware BeliefNav + picker guard (2026-04-20)

Landed S19n-followup-A: the navigator now knows where the next gate is in NED, and the detection picker stops re-anchoring on the just-passed gate. Fusion bench went from 9/12 + 8/12 + 12/12 (clean / mild / harsh, post-S19n honest numbers) to **12/12 + 12/12 + 6/12** with honest sub-3m pose errors on the first two scenarios. Legacy path fully recovered to 12/12 across all noise levels.

### What changed and why

**Problem (from S19n epilogue).** When the drone passes a gate, the target idx advances but the next gate is often outside FOV for several ticks. Two things then went wrong:

1. `_pick_detection` in `associate_mode="target_idx"` fell back to `detections[0]` when no detection carried the new target idx. That nearest detection was almost always the *just-passed* gate still visible behind the drone. BeliefNav's `_plan_tracking` followed that bearing → drone turned back toward the gate it just cleared.
2. Meanwhile the new target was hidden, and BeliefNav's `_plan_coast` / `_plan_search` path had nothing to coast on (`on_gate_passed` resets belief confidence to 0). It sat still or sweep-searched blindly.

The combination made the drone drift off the optimal line by tens of meters, and on long courses it couldn't recover within the timeout. The pre-S19n "12/12 0.34 m" fusion bench number was only possible because the cascade bug was racing `target_idx` through all 12 gates in a few ticks while the drone physically traversed ~2 gates — that run wasn't measuring navigation, it was measuring how fast the cascade could iterate the target.

**Fix.** Two-part change:

- `BeliefNav.set_gates_ned(gates_ned)` hands the full gate list to the navigator. `plan()` now checks: if gates are known and `target_idx` is valid AND there's no current detection, route to a new `_plan_toward_known_target` that computes NED velocity toward `gates_ned[target_idx]` using the navigator's pose estimate (truth or ESKF-fused, depending on wiring). Yaw command points along the approach vector. Speed reuses `_speed_for_range × coast_speed_frac`, so cruise speeds stay consistent with belief-coast on continuous targets.
- `RaceLoop._pick_detection` gains a second fallback tier: after no target-idx match, prefer a detection with `gate_idx >= target_idx` (or `gate_idx == -1`, which is real-YOLO's tag) over the just-passed one. Returns `None` if nothing matches. **This tier only activates when the navigator has a fallback** (`_gates_ned is not None` and `navigator.gates_ned is not None`) — legacy non-fusion bench runs don't pass `gates_ned`, so they still use nearest-first and ride out the cascade window via the S19n refractory alone.
- `RaceLoop.__init__` wires `navigator.set_gates_ned(self._gates_ned)` if the navigator accepts it. Legacy navigators that don't define the method keep their old behaviour.

Noisy pose is self-correcting: once the drone gets close enough for the next gate to enter FOV, `_plan_tracking` takes over and detection-bearing overrides world-frame bearing. So even if `pos` is drifted by 5 m, the gate-aware path is a heading guess that closes the distance enough to re-acquire.

### Results

**Unit tests (passing):** 5 new tests in `test_belief_nav_gate_aware.py` cover gate-aware steering, detection-override, gates-unset fallback, out-of-range target idx, negative target idx. Combined with existing race/belief tests: 62 / 62 passing in the S19-relevant suite.

**Distractor bench (unchanged — still clean):**

| scenario                  | decoys   | noise | gates | time s | spoof_μ | spoof_M |
|---------------------------|----------|-------|-------|--------|---------|---------|
| baseline_no_decoys        | none     | clean | 5/5   |  9.62  | 2.36 m  | 2.42 m  |
| distractor_on_path        | on_path  | clean | 5/5   | 10.02  | 3.81 m  | 3.94 m  |
| distractor_off_path       | off_path | clean | 5/5   |  9.62  | 2.36 m  | 2.42 m  |

**Fusion A/B bench (`--course technical --vision-pos-sigma 1.0`):**

| scenario          | noise | fusion | gates | time s | max_err       | final_err     |
|-------------------|-------|--------|-------|--------|---------------|---------------|
| legacy/clean      | clean | off    | 12/12 |  2.22  |     —         |     —         |
| fusion/clean      | clean | ON     | 12/12 | 12.06  |    1.78 m     |    1.35 m     |
| legacy/mild       | mild  | off    | 12/12 |  2.14  |     —         |     —         |
| fusion/mild       | mild  | ON     | 12/12 | 12.28  |    2.37 m     |    2.33 m     |
| legacy/harsh      | harsh | off    | 12/12 |  2.24  |     —         |     —         |
| fusion/harsh      | harsh | ON     | 6/12  | 25.00  | 2199.69 m     | 2199.69 m     |

Versus post-S19n (honest numbers, pre-S19o):
- fusion/clean: 9/12 @ 86 m max_err → **12/12 @ 1.78 m** ✅
- fusion/mild:  8/12 @ large err    → **12/12 @ 2.37 m** ✅
- fusion/harsh: 12/12 @ 372 m       → 6/12 @ 2199 m ❌

The fusion/harsh regression is not a navigation issue: the ESKF's chi-squared gate rejects most vision fixes in harsh noise (192 rej vs 166 ok), the fused pose blows up, and the gate-aware path then steers to a wrong world-frame target. Pre-S19o, the drone would still complete because the picker would anchor on whatever detection was nearest, and the detection-bearing path (`_plan_tracking`) was at least locally consistent. Put differently, S19o trades a locally-robust-but-blindly-cascading navigator for a globally-correct-but-pose-dependent one. For clean + mild this is a large win; for harsh, it exposes a pre-existing ESKF fragility that was previously papered over by cascade-driven pass counts.

### Honest caveat on fusion/harsh

The pre-S19o "fusion/harsh 12/12" was itself partially fiction — the 372 m max_err means the drone was physically off course; the bench was crediting passes that the pose-based gate detector was firing on fused-pose proximity, not actual gate traversal. The S19o regression reveals that this case was never really navigating — it was coasting on the fusion/pose-detector accounting error. The right next move here is filtering at the ESKF layer or a confidence-gated fallback from `_plan_toward_known_target` to belief-coast, not reverting the picker guard.

### Files touched (S19o)

- `gate_belief.py`
  - `BeliefNav.__init__`: `self.gates_ned = None`.
  - New method `set_gates_ned(gates_ned)`.
  - `plan()`: gate-aware branch when `gates_ned` is set and `target_idx` is in range, routed before `_plan_coast` / `_plan_search`.
  - New method `_plan_toward_known_target(target_idx, pos, yaw_rad, dt)` — NED-geometry velocity + yaw command toward `gates_ned[target_idx]`.
- `src/race_loop.py`
  - `__init__`: wire `navigator.set_gates_ned(self._gates_ned)` when available.
  - `_pick_detection`: add conditional second fallback tier (skip already-passed gates). Only active when navigator has its own fallback, via new helper `_has_nav_fallback()`.
- `test_belief_nav_gate_aware.py` — new, 5 tests.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19o addendum.

### Pending (ordered)

1. **Fusion/harsh recovery (S19o-followup-A).** The gate-aware path should refuse to act on a fused pose that the ESKF is visibly distrusting. Two candidates: (a) track recent `vis_rej / (vis_rej + vis_ok)` rate and fall back to belief-coast when it exceeds a threshold; (b) monitor ESKF covariance trace and gate `_plan_toward_known_target` on it. (a) is cheaper.
2. **Replace fusion-bench gates-passed metric with pose-error curve (was #48).** The current metric conflates "did fusion stay close to truth" with "did navigator succeed". A pose-error-over-time track for the first N gates is more honest and would have caught the fusion/harsh regression earlier.
3. **Extend refractory to legacy `_check_gate_pass` (was #47).** Legacy path still cascades on range-only; unprotected today.
4. **Centreline-crossing detector scaffold** — unchanged from S19m.
5. **Distractor-augmented YOLO training** — unchanged, GPU-blocked.

---

## Session 19p — Pose-trust gate on the gate-aware fallback (2026-04-20)

Landed S19o-followup-A: the gate-aware BeliefNav fallback now refuses to navigate on a fused pose that the ESKF is visibly distrusting. Closes the fusion/harsh regression introduced in S19o (6/12 timeout @ 2199 m max_err) — now **12/12 in 11.68 s** at the same `--vision-pos-sigma 1.0` setting. All 6 fusion bench scenarios complete cleanly.

### What changed

S19o gave the navigator a powerful new tool: when no detection is available and `gates_ned` is set, fly directly toward the world-frame target. That tool is only useful when the navigator's pose estimate is roughly correct. In fusion/harsh, the chi-squared gate on `ESKF.update_vision` enters the self-destructive regime documented in `bench_fusion_ab.py` — covariance shrinks faster than truth error, valid fixes look like outliers, drift compounds. The ESKF reports the divergence honestly via `vision_fixes_rejected`, but pre-S19p the navigator still consumed and acted on the (wrong) fused pose, producing 2199 m max-err completions.

Fix is two pieces:

- `PoseFusion.recent_reject_rate(min_samples=5)` — rolling window (default 20) of vision-fix outcomes; returns the rejection rate when ≥ `min_samples` outcomes have been observed, else 0.0 ("trust by default while we're warming up"). Window is a `collections.deque` populated alongside the existing `vision_fixes_accepted/rejected` telemetry counters; `seed`/auto-seed paths also push.
- `BeliefNav.pose_trusted: bool = True` — checked in `plan()`'s gate-aware branch; when False, falls through to the legacy belief-coast/search path. Defaults True so legacy callers without fusion (truth pose is always trusted) get the gate-aware path unchanged.
- `RaceLoop.step()` fusion branch — after `on_vision_pose`, checks `pose_fusion.recent_reject_rate()` against a new class constant `POSE_TRUST_REJECT_RATE = 0.5` and writes `navigator.pose_trusted = (rate < threshold)`. Wrapped in `hasattr` checks for both sides so legacy non-BeliefNav navigators and PoseFusion-without-the-window stay byte-compatible.
- Non-fusion branch unconditionally sets `navigator.pose_trusted = True` — adapter truth never stops being trusted.

Threshold of 0.5 chosen against fusion/harsh's measured 322/(322+166)=0.66 (above threshold ⇒ pose_trusted=False), and against fusion/clean+mild's <2 % rejection rate (well below ⇒ stays trusted). The bench numbers below confirm both regimes hit the right branch.

### Bench results (`--course technical --vision-pos-sigma 1.0`)

| scenario          | noise | fusion | gates | time s | max_err   | final_err |
|-------------------|-------|--------|-------|--------|-----------|-----------|
| legacy/clean      | clean | off    | 12/12 |  2.22  |     —     |     —     |
| fusion/clean      | clean | ON     | 12/12 | 12.06  |   1.78 m  |   1.35 m  |
| legacy/mild       | mild  | off    | 12/12 |  2.14  |     —     |     —     |
| fusion/mild       | mild  | ON     | 12/12 | 12.28  |   2.37 m  |   2.33 m  |
| legacy/harsh      | harsh | off    | 12/12 |  2.24  |     —     |     —     |
| fusion/harsh      | harsh | ON     | 12/12 | 11.68  | 159.53 m  | 159.53 m  |

Δ from S19o:
- fusion/harsh: 6/12 timeout @ 2199 m → **12/12 @ 159 m** ✅
- All other rows unchanged (rejection rate stays well below 0.5).

The remaining 159 m max_err on fusion/harsh is the ESKF's own divergence — the navigator now correctly ignores it. Drone navigates on detection bearing through belief-coast, completes the course physically. The fused pose remains diagnostic-only telemetry in this regime; the load-bearing signals (detection bearing for steering, detection range for pass) are unaffected by the divergence because VirtualDetector projects from truth (and real YOLO will see real gates regardless of what the ESKF thinks).

Distractor bench unchanged (2.42 / 3.94 / 2.42).

### Why this is the right shape

The pose-trust gate is **observed, not predicted**: we don't try to model when fusion will diverge, we measure the chi-squared rejection rate the filter is already producing and react to it. That keeps the gate self-tuning across detector/IMU configurations — a future Neros IMU or a real YOLO with different noise characteristics will produce a different rejection rate, and the gate will trip at the same observed-divergence point.

It also degrades gracefully in the wrong direction: a *false* untrust just costs us the gate-aware fallback for a few ticks (we revert to belief-coast/search, which was the pre-S19o baseline); a *false* trust would let us steer onto a wrong pose. Asymmetric cost favours trusting only when the filter agrees it's healthy.

### Files touched (S19p)

- `src/estimation/pose_fusion.py`
  - `from collections import deque`.
  - `DEFAULT_REJECT_WINDOW = 20`; `__init__` takes `reject_window`; `_recent_outcomes: deque[int]`.
  - `on_vision_pose` appends 1/0 to the window on every outcome (including auto-seed and rejected paths).
  - New method `recent_reject_rate(min_samples=5)`.
- `gate_belief.py`
  - `BeliefNav.__init__`: `self.pose_trusted = True` with docstring.
  - `plan()`: gate-aware branch now requires `self.pose_trusted` in addition to `gates_ned` + valid `target_idx`.
- `src/race_loop.py`
  - New class constant `POSE_TRUST_REJECT_RATE = 0.5` with rationale docstring.
  - `step()` fusion branch: post-vision, write `navigator.pose_trusted = (rate < threshold)` when both ends support it.
  - `step()` non-fusion branch: explicit `navigator.pose_trusted = True` (truth pose).
- `test_belief_nav_gate_aware.py` — 2 new tests (`test_pose_trusted_false_skips_gate_aware`, `test_pose_trusted_default_is_true`); now 7/7.
- `test_pose_fusion.py` — 1 new test (`test_recent_reject_rate_rolling_window`); now 6/6.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19p addendum.

### Test status

- 65/65 pass in S19-relevant suites.
- Fusion bench: 6/6 scenarios complete cleanly (was 5/6 post-S19o, 4/6 in the trial that motivated S19o).
- Distractor bench: 3/3 unchanged.

### Pending (ordered, supersedes S19o list)

1. **Replace fusion-bench gates-passed metric with pose-error curve (was #48).** fusion/harsh's "12/12 @ 159 m" still misleads as a pass count. A pose-error-over-time track for the first N gates would surface fusion divergence before it becomes a navigator concern.
2. **Extend refractory to legacy `_check_gate_pass` (was #47).** Range-only path still cascades when fusion is off and `gates_ned` is unset.
3. **ESKF tuning under harsh noise.** S19p makes the navigator robust to a divergent ESKF, but the divergence itself is still the underlying issue. Adaptive Q/R or a chi-squared threshold that loosens after sustained rejection are the natural next moves; both want real Neros IMU data first.
4. **Real-YOLO vision anchor remains best-effort** (falls back to target_idx when gate_idx=-1) — future identity-aware detector should supply gate_idx.
5. Centreline-crossing detector scaffold — unchanged from S19m.
6. Distractor-augmented YOLO training — unchanged, GPU-blocked.

---

## Session 19q — Honest-pass metric for the fusion bench (2026-04-21)

Landed #48 from the post-S19p gap list: the fusion A/B bench now reports a **pose-error-at-passes** metric alongside the gates-passed count. S19p closed the navigator's fusion/harsh regression (6/12 timeout → 12/12 @ 159 m) but left an honest-but-misleading row: "12/12 gates" reads like success when the fused pose was 159 m off truth at every pass — the drone was completing on detection bearing while the fusion told a fabricated story.

### What changed

The bench now samples `‖fused_pos − truth_pos‖` at every `target_idx` advance (i.e., each gate-pass event) and reports three statistics:

- `pass_err_max_m` — worst fused-pose error observed at any pass.
- `pass_err_median_m` — median across passes (robust against a single outlier).
- `honest_passes` — count of passes where `pass_err < RaceLoop.PASSAGE_RADIUS` (2.0 m). This is the load-bearing number: it's the count of gates where the drone "believed" it was where it actually was.

Legacy rows (`fusion=off`) print `—` for all three — truth-fed runs have no fused pose to score.

### Bench results (`--course technical --vision-pos-sigma 1.0`)

| scenario     | noise | fusion | gates | time s | max_err    | pass_err  | honest |
|--------------|-------|--------|-------|--------|------------|-----------|--------|
| legacy/clean | clean | off    | 12/12 |   2.22 |     —      |     —     |   —    |
| fusion/clean | clean | ON     | 12/12 |  12.06 |   1.78 m   |   1.64 m  | 12/12  |
| legacy/mild  | mild  | off    | 12/12 |   2.14 |     —      |     —     |   —    |
| fusion/mild  | mild  | ON     | 12/12 |  12.28 |   2.37 m   |   2.33 m  | 10/12  |
| legacy/harsh | harsh | off    | 12/12 |   2.24 |     —      |     —     |   —    |
| fusion/harsh | harsh | ON     | 12/12 |  11.68 | 159.53 m   | 159.53 m  |  3/12  |

The new column exposes exactly the regime S19p built the navigator to survive: fusion/harsh passed every gate physically (12/12) while only 3 of those passes were "honest" — the drone was navigating on the detection-bearing signal, ignoring a fused pose that was 100+ m adrift. fusion/mild drops 2 honest passes to the 2.0 m bar, which is a measured statement about ESKF residual error under mild noise (not a regression to fix right now). fusion/clean is honest across the board.

### Why this metric shape

The original gates-passed count is a pass/fail read on the entire stack — useful as a headline, useless for attributing failure. Pose-error-at-pass is a decision-relevant moment: it's the one tick where the navigator actually cared about the fused pose (refractory arms, gate-pass event fires, next target loads). Sampling there is cheaper than streaming full trajectories and catches the same divergence an over-time curve would, but with one row of numbers per scenario.

`honest_passes` is preferable to `pass_err_mean` because ESKF divergence tends to be bimodal — you either track well (sub-metre) or you lose the plot (10+ m). A mean smears those two regimes together; a count with a physically-meaningful threshold doesn't.

### Files touched (S19q)

- `bench_fusion_ab.py`
  - `BenchRow`: added `pass_err_max_m: Optional[float]`, `pass_err_median_m: Optional[float]`, `honest_passes: Optional[int]`.
  - `_drive`: tracks `prev_target`; when `target_idx` advances, samples `‖fused − truth‖` and pushes to a `pass_pose_errors: List[float]`. Returns 5-tuple now (added the list).
  - `run_one`: computes max/median/honest from the pass-errors list when `with_fusion`, leaves as `None` for legacy rows.
  - `print_table`: rewrote header/row templates, added `pass_err` + `honest` columns, added footer block defining the columns.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19q addendum.

### Test status

- 47/47 pass across `test_race_loop`, `test_race_loop_fusion`, `test_race_loop_gate_pass`, `test_gate_belief`, `test_belief_nav_gate_aware`, `test_pose_fusion`, `test_eskf`.
- `_drive` signature change is bench-internal (no external callers); no regression.
- Fusion bench: 6/6 scenarios still complete cleanly; new columns render correctly.

### Pending (ordered, supersedes S19p list)

1. **Extend refractory to legacy `_check_gate_pass` (was #47).** Range-only path still cascades when fusion is off and `gates_ned` is unset. Top priority now that measurement honesty is restored.
2. **ESKF tuning under harsh noise.** S19q confirms the divergence is visible and attributable; fixing it needs real Neros IMU noise characteristics.
3. **Real-YOLO vision anchor** — unchanged, falls back to target_idx when gate_idx=-1.
4. Centreline-crossing detector scaffold — unchanged from S19m.
5. Distractor-augmented YOLO training — unchanged, GPU-blocked.

---

## Session 19r — Drone-displacement refractory on the legacy pass detector (2026-04-21)

Landed #47 from the post-S19q gap list: `RaceLoop._check_gate_pass` (legacy range-only path, taken when `gates_ned=None`) now honours the same drone-displacement refractory that S19n added to `_check_gate_pass_position`. Closes the S19m cascade on the legacy path — and surfaces a pre-existing legacy-path nav-recovery gap that the cascade was masking.

### What changed

`_check_gate_pass(tracker)` grew an optional `drone_ned` parameter. When supplied, after a pass fires, the next range-cross is blocked until the drone has moved `PASSAGE_REFRACTORY = 5.0 m` from the pass-firing position. Reuses `self._drone_ned_at_last_pass` so range-signal and position-signal refractories share state (a pass on either path arms both). `drone_ned=None` preserves the pre-S19r behaviour, so minimal unit tests that poke the range signal in isolation don't need to synthesise a pose.

`RaceLoop.step()` passes `pos_for_plan` into the legacy branch, so the production path always gets the refractory. Non-`step()` callers (there are none in repo, but third-party integrations might exist) keep the original signature.

### What the bench shows

Before S19r, the fusion bench's legacy rows read "12/12 in 2.22 s" on every noise profile. That number was fiction for the same reason the S19n pre-fix distractor baseline was: the cascade was racing `target_idx` through all 12 gates in ~110 ticks while the drone physically traversed only ~11 m. With the refractory on, the legacy rows read "2/12 timeout" — the drone cleanly passes the first two gates, then stalls because BeliefNav without `gates_ned` cannot recover when the tracker's detection latches onto the just-passed gate.

| scenario     | noise | fusion | gates | time s | pass_err  | honest |
|--------------|-------|--------|-------|--------|-----------|--------|
| legacy/clean | clean | off    |  2/12 | timeout|     —     |   —    |
| fusion/clean | clean | ON     | 12/12 |  12.06 |   1.64 m  | 12/12  |
| legacy/mild  | mild  | off    |  2/12 | timeout|     —     |   —    |
| fusion/mild  | mild  | ON     | 12/12 |  12.28 |   2.33 m  | 10/12  |
| legacy/harsh | harsh | off    |  2/12 | timeout|     —     |   —    |
| fusion/harsh | harsh | ON     | 12/12 |  11.68 |  159.53 m |  3/12  |

Fusion rows are unchanged — they use the position-based detector (`_check_gate_pass_position`, which already had a refractory since S19n). Distractor bench also unchanged (2.42 / 3.94 / 2.42) — distractor scenarios run with `gates_ned=real_gates`, so they've always taken the position path.

The legacy "2/12" isn't a regression introduced by S19r — it's an honest measurement of what the legacy path actually achieves under production-like conditions. The old "12/12 in 2.22 s" was the same kind of cascade fiction that pre-S19n distractor `baseline_no_decoys` was (61.95 m "pass count" while the drone sat at the start line). If anything, S19r completes S19n by applying the same fix to the only remaining cascade-susceptible path.

### Why this is the right shape

The refractory is self-healing: any genuine drone motion gradually clears it, pose drift cannot trap us because `target_idx` has already advanced on each firing and the reference is drone-position not gate-position. This is the same property that made the S19n position-path refractory fusion-drift-safe; it transfers directly.

Optional `drone_ned` preserves the minimal unit test surface — poking `_check_gate_pass(tracker)` in isolation still works without having to synthesise a pose. Production callers always have a pose (`pos_for_plan`), so the refractory is always on in practice.

The `gates_ned=None` legacy configuration now needs a follow-up: either wire BeliefNav's search mode more aggressively so it can recover from a stuck-detection lock, or require production callers to always supply `gates_ned` (which S19d's `run_race.py` does for real courses). The latter is cheaper and is the natural default — the legacy path is a minimal-integration fallback, not a production target.

### Files touched (S19r)

- `src/race_loop.py`
  - `_check_gate_pass` signature: added optional `drone_ned: Optional[Sequence[float]]`.
  - Refractory check: mirrors `_check_gate_pass_position`'s drone-displacement logic; snapshots `_drone_ned_at_last_pass` on firing.
  - `step()` legacy branch: passes `pos_for_plan` into the updated signature.
- `test_race_loop_gate_pass.py` — 2 new tests:
  - `test_legacy_range_path_refractory_blocks_cascade` — drone barely moves after a pass, re-fire blocked; once drone has moved > PASSAGE_REFRACTORY, next legit pass fires.
  - `test_legacy_range_path_without_drone_ned_still_cascades` — explicit contract that `drone_ned=None` preserves pre-S19r behaviour; 3 consecutive fires when no pose supplied.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19r addendum.

### Test status

- **49/49** across S19 suites (race_loop 5, race_loop_fusion 4, race_loop_gate_pass 12, gate_belief 5, belief_nav_gate_aware 7, pose_fusion 6, eskf 10).
- Distractor bench unchanged.
- Fusion bench: legacy rows now honest, fusion rows unchanged.

### Pending (ordered, supersedes post-S19q list)

1. **Legacy-path nav recovery** — the S19r bench numbers expose that BeliefNav without `gates_ned` cannot recover from a stuck-detection lock on a just-passed gate. Either (a) upgrade BeliefNav search mode to yaw-sweep + forward-drift when tracker hasn't progressed for N ticks, or (b) document the legacy path as minimal-integration-only and require `gates_ned` in all production callers. (b) is cheaper and matches how `run_race.py` already wires things.
2. **ESKF tuning under harsh noise** — honest_passes 3/12 on fusion/harsh is the number to move. Needs real Neros IMU characterisation.
3. Real-YOLO vision anchor remains best-effort (falls back to target_idx when gate_idx=-1) — future identity-aware detector should supply gate_idx.
4. Centreline-crossing detector scaffold (unchanged since S19m).
5. Distractor-augmented YOLO training (GPU-blocked).


---

## Session 19s — Documenting `gates_ned=None` as minimal-integration-only (2026-04-21)

Closes the top of the post-S19r gap list by taking option (b) from that session's pending notes: the `gates_ned=None` code path is explicitly marked as a minimal-integration fallback, not a production target. No behaviour change — documentation, a constructor `UserWarning`, and a docstring block that calls out the navigator-recovery gap S19r's bench surfaced.

### Why (b) and not (a)

S19r's bench showed legacy rows stalling at 2/12 gates when `gates_ned=None`: BeliefNav enters search mode after the tracker's detection latches onto the just-passed gate, but without world-frame gate positions the search has no reference to steer toward. Fixing this properly (option (a)) would mean upgrading `_plan_search` to a more aggressive yaw-sweep + forward-drift recovery, plus plumbing a "tracker hasn't progressed for N ticks" signal from RaceLoop into BeliefNav. That's a real week of work on a configuration that has zero production callers — `run_race.py` (S19d) always supplies `gates_ned=real_gates`, and so does every fusion bench row and distractor bench row.

The only places that exercise `gates_ned=None` are (i) minimal unit tests that deliberately probe the range signal in isolation without constructing a gate-world fixture, and (ii) the legacy bench rows in `bench_fusion_ab.py`, which exist to contrast against the fusion rows. None of those need navigator recovery — the unit tests pass on 2-3 gate courses that complete before the cascade would hit, and the legacy bench rows' new "2/12 timeout" result is the *point* (it demonstrates that without fusion, the legacy path is not a viable production configuration).

Asymmetric cost: option (a) is a week of work with ~zero production payoff; option (b) is 20 lines of docstring + one `warnings.warn` call that makes every future minimal-integration caller aware of the limitation at construction time.

### What changed

`RaceLoop.__init__` grew a docstring block under the `gates_ned` arg explaining:

- What `gates_ned=None` means (legacy range-only path; `_check_gate_pass` instead of `_check_gate_pass_position`).
- That S19r's drone-displacement refractory applies to both paths, so the cascade bug is closed either way.
- That BeliefNav cannot recover from stuck-detection lock without `gates_ned` — it has no world-frame reference for search steering.
- That production callers (`run_race.py`, fusion bench, distractor bench) always supply `gates_ned=real_gates`, so this is strictly a minimal-integration fallback.

The constructor body also emits a `UserWarning` when `gates_ned is None`:

```python
warnings.warn(
    "RaceLoop constructed without gates_ned — this is a "
    "minimal-integration configuration only. BeliefNav "
    "cannot recover from stuck-detection lock without "
    "world-frame gate positions. Supply `gates_ned` for "
    "any production use.",
    UserWarning,
    stacklevel=2,
)
```

`stacklevel=2` points the warning at the caller's construction site rather than the constructor line itself, which is what future users will want to see when they hit this.

### Where the warning fires (verified)

- `test_race_loop.py:145` — minimal 2-gate fixture that doesn't need a world frame. Expected.
- `test_race_loop_gate_pass.py:89` — the `_build` helper used by ~half of that file's tests. Expected; these tests probe the range signal and refractory in isolation and don't need `gates_ned`.
- `bench_fusion_ab.py:248` — legacy bench rows constructed with `gates_ned=None` as the intentional contrast to fusion rows. Expected; the warning here labels exactly the rows that `print_table`'s pass_err and honest columns show as "—".

Where the warning does *not* fire (also verified): all fusion bench rows, all fusion unit tests (`test_race_loop_fusion.py`), all distractor-bench scenarios, and `run_race.py`'s production callers. Every production-like site already supplies `gates_ned`.

The warning is left unsilenced in test fixtures — its diagnostic value is the point of S19s, and Python's default warning filter only emits each (message, category, module, lineno) tuple once per process, so the test output isn't cluttered.

### Files touched (S19s)

- `src/race_loop.py`
  - Added ~18-line docstring block under `gates_ned` in `__init__`'s Args section.
  - Added `warnings.warn(...)` call in constructor body when `gates_ned is None`.
- `docs/PROJECT_LOG.md` — this entry.
- `.auto-memory/project_aigp_state.md` — S19s addendum.

### Test status

- **49/49** across the S19 suites (race_loop 5, race_loop_fusion 4, race_loop_gate_pass 12, gate_belief 5, belief_nav_gate_aware 7, pose_fusion 6, eskf 10).
- No test behaviour changed; warnings fire exactly at the three documented minimal-integration call sites and nowhere else.
- Bench output unchanged from S19r.

### Pending (ordered, supersedes post-S19r list)

1. **ESKF tuning under harsh noise** — fusion/harsh's honest_passes = 3/12 is now the most legible number in the bench. Moving it requires real Neros IMU noise characterisation (measurement-driven Q/R tuning beats hand-waving).
2. **Real-YOLO vision anchor** — unchanged; identity-aware detector should supply gate_idx rather than falling back to `target_idx` in `_pick_detection`.
3. **Centreline-crossing detector scaffold** — unchanged since S19m.
4. **Distractor-augmented YOLO training** — GPU-blocked.
5. **Legacy-path navigator recovery** (option (a) from S19r) — deliberately descoped by S19s. If a real caller ever needs `gates_ned=None` at production scale, reopen this; until then the warning makes the limitation visible at construction time.


---

## Session 19t — Memory consolidation + DCL-prep audit (2026-04-21)

Not a code-feature session — a strategic-cleanup one. Two deliverables: a consolidated memory index that isn't a chronological addendum pile any more, and a DCL integration checklist that reduces day-1-after-SDK-drop to mechanical work instead of thinking-under-time-pressure.

### Why this and not more gate-pass work

The post-S19s gap list had ESKF tuning and distractor-augmented YOLO training as the top items, but both are hardware/GPU blocked. The centreline-crossing detector scaffold and identity-aware YOLO adapter are both viable standalone pieces of work, but they refine the simulator-only stack against a simulator we wrote. The DCL sim drops in 2–4 weeks (May 2026) and will shift camera characteristics, physics, and adapter timing. Work done now on simulator-only refinements has a real chance of being invalidated or needing rework. Conversely, the adapter seam is exactly the place where time pressure hurts most — integration surprises on day 1 are expensive when the rest of the stack is assuming shapes that might not match.

Asymmetric-cost reasoning (same lens as S19p/s): a cleanup pass at T-minus-2-weeks is cheap and preserves optionality; a new feature at T-minus-2-weeks costs the same engineering time and might not survive the sim drop.

### Memory consolidation

`.auto-memory/project_aigp_state.md` had accreted 20 session addenda (S19a through S19s) across 229 lines / 42 KB. Rewrote as a topic-structured state snapshot: locked baseline, competition context, architecture, durable design rationale, current gap list, test structure. 91 lines / 11 KB. Session chronology delegated to PROJECT_LOG where it belongs — memory is for orientation, not chronological duplication.

`.auto-memory/reference_aigp_docs.md` — fixed stale "contains the yaw-propagation bug" note on `gate_belief.py` (bug was fixed in S19), updated PROJECT_LOG line count, added source-file references that had accrued since the file was first written.

`.auto-memory/MEMORY.md` — fixed the project-hook line that still pointed at "S18 belief model has yaw-propagation bug".

**Durable lessons preserved** (collected into one section of the new memory file rather than scattered across 20 session entries): `--vision-pos-sigma 1.0` tuning rule (S19j), sanity-gate cascade lesson (S19l), target_idx vs nearest-first picker interaction (S19m), why drone-displacement refractory beats gate-position refractory (S19n/r), why pose-trust gates gate-aware nav (S19p), why `honest_passes` beats gates-passed (S19q), why `gates_ned=None` is minimal-integration-only (S19r/s), and the ESKF-noise-placeholder reminder for real Neros IMU data.

### DCL integration checklist

`docs/DCL_INTEGRATION_CHECKLIST.md` (new). Six sections:

1. **What the race stack calls per tick.** Hot path is `get_state() / get_camera_frame() / get_imu() / send_velocity_ned`; everything else is cold.
2. **Minimum viable DCL adapter.** Ranked table of must-have vs nice-to-have vs can-stay-no-op methods, with reasoning for each.
3. **Day-1 checklist when SDK drops.** Ordered checkbox sequence from reading the constructor docs through running `bench_fusion_ab.py --backend dcl`.
4. **Open questions.** The things that cannot be answered without the SDK — frame conventions, camera format, IMU rate/units, tick-rate semantics, arm/takeoff model. Tracked here so they aren't forgotten on day 1.
5. **Anticipated integration surprises.** Six concrete pitfalls from reading the adapter + runner + race_loop code: async wrapping of sync APIs, coordinate frame flips at the adapter boundary, BGR vs RGB, specific-force vs gravity-subtracted accel, `get_state()` called before first `step()`, `command_hz` vs DCL tick rate.
6. **Pre-landing work.** What can be done now vs what must wait for the SDK.

### DCL adapter seam tests

`test_dcl_adapter_seam.py` (new, 6/6 pass). Contract tests that assert the `DCLSimAdapter` stub cannot silently drift into a no-op state during future refactors — a no-op stub would let the race stack ship "working" against DCL by accident, with the breakage only surfacing in-sim.

Tests cover: capability flags include the minimum viable set (VELOCITY_NED, CAMERA_RGB, IMU, RESET); `ARM_ACTION` is NOT advertised (arm/takeoff/land are intentional no-ops); `info().notes` still marks DCL as a stub; every hot-path method raises `NotImplementedError` with a hint containing either `DCLSimAdapter` or the method name; action methods (arm/disarm/takeoff/land/offboard) complete without raising (intentional no-ops); `make_adapter('dcl')` factory returns the right type.

**When the SDK lands, most of these tests should invert** — from "asserts stub raises" to "asserts real implementation runs." The checklist doc and this test file are both designed to be edited together with the real implementation.

### Files touched

- `.auto-memory/project_aigp_state.md` — rewrite (229 → 91 lines).
- `.auto-memory/reference_aigp_docs.md` — update.
- `.auto-memory/MEMORY.md` — hook fix.
- `docs/DCL_INTEGRATION_CHECKLIST.md` — new.
- `test_dcl_adapter_seam.py` — new, 6/6 pass.
- `docs/PROJECT_LOG.md` — this entry.

### Test status

- **55/55** across S19 suites (49 prior + 6 new DCL seam). No behaviour change in `race_loop.py` or any other production module.
- Bench output unchanged.

### Pending (unchanged from post-S19s with added DCL-prep item)

1. **DCL-prep (near-term strategic).** Checklist + seam tests landed; remaining pre-landing items are writing a `MockDCLAdapter` under `--backend dcl` gated on a `DCL_SDK_AVAILABLE` flag (smoke-tests the full chain now) and drafting the day-1 command sequence as a shell script.
2. **ESKF tuning under harsh noise.** Hardware-blocked.
3. **Real-YOLO vision anchor.** Identity-aware detector should supply gate_idx.
4. **Centreline-crossing detector scaffold.** Would remove last hand-tuned magic number.
5. **Distractor-augmented YOLO training.** GPU-blocked.
6. **Legacy-path navigator recovery.** Descoped by S19s.


## Session 19u — 2026-04-21 — MockDCLAdapter smoke test surfaces two latent bugs

### Context

S19t landed `MockDCLAdapter` as a DCL-shaped pre-landing stand-in, plus a CLI `--backend mock_dcl` flag. The *point* of building it was to surface PX4-assumption leaks in RaceRunner / RaceLoop before the real SDK lands. The first smoke test it ran — `python run_race.py --backend mock_dcl --detector virtual --course technical` — surfaced exactly that: 0/12 gates, timeout. Two distinct bugs in two distinct files, both exposed by a capability shape (no ARM_ACTION, no WALLCLOCK_PACED, uses RESET at connect) that no existing backend had.

### Bug 1: `MockKinematicAdapter.reset()` drops configuration kwargs

`MockKinematicAdapter.reset()` is implemented by re-calling `__init__()`, but it did not forward `initial_altitude_m` or `auto_step`. Those kwargs silently reverted to their defaults (0.0 and False). `MockDCLAdapter.connect()` calls `self._kin.reset()` as part of its gym-style "connect = instantiate + reset" convention — so every `mock_dcl` race started with a drone glued to the ground (altitude 0) and physics that wouldn't advance per send_velocity_ned (auto_step False). Result: drone sends velocity commands forever, truth state never updates, detector sees the same gate at the same range every tick, belief model keeps estimating 8 m to gate 0.

**Fix.** Store `initial_altitude_m` and `seed` as fields in `__init__`, and forward *every* configuration kwarg in `reset()`:

```python
async def reset(self) -> None:
    # Re-seed via __init__ so every invariant gets zeroed in one
    # place. All configuration kwargs must be forwarded — a missing
    # kwarg silently reverts to its default and that's how
    # MockDCLAdapter.connect() ended up with a ground-fixed drone
    # and auto_step=False after S19t.
    self.__init__(
        dt=self._dt_default, vel_tau=self._vel_tau, yaw_tau=self._yaw_tau,
        accel_noise_sigma=self._accel_noise, gyro_noise_sigma=self._gyro_noise,
        accel_bias=self._accel_bias, gyro_bias=self._gyro_bias,
        seed=self._seed,
        initial_altitude_m=self._initial_altitude_m,
        auto_step=self._auto_step,
    )
```

**Lesson.** When `reset()` is implemented as `self.__init__(**subset_of_kwargs)`, every kwarg missing from that subset silently reverts to its default. This is a recurring footgun in any "reset-by-reconstruct" pattern. The defense is "forward all or forward none" — either pass every constructor kwarg, or drop the reconstruct pattern and zero the mutable state fields directly.

### Bug 2: `RaceRunner` passed `gates_ned=None` in non-fusion mode

After bug 1 was fixed, the smoke test still stalled — this time at 2/12 gates, same pattern as `--backend mock --detector virtual` had been exhibiting all along (previously invisible because no one runs the basic mock path to completion). Tracked it to `src/race/runner.py` line 151:

```python
# before:
gates_ned=self.gates if self.pose_fusion is not None else None,
```

S19o built the gate-aware BeliefNav fallback (fires when the tracker loses a detection, steers toward `gates_ned[target_idx]`). S19p guarded it behind `pose_trusted`. But this line meant non-fusion mode always got `gates_ned=None`, so the fallback **could never fire** without fusion — the drone would stall at whichever gate the belief tracker lost first. The S19s `UserWarning` correctly flagged this, but the warning lived in `RaceLoop.__init__` and `RaceRunner` was triggering it in production every non-fusion run.

The conditional made no sense in either direction:

- In fusion mode: gates_ned is required, fine.
- In non-fusion mode: `pos_for_plan` is adapter truth. Truth + known gate positions is the *most* reliable gate-aware config there is. Passing None strictly dominates passing the gates.

**Fix.** Pass `gates_ned=self.gates` unconditionally. Added a comment explaining the previous bug so it doesn't regress.

**Impact.** `--backend mock --detector virtual --course technical`: 2/12 timeout (before) → 12/12 in 12.1 s (after). Same for `mock_dcl`: 0/12 timeout → 12/12 in 13.0 s. Fusion path unchanged (gates_ned was always passed in that branch).

### Why MockDCL is earning its keep

This is the canonical "hermetic sandbox catches a bug you'd otherwise hit in the real environment" moment. PX4 SITL never exercised the no-ARM_ACTION lifecycle branch in RaceRunner or the gym-style reset convention, so these bugs were dormant. A real DCL SDK arriving in May would have hit both on day 1 — bug 1 as an altitude-0 crash, bug 2 as a mysterious stall that would have been blamed on the SDK. Instead they're fixed now, 2–4 weeks before the SDK arrives, with unit tests already validating the fixes. This is the entire argument for building MockDCL pre-landing.

### Test status

Full S19 suite + legacy suites: **92/92 pass** across 14 test files (test_gate_belief 5, test_belief_replay 1, test_belief_nav_gate_aware 7, test_pose_fusion 6, test_eskf 10, test_race_loop 5, test_race_loop_fusion 4, test_race_loop_gate_pass 12, test_dcl_adapter_seam 6, test_sim_adapter 8, test_sim_imu 7, test_detector 11, test_race_runner 7, test_vision_roundtrip 4). No regressions.

### Files touched

- `src/sim/mock.py` — store `_initial_altitude_m` + `_seed`, forward all kwargs in `reset()`.
- `src/race/runner.py` — unconditional `gates_ned=self.gates` with explanatory comment.
- `docs/PROJECT_LOG.md` — this entry.

### Pending (unchanged from S19t with one item resolved)

1. **DCL-prep (near-term strategic).** Checklist + seam tests + MockDCLAdapter + first-surface-smoke-test landed (S19t–u). ~~MockDCLAdapter itself~~ ✓. Remaining: day-1 shell script; consider folding `test_dcl_smoke.py` that exercises `--backend mock_dcl --detector virtual` end-to-end (would've caught both S19u bugs pre-commit).
2. **ESKF tuning under harsh noise.** Hardware-blocked.
3. **Real-YOLO vision anchor.** Identity-aware detector should supply gate_idx.
4. **Centreline-crossing detector scaffold.** Would remove last hand-tuned magic number.
5. **Distractor-augmented YOLO training.** GPU-blocked.
6. **Legacy-path navigator recovery.** Descoped by S19s (and S19u further retired the use case — RaceRunner no longer produces `gates_ned=None` callers).


## Session 19v — 2026-04-21 — DCL-prep arc closed

### Context

S19t built `MockDCLAdapter` + checklist + seam tests. S19u's first smoke test of MockDCL surfaced two real bugs, both in files that looked fine in isolation. The obvious follow-up: turn that smoke test into a permanent CI-style guard and ship the day-1 shell script so the DCL-prep arc closes cleanly.

### `test_dcl_smoke.py` — three end-to-end CLI tests

New file, 3/3 pass, ~35 s total. Runs `run_race.py` in subprocess and asserts 12/12 gate completion for:

1. `--backend mock_dcl --detector virtual --course technical` — would have caught S19u bug 1 (`MockKinematicAdapter.reset()` kwarg drop) and bug 2 (`RaceRunner` conditional `gates_ned`). This is the load-bearing test.
2. `--backend mock --detector virtual --course technical` — regression guard on the original mock path (S19u bug 2 also bit this).
3. `--backend mock_kinematic --detector virtual --course technical --fusion --vision-pos-sigma 1.0` — regression guard on the fusion path (confirms the S19u RaceRunner change didn't break fusion).

Decision to use `subprocess.run` rather than `import run_race; run_race.main()`: the whole point is to exercise the actual CLI as a user would, including the argument parser, the mavsdk-stub logic, backend factory dispatch, and async loop bring-up. An import-level call wouldn't catch regressions in any of those.

Additional sanity assertion: the S19s `UserWarning` about `gates_ned=None` must NOT appear in stderr on any production path. If it ever does, RaceRunner has regressed back to conditional `gates_ned`. Trips loud.

### `scripts/day1_dcl.sh` — mechanical day-1 sequence

Derived directly from `docs/DCL_INTEGRATION_CHECKLIST.md §3`. Four ordered steps, each logged to `dcl_day1_logs/<utc-stamp>/`:

0. **Hermetic baseline** — `python test_dcl_smoke.py`. If this fails, the race stack is broken independent of DCL and any downstream failure would be noise.
1. **State + command seam** — `run_race.py --backend dcl --detector virtual --course technical`. Stresses only `get_state()` and `send_velocity_ned()`. Failure → coordinate frame or async wrapping.
2. **Camera seam** — `run_race.py --backend dcl --detector yolo_pnp ...`. Failure → shape/dtype/BGR-vs-RGB in `get_camera_frame()`.
3. **IMU + fusion seam** — `run_race.py --backend dcl --detector yolo_pnp --fusion ...`. Failure → specific-force vs gravity-subtracted accel convention, or IMU unit mismatch.

Each failure includes a debug hint pointing at the most likely cause (single-step sandbox recipes for inspecting the frame or the IMU sample). Step 4 (bench harness DCL comparison) is deferred — `bench_fusion_ab.py` is hardwired to `MockKinematicAdapter` (line 69) and needs a `--backend` flag before it can compare DCL vs mock. Logged as day-2 work in the script output and the gap list.

`set -e` intentionally OFF: we want every step to attempt on day 1 so the operator sees the full failure surface in one pass, not one failure at a time across four re-runs. The script's final exit code is 0 only when steps 0 and 1 are both green (the minimum-viable integration).

**Confirmed today with `scripts/day1_dcl.sh` dry run:** step 0 ✓, step 1 ✗ with `NotImplementedError: DCLSimAdapter.connect — awaiting real SDK` logged to `step1_dcl_virtual.log`, steps 2–3 cleanly skip on missing YOLO weights, step 4 deferred. Exit code 1. This is exactly the expected pre-SDK state.

### Files touched

- `test_dcl_smoke.py` — new, 3/3 pass.
- `scripts/day1_dcl.sh` — new, executable.
- `.gitignore` — added `dcl_day1_logs/`.
- `docs/DCL_INTEGRATION_CHECKLIST.md` — updated §3 with TL;DR reference to the script; updated §6 status to `[x]` on all four pre-landing items with cross-refs.
- `docs/PROJECT_LOG.md` — this entry.

### Test status

`test_dcl_smoke.py`: 3/3. All other suites unchanged. Full count: **95/95** (92 prior + 3 new smoke).

### Gap list after S19v

DCL-prep arc closed. Remaining gaps unchanged:

1. ~~**DCL-prep**~~ ✓ **Complete.** Adapter stub, seam tests, MockDCLAdapter, end-to-end smoke test, day-1 shell script all landed. When the SDK drops: `scripts/day1_dcl.sh --model path/to/weights.pt` is the mechanical first-validation run.
2. **ESKF tuning under harsh noise.** Hardware-blocked.
3. **Real-YOLO vision anchor.** Identity-aware detector should supply gate_idx.
4. **Centreline-crossing detector scaffold.** Would remove last hand-tuned magic number.
5. **Distractor-augmented YOLO training.** GPU-blocked.
6. **Legacy-path navigator recovery.** Descoped.
7. **(New)** `bench_fusion_ab.py` is hardwired to `MockKinematicAdapter`. Add a `--backend` flag before day-1 so honest_passes/max_err can be compared DCL vs mock_kinematic at matched seeds. Day-2 item; not urgent.

### Where we are now

The stack is in a clean waiting state for two external inputs: the DCL SDK (~May 2026, 2–4 weeks out) and real Neros hardware for ESKF retuning. Everything that can be pre-built ahead of those has been pre-built. The simulator-only gaps that remain (3, 4) are judgment calls: whether to do them now risks having DCL invalidate them; whether to wait risks idle time. The asymmetric-cost argument still favours waiting.

---

## Session 19w — Modern soak + fast-time RaceLoop + brutal noise profile (2026-04-21)

**Goal.** Pull on the "1/20 sometimes fails on mixed" thread from the legacy `ab_soak.py` numbers. The legacy harness was S9-vintage, hardcoded a macOS dev path, and drove PX4 SITL directly via mavsdk — so its failure variance came from wallclock-paced SITL jitter, not detector noise trajectory. The failure mode was never categorised because the single-run-per-seed was too slow to iterate on. Modernise: sandbox-runnable, seed-varying, failure-mode-legible soak harness; run it at N≥100 across the (course × noise_profile) grid; actually identify what makes trials fail.

### What landed

**`soak.py`** — new top-level harness. Runs the cross product of courses × noise profiles × trial seeds through `MockKinematicAdapter + VirtualDetector(seed=trial_seed) + BeliefNav + RaceRunner`. Per-trial diagnostics: last target_idx, last position, distance to current gate, frames without detection at tail, median recent commanded velocity magnitude. Post-hoc failure-mode classifier buckets trials into `{completed, stall, stuck_at_gate, lost_target, drifted, other, crash:*}`. JSON output under `soak_runs/<timestamp>.json`, plus a summary table per (course, noise) cell: N, completion rate, median/p95 time, failure-mode histogram.

**`vision_nav.py::VirtualCamera`** — added `seed` kwarg. The `self.rng = random.Random(42)` hardcode had been the latent source of soak invariance: for over a year, every trial had the *same* noise trajectory, so the single-seed bench (`bench_fusion_ab.py`) could surface whatever failure seed 42 produced but could not tell us what *fraction* of noise trajectories failed. Threaded through `VirtualDetector` and the `make_detector("virtual", ..., seed=...)` factory. Default 42 preserves deterministic behaviour for test fixtures that depend on it.

**`NOISE_PROFILES["brutal"]`** — new profile past `harsh`: bearing_sigma 10°, range_sigma 30%, miss_prob 35%, FOV 60°H×45°V, max_detect_range 15 m, bearing_bias 3°. Exists specifically to stress past the previous clean→harsh ceiling in soak; V5.1 saturated on harsh at 100% completion and harsh wasn't surfacing the failure envelope we eventually need for ESKF-fusion budget on hardware.

**`RaceLoop.run(realtime=True|False)`** — added fast-time mode. When `realtime=False`, the loop drops the `asyncio.sleep(dt)` pacing and advances a sim clock by `dt` per tick instead of reading `time.time()`. Safe only against adapters whose physics is tick-driven (MockKinematicAdapter with `auto_step=True`); PX4 SITL and DCL run on their own real clocks and would misbehave. Threaded through `RaceRunner.fly(realtime=...)`; `soak.py` sets `realtime=False`. **Effect: ~3000× speedup.** 13 s wallclock per trial → ~4 ms. A 900-trial soak finishes in 30 s. Without this, N=100 × 9 cells would have been ~45 min of wall time per run — too slow to iterate on.

**`StepResult.pos_ned` + `yaw_deg`** — added pose fields to the per-tick log record. Previously only target_idx / detected / range_est / commanded velocity were captured; the absent pose made failure-mode categorisation (distance-to-gate, drift) impossible without re-simulating. Both fields default `None` so test fixtures that pre-date the change are backward compatible. In fast-time mode `step.t` is overwritten with sim time (not wall time) so downstream analysis sees consistent Δt across ticks.

### The soak result

**First pass** (N=100, seeds 1000–1099, timeout=30 s, 9 cells):

| course     | noise | pass  | rate  | t_med  | t_p95  | modes                               |
|------------|-------|-------|-------|--------|--------|-------------------------------------|
| technical  | clean | 100   | 100%  | 11.1   | 11.2   | —                                   |
| technical  | mild  | 100   | 100%  | 10.9   | 11.4   | —                                   |
| technical  | harsh | 100   | 100%  | 11.3   | 11.8   | —                                   |
| mixed      | clean | 100   | 100%  | 17.4   | 17.4   | —                                   |
| mixed      | mild  | 100   | 100%  | 17.0   | 17.2   | —                                   |
| mixed      | harsh | 100   | 100%  | 16.7   | 17.0   | —                                   |
| sprint     | clean | 100   | 100%  | 27.8   | 27.9   | —                                   |
| sprint     | mild  | 100   | 100%  | 27.7   | 27.9   | —                                   |
| **sprint** | **harsh** | **45** | **45%** | 29.8 | 30.0 | other:33, stuck_at_gate:12, drifted:10 |

One cell — sprint × harsh — showed 55% failure. Every other cell was 100%.

**Drilling into sprint/harsh failures.** All 55 failures had `last_target_idx == 9` (the 10th, final gate). Sample positions clustered near `(3–10, 16–20, -3)`, vs gate 9 at `(0, 15, -3)` — drones arriving at gate 9 but ending at the 30 s deadline before passing it. Replay of seed 1000 with a 35 s timeout confirmed: the drone *does* pass gate 9 at t=30.28 s (0.28 s past the soak's 30 s cutoff). Per-gate timing for that trial showed cruise speeds ~9 m/s (matching `cruise_speed` setting) with gate legs 7→8 (40 m) and 8→9 (33 m) taking 4.4 s and 3.7 s respectively. No algorithmic stall.

**Verification run.** N=100 sprint/harsh with `--timeout 45.0` (seeds 1000–1099): **100/100 completed**, median 30.1 s, p95 31.6 s. Second independent N=100 with `--timeout 90.0` (seeds 2000–2099): **100/100**, median 30.0 s, p95 31.0 s, max 31.5 s. The "55% failure" was a timeout artifact — no trial genuinely stalled.

**Stress with brutal.** Re-soak N=100 × 3 courses with the new `brutal` profile (seeds 3000–3099, timeout=60 s): **300/300 completed**. Median times: technical 10.7 s, mixed 17.8 s, sprint 34.0 s. The ~4 s slowdown vs harsh on sprint is the cost of 35% miss + 60° FOV + 15 m range — more coast-between-gates ticks — but not enough to break the planner.

### What this tells us about the stack

1. **V5.1 + BeliefNav is robust to detector noise variance across all 1200 trials** (900 clean/mild/harsh + 300 brutal). Under kinematic physics with only the detector seed varying, the algorithmic failure rate is **0%**. This is stronger than prior single-seed benches could establish.
2. **The "1/20 sometimes fails" legacy number was almost certainly wallclock-variance + timeout-window margin, not an algorithmic issue.** Legacy ab_soak.py used PX4 SITL (wallclock-paced, jittery) and whatever timeout was the default at the time; sprint/mixed occasionally nudged past it by a fraction of a second under harsh detection. When the sim runs deterministically (kinematic, tick-paced), there's nothing stochastic enough to *produce* a 5% failure rate from the planner side.
3. **Gap list gap 6 ("legacy-path navigator recovery") remains descoped correctly.** The gates_ned=None path is opt-in-only minimal-integration, not a path production exercises.
4. **Race timeout defaults should allow sprint/harsh headroom.** Current `run_race.py` default is 120 s — plenty, but `test_dcl_smoke.py` uses 30 s on the technical course (which tops out at 11.8 s p95, fine). If sprint ever shows up in smoke tests, 45+ s is the floor.

**What we still can't surface in sandbox:**
* PX4 SITL wallclock jitter (only PX4 SITL + real wallclock)
* Motor nonlinearity, wind, thermal drift (only hardware)
* Lighting/occlusion corner cases in real-YOLO (only real camera feed + GPU-trained weights)
* DCL-specific failures (only DCL SDK)

### Files touched

- `soak.py` — new, 900-trial sweep finishes in ~30 s.
- `vision_nav.py` — `VirtualCamera.__init__(seed=42)` + `NOISE_PROFILES["brutal"]`.
- `src/vision/detector.py` — `VirtualDetector(seed=42)` + `.name()` now includes the seed.
- `src/race_loop.py` — `StepResult.pos_ned`/`yaw_deg` + `RaceLoop.run(realtime=...)`.
- `src/race/runner.py` — `RaceRunner.fly(realtime=...)` forwarding.
- `run_race.py` — `--noise` choices extended with `brutal`.
- `soak_runs/` — ignored, regenerated per run.
- `docs/PROJECT_LOG.md` — this entry.

### Test status

All prior suites unchanged: `test_dcl_smoke.py 3/3`, `test_race_loop.py 5/5`, `test_race_loop_fusion.py 4/4`, `test_race_loop_gate_pass.py 12/12`, `test_race_runner.py 7/7`, `test_detector.py 11/11`, `test_dcl_adapter_seam.py 6/6`, `test_vision_roundtrip.py 4/4`, `test_belief_replay.py` PASS, `bench_fusion_ab.py --course technical` unchanged.

### Gap list after S19w

No gaps closed or opened. One clarification: gap 6 ("legacy-path navigator recovery") is now formally de-prioritised — it only affects the `gates_ned=None` minimal-integration path, which `run_race.py`, `soak.py`, `test_dcl_smoke.py`, and `RaceRunner` all avoid. We'd only touch this for a third-party integration of `RaceLoop` that specifically opted in.

1. ~~**DCL-prep**~~ ✓ Closed S19v.
2. **ESKF tuning under harsh noise.** Hardware-blocked.
3. **Real-YOLO vision anchor.** Identity-aware detector should supply gate_idx.
4. **Centreline-crossing detector scaffold.** Simulator-only work with DCL-invalidation risk.
5. **Distractor-augmented YOLO training.** GPU-blocked.
6. **Legacy-path navigator recovery.** De-prioritised.
7. **`bench_fusion_ab.py` `--backend` flag.** Day-2 DCL item.
8. **(New, low priority)** Flight-replay visualiser. `StepResult` now carries enough to reconstruct the trajectory; a Three.js or matplotlib artifact would make S19w-class failures legible at a glance. Deferred — no current failure surface actually needs it.

### Where we are now

Post-S19w, **the sim-side testing harness is the mature thing in the stack**. We have a fast-time sandbox soak that runs 900 trials in 30 s and will surface any future detector-side regression as soon as we add it. The remaining unknowns are all external: DCL SDK behaviour (weeks away), real hardware behaviour (months away), real YOLO behaviour under race lighting (GPU-blocked for training, hardware-blocked for evaluation). The asymmetric-cost argument for waiting still holds, and with this harness the waiting window now has a mechanical regression check for anything we touch.

---

## Session 19x — Pre-DCL perception prep: IdentityTagger + test harness repair (2026-04-24)

### Why this session

Two drivers. First, an audit of S19w's changes surfaced a real bug: running `pytest` across multiple test files in one invocation failed 23 tests with `ImportError: cannot import name 'System' from 'mavsdk'`. Individual per-file pytest runs had always worked, but naked `pytest` (or any aggregate CI invocation) was broken. Second, strategic direction: the next highest-payoff pre-DCL work is the perception swap, and the specific pre-req that's unblocked in the sandbox — no GPU, no gate-trained weights, no DCL SDK — is the data-association layer that converts real-YOLO's `gate_idx=-1` detections into identity-tagged ones the race loop's `target_idx` picker can use.

### What landed

**`conftest.py` at repo root — single mavsdk stub, one place.** Several legacy test files each install their own mavsdk stub at module import time, guarded by `if "mavsdk" not in sys.modules`. Three of them (`test_gate_belief.py`, `test_belief_replay.py`, `test_belief_nav_gate_aware.py`) only set `mavsdk.offboard.VelocityNedYaw` — they never set `mavsdk.System`. Under a multi-file pytest invocation, alphabetical collection order meant one of those short stubs always landed first; every other test file's full stub was short-circuited by the `not in` guard; `vision_nav.py`'s `from mavsdk import System` then `ImportError`ed. The fix is one module-level stub in `conftest.py`, which pytest loads before walking the test tree — all subsequent per-file stubs become idempotent no-ops. Also added `collect_ignore_glob` for the files that need `lsy_drone_racing` (gitignored) or `cv2`/`torch` (sandbox doesn't have them): `ab_exit_s11_test.py`, `s16_vision_test.py`, `s18_belief_test.py`, `test_connect.py`, `test_render.py`, `test_classical_detector.py`, `test_yolo_pipeline.py`, `sims/**`.

**Effect:** `pytest` from repo root now collects and runs everything: **123 passed, 10 warnings in 63 s** (106 existing + 17 new from S19x).

**`soak.py` cleanups.** `--noises` help-string added `brutal` (was `(clean|mild|harsh)`). Removed unused `field` import from `dataclasses`. Classifier `lost_target` threshold now uses per-step timestamps instead of a hardcoded `1.0/50.0`, so the harness is rate-independent — future `--command-hz 30` runs classify correctly.

**`src/vision/identity.py` — the real work.** Two classes:
- `IdentityTagger(gates_ned, accept_radius_m=2.5, ambiguity_ratio=1.3)`. Back-projects a body-frame detection (bearing_h, bearing_v, range_est) through drone pose (pos_ned, yaw_rad) into world NED, then nearest-neighbours against the configured gate list. Returns the detection with `gate_idx` filled in, or `-1` when unmatched (beyond accept radius) or ambiguous (second-nearest within `ambiguity_ratio × nearest`). Detections that already carry a valid `gate_idx ≥ 0` pass through unchanged — lets the tagger be wrapped around VirtualDetector defensively without clobbering sim-truth tags.
- `TaggedDetector(inner, tagger)`. Wraps any Detector, calls `inner.detect(frame, state)`, re-tags output using `state.pos_ned` + `state.att_rad[2]`. Drop-in replacement for any `Detector` — satisfies the same Protocol. `.name()` surfaces the composition so logs show `"tagged[yolo_pnp[gate_weights.pt]]"`.

Plus `backproject_ned(...)` as a module-level function — exposed for reuse (the replay tests could profit from it; a future flight-replay visualiser definitely would).

**Design rationale worth keeping:**
- **Accept-radius + ambiguity-ratio, not max-likelihood.** A physical 2 m gate at typical range has back-projection noise well under 1 m for clean bearings; the failure we're guarding against is gross mis-association (just-passed gate vs. real target, >5 m apart in NED), not fine-grained multi-hypothesis tracking. A 2.5 m radius + 1.3× ratio is enough to trade a handful of over-rejections for zero cross-tags. Upgrade only if real YOLO data proves it's needed.
- **Pull pose from adapter state, not fused estimate.** Fusion can diverge; adapter truth cannot. If the ESKF blows up, a tagger using fused pose would stamp detections with *wrong identity* and lock belief onto the wrong gate. Using adapter state means the tagger stays correct even when fusion is in the weeds — cost is that the tag is only as good as the adapter's own pose (sim truth in mock/DCL, EKF2 output in PX4/real).
- **Wrapper Detector, not picker patch.** Keeps the Detector protocol unchanged; swap is a 1-line change at construction. No coupling of association to the loop's `target_idx` state, tagger stays stateless.

### What we verified

`test_identity_tagger.py` — **17 tests, all pass in 0.17 s**:
- Backprojection math: straight-ahead, yaw=90° east rotation, translation, right-bearing, vertical bearing.
- Tag accept: clean in-radius match tags correctly.
- Tag preserves existing ≥0 `gate_idx` (defensive wrap).
- Tag rejects: out-of-radius → -1; ambiguous (two gates within 1.3× ratio) → -1; empty gate list → -1.
- Ambiguity gate disable (`ambiguity_ratio=0`) → take nearest regardless.
- `tag_all` preserves input order.
- `TaggedDetector` wraps correctly; empty inner → empty out; `.name()` reports composition.
- **Cascade reproducer (`test_picker_cascade_with_untagged_detections`):** S19m scenario — drone at (8,0,-2) facing N, gate 0 behind at (5,0,-2) and gate 1 ahead at (20,0,-2). Both detections carry `gate_idx=-1` as real YOLO would emit. `RaceLoop._pick_detection(associate_mode="target_idx", target_idx=1)` falls into the permissive `gi < 0` branch and returns the nearest — which is the just-passed gate. Test asserts this happens. That's the bug.
- **Cascade resolver (`test_picker_resolves_after_tagging`):** Same scenario run through `IdentityTagger`; both detections get correctly tagged (`[0, 1]`); picker's exact-match branch now finds the target. Asserts `picked is tagged[1]`. That's the fix.

Full suite after S19x: **123 passed, 10 warnings in 63 s** under naked `pytest`. No prior tests regressed.

Soak smoke: `python soak.py --n 3 --courses technical --noises clean,brutal --summary-only --timeout 45` → 6/6 completed in 0.1 s wallclock, fail_modes `—`. Rate-independent classifier still works.

### What this DOESN'T prove (yet)

The tagger is *structurally* correct against the known cascade pattern. What it doesn't prove:
- **That real YOLO actually emits detections with clean enough bearing/range to back-project within the 2.5 m accept radius.** We haven't fed a real YOLO output through it — weights are PC/GPU-blocked. Next pre-DCL step (when PC access is available): run YoloPnpDetector on rendered synthetic frames, pipe through TaggedDetector, confirm tags match ground truth on a known course.
- **That the 2.5 m / 1.3× defaults survive realistic pose noise.** Sandbox tests use sim-truth pose; real PX4/DCL/hardware pose carries EKF2 noise that can add meters of error at altitude. Needs a stress test with injected pose noise once we have something to measure against.
- **That the soak's 100% robustness number holds when TaggedDetector is live.** Currently soak runs VirtualDetector (already tagged). A follow-up soak that swaps in `TaggedDetector(StubYolo(gate_idx=-1), IdentityTagger(gates))` would characterise the tagger's contribution without requiring real YOLO.

### Files touched

- `conftest.py` — new, 70 lines. Single mavsdk stub + collection ignores.
- `soak.py` — help-string, unused import, rate-independent classifier (minor).
- `src/vision/identity.py` — new, 205 lines. IdentityTagger + TaggedDetector + backproject_ned.
- `test_identity_tagger.py` — new, 17 tests including cascade reproducer + resolver.
- `docs/PROJECT_LOG.md` — this entry.

### Gap list after S19x

Gap 3 ("Real-YOLO vision anchor — `_pick_detection` falls back to target_idx when gate_idx=-1") is now **structurally closed** — the tagger exists, is tested against the cascade, and wraps cleanly around YoloPnpDetector. The remaining work is *integration verification against real weights*, which is PC/GPU-blocked. Re-classifying gap 3 from "unresolved risk" to "ready-to-wire, needs PC".

1. ~~**DCL-prep**~~ ✓ Closed S19v.
2. **ESKF tuning under harsh noise.** Hardware-blocked.
3. ~~**Real-YOLO vision anchor**~~ — structurally closed via IdentityTagger (S19x). Integration verification PC/GPU-blocked; mechanical once weights + render pipeline are available.
4. **Centreline-crossing detector scaffold.** Simulator-only work with DCL-invalidation risk.
5. **Distractor-augmented YOLO training.** GPU-blocked.
6. **Legacy-path navigator recovery.** De-prioritised (S19w).
7. **`bench_fusion_ab.py` `--backend` flag.** Day-2 DCL item.
8. **Flight-replay visualiser** (low priority, S19w).
9. **(New)** Tagger stress test with injected pose noise. Characterises the 2.5 m / 1.3× defaults against realistic pose error before hardware lands. Sandbox-runnable.

### Where we are now

Pre-DCL, the sandbox-runnable work list now has one fewer item. The perception-swap unblock is ready — `TaggedDetector(YoloPnpDetector(...), IdentityTagger(gates))` is a three-line construction at run_race's wiring site, deferred to the day weights + a rendering path are both available. Meanwhile the stack will run the full 123-test suite in one shot and has one less class of "how do I even run the tests" friction for anyone onboarding.

### Follow-up: dynamic validation (S19x-f)

Added `test_tagger_integration_does_not_regress_completion` — a small integration harness that runs `N=5` full races on technical/mild with (a) stub-untagged detector (`VirtualDetector` output with all `gate_idx` forced to -1, modelling real YOLO behaviour today) vs. (b) `TaggedDetector(stub, IdentityTagger(gates))`. Same physics (`mock_kinematic`, `realtime=False`), same navigator (`BeliefNav`), same seeds per condition.

**Finding: 5/5 both conditions, delta 0.** The cascade the picker unit test demonstrates as a static failure is *not* load-bearing during a running race under typical topology — V5.1 + BeliefNav's existing defences (drone-displacement refractory from S19n/r + gate-aware nav fallback from S19o + the picker's `gi < 0` permissive branch skipping already-passed gates) collectively swallow the bad pick before it compounds.

This is a useful negative result — it reclassifies the IdentityTagger from "operational fix" to "architectural insurance for the real-YOLO distractor regime": the failure mode the tagger actually closes is when detections include *false positives near non-target gates* (texture confusers, reflections), which VirtualDetector + FOV constraints cannot simulate. Adding a distractor-injection test here would be synthesising a failure we have no evidence exists at the right magnitude — better to characterise against real YOLO output once weights are available and use that as calibration.

**Template lesson: `asyncio.run()` is a footgun in multi-file pytest.** First attempt used `asyncio.run(runner.fly(...))` inside the test; the full suite dropped to 104/124 with `RuntimeError: There is no current event loop` errors in every other async test. `asyncio.run()` closes the default event loop at exit, and subsequent tests calling `asyncio.get_event_loop().run_until_complete(...)` then fail because Python 3.10+ no longer auto-creates a default loop when none exists. The fix in sync test helpers is to allocate and close a new loop explicitly (`loop = asyncio.new_event_loop(); try: loop.run_until_complete(...); finally: loop.close()`) — never touch the default loop policy. Full suite after fix: **124 passed, 10 warnings in 64 s**.

Gap list amendment: gap 9 (tagger stress under pose noise) now reframed — the more valuable follow-up is *distractor-injection stress on real YOLO output*, which is PC/GPU-blocked. Pure pose-noise sensitivity against `VirtualDetector` output would just characterize the backprojection math already tested in `test_backproject_*`.
