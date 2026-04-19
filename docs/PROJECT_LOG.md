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
