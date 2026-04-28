# AI Grand Prix — Mac → PC Handoff Document

**Date:** 2026-04-22
**Purpose:** Transfer all context so development continues seamlessly on the Windows PC with GPU.

---

## 1. What This Project Is

**Competition:** Anduril AI Grand Prix — $500K prize pool, autonomous drone racing.
**Format:** Time-trial through gates. All gates in order, no human input during run = DQ.
**You:** Solo developer (Conrad Weeden).

**Timeline:**
| Date | Event |
|------|-------|
| Now (Apr 22) | Pre-sim development on Mac + PC |
| ~May 2026 | DCL competition sim drops (starting gun) |
| May–June | VQ1: simple course, highlighted gates, easy perception |
| June–July | VQ2: complex 3D-scanned environment, no visual aids |
| End of July | VQ2 cutoff |
| September | Physical qualifier (SoCal, indoor) |
| November | Championship (Columbus OH, $500K) |

---

## 2. Repository Structure

```
ai-grand-prix/
├── run_race.py              # Top-level entrypoint — wires everything together
├── gate_belief.py           # GateBelief + BeliefNav (EMA belief model, yaw-fixed)
├── vision_nav.py            # VirtualCamera + GateTracker + VisionNav (legacy sim vision)
├── px4_v51_baseline.py      # V5.1 phase-based planner (LOCKED baseline)
├── bench.py                 # PX4 SITL bench harness (hardened restart, singleton lock)
├── soak.py                  # Multi-seed soak test runner
│
├── src/
│   ├── race_loop.py         # RaceLoop — core tick loop (detect → plan → command → pass check)
│   ├── courses.py           # Gate coordinate lists (sprint, technical, mixed)
│   ├── race/
│   │   ├── runner.py        # RaceRunner — lifecycle (connect → arm → takeoff → loop → land)
│   │   └── gate_sequencer.py # GateSequencer — track-agnostic gate discovery mode
│   ├── sim/
│   │   ├── adapter.py       # SimAdapter protocol + PX4SITLAdapter + factory
│   │   ├── mock.py          # MockKinematicAdapter (offline testing, no PX4)
│   │   └── mock_dcl.py      # MockDCLAdapter (DCL-shaped pre-landing scaffold)
│   ├── vision/
│   │   ├── detector.py      # Detector protocol + VirtualDetector + YoloPnpDetector + factory
│   │   ├── classical_detector.py  # ClassicalGateDetector (HSV color + contour, for VQ1)
│   │   ├── pnp_pose.py      # GatePoseEstimator (OpenCV solvePnP)
│   │   └── gate_yolo/
│   │       ├── detect_keypoints.py    # GateKeypointDetector (YOLOv8-pose → PnP)
│   │       ├── generate_data_keypoints.py  # Synthetic YOLO training data generator
│   │       ├── yolov8n-pose.pt        # Base YOLOv8-nano-pose (NOT gate-trained)
│   │       ├── yolov8n.pt             # Base YOLOv8-nano (NOT gate-trained)
│   │       └── training_history.json  # 2 training cycles logged (mAP50=0.96)
│   ├── estimation/
│   │   ├── eskf.py           # Error-State Kalman Filter
│   │   └── pose_fusion.py    # PoseFusion (ESKF + vision gating)
│   └── control/
│       └── (PX4 control scripts)
│
├── docs/
│   ├── BATTLE_PLAN.md        # Competition strategy + phased development plan
│   ├── STRATEGY.md           # Technical architecture + design rationale
│   ├── STATUS_2026-04-20.md  # Comprehensive project status as of Apr 20
│   ├── COMPETITION_REVIEW_2026-04-20.md  # Competition rules analysis
│   ├── DCL_INTEGRATION_CHECKLIST.md      # Day-1 DCL integration punch list
│   └── PROJECT_LOG.md        # Session-by-session development log
│
├── test_gate_sequencer.py    # 11 tests — gate sequencer ✓
├── test_classical_detector.py # 9 tests — classical CV detector ✓
├── test_gate_belief.py       # 5 tests — belief model ✓
├── test_belief_replay.py     # yaw dropout replay (0.22m fixed vs 20m buggy) ✓
├── test_belief_yaw_ab.py     # offline A/B: mild 0/3→3/3, search 83%→7% ✓
├── test_yolo_pipeline.py     # YOLO contract test (skips without ultralytics) ✓
├── test_detector.py          # VirtualDetector tests ✓
├── test_race_loop_gate_pass.py # Gate pass detection tests ✓
├── test_eskf.py              # ESKF unit tests ✓
├── test_pose_fusion.py       # PoseFusion tests ✓
├── test_dcl_adapter_seam.py  # DCL adapter shape tests ✓
├── test_dcl_smoke.py         # DCL smoke tests ✓
└── requirements.txt          # Full pip freeze (includes ultralytics, torch, etc.)
```

---

## 3. What's Been Done (Sessions 1–22)

### Epoch 1: Foundation (Apr 6–10)
- PX4 SITL harness, MAVSDK skeleton flying
- YOLOv8 gate detector trained on synthetic data (mAP50=0.96)
- V4→V6 phase-based planner, first autonomous gate completions

### Epoch 2: Control Tuning (Apr 10–14)
- 6+ paired A/B experiments
- Adopted: `px4_speed_ceiling=9.5`, `max_speed=12.0`, `transition_blend`
- Identified PX4 ~2.5 m/s tracking floor as environment ceiling
- **LOCKED baseline:** V5.1, mixed 22.8–24.5s, technical 13.3–13.7s, **100% completion across 1200 sandbox trials** (S19w re-validation; the older "95%" was a timeout artifact on sprint×harsh)

### Epoch 3: Perception (Apr 14–20)
- VirtualCamera + GateTracker + VisionNav pipeline
- GateBelief + BeliefNav with EMA fusion
- S18 A/B exposed yaw propagation bug (mild 0/3 with 83% search)
- S19 fixed the yaw bug (`_prev_yaw_rad` for body→NED / `yaw_rad` for NED→body)
- ESKF + PoseFusion for IMU-vision fusion
- SimAdapter abstraction (PX4, MockKinematic, MockDCL)
- RaceLoop + RaceRunner architecture

### Epoch 4: Race-Ready Stack (Apr 20–22)
- **GateSequencer** (`src/race/gate_sequencer.py`): Track-agnostic gate discovery. Detects, latches, passes, suppresses, predicts next gate position. CLI: `--discovery` flag. 11 tests pass.
- **ClassicalGateDetector** (`src/vision/classical_detector.py`): HSV color + contour for VQ1 highlighted gates. 9 color profiles, range from apparent size, body-frame bearings. CLI: `--detector classical`. 9 tests pass.
- **YOLO pipeline wired**: `YoloPnpDetector` → `GateDetection` → navigator chain complete. Blocked on trained gate model (base `yolov8n-pose.pt` detects human poses, not gates). Contract tests pass.
- **Belief yaw fix validated**: Offline mock A/B confirms mild/belief 0/3 → 3/3, search 83% → 7%. Full PX4 retest recommended.

### Epoch 5: DCL-prep + Sandbox Soak + Identity Association (Apr 22–24) — MOST RECENT
- **DCL scaffolding closed (S19v):** `MockDCLAdapter` + 6 seam contract tests + `test_dcl_smoke.py` (3 end-to-end CLI tests) + `scripts/day1_dcl.sh` (mechanical day-1 sequence with per-step logging). **Day-1 DCL action is now one shell command.**
- **Sandbox soak harness (S19w):** `soak.py` runs 900 trials in 30 s wallclock via `realtime=False` fast-time mode. Seed variance threaded through VirtualCamera → VirtualDetector → make_detector. New `brutal` noise profile. Per-tick `StepResult.pos_ned`/`yaw_deg` instrumentation. 1200-trial soak (3 courses × 4 noise profiles × ~100 seeds) showed **0% algorithmic failure**; the legacy "1/20 fails" was a timeout artifact on sprint×harsh, confirmed by re-running at 45 s and 90 s budgets (100/100 both).
- **IdentityTagger for real-YOLO association (S19x):** `src/vision/identity.py` backprojects detections through adapter pose to NED + nearest-gate match + ambiguity gate. Wraps any Detector via `TaggedDetector`. Closes gap 3 (real-YOLO vision anchor) structurally; integration mechanical via new `--tag-identities` CLI flag. 17 unit tests + 1 dynamic integration test. Back-check finding (S19x-f): V5.1 swallows the cascade in typical topology, so the tagger is insurance for the real-YOLO distractor regime, not operationally load-bearing today.
- **Test harness repair (S19x):** `conftest.py` at repo root installs a canonical mavsdk stub once; fixes 23 cross-file `ImportError`s that had been silently breaking aggregate `pytest`. Full suite now **124/124 in ~64 s under naked `pytest`**.

---

## 4. What's NOT Done Yet

### Immediate (Do on PC)

| # | Task | Why PC | Effort |
|---|------|--------|--------|
| 1 | **Buy DCL The Game on Steam ($5)** | Windows only, need GPU for frame capture | 10 min |
| 2 | **Screen capture pipeline** | `dxcam` on Windows, capture gate frames while flying manually | 1 day |
| 3 | **Retrain YOLOv8-pose on DCL gate visuals** | Needs GPU (RTX 3060+ = 15-30 min vs hours on CPU) | 1–2 days |
| 4 | **Tune classical detector HSV ranges** | Need real DCL gate color samples | 1 hour |
| 5 | **Validate PnP gate dimensions** | Confirm gate physical size matches 2.0m assumption | 30 min |
| 6 | **Full PX4 A/B for yaw fix + soak re-validation** | PX4 SITL only runs well on your PC. Re-run `soak.py`-equivalent soak vs PX4 at N≥20 to corroborate the 100% sandbox claim against real physics. | 2–3 hours |
| 7 | **First real-YOLO end-to-end run** | `python run_race.py --backend mock_kinematic --detector yolo_pnp --model-path <weights.pt> --tag-identities`. Highest-information first experiment once weights are loadable. | 1 hour |

### Before DCL Sim Drops (~May)

| # | Task | Where | Effort | Status |
|---|------|-------|--------|--------|
| 8 | Minimum-snap trajectory optimizer | Mac or PC | 2–3 days | deferred — V5.1 locked and at 100% sandbox |
| 9 | Look-ahead navigation (2-3 gates) | Mac or PC | 1 day | deferred — same |
| 10 | Soak test harness | Mac or PC | — | **✓ DONE (S19w)** — `soak.py`, `realtime=False`, seed variance, classifier |
| 11 | Replay/analysis tooling (flight path viz) | Mac or PC | 1 day | deferred (gap 8, low priority) — `StepResult` already instrumented |
| 12 | Distractor-augmented YOLO training data-gen pipeline | PC (CPU-only for gen) | 1 day | **Round 2 critical** — data-gen can ship pre-GPU |

### When DCL Sim Drops (May, Day 1)

Follow `scripts/day1_dcl.sh` (S19v) — mechanical sequence with per-step forensic logging to `dcl_day1_logs/`.

| # | Task | Details |
|---|------|---------|
| 13 | Download sim, extract camera intrinsics | Update PnP focal length / principal point |
| 14 | Fill in `DCLSimAdapter` work methods | `src/sim/adapter.py::DCLSimAdapter` is a stub (raises `NotImplementedError`). `MockDCLAdapter` (S19t) is a drop-in shape reference. Invert the seam contract tests (`test_dcl_adapter_seam.py`) when real SDK methods compile. |
| 15 | End-to-end smoke: `scripts/day1_dcl.sh --model <path-to-weights.pt>` | Walks through connect → arm → takeoff → offboard → race. `test_dcl_smoke.py` (S19v) provides the same 3-test shape pre-integration. |
| 16 | First competition run: `python run_race.py --backend dcl --detector yolo_pnp --model-path <weights.pt> --tag-identities` | Tagger flag turns on real-YOLO identity association. |
| 17 | Capture 500+ frames from sim, retrain YOLO | Fine-tune on competition visuals |
| 18 | Submit first qualifying run | Slow but clean > fast but crashes |

### Speed Phase (June–July)

| # | Task |
|---|------|
| 16 | Profile time loss by sector |
| 17 | Racing line optimization with smooth splines |
| 18 | Graduate to attitude control |
| 19 | RL controller (PPO) — high-ceiling, only if everything else is clean |

---

## 5. How to Set Up on PC

### 5.1 Get the Code

```bash
# Option A: Push from Mac first
cd ~/ai-grand-prix
git add -A
git commit -m "Pre-PC-handoff: gate sequencer + classical detector + yaw fix"
git remote add origin <your-github-url>
git push -u origin main

# Then on PC:
git clone <your-github-url>
cd ai-grand-prix
```

```bash
# Option B: USB / file copy (if no remote)
# Copy the entire ai-grand-prix/ folder to PC
```

### 5.2 Python Environment (PC)

```bash
# Python 3.11+ recommended (3.13 works too)
python -m venv .venv
.venv\Scripts\activate

# Core deps
pip install numpy opencv-python mavsdk ultralytics torch torchvision

# Full deps (may need adjustment for Windows)
pip install -r requirements.txt
```

### 5.3 Verify Tests Pass

```bash
python test_gate_sequencer.py      # 11 tests
python test_classical_detector.py  # 9 tests
python test_gate_belief.py         # 5 tests
python test_belief_replay.py       # yaw fix validation
python test_yolo_pipeline.py       # YOLO contract (should NOT skip on PC with ultralytics)
python test_belief_yaw_ab.py       # offline A/B
```

### 5.4 YOLO on PC (with GPU)

```bash
# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test YOLO loads
python -c "from ultralytics import YOLO; m = YOLO('src/vision/gate_yolo/yolov8n-pose.pt'); print('OK')"

# Run YOLO pipeline test (should actually run, not skip)
python test_yolo_pipeline.py
```

### 5.5 PX4 SITL (if needed on PC)

```bash
# Only needed for the full PX4 A/B retest (item #6)
# Follow SETUP.md for PX4-Autopilot clone + build
python s18_belief_test.py  # 24-flight A/B, ~45 min
```

---

## 6. Key CLI Commands

```bash
# Smoke test with mock backend (no PX4, no sim)
python run_race.py --backend mock --detector virtual --course technical --timeout 30

# Discovery mode (track-agnostic gate sequencer)
python run_race.py --backend mock --detector virtual --course technical --discovery

# Classical CV detector (for when you have real frames)
python run_race.py --backend mock_dcl --detector classical --discovery

# YOLO detector (needs trained model)
python run_race.py --backend dcl --detector yolo_pnp --model-path path/to/trained_gate_model.pt

# PX4 SITL (needs PX4 running)
python run_race.py --backend px4_sitl --detector virtual --course technical
```

---

## 7. Architecture Quick Reference

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐
│ SimAdapter   │────▶│ Detector │────▶│ RaceLoop │────▶│  BeliefNav   │
│ (PX4/DCL/   │     │ (Virtual/│     │ (detect, │     │  (plan vel   │
│  Mock)       │     │  YOLO/   │     │  pass,   │     │   commands)  │
│              │◀────│  Classic) │     │  command) │     │              │
│ state+frame  │     │          │     │          │     │              │
└─────────────┘     └──────────┘     └──────────┘     └──────────────┘
                                          │
                                    ┌─────┴──────┐
                                    │GateSequencer│ (optional, --discovery)
                                    │ detect→latch│
                                    │ →pass→next  │
                                    └────────────┘
```

**Detector protocol:** `detect(frame, state) → List[GateDetection]`
- `VirtualDetector` — projects known gate coords synthetically (testing)
- `YoloPnpDetector` — YOLOv8-pose + PnP on real frames (needs trained model)
- `ClassicalGateDetector` — HSV color + contour (VQ1 highlighted gates, no model needed)

**Factory:** `make_detector("virtual" | "yolo_pnp" | "classical", **kwargs)`

---

## 8. Immediate PC Action Plan (Priority Order)

1. **Clone repo + set up venv + verify tests pass** (30 min)
2. **Buy DCL The Game on Steam** ($5, 10 min)
3. **Build screen capture pipeline** — fly manually, record frames with gate annotations
4. **Retrain YOLOv8-pose** on captured DCL gate frames (`src/vision/gate_yolo/`)
5. **Tune classical detector** — screenshot a gate, measure HSV values, update profiles
6. **Run full PX4 A/B** (`python s18_belief_test.py`) to confirm yaw fix on real dynamics
7. **Wait for DCL sim** → same-day integration → first submission

---

## 9. Files That Need Attention on PC

| File | What to Do |
|------|-----------|
| `src/vision/gate_yolo/yolov8n-pose.pt` | Replace with gate-trained model after retraining |
| `src/vision/classical_detector.py` | Tune `PROFILES` dict HSV ranges to match actual DCL gate colors |
| `src/vision/pnp_pose.py` | Verify `GATE_PHYSICAL_SIZE_M = 2.0` matches DCL gates |
| `requirements.txt` | May need Windows-specific torch install (`--index-url cu124`) |
| `run_race.py` | Ready to go, just swap `--backend dcl` when sim drops |

---

## 10. Git Status (Uncommitted Changes)

There are **43 uncommitted files** including all the new work from Epochs 3-4. Before transferring:

```bash
cd ~/ai-grand-prix
git add -A
git commit -m "Epoch 3-4: gate sequencer, classical detector, belief yaw fix, DCL scaffolding"
```

Key new files to commit:
- `src/race/gate_sequencer.py` — GateSequencer
- `src/vision/classical_detector.py` — ClassicalGateDetector
- `src/vision/detector.py` — Detector protocol + factory
- `src/race_loop.py` — RaceLoop
- `src/race/runner.py` — RaceRunner
- `src/sim/adapter.py`, `mock.py`, `mock_dcl.py` — SimAdapter stack
- `src/estimation/eskf.py`, `pose_fusion.py` — ESKF + PoseFusion
- `run_race.py` — Top-level entrypoint
- All `test_*.py` files

---

*This document is the complete context needed to continue development on PC. Read it, set up the environment, run the tests, and start with DCL The Game frame capture.*
