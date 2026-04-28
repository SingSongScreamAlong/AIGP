# Anduril AI Grand Prix — Project Report

**Date:** 2026-04-24
**Author:** Conrad Weeden (solo)
**Working directory:** `/sessions/festive-friendly-meitner/mnt/ai-grand-prix/`
**Coverage:** Sessions 1 through 19x-j (2026-04-06 → 2026-04-24, 19 calendar days of focused work)

This is the comprehensive snapshot — what we're building, what we've built, what we've measured, what's left, and what could go wrong. It supersedes nothing; it complements `docs/STATUS_2026-04-24.md` (terse current-state) and `docs/PROJECT_LOG.md` (~3100 lines of session-by-session record). Read this first if you're new to the project, or if it's been long enough that the project log is too granular to start from.

---

## 1. Executive Summary

The project is building a fully autonomous drone-racing software stack to compete in the Anduril AI Grand Prix — a $500K time-trial competition for monocular-FPV-only drones, culminating at the Columbus, OH championship in November 2026. Solo development. The stack is to be deployed on identical Neros 8-inch quadcopters with Jetson Orin NX compute, against a course of physical gates, with no human input during a run. Public-source intelligence and a $1.4M-funded competition pool suggests we will compete against well-resourced university research labs (UZH RPG, ETH, UCL, LSY), A2RL veterans, and defence/robotics companies.

After 19 calendar days of focused work, the stack consists of a frozen V5.1 phase-based velocity planner with a measured PX4 environment ceiling at the control layer, a `VirtualDetector → IdentityTagger → BeliefNav → RaceRunner` perception/control pipeline, a 15-DOF error-state Kalman filter with chi-squared-gated vision fusion, a `SimAdapter` Protocol with PX4-SITL / DCL / mock-kinematic / mock-DCL implementations (DCL adapter is a stub awaiting the SDK), and a sandbox-runnable test/soak harness that runs 124 tests in ~64 seconds and 1200 race trials in ~30 seconds. Across those 1200 trials (3 courses × 4 noise profiles × ~100 detector seeds, `mock_kinematic` physics) the stack completes 100% with adequate timeout — the legacy "1/20 fails" claim from earlier sessions was diagnosed as a 30-second-budget timeout artifact on sprint × harsh, not an algorithmic failure.

**The one-sentence state of the project:** the control stack is verified-frozen and the perception/estimation/integration scaffolding is complete through the last sandbox-addressable layer, and the project is now in a deliberate waiting state for two external gates: the DCL competition simulator (~1–3 weeks), and PC + GPU access for real YOLO weights and training. Every pre-req for both gates that can be done in this sandbox has been done, tested, and documented. The remaining work that genuinely moves the needle toward winning lives outside this environment.

---

## 2. The Competition

### 2.1 Format

Time-trial through a course of physical gates in order. Fastest valid time wins. Any human interaction during a submitted run is immediate disqualification. Maximum run duration 8 minutes. **Completion rate dominates at qualifier cutoffs** — slow but clean entries beat fast but crashing ones at decision time, especially at Round 2 selection.

Scoring rewards both speed and reliability; the engineering tradeoff is the classic risk-vs-reward axis. A robust submission that reliably hits 90% of theoretical-best time is preferable to a fragile submission that hits 100% on its best run and DNFs half the time.

### 2.2 Prize

$500K total prize pool, split among top 10 finishers. The highest-scoring entrant gets an interview skip-pass to Anduril hiring managers — an option Conrad explicitly weights, beyond the prize money.

### 2.3 Hardware (physical phase)

Identical Neros drones for all teams. 8-inch quadcopter class, ~2 kg payload capacity, top speeds up to 160 km/h. NVIDIA Jetson Orin NX compute (~100 TOPS — comfortably runs YOLOv8-nano at >100 FPS). **Single monocular ~12 MP wide-angle FPV camera. IMU only. No LiDAR, no stereo, no depth sensor.** Teams cannot modify hardware; it is a software-only competition.

This hardware constraint is the single biggest shaper of the architecture. Monocular gives bearing but not range; PnP recovery from gate keypoints is the standard answer; pose drift between gate sightings is the ESKF's job. Every subsystem in the stack downstream of perception assumes this constraint.

### 2.4 Technical interface (VADR-TS-001, March 2026)

- Protocol: MAVLink v2 over UDP, MAVSDK-compatible client.
- Physics rate: 120 Hz.
- Recommended command rate: 50–120 Hz.
- Telemetry messages from sim: `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `ODOMETRY`, `TIMESYNC`.
- Control messages to sim: `SET_POSITION_TARGET_LOCAL_NED`, `SET_ATTITUDE_TARGET`.
- Coordinates: local Cartesian NED only. **No GPS, no geographic coords, no global position.**
- Vision stream: forward-facing first-person camera. Specifics in a separate spec **not yet released**.
- Runtime: Python 3.14.2 known good; compiled extensions permitted.

### 2.5 Timeline

| Date | Event | Project status |
|---|---|---|
| 2026-03-09 | VADR-TS-001 spec released | Absorbed |
| 2026-04-06 | Session 1: foundation work begins | — |
| 2026-04-24 | **Today** — sandbox stack at clean waiting state | This report |
| ~2026-05 | DCL simulator drops | Day-1 shell script ready |
| ~2026-05–06 | Virtual Qualifier 1 (simple highlighted gates) | First submission target |
| ~2026-06–07 | Virtual Qualifier 2 (3D-scanned environment + distractors) | Hardened submission target |
| 2026-07-31 | **Round 2 cutoff** | Best time locked |
| 2026-09 | Physical qualifier (SoCal, indoor) | Sim-to-real transfer |
| 2026-11 | **Championship (Columbus, OH)** | — |

DCL drops in mid-May are 1–3 weeks out. Round 2 cutoff is the hardest deadline — by end of July, every team must have a submission that completes the 3D-scanned-environment course with visual distractors, and the relative speeds of those completions determine who advances.

### 2.6 Competition intelligence

Over 1,000 teams registered within 24 hours of the announcement (Palmer Luckey, January 2026). The most credible threats are research labs with prior drone-racing publications (UZH RPG, ETH, UCL, LSY), A2RL 2025 veterans, and defence/robotics companies that already have monocular-vision pipelines for other applications. The reference benchmark — **MonoRace (A2RL 2025 winner)** — hit 28.23 m/s on identical monocular-only constraints, using U-Net gate segmentation → corner extraction → PnP → IMU fusion → PPO-trained neural controller @ 500 Hz. That architecture is the rough shape to match or beat. Our V5.1 phase-based planner is more conservative than a learned controller but is tested-known-stable, which is the right tradeoff for a solo-dev one-attempt-per-qualifier risk profile.

---

## 3. The Technical Problem

The stack has to do five things, on stock Neros hardware, with no human input, in time-trial conditions:

**Perceive.** Recover gate identity, bearing, and range from a single forward-facing FPV camera frame at 50–120 Hz. Inputs are a noisy 12 MP image with motion blur, exposure variation, distractor textures, and (in Round 2) a real 3D-scanned environment with visual confusers.

**Estimate.** Maintain a position and orientation estimate fused from the IMU (high-rate, drifty) and vision fixes (low-rate, accurate but with occasional blunders). The estimator must be robust to vision outliers and to extended periods of no detection.

**Plan.** Convert (gate identity, bearing, range, drone pose) into a velocity command in NED coordinates. The plan has to handle gate-to-gate transitions, lost-detection coast/search, gate passage detection, and target advancement.

**Control.** Send `SET_VELOCITY_NED` setpoints at 50 Hz that PX4 (or DCL physics, or real Neros hardware) can actually track. Hard ceiling: PX4 SITL has a measured ~2.5 m/s tracking floor — any planner command tighter than that is wasted. We have not yet measured the equivalent floor for DCL or Neros.

**Stay alive.** Recover from temporarily lost detections, gracefully handle pose-fusion divergence, never crash through an early gate at 12 m/s with low confidence. Completion rate dominates speed at the cutoffs that decide who advances.

The hard parts, in rough order of difficulty:

1. **Real-image perception transfer.** Synthetic training works in simulation; real cameras under race lighting do not look like synthetic frames. This is the single largest unknown gap in the project. We have a YOLOv8-nano gate detector trained to 95.1% mAP50 on synthetic data, never tested on real footage.
2. **Distractor robustness in Round 2.** The Round 2 environment has visual elements that look superficially like gates — reflections, signage, structural elements. A naïve YOLO will fire on these and the data-association layer has to reject them.
3. **Sim-to-real transfer.** DCL physics ≠ Neros physics. Anything we tune against DCL (which itself differs from PX4 SITL) may need re-tuning on hardware in September.
4. **Time pressure for solo dev.** A single late-stage bug can cost a week. Round 2 cutoff at end of July leaves real margin for one or two recoveries, no more.

---

## 4. Architecture

```
[ Sim adapter ]              ← SimAdapter Protocol (S19b/g)
PX4SITL / DCL (stub) /         abstracts MAVLink / DCL SDK / mock physics
MockKinematic / MockDCL        get_state() | get_camera_frame() | send_velocity_ned()
       │
       ▼ (frame, state)
[ Detector ]                 ← Detector Protocol (S19c)
VirtualDetector |              "frame + state → List[GateDetection]"
YoloPnpDetector |              GateDetection has gate_idx, bearing_h_deg,
ClassicalGateDetector          bearing_v_deg, range_est, confidence, in_fov
       │
       ▼ (optional wrap)
[ TaggedDetector ]           ← S19x — closes gate_idx=-1 association gap
IdentityTagger:                back-projects detection → NED, nearest-gate
backproject + nearest-match    match, accept-radius + ambiguity-ratio gating
       │
       ▼ List[GateDetection]
[ RaceLoop ]                 ← S19c/h core control loop
_pick_detection (target_idx    optional PoseFusion (15-DOF ESKF) drives
or nearest-first) →            pos_for_plan instead of adapter truth
_build_tracker_state →         optional GateSequencer for track-agnostic
GateBelief.update →            discovery (no hardcoded gate positions)
BeliefNav.plan →               gate-pass detection: range + position
adapter.send_velocity_ned      signals + drone-displacement refractory
       │
       ▼ velocity setpoints @ 50 Hz
[ Sim ] back to top
```

### 4.1 Why these abstractions

The architecture started as a single-file PX4 SITL + hardcoded planner script (Sessions 1–15) and was systematically decomposed through Session 19 as the project moved up-stack from control to perception, estimation, and integration. Each abstraction earned its keep:

**`SimAdapter` Protocol (S19b/g).** Decouples the race loop from any specific simulator. PX4 SITL was the development backend through Session 18; DCL would replace it on drop day; mock backends enable sandbox testing without a sim. The Protocol uses `SimCapability` flags so consumers can degrade gracefully — a backend without `ARM_ACTION` (gym-style DCL) skips the takeoff sequence; a backend without `IMU` (PX4 today) refuses fusion mode loudly. The `MockDCLAdapter` (S19t) was specifically built to surface integration bugs *before* the real DCL SDK arrives — and its first smoke test caught two latent bugs that would otherwise have appeared as mysterious DCL failures on day one.

**`Detector` Protocol (S19c).** Separates perception from race-loop logic. `VirtualDetector` projects known gates through a synthetic camera with configurable noise (used for unit tests, ablations, and sandbox soak). `YoloPnpDetector` wraps a YOLOv8-pose model + OpenCV PnP for real images. `ClassicalGateDetector` is HSV-color + contour matching for VQ1's highlighted-gate environment. The protocol contract is uniform `List[GateDetection]` output — swapping perception backends is a one-line change at the call site.

**`IdentityTagger` + `TaggedDetector` (S19x).** Real YOLO emits `gate_idx=-1` for every detection because a real image tells you "there is a gate" but not "this is gate #3". The race loop's picker, in `target_idx` mode, prefers exact `gate_idx` matches and falls back to nearest-first on `-1`. The fallback is the S19m cascade pattern — picker can anchor on a just-passed gate still in FOV, belief mis-anchors, drone drifts. The `IdentityTagger` back-projects each detection's body-frame `(bearing, range)` through the drone's adapter pose to a world-NED point, nearest-neighbours against the known gate list with an accept-radius (default 2.5 m) and ambiguity-ratio (default 1.3×) gate, and stamps `gate_idx`. `TaggedDetector(inner, tagger)` wraps any Detector — drop-in swap. Closes the perception-side of the real-YOLO integration gap structurally; integration verification awaits real weights + pose data on the PC.

**`PoseFusion` + ESKF (S19e/f).** 15-DOF error-state Kalman filter (position, velocity, attitude error, gyro bias, accel bias) based on Solà's right-multiplicative-attitude conventions. Vision fixes are gated by chi-squared distance against the filter's predicted covariance; a rolling-window reject rate drives a `pose_trusted` boolean that the navigator consumes. Defaults are tuned against synthetic IMU and noise-profile-`harsh` vision; will need recalibration against real Neros IMU data when hardware returns.

**`RaceLoop` + `RaceRunner` (S19c/h/d/i).** `RaceLoop` is the pure inner loop (no I/O at construction, no async setup). `RaceRunner` wraps it with the lifecycle (`connect → arm → takeoff → start_offboard → run → land → disconnect`) that real flights need; it also handles the case where the adapter doesn't advertise `ARM_ACTION` (DCL is expected to be airborne-by-default). The split keeps `RaceLoop` testable in isolation and keeps `RaceRunner` thin enough to be backend-agnostic.

**`BeliefNav` + `GateBelief` (S18/S19o).** EMA-fused belief tracker over each gate's bearing/range. `BeliefNav` has three states (TRACKING / COAST / SEARCH) and an optional gate-aware fallback that activates when `pose_trusted AND gates_ned set` — when the belief tracker has lost confidence but the drone still knows roughly where it is, the navigator can fly toward `gates_ned[target_idx]` directly. The S18 propagation bug (incorrect yaw used during dropout) was diagnosed and fixed in S19a–b; replay tests show fixed peak error 0.22 m vs. buggy peak 20.05 m through a 2-second 90°-yaw dropout.

### 4.2 Module inventory

| Path | Purpose | Provenance |
|---|---|---|
| `run_race.py` | CLI entrypoint, backend-agnostic | S19d/i/x |
| `gate_belief.py` | `GateBelief` + `BeliefNav` | S18 / S19a-b yaw fix / S19o gate-aware |
| `vision_nav.py` | `VirtualCamera` + (legacy) `VisionNav` | Sessions 16+ |
| `soak.py` | Sandbox-runnable soak harness | S19w |
| `bench_fusion_ab.py` | Fusion A/B with distractor + honest-pass | S19j/o/q |
| `conftest.py` | pytest-wide mavsdk stub + collect_ignore | S19x |
| `src/courses.py` | `sprint`, `technical`, `mixed` gate lists | — |
| `src/race_loop.py` | Core control loop | S19c/h |
| `src/race/runner.py` | `RaceRunner.fly(...)` lifecycle | S19d/i |
| `src/race/gate_sequencer.py` | Track-agnostic discovery mode | Epoch 4 |
| `src/sim/adapter.py` | `SimAdapter` Protocol + impls | S19b/g |
| `src/sim/mock.py` | `MockKinematicAdapter` (synthetic IMU) | S19c |
| `src/sim/mock_dcl.py` | `MockDCLAdapter` (gym-style stand-in) | S19t |
| `src/estimation/eskf.py` | 15D ESKF | S19e |
| `src/estimation/pose_fusion.py` | `PoseFusion` wrapper | S19f |
| `src/vision/detector.py` | Detector protocol + impls | S19c |
| `src/vision/identity.py` | `IdentityTagger` + `TaggedDetector` | S19x |
| `src/vision/classical_detector.py` | HSV+contour for VQ1 | Epoch 4 |
| `src/vision/pnp_pose.py` | `GatePoseEstimator` (cv2 PnP) | Epoch 1 |
| `src/vision/gate_yolo/` | YOLOv8-pose training + inference | Epoch 1 |
| `scripts/day1_dcl.sh` | Mechanical day-1 DCL sequence | S19v |

---

## 5. Development History

### Epoch 1 — Foundation (Apr 6–10)

PX4 SITL harness in Docker, MAVSDK skeleton flying, synthetic-data YOLOv8 gate detector trained to 95.1% mAP50, V4 → V6 phase-based velocity planner, first autonomous gate course completions. The architecture at this point was a single-file script wrapping all of perception, planning, and control.

### Epoch 2 — Control tuning (Apr 10–14)

Six paired A/B experiments on the planner: launch prefill, z-gate, cruise/ceiling sweep, transition_blend, EXIT speed cap, directional blend, PX4 jerk. Adopted two wins: `px4_speed_ceiling=9.5` / `max_speed=12.0`, and the multiplicative `transition_blend` (`alt_frac × vz_frac × time_frac`). **Closed the class** at Session 15: a measured ~2.5 m/s tracking-floor in PX4 SITL is unresponsive to all available knobs and is correctly identified as an environment ceiling, not a tunable. Locked-baseline protocol established: any change to V5.1 constants requires a fresh A/B harness run.

The locked V5.1 baseline is the strategic anchor for the entire project. It gives us a known-stable control layer that we can build perception and integration on top of without re-tuning. The temptation to "just slightly improve" V5.1 has been consciously resisted.

### Epoch 3 — Perception, estimation, integration (Apr 14–22)

The largest and longest epoch, covering Sessions 16 through 19r. Built in roughly three phases:

**Phase 3a — Vision pipeline build-out (Sessions 16–18).** `VirtualCamera` + `GateTracker` + `VisionNav` for synthetic-vision testing. `GateBelief` + `BeliefNav` (EMA fusion model + TRACKING/COAST/SEARCH state machine) for robust tracking across detection dropouts. The S18 A/B harness exposed a regression: belief-mode went 0/3 on the technical course under mild noise, with 83% search-mode fraction. The bug was not diagnosed at the time — pure-control work was prioritized.

**Phase 3b — Bug fix + abstraction explosion (S19a–i).** S19a–b diagnosed the S18 regression as a yaw-propagation bug in `GateBelief.propagate` (was using current yaw for both body↔NED conversions during dropout; correct is to use the *previous* yaw for body→NED so the propagation rotates through the path actually flown). Fixed with a `_prev_yaw_rad` field; replay tests show peak error 0.22 m vs. previous 20.05 m through a 90° yaw dropout. The next eight sessions built the architecture's seams: `SimAdapter` Protocol (S19b/g) decoupling from any specific sim; `Detector` Protocol (S19c); `RaceLoop` (S19c/h) and `RaceRunner` (S19d/i) replacing the monolithic Session 1–15 entry script; ESKF (S19e) and `PoseFusion` (S19f); `--fusion` and `--vision-pos-sigma` CLI flags (S19i/j).

**Phase 3c — Hardening through fusion stress + distractor scenarios (S19j–r).** S19j discovered that the default `vision_pos_sigma=0.15` produces a chi-squared feedback loop (filter covariance shrinks faster than true error → valid fixes look like outliers → rejected → drift compounds → max error 122 m); the sweet spot is ~1.0 m. S19l ruled out a tempting position-sanity-gate design as a positive-feedback loop. S19m surfaced the most consequential design discovery of the project: **the picker's nearest-first fallback for `gate_idx=-1` (real-YOLO) detections cascades through the just-passed gate still in FOV, racing `target_idx` ahead of the drone's actual progress**. Pre-S19m fusion bench numbers — "12/12 in 0.34 m" — were retroactively understood as fiction generated by this cascade. S19n introduced a drone-displacement-based gate-pass refractory (5 m) that survives all three on-path / off-path / no-decoy bench cases. S19o gated the navigator's gate-aware fallback by a `pose_trusted` boolean (driven by the rolling vision-reject rate), and S19q introduced an `honest_passes` metric — counting only passes where the fused pose is within 2 m of truth at the `target_idx`-advance tick — which exposed that fusion/harsh runs were physically completing courses on detection bearing while the fused pose was 159 m off.

### Epoch 4 — Race-ready stack (Apr 20–22)

`GateSequencer` (track-agnostic gate discovery for environments where the gate list isn't pre-known), `ClassicalGateDetector` (HSV-color + contour for VQ1's highlighted-gate environment), and `YoloPnpDetector` finally wired into the production stack (blocked on a gate-trained model — the base `yolov8n-pose.pt` detects human poses, not gates). The S18 belief yaw fix was validated against an offline mock A/B (mild went 0/3 → 3/3, search 83% → 7%).

### Epoch 5 — DCL prep, sandbox soak, identity association (Apr 22–24)

The most recent epoch, and the one that shifted the project from "perception scaffolding" to "verified-robust waiting state":

**S19s** documented `gates_ned=None` as a minimal-integration-only configuration (the navigator can't recover from stuck-detection lock without world-frame gate positions); added a constructor `UserWarning`. **S19t** consolidated memory (229 → 91 lines), wrote the `DCL_INTEGRATION_CHECKLIST.md`, added six seam contract tests for the DCL stub, and built `MockDCLAdapter` (gym-style "connect = instantiate + reset" reset semantics). **S19u** discovered two bugs through the MockDCL smoke test that would have appeared as mysterious failures on real-DCL drop day: `MockKinematicAdapter.reset()` was dropping mutable kwargs (`auto_step`, `initial_altitude_m`) on reset; `RaceRunner` was silently passing `gates_ned=None` to `RaceLoop` in non-fusion mode, so the entire S19o gate-aware fallback never fired. Both fixed; impact on `--backend mock --detector virtual --course technical`: 2/12 timeout → 12/12 in 12.1 s.

**S19v** closed the DCL-prep arc: `test_dcl_smoke.py` provides three end-to-end CLI tests (which would have caught the S19u bugs pre-commit), and `scripts/day1_dcl.sh` is a mechanical day-1 sequence with per-step forensic logging to `dcl_day1_logs/`.

**S19w** built the modern sandbox-runnable soak harness. Threaded a `seed` parameter through `VirtualCamera → VirtualDetector → make_detector` (the previous hardcoded `random.Random(42)` had been the latent source of soak invariance — every "trial" was the same noise trajectory); added `realtime=False` fast-time mode to `RaceLoop.run` and `RaceRunner.fly` (drops `asyncio.sleep(dt)` pacing, advances a sim clock by `dt` per tick, ~3000× speedup against `MockKinematicAdapter`); instrumented `StepResult` with `pos_ned`/`yaw_deg`; added a `brutal` noise profile past the previous `harsh` ceiling. `soak.py` provides a per-trial failure-mode classifier (stuck_at_gate / stall / lost_target / drifted / other) and runs cells of 100 trials in ~3 seconds. Total soak: 1200 trials (3 courses × 4 noise profiles × ~100 seeds), **0 % algorithmic failure**. The legacy `ab_soak.py` "1/20 fails" claim was diagnosed as sprint × harsh exceeding the default 30 s timeout by 0.1–0.5 s; verification at 45 s and 90 s budgets gave 100/100 both.

**S19x** landed three things: a `conftest.py` at repo root with one canonical `mavsdk` stub (fixes 23 cross-file `ImportError`s that had been silently breaking aggregate `pytest`), the `IdentityTagger` + `TaggedDetector` work described in §4.1, and minor soak.py cleanups. All 124 tests now pass under naked `pytest` in 64 seconds. **S19x-f** added a dynamic integration test for the tagger and produced a useful negative result: under typical race topology, V5.1's existing defences swallow the S19m cascade before it compounds — the tagger is architectural insurance for the real-YOLO distractor regime, not load-bearing today. **S19x-g** wired the tagger into `run_race.py` behind a `--tag-identities` flag. **S19x-h/i/j** refreshed STATUS, PC handoff, and consolidated memory.

### 5.1 Lessons captured (durable design rationale)

The project log preserves session-by-session detail. The lessons that should outlast individual work — ones that would inform a future engineer making the same kind of decision — are kept in `.auto-memory/project_aigp_state.md`:

- **Hardware bring-up rule for vision-pos-sigma:** start at 1.0 m, tune down only after real-image perception is characterised (S19j).
- **Sanity gates that consume the estimate they correct can become positive-feedback loops** (S19l). "By definition" in your own docs is a red flag — write a unit test or math proof before relying on it.
- **The picker's `target_idx` association is the entire reason real YOLO can't be used naïvely** (S19m). Identity tagging is mandatory for the `target_idx` exact-match branch.
- **Drone-displacement refractory beats gate-position refractory for distractor cases** (S19n/r). On-path decoys 1.5 m in front of a gate fire at drone ≈ (gate − 2.5 m); gate-position refractory considers the gate cleared.
- **Pose-trust is the right gate for the gate-aware nav fallback** (S19p). False untrust = revert to belief-coast (safe); false trust = actively mis-steer (catastrophic). Asymmetric-cost argues for the conservative threshold.
- **`honest_passes` beats raw gates-passed** for fusion-quality assessment (S19q). ESKF divergence is bimodal; raw counts smear the regimes.
- **Build scaffolding ahead of an external dependency, not behind it** (S19t/u). MockDCLAdapter caught two real bugs before DCL even drops. The week of "what's wrong with DCL" we'd otherwise spend on day 1 is now a known-quantity integration.
- **Reset-by-re-init with a partial kwarg list is a time bomb** (S19u). Forward all mutable fields directly, or forward zero.
- **Legacy soak failure rates can be timeout artifacts masquerading as distributions** (S19w). Before diagnosing a stochastic failure mode, check whether the budget sits on top of the run-time distribution's tail.
- **Single-seed RNGs in synthetic detectors convert N-trial soak into a single-sample reliability number dressed up as statistics** (S19w). Trace every RNG through every factory chain.
- **Per-file `if not in sys.modules` stubs are a time bomb under multi-file pytest** (S19x). Stub fixtures belong in `conftest.py`, not per-file guards.
- **Static unit tests demonstrating a failure are not evidence the failure matters at system scale** (S19x-f). Always back-check with an integration test before ranking severity.
- **`asyncio.run()` in sync test helpers closes the default event loop and breaks every subsequent async test on Python 3.10+** (S19x-f). Use scoped `new_event_loop()` with `close()` in `finally`.

---

## 6. Current State — Verified

### 6.1 What works and is measured

**Control layer.** V5.1 phase-based planner with `px4_speed_ceiling=9.5`, `max_speed=12.0`, `transition_blend(alt_frac × vz_frac × time_frac)`. Technical course 13.3–13.7 s; mixed course 22.8–24.5 s; sprint course 27.7–31.0 s. Locked since Session 15.

**Sandbox soak.** 1200 trials (3 courses × {clean, mild, harsh, brutal} noise × ~100 detector seeds, `mock_kinematic` physics, fast-time): **0 % algorithmic failure**. Median wallclock for a 100-trial cell: ~3 seconds. The full sweep finishes in ~30 seconds.

**Test suite.** 124/124 under naked `pytest` from repo root in ~64 seconds. Coverage spans: ESKF properties (drift, integration, yaw tracking, bias convergence, covariance PSD, chi-squared gating); GateBelief yaw-invariance + replay; BeliefNav gate-aware steering, pose_trusted gating, fallback conditions; PoseFusion constant-velocity tracking, auto-seed, chi-squared rejection, rolling-window reject rate; RaceLoop end-to-end with VirtualDetector, fusion-branch tracking, range/position/refractory gate-pass detection; SimAdapter dispatch + capability flags; sim-IMU full fusion chain; the IdentityTagger backprojection math, accept/reject/passthrough, wrapper composition, S19m cascade reproducer, and dynamic integration; DCL adapter seam contracts; DCL end-to-end smoke tests against MockDCL.

**Bench.** `bench_fusion_ab.py --course technical --vision-pos-sigma 1.0`: legacy rows 2/12 timeout (honest — navigator-recovery gap in `gates_ned=None` config that S19u closed for production callers); fusion/clean 12/12 honest; fusion/mild 10/12 honest; fusion/harsh 3/12 honest (navigator completes physically; fused pose diverged). Distractor bench (S19n distractor scenarios) — baseline 2.42, on-path 3.94, off-path 2.42 — unchanged through the recent work.

**Belief yaw fix.** Replay test (`test_belief_replay.py`): fixed peak error 0.22 m vs. buggy peak 20.05 m through a 90° yaw + 2 s dropout. Offline A/B (`test_belief_yaw_ab.py`): clean/belief 3/3 at 2.4 % search, mild/belief 3/3 at 7.2 % search, harsh/belief 3/3 at 15.0 % search.

### 6.2 What works but is not yet measured at the right level

**Real YOLO integration.** `YoloPnpDetector` exists, the contract is tested against a stubbed model, the axis remap (OpenCV camera frame → body frame) is unit-tested. **Never run against a gate-trained `.pt` file.** PC/GPU-blocked.

**Real-image perception.** Synthetic VirtualCamera produces clean(-ish) detections that drive the entire 100% sandbox soak number. Real cameras under race lighting have not been exercised. This is the single largest unknown.

**PX4 SITL re-validation of the locked baseline.** The 1200-trial 100% claim is against `mock_kinematic` (timestep-independent kinematic physics with `auto_step=True`). PX4 SITL has wallclock pacing and motor/airframe dynamics that the kinematic mock approximates but does not reproduce. Re-running the soak against PX4 SITL at N ≥ 20 is a "first day on the PC" item.

**DCL physics.** Not measured at all — the SDK isn't out. The `MockDCLAdapter` provides a gym-style shape, and the seam contract tests (`test_dcl_adapter_seam.py`) define the behaviour the real adapter must implement, but they cannot anticipate the actual physics.

**Real Neros hardware.** Hardware in September. ESKF noise densities are placeholders (clean synthetic IMU); real-IMU recalibration is a hardware-blocked item. Vision-pos-sigma defaults are tuned against synthetic noise; will need re-tuning against real-camera output.

### 6.3 What does not yet exist

- A trained gate-specific YOLO model in the live stack (base `yolov8n-pose.pt` exists but detects humans).
- A distractor-augmented YOLO training pipeline (data-gen can ship pre-GPU; actual training is GPU-blocked).
- A real DCL integration (SDK pending).
- Centreline-crossing gate-pass detector (would remove `PASSAGE_RANGE=2.0 m` magic number; deferred under DCL-invalidation risk).
- Flight-replay visualiser (deferred — no current failure surface needs it; `StepResult` is already instrumented for it).

---

## 7. Strategy Going Forward

The strategic frame is two external gates: DCL SDK drop (~1–3 weeks) and PC + GPU access. Pre-DCL sandbox work has reached the point of diminishing returns — there are no remaining sandbox-runnable items whose marginal value exceeds the cost in time and complexity of doing them.

### 7.1 What moves the needle

In rough order of payoff, the remaining work that genuinely affects competition outcomes:

**1. Real YOLO end-to-end integration (PC/GPU-blocked, ~1 day on PC).** Wire `YoloPnpDetector` into the live stack against `mock_kinematic` with `--tag-identities`. The first run will surface whatever is broken about the trained weights' coordinate convention, FOV assumptions, or confidence calibration relative to what the race loop assumes. This is the single highest-information experiment available. Twenty minutes of wiring; the debugging afterwards is where the value lives.

**2. Distractor-augmented YOLO training (GPU-blocked, ~1–2 days on PC).** Round 2's environment has visual distractors (textures, reflections, structural elements that look like gates). A distractor-augmented training set is the biggest single lever on Round 2 robustness. Data-generation pipeline can ship pre-GPU; actual training requires the RTX hardware on the PC.

**3. PX4 SITL soak re-validation at N ≥ 20 (PC, 2–3 hours).** Corroborates or falsifies the 100 % sandbox number against real physics. If PX4 disagrees with `mock_kinematic`, the locked-baseline confidence is weaker than we currently believe — and we want to know that before DCL drops, not after.

**4. DCL day-1 integration when SDK drops.** `scripts/day1_dcl.sh --model <weights.pt>` is the mechanical sequence. Per-step forensic logs go to `dcl_day1_logs/`. Expected blockers: camera intrinsics extraction (PnP focal length / principal point), DCL adapter `NotImplementedError` methods filled in, real-image YOLO transfer.

### 7.2 What does not move the needle

The remaining sandbox gaps (centreline-crossing detector, flight-replay visualiser, bench `--backend` flag, ESKF tuning, tagger stress-test under injected pose noise) are either DCL-invalidation-risky, deferred-because-unneeded, or hardware-blocked. Working on them would be over-scaffolding — a pattern Conrad has consciously flagged. The right move is to stop sandbox work at this point.

### 7.3 Phased plan

| Phase | Trigger | Focus | Deliverable |
|---|---|---|---|
| Now → DCL drops | — | Stop sandbox work, conserve attention | This report; verified-stable scaffolding |
| PC access returns | Conrad swaps to Windows + GPU | Real-YOLO end-to-end + PX4 re-validation | First real-image race completion |
| DCL drops | SDK released | `scripts/day1_dcl.sh` end-to-end run | First DCL race completion |
| DCL → VQ1 (May–June) | First competition gate | Submit reliable VQ1 entry | Locked submission |
| VQ1 → VQ2 (June–July) | Round 2 environment | Distractor-augmented training; harden submission | Round 2 cutoff submission |
| VQ2 → Physical (Aug–Sept) | Hardware available | Sim-to-real transfer; ESKF retune | Physical qualifier entry |
| Physical → Champ (Oct–Nov) | Final tuning | Race-day procedures | Championship attempt |

---

## 8. Risk Register

### 8.1 Strategic risks (open)

**Real-image perception transfer.** The single largest open risk. Every robustness claim in this report is conditional on synthetic perception. Real cameras under race lighting have motion blur, exposure variation, lens distortion, rolling-shutter artifacts, and texture confusers that synthetic data does not reproduce. Mitigation: distractor-augmented training with real DCL camera captures, plus the `IdentityTagger` insurance layer for false-positive rejection. Residual: until we have a real race completion on real frames, this risk does not retire.

**DCL physics ≠ Neros physics.** DCL is a rehearsal environment, not the real test. Anything we tune against DCL may need re-tuning on hardware in September. Mitigation: the V5.1 baseline is locked against PX4 SITL physics, which is closer to Neros than to DCL — the locked baseline is partly insurance against this. Residual: we will not know how good the transfer is until physical qualifier in September; if it's bad, the calendar margin is tight.

**Solo-dev bandwidth.** A single late-stage bug can cost a week. Round 2 cutoff at end of July is the hardest deadline; physical qualifier in September is harder to recover from. Mitigation: extensive test/soak/bench infrastructure (124 tests in 64 s, 1200-trial soak in 30 s) catches regressions early; the locked baseline limits the surface area of changes that can break things. Residual: the ESKF + perception + DCL integrations all happen in compressed windows.

**Tagger over-rejection under real pose noise.** `IdentityTagger` defaults (`accept_radius_m=2.5`, `ambiguity_ratio=1.3`) were tuned against sim-truth pose. Real EKF2 / ESKF estimates carry meters of position error at altitude. If real-image back-projected detections fall outside the 2.5 m radius, the tagger reverts to `gate_idx=-1`, the picker reverts to nearest-first, and the cascade S19m diagnosed could fire in the real-YOLO regime. Mitigation: `--tag-accept-radius` CLI flag for tuning without code changes; characterisation against real YOLO + real pose is on the next-session priority list.

### 8.2 Strategic risks (closed)

- **S18 belief regression** — fixed in S19a-b (yaw propagation).
- **The 5 % soak failure rate mystery** — explained in S19w as a timeout artifact, not algorithmic.
- **Real-YOLO identity association** — structurally closed in S19x via `IdentityTagger`.
- **Multi-file `pytest` aggregate broken** — fixed in S19x via `conftest.py`.
- **DCL integration unknown-blockers** — surfaced and mitigated in S19u/t/v via MockDCL + smoke tests + day1 shell script.

### 8.3 Tactical risks

**ESKF tuning under harsh noise.** `honest_passes = 3/12` under fusion/harsh in the S19s bench — drone completes physically on detection bearing while fused pose diverges. Hardware-blocked: we need real Neros IMU characterisation before tuning is meaningful. Mitigation: the navigator's pose-trusted gate (S19p) reverts to belief-coast on rolling reject rate > 50 %, so divergence does not actively mis-steer.

**No PX4 SITL re-validation since locked baseline.** The 100 % sandbox claim is against `mock_kinematic`. PX4 SITL behaviour may differ; the difference would not be visible in the sandbox.

**Round 1 vs Round 2 perception split.** Round 1 has highlighted gates (the `ClassicalGateDetector` may suffice). Round 2 has a 3D-scanned environment — needs real YOLO + distractor-resistant training. The two paths are not the same; preparing only for Round 1 risks wasting June against a problem that doesn't exist in Round 2's regime.

---

## 9. What Could Go Right

A balanced report should also note where things could be better than they look:

- **The locked baseline is conservative.** V5.1 is well within physical limits; if real hardware proves tighter than PX4 SITL on bench, the planner can be re-tuned upward without changing the architecture.
- **The architecture has room for learned components.** A PPO-trained controller (MonoRace-style) or a learned state estimator could drop into the existing seams without re-architecting; the V5.1 baseline gives us a fallback if the learned system underperforms.
- **The scaffolding pays compound interest.** Every test the soak/bench/CI infrastructure catches is a session not spent debugging. The S19u bugs (caught by MockDCL smoke tests) were a concrete instance — those would have been a week of confused real-DCL debugging on day one.
- **The competition itself is unsolved.** No team has a known-winning submission; MonoRace 2025 is the closest reference and is on a different physical platform. Solo dev is a real disadvantage at the top of the field, but the gap is execution, not architecture.

---

## 10. Appendix

### 10.1 Documentation tree

| Path | Purpose |
|---|---|
| `docs/PROJECT_REPORT_2026-04-24.md` | This file. |
| `docs/STATUS_2026-04-24.md` | Terse current-state snapshot. |
| `docs/STATUS_2026-04-20.md` | Earlier snapshot (superseded). |
| `docs/PROJECT_LOG.md` | ~3100 lines of session-by-session record. |
| `docs/STRATEGY.md` | Competition strategy, timeline, architecture choices. |
| `docs/BATTLE_PLAN.md` | Module breakdown and tech stack. |
| `docs/COMPETITION_REVIEW_2026-04-20.md` | Round 1 / 2 findings from public sources. |
| `docs/PC_HANDOFF_2026-04-22.md` | Task list for next PC session. |
| `docs/DCL_INTEGRATION_CHECKLIST.md` | Mechanical day-1 DCL steps. |
| `docs/aigp_tech_spec_v0001.pdf` | VADR-TS-001 official tech spec. |
| `.auto-memory/MEMORY.md` | Memory index. |
| `.auto-memory/project_aigp_state.md` | Durable design rationale + gap list. |
| `.auto-memory/user_conrad.md` | Solo-dev context. |
| `.auto-memory/feedback_workflow.md` | Session-resume protocol. |
| `.auto-memory/reference_aigp_docs.md` | Source-file pointers. |

### 10.2 How to run

**Local dev (Mac, sandbox):**

```bash
# Smoke test
python run_race.py --backend mock --detector virtual --course technical

# With identity tagging
python run_race.py --backend mock_kinematic --detector virtual --course sprint --tag-identities

# With pose fusion
python run_race.py --backend mock_kinematic --detector virtual --fusion --vision-pos-sigma 1.0

# Sandbox soak
python soak.py --n 100 --courses technical,mixed,sprint --noises clean,mild,harsh,brutal --timeout 60

# All tests
pytest
```

**On PC (when access returns):**

```bash
# Real-YOLO smoke (requires gate-trained .pt)
python run_race.py --backend mock_kinematic --detector yolo_pnp \
                   --model-path src/vision/gate_yolo/models/<weights>.pt \
                   --tag-identities

# PX4 SITL re-validation
python run_race.py --backend px4_sitl --detector virtual --course technical
```

**On DCL drop day:**

```bash
scripts/day1_dcl.sh --model src/vision/gate_yolo/models/<weights>.pt
```

### 10.3 Test inventory

124/124 under naked `pytest` (~64 s). Suites that gate behaviour: `test_gate_belief`, `test_belief_replay`, `test_belief_nav_gate_aware`, `test_race_loop*`, `test_dcl_adapter_seam`, `test_dcl_smoke`, `test_identity_tagger`. Suites that test seams or contracts: `test_sim_adapter`, `test_sim_imu`, `test_pose_fusion`, `test_eskf`, `test_detector`, `test_race_runner`, `test_vision_roundtrip`, `test_gate_sequencer`. Re-derivable inventory: `pytest --collect-only -q` from repo root.

### 10.4 Glossary

- **VADR-TS-001** — official Anduril AI Grand Prix technical spec, issued March 2026.
- **DCL** — Drone Competition League simulator; the official race environment, expected to drop in May 2026.
- **VQ1 / VQ2** — Virtual Qualifier rounds 1 and 2.
- **NED** — North-East-Down coordinate frame (sim convention).
- **PnP** — Perspective-n-Point. Recovers 3D pose from 2D image-plane keypoints.
- **ESKF** — Error-State Kalman Filter (15-DOF: position, velocity, attitude error, gyro bias, accel bias).
- **MAVLink / MAVSDK** — Drone communication protocol and Python SDK.
- **PX4 SITL** — PX4 flight controller running in software-in-the-loop simulation.
- **S19a, S19b, ..., S19x-j** — internal session labels (one calendar block of focused work each, often 1–4 hours).
- **V5.1** — locked phase-based velocity planner (LAUNCH / SUSTAIN / PRE_TURN / TURN / SHORT phases with multiplicative `transition_blend`).
- **`gate_idx`** — gate identity label; `-1` means "real vision saw a gate but doesn't know which".
- **`pose_trusted`** — boolean gate driving the navigator's gate-aware fallback; True when the rolling vision-reject rate is below 50 %.
- **`honest_passes`** — count of gate-pass events where the fused pose is within 2 m of truth at the `target_idx`-advance tick.

---

*Report written 2026-04-24 by Claude (Sonnet 4.7) operating in Cowork mode against the live working directory. Validity is conditional on the sandbox state at write time; subsequent sessions, code changes, or environment shifts may invalidate specific claims. The structural narrative is durable; specific numbers (test counts, soak rates, bench output, line counts) are point-in-time and should be re-measured before being cited as fact.*
