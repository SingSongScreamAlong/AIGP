# DCL Simulator Integration Checklist

**Purpose:** Day-1-after-DCL-SDK-drops checklist. The DCL AI Race League Python API is expected ~May 2026; until it lands, everything here is what can be pre-staged. When the SDK arrives, this doc becomes a punch list.

**Status as of 2026-04-21 (post-S19u):** Adapter seam exists (`src/sim/adapter.py::DCLSimAdapter`). 8 methods raise `NotImplementedError` with structured hints. 6 methods are no-ops (arm/takeoff/land/offboard) because gym-style envs usually don't model those. Capability flags are a best guess: `VELOCITY_NED | POSITION_NED | CAMERA_RGB | IMU | RESET`. Pre-landing scaffolding complete: `MockDCLAdapter` + `test_dcl_smoke.py` + `scripts/day1_dcl.sh`.

---

## 1. What the race stack actually calls per tick

The hot path through `RaceLoop.step()` (see `src/race_loop.py`):

1. `await adapter.get_state() → SimState` — drone pose for navigator + gate-pass detector
2. `await adapter.get_camera_frame() → Optional[np.ndarray]` — HxWx3 uint8, fed to detector
3. `await adapter.get_imu() → Optional[IMUReading]` — **fusion path only** (skipped when `pose_fusion=None`)
4. `await adapter.send_velocity_ned(vn, ve, vd, yaw_deg)` — planner output

Everything else is cold path (connect / lifecycle / offboard / land) and fires at most a handful of times per flight.

## 2. Minimum viable DCL adapter for first race attempt

**Must-have before first `python3 run_race.py --backend dcl` completes:**

| Method | Hot? | Why we need it | Notes |
|---|---|---|---|
| `connect()` | cold | Create env; RaceRunner calls this first | Gym-style `env = dcl.make(...)` pattern expected |
| `disconnect()` | cold | Clean shutdown; called in `finally` | Likely just `env.close()` |
| `get_state()` | **hot** | Navigator + gate-pass detector consume pose/vel/yaw | Map obs['pose']/obs['twist'] → SimState |
| `get_camera_frame()` | **hot** | Vision detector needs frames | HxWx3 uint8. Confirm BGR vs RGB on day 1 |
| `send_velocity_ned()` | **hot** | V5.1 planner emits velocity commands | Race will not move without this |

**Nice-to-have for full feature parity:**

| Method | Hot? | Why | Notes |
|---|---|---|---|
| `get_imu()` | **hot** (fusion) | `--fusion` flag requires it | Confirm body-FRD + SI units; convert if not |
| `reset()` | cold | Between-race isolation; useful for bench harnesses | Gym `env.reset(seed=...)` |
| `send_position_ned()` | cold | Hover / precision mode (not on hot path in V5.1) | Low priority |
| `send_attitude()` | cold | May not be supported by DCL at all | If unsupported, strip `SimCapability.ATTITUDE` |

**Can stay no-op:**
- `arm() / disarm() / takeoff() / land() / start_offboard() / stop_offboard()` — gym envs usually model the drone as already flying. Leave as `pass`, since `RaceRunner` already gates arm/takeoff/land on `SimCapability.ARM_ACTION`. Do NOT add `ARM_ACTION` to DCL capabilities unless the SDK actually exposes these.

## 3. Day-1 checklist when the SDK drops

**TL;DR:** `scripts/day1_dcl.sh [--model PATH]` runs the validation sequence below and logs each step to `dcl_day1_logs/<timestamp>/`. Manual version follows — top to bottom, each item a specific, checkable step.

- [ ] **Read the SDK docs for the constructor shape.** Gym-style `dcl.make(scenario=...)` vs something stateful vs something else. Adjust `DCLSimAdapter.__init__` to match.
- [ ] **Wire `connect()`.** If the SDK is sync, wrap in `asyncio.to_thread(...)` so the async contract holds. Store the env handle on `self._env`.
- [ ] **Wire `get_state()`.** Identify the obs keys for position, velocity, attitude. Double-check frame: NED world? ENU? Body? The race stack assumes NED world for position/velocity, radians for attitude. Convert at the adapter boundary.
- [ ] **Wire `get_camera_frame()`.** Check shape (HxWx3), dtype (uint8 expected), color order (BGR vs RGB). Document whatever DCL ships and convert here so the detector stays invariant. If the SDK returns a torch/jax tensor, `.cpu().numpy().astype(np.uint8)` it.
- [ ] **Wire `send_velocity_ned()`.** Confirm sign conventions (DCL may use ENU or FLU body internally; adapter's job is to present NED to the race stack). Confirm yaw is degrees on the wire.
- [ ] **Run `python3 run_race.py --backend dcl --detector virtual --course technical --timeout 30`.** Virtual detector works without a camera — it isolates the adapter. If this completes, the state/command seam is good.
- [ ] **Run `python3 run_race.py --backend dcl --detector yolo_pnp --course technical --model-path ... --timeout 30`.** Adds the camera seam. Inspect a frame programmatically (`adapter.get_camera_frame()` in a sandbox script) before running the whole race.
- [ ] **If enabling fusion: wire `get_imu()`.** Verify body-FRD convention (accel measures specific force in m/s², gravity-loaded; at rest level → [0, 0, -9.81]). Convert gyro from deg/s if needed. Then run `python3 run_race.py --backend dcl --detector yolo_pnp --fusion --vision-pos-sigma 1.0 ...`.
- [ ] **Validate with the `bench_fusion_ab.py` harness** on the DCL backend once the above CLI runs complete. Compare pass counts, honest_passes, and pose error to the mock_kinematic numbers as a sanity check.

## 4. Open questions — fill in on day 1

These cannot be answered without the SDK. Track them here so they're not forgotten.

- **Coordinate frame conventions.** NED? ENU? Body-FLU? Attitude as Euler / quat / rotation matrix? Yaw sign convention?
- **Camera format.** RGB vs BGR. Intrinsics — does the SDK publish them, or are they fixed per scenario?
- **IMU rate and format.** Hz, accel units (m/s² vs g), gyro units (rad/s vs deg/s), bias behaviour.
- **Tick rate.** Gym envs often advance one physics step per `step()` call. Does DCL? Or is it wallclock-paced? `SimCapability.WALLCLOCK_PACED` setting depends on the answer.
- **Reset semantics.** Does `reset(seed=...)` deterministically re-generate the course, or just reset the drone?
- **Arm/takeoff model.** Does DCL start the drone airborne, or does it model a ground state? If the latter, the arm/takeoff no-ops need real implementations, and `ARM_ACTION` capability should be added.

## 5. Anticipated integration surprises

From reading the adapter + runner + race_loop code:

1. **Async wrapping of sync APIs.** If DCL's `env.step()` is synchronous (most gym envs are), every method needs `await asyncio.to_thread(env.step, ...)` to avoid blocking the async race loop. This is fine but easy to forget and will show up as "race feels slow / command_hz is missed" if not done.
2. **Coordinate frame flips at the adapter boundary.** The race stack assumes NED consistently. If DCL uses ENU or something else internally, the adapter must convert on both telemetry-out (`get_state`) and command-in (`send_velocity_ned`). A single sign flip in one direction is catastrophic and silent.
3. **Camera BGR vs RGB.** YOLO was trained on whatever colour order the training data was in; document on day 1 and convert at the adapter boundary. If it's wrong the detector will still produce boxes but they'll be wrong colour.
4. **IMU specific-force vs raw acceleration.** `IMUReading.accel_body` is specific force (gravity-loaded — level at rest reads ≈ [0, 0, -9.81]). If DCL publishes gravity-subtracted acceleration instead, the ESKF will believe the drone is in free fall and diverge immediately. Check with a stationary-drone sanity read first thing.
5. **`get_state()` called before first `step()`.** The race loop's `get_state()` fires on every tick including tick 0. If DCL requires a `step()` before observations are valid, `connect()` must do an initial no-op step or the first tick will fail.
6. **`command_hz` matches DCL's tick rate.** Default is 50 Hz. If DCL's physics is 120 Hz (matching the spec), the race loop should still work since `send_velocity_ned` is rate-limited at the loop layer, but if DCL is e.g. 30 Hz and the loop fires 50 Hz, commands get dropped silently. Worth logging the first few tick timings to confirm.

## 6. Pre-landing work that can start now

**Without the SDK, already doable:**

- [x] Adapter stub exists with structured hints.
- [x] `MockDCLAdapter` (`src/sim/mock_dcl.py`) mimics DCL's expected capability shape (no ARM_ACTION, no WALLCLOCK_PACED, CAMERA_RGB + IMU + RESET) backed by `MockKinematicAdapter` physics. Wired into `run_race.py` as `--backend mock_dcl`. Landed S19t; first smoke test surfaced two real bugs (S19u) — see `docs/PROJECT_LOG.md` S19u for details.
- [x] `test_dcl_adapter_seam.py` — 6 contract tests ensuring `DCLSimAdapter` stub cannot drift into no-ops. **When the SDK lands, most of these tests invert.**
- [x] `test_dcl_smoke.py` — end-to-end CLI smoke test running `--backend mock_dcl` + `mock` + `mock_kinematic+fusion` and asserting 12/12 gate completion. Catches the class of bugs S19u found (reset-kwarg-drop, gates_ned handling). Runs in ~35 s.
- [x] `scripts/day1_dcl.sh` — executable day-1 sequence. Runs the smoke test as a baseline, then the three `--backend dcl` validation commands, logging each step to `dcl_day1_logs/<utc-stamp>/` for forensic comparison.

## 7. What lives where

- Adapter seam: `src/sim/adapter.py::DCLSimAdapter`
- Capability flag checks: `src/race/runner.py::fly` (lines ~120, ~133)
- Hot-path calls: `src/race_loop.py::step` (see `grep "self.adapter\."`)
- CLI wiring: `run_race.py::build_adapter` (the `backend in ("px4_sitl", "dcl")` branch)
- Fusion-bench harness: `bench_fusion_ab.py` — run against DCL as soon as fusion path is wired

---

**This doc is a plan, not a commitment.** The real DCL SDK will expose shapes we haven't predicted, and some of the anticipated surprises may not be surprises at all. The value of this checklist is reducing day-1 thinking to day-1 *doing*.
