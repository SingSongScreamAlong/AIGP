"""RaceLoop — the end-to-end integration path — Session 19.

Ties together:
    SimAdapter    (state + optional camera frame + command sink)
    Detector      (frame + state → List[GateDetection])
    BeliefNav     (tracker state → velocity command)
    PoseFusion    (optional — IMU + vision fusion; Session 19h)

Runs at command_hz (default 50). Sends velocity-NED or attitude
commands through the adapter every tick, depending on adapter
capabilities.

Design goals:
  * Swappable backends. The loop is the same whether the adapter is
    PX4 SITL + VirtualDetector (today) or DCL + YoloPnpDetector (May).
  * Capability-aware. If the adapter reports no CAMERA_RGB, we pass
    frame=None to the detector (VirtualDetector ignores it, YOLO
    detector returns empty — the belief model correctly falls into
    COAST/SEARCH).
  * Gate-pass detection has two modes:
      - range-based (legacy default): detected range crossed below
        PASSAGE_RANGE. Used when we have no NED gate positions, which
        is the bare-minimum skeleton path.
      - position-based (preferred when gates_ned is supplied): fires
        on the local-minimum of ‖drone_ned − gate_ned‖ while within
        PASSAGE_RADIUS, OR'd with the legacy range signal so timing
        matches the pre-S19k detector on clean runs. Robust to
        detection dropouts at the gate. Rejects decoys far from the
        drone's physical path (can't hit a position local-minimum
        we never approached), but offers NO defense against decoys
        placed near the target gate on the drone's flight path —
        that's detector-layer (YOLO-training) work.

        The range signal carries a PASSAGE_REFRACTORY block after
        each pass (S19n): blocks re-firing on the just-passed gate's
        detection until the drone is geometrically past it. Without
        this, `associate_mode="nearest"` (the real-YOLO threat model)
        cascades through every subsequent target within a few ticks
        of reaching the first gate, even with zero decoys.

        See _check_gate_pass_position docstring and the S19l/S19m
        project-log entries for the ruled-out sanity-gate experiment
        and the cascade finding that motivated the refractory.
  * Structured result types so tests and analytics can observe a run
    without grepping print statements.
  * Optional ESKF/PoseFusion integration. When `pose_fusion` is supplied,
    the loop feeds IMU + backprojected vision fixes into the filter and
    uses the fused pose (instead of `adapter.get_state()`) to drive the
    navigator. Gated by `SimCapability.IMU`; unchanged when pose_fusion
    is None.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from sim.adapter import SimCapability
from control.attitude_controller import AttitudeController

# We import from sim.adapter at type-check time; at runtime the loop
# only touches duck-typed state / cmd fields, so we don't depend on
# mavsdk being installed here.


# ─────────────────────────────────────────────────────────────────────
# TrackerState — the shape BeliefNav.plan wants
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrackerState:
    """What the navigator consumes. Matches vision_nav.TrackerState fields
    that BeliefNav.plan actually reads."""
    target_idx: int
    detected: bool
    bearing_h: float        # rad (body-frame horizontal bearing)
    bearing_v: float        # rad (body-frame vertical bearing)
    range_est: float        # m
    confidence: float
    frames_since_seen: int
    search_mode: bool


# ─────────────────────────────────────────────────────────────────────
# Per-step and per-run result records
# ─────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """One tick of the loop — enough to log or replay.

    `pos_ned` and `yaw_deg` reflect the pose the *navigator consumed*
    this tick (fused pose when `pose_fusion` is on, adapter truth
    otherwise). Added S19w so soak logs can reconstruct the drone's
    trajectory for failure-mode categorisation and replay; default is
    None so the field is optional in test fixtures that pre-date it.
    """
    t: float
    target_idx: int
    detected: bool
    range_est: float
    cmd_vn: float
    cmd_ve: float
    cmd_vd: float
    cmd_yaw_deg: float
    passed_gate: bool = False
    pos_ned: Optional[Tuple[float, float, float]] = None
    yaw_deg: Optional[float] = None


@dataclass
class RunResult:
    gates_passed: int
    gate_count: int
    total_time_s: float
    timeout: bool
    steps: List[StepResult] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.gates_passed >= self.gate_count and not self.timeout


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────

class RaceLoop:
    """One race attempt. Construct, call run(), read RunResult."""

    PASSAGE_RANGE = 2.5            # m — gate considered passed when range < this
    PASSAGE_RADIUS = 2.0           # m — position-based: local minimum of |drone-gate|
    # m — after a pass fires, block the range signal until drone has moved
    # this far from where it was when the pass fired (drone displacement,
    # not distance to gate). Guards against the S19m baseline cascade:
    # under associate_mode="nearest" (the honest real-YOLO threat model,
    # since YOLO tags gate_idx=-1), the just-passed gate's detection
    # stays within PASSAGE_RANGE for several ticks and the range signal
    # re-fires for each successive target_idx unchecked.
    #
    # Drone-displacement reference (rather than gate-position reference)
    # is critical for the on-path-decoy case: a decoy 1.5 m in front of
    # gate K fires the range signal early at drone position (gate_K - 2.5);
    # drone is then ~3 m from gate K, which a gate-position-based
    # refractory would consider already-cleared. Drone-displacement
    # forces the refractory to last until the drone has physically
    # carried itself past the spurious detection.
    #
    # Sized at 2 × PASSAGE_RANGE = 5.0 m — wide enough to traverse the
    # baseline cascade window, narrow enough that on technical courses
    # (gate spacing 5.7–8 m) the next legitimate range fire is not
    # delayed beyond when the drone reaches gate K+1's range bubble.
    # Position signal is the always-on backup if range is gated past a
    # legitimate fire (position fires inside PASSAGE_RADIUS=2 m of the
    # NEW target).
    #
    # Applied to the range signal only — the position signal is not
    # susceptible to the cascade (requires drone within PASSAGE_RADIUS
    # of the NEW target gate). See S19m PROJECT_LOG entry.
    PASSAGE_REFRACTORY = 5.0
    SEARCH_TIMEOUT_FRAMES = 50     # frames without detection before search mode
    COMMAND_HZ = 50
    # S19p pose-trust threshold for the gate-aware BeliefNav fallback.
    # When the rolling rejection rate on PoseFusion vision fixes exceeds
    # this, the navigator's `pose_trusted` flag is cleared and the gate-
    # aware path falls through to belief-coast/search. 0.5 means "half
    # the recent fixes are being rejected by the chi-squared gate" —
    # that's a clear signal the filter is in the self-destructive
    # divergence documented in bench_fusion_ab.py. Tuned against
    # fusion/harsh on technical (192 rej vs 166 ok ⇒ 0.54). Below the
    # threshold, gate-aware path stays on because clean/mild fusion
    # benches reject fewer than 2% of fixes.
    POSE_TRUST_REJECT_RATE = 0.5

    def __init__(
        self,
        adapter,
        detector,
        navigator,
        gate_count: int,
        command_hz: int = COMMAND_HZ,
        associate_mode: str = "nearest",
        pose_fusion=None,
        gates_ned: Optional[Sequence[Tuple[float, float, float]]] = None,
        vision_pos_sigma: float = 0.15,
        gate_sequencer=None,
        attitude_controller: Optional[AttitudeController] = None,
    ):
        """
        Args:
            adapter: Anything implementing SimAdapter (get_state,
                get_camera_frame, send_velocity_ned, capabilities).
            detector: Anything implementing Detector (detect,
                name). Emits GateDetection list in body frame.
            navigator: BeliefNav-compatible object with
                .plan(tracker_state, pos, vel, yaw_deg, dt) returning
                something with .north_m_s/.east_m_s/.down_m_s/.yaw_deg.
            gate_count: expected number of gates in the race.
            command_hz: command rate (default 50).
            associate_mode: "nearest" picks the nearest detection;
                "target_idx" prefers a detection tagged with the
                current target (VirtualDetector only).
            pose_fusion: Optional PoseFusion instance. When supplied,
                the loop drives it with adapter.get_imu() samples and
                backprojected vision fixes, and feeds its fused pose
                into the navigator. When None (default), navigator
                consumes adapter.get_state() pose directly.
            gates_ned: Gate NED positions (N, E, D). Required when
                pose_fusion is supplied — used to backproject body-frame
                detections into drone-frame world-position fixes. May
                also be supplied WITHOUT pose_fusion to enable the
                position-based gate-pass detector (uses adapter truth
                pose in that case).

                **Omitting gates_ned (`gates_ned=None`) is a minimal-
                integration configuration, not a production target.**
                The S19r bench (bench_fusion_ab.py legacy rows) surfaces
                a navigator-recovery gap on this path: when the tracker
                locks onto a just-passed gate's detection and the picker
                has no navigator-side fallback, BeliefNav steers
                *toward* that detection — which is now spatially
                behind the drone — with no world-frame reference to
                recover to. Closing that gap requires upgrading
                BeliefNav's search mode (yaw-sweep + more aggressive
                forward drift on N ticks without target_idx progress);
                no production caller needs this today because
                `run_race.py`, RaceRunner tests, and fusion bench rows
                all supply `gates_ned`. A `UserWarning` is emitted at
                construction if you opt into the minimal path, so test
                fixtures that silence warnings don't mask accidental
                production misuse.
            vision_pos_sigma: 1-σ noise on the backprojected position
                fix (meters). Default 0.15 m is only appropriate for
                *clean* synthetic projection chains on short, dense
                courses. On longer courses or noisier vision profiles,
                the chi-squared gate in ESKF.update_vision becomes
                self-destructive: once drift accumulates, all fixes
                look like outliers and get rejected, so drift compounds.
                S19j bench showed 0.15 → 122 m pose error on sprint/mild,
                while 1.0 keeps it under 1.1 m. Rule of thumb:
                err_pos ≈ bearing_sigma_rad × typical_range + range_sigma.
                See bench_fusion_ab.py for the regression harness.
        """
        self.adapter = adapter
        self.detector = detector
        self.navigator = navigator
        self.gate_count = gate_count
        self.dt = 1.0 / command_hz
        self.associate_mode = associate_mode

        # Attitude controller -----------------------------------------
        # When the adapter supports ATTITUDE but not VELOCITY_NED (e.g.
        # DCL), we need to translate velocity commands into T/R/P/Y.
        # Auto-create a default AttitudeController if none was supplied
        # and the adapter has the ATTITUDE capability.
        if attitude_controller is not None:
            self._attitude_ctrl = attitude_controller
        elif (hasattr(adapter, 'capabilities')
              and SimCapability.ATTITUDE in adapter.capabilities
              and SimCapability.VELOCITY_NED not in adapter.capabilities):
            self._attitude_ctrl = AttitudeController()
        else:
            self._attitude_ctrl = None

        # Fusion plumbing ----------------------------------------------
        self.pose_fusion = pose_fusion
        self._gates_ned = None
        if pose_fusion is not None and gates_ned is None:
            raise ValueError(
                "pose_fusion requires gates_ned to backproject "
                "detections into NED world-position fixes."
            )
        if gates_ned is not None:
            import numpy as _np
            self._gates_ned = [_np.asarray(g, dtype=float) for g in gates_ned]
        else:
            # S19s — legacy minimal-integration path. Navigator cannot
            # recover from stuck-detection lock (see constructor
            # docstring for the full rationale and the S19r bench).
            # UserWarning so it shows up with default filters but can
            # be silenced by tests that intentionally exercise the
            # path. `stacklevel=2` points at the caller, not this line.
            # S20: suppress warning when GateSequencer is active —
            # gates_ned=None is intentional in discovery mode.
            if gate_sequencer is None:
                import warnings
                warnings.warn(
                    "RaceLoop constructed without gates_ned — this is a "
                    "minimal-integration configuration only. BeliefNav "
                    "cannot recover from stuck-detection lock without "
                    "world-frame gate positions. Supply `gates_ned` for "
                    "any production use.",
                    UserWarning,
                    stacklevel=2,
                )
        self._vision_pos_sigma = float(vision_pos_sigma)

        # S20 track-agnostic gate sequencer. When supplied, detection
        # filtering and gate-pass detection are delegated to the sequencer
        # instead of the built-in target_idx + gates_ned logic. The
        # sequencer discovers gates from vision alone, with no a priori
        # knowledge of the course layout.
        self.gate_sequencer = gate_sequencer

        # Hand gate positions to the navigator so it can turn toward the
        # current target when no detection is available (S19o gate-aware
        # fallback). Legacy navigators without this method keep their old
        # belief-coast/search behaviour.
        if self._gates_ned is not None and hasattr(navigator, "set_gates_ned"):
            navigator.set_gates_ned(self._gates_ned)

        # Running state
        self.target_idx = 0
        self.frames_since_seen = 0
        self.last_range: Optional[float] = None
        self._last_gate_dist: Optional[float] = None
        # Drone NED at the moment the most-recent pass fired. When
        # non-None, the range signal is refractory-blocked until the
        # drone has moved PASSAGE_REFRACTORY from this reference.
        # Using drone-displacement (not gate-position) is deliberate:
        # see PASSAGE_REFRACTORY class-level comment and S19m/S19n
        # PROJECT_LOG.
        self._drone_ned_at_last_pass = None
        self._start_time: Optional[float] = None

    # ── Detection → TrackerState ────────────────────────────────
    def _pick_detection(self, detections):
        """Associate raw detections to the current target gate.

        Defaults to nearest-first. With 'target_idx' mode:
          1. Prefer a detection whose gate_idx matches the current target.
          2. **Only when a navigator-side fallback exists** (we have
             `gates_ned` AND the navigator implements `set_gates_ned`):
             skip already-passed gates in the nearest-first fallback.
             That avoids re-anchoring on the just-passed gate still in
             FOV — the cascade that made the pre-S19n fusion bench look
             perfect while the drone physically drove off course. Safe
             here because the navigator can fly toward the next target
             from `gates_ned[target_idx]` even with no detection.
          3. Without that fallback (legacy non-fusion runs with no
             gates_ned, or a navigator that doesn't accept gates_ned),
             fall back to nearest-first regardless of gate_idx — the
             post-pass refractory still blocks fake range-passes, so
             cascades don't fire even if the picker returns the
             just-passed gate. The drone will wobble back briefly,
             then re-acquire the real target as it comes into FOV.

        Only VirtualDetector tags gate_idx today; real-vision paths
        carry gate_idx=-1 and keep the legacy nearest-first behaviour.
        """
        if not detections:
            return None
        if self.associate_mode == "target_idx":
            for d in detections:
                if getattr(d, "gate_idx", -1) == self.target_idx:
                    return d
            if self._has_nav_fallback():
                for d in detections:
                    gi = getattr(d, "gate_idx", -1)
                    if gi < 0 or gi >= self.target_idx:
                        return d
                return None
        return detections[0]  # nearest-first already ensured by detectors

    def _has_nav_fallback(self) -> bool:
        """True when the navigator can plan without a detection.

        S19o: gate-aware BeliefNav can fly toward `gates_ned[target_idx]`
        when set. Without that, returning None from `_pick_detection`
        would silently put the drone into a blind coast/search loop —
        worse than the just-passed-gate cascade we'd be guarding against.
        """
        return (
            self._gates_ned is not None
            and hasattr(self.navigator, "gates_ned")
            and getattr(self.navigator, "gates_ned", None) is not None
        )

    def _build_tracker_state(self, detections) -> TrackerState:
        picked = self._pick_detection(detections)
        if picked is None:
            self.frames_since_seen += 1
            return TrackerState(
                target_idx=self.target_idx,
                detected=False,
                bearing_h=0.0,
                bearing_v=0.0,
                range_est=self.last_range or 20.0,
                confidence=0.0,
                frames_since_seen=self.frames_since_seen,
                search_mode=self.frames_since_seen >= self.SEARCH_TIMEOUT_FRAMES,
            )

        self.frames_since_seen = 0
        self.last_range = picked.range_est
        return TrackerState(
            target_idx=self.target_idx,
            detected=True,
            bearing_h=math.radians(picked.bearing_h_deg),
            bearing_v=math.radians(picked.bearing_v_deg),
            range_est=picked.range_est,
            confidence=picked.confidence,
            frames_since_seen=0,
            search_mode=False,
        )

    # ── Gate-pass heuristic ─────────────────────────────────────
    def _advance_target(self) -> None:
        """Shared bookkeeping when the current gate is considered passed."""
        self.target_idx += 1
        self.frames_since_seen = 0
        self.last_range = None
        self._last_gate_dist = None
        # Tell the navigator to reset its belief for the new target
        if hasattr(self.navigator, "on_gate_passed"):
            try:
                self.navigator.on_gate_passed()
            except TypeError:
                # Older signatures took a speed arg
                self.navigator.on_gate_passed(0.0)

    def _check_gate_pass(
        self,
        tracker: TrackerState,
        drone_ned: Optional[Sequence[float]] = None,
    ) -> bool:
        """Range-based gate pass. Detected range crossed below
        PASSAGE_RANGE. This is the skeleton default used when no
        NED gate positions are available; swap to the position-based
        detector when gates_ned is supplied.

        **S19r refractory (legacy path):** when `drone_ned` is supplied,
        the range signal is gated on the same drone-displacement
        refractory as `_check_gate_pass_position` — after a pass fires,
        the next range-cross is blocked until the drone has moved
        PASSAGE_REFRACTORY from the pass-firing position. This closes
        the same baseline cascade S19n fixed in the position path:
        without the refractory, nearest-first association can keep
        picking the just-passed gate's detection for several ticks,
        and the range signal re-fires for each new target_idx, racing
        target_idx through the remaining gates while the drone barely
        moves.

        `drone_ned=None` preserves the pre-S19r behaviour (for callers
        that truly have no pose available — e.g. a minimal unit test
        that pokes the range signal in isolation). Production callers
        (`step()`) always pass the navigator's pose source.
        """
        if self.target_idx >= self.gate_count:
            return False

        # S19r — drone-displacement refractory on the range signal.
        # Mirrors `_check_gate_pass_position`'s refractory arm; same
        # self-healing property (any genuine drone motion gradually
        # clears it, pose drift cannot trap us here because target_idx
        # advances on each firing).
        if drone_ned is not None and self._drone_ned_at_last_pass is not None:
            import numpy as np
            drone_arr = np.asarray(drone_ned, dtype=float)
            disp_since_pass = float(np.linalg.norm(
                drone_arr - self._drone_ned_at_last_pass
            ))
            if disp_since_pass < self.PASSAGE_REFRACTORY:
                return False
            # Past the refractory — clear the reference so it doesn't
            # influence subsequent firings.
            self._drone_ned_at_last_pass = None

        if tracker.detected and tracker.range_est < self.PASSAGE_RANGE:
            if drone_ned is not None:
                import numpy as np
                self._drone_ned_at_last_pass = np.asarray(
                    drone_ned, dtype=float
                ).copy()
            self._advance_target()
            return True
        return False

    def _check_gate_pass_position(
        self,
        tracker: TrackerState,
        drone_ned: Sequence[float],
    ) -> bool:
        """Combined range- and position-based gate pass, used whenever
        we have `gates_ned`. Two signals, OR'd:

          1. **Range signal (legacy timing)**: detected range <
             PASSAGE_RANGE. Fires on the same tick as the pre-S19k
             detector did. Retained *without* a position sanity gate
             after a failed S19l experiment: AND-gating range with
             a sanity radius permanently stalls the race once fused
             pose drifts past the radius, because the drone orbits
             the target gate (which is physically close but fused-
             NED-far), which starves fusion of vision updates, which
             compounds the drift. Chicken-and-egg.

             **S19n refractory**: after a pass fires, the range signal
             is blocked until drone has moved PASSAGE_REFRACTORY
             beyond the just-passed gate. Guards against the S19m
             baseline cascade (nearest-first association keeps
             picking the just-passed gate's detection for several
             ticks; range signal re-fires for each new target_idx
             unchecked). The refractory blocks SPURIOUS passes — not
             required ones, unlike S19l's sanity-gate — so it's
             fusion-drift-safe: by the time it kicks in, target_idx
             and the navigator have already advanced past the gate,
             so drift recovers naturally via vision updates on the
             next target instead of feeding back into the loop.

          2. **Position local-minimum**: ‖drone_ned − gate‖ hits a
             local minimum while within PASSAGE_RADIUS. Catches
             detection dropouts at close range (YOLO boxes collapse
             when the gate fills frame); and because it's gated on
             actual drone proximity to the target gate's known NED,
             cannot fire on decoys the drone never physically
             approached. Position signal is NOT refractory-blocked:
             it already requires drone within PASSAGE_RADIUS of the
             NEW target, so it cannot cascade on the just-passed
             gate.

        **Distractor defense is best-effort and lives at the detector
        layer, not here.** Signal 1 is spoofable by a decoy at short
        camera range. Signal 2 is NOT spoofable by a decoy the drone
        never physically approaches, but IS spoofable by a decoy
        placed on the drone's flight path near the target gate. The
        canonical Round 2 defense is distractor-augmented YOLO
        training (detector layer), with this loop's position signal
        as a defense against the specific "decoy far from drone path"
        attack vector. See the S19l and S19m project-log entries for
        the experiments that ruled out range-signal sanity gating and
        the cascade that the S19n refractory addresses.

        Uses whichever pose is canonical for the loop (fused when
        fusion is on, adapter truth otherwise) — supplied by the
        caller via `drone_ned`.
        """
        if self.target_idx >= self.gate_count:
            return False
        import numpy as np
        gate = self._gates_ned[self.target_idx]
        drone_arr = np.asarray(drone_ned, dtype=float)
        dist = float(np.linalg.norm(drone_arr - gate))
        prev = self._last_gate_dist
        self._last_gate_dist = dist

        # S19n refractory — has the drone moved far enough since the
        # last pass fired? Drone-displacement reference is robust to
        # where the spurious detection actually was (decoy in front of
        # gate, just-passed gate behind, etc.). Evaluated every tick
        # so the check is self-healing: pose drift cannot trap us in
        # refractory because target_idx has already advanced and any
        # genuine drone motion gradually clears the refractory.
        range_refractory = False
        if self._drone_ned_at_last_pass is not None:
            disp_since_pass = float(np.linalg.norm(
                drone_arr - self._drone_ned_at_last_pass
            ))
            if disp_since_pass < self.PASSAGE_REFRACTORY:
                range_refractory = True
            else:
                self._drone_ned_at_last_pass = None

        # Signal 1 — range-based, refractory-gated.
        range_fires = (
            not range_refractory
            and tracker.detected
            and tracker.range_est < self.PASSAGE_RANGE
        )

        # Signal 2 — position local-minimum inside PASSAGE_RADIUS.
        # Not refractory-gated (see class docstring).
        position_fires = (
            prev is not None
            and prev < self.PASSAGE_RADIUS
            and dist > prev
        )

        if range_fires or position_fires:
            # Snapshot drone position BEFORE _advance_target so the
            # refractory references where we were when the pass fired,
            # not where we are after subsequent ticks.
            self._drone_ned_at_last_pass = drone_arr.copy()
            self._advance_target()
            return True
        return False

    # ── Fusion helpers ──────────────────────────────────────────
    async def _ingest_imu_async(self) -> None:
        """Drive PoseFusion with an IMU sample, if available. Duck-
        bridges `sim.adapter.IMUReading` → `pose_fusion.IMUSample` since
        the field shapes are identical."""
        # Deferred import — keep race_loop standalone of estimation.
        from estimation import IMUSample
        reading = await self.adapter.get_imu()
        if reading is None:
            return
        self.pose_fusion.on_imu(IMUSample(
            accel_body=reading.accel_body,
            gyro_body=reading.gyro_body,
            timestamp=reading.timestamp,
        ))

    def _ingest_vision_from_detection(self, picked, fused_yaw: float) -> None:
        """Backproject a body-frame detection of the picked gate into a
        drone-NED position fix, and feed it to PoseFusion.

        Anchors on ``picked.gate_idx`` — NOT ``self.target_idx`` — when
        the detection carries a valid gate index. The picker can hand us
        a detection whose gate_idx differs from target_idx (e.g.
        target_idx mode falling back to nearest when no detection is
        tagged with the current target). Pre-S19n, the baseline cascade
        masked this by aggressively advancing target_idx to match
        whatever got picked; S19n's refractory exposed the bug because
        target_idx now lags the picked detection by several gates.
        Anchoring on picked.gate_idx is what the detection actually
        represents.

        When gate_idx is -1 (real YOLO, which doesn't tag identity) we
        fall back to target_idx. This is a load-bearing approximation
        for real-vision runs — a future identity-aware detector or PnP
        layer should supply gate_idx so we never hit this path.

        Uses the currently-fused yaw to rotate body→NED. Passes the same
        yaw as the "measurement" to keep the yaw innovation at zero —
        this is a pos-only fusion update. When YOLO ships proper PnP
        gate-orientation, we can upgrade this to publish yaw too.
        """
        if picked is None:
            return
        # Anchor gate: prefer detection's gate_idx; fall back to target.
        gi = getattr(picked, "gate_idx", -1)
        if 0 <= gi < len(self._gates_ned):
            anchor_idx = gi
        elif self.target_idx < len(self._gates_ned):
            anchor_idx = self.target_idx
        else:
            return
        import numpy as np
        bh = math.radians(picked.bearing_h_deg)
        bv = math.radians(picked.bearing_v_deg)
        r = float(picked.range_est)
        if r <= 0.1:
            return
        # Body-frame vector to gate (FRD: X forward, Y right, Z down).
        body_to_gate = np.array([
            r * math.cos(bv) * math.cos(bh),
            r * math.cos(bv) * math.sin(bh),
            r * math.sin(bv),
        ])
        # Rotate body → NED assuming level flight (roll=pitch=0). This
        # is the same simplification PoseFusion's predict() uses for the
        # non-attitude tests; BeliefNav+V5.1 also assume level.
        cy, sy = math.cos(fused_yaw), math.sin(fused_yaw)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]])
        world_to_gate = Rz @ body_to_gate
        gate_ned = self._gates_ned[anchor_idx]
        drone_ned_meas = gate_ned - world_to_gate
        self.pose_fusion.on_vision_pose(
            p_world=drone_ned_meas,
            yaw_rad=fused_yaw,       # zero yaw innovation → pos-only update
            pos_sigma=self._vision_pos_sigma,
        )

    # ── Per-tick step ───────────────────────────────────────────
    async def step(self) -> StepResult:
        """One command tick. Public so tests can drive the loop
        deterministically without asyncio.sleep."""
        state = await self.adapter.get_state()
        frame = await self.adapter.get_camera_frame()
        detections = self.detector.detect(frame, state)

        # S20 — when a GateSequencer is active, let it filter detections
        # and detect gate passes. The sequencer replaces the built-in
        # target_idx tracking for unknown courses.
        _seq_passed = False
        if self.gate_sequencer is not None:
            yaw_rad = float(state.att_rad[2])
            elapsed = time.time() - (self._start_time or time.time())
            seq_target, _seq_passed = self.gate_sequencer.update(
                detections, list(state.pos_ned), yaw_rad, elapsed,
            )
            # Sync target_idx from sequencer so the rest of the loop
            # (navigator gate advancement, run termination) stays consistent.
            if self.gate_sequencer.gates_passed > self.target_idx:
                for _ in range(self.gate_sequencer.gates_passed - self.target_idx):
                    self._advance_target()
            # Feed the sequencer's picked detection to the navigator via
            # the standard tracker-state path. Build a tracker state from
            # the single picked detection (or None).
            detections = [seq_target] if seq_target is not None else []
            # Feed the sequencer's nav gate list (passed gates + predicted
            # next gate) to the navigator for gate-aware fallback. The
            # predicted gate at index `gates_passed` is where target_idx
            # points, so the navigator steers toward it when no detection
            # is available — far better than blind search.
            nav_gates = self.gate_sequencer.get_nav_gate_list()
            if nav_gates and hasattr(self.navigator, "set_gates_ned"):
                import numpy as _np
                self._gates_ned = [_np.asarray(g, dtype=float) for g in nav_gates]
                self.navigator.set_gates_ned(self._gates_ned)

        tracker = self._build_tracker_state(detections)

        # --- Optional pose fusion ---
        if self.pose_fusion is not None:
            # Drive IMU first so the filter is up-to-date before the
            # vision update.
            await self._ingest_imu_async()
            picked = self._pick_detection(detections)
            # Seed the filter from adapter truth if it hasn't been
            # initialized yet (first tick). Caller can pre-seed too.
            if not self.pose_fusion.is_seeded:
                import numpy as np
                self.pose_fusion.seed(
                    p=np.asarray(state.pos_ned, dtype=float),
                    v=np.asarray(state.vel_ned, dtype=float),
                    yaw_rad=float(state.att_rad[2]),
                    bias_sigma=0.1,    # allow mild bias absorption
                )
            if picked is not None and picked.confidence > 0.0:
                _, _, yaw_now = self.pose_fusion.pose()
                self._ingest_vision_from_detection(picked, yaw_now)
            pos_ned, vel_ned, yaw_rad = self.pose_fusion.pose()
            pos_for_plan = list(pos_ned)
            vel_for_plan = list(vel_ned)
            yaw_deg = math.degrees(yaw_rad)

            # S19p pose-trust gate. Check the rolling rejection rate;
            # if the chi-squared gate is dropping most fixes, the fused
            # pose is unreliable and the navigator's gate-aware fallback
            # would commit to a wrong world-frame target. Flip
            # `pose_trusted` so BeliefNav falls through to belief-coast/
            # search. Only active when the navigator has the attribute
            # (backward-compatible with non-BeliefNav navigators).
            if hasattr(self.navigator, "pose_trusted") and hasattr(
                self.pose_fusion, "recent_reject_rate"
            ):
                reject_rate = self.pose_fusion.recent_reject_rate()
                self.navigator.pose_trusted = (
                    reject_rate < self.POSE_TRUST_REJECT_RATE
                )
        else:
            pos_for_plan = list(state.pos_ned)
            vel_for_plan = list(state.vel_ned)
            yaw_deg = math.degrees(state.att_rad[2])
            # No fusion → adapter truth. Always trusted.
            if hasattr(self.navigator, "pose_trusted"):
                self.navigator.pose_trusted = True

        cmd = self.navigator.plan(
            tracker,
            pos_for_plan,
            vel_for_plan,
            yaw_deg,
            self.dt,
        )

        # VelocityNedYaw has north_m_s/east_m_s/down_m_s/yaw_deg.
        vn = float(cmd.north_m_s)
        ve = float(cmd.east_m_s)
        vd = float(cmd.down_m_s)
        cy = float(cmd.yaw_deg)

        if self._attitude_ctrl is not None:
            # Attitude path: convert velocity NED → T/R/P/Y.
            att_cmd = self._attitude_ctrl.convert(
                desired_vn=vn,
                desired_ve=ve,
                desired_vd=vd,
                desired_yaw_deg=cy,
                current_vel_ned=state.vel_ned,
                current_yaw_deg=yaw_deg,
                dt=self.dt,
            )
            await self.adapter.send_attitude(
                att_cmd.roll_deg, att_cmd.pitch_deg,
                att_cmd.yaw_deg, att_cmd.throttle,
            )
        else:
            await self.adapter.send_velocity_ned(vn, ve, vd, cy)

        # Gate-pass dispatch. When a GateSequencer is active, it already
        # handled pass detection in the step() preamble above — use its
        # result. Otherwise, position-based when we know where the gates
        # are in NED; falls back to range-based otherwise.
        if self.gate_sequencer is not None:
            passed = _seq_passed
        elif self._gates_ned is not None:
            passed = self._check_gate_pass_position(tracker, pos_for_plan)
        else:
            # Pass pos_for_plan so the S19r refractory can engage — see
            # _check_gate_pass docstring for why the drone-displacement
            # reference is safe under pose drift.
            passed = self._check_gate_pass(tracker, pos_for_plan)
        t = time.time() - (self._start_time or time.time())
        return StepResult(
            t=t, target_idx=self.target_idx, detected=tracker.detected,
            range_est=tracker.range_est,
            cmd_vn=vn, cmd_ve=ve, cmd_vd=vd, cmd_yaw_deg=cy,
            passed_gate=passed,
            pos_ned=(float(pos_for_plan[0]),
                     float(pos_for_plan[1]),
                     float(pos_for_plan[2])),
            yaw_deg=float(yaw_deg),
        )

    # ── Full run ────────────────────────────────────────────────
    async def run(
        self,
        timeout_s: float = 120.0,
        log_steps: bool = False,
        realtime: bool = True,
    ) -> RunResult:
        """Fly until all gates pass or timeout.

        Caller is responsible for arming / takeoff / start_offboard
        before calling run(). The loop only issues velocity setpoints.

        Args:
            realtime: When True (default, production path), pace each
                tick at command_hz via asyncio.sleep(dt) and measure
                elapsed with wall-clock time. When False (sandbox soak
                via `soak.py`), drop the sleep and advance time by `dt`
                per tick — simulated time, not wall time. Safe *only*
                for adapters whose physics doesn't depend on real
                wallclock (MockKinematicAdapter with auto_step=True is
                the validated case). PX4 SITL and DCL run on their own
                clocks and will misbehave here, so stick with the
                default for anything not strictly sandbox.
        """
        self._start_time = time.time()
        steps: List[StepResult] = []
        timeout = False
        sim_elapsed = 0.0
        if realtime:
            elapsed_fn = lambda: time.time() - self._start_time  # noqa: E731
        else:
            elapsed_fn = lambda: sim_elapsed  # noqa: E731
        while self.target_idx < self.gate_count:
            elapsed = elapsed_fn()
            if elapsed >= timeout_s:
                timeout = True
                break
            step = await self.step()
            if log_steps:
                # Overwrite wall-clock `t` with sim time in fast mode so
                # downstream analysis (soak failure-mode classifier,
                # replay) sees consistent dt across ticks.
                if not realtime:
                    step.t = sim_elapsed
                steps.append(step)
            if realtime:
                await asyncio.sleep(self.dt)
            else:
                sim_elapsed += self.dt
                # Yield to the event loop so asyncio scheduling still
                # works (adapter coroutines, etc.) without wall delay.
                await asyncio.sleep(0)

        total = elapsed_fn() if not realtime else (
            time.time() - self._start_time
        )
        return RunResult(
            gates_passed=self.target_idx,
            gate_count=self.gate_count,
            total_time_s=total,
            timeout=timeout,
            steps=steps,
        )
