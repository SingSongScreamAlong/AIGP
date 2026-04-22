"""Session 19j — Fusion A/B stress harness.

Runs the RaceLoop end-to-end across the cross product of:

    pose source  ∈ {legacy (adapter truth), fusion (ESKF-fused)}
    vision noise ∈ {clean, mild, harsh}

on a fixed course with a fixed MockKinematicAdapter seed, and produces
a compact table of gates-passed, total time, and — for fusion runs —
peak and final fused-vs-truth error.

Goal:
  * Empirically demonstrate that fusion completes at least the same
    courses legacy does under clean/mild/harsh vision noise.
  * Quantify how well fusion tracks truth as vision quality degrades.
  * Flush out any regressions where fusion over-corrects on a noisy
    fix and diverges from truth.

Not a unit test — this is a characterization harness. Runs in ~30s and
outputs a table the eng team can paste into the project log.

Usage:
    python3 bench_fusion_ab.py
    python3 bench_fusion_ab.py --course technical
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# Stub mavsdk so legacy importers don't blow up.
if "mavsdk" not in sys.modules:
    m = types.ModuleType("mavsdk")
    o = types.ModuleType("mavsdk.offboard")

    class _System:
        def __init__(self, *a, **k): pass
    m.System = _System

    class _VNY:
        def __init__(self, vn, ve, vd, yd):
            self.north_m_s = vn
            self.east_m_s = ve
            self.down_m_s = vd
            self.yaw_deg = yd
    o.VelocityNedYaw = _VNY
    for n in ("PositionNedYaw", "Attitude"):
        setattr(o, n, type(n, (), {"__init__": lambda self, *a, **k: None}))
    o.OffboardError = type("OffboardError", (Exception,), {})
    sys.modules["mavsdk"] = m
    sys.modules["mavsdk.offboard"] = o


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src"))


from sim.mock import MockKinematicAdapter
from vision.detector import VirtualDetector
from gate_belief import BeliefNav
from race_loop import RaceLoop
from estimation import PoseFusion
from courses import get_course


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BenchRow:
    name: str
    noise: str
    fusion: bool
    gates_passed: int
    gate_count: int
    total_time_s: float
    timed_out: bool
    max_pose_err_m: Optional[float]     # fusion only, over full run
    final_pose_err_m: Optional[float]   # fusion only
    imu_seen: int
    vis_ok: int
    vis_rej: int
    # Distractor-bench fields (S19l). For non-distractor rows these
    # stay None so the table still renders cleanly.
    spoof_mean_m: Optional[float] = None    # mean drone-dist-to-real-target at each pass event
    spoof_max_m: Optional[float] = None     # max of same
    decoy_kind: Optional[str] = None        # "on_path" / "off_path" / None
    # Pass-event pose-error fields (S19q). Per-pass `‖fused − truth‖`
    # sampled at the tick the pass fires — the decision-relevant
    # moment for the navigator and gate-pass detector. `honest_passes`
    # is the count of passes where fused pose was within
    # PASSAGE_RADIUS of truth; it's the single number that makes a
    # "12/12 @ 159 m" completion visibly untrustworthy.
    pass_err_max_m: Optional[float] = None
    pass_err_median_m: Optional[float] = None
    honest_passes: Optional[int] = None     # count where pass_err < PASSAGE_RADIUS


# ─────────────────────────────────────────────────────────────────────
# Distractor detector — VirtualCamera × 2 with decoy gates marked
# gate_idx=-1 (matches real YoloPnpDetector's convention: the detector
# sees boxes but doesn't know which course index they are).
# ─────────────────────────────────────────────────────────────────────

class DistractorVirtualDetector:
    """Two VirtualCameras stacked: the real course (tagged with sim
    indices) and a decoy set (tagged with gate_idx=-1 to mimic YOLO's
    "unknown" tagging in production). Detections are concatenated and
    sorted nearest-first. Exists only in this bench; tests that want
    the clean VirtualDetector behaviour don't see it.

    This intentionally does NOT use `associate_mode='target_idx'` at
    the RaceLoop side — real YOLO returns gate_idx=-1 always, so the
    honest threat model is nearest-first association.
    """

    def __init__(self, real_gates, decoys, noise_profile: str = "clean"):
        from vision_nav import VirtualCamera
        self._real = VirtualCamera(real_gates, noise_profile)
        self._decoy = VirtualCamera(decoys, noise_profile)
        self._profile = noise_profile

    def detect(self, frame, state):
        import dataclasses as _dc
        pos = list(state.pos_ned)
        vel = list(state.vel_ned)
        yaw_deg = math.degrees(state.att_rad[2])
        real = self._real.observe(pos, vel, yaw_deg)
        deco = self._decoy.observe(pos, vel, yaw_deg)
        deco = [_dc.replace(d, gate_idx=-1) for d in deco]
        combined = real + deco
        combined.sort(key=lambda g: g.range_est)
        return combined

    def name(self) -> str:
        return f"virtual+distractor[{self._profile}]"


# A synthetic 5-gate straight-line course used *only* for the
# distractor bench. Decoupled from `courses.py` so the experiment
# doesn't need to reason about the production courses' figure-8 /
# loop-back geometries. Drone spawns at the origin (matching
# MockKinematicAdapter's default) and flies +N; every gate is in
# front of it at the start, so nearest-first associator picks the
# closest in-front target cleanly and we avoid the startup-cascade
# artefact where the spawn point is coincidentally ≤ PASSAGE_RANGE
# from a non-target real gate on the production courses.
DISTRACTOR_COURSE = [
    (15.0, 0.0, -2.5),
    (30.0, 0.0, -2.5),
    (45.0, 0.0, -2.5),
    (60.0, 0.0, -2.5),
    (75.0, 0.0, -2.5),
]


def _generate_decoys(real_gates, kind: str):
    """Deterministic decoy layouts for the S19l/S19m scenarios.

    "on_path":  decoy 1.5 m in -N direction from each real gate (roughly
                 on the drone's approach vector for the courses we ship,
                 which advance +N). Decoy appears NEARER than the real
                 target as the drone closes in. Realistic worst case
                 for the range signal.
    "off_path": cluster of 4 decoys at the corners of a ~50×50 m box
                 around the course centroid, at flight altitude. Placed
                 far enough from every real gate that the drone never
                 physically approaches within PASSAGE_RANGE, but close
                 enough (and at the right altitude) that the detector
                 still sees them when the drone yaws. Demonstrates the
                 position-signal defense: detections exist all run long
                 but neither signal fires because the drone never hits
                 a real-target local-min on a decoy, and range_est is
                 always well above PASSAGE_RANGE.

    Altitude offset (rather than horizontal) was tried first for
    off_path but decoys 15 m below the flight envelope fall outside
    the vertical FOV of even the `clean` profile (fov_v=90°) once
    forward range exceeds ~15 m, so the detector returns nothing and
    we fail to exercise the pass detector against detected decoys.
    Horizontal corner placement keeps decoys in FOV at plausible yaw
    offsets while staying physically out of reach.

    Coordinates returned as (N, E, D) tuples — same shape as `real_gates`.
    """
    if kind == "none":
        return []
    if kind == "on_path":
        return [(g[0] - 1.5, g[1], g[2]) for g in real_gates]
    if kind == "off_path":
        cN = sum(g[0] for g in real_gates) / len(real_gates)
        cE = sum(g[1] for g in real_gates) / len(real_gates)
        alt = sum(g[2] for g in real_gates) / len(real_gates)
        # Compute the course's half-extent so the "corner" offset
        # scales with the course. Minimum 25 m to keep small/tight
        # courses like `technical` (half-extent ~8 m) from putting
        # decoys too close to the drone.
        half_N = max(abs(g[0] - cN) for g in real_gates)
        half_E = max(abs(g[1] - cE) for g in real_gates)
        off_N = max(25.0, 2.0 * half_N)
        off_E = max(25.0, 2.0 * half_E)
        offsets = [(+off_N, +off_E), (-off_N, +off_E),
                   (+off_N, -off_E), (-off_N, -off_E)]
        return [(cN + dN, cE + dE, alt) for (dN, dE) in offsets]
    raise ValueError(f"unknown decoy kind: {kind!r}")


# ─────────────────────────────────────────────────────────────────────
# Per-run driver
# ─────────────────────────────────────────────────────────────────────

def _build_stack(gates, noise_profile: str, with_fusion: bool, seed: int = 0,
                 vision_pos_sigma: float = 0.20):
    """Construct a fresh stack for one A/B row. Every run gets its own
    adapter/detector/navigator/fusion so RNG state and belief state
    don't leak between configurations."""
    adapter = MockKinematicAdapter(
        dt=1.0 / 50,
        vel_tau=0.05,
        yaw_tau=0.10,
        auto_step=True,
        initial_altitude_m=1.0,
        seed=seed,
    )
    detector = VirtualDetector(gates=gates, noise_profile=noise_profile)
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    pf: Optional[PoseFusion] = None
    if with_fusion:
        pf = PoseFusion()
        pf.seed(
            p=np.array([0.0, 0.0, -1.0]),
            v=np.zeros(3),
            yaw_rad=0.0,
            bias_sigma=0.1,
        )
    loop = RaceLoop(
        adapter=adapter,
        detector=detector,
        navigator=navigator,
        gate_count=len(gates),
        command_hz=50,
        associate_mode="target_idx",
        pose_fusion=pf,
        gates_ned=gates if with_fusion else None,
        vision_pos_sigma=vision_pos_sigma,
    )
    return loop, adapter, pf


async def _drive(loop: RaceLoop, adapter: MockKinematicAdapter,
                 pf: Optional[PoseFusion], timeout_s: float) -> Tuple[bool, float, Optional[float], Optional[float], List[float]]:
    """Drive the loop tick-by-tick so we can sample pose error each
    tick without relying on loop.run()'s asyncio.sleep pacing.

    Returns (timed_out, elapsed, max_err, final_err, pass_pose_errors).
    `pass_pose_errors[i]` is `‖fused_pos − truth_pos‖` at the tick
    `target_idx` advanced past gate i. Empty list when no fusion.
    """
    max_err = 0.0
    final_err = 0.0
    elapsed = 0.0
    dt = loop.dt
    pass_pose_errors: List[float] = []
    prev_target = loop.target_idx
    n_ticks_budget = int(timeout_s / dt)
    for _ in range(n_ticks_budget):
        await loop.step()
        if pf is not None:
            truth = np.array(adapter._pos, dtype=float)
            fused = pf.pose()[0]
            err = float(np.linalg.norm(fused - truth))
            if err > max_err:
                max_err = err
            final_err = err
            # Record pose error at each gate-pass event (fusion runs
            # only — legacy/non-fusion doesn't have a separate fused
            # pose to compare against truth).
            new_target = loop.target_idx
            if new_target > prev_target:
                for _just_passed in range(prev_target, new_target):
                    pass_pose_errors.append(err)
                prev_target = new_target
        elapsed += dt
        if loop.target_idx >= loop.gate_count:
            break
    timed_out = loop.target_idx < loop.gate_count
    return (
        timed_out,
        elapsed,
        (max_err if pf is not None else None),
        (final_err if pf is not None else None),
        pass_pose_errors,
    )


def _build_distractor_stack(real_gates, decoys, noise_profile: str,
                             seed: int = 0):
    """Wires `DistractorVirtualDetector` into a legacy (non-fusion)
    stack using `associate_mode="nearest"` — the honest real-YOLO
    threat model. Fusion is intentionally OFF: we want to isolate the
    gate-pass detector's behaviour against decoys, not bundle it
    with ESKF drift artefacts. PoseFusion can be re-added as a
    follow-up if the clean-pose results surface interesting behaviour.
    """
    adapter = MockKinematicAdapter(
        dt=1.0 / 50,
        vel_tau=0.05,
        yaw_tau=0.10,
        auto_step=True,
        initial_altitude_m=1.0,
        seed=seed,
    )
    detector = DistractorVirtualDetector(real_gates, decoys, noise_profile)
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    loop = RaceLoop(
        adapter=adapter,
        detector=detector,
        navigator=navigator,
        gate_count=len(real_gates),
        command_hz=50,
        associate_mode="nearest",   # ← real-YOLO-like (gate_idx=-1 on decoys)
        pose_fusion=None,
        gates_ned=real_gates,        # enables position-based dispatch
    )
    return loop, adapter


async def _drive_distractor(loop: RaceLoop, adapter: MockKinematicAdapter,
                             real_gates, timeout_s: float):
    """Drive the loop while recording `drone_pos_at_pass` for each
    gate-pass event. Returns (timed_out, elapsed, pass_distances),
    where `pass_distances[i]` is how far the drone was from
    real_gates[i] at the tick `target_idx` advanced past i.
    """
    pass_distances: List[float] = []
    prev_target = loop.target_idx
    elapsed = 0.0
    dt = loop.dt
    n_ticks_budget = int(timeout_s / dt)
    for _ in range(n_ticks_budget):
        await loop.step()
        new_target = loop.target_idx
        if new_target > prev_target:
            # One or more passes fired this tick. With the current
            # detector it's always exactly one, but handle the general
            # case for safety.
            for just_passed in range(prev_target, new_target):
                gate = real_gates[just_passed]
                drone = adapter._pos
                d = float(np.linalg.norm(
                    np.array(drone, dtype=float) - np.array(gate, dtype=float)
                ))
                pass_distances.append(d)
            prev_target = new_target
        elapsed += dt
        if loop.target_idx >= loop.gate_count:
            break
    timed_out = loop.target_idx < loop.gate_count
    return timed_out, elapsed, pass_distances


def run_distractor_one(name: str, real_gates, decoy_kind: str,
                       noise: str = "clean", timeout_s: float = 30.0,
                       seed: int = 0) -> BenchRow:
    """One distractor scenario. Does NOT use fusion — see
    `_build_distractor_stack` docstring for rationale.
    """
    decoys = _generate_decoys(real_gates, decoy_kind)
    loop, adapter = _build_distractor_stack(real_gates, decoys, noise, seed)
    timed_out, elapsed, pass_dists = asyncio.get_event_loop().run_until_complete(
        _drive_distractor(loop, adapter, real_gates, timeout_s)
    )
    spoof_mean = (sum(pass_dists) / len(pass_dists)) if pass_dists else None
    spoof_max = max(pass_dists) if pass_dists else None
    return BenchRow(
        name=name,
        noise=noise,
        fusion=False,
        gates_passed=loop.target_idx,
        gate_count=loop.gate_count,
        total_time_s=elapsed,
        timed_out=timed_out,
        max_pose_err_m=None,
        final_pose_err_m=None,
        imu_seen=0,
        vis_ok=0,
        vis_rej=0,
        spoof_mean_m=spoof_mean,
        spoof_max_m=spoof_max,
        decoy_kind=decoy_kind,
    )


def run_one(name: str, gates, noise: str, with_fusion: bool,
            timeout_s: float = 30.0, seed: int = 0,
            vision_pos_sigma: float = 0.20) -> BenchRow:
    loop, adapter, pf = _build_stack(
        gates, noise, with_fusion, seed=seed,
        vision_pos_sigma=vision_pos_sigma,
    )
    timed_out, elapsed, max_err, final_err, pass_errs = asyncio.get_event_loop().run_until_complete(
        _drive(loop, adapter, pf, timeout_s)
    )
    imu_seen = vis_ok = vis_rej = 0
    if pf is not None:
        tel = pf.telemetry
        imu_seen = tel.imu_samples_seen
        vis_ok = tel.vision_fixes_accepted
        vis_rej = tel.vision_fixes_rejected
    # Pass-event stats (fusion only).
    pass_err_max = pass_err_median = None
    honest_passes: Optional[int] = None
    if with_fusion and pass_errs:
        pass_err_max = max(pass_errs)
        s = sorted(pass_errs)
        mid = len(s) // 2
        pass_err_median = (
            s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
        )
        honest_passes = sum(1 for d in pass_errs if d < RaceLoop.PASSAGE_RADIUS)
    elif with_fusion:
        # Fusion on but no passes fired — honest count is 0.
        honest_passes = 0
    return BenchRow(
        name=name,
        noise=noise,
        fusion=with_fusion,
        gates_passed=loop.target_idx,
        gate_count=loop.gate_count,
        total_time_s=elapsed,
        timed_out=timed_out,
        max_pose_err_m=max_err,
        final_pose_err_m=final_err,
        imu_seen=imu_seen,
        vis_ok=vis_ok,
        vis_rej=vis_rej,
        pass_err_max_m=pass_err_max,
        pass_err_median_m=pass_err_median,
        honest_passes=honest_passes,
    )


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def _fmt_err(x: Optional[float]) -> str:
    return "   —  " if x is None else f"{x:6.3f}"


def _fmt_int(x: int) -> str:
    return "—" if x == 0 else str(x)


def _fmt_honest(r: BenchRow) -> str:
    """Render 'honest/total' for fusion rows, '—' for legacy."""
    if not r.fusion or r.honest_passes is None:
        return "    —   "
    return f"  {r.honest_passes:>2}/{r.gates_passed:<2}  "


def print_table(rows: List[BenchRow]) -> None:
    print()
    print("┌────────────────────┬───────┬────────┬────────┬────────┬──────────┬──────────┬──────────┬──────────┬──────┬────────┬────────┐")
    print("│ scenario           │ noise │ fusion │ gates  │ time s │ max_err  │ final_err│ pass_err │  honest  │ imu  │ vis_ok │ vis_rej│")
    print("├────────────────────┼───────┼────────┼────────┼────────┼──────────┼──────────┼──────────┼──────────┼──────┼────────┼────────┤")
    for r in rows:
        status = f"{r.gates_passed}/{r.gate_count}"
        if r.timed_out:
            status += "!"
        print(
            f"│ {r.name:<18} │ {r.noise:<5} │ {'ON ' if r.fusion else 'off':<6} │"
            f" {status:>6} │ {r.total_time_s:6.2f} │ {_fmt_err(r.max_pose_err_m)} m │"
            f" {_fmt_err(r.final_pose_err_m)} m │ {_fmt_err(r.pass_err_max_m)} m │"
            f" {_fmt_honest(r)} │ {_fmt_int(r.imu_seen):>4} │"
            f" {_fmt_int(r.vis_ok):>6} │ {_fmt_int(r.vis_rej):>6} │"
        )
    print("└────────────────────┴───────┴────────┴────────┴────────┴──────────┴──────────┴──────────┴──────────┴──────┴────────┴────────┘")
    print(f"  pass_err = max  ‖fused_pos − truth_pos‖  sampled at gate-pass events (S19q)")
    print(f"  honest   = count of passes where fused pose was < PASSAGE_RADIUS="
          f"{RaceLoop.PASSAGE_RADIUS:.1f} m from truth at pass time.")
    print(f"  A high gate count with a low honest count means the drone physically")
    print(f"  completed on detection bearing while the fused pose was diverged.")


def print_distractor_table(rows: List[BenchRow]) -> None:
    """Dedicated distractor table. Surfaces the pass-quality metric
    that the standard fusion table doesn't carry."""
    print()
    print("─── distractor scenarios (S19l/S19m) ───")
    print("┌────────────────────────┬──────────┬───────┬────────┬────────┬──────────┬─────────┐")
    print("│ scenario               │ decoys   │ noise │ gates  │ time s │ spoof_μ  │ spoof_M │")
    print("├────────────────────────┼──────────┼───────┼────────┼────────┼──────────┼─────────┤")
    for r in rows:
        status = f"{r.gates_passed}/{r.gate_count}"
        if r.timed_out:
            status += "!"
        sp_m = "   —  " if r.spoof_mean_m is None else f"{r.spoof_mean_m:6.2f}"
        sp_M = "  —  " if r.spoof_max_m is None else f"{r.spoof_max_m:5.2f}"
        print(
            f"│ {r.name:<22} │ {(r.decoy_kind or '—'):<8} │ {r.noise:<5} │"
            f" {status:>6} │ {r.total_time_s:6.2f} │ {sp_m} m │ {sp_M} m │"
        )
    print("└────────────────────────┴──────────┴───────┴────────┴────────┴──────────┴─────────┘")
    print("  spoof_μ = mean  ‖drone_pos − real_target‖  at each pass-event tick")
    print("  spoof_M = max   ‖drone_pos − real_target‖  at each pass-event tick")
    print(f"  honest pass: ≤ PASSAGE_RADIUS={RaceLoop.PASSAGE_RADIUS:.1f} m "
          f"(or ≤ PASSAGE_RANGE={RaceLoop.PASSAGE_RANGE:.1f} m for range-only)")
    # S19m FINDING: if baseline_no_decoys shows spoof_M >> PASSAGE_RANGE,
    # you are not seeing decoy effects — you are seeing the nearest-first
    # + range-signal *baseline cascade*:
    #
    #   When target advances past gate K, gate K's detection is still
    #   the closest (drone lingers within PASSAGE_RANGE for several
    #   ticks), so `associate_mode="nearest"` keeps returning it. The
    #   range signal fires for WHATEVER target_idx is now, cascading
    #   through targets K+1, K+2, … within a handful of ticks. Decoys
    #   add almost nothing to this — the cascade dominates.
    #
    # This is a baseline correctness finding for real-YOLO deployment,
    # NOT a distractor-only problem. The fusion A/B bench doesn't
    # surface it because it uses `associate_mode="target_idx"`, which
    # refuses to pick gate K's detection for target K+1 (gate_idx
    # mismatch). Real YOLO emits gate_idx=-1 for every detection, so
    # target_idx mode degenerates to nearest-fallback — which cascades.
    #
    # Candidate fixes (future session): post-pass refractory period
    # (displacement- or tick-based), or require detected gate_idx
    # to match current target. Both live in race_loop._check_gate_pass*.
    # See PROJECT_LOG S19m entry.


def summarize(rows: List[BenchRow]) -> None:
    print()
    print("─── summary ───")
    # Fusion completion vs legacy at each noise level
    by_noise = {}
    for r in rows:
        by_noise.setdefault(r.noise, []).append(r)
    for noise, rs in by_noise.items():
        legacy = [r for r in rs if not r.fusion]
        fused = [r for r in rs if r.fusion]
        if not legacy or not fused:
            continue
        L, F = legacy[0], fused[0]
        lstr = f"{L.gates_passed}/{L.gate_count}"
        fstr = f"{F.gates_passed}/{F.gate_count}"
        delta_s = F.total_time_s - L.total_time_s
        err_str = (
            f"  max_err={F.max_pose_err_m:.2f}m final_err={F.final_pose_err_m:.2f}m"
            if F.max_pose_err_m is not None else ""
        )
        print(
            f"  {noise:5}: legacy {lstr}  fusion {fstr}  "
            f"Δt={delta_s:+.2f}s{err_str}"
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fusion A/B stress harness.")
    p.add_argument("--course", default="technical",
                   choices=("sprint", "technical", "mixed"))
    p.add_argument("--timeout", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vision-pos-sigma", type=float, default=0.20,
                   dest="vision_pos_sigma",
                   help="1-σ (m) for backprojected vision position fix. "
                        "S19j finding: 0.20 works clean but diverges on long "
                        "mild/harsh runs; ~1.0 is robust.")
    p.add_argument("--distractors", action="store_true",
                   help="Run S19l/S19m distractor-realism scenarios "
                        "(distractor_on_path, distractor_off_path) alongside "
                        "the fusion A/B scenarios.")
    p.add_argument("--only-distractors", action="store_true",
                   help="Run ONLY the distractor scenarios; skip the "
                        "6-row fusion A/B table.")
    args = p.parse_args(argv)

    gates = get_course(args.course)
    print(f"Course '{args.course}': {len(gates)} gates, seed={args.seed}")

    scenarios = [
        # Always run legacy first at each noise level — sets the baseline
        # row against which fusion's delta is measured.
        ("legacy/clean",  "clean",  False),
        ("fusion/clean",  "clean",  True),
        ("legacy/mild",   "mild",   False),
        ("fusion/mild",   "mild",   True),
        ("legacy/harsh",  "harsh",  False),
        ("fusion/harsh",  "harsh",  True),
    ]

    rows: List[BenchRow] = []
    if not args.only_distractors:
        for name, noise, fusion in scenarios:
            row = run_one(name, gates, noise, fusion,
                          timeout_s=args.timeout, seed=args.seed,
                          vision_pos_sigma=args.vision_pos_sigma)
            rows.append(row)
        print(f"vision_pos_sigma = {args.vision_pos_sigma} m")
        print_table(rows)
        summarize(rows)

    distractor_rows: List[BenchRow] = []
    if args.distractors or args.only_distractors:
        # Distractor bench runs on a synthetic straight-line course
        # (not the user-selected course) to isolate pass-detector
        # behaviour from production-course geometry artefacts. See
        # DISTRACTOR_COURSE comment.
        dc = DISTRACTOR_COURSE
        print(f"\ndistractor course: {len(dc)} gates (straight-line, +N)")
        # Baseline: no decoys. Establishes the honest pass distance
        # against which the two decoy rows are compared.
        baseline = run_distractor_one(
            "baseline_no_decoys", dc, "none", noise="clean",
            timeout_s=args.timeout, seed=args.seed,
        )
        distractor_rows.append(baseline)
        for dk in ("on_path", "off_path"):
            name = f"distractor_{dk}"
            row = run_distractor_one(
                name, dc, dk, noise="clean",
                timeout_s=args.timeout, seed=args.seed,
            )
            distractor_rows.append(row)
        print_distractor_table(distractor_rows)

    # Exit code: 0 if all completed, 1 otherwise.
    all_rows = rows + distractor_rows
    failed = [r for r in all_rows if r.timed_out]
    if failed:
        print(f"\n{len(failed)}/{len(all_rows)} scenarios TIMED OUT:")
        for r in failed:
            print(f"  - {r.name} ({r.gates_passed}/{r.gate_count})")
        return 1
    print(f"\nall {len(all_rows)} scenarios completed cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
