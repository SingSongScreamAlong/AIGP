"""Session 19w — Modern sandbox-runnable soak harness.

Replaces the S9-vintage `ab_soak.py` (which drives PX4 SITL via
mavsdk, hardcodes a macOS developer path, and uses the pre-S19
`px4_v51_baseline` planner). This one runs in the sandbox with no
external sim, so it can be iterated on without hardware.

What it does:
    for each (course × noise_profile × seed) cell, run one race through
        MockKinematicAdapter + VirtualDetector(seed=trial_seed) +
        BeliefNav + RaceRunner
    capture per-tick StepResult + the final gate count + time
    emit a JSON result file and a human-readable summary table

Why the seed matters:
    VirtualCamera's miss_prob + noise RNG was hardcoded to
    random.Random(42) for over a year. The single-seed bench
    (`bench_fusion_ab.py`) can only see whatever failure a seed-42
    noise trajectory produces; it cannot tell us what fraction of
    random noise trajectories will fail. The legacy `ab_soak.py`
    surfaced a 1/20 "sometimes fails" number whose failure-mode was
    never categorised because it rode on wall-clock-paced PX4 SITL
    jitter rather than detector variance. This harness varies the
    detector seed so the failure mode is legible and reproducible.

Failure-mode categories (post-hoc, from per-trial StepResult):
    "completed"         — all gates passed inside timeout
    "stall"             — timed out with target_idx fixed for > STALL_WINDOW_S
                          and commanded velocity ~0
    "stuck_at_gate"     — timed out, drone is within STUCK_RADIUS of the
                          current target gate but never fires the pass
    "lost_target"       — timed out, detected=False for > LOST_WINDOW_S
                          near the end of the run
    "drifted"           — timed out, drone is > DRIFT_RADIUS from ALL
                          remaining gates at the last tick
    "other"             — failed but didn't fit any of the above buckets

Usage:
    python3 soak.py                                     # default: N=50 technical
    python3 soak.py --n 100 --course technical
    python3 soak.py --courses technical,mixed --noises clean,mild --n 50
    python3 soak.py --out soak_run1.json --summary-only

Deliberate limits:
    * MockKinematicAdapter only — no PX4, no DCL. If a failure mode
      only shows up in PX4/DCL physics (wallclock jitter, wind, motor
      nonlinearity), this harness will miss it. That's the price of
      sandbox-runnable.
    * No parallelism. ~0.6 s/trial at dt=1/50 on the technical course;
      a 100-trial run finishes in ~60 s serially. Parallel asyncio
      trials would share the event loop's wall clock and interfere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── mavsdk stub (sandbox has no mavsdk) ────────────────────────────
if "mavsdk" not in sys.modules:
    _m = types.ModuleType("mavsdk")
    _o = types.ModuleType("mavsdk.offboard")

    class _S:
        def __init__(self, *a, **k): pass
    _m.System = _S

    class _VNY:
        def __init__(self, vn, ve, vd, yd):
            self.north_m_s, self.east_m_s = vn, ve
            self.down_m_s, self.yaw_deg = vd, yd
    _o.VelocityNedYaw = _VNY
    for _n in ("PositionNedYaw", "Attitude"):
        setattr(_o, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    _o.OffboardError = type("OffboardError", (Exception,), {})
    sys.modules["mavsdk"] = _m
    sys.modules["mavsdk.offboard"] = _o


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))


from courses import get_course                       # noqa: E402
from gate_belief import BeliefNav                    # noqa: E402
from race.runner import RaceRunner                   # noqa: E402
from race_loop import StepResult                     # noqa: E402
from sim.adapter import make_adapter                 # noqa: E402
from vision.detector import make_detector            # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Per-trial record + failure classifier
# ─────────────────────────────────────────────────────────────────────

STALL_WINDOW_S = 3.0        # target_idx unchanged for this long + |cmd| ~ 0 → stall
STALL_CMD_EPS = 0.3         # m/s threshold for "not really commanding motion"
LOST_WINDOW_S = 3.0         # no detections for this long near the end → lost_target
STUCK_RADIUS = 4.0          # m — "right at the gate" proximity for stuck_at_gate
DRIFT_RADIUS = 15.0         # m — "gave up and floated away" threshold


@dataclass
class TrialResult:
    course: str
    noise: str
    trial_idx: int
    seed: int
    completed: bool
    gates_passed: int
    gate_count: int
    time_s: float
    timeout: bool
    failure_mode: Optional[str]
    # Compact diagnostic fields — always present so the JSON is
    # uniform across trials. Each is the value AT THE LAST TICK.
    last_target_idx: int
    last_pos: Tuple[float, float, float]
    last_range_est: float
    last_detected: bool
    frames_lost_at_end: int
    dist_to_current_gate_m: Optional[float]
    steps_count: int


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def classify_trial(
    steps: List[StepResult],
    gates: List[Tuple[float, float, float]],
    completed: bool,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (failure_mode, diagnostic_fields).

    Called after every trial; for completed trials returns ("completed", {...})
    so the caller can still pack diagnostics uniformly.
    """
    if not steps:
        return "no_steps_logged", {}

    last = steps[-1]
    pos = last.pos_ned or (0.0, 0.0, 0.0)

    # How long has the current target been stuck at its value?
    t_target_locked_at = steps[-1].t
    cur_target = last.target_idx
    for s in reversed(steps):
        if s.target_idx != cur_target:
            break
        t_target_locked_at = s.t
    target_locked_for = last.t - t_target_locked_at

    # Frames without detection at the tail
    frames_lost_at_end = 0
    for s in reversed(steps):
        if s.detected:
            break
        frames_lost_at_end += 1

    # Commanded motion at the tail
    recent = steps[-min(50, len(steps)) :]
    cmd_mag_p50 = statistics.median(
        math.sqrt(s.cmd_vn ** 2 + s.cmd_ve ** 2 + s.cmd_vd ** 2)
        for s in recent
    )

    # Where is the drone relative to the current target and all remaining
    # gates?
    dist_to_current = None
    dist_to_all_remaining = []
    if 0 <= cur_target < len(gates):
        dist_to_current = _dist(pos, gates[cur_target])
        for g in gates[cur_target:]:
            dist_to_all_remaining.append(_dist(pos, g))

    diag = {
        "last_target_idx": cur_target,
        "last_pos": pos,
        "last_range_est": last.range_est,
        "last_detected": last.detected,
        "frames_lost_at_end": frames_lost_at_end,
        "dist_to_current_gate_m": dist_to_current,
        "target_locked_for_s": round(target_locked_for, 2),
        "recent_cmd_mag_p50": round(cmd_mag_p50, 3),
        "dist_to_nearest_remaining_m": round(
            min(dist_to_all_remaining), 2
        ) if dist_to_all_remaining else None,
    }

    if completed:
        return "completed", diag

    # ── Failure-mode classification ──────────────────────────────────
    # Priority matters: stuck_at_gate → stall (cmd low) → lost → drift → other.
    # "stuck_at_gate" beats "stall" when the drone is arriving at the right
    # gate but the pass detector never fires — that's a different bug than
    # the drone going limp.
    if dist_to_current is not None and dist_to_current < STUCK_RADIUS \
            and target_locked_for > STALL_WINDOW_S:
        return "stuck_at_gate", diag
    if cmd_mag_p50 < STALL_CMD_EPS and target_locked_for > STALL_WINDOW_S:
        return "stall", diag
    if frames_lost_at_end * (1.0 / 50.0) > LOST_WINDOW_S:
        return "lost_target", diag
    if dist_to_all_remaining and min(dist_to_all_remaining) > DRIFT_RADIUS:
        return "drifted", diag
    return "other", diag


# ─────────────────────────────────────────────────────────────────────
# Single-trial driver
# ─────────────────────────────────────────────────────────────────────

async def run_one_trial(
    course_name: str,
    noise_profile: str,
    trial_idx: int,
    seed: int,
    timeout_s: float,
    command_hz: int,
) -> TrialResult:
    gates = get_course(course_name)

    adapter = make_adapter(
        "mock_kinematic",
        dt=1.0 / command_hz,
        vel_tau=0.05,
        yaw_tau=0.10,
        auto_step=True,
        initial_altitude_m=1.0,
    )
    detector = make_detector(
        "virtual",
        gates=gates,
        noise_profile=noise_profile,
        seed=seed,
    )
    navigator = BeliefNav(max_speed=12.0, cruise_speed=9.0)
    runner = RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=gates,
        takeoff_altitude_m=2.0,
        command_hz=command_hz,
    )
    result = await runner.fly(
        timeout_s=timeout_s,
        log_steps=True,
        realtime=False,
    )

    mode, diag = classify_trial(result.run.steps, gates, result.completed)

    return TrialResult(
        course=course_name,
        noise=noise_profile,
        trial_idx=trial_idx,
        seed=seed,
        completed=bool(result.completed),
        gates_passed=result.run.gates_passed,
        gate_count=result.run.gate_count,
        time_s=round(result.run.total_time_s, 3),
        timeout=bool(result.run.timeout),
        failure_mode=mode,
        last_target_idx=diag.get("last_target_idx", -1),
        last_pos=tuple(diag.get("last_pos", (0.0, 0.0, 0.0))),
        last_range_est=float(diag.get("last_range_est", 0.0) or 0.0),
        last_detected=bool(diag.get("last_detected", False)),
        frames_lost_at_end=int(diag.get("frames_lost_at_end", 0)),
        dist_to_current_gate_m=(
            round(diag["dist_to_current_gate_m"], 2)
            if diag.get("dist_to_current_gate_m") is not None else None
        ),
        steps_count=len(result.run.steps),
    )


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────

def _fmt_summary(trials: List[TrialResult]) -> str:
    lines: List[str] = []
    # Group by (course, noise)
    groups: Dict[Tuple[str, str], List[TrialResult]] = {}
    for t in trials:
        groups.setdefault((t.course, t.noise), []).append(t)

    lines.append("")
    lines.append("SOAK SUMMARY")
    lines.append("═" * 78)
    lines.append(
        f"{'course':>10} {'noise':>6} {'N':>4} {'pass':>6} {'rate':>7} "
        f"{'t_med':>7} {'t_p95':>7} {'fail_modes':<30}"
    )
    lines.append("─" * 78)

    for (course, noise), ts in sorted(groups.items()):
        n = len(ts)
        passed = sum(1 for t in ts if t.completed)
        rate = passed / n if n else 0.0
        times_passed = [t.time_s for t in ts if t.completed]
        t_med = (
            round(statistics.median(times_passed), 2)
            if times_passed else float("nan")
        )
        if len(times_passed) >= 2:
            t_p95 = round(
                statistics.quantiles(times_passed, n=20)[-1], 2
            )
        else:
            t_p95 = t_med

        mode_counts: Dict[str, int] = {}
        for t in ts:
            if not t.completed:
                mode_counts[t.failure_mode or "other"] = \
                    mode_counts.get(t.failure_mode or "other", 0) + 1
        modes_str = ", ".join(
            f"{m}:{c}" for m, c in sorted(
                mode_counts.items(), key=lambda kv: -kv[1]
            )
        ) if mode_counts else "—"

        lines.append(
            f"{course:>10} {noise:>6} {n:>4} {passed:>6} "
            f"{rate*100:>6.1f}% {t_med:>7} {t_p95:>7} {modes_str:<30}"
        )
    lines.append("─" * 78)
    lines.append(f"total trials: {len(trials)}  "
                 f"completed: {sum(1 for t in trials if t.completed)}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

async def _run_all(args: argparse.Namespace) -> List[TrialResult]:
    courses = [c.strip() for c in args.courses.split(",") if c.strip()]
    noises = [n.strip() for n in args.noises.split(",") if n.strip()]

    trials: List[TrialResult] = []
    grand_total = len(courses) * len(noises) * args.n
    completed_trials = 0
    t_wall_start = time.time()

    for course in courses:
        for noise in noises:
            for i in range(args.n):
                seed = args.seed_base + i
                t0 = time.time()
                try:
                    tr = await run_one_trial(
                        course_name=course,
                        noise_profile=noise,
                        trial_idx=i,
                        seed=seed,
                        timeout_s=args.timeout,
                        command_hz=args.command_hz,
                    )
                except Exception as e:
                    tr = TrialResult(
                        course=course, noise=noise, trial_idx=i, seed=seed,
                        completed=False, gates_passed=0, gate_count=0,
                        time_s=0.0, timeout=False,
                        failure_mode=f"crash:{type(e).__name__}",
                        last_target_idx=-1,
                        last_pos=(0.0, 0.0, 0.0),
                        last_range_est=0.0, last_detected=False,
                        frames_lost_at_end=0,
                        dist_to_current_gate_m=None,
                        steps_count=0,
                    )
                trials.append(tr)
                completed_trials += 1
                tag = "✓" if tr.completed else f"✗ {tr.failure_mode}"
                print(
                    f"[{completed_trials:>3}/{grand_total}] "
                    f"{course:>9s}/{noise:<5s} seed={seed:<4} "
                    f"{tag:<24} {tr.gates_passed:>2}/{tr.gate_count:<2} "
                    f"{tr.time_s:>5.2f}s  "
                    f"(wall {time.time() - t0:.1f}s)",
                    flush=True,
                )
    print(f"\nsoak wall-clock: {time.time() - t_wall_start:.1f}s")
    return trials


def _write_json(path: Path, trials: List[TrialResult]) -> None:
    payload = {
        "schema": "soak.v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_trials": len(trials),
        "trials": [asdict(t) for t in trials],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=50, help="trials per cell")
    p.add_argument("--courses", default="technical",
                   help="comma-separated list of course names")
    p.add_argument("--noises", default="clean",
                   help="comma-separated list of noise profiles "
                        "(clean|mild|harsh)")
    p.add_argument("--seed-base", type=int, default=1000, dest="seed_base",
                   help="first trial's seed; subsequent trials use "
                        "seed_base+1, seed_base+2, ...")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="per-trial race timeout (s)")
    p.add_argument("--command-hz", type=int, default=50, dest="command_hz")
    p.add_argument("--out", default=None,
                   help="JSON output path; default: soak_runs/<timestamp>.json")
    p.add_argument("--summary-only", action="store_true", dest="summary_only",
                   help="print the summary but skip JSON dump")
    args = p.parse_args(argv)

    trials = asyncio.run(_run_all(args))

    print(_fmt_summary(trials))

    if not args.summary_only:
        out_path = Path(args.out) if args.out else (
            _HERE / "soak_runs"
            / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        )
        _write_json(out_path, trials)

    return 0


if __name__ == "__main__":
    sys.exit(main())
