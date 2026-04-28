#!/usr/bin/env python3
"""Autonomous overnight test & improvement loop.

Kick this off and walk away.  It will:

  1. Run unit tests            (attitude controller + mock DCL adapter)
  2. Run soak tests            (full race loop, all courses × noise profiles)
  3. Run PID gain sweeps       (grid search for faster race times)
  4. Run speed sweeps          (vary cruise/max speed and measure completion)
  5. Aggregate results         (JSON log + human-readable summary)

Every experiment is self-contained and logged.  If one crashes, the loop
continues and flags the failure.

Usage:
    python3 run_overnight.py                   # full run
    python3 run_overnight.py --quick           # smoke test (tiny grid)
    python3 run_overnight.py --hours 8         # cap wall-clock time
    python3 run_overnight.py --sweep-only      # skip unit tests
    nohup python3 run_overnight.py &           # truly overnight

Results go to: overnight_results/<timestamp>/
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Python path setup ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Stub mavsdk ──────────────────────────────────────────────────
try:
    import mavsdk  # noqa: F401
except ImportError:
    import types
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

# ── Imports ──────────────────────────────────────────────────────
from sim.mock_dcl import MockDCLAdapter
from control.attitude_controller import AttitudeController, PIDGains
from race_loop import RaceLoop
from courses import get_course, list_courses
from vision.detector import make_detector
from gate_belief import BeliefNav

import warnings
warnings.filterwarnings("ignore")


# ── Output directory ─────────────────────────────────────────────
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_OUT = Path(f"overnight_results/{_TS}")
_OUT.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _OUT / "log.jsonl"
_SUMMARY_FILE = _OUT / "summary.txt"

# Global counters
_total_experiments = 0
_total_passed = 0
_total_failed = 0
_total_crashed = 0


def _log(entry: Dict[str, Any]) -> None:
    """Append one JSON line to the log file."""
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _print_and_log(msg: str, entry: Optional[Dict] = None) -> None:
    print(msg)
    if entry:
        _log(entry)


# ── Single race runner ───────────────────────────────────────────

async def run_single_race(
    course_name: str,
    noise_profile: str,
    command_hz: int = 50,
    timeout_s: float = 120.0,
    # Attitude controller params
    hover_throttle: float = 9.81 / 20.0,
    vel_kp: float = 5.0,
    vel_ki: float = 0.5,
    vel_kd: float = 1.0,
    alt_kp: float = 3.0,
    alt_ki: float = 1.0,
    alt_kd: float = 0.5,
    yaw_kp: float = 2.0,
    max_roll_deg: float = 35.0,
    max_pitch_deg: float = 35.0,
    # Navigator params
    max_speed: float = 8.0,
    cruise_speed: float = 6.0,
) -> Dict[str, Any]:
    """Run one full race and return metrics dict."""
    gates = get_course(course_name)

    adapter = MockDCLAdapter(initial_altitude_m=2.0)
    await adapter.connect()

    detector = make_detector("virtual", gates=gates, noise_profile=noise_profile)
    navigator = BeliefNav(max_speed=max_speed, cruise_speed=cruise_speed)

    ctrl = AttitudeController(
        hover_throttle=hover_throttle,
        max_roll_deg=max_roll_deg,
        max_pitch_deg=max_pitch_deg,
        vel_north_gains=PIDGains(
            kp=vel_kp, ki=vel_ki, kd=vel_kd,
            output_limit=max_pitch_deg, integral_limit=max_pitch_deg * 0.5,
        ),
        vel_east_gains=PIDGains(
            kp=vel_kp, ki=vel_ki, kd=vel_kd,
            output_limit=max_roll_deg, integral_limit=max_roll_deg * 0.5,
        ),
        vel_down_gains=PIDGains(
            kp=alt_kp, ki=alt_ki, kd=alt_kd,
            output_limit=0.4, integral_limit=0.2,
        ),
        yaw_gains=PIDGains(
            kp=yaw_kp, ki=0.1, kd=0.02,
            output_limit=90.0, integral_limit=30.0,
        ),
    )

    loop = RaceLoop(
        adapter=adapter, detector=detector, navigator=navigator,
        gate_count=len(gates), command_hz=command_hz,
        gates_ned=gates, attitude_controller=ctrl,
    )

    result = await loop.run(timeout_s=timeout_s, log_steps=False, realtime=False)
    await adapter.disconnect()

    return {
        "course": course_name,
        "noise": noise_profile,
        "gates_passed": result.gates_passed,
        "gate_count": result.gate_count,
        "total_time_s": result.total_time_s,
        "timeout": result.timeout,
        "completed": result.gates_passed >= result.gate_count and not result.timeout,
        "steps": len(result.steps),
    }


# ── Phase 1: Unit tests ─────────────────────────────────────────

def run_unit_tests() -> Dict[str, Any]:
    """Run existing unit test scripts and capture results."""
    import subprocess
    results = {}
    for script in ["run_attitude_tests.py", "run_mock_dcl_tests.py"]:
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=120,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            results[script] = {
                "exit_code": r.returncode,
                "passed": r.returncode == 0,
                "wall_s": time.time() - t0,
                "stdout_tail": r.stdout[-500:] if r.stdout else "",
                "stderr_tail": r.stderr[-300:] if r.stderr else "",
            }
        except Exception as e:
            results[script] = {
                "exit_code": -1,
                "passed": False,
                "wall_s": time.time() - t0,
                "error": str(e),
            }
    return results


# ── Phase 2: Soak tests ─────────────────────────────────────────

async def run_soak_tests() -> List[Dict[str, Any]]:
    """Run full race soak across all courses and noise profiles."""
    results = []
    for course_name in list_courses():
        for noise in ("clean", "mild"):
            t0 = time.time()
            try:
                metrics = await run_single_race(course_name, noise)
                metrics["wall_s"] = time.time() - t0
                metrics["crashed"] = False
                results.append(metrics)
            except Exception as e:
                results.append({
                    "course": course_name, "noise": noise,
                    "crashed": True, "error": str(e),
                    "wall_s": time.time() - t0,
                })
    return results


# ── Phase 3: PID gain sweep ─────────────────────────────────────

async def run_pid_sweep(quick: bool = False) -> List[Dict[str, Any]]:
    """Grid search over velocity PID gains to minimize race time."""
    if quick:
        kp_vals = [5.0, 7.0]
        ki_vals = [0.5]
        kd_vals = [1.0]
    else:
        kp_vals = [3.0, 5.0, 7.0, 9.0]
        ki_vals = [0.3, 0.5, 0.8]
        kd_vals = [0.5, 1.0, 1.5]

    course = "technical"  # tight course = most sensitive to gains
    noise = "clean"
    results = []

    total = len(kp_vals) * len(ki_vals) * len(kd_vals)
    i = 0

    for kp, ki, kd in product(kp_vals, ki_vals, kd_vals):
        i += 1
        label = f"kp={kp} ki={ki} kd={kd}"
        t0 = time.time()
        try:
            metrics = await run_single_race(
                course, noise,
                vel_kp=kp, vel_ki=ki, vel_kd=kd,
            )
            metrics.update({
                "sweep": "pid", "vel_kp": kp, "vel_ki": ki, "vel_kd": kd,
                "wall_s": time.time() - t0, "crashed": False,
            })
            status = f"{'OK' if metrics['completed'] else 'FAIL'} {metrics['total_time_s']:.1f}s"
        except Exception as e:
            metrics = {
                "sweep": "pid", "vel_kp": kp, "vel_ki": ki, "vel_kd": kd,
                "crashed": True, "error": str(e),
                "wall_s": time.time() - t0,
            }
            status = f"CRASH"

        print(f"  [{i}/{total}] {label} → {status}")
        results.append(metrics)

    return results


# ── Phase 4: Speed sweep ────────────────────────────────────────

async def run_speed_sweep(quick: bool = False) -> List[Dict[str, Any]]:
    """Vary cruise/max speed and measure race completion time."""
    if quick:
        speed_pairs = [(4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
    else:
        speed_pairs = [
            (3.0, 5.0), (4.0, 6.0), (5.0, 7.0), (6.0, 8.0),
            (7.0, 9.0), (8.0, 10.0), (9.0, 12.0), (10.0, 14.0),
        ]

    results = []
    for course in list_courses():
        for cruise, maxs in speed_pairs:
            t0 = time.time()
            label = f"{course} cruise={cruise} max={maxs}"
            try:
                metrics = await run_single_race(
                    course, "clean",
                    cruise_speed=cruise, max_speed=maxs,
                )
                metrics.update({
                    "sweep": "speed", "cruise_speed": cruise, "max_speed": maxs,
                    "wall_s": time.time() - t0, "crashed": False,
                })
                status = f"{'OK' if metrics['completed'] else 'FAIL'} {metrics['total_time_s']:.1f}s"
            except Exception as e:
                metrics = {
                    "sweep": "speed", "course": course,
                    "cruise_speed": cruise, "max_speed": maxs,
                    "crashed": True, "error": str(e),
                    "wall_s": time.time() - t0,
                }
                status = "CRASH"

            print(f"  {label} → {status}")
            results.append(metrics)

    return results


# ── Phase 5: Altitude gain sweep ────────────────────────────────

async def run_alt_sweep(quick: bool = False) -> List[Dict[str, Any]]:
    """Vary altitude PID gains to improve vertical stability."""
    if quick:
        alt_kp_vals = [2.0, 3.0, 4.0]
    else:
        alt_kp_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    results = []
    course = "technical"
    for alt_kp in alt_kp_vals:
        t0 = time.time()
        try:
            metrics = await run_single_race(
                course, "clean", alt_kp=alt_kp,
            )
            metrics.update({
                "sweep": "alt_pid", "alt_kp": alt_kp,
                "wall_s": time.time() - t0, "crashed": False,
            })
            status = f"{'OK' if metrics['completed'] else 'FAIL'} {metrics['total_time_s']:.1f}s"
        except Exception as e:
            metrics = {
                "sweep": "alt_pid", "alt_kp": alt_kp,
                "crashed": True, "error": str(e),
                "wall_s": time.time() - t0,
            }
            status = "CRASH"

        print(f"  alt_kp={alt_kp} → {status}")
        results.append(metrics)

    return results


# ── Summary builder ──────────────────────────────────────────────

def build_summary(
    unit_results: Dict,
    soak_results: List[Dict],
    pid_results: List[Dict],
    speed_results: List[Dict],
    alt_results: List[Dict],
    total_wall_s: float,
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  OVERNIGHT RUN SUMMARY")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total wall time: {total_wall_s:.0f}s ({total_wall_s/60:.1f} min)")
    lines.append("=" * 60)

    # Unit tests
    lines.append("\n── UNIT TESTS ──")
    for script, r in unit_results.items():
        status = "PASS" if r.get("passed") else "FAIL"
        lines.append(f"  {script}: {status} ({r.get('wall_s', 0):.1f}s)")

    # Soak tests
    lines.append("\n── SOAK TESTS ──")
    lines.append(f"  {'Course':<15} {'Noise':<8} {'Gates':<10} {'SimTime':<10} {'Status'}")
    lines.append("  " + "-" * 55)
    for r in soak_results:
        if r.get("crashed"):
            lines.append(f"  {r['course']:<15} {r['noise']:<8} {'CRASH':<10} {'-':<10} CRASH")
        else:
            gp = f"{r['gates_passed']}/{r['gate_count']}"
            status = "OK" if r["completed"] else "FAIL"
            lines.append(f"  {r['course']:<15} {r['noise']:<8} {gp:<10} {r['total_time_s']:<10.1f} {status}")

    # PID sweep — find best
    lines.append("\n── PID GAIN SWEEP (technical/clean) ──")
    completed_pid = [r for r in pid_results if r.get("completed")]
    if completed_pid:
        best = min(completed_pid, key=lambda r: r["total_time_s"])
        lines.append(f"  Best: kp={best['vel_kp']} ki={best['vel_ki']} kd={best['vel_kd']} "
                     f"→ {best['total_time_s']:.1f}s")
        # Show top 5
        top5 = sorted(completed_pid, key=lambda r: r["total_time_s"])[:5]
        lines.append(f"\n  Top 5:")
        for r in top5:
            lines.append(f"    kp={r['vel_kp']:<5} ki={r['vel_ki']:<5} kd={r['vel_kd']:<5} → {r['total_time_s']:.1f}s")
    else:
        lines.append("  No completed runs!")

    # Speed sweep — find best per course
    lines.append("\n── SPEED SWEEP ──")
    for course in list_courses():
        course_runs = [r for r in speed_results
                       if r.get("course") == course and r.get("completed")]
        if course_runs:
            best = min(course_runs, key=lambda r: r["total_time_s"])
            lines.append(f"  {course}: best cruise={best['cruise_speed']} "
                         f"max={best['max_speed']} → {best['total_time_s']:.1f}s")
        else:
            lines.append(f"  {course}: no completed runs")

    # Alt sweep
    lines.append("\n── ALTITUDE PID SWEEP ──")
    completed_alt = [r for r in alt_results if r.get("completed")]
    if completed_alt:
        best = min(completed_alt, key=lambda r: r["total_time_s"])
        lines.append(f"  Best alt_kp={best['alt_kp']} → {best['total_time_s']:.1f}s")

    # Overall stats
    all_results = soak_results + pid_results + speed_results + alt_results
    n_total = len(all_results)
    n_completed = sum(1 for r in all_results if r.get("completed"))
    n_crashed = sum(1 for r in all_results if r.get("crashed"))
    lines.append(f"\n── TOTALS ──")
    lines.append(f"  Experiments: {n_total}")
    lines.append(f"  Completed:   {n_completed}")
    lines.append(f"  Failed:      {n_total - n_completed - n_crashed}")
    lines.append(f"  Crashed:     {n_crashed}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────

async def main():
    import argparse
    p = argparse.ArgumentParser(description="Autonomous overnight test loop.")
    p.add_argument("--quick", action="store_true",
                   help="Small grid for smoke testing (~2 min).")
    p.add_argument("--hours", type=float, default=24.0,
                   help="Max wall-clock hours before stopping.")
    p.add_argument("--sweep-only", action="store_true",
                   help="Skip unit tests, go straight to sweeps.")
    args = p.parse_args()

    deadline = time.time() + args.hours * 3600
    t_start = time.time()

    print("=" * 60)
    print("  OVERNIGHT AUTONOMOUS TEST LOOP")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max runtime: {args.hours}h")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Output: {_OUT}")
    print("=" * 60)

    # Phase 1: Unit tests
    unit_results = {}
    if not args.sweep_only:
        print("\n[Phase 1/5] Unit tests...")
        unit_results = run_unit_tests()
        for script, r in unit_results.items():
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"  {script}: {status}")
        _log({"phase": "unit_tests", "results": unit_results})

    if time.time() > deadline:
        print("Time limit reached after unit tests.")
        return

    # Phase 2: Soak tests
    print("\n[Phase 2/5] Soak tests (all courses × noise)...")
    soak_results = await run_soak_tests()
    for r in soak_results:
        if r.get("crashed"):
            print(f"  {r['course']}/{r['noise']}: CRASH — {r.get('error', '?')}")
        else:
            status = "OK" if r["completed"] else "FAIL"
            print(f"  {r['course']}/{r['noise']}: {status} {r['total_time_s']:.1f}s")
    _log({"phase": "soak", "results": soak_results})

    if time.time() > deadline:
        print("Time limit reached after soak tests.")
        return

    # Phase 3: PID sweep
    print(f"\n[Phase 3/5] PID gain sweep ({'quick' if args.quick else 'full'})...")
    pid_results = await run_pid_sweep(quick=args.quick)
    _log({"phase": "pid_sweep", "results": pid_results})

    if time.time() > deadline:
        print("Time limit reached after PID sweep.")
        return

    # Phase 4: Speed sweep
    print(f"\n[Phase 4/5] Speed sweep ({'quick' if args.quick else 'full'})...")
    speed_results = await run_speed_sweep(quick=args.quick)
    _log({"phase": "speed_sweep", "results": speed_results})

    if time.time() > deadline:
        print("Time limit reached after speed sweep.")
        return

    # Phase 5: Altitude PID sweep
    print(f"\n[Phase 5/5] Altitude PID sweep...")
    alt_results = await run_alt_sweep(quick=args.quick)
    _log({"phase": "alt_sweep", "results": alt_results})

    # ── Build and write summary ──────────────────────────────
    total_wall = time.time() - t_start
    summary = build_summary(
        unit_results, soak_results, pid_results,
        speed_results, alt_results, total_wall,
    )
    print(f"\n{summary}")

    with open(_SUMMARY_FILE, "w") as f:
        f.write(summary)

    print(f"\nResults saved to: {_OUT}")
    print(f"  Log:     {_LOG_FILE}")
    print(f"  Summary: {_SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
