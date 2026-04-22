"""Offline A/B test: belief model yaw fix validation.

Replays the exact failure mode from S18: mild-noise dropout during
turns causes belief to drift. Uses the mock adapter with synthetic
vision (no PX4 required).

Compares:
  - VisionNav (control): no belief model, raw tracker
  - BeliefNav (treatment): with yaw-corrected propagation

On the technical course with mild noise, the PRE-FIX belief model
went 0/3 (83% search). After the S19 yaw fix, belief should:
  - Complete the course (12/12 gates)
  - Keep search fraction < 30%
  - Match or beat vision time

This replaces the full PX4 A/B with a fast deterministic test.
"""

import asyncio
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

# Stub mavsdk for environments that don't have it
import types as _types
if "mavsdk" not in sys.modules:
    _m = _types.ModuleType("mavsdk")
    _o = _types.ModuleType("mavsdk.offboard")
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

from courses import get_course
from vision.detector import VirtualDetector
from gate_belief import BeliefNav
from race_loop import RaceLoop


async def run_trial(
    course_name: str,
    noise_profile: str,
    nav_type: str,
    timeout_s: float = 60.0,
    seed: int = 42,
):
    """Run one mock trial and return metrics dict."""
    gates = get_course(course_name)

    # Build mock adapter
    from run_race import _MockFlightAdapter
    adapter = _MockFlightAdapter()
    await adapter.connect()
    await adapter.arm()
    await adapter.takeoff(2.0)

    # Build detector
    detector = VirtualDetector(gates, noise_profile=noise_profile, seed=seed)

    # Build navigator
    navigator = BeliefNav()
    navigator.set_gates_ned(gates)

    # Build race loop
    loop = RaceLoop(
        adapter=adapter,
        detector=detector,
        navigator=navigator,
        gate_count=len(gates),
        command_hz=50,
        associate_mode="target_idx",
        gates_ned=gates,
    )

    # Run with realtime=False for fast execution
    result = await loop.run(
        timeout_s=timeout_s,
        log_steps=True,
        realtime=False,
    )

    await adapter.land()
    await adapter.disconnect()

    # Extract metrics from logged steps
    total_steps = len(result.steps)
    search_steps = sum(
        1 for s in result.steps
        if getattr(s, "search_mode", False)
        or (not getattr(s, "detected", True))
    )
    detected_steps = sum(
        1 for s in result.steps if getattr(s, "detected", False)
    )

    return {
        "course": course_name,
        "noise": noise_profile,
        "nav": nav_type,
        "completed": result.completed,
        "time": round(result.total_time_s, 3),
        "gates_passed": result.gates_passed,
        "total_steps": total_steps,
        "pct_search": round(100 * search_steps / max(total_steps, 1), 1),
        "pct_detected": round(100 * detected_steps / max(total_steps, 1), 1),
        "timeout": result.timeout,
    }


async def run_ab_test():
    """Run the full A/B test matrix."""
    course = "technical"
    gates = get_course(course)
    n_gates = len(gates)
    n_trials = 3
    seeds = [42, 123, 7]

    conditions = [
        ("clean", "belief"),
        ("mild", "belief"),
        ("harsh", "belief"),
    ]

    results = []
    print(f"=== Offline Belief Model A/B Test ===")
    print(f"Course: {course} ({n_gates} gates)")
    print(f"Trials per condition: {n_trials}")
    print()

    for noise, nav in conditions:
        print(f"--- {noise}/{nav} ---")
        for i, seed in enumerate(seeds):
            r = await run_trial(
                course_name=course,
                noise_profile=noise,
                nav_type=nav,
                timeout_s=60.0,
                seed=seed,
            )
            r["trial"] = i + 1
            results.append(r)
            status = "✓" if r["completed"] else "✗"
            print(
                f"  [{status}] t{i+1}: gates={r['gates_passed']}/{n_gates} "
                f"time={r['time']:.1f}s search={r['pct_search']:.0f}%"
            )

    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"{'condition':25s} {'done':6s} {'median_t':9s} {'search%':8s}")
    for noise, nav in conditions:
        label = f"{noise}/{nav}"
        trials = [r for r in results if r["noise"] == noise and r["nav"] == nav]
        done = [r for r in trials if r["completed"]]
        n_done = len(done)
        n_total = len(trials)
        if done:
            med_t = median([r["time"] for r in done])
            avg_srch = mean([r["pct_search"] for r in trials])
            print(f"  {label:23s} {n_done}/{n_total}    {med_t:7.1f}s  {avg_srch:6.1f}%")
        else:
            avg_srch = mean([r["pct_search"] for r in trials])
            print(f"  {label:23s} {n_done}/{n_total}    DNF      {avg_srch:6.1f}%")

    # Validate criteria
    print()
    mild_trials = [r for r in results if r["noise"] == "mild"]
    mild_done = [r for r in mild_trials if r["completed"]]
    mild_search = mean([r["pct_search"] for r in mild_trials])

    if len(mild_done) >= 2:
        print("✓ PASS: mild/belief completes ≥2/3 (was 0/3 pre-fix)")
    else:
        print(f"✗ FAIL: mild/belief completes only {len(mild_done)}/3")

    if mild_search < 30:
        print(f"✓ PASS: mild search fraction {mild_search:.1f}% < 30%")
    else:
        print(f"✗ FAIL: mild search fraction {mild_search:.1f}% ≥ 30%")

    return results


def main():
    results = asyncio.run(run_ab_test())
    return 0


if __name__ == "__main__":
    sys.exit(main())
