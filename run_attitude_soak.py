"""Offline soak test: full race loop through the attitude control path.

Exercises the complete pipeline end-to-end:
    Navigator → AttitudeController → MockDCLAdapter → attitude dynamics

Runs in fast mode (realtime=False) so simulated time advances per tick
without wall-clock pacing — a 120 s simulated race finishes in seconds.

Tests multiple courses and noise profiles to surface any failure modes
in the attitude path that the unit tests wouldn't catch.
"""

from __future__ import annotations

import asyncio
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Stub mavsdk if not installed (same as run_race.py)
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

from sim.mock_dcl import MockDCLAdapter
from sim.adapter import SimCapability
from control.attitude_controller import AttitudeController
from race_loop import RaceLoop

passed = 0
failed = 0
total_start = time.time()


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


async def run_soak(
    course_name: str,
    gates_ned: list,
    noise_profile: str = "clean",
    timeout_s: float = 60.0,
    command_hz: int = 50,
    hover_throttle: float = 9.81 / 20.0,
) -> dict:
    """Run one full race through the attitude path and return metrics."""

    # Build adapter
    adapter = MockDCLAdapter(initial_altitude_m=2.0)
    await adapter.connect()

    # Build detector (VirtualDetector from truth pose)
    from vision.detector import make_detector
    detector = make_detector("virtual", gates=gates_ned, noise_profile=noise_profile)

    # Build navigator
    from gate_belief import BeliefNav
    navigator = BeliefNav(max_speed=8.0, cruise_speed=6.0)

    # Build attitude controller with tuned hover throttle
    ctrl = AttitudeController(hover_throttle=hover_throttle)

    # Build and run RaceLoop
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = RaceLoop(
            adapter=adapter,
            detector=detector,
            navigator=navigator,
            gate_count=len(gates_ned),
            command_hz=command_hz,
            gates_ned=gates_ned,
            attitude_controller=ctrl,
        )

    result = await loop.run(
        timeout_s=timeout_s,
        log_steps=True,
        realtime=False,
    )

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


async def run_all():
    global passed, failed

    print("=" * 60)
    print("ATTITUDE PATH SOAK TEST")
    print("=" * 60)

    # ── Course definitions ─────────────────────────────────
    # Import courses
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from courses import get_course

    courses = {}
    for name in ("technical", "sprint"):
        try:
            courses[name] = get_course(name)
        except Exception as e:
            print(f"  SKIP: course '{name}' not available ({e})")

    if not courses:
        print("ERROR: No courses available. Cannot run soak test.")
        sys.exit(1)

    # ── Test matrix ────────────────────────────────────────
    test_cases = []
    for course_name, gates in courses.items():
        for noise in ("clean", "mild"):
            test_cases.append((course_name, gates, noise))

    # ── Run each case ──────────────────────────────────────
    results = []
    for course_name, gates, noise in test_cases:
        label = f"{course_name}/{noise}"
        print(f"\n[{label}] ({len(gates)} gates)")
        t0 = time.time()

        try:
            metrics = await run_soak(
                course_name=course_name,
                gates_ned=gates,
                noise_profile=noise,
                timeout_s=120.0,
            )
        except Exception as e:
            print(f"  CRASH: {e}")
            check(f"{label} no crash", False, str(e))
            continue

        wall_s = time.time() - t0
        results.append(metrics)

        gp = metrics["gates_passed"]
        gc = metrics["gate_count"]
        sim_t = metrics["total_time_s"]
        steps = metrics["steps"]
        completed = metrics["completed"]
        timed_out = metrics["timeout"]

        print(f"  Gates: {gp}/{gc}  SimTime: {sim_t:.1f}s  Steps: {steps}  Wall: {wall_s:.1f}s")

        check(f"{label} no crash", True)
        check(f"{label} completed", completed,
              f"{gp}/{gc} timeout={timed_out}")

        # If completed, check sim time is reasonable. Attitude-controlled
        # flights are slower than direct velocity control — the PID needs
        # time to tilt, accelerate, and stabilize. 90s is generous.
        if completed:
            check(f"{label} sim_time < 90s", sim_t < 90.0,
                  f"took {sim_t:.1f}s")

    # ── Summary ────────────────────────────────────────────
    total_wall = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"SOAK RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"Wall time: {total_wall:.1f}s")

    if failed == 0:
        print("ALL SOAK TESTS PASSED")
    else:
        print("SOME SOAK TESTS FAILED")
        # Print summary table
        print("\n  Course/Noise      Gates   SimTime  Status")
        print("  " + "-" * 50)
        for r in results:
            status = "OK" if r["completed"] else "FAIL"
            print(f"  {r['course']:12s}/{r['noise']:6s}  "
                  f"{r['gates_passed']:2d}/{r['gate_count']:2d}  "
                  f"{r['total_time_s']:6.1f}s   {status}")
        sys.exit(1)


asyncio.run(run_all())
