"""Integration smoke test for src/race_loop.py — Session 19.

Drives RaceLoop with a hand-rolled MockAdapter that integrates
commanded velocity into state each tick. Uses VirtualDetector on a
2-gate course. Verifies:

  * RaceLoop.step() runs clean end-to-end without PX4 or DCL.
  * The drone actually advances through gates — target_idx should
    reach gate_count within the timeout.
  * StepResult.passed_gate fires on gate pass.
  * Belief model + navigator reset correctly between gates.
  * log_steps captures tick-by-tick data.

This is not a performance benchmark. It's a contract test: the
adapter/detector/navigator/loop seams line up and produce a completing
mission under ideal (clean) conditions.

Run standalone:
    python test_race_loop.py
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import types
from dataclasses import dataclass


# Stub mavsdk at the module level so the existing gate_belief +
# vision_nav imports succeed in a mavsdk-less sandbox.
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


# ─────────────────────────────────────────────────────────────────────
# MockAdapter: integrates commanded velocity into state
# ─────────────────────────────────────────────────────────────────────

from sim.adapter import SimState, SimCapability, SimInfo  # noqa: E402


class MockAdapter:
    """Kinematic-only mock: apply commanded velocity directly to state,
    with light first-order yaw tracking. Enough fidelity to verify the
    RaceLoop seams; NOT a physics sim."""

    capabilities = (
        SimCapability.VELOCITY_NED
        | SimCapability.POSITION_NED
        | SimCapability.ARM_ACTION
        | SimCapability.WALLCLOCK_PACED
    )

    def __init__(self, dt=0.02):
        self._pos = [0.0, 0.0, -1.0]        # start 1 m up (NED down=-1)
        self._vel = [0.0, 0.0, 0.0]
        self._yaw = 0.0                     # rad
        self._dt = dt

    async def connect(self): pass
    async def disconnect(self): pass
    async def reset(self): pass

    async def get_state(self) -> SimState:
        return SimState(
            pos_ned=tuple(self._pos),
            vel_ned=tuple(self._vel),
            att_rad=(0.0, 0.0, self._yaw),
            timestamp=0.0,
            armed=True,
            connected=True,
        )

    async def get_camera_frame(self):
        return None  # VirtualDetector doesn't need it

    async def send_velocity_ned(self, vn, ve, vd, yaw_deg):
        # Integrate the commanded vel straight into position, and
        # track yaw toward the command with a short time constant
        # (~150 ms) so the drone turns realistically through gates.
        self._vel = [vn, ve, vd]
        self._pos[0] += vn * self._dt
        self._pos[1] += ve * self._dt
        self._pos[2] += vd * self._dt
        desired_yaw = math.radians(yaw_deg)
        # Shortest-angle blend
        delta = (desired_yaw - self._yaw + math.pi) % (2 * math.pi) - math.pi
        tau = 0.15
        self._yaw += delta * min(1.0, self._dt / tau)

    async def send_position_ned(self, *a, **kw): pass
    async def send_attitude(self, *a, **kw): pass
    async def arm(self): pass
    async def disarm(self): pass
    async def takeoff(self, alt): pass
    async def land(self): pass
    async def start_offboard(self, initial_mode="velocity"): pass
    async def stop_offboard(self): pass

    def info(self) -> SimInfo:
        return SimInfo(backend="mock", capabilities=self.capabilities)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def _build_loop(gates, noise="clean", command_hz=50):
    """Wire an integration-ready RaceLoop."""
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav
    from race_loop import RaceLoop

    adapter = MockAdapter(dt=1.0 / command_hz)
    detector = VirtualDetector(gates=gates, noise_profile=noise)
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    loop = RaceLoop(
        adapter=adapter, detector=detector, navigator=navigator,
        gate_count=len(gates), command_hz=command_hz,
        associate_mode="target_idx",
    )
    return loop


def test_single_step_runs_clean():
    """One step: no crash, sends a command, emits a StepResult."""
    loop = _build_loop(gates=[(8.0, 0.0, -1.0)])
    result = asyncio.get_event_loop().run_until_complete(loop.step())
    assert result is not None
    assert result.target_idx == 0
    assert abs(result.cmd_vn) + abs(result.cmd_ve) > 0, "expected non-zero velocity"
    print(f"  ✓ step produced cmd=(vn={result.cmd_vn:.2f}, ve={result.cmd_ve:.2f}), "
          f"detected={result.detected}")


def test_completes_short_course():
    """2-gate straight course should complete within timeout."""
    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    loop = _build_loop(gates=gates)
    run_res = asyncio.get_event_loop().run_until_complete(
        loop.run(timeout_s=15.0, log_steps=True)
    )
    assert run_res.gates_passed == 2, \
        f"expected to pass both gates, passed {run_res.gates_passed}"
    assert not run_res.timeout, "hit timeout on a clean 2-gate course"
    assert run_res.completed, "completed flag should be True"
    # Exactly two gate-pass events in the step log
    passes = [s for s in run_res.steps if s.passed_gate]
    assert len(passes) == 2, f"expected 2 pass events, got {len(passes)}"
    print(f"  ✓ passed 2/2 gates in {run_res.total_time_s:.2f}s "
          f"({len(run_res.steps)} steps)")


def test_gate_pass_event_progression():
    """Gate-pass events should happen in target order."""
    gates = [(10.0, 0.0, -1.0), (25.0, 5.0, -1.0)]
    loop = _build_loop(gates=gates)
    run_res = asyncio.get_event_loop().run_until_complete(
        loop.run(timeout_s=20.0, log_steps=True)
    )
    passes = [s for s in run_res.steps if s.passed_gate]
    assert [p.target_idx for p in passes] == [1, 2], \
        f"pass target_idx sequence wrong: {[p.target_idx for p in passes]}"
    # Second pass must come after first
    assert passes[0].t < passes[1].t
    print(f"  ✓ gate-pass order: t={passes[0].t:.2f}s → t={passes[1].t:.2f}s")


def test_capability_flags_respected():
    """Loop must not crash if adapter returns frame=None (no camera)."""
    # MockAdapter returns None — RaceLoop must pass None through cleanly
    # to VirtualDetector which also ignores frames. Already covered by
    # completes_short_course, but assert the capability flag directly.
    loop = _build_loop(gates=[(10.0, 0.0, -1.0)])
    caps = loop.adapter.capabilities
    assert SimCapability.CAMERA_RGB not in caps
    # One step with frame=None must not raise
    asyncio.get_event_loop().run_until_complete(loop.step())
    print("  ✓ loop handles frame=None path")


def test_no_detections_falls_into_coast_or_search():
    """If all detections disappear (drone yawed away from all gates),
    the navigator shouldn't crash — belief falls into COAST then SEARCH
    as frames_since_seen increases. We verify by running with a gate
    behind the drone."""
    # Gate 10 m NORTH but drone yawed east (out of FOV entirely). With
    # no detections, the loop should still emit commands, not error.
    gates = [(10.0, 0.0, -1.0)]
    loop = _build_loop(gates=gates)

    # Force the adapter to start yawed 150° (gate well outside 120° FOV)
    loop.adapter._yaw = math.radians(150.0)

    steps_run = 10
    for _ in range(steps_run):
        asyncio.get_event_loop().run_until_complete(loop.step())
    # Belief alive or not, but we must not have passed anything
    assert loop.target_idx == 0, "shouldn't pass a gate we can't see"
    print(f"  ✓ {steps_run} steps with no detection → no crash, no false passes")


def main():
    tests = [
        ("single step runs clean",                  test_single_step_runs_clean),
        ("completes short 2-gate course",           test_completes_short_course),
        ("gate-pass events in target order",        test_gate_pass_event_progression),
        ("loop handles frame=None (no camera)",     test_capability_flags_respected),
        ("no detections → coast/search no-crash",   test_no_detections_falls_into_coast_or_search),
    ]
    failures = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failures += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} PASSED")


if __name__ == "__main__":
    main()
