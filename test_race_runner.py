"""Integration tests for src/race/runner.py — Session 19.

Verifies the RaceRunner lifecycle:
  * fly() calls connect → arm → takeoff → start_offboard → run → land → disconnect
    in order, with the right args, only when the adapter supports them.
  * Adapters without ARM_ACTION (gym-style) skip arm/takeoff/land and
    jump straight to start_offboard → run.
  * Teardown runs on both success and exception paths.
  * FlightResult carries the right flags.

Uses a spying MockAdapter that records the method call sequence so the
lifecycle ordering is easy to assert.

Run standalone:
    python test_race_runner.py
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import types


# Stub mavsdk so gate_belief + vision_nav import cleanly.
if "mavsdk" not in sys.modules:
    m = types.ModuleType("mavsdk")
    o = types.ModuleType("mavsdk.offboard")

    class _S:
        def __init__(self, *a, **k): pass
    m.System = _S

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


from sim.adapter import SimState, SimCapability, SimInfo  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Spying MockAdapter — records the call sequence
# ─────────────────────────────────────────────────────────────────────

class SpyAdapter:
    """Kinematic mock + call-log for lifecycle assertions."""

    def __init__(
        self,
        dt: float = 0.02,
        capabilities=None,
        backend: str = "spy",
        fail_at: str | None = None,
    ):
        self._pos = [0.0, 0.0, -1.0]
        self._vel = [0.0, 0.0, 0.0]
        self._yaw = 0.0
        self._dt = dt
        self.capabilities = (
            capabilities if capabilities is not None
            else (SimCapability.VELOCITY_NED | SimCapability.ARM_ACTION)
        )
        self._backend = backend
        self._fail_at = fail_at
        self.calls: list[tuple[str, tuple]] = []

    def _log(self, name: str, *args):
        self.calls.append((name, args))
        if self._fail_at == name:
            raise RuntimeError(f"forced failure in {name}()")

    async def connect(self): self._log("connect")
    async def disconnect(self): self._log("disconnect")
    async def reset(self): self._log("reset")

    async def get_state(self):
        return SimState(
            pos_ned=tuple(self._pos),
            vel_ned=tuple(self._vel),
            att_rad=(0.0, 0.0, self._yaw),
            timestamp=0.0, armed=True, connected=True,
        )

    async def get_camera_frame(self):
        return None

    async def send_velocity_ned(self, vn, ve, vd, yaw_deg):
        self._vel = [vn, ve, vd]
        self._pos[0] += vn * self._dt
        self._pos[1] += ve * self._dt
        self._pos[2] += vd * self._dt
        desired = math.radians(yaw_deg)
        delta = (desired - self._yaw + math.pi) % (2 * math.pi) - math.pi
        self._yaw += delta * min(1.0, self._dt / 0.15)
        # Don't log every tick — would drown the call log.

    async def send_position_ned(self, *a, **k): self._log("send_position_ned", *a)
    async def send_attitude(self, *a, **k): self._log("send_attitude", *a)
    async def arm(self): self._log("arm")
    async def disarm(self): self._log("disarm")
    async def takeoff(self, alt): self._log("takeoff", alt)
    async def land(self): self._log("land")
    async def start_offboard(self, initial_mode="velocity"): self._log("start_offboard", initial_mode)
    async def stop_offboard(self): self._log("stop_offboard")

    def info(self):
        return SimInfo(backend=self._backend, capabilities=self.capabilities)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _build_runner(adapter, gates, noise="clean"):
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav
    from race.runner import RaceRunner

    detector = VirtualDetector(gates=gates, noise_profile=noise)
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    return RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=gates, takeoff_altitude_m=2.0, command_hz=50,
    )


def _call_names(calls):
    return [name for (name, _args) in calls]


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_full_lifecycle_with_arm_action():
    """connect → arm → takeoff → start_offboard → land → stop_offboard →
    disconnect, in that order, with takeoff_alt passed through."""
    adapter = SpyAdapter()
    runner = _build_runner(adapter, gates=[(8.0, 0.0, -1.0)])
    result = asyncio.get_event_loop().run_until_complete(runner.fly(timeout_s=10.0))

    names = _call_names(adapter.calls)
    # Required subsequence (ignores ticks / get_state / send_velocity):
    expected_order = [
        "connect", "arm", "takeoff", "start_offboard",
        "stop_offboard", "land", "disconnect",
    ]
    indices = []
    for step in expected_order:
        try:
            indices.append(names.index(step))
        except ValueError:
            raise AssertionError(f"missing {step!r} in call log: {names}")
    assert indices == sorted(indices), \
        f"lifecycle out of order: {list(zip(expected_order, indices))}"

    # takeoff altitude passed through
    takeoff_call = next(c for c in adapter.calls if c[0] == "takeoff")
    assert takeoff_call[1] == (2.0,), f"takeoff args: {takeoff_call[1]}"

    assert result.took_off is True
    assert result.landed is True
    assert result.run.completed
    assert result.backend == "spy"
    print(f"  ✓ lifecycle: {expected_order}, took off & landed, "
          f"ran {len(result.run.steps)} steps")


def test_no_arm_capability_skips_arm_takeoff_land():
    """Gym-style adapter (no ARM_ACTION): skip arm/takeoff/land; go
    straight from connect → start_offboard → run → stop_offboard →
    disconnect. Matches the expected DCL path."""
    adapter = SpyAdapter(
        capabilities=(SimCapability.VELOCITY_NED | SimCapability.CAMERA_RGB),
        backend="mock_dcl",
    )
    runner = _build_runner(adapter, gates=[(8.0, 0.0, -1.0)])
    result = asyncio.get_event_loop().run_until_complete(runner.fly(timeout_s=10.0))

    names = _call_names(adapter.calls)
    assert "arm" not in names, f"arm called on no-ARM_ACTION adapter: {names}"
    assert "takeoff" not in names, "takeoff called on no-ARM_ACTION adapter"
    assert "land" not in names, "land called on no-ARM_ACTION adapter"
    # Must still have connected and started offboard
    assert names.index("connect") < names.index("start_offboard")
    assert "disconnect" in names

    assert result.took_off is False
    assert result.landed is False
    assert result.run.completed
    print(f"  ✓ gym-style: arm/takeoff/land skipped, ran "
          f"{len(result.run.steps)} steps")


def test_teardown_runs_even_when_inner_loop_throws():
    """If something inside RaceLoop.run() raises, the runner must still
    stop_offboard + land + disconnect. We force it by yanking the
    navigator's plan() to raise."""
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav
    from race.runner import RaceRunner

    adapter = SpyAdapter()
    # Navigator that raises on first plan()
    class BoomNav(BeliefNav):
        def plan(self, *a, **k):
            raise RuntimeError("boom")
    navigator = BoomNav(max_speed=10.0, cruise_speed=8.0)
    detector = VirtualDetector(gates=[(8.0, 0.0, -1.0)], noise_profile="clean")
    runner = RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=[(8.0, 0.0, -1.0)], command_hz=50,
    )

    loop = asyncio.get_event_loop()
    raised = False
    try:
        loop.run_until_complete(runner.fly(timeout_s=2.0))
    except RuntimeError as e:
        assert "boom" in str(e)
        raised = True
    assert raised, "inner exception should have propagated"

    names = _call_names(adapter.calls)
    for step in ("stop_offboard", "land", "disconnect"):
        assert step in names, \
            f"teardown step {step!r} skipped after inner failure: {names}"
    print("  ✓ teardown runs on inner exception; stop_offboard + land + disconnect called")


def test_teardown_errors_do_not_mask_primary():
    """A teardown method that raises must be swallowed; primary error
    still propagates with its original type."""
    from race.runner import RaceRunner
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav

    # Make the adapter's `land()` throw — but we also need the inner
    # loop to raise so there's a primary. Use the BoomNav pattern.
    class BoomNav(BeliefNav):
        def plan(self, *a, **k):
            raise RuntimeError("primary boom")

    adapter = SpyAdapter(fail_at="land")
    runner = RaceRunner(
        adapter=adapter,
        detector=VirtualDetector(gates=[(8.0, 0.0, -1.0)], noise_profile="clean"),
        navigator=BoomNav(max_speed=10.0, cruise_speed=8.0),
        gates=[(8.0, 0.0, -1.0)], command_hz=50,
    )

    loop = asyncio.get_event_loop()
    raised_msg = None
    try:
        loop.run_until_complete(runner.fly(timeout_s=2.0))
    except RuntimeError as e:
        raised_msg = str(e)
    assert raised_msg == "primary boom", \
        f"expected primary to propagate, got {raised_msg!r}"
    # disconnect must still have run even though land raised
    assert "disconnect" in _call_names(adapter.calls)
    print("  ✓ teardown error swallowed; primary propagates; disconnect still ran")


def test_fusion_runner_completes_on_mock_kinematic():
    """Full lifecycle with pose_fusion turned on against the real
    MockKinematicAdapter (which actually synthesizes IMU). Runner should
    complete the course, set fusion_on=True on the result, and the
    telemetry should show both IMU samples and accepted vision fixes."""
    from race.runner import RaceRunner
    from sim.mock import MockKinematicAdapter
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav
    from estimation import PoseFusion

    gates = [(10.0, 0.0, -1.0), (25.0, 0.0, -1.0)]
    adapter = MockKinematicAdapter(
        dt=1.0 / 50,
        vel_tau=0.05,
        yaw_tau=0.10,
        auto_step=True,
        initial_altitude_m=1.0,
    )
    detector = VirtualDetector(gates=gates, noise_profile="clean")
    navigator = BeliefNav(max_speed=10.0, cruise_speed=8.0)
    pf = PoseFusion()

    runner = RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=gates, command_hz=50, pose_fusion=pf,
        vision_pos_sigma=0.20,
    )
    result = asyncio.get_event_loop().run_until_complete(
        runner.fly(timeout_s=15.0)
    )
    assert result.fusion_on is True
    assert result.run.gates_passed == 2, \
        f"fusion runner did not complete: {result.run.gates_passed}/2"
    tel = pf.telemetry
    assert tel.imu_samples_seen > 50, \
        f"IMU stream wasn't flowing: {tel.imu_samples_seen}"
    assert tel.vision_fixes_accepted > 0, \
        f"no vision fixes accepted: {tel.vision_fixes_accepted}"
    print(f"  ✓ fusion runner: {result.run.gates_passed}/2 in "
          f"{result.run.total_time_s:.2f}s, imu={tel.imu_samples_seen}, "
          f"vis_ok={tel.vision_fixes_accepted}")


def test_fusion_rejects_adapter_without_imu_capability():
    """Runner must fail loud before side effects if pose_fusion is
    supplied but the adapter doesn't advertise IMU. This catches the
    'flip --fusion on against PX4' failure mode early — you'd otherwise
    spend a whole flight with get_imu() returning None every tick."""
    from race.runner import RaceRunner
    from vision.detector import VirtualDetector
    from gate_belief import BeliefNav
    from estimation import PoseFusion

    # SpyAdapter has VELOCITY_NED|ARM_ACTION — no IMU. Perfect.
    adapter = SpyAdapter()
    runner = RaceRunner(
        adapter=adapter,
        detector=VirtualDetector(gates=[(8.0, 0.0, -1.0)], noise_profile="clean"),
        navigator=BeliefNav(max_speed=10.0, cruise_speed=8.0),
        gates=[(8.0, 0.0, -1.0)], command_hz=50,
        pose_fusion=PoseFusion(),
    )
    loop = asyncio.get_event_loop()
    raised = False
    try:
        loop.run_until_complete(runner.fly(timeout_s=2.0))
    except RuntimeError as e:
        msg = str(e)
        assert "IMU" in msg or "imu" in msg, f"wrong error message: {e}"
        raised = True
    assert raised, "runner should reject fusion against IMU-less adapter"
    # Importantly: no side effects should have fired (no connect, etc.)
    names = _call_names(adapter.calls)
    assert "connect" not in names, \
        f"adapter touched before validation failed: {names}"
    print("  ✓ rejects fusion on IMU-less adapter before connect")


def test_course_import_is_mavsdk_free():
    """Importing courses must not pull in mavsdk. Critical because
    run_race.py will import it at startup and shouldn't force a
    mavsdk dependency on DCL-only runs."""
    import importlib
    # Pre-check: mavsdk IS in sys.modules because we stubbed it above.
    # We can't assert absence globally, but we CAN assert that
    # src/courses.py doesn't import mavsdk by inspecting its AST.
    import ast, pathlib
    src = pathlib.Path(_HERE, "src", "courses.py").read_text()
    tree = ast.parse(src)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "mavsdk" not in imports, \
        f"courses.py imports mavsdk: {imports}"
    # And it loads
    from courses import COURSES, list_courses, get_course
    assert "technical" in list_courses()
    assert len(get_course("technical")) == 12
    print(f"  ✓ courses.py is mavsdk-free; {len(COURSES)} courses available")


def main():
    tests = [
        ("full lifecycle with ARM_ACTION",            test_full_lifecycle_with_arm_action),
        ("no ARM_ACTION skips arm/takeoff/land",      test_no_arm_capability_skips_arm_takeoff_land),
        ("teardown runs on inner exception",          test_teardown_runs_even_when_inner_loop_throws),
        ("teardown errors don't mask primary",        test_teardown_errors_do_not_mask_primary),
        ("fusion runner completes on mock_kinematic", test_fusion_runner_completes_on_mock_kinematic),
        ("fusion rejects IMU-less adapter",           test_fusion_rejects_adapter_without_imu_capability),
        ("courses module is mavsdk-free",             test_course_import_is_mavsdk_free),
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
