"""Contract tests for the DCLSimAdapter stub.

Until the DCL AI Race League SDK ships (~May 2026), `DCLSimAdapter` is a
stub that raises NotImplementedError with structured hints on every
work method. These tests exist so the stub cannot silently drift into a
no-op state during future refactors — a no-op stub would let the race
stack ship "working" against DCL by accident, which would only surface
in-sim.

When the real SDK lands and the stub is filled in, most of these tests
should INVERT: assert the method runs without raising. See
docs/DCL_INTEGRATION_CHECKLIST.md for the day-1 punch list.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sim.adapter import (  # noqa: E402
    DCLSimAdapter,
    SimCapability,
    SimInfo,
    make_adapter,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _run(coro):
    """Minimal async runner for the tests that call async methods."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _assert_raises_not_implemented(coro, method_name: str) -> None:
    try:
        _run(coro)
    except NotImplementedError as e:
        # The hint message should name the adapter + method so debugging
        # in a real race is direct. Not a strict format check — just a
        # sanity guard that the hint wasn't stripped.
        msg = str(e)
        assert "DCLSimAdapter" in msg or method_name in msg, (
            f"{method_name} raised NotImplementedError but the hint is "
            f"missing both 'DCLSimAdapter' and the method name: {msg!r}"
        )
        return
    raise AssertionError(
        f"Expected {method_name} to raise NotImplementedError (stub), "
        f"but it completed without raising. Was the stub silently "
        f"replaced with a no-op?"
    )


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_capabilities_include_the_minimum_viable_set() -> None:
    """DCL must advertise VELOCITY_NED + CAMERA_RGB + IMU for the hot path.

    If any of these get stripped, consumers that gate on capabilities
    (RaceRunner's fusion validation, detector wiring) will start failing
    loud instead of passing a misconfigured race through.
    """
    a = DCLSimAdapter()
    assert SimCapability.VELOCITY_NED in a.capabilities, (
        "Hot-path command disappeared from DCL capabilities."
    )
    assert SimCapability.CAMERA_RGB in a.capabilities, (
        "CAMERA_RGB required for vision — the whole point of running "
        "against DCL vs PX4 SITL is that DCL has a camera."
    )
    assert SimCapability.IMU in a.capabilities, (
        "IMU required for the --fusion code path."
    )
    assert SimCapability.RESET in a.capabilities, (
        "RESET required for bench-style between-race isolation."
    )


def test_arm_action_is_NOT_advertised() -> None:
    """Gym-style sims don't model arm/takeoff/land.

    RaceRunner uses `SimCapability.ARM_ACTION` to decide whether to run
    the arm → takeoff ramp. DCL leaves these as no-ops, so the flag
    must stay off or RaceRunner will call arm() → pass (silent), then
    takeoff() → pass (silent), then try to race a drone that may be
    ground-fixed.
    """
    a = DCLSimAdapter()
    assert SimCapability.ARM_ACTION not in a.capabilities, (
        "ARM_ACTION was added to DCL capabilities — but the stub methods "
        "are no-ops. If DCL actually exposes arm/takeoff/land on the "
        "real SDK, fill the methods AND add the flag together. Never "
        "one without the other."
    )


def test_info_returns_stub_marker() -> None:
    """info() should signal 'this is still a stub' so anything reading
    the SimInfo notes can tell."""
    a = DCLSimAdapter()
    info = a.info()
    assert isinstance(info, SimInfo)
    assert info.backend == "dcl"
    assert "STUB" in info.notes.upper() or "AWAITING" in info.notes.upper(), (
        f"info().notes no longer marks DCL as a stub: {info.notes!r}. "
        f"If the real SDK has landed, flip the marker AND this test."
    )


def test_hot_path_methods_raise_not_implemented() -> None:
    """Every method on the hot-path still raises the documented stub
    error. If any silently becomes a no-op, RaceLoop will appear to run
    but produce nonsense (empty state, no camera, no commands reaching
    the sim)."""
    a = DCLSimAdapter()
    _assert_raises_not_implemented(a.connect(), "connect")
    _assert_raises_not_implemented(a.disconnect(), "disconnect")
    _assert_raises_not_implemented(a.reset(), "reset")
    _assert_raises_not_implemented(a.get_state(), "get_state")
    _assert_raises_not_implemented(a.get_camera_frame(), "get_camera_frame")
    _assert_raises_not_implemented(a.get_imu(), "get_imu")
    _assert_raises_not_implemented(
        a.send_velocity_ned(0.0, 0.0, 0.0, 0.0), "send_velocity_ned"
    )
    _assert_raises_not_implemented(
        a.send_position_ned(0.0, 0.0, 0.0, 0.0), "send_position_ned"
    )
    _assert_raises_not_implemented(
        a.send_attitude(0.0, 0.0, 0.0, 0.5), "send_attitude"
    )


def test_action_methods_are_no_ops() -> None:
    """arm/disarm/takeoff/land/offboard are intentionally no-ops on DCL
    (gym-style drones are airborne at reset). These MUST complete
    without raising so RaceRunner's lifecycle wrapper doesn't blow up
    on DCL when `ARM_ACTION` happens to leak into capabilities."""
    a = DCLSimAdapter()
    _run(a.arm())
    _run(a.disarm())
    _run(a.takeoff(2.0))
    _run(a.land())
    _run(a.start_offboard())
    _run(a.stop_offboard())
    # If we got here without an exception, the no-ops held.


def test_factory_builds_dcl_adapter() -> None:
    """`make_adapter('dcl')` should return a DCLSimAdapter instance.
    Loose check — we don't want to import PX4 machinery or hit network
    just to validate the factory maps the string."""
    a = make_adapter("dcl")
    assert isinstance(a, DCLSimAdapter)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        ("DCL capabilities include minimum viable set",
         test_capabilities_include_the_minimum_viable_set),
        ("ARM_ACTION is NOT advertised (no-op guard)",
         test_arm_action_is_NOT_advertised),
        ("info() still marks DCL as a stub",
         test_info_returns_stub_marker),
        ("hot-path methods raise NotImplementedError",
         test_hot_path_methods_raise_not_implemented),
        ("action methods are no-ops (don't raise)",
         test_action_methods_are_no_ops),
        ("make_adapter('dcl') returns DCLSimAdapter",
         test_factory_builds_dcl_adapter),
    ]
    passed = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAILED — {e}")
            return 1
        except Exception as e:
            print(f"  ✗ ERROR — {type(e).__name__}: {e}")
            return 1
        print(f"  ✓ {name}")
        passed += 1
    print(f"\n{passed}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
