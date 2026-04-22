"""Tests for src/sim/adapter.py — Session 19 scaffold.

Checks we can enforce at import time, without PX4 or DCL actually running:

  * SimState defaults.
  * SimCapability flag combinations behave like bit sets.
  * make_adapter dispatches correctly and rejects unknown backends.
  * PX4SITLAdapter declares sensible capabilities and reports
    backend="px4_sitl" via info().
  * DCLSimAdapter capabilities include CAMERA_RGB + RESET (the two
    reasons to prefer it over PX4 once the real sim drops).
  * Every DCLSimAdapter method that claims to be a stub actually
    raises NotImplementedError rather than silently no-oping. The
    arm/takeoff/land family is allowed to pass because gym-style envs
    don't usually model those.
  * PX4 adapter construction does not perform any network I/O (i.e.
    connect() is the first call that talks to the wire).

Run standalone:
    python test_sim_adapter.py
"""

from __future__ import annotations

import asyncio
import sys
import types


# Stub mavsdk so the PX4 adapter can at least be constructed and
# inspected in the sandbox. Real connect() would still require the
# actual mavsdk wheel + a running PX4 SITL.
if "mavsdk" not in sys.modules:
    mavsdk_stub = types.ModuleType("mavsdk")
    offboard_stub = types.ModuleType("mavsdk.offboard")

    class _System:
        def __init__(self, *a, **kw):
            self.core = None
            self.telemetry = None
            self.action = None
            self.offboard = None

    class _VNY:
        def __init__(self, vn, ve, vd, yd): pass

    class _PNY:
        def __init__(self, n, e, d, yd): pass

    class _Att:
        def __init__(self, r, p, y, t): pass

    mavsdk_stub.System = _System
    offboard_stub.VelocityNedYaw = _VNY
    offboard_stub.PositionNedYaw = _PNY
    offboard_stub.Attitude = _Att
    offboard_stub.OffboardError = type("OffboardError", (Exception,), {})

    sys.modules["mavsdk"] = mavsdk_stub
    sys.modules["mavsdk.offboard"] = offboard_stub


# Make src/sim importable
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from sim.adapter import (  # noqa: E402
    SimState,
    SimCapability,
    SimInfo,
    PX4SITLAdapter,
    DCLSimAdapter,
    make_adapter,
)


def test_sim_state_defaults():
    s = SimState()
    assert s.pos_ned == (0.0, 0.0, 0.0)
    assert s.vel_ned == (0.0, 0.0, 0.0)
    assert s.att_rad == (0.0, 0.0, 0.0)
    assert s.armed is False
    assert s.connected is False
    print("  ✓ SimState defaults sensible")


def test_capability_flags_combine():
    c = SimCapability.VELOCITY_NED | SimCapability.CAMERA_RGB
    assert SimCapability.VELOCITY_NED in c
    assert SimCapability.CAMERA_RGB in c
    assert SimCapability.ATTITUDE not in c
    assert (c & SimCapability.VELOCITY_NED) == SimCapability.VELOCITY_NED
    print("  ✓ SimCapability flag math works")


def test_make_adapter_dispatch():
    a = make_adapter("px4_sitl")
    assert isinstance(a, PX4SITLAdapter)
    d = make_adapter("dcl", scenario="round1_simple", seed=7)
    assert isinstance(d, DCLSimAdapter)
    assert d.scenario == "round1_simple"
    assert d.seed == 7
    try:
        make_adapter("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("make_adapter accepted unknown backend")
    print("  ✓ make_adapter dispatches and rejects unknowns")


def test_px4_adapter_capabilities():
    a = PX4SITLAdapter()
    caps = a.capabilities
    assert SimCapability.VELOCITY_NED in caps
    assert SimCapability.POSITION_NED in caps
    assert SimCapability.ATTITUDE in caps
    assert SimCapability.ARM_ACTION in caps
    assert SimCapability.WALLCLOCK_PACED in caps
    # PX4 SITL via mavsdk has no camera, no scenario reset.
    assert SimCapability.CAMERA_RGB not in caps
    assert SimCapability.RESET not in caps

    info = a.info()
    assert isinstance(info, SimInfo)
    assert info.backend == "px4_sitl"
    print("  ✓ PX4 adapter declares the right capabilities")


def test_px4_adapter_no_io_on_construction():
    # Constructing the adapter must not touch the network. We verify
    # indirectly: nothing has been "connected" yet, and get_camera_frame
    # returns None synchronously-adapted-to-async without a live drone.
    a = PX4SITLAdapter()
    assert a._drone is None
    frame = asyncio.get_event_loop().run_until_complete(a.get_camera_frame())
    assert frame is None
    print("  ✓ PX4 adapter is inert until connect() is called")


def test_dcl_capabilities_include_camera_and_reset():
    d = DCLSimAdapter()
    caps = d.capabilities
    assert SimCapability.CAMERA_RGB in caps, \
        "DCL stub must advertise CAMERA_RGB — that's the whole point of using it."
    assert SimCapability.RESET in caps, \
        "DCL stub must advertise RESET — gym-style envs reset."
    assert d.info().backend == "dcl"
    print("  ✓ DCL adapter advertises camera + reset")


def test_dcl_stub_methods_fail_loudly():
    """DCL stub methods that do real work must raise NotImplementedError.
    No-op methods are explicitly allowed for arm/takeoff/land since those
    don't map to gym-style envs."""
    d = DCLSimAdapter()
    loop = asyncio.get_event_loop()

    must_raise = [
        ("connect",            d.connect()),
        ("disconnect",         d.disconnect()),
        ("reset",              d.reset()),
        ("get_state",          d.get_state()),
        ("get_camera_frame",   d.get_camera_frame()),
        ("send_velocity_ned",  d.send_velocity_ned(1, 0, 0, 0)),
        ("send_position_ned",  d.send_position_ned(0, 0, 0, 0)),
        ("send_attitude",      d.send_attitude(0, 0, 0, 0.5)),
    ]
    for name, coro in must_raise:
        try:
            loop.run_until_complete(coro)
        except NotImplementedError:
            continue
        except Exception as e:
            raise AssertionError(
                f"DCL stub {name}() raised {type(e).__name__} instead of "
                f"NotImplementedError: {e}"
            )
        else:
            raise AssertionError(
                f"DCL stub {name}() did not raise — would silently no-op in race"
            )

    # Allowed to pass-through (gym envs don't model arm/takeoff):
    allowed_passthrough = [
        ("arm",            d.arm()),
        ("disarm",         d.disarm()),
        ("takeoff",        d.takeoff(2.0)),
        ("land",           d.land()),
        ("start_offboard", d.start_offboard()),
        ("stop_offboard",  d.stop_offboard()),
    ]
    for name, coro in allowed_passthrough:
        try:
            loop.run_until_complete(coro)
        except Exception as e:
            raise AssertionError(
                f"DCL stub {name}() should be a harmless no-op, got {type(e).__name__}: {e}"
            )
    print("  ✓ DCL stub: real methods raise, flight-mode no-ops pass")


def test_protocol_method_set_matches():
    """Both adapters should expose the same method names. A method that
    exists on one but not the other is a bug — a call site that works
    with PX4 will crash at runtime on DCL, or vice versa."""
    px4_methods = {m for m in dir(PX4SITLAdapter) if not m.startswith("_")}
    dcl_methods = {m for m in dir(DCLSimAdapter) if not m.startswith("_")}
    # Both must be supersets of the core contract
    expected = {
        "connect", "disconnect", "reset",
        "get_state", "get_camera_frame",
        "send_velocity_ned", "send_position_ned", "send_attitude",
        "arm", "disarm", "takeoff", "land",
        "start_offboard", "stop_offboard",
        "info", "capabilities",
    }
    for method in expected:
        assert method in px4_methods, f"PX4 adapter missing {method}()"
        assert method in dcl_methods, f"DCL adapter missing {method}()"
    print(f"  ✓ Both adapters implement all {len(expected)} required members")


def main():
    tests = [
        ("SimState defaults",                       test_sim_state_defaults),
        ("SimCapability flag math",                 test_capability_flags_combine),
        ("make_adapter dispatch",                   test_make_adapter_dispatch),
        ("PX4 adapter capabilities",                test_px4_adapter_capabilities),
        ("PX4 adapter inert on construction",       test_px4_adapter_no_io_on_construction),
        ("DCL adapter advertises camera + reset",   test_dcl_capabilities_include_camera_and_reset),
        ("DCL stub methods fail loudly",            test_dcl_stub_methods_fail_loudly),
        ("Protocol method set matches",             test_protocol_method_set_matches),
    ]
    failures = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} PASSED")


if __name__ == "__main__":
    main()
