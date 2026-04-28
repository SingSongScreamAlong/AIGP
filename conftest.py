"""Pytest conftest — Session 19x.

Two jobs:

1. Stub ``mavsdk`` and ``mavsdk.offboard`` once per session so the
   repo's imports (``vision_nav``, ``gate_belief``, ``sim.adapter``,
   ``run_race``, ...) all succeed under a ``pytest`` invocation that
   collects multiple test files in one shot.

   The bug this fixes: several legacy tests (``test_gate_belief.py``,
   ``test_belief_replay.py``, ``test_belief_nav_gate_aware.py``) each
   install their own stub that sets ``VelocityNedYaw`` on
   ``mavsdk.offboard`` but never sets ``System`` on ``mavsdk``. Whichever
   of those lands first wins the ``if "mavsdk" not in sys.modules:``
   guard in every other test file; later files' full stubs are
   short-circuited; ``from mavsdk import System`` (executed by
   ``vision_nav.py``) then ``ImportError``s. In per-file runs, only one
   stub ever gets installed, so the bug is invisible. Under
   ``pytest`` / aggregate CI, 23 tests fail with a confusing import
   error. Installing the complete stub here, before any test module is
   imported, makes all individual per-test stubs idempotent no-ops.

2. Keep ``pytest`` (with no arguments) from erroring at collection time
   on files that depend on environment-only packages. These are valid
   on the dev PC but not in the sandbox:
     * ``lsy_drone_racing`` — gitignored submodule; sim-only tests
     * ``cv2`` / ``torch`` / ``ultralytics`` — GPU-stack deps

   A CI job with the right environment can pop items off
   ``collect_ignore_glob`` or override it.
"""

from __future__ import annotations

import sys
import types


# ── mavsdk stub ─────────────────────────────────────────────────────
# Must run before any test module is imported. conftest.py at repo root
# is loaded by pytest before it walks the test tree, so this is safe.

if "mavsdk" not in sys.modules:
    _m = types.ModuleType("mavsdk")
    _o = types.ModuleType("mavsdk.offboard")

    class _System:  # mavsdk.System
        def __init__(self, *a, **k):
            pass

    _m.System = _System

    class _VelocityNedYaw:  # mavsdk.offboard.VelocityNedYaw
        def __init__(self, vn, ve, vd, yd):
            self.north_m_s = vn
            self.east_m_s = ve
            self.down_m_s = vd
            self.yaw_deg = yd

    _o.VelocityNedYaw = _VelocityNedYaw

    # Other offboard classes the repo pulls in — full shape not needed
    # for the tests we run in sandbox; just make the names importable.
    for _name in ("PositionNedYaw", "Attitude", "ActuatorControl",
                  "AccelerationNed"):
        setattr(_o, _name, type(_name, (),
                                 {"__init__": lambda self, *a, **k: None}))

    _o.OffboardError = type("OffboardError", (Exception,), {})

    _m.offboard = _o
    sys.modules["mavsdk"] = _m
    sys.modules["mavsdk.offboard"] = _o


# ── Collection ignores ─────────────────────────────────────────────
# pytest uses ``collect_ignore_glob`` at the rootdir to skip files at
# collection time. These either need the ``lsy_drone_racing`` submodule
# (gitignored; lives only on the dev PC) or GPU libs (cv2/torch).

collect_ignore_glob = [
    # lsy_drone_racing submodule
    "ab_exit_s11_test.py",
    "s16_vision_test.py",
    "s18_belief_test.py",
    "test_connect.py",
    "test_render.py",
    # OpenCV / YOLO deps
    "test_classical_detector.py",
    "test_yolo_pipeline.py",
    # Submodule tree
    "sims/**",
]
