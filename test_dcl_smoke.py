"""End-to-end smoke test for the `--backend mock_dcl` CLI path.

This test would have caught both S19u bugs pre-commit:
  * `MockKinematicAdapter.reset()` dropping `auto_step` + `initial_altitude_m`
  * `RaceRunner` passing `gates_ned=None` in non-fusion mode

Purpose: every commit touching `run_race.py`, `src/race/runner.py`, the
sim adapter layer, or the mock kinematics should be followed by
`python test_dcl_smoke.py` before push. Completes in ~15 seconds on
the sandbox; cheap enough to run as a pre-push gate.

Runs the full CLI chain — argument parser, backend factory, detector
factory, RaceRunner lifecycle, RaceLoop ticks, flight summary print —
via subprocess. We do NOT import `run_race.main()` and call it
directly: the whole point is to exercise what a user would actually
run, including mavsdk-stub logic and the asyncio loop bring-up.

When the real DCL SDK lands, add a parallel `test_dcl_real_smoke.py`
that runs `--backend dcl` against whatever the minimum-viable scenario
is. This file stays as the always-green hermetic version.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _run_race(*args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    """Invoke `python run_race.py` with the given args; return
    (returncode, stdout, stderr). 60 s wall-clock budget is generous —
    the technical course typically completes in 13 s."""
    cmd = [sys.executable, str(_REPO / "run_race.py"), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_gates_passed(stdout: str) -> tuple[int, int]:
    """Pull the `Gates passed : N/M` line out of the summary block."""
    for line in stdout.splitlines():
        if line.strip().startswith("Gates passed"):
            # Format: "Gates passed  : 12/12"
            frac = line.split(":", 1)[1].strip()
            num, denom = frac.split("/")
            return int(num), int(denom)
    raise AssertionError(
        f"No 'Gates passed' line in stdout. Did the CLI change its "
        f"summary format? stdout:\n{stdout}"
    )


def _parse_time_s(stdout: str) -> float:
    """Pull the `Time : X.XX s` line. Strips the `(TIMEOUT)` suffix."""
    for line in stdout.splitlines():
        if line.strip().startswith("Time"):
            # "Time          : 13.04 s" or "... 30.01 s (TIMEOUT)"
            rhs = line.split(":", 1)[1].strip()
            num = rhs.split()[0]
            return float(num)
    raise AssertionError("No 'Time' line in stdout")


def _parse_completed(stdout: str) -> bool:
    for line in stdout.splitlines():
        if line.strip().startswith("Completed"):
            return "true" in line.split(":", 1)[1].strip().lower()
    raise AssertionError("No 'Completed' line in stdout")


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_mock_dcl_technical_course_completes() -> None:
    """The headline smoke test: mock_dcl + virtual detector + technical
    course should fly to completion in well under the timeout.

    This is the test that would have caught S19u bugs 1 and 2.
    Pre-S19u fix: 0/12, TIMEOUT. Post-fix: 12/12, ~13 s.
    """
    rc, out, err = _run_race(
        "--backend", "mock_dcl",
        "--detector", "virtual",
        "--course", "technical",
        "--timeout", "30",
    )
    gates_passed, gate_count = _parse_gates_passed(out)
    assert gate_count == 12, (
        f"Technical course no longer has 12 gates? Got {gate_count}. "
        f"Update this test if the course definition changed."
    )
    assert gates_passed == 12, (
        f"mock_dcl stalled at {gates_passed}/{gate_count}. This is "
        f"the S19u regression class: check `MockKinematicAdapter.reset()` "
        f"kwarg forwarding and `RaceRunner`'s `gates_ned` handling. "
        f"stderr:\n{err}\nstdout tail:\n{out[-800:]}"
    )
    assert _parse_completed(out), (
        f"Completed flag False despite 12/12 passes. FlightResult "
        f"may have been restructured. stdout:\n{out[-800:]}"
    )
    t = _parse_time_s(out)
    assert t < 20.0, (
        f"Technical course took {t:.1f} s — baseline is ~13 s. "
        f"Something slowed the loop down. Check `asyncio.sleep(dt)` "
        f"pacing and MockKinematicAdapter.auto_step."
    )
    # Sanity: the S19s UserWarning for `gates_ned=None` should NOT
    # fire in any production path. If it does, RaceRunner regressed
    # back to conditional `gates_ned`.
    assert "gates_ned" not in err or "minimal-integration" not in err, (
        f"S19s UserWarning fired — RaceRunner may have regressed to "
        f"`gates_ned=None` in non-fusion mode (S19u bug 2). stderr:\n{err}"
    )
    assert rc == 0, (
        f"run_race.py exited with code {rc}. Should be 0 on "
        f"successful race. stderr:\n{err}"
    )


def test_mock_backend_also_completes() -> None:
    """Regression guard on `--backend mock` — S19u bug 2 also bit this
    path (stall at 2/12). Keeping both backends covered prevents
    future refactors from un-fixing the general case while leaving
    mock_dcl alone."""
    rc, out, err = _run_race(
        "--backend", "mock",
        "--detector", "virtual",
        "--course", "technical",
        "--timeout", "30",
    )
    gates_passed, gate_count = _parse_gates_passed(out)
    assert gates_passed == gate_count == 12, (
        f"`--backend mock` regression: {gates_passed}/{gate_count}. "
        f"If mock_dcl passes but mock fails, the bug is backend-"
        f"specific and not in RaceRunner. stderr:\n{err}"
    )
    assert rc == 0


def test_fusion_backend_still_completes() -> None:
    """Fusion path regression guard. The S19u fix to RaceRunner
    changed the `gates_ned` handling; this confirms the fusion path
    still works (it always got `gates_ned` before, but the post-fix
    path is unconditional so it's worth re-checking)."""
    rc, out, err = _run_race(
        "--backend", "mock_kinematic",
        "--detector", "virtual",
        "--course", "technical",
        "--fusion",
        "--vision-pos-sigma", "1.0",
        "--timeout", "30",
    )
    gates_passed, gate_count = _parse_gates_passed(out)
    assert gates_passed == gate_count == 12, (
        f"Fusion path regression: {gates_passed}/{gate_count}. "
        f"The S19u RaceRunner change should NOT have affected fusion "
        f"behaviour (gates_ned was always passed in that branch). "
        f"stderr:\n{err}"
    )
    assert rc == 0


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        ("mock_dcl + virtual completes technical course (S19u bug repro)",
         test_mock_dcl_technical_course_completes),
        ("mock + virtual completes technical course",
         test_mock_backend_also_completes),
        ("mock_kinematic + --fusion still completes",
         test_fusion_backend_still_completes),
    ]
    passed = 0
    for name, fn in tests:
        print(f"• {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAILED — {e}")
            return 1
        except subprocess.TimeoutExpired as e:
            print(f"  ✗ TIMEOUT — subprocess hung: {e}")
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
