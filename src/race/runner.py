"""RaceRunner — full flight lifecycle over a SimAdapter — Session 19.

Wraps the perception + planning + control inner loop (RaceLoop) with
the vehicle lifecycle: connect, arm, takeoff, start offboard, race, land,
disconnect. This is the piece that makes `control_skeleton_v5.py`
obsolete as an entrypoint — anything going forward should construct a
RaceRunner with an injected adapter + detector + navigator.

Keep it dependency-light:
  * imports RaceLoop from the parent package
  * NEVER imports mavsdk directly — that's the adapter's job
  * no file I/O at import time (logging belongs to the caller)

Lifecycle rules, tuned for the current PX4 path and designed to also
work against DCL (which will probably no-op arm/takeoff/land):
  1. If the adapter advertises ARM_ACTION and isn't connected, do the
     full arm → takeoff → start_offboard ramp before handing to RaceLoop.
  2. If the adapter lacks ARM_ACTION (gym-style DCL), jump straight to
     start_offboard (which is also a no-op for DCL) and run the race.
  3. Always attempt to land + disconnect in a `finally`. Exceptions in
     land/disconnect are suppressed so they don't mask the primary error.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Make sibling package importable when executed from repo root or /src.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from race_loop import RaceLoop, RunResult  # noqa: E402
from sim.adapter import SimCapability       # noqa: E402


@dataclass
class FlightResult:
    """What the caller gets back after a full flight.

    Wraps RunResult with lifecycle metadata — whether takeoff/land
    completed, whether the adapter had a camera, whether we hit timeout.
    `fusion_on` is True when the loop planned off a fused pose rather
    than adapter truth.
    """
    run: RunResult
    took_off: bool
    landed: bool
    backend: str
    detector_name: str
    fusion_on: bool = False

    @property
    def completed(self) -> bool:
        return self.run.completed


class RaceRunner:
    """End-to-end race flight, backend-agnostic.

    Construct with already-built adapter, detector, and navigator. Call
    `await runner.fly()` once — it returns a FlightResult.
    """

    DEFAULT_TAKEOFF_ALT = 2.0
    DEFAULT_TIMEOUT_S = 120.0

    def __init__(
        self,
        adapter,
        detector,
        navigator,
        gates: List[tuple],
        takeoff_altitude_m: float = DEFAULT_TAKEOFF_ALT,
        command_hz: int = 50,
        associate_mode: str = "target_idx",
        pose_fusion=None,
        vision_pos_sigma: float = 0.15,
        gate_sequencer=None,
    ):
        """
        Args:
            pose_fusion: Optional PoseFusion instance. When supplied,
                RaceLoop drives it with adapter IMU + backprojected
                vision fixes and plans off the fused pose instead of
                adapter truth. Requires the adapter to advertise
                `SimCapability.IMU`; otherwise the loop will call
                `adapter.get_imu()` and get None back every tick, which
                makes the filter useless. Validation happens in fly().
            vision_pos_sigma: 1-σ (m) noise on the backprojected
                drone-NED position fix; forwarded to RaceLoop.
        """
        self.adapter = adapter
        self.detector = detector
        self.navigator = navigator
        self.gates = gates
        self.takeoff_altitude_m = takeoff_altitude_m
        self.command_hz = command_hz
        self.associate_mode = associate_mode
        self.pose_fusion = pose_fusion
        self.vision_pos_sigma = float(vision_pos_sigma)
        self.gate_sequencer = gate_sequencer

    async def fly(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        log_steps: bool = False,
        realtime: bool = True,
    ) -> FlightResult:
        """
        Args:
            realtime: Forwarded to RaceLoop.run(). Sandbox soak (S19w)
                sets False to drop wallclock pacing — only safe against
                MockKinematicAdapter. See RaceLoop.run for caveats.
        """
        took_off = False
        landed = False
        run_res: Optional[RunResult] = None
        backend = getattr(self.adapter.info(), "backend", "?")
        error: Optional[BaseException] = None

        try:
            # ── 0. Validate fusion config before side effects ──────
            # Fail loud here rather than noticing at tick-1 that
            # get_imu() returns None every call.
            if self.pose_fusion is not None:
                if SimCapability.IMU not in self.adapter.capabilities:
                    raise RuntimeError(
                        f"pose_fusion supplied but adapter backend "
                        f"'{backend}' does not advertise SimCapability.IMU. "
                        f"Switch to mock_kinematic, or wait until "
                        f"PX4SITLAdapter.get_imu() is wired up."
                    )

            # ── 1. Connect ──────────────────────────────────────────
            await self.adapter.connect()

            # ── 2. Arm / takeoff (skipped if adapter doesn't support) ──
            if SimCapability.ARM_ACTION in self.adapter.capabilities:
                await self.adapter.arm()
                await self.adapter.takeoff(self.takeoff_altitude_m)
                took_off = True
            # DCL-style adapters without ARM_ACTION are assumed airborne.

            # ── 3. Start offboard (velocity-mode race loop) ────────
            await self.adapter.start_offboard(initial_mode="velocity")

            # ── 4. Run the race ─────────────────────────────────────
            # gates_ned is passed UNCONDITIONALLY. Earlier, this was
            # gated on `pose_fusion is not None`, which meant non-fusion
            # races got `gates_ned=None` and the S19o gate-aware
            # BeliefNav fallback could not fire — the drone would stall
            # at whichever gate the belief tracker lost track of (repro:
            # `--backend mock --detector virtual --course technical`,
            # 2/12 gates, timeout). Non-fusion mode uses adapter truth
            # for pos, so steering to a known world-frame gate position
            # is always safe. The fusion mode still works as before
            # (pose_trusted gates gate-aware nav).
            # When a GateSequencer is active (discovery mode), don't
            # pass gates_ned to the loop — the sequencer discovers gate
            # positions from vision. The loop will still get gate_count
            # from gates (or sequencer) for run termination.
            use_gates_ned = self.gates if self.gate_sequencer is None else None
            loop = RaceLoop(
                adapter=self.adapter,
                detector=self.detector,
                navigator=self.navigator,
                gate_count=len(self.gates),
                command_hz=self.command_hz,
                associate_mode=self.associate_mode,
                pose_fusion=self.pose_fusion,
                gates_ned=use_gates_ned,
                vision_pos_sigma=self.vision_pos_sigma,
                gate_sequencer=self.gate_sequencer,
            )
            run_res = await loop.run(
                timeout_s=timeout_s,
                log_steps=log_steps,
                realtime=realtime,
            )

        except BaseException as e:
            error = e

        finally:
            # Best-effort teardown. Swallow teardown exceptions so they
            # don't mask the primary error.
            try:
                await self.adapter.stop_offboard()
            except Exception:
                pass
            if took_off and SimCapability.ARM_ACTION in self.adapter.capabilities:
                try:
                    await self.adapter.land()
                    landed = True
                except Exception:
                    pass
            try:
                await self.adapter.disconnect()
            except Exception:
                pass

        if error is not None:
            # Re-raise the primary error after teardown has run.
            raise error

        assert run_res is not None  # reachable only on clean run
        return FlightResult(
            run=run_res,
            took_off=took_off,
            landed=landed,
            backend=backend,
            detector_name=self.detector.name(),
            fusion_on=self.pose_fusion is not None,
        )
