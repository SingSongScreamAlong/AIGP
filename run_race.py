"""Top-level race entrypoint — Session 19.

Replaces `python3 src/control/control_skeleton_v5.py` as the user-facing
way to run a race. Wires a SimAdapter + Detector + BeliefNav + RaceRunner
from command-line flags.

Examples:
    # Current dev path: PX4 SITL + synthetic vision on the technical course
    python3 run_race.py --backend px4_sitl --detector virtual --course technical

    # Real-vision path (requires trained YOLO model + a backend with CAMERA_RGB)
    python3 run_race.py --backend dcl --detector yolo_pnp \
                        --model-path src/vision/gate_yolo/models/gate_corners_v1.pt

    # Offline mock (no PX4, no DCL, no camera) — smoke tests only
    python3 run_race.py --backend mock --detector virtual --course sprint --timeout 30
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))


def build_adapter(backend: str, connection_string: str):
    """Construct a SimAdapter by backend name. MockAdapter is inline
    because it's only useful here (identical kinematics to the test
    harness)."""
    if backend == "mock":
        return _MockFlightAdapter()
    if backend == "mock_kinematic":
        # Kinematic mock with synthetic IMU — for --fusion smoke tests.
        from sim.adapter import make_adapter
        return make_adapter(
            "mock_kinematic",
            dt=1.0 / 50,
            vel_tau=0.05,
            yaw_tau=0.10,
            auto_step=True,
            initial_altitude_m=1.0,
        )
    if backend == "mock_dcl":
        # Pre-landing DCL stand-in. Advertises DCL's expected capability
        # shape (no ARM_ACTION, CAMERA_RGB returning placeholder frames).
        # See src/sim/mock_dcl.py + docs/DCL_INTEGRATION_CHECKLIST.md.
        from sim.adapter import make_adapter
        return make_adapter("mock_dcl")
    if backend in ("px4_sitl", "dcl"):
        # Lazy import: dcl doesn't need mavsdk; px4_sitl does.
        from sim.adapter import make_adapter
        if backend == "px4_sitl":
            return make_adapter("px4_sitl", connection_string=connection_string)
        return make_adapter("dcl")
    raise ValueError(f"Unknown backend {backend!r}")


def build_detector(kind: str, gates, noise_profile: str, model_path: str | None,
                   color_profiles=None):
    from vision.detector import make_detector
    if kind == "virtual":
        return make_detector("virtual", gates=gates, noise_profile=noise_profile)
    if kind == "yolo_pnp":
        if not model_path:
            raise ValueError("--model-path required for --detector yolo_pnp")
        return make_detector("yolo_pnp", model_path=model_path)
    if kind == "classical":
        profiles = color_profiles or ["any_bright"]
        return make_detector("classical", color_profiles=profiles)
    raise ValueError(f"Unknown detector {kind!r}")


# ─────────────────────────────────────────────────────────────────────
# MockFlightAdapter — same as test harness, exposed as a CLI backend
# ─────────────────────────────────────────────────────────────────────

class _MockFlightAdapter:
    """Kinematic-only mock for dry-run and smoke testing without PX4/DCL.

    Not part of the main src/sim/adapter.py module because this adapter
    is only useful in `run_race.py --backend mock` — production paths
    always use a real sim.
    """

    from sim.adapter import SimState, SimCapability, SimInfo
    capabilities = (
        SimCapability.VELOCITY_NED | SimCapability.POSITION_NED
        | SimCapability.ARM_ACTION | SimCapability.WALLCLOCK_PACED
    )

    def __init__(self, dt: float = 0.02):
        self._pos = [0.0, 0.0, -1.0]
        self._vel = [0.0, 0.0, 0.0]
        self._yaw = 0.0
        self._dt = dt

    async def connect(self): print("[mock] connect")
    async def disconnect(self): print("[mock] disconnect")
    async def reset(self): pass

    async def get_state(self):
        from sim.adapter import SimState
        return SimState(
            pos_ned=tuple(self._pos),
            vel_ned=tuple(self._vel),
            att_rad=(0.0, 0.0, self._yaw),
            timestamp=time.time(), armed=True, connected=True,
        )

    async def get_camera_frame(self): return None

    async def send_velocity_ned(self, vn, ve, vd, yaw_deg):
        self._vel = [vn, ve, vd]
        self._pos[0] += vn * self._dt
        self._pos[1] += ve * self._dt
        self._pos[2] += vd * self._dt
        desired = math.radians(yaw_deg)
        delta = (desired - self._yaw + math.pi) % (2 * math.pi) - math.pi
        self._yaw += delta * min(1.0, self._dt / 0.15)

    async def send_position_ned(self, *a, **k): pass
    async def send_attitude(self, *a, **k): pass
    async def arm(self): print("[mock] armed")
    async def disarm(self): pass
    async def takeoff(self, alt):
        self._pos[2] = -alt
        print(f"[mock] takeoff to {alt} m")
    async def land(self): print("[mock] land")
    async def start_offboard(self, initial_mode="velocity"):
        print(f"[mock] start_offboard ({initial_mode})")
    async def stop_offboard(self): pass

    def info(self):
        from sim.adapter import SimInfo
        return SimInfo(backend="mock", capabilities=self.capabilities,
                       notes="kinematic-only; no physics; for smoke tests.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="AI Grand Prix race runner — backend-agnostic entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend",
                   choices=("px4_sitl", "dcl", "mock", "mock_kinematic", "mock_dcl"),
                   default="mock",
                   help="Sim backend to fly against. mock_kinematic synthesizes "
                        "an IMU stream (needed for --fusion). mock_dcl mimics "
                        "DCL's expected shape for pre-landing smoke tests.")
    p.add_argument("--detector", choices=("virtual", "yolo_pnp", "classical"), default="virtual",
                   help="Perception backend. 'classical' uses HSV color "
                        "thresholding for VQ1 highlighted gates.")
    p.add_argument("--course", default="technical",
                   help="Course name (one of: sprint, technical, mixed).")
    p.add_argument("--noise", default="clean",
                   choices=("clean", "mild", "harsh", "brutal"),
                   help="Noise profile for VirtualDetector (ignored for yolo_pnp).")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Race timeout in seconds.")
    p.add_argument("--takeoff-alt", type=float, default=2.0, dest="takeoff_alt",
                   help="Takeoff altitude in meters (ignored if adapter lacks ARM_ACTION).")
    p.add_argument("--command-hz", type=int, default=50,
                   help="Velocity command rate.")
    p.add_argument("--max-speed", type=float, default=12.0,
                   help="Planner max speed.")
    p.add_argument("--cruise-speed", type=float, default=9.0,
                   help="Planner cruise speed.")
    p.add_argument("--connection", default="udpin://0.0.0.0:14540",
                   help="MAVSDK connection string (px4_sitl only).")
    p.add_argument("--model-path", default=None,
                   help="Path to YOLOv8-pose .pt weights (yolo_pnp detector only).")
    p.add_argument("--log-steps", action="store_true",
                   help="Capture per-tick StepResult entries in the result.")
    p.add_argument("--fusion", action="store_true",
                   help="Drive the navigator off a PoseFusion (ESKF) estimate "
                        "instead of adapter truth. Requires the backend to "
                        "advertise SimCapability.IMU (today: mock_kinematic).")
    p.add_argument("--vision-pos-sigma", type=float, default=0.15,
                   dest="vision_pos_sigma",
                   help="1-σ (m) noise on backprojected vision position fixes "
                        "when --fusion is set.")
    p.add_argument("--discovery", action="store_true",
                   help="Track-agnostic mode: discover gates from vision instead "
                        "of using hardcoded gate positions. Uses GateSequencer "
                        "for gate pass detection and suppression.")
    p.add_argument("--gate-count", type=int, default=None,
                   dest="gate_count",
                   help="Expected number of gates in discovery mode. If not set, "
                        "races until timeout. Ignored without --discovery.")
    args = p.parse_args(argv)

    # Stub mavsdk if backend != px4_sitl, so running --backend mock/dcl on
    # a machine without mavsdk installed works. Only do this if mavsdk is
    # not already importable to avoid clobbering a real install.
    if args.backend != "px4_sitl":
        try:
            import mavsdk  # noqa: F401
        except Exception:
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

    # Resolve course → gates
    from courses import get_course
    gates = get_course(args.course)
    print(f"Course '{args.course}': {len(gates)} gates")

    # Build the pieces
    adapter = build_adapter(args.backend, args.connection)
    detector = build_detector(args.detector, gates, args.noise, args.model_path)

    # Optional track-agnostic gate sequencer
    gate_sequencer = None
    if args.discovery:
        from race.gate_sequencer import GateSequencer
        gc = args.gate_count if args.gate_count else len(gates)
        gate_sequencer = GateSequencer(gate_count=gc)
        print(f"Discovery mode: GateSequencer active (gate_count={gc})")

    from gate_belief import BeliefNav
    navigator = BeliefNav(max_speed=args.max_speed, cruise_speed=args.cruise_speed)

    # Optional pose fusion — fail loud early if the user asked for it
    # against a backend that can't feed the filter.
    pose_fusion = None
    if args.fusion:
        from sim.adapter import SimCapability
        if SimCapability.IMU not in adapter.capabilities:
            print(
                f"ERROR: --fusion requires a backend with SimCapability.IMU. "
                f"'{args.backend}' does not. Use --backend mock_kinematic for "
                f"a sandbox run, or wait until PX4SITLAdapter.get_imu() is wired.",
                file=sys.stderr,
            )
            return 2
        from estimation import PoseFusion
        pose_fusion = PoseFusion()

    from race.runner import RaceRunner
    runner = RaceRunner(
        adapter=adapter, detector=detector, navigator=navigator,
        gates=gates, takeoff_altitude_m=args.takeoff_alt,
        command_hz=args.command_hz,
        pose_fusion=pose_fusion,
        vision_pos_sigma=args.vision_pos_sigma,
        gate_sequencer=gate_sequencer,
    )

    print(f"Backend : {adapter.info().backend} (caps={adapter.info().capabilities!s})")
    print(f"Detector: {detector.name()}")
    print(f"Planner : BeliefNav cruise={args.cruise_speed} max={args.max_speed}")
    print(f"Fusion  : {'ON (PoseFusion/ESKF)' if pose_fusion else 'off (adapter truth)'}")
    print()

    # Fly
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            runner.fly(timeout_s=args.timeout, log_steps=args.log_steps)
        )
    finally:
        loop.close()

    # Report
    r = result.run
    print()
    print("──── FLIGHT SUMMARY ────")
    print(f"Backend       : {result.backend}")
    print(f"Detector      : {result.detector_name}")
    print(f"Took off      : {result.took_off}")
    print(f"Landed        : {result.landed}")
    print(f"Gates passed  : {r.gates_passed}/{r.gate_count}")
    print(f"Time          : {r.total_time_s:.2f} s"
          + (" (TIMEOUT)" if r.timeout else ""))
    print(f"Completed     : {result.completed}")
    if result.fusion_on and pose_fusion is not None:
        tel = pose_fusion.telemetry
        print(f"Fusion        : ON  imu={tel.imu_samples_seen}  "
              f"vis_ok={tel.vision_fixes_accepted}  "
              f"vis_rej={tel.vision_fixes_rejected}  "
              f"imu_drop={tel.imu_samples_dropped}")
    if args.log_steps:
        print(f"Steps logged  : {len(r.steps)}")

    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
