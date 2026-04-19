"""
AI Grand Prix — Control Skeleton
==================================
MAVSDK-based autonomy loop. Connects to any MAVLink sim (mock or real).
Receives telemetry, plans, sends commands.

Usage:
    # Terminal 1: run the mock sim
    python3.13 mock_sim.py

    # Terminal 2: run this controller
    python3.13 control_skeleton.py

Architecture:
    Telemetry (ODOMETRY, ATTITUDE, IMU) → State → Planner → Commander → MAVLink
"""

import asyncio
import time
import math
import json
import os
from datetime import datetime
from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    PositionNedYaw,
    VelocityNedYaw,
    Attitude,
)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class DroneState:
    """Current estimated state of the drone."""

    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]       # NED meters
        self.vel = [0.0, 0.0, 0.0]       # NED m/s
        self.att = [0.0, 0.0, 0.0]       # roll, pitch, yaw (rad)
        self.att_rate = [0.0, 0.0, 0.0]  # rad/s
        self.timestamp = 0.0
        self.connected = False
        self.armed = False
        self.in_air = False


# ─────────────────────────────────────────────
# Telemetry Logger
# ─────────────────────────────────────────────

class TelemetryLogger:
    """Logs all state + commands for replay analysis."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"flight_{timestamp}.jsonl")
        self.file = open(self.log_path, 'w')
        self.count = 0
        print(f"  [LOG] Logging to {self.log_path}")

    def log(self, event_type, data):
        """Write a log entry."""
        entry = {
            "t": time.time(),
            "type": event_type,
            **data
        }
        self.file.write(json.dumps(entry) + '\n')
        self.count += 1
        if self.count % 500 == 0:
            self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()
        print(f"  [LOG] Wrote {self.count} entries to {self.log_path}")


# ─────────────────────────────────────────────
# Gate Sequencer (stub)
# ─────────────────────────────────────────────

class GateSequencer:
    """Tracks gate order and current target.
    For now: hard-coded waypoint list as gate proxies.
    """

    def __init__(self, gates=None):
        # Default: a square pattern at 2m altitude (NED: z=-2)
        if gates is None:
            gates = [
                (5.0, 0.0, -2.0),    # gate 1: 5m forward
                (5.0, 5.0, -2.0),    # gate 2: right turn
                (0.0, 5.0, -2.0),    # gate 3: back
                (0.0, 0.0, -2.0),    # gate 4: home
            ]
        self.gates = gates
        self.current_idx = 0
        self.completed = False

    @property
    def current_gate(self):
        if self.current_idx < len(self.gates):
            return self.gates[self.current_idx]
        return None

    @property
    def gate_count(self):
        return len(self.gates)

    def check_gate_pass(self, pos, threshold=2.5):
        """Check if we've passed through the current gate."""
        gate = self.current_gate
        if gate is None:
            return False

        dx = pos[0] - gate[0]
        dy = pos[1] - gate[1]
        dz = pos[2] - gate[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < threshold:
            self.current_idx += 1
            if self.current_idx >= len(self.gates):
                self.completed = True
            return True
        return False

    def next_gate_after(self):
        """Peek at the gate after the current one (for look-ahead)."""
        idx = self.current_idx + 1
        if idx < len(self.gates):
            return self.gates[idx]
        return None


# ─────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────

class Planner:
    """V5.1: Phase-aware + V4-aggressive hybrid.
    
    Keeps V5 phase detection (LAUNCH/BUILD/SUSTAIN/PRE_TURN/TURN/SHORT).
    Restores V4 aggression: higher speed targets, later braking, punchier launch.
    
    Changes from V5:
    - BUILD/SUSTAIN: full cruise speed, no caps (V4 level)
    - PRE_TURN: later onset (decel_distance * 0.8 safety margin, down from 1.3)
    - SHORT: brief aggressive build allowed, not mini safe mode
    - LAUNCH: faster ramp (2.0s, punchier after initial stabilization)
    - Smoothing: BUILD/SUSTAIN at 12 m/s/s (up from 10/8), TURN kept at 6
    - Turn entry speed: 0.4 + 0.6*cos (V4 formula, not V5's 0.35 + 0.65)
    - PX4 model: speed_ceiling 8.5, util_straight 0.78 (V5 values, validated)
    """
    
    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5):
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.base_blend = base_blend
        self.blend_radius = base_blend
        self.mode = "velocity"
        
        # PX4 response model
        self.px4_max_accel = 6.0
        self.px4_max_decel = 4.0
        self.px4_speed_ceiling = 8.5
        self.px4_util_straight = 0.78
        self.px4_util_turn = 0.55
        self.px4_hover_ramp_time = 2.0  # punchier: 2.0s (was 2.5 in V5, 3.0 in V4)
        
        # State
        self.last_cmd_speed = 0.0
        self.mission_start_time = None
        self.gates_passed = 0
        self.is_first_leg = True
        self.prev_gate_speed = 0.0
    
    def _turn_entry_speed(self, ta):
        """V4 formula restored: 0.4 + 0.6*cos(ta/2)"""
        return min(self.cruise_speed * (0.4 + 0.6 * math.cos(ta / 2.0)), self.px4_speed_ceiling)
    
    def _decel_distance(self, cs, ta):
        """Shorter safety margin than V5: 0.8x (was 1.3x)"""
        if ta < 0.3: return 0.0  # raised threshold: only pre-decel for real turns
        ts = self._turn_entry_speed(ta)
        sd = max(0, cs - ts)
        if sd <= 0: return 0.0
        t = sd / self.px4_max_decel
        d = cs * t - 0.5 * self.px4_max_decel * t * t
        return max(d * 0.8, self.base_blend)  # tighter: 0.8x margin, min = blend only
    
    def _phase(self, dxy, cs, ta, db):
        """Phase detection with later PRE_TURN onset."""
        if self.is_first_leg and self.mission_start_time:
            if time.time() - self.mission_start_time < self.px4_hover_ramp_time:
                return 'LAUNCH'
        if dxy <= db:
            return 'TURN'
        dd = self._decel_distance(cs, ta)
        if dxy <= dd and ta > 0.3:
            return 'PRE_TURN'
        # Short leg: only if VERY short. Raised from 10m to 6m.
        if dxy < 6.0 and ta > 0.8:
            return 'SHORT'
        return 'SUSTAIN'  # merged BUILD into SUSTAIN - always push hard
    
    def _px4_cmd(self, desired, ta_for_util):
        """Utilization inversion."""
        tr = ta_for_util / math.pi
        util = self.px4_util_straight * (1 - tr) + self.px4_util_turn * tr
        return min(desired / max(util, 0.3), self.max_speed)
    
    def _smooth(self, target, phase, dt=0.02):
        """Phase-aware smoothing - V5.1: faster BUILD/SUSTAIN rates."""
        rates = {
            'LAUNCH': 6.0,    # up from 4.0 - punchier launch
            'SUSTAIN': 12.0,  # up from 8.0 - V4+ aggression
            'PRE_TURN': 10.0, # kept moderate for controlled decel
            'TURN': 6.0,      # kept gentle through turns
            'SHORT': 10.0,    # up from 8.0
        }
        mr = rates.get(phase, 10.0)
        md = mr * dt
        if target > self.last_cmd_speed:
            s = min(target, self.last_cmd_speed + md)
        else:
            s = max(target, self.last_cmd_speed - md * 1.5)
        self.last_cmd_speed = s
        return s
    
    def on_gate_passed(self, speed):
        self.gates_passed += 1
        self.is_first_leg = False
        self.prev_gate_speed = speed
    
    def plan_velocity(self, state, target_gate, next_gate=None):
        """V5.1 velocity planning."""
        if target_gate is None:
            return VelocityNedYaw(0, 0, 0, 0)
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
        
        tx, ty, tz = target_gate
        dx, dy, dz = tx - state.pos[0], ty - state.pos[1], tz - state.pos[2]
        dist_xy = math.sqrt(dx*dx + dy*dy)
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
        yaw_deg = math.degrees(math.atan2(dy, dx))
        
        if dist_3d < 0.1:
            return VelocityNedYaw(0, 0, 0, yaw_deg)
        
        ux, uy, uz = dx/dist_3d, dy/dist_3d, dz/dist_3d
        cs = math.sqrt(state.vel[0]**2 + state.vel[1]**2)
        
        # Turn angle
        ta = 0.0
        if next_gate is not None:
            nx, ny = next_gate[0] - tx, next_gate[1] - ty
            nd = math.sqrt(nx*nx + ny*ny)
            if nd > 0.1 and dist_xy > 0.1:
                ax, ay = dx / dist_xy, dy / dist_xy
                bx, by = nx / nd, ny / nd
                dot = max(-1.0, min(1.0, ax*bx + ay*by))
                ta = math.acos(dot)
        
        # Dynamic blend
        db = self.base_blend + 0.25 * cs + (ta / math.pi) * 2.0
        self.blend_radius = db
        
        # Phase
        phase = self._phase(dist_xy, cs, ta, db)
        
        # Speed target per phase
        if phase == 'LAUNCH':
            elapsed = time.time() - self.mission_start_time
            ramp = min(1.0, elapsed / self.px4_hover_ramp_time)
            ramp = ramp * ramp  # quadratic but faster (2.0s window)
            desired = max(self.cruise_speed * ramp, 2.5)
        
        elif phase == 'SHORT':
            # Brief aggressive build, but cap at turn-entry speed
            te = self._turn_entry_speed(ta)
            # Allow accelerating up to turn entry speed, not just limping
            desired = min(max(cs + 2.0, te * 0.8), te)
        
        elif phase == 'SUSTAIN':
            # Full cruise - V4 level aggression
            desired = self.cruise_speed
        
        elif phase == 'PRE_TURN':
            te = self._turn_entry_speed(ta)
            dd = self._decel_distance(cs, ta)
            if dd > 0.1 and dd > db:
                progress = 1.0 - (dist_xy - db) / (dd - db)
                progress = max(0.0, min(1.0, progress))
                desired = cs + (te - cs) * progress
            else:
                desired = te
        
        elif phase == 'TURN':
            te = self._turn_entry_speed(ta)
            blend = max(0.0, min(1.0, 1.0 - (dist_xy / db)))
            # V4 apex formula: 0.2 + 0.2*(ta/pi)
            ad = 0.2 + 0.2 * (ta / math.pi)
            af = 1.0 - ad * math.sin(blend * math.pi)
            desired = te * af
        
        else:
            desired = self.cruise_speed
        
        desired = min(desired, self.px4_speed_ceiling)
        
        # PX4 command inversion - straight-line util for non-turn phases
        cmd_spd = self._px4_cmd(desired, ta if phase in ('PRE_TURN', 'TURN', 'SHORT') else 0.0)
        
        # Direction
        if phase == 'TURN' and next_gate is not None and dist_xy < db:
            blend = max(0.0, min(1.0, 1.0 - (dist_xy / db)))
            nx2, ny2, nz2 = next_gate[0]-tx, next_gate[1]-ty, next_gate[2]-tz
            nd2 = math.sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2)
            if nd2 > 0.1:
                nux, nuy, nuz = nx2/nd2, ny2/nd2, nz2/nd2
            else:
                nux, nuy, nuz = ux, uy, uz
            bxd = ux*(1-blend) + nux*blend
            byd = uy*(1-blend) + nuy*blend
            bzd = uz*(1-blend) + nuz*blend
            bm = math.sqrt(bxd*bxd + byd*byd + bzd*bzd)
            if bm > 0.01:
                bxd, byd, bzd = bxd/bm, byd/bm, bzd/bm
            sp = self._smooth(cmd_spd, phase)
            vx, vy, vz = bxd*sp, byd*sp, bzd*sp
            yaw_deg = math.degrees(math.atan2(byd, bxd))
        else:
            sp = self._smooth(cmd_spd, phase)
            vx, vy, vz = ux*sp, uy*sp, uz*sp
        
        # Altitude
        vz = (tz - state.pos[2]) * 3.0
        
        # Cap
        st = math.sqrt(vx*vx + vy*vy + vz*vz)
        if st > self.max_speed:
            s = self.max_speed / st
            vx *= s; vy *= s; vz *= s
        
        return VelocityNedYaw(vx, vy, vz, yaw_deg)
    
    def plan_position(self, state, target_gate, next_gate=None):
        if target_gate is None:
            return PositionNedYaw(state.pos[0], state.pos[1], state.pos[2], 0.0)
        tx, ty, tz = target_gate
        dx, dy = tx - state.pos[0], ty - state.pos[1]
        yaw_deg = math.degrees(math.atan2(dy, dx))
        if next_gate is not None:
            tx += (next_gate[0] - tx) * 0.3
            ty += (next_gate[1] - ty) * 0.3
        else:
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.1:
                tx += (dx / dist) * 0.5; ty += (dy / dist) * 0.5
        return PositionNedYaw(tx, ty, tz, yaw_deg)
    
    def plan(self, state, target_gate, next_gate=None):
        if self.mode == "velocity":
            return self.plan_velocity(state, target_gate, next_gate)
        else:
            return self.plan_position(state, target_gate, next_gate)


# ─────────────────────────────────────────────
# Main Controller
# ─────────────────────────────────────────────

class Controller:
    """Main autonomy loop."""

    def __init__(self, connection_string="udpin://0.0.0.0:14540"):
        self.conn_str = connection_string
        self.drone = System()
        self.state = DroneState()
        self.logger = TelemetryLogger()
        self.sequencer = GateSequencer()
        self.planner = Planner(max_speed=12.0, cruise_speed=10.0, base_blend=2.5)
        self.command_hz = 50
        self.start_time = None

    async def connect(self):
        """Connect to the MAVLink sim."""
        print(f"[CTRL] Connecting to {self.conn_str}...")
        await self.drone.connect(system_address=self.conn_str)

        print("[CTRL] Waiting for connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[CTRL] Connected!")
                self.state.connected = True
                break

    async def start_telemetry(self):
        """Subscribe to telemetry streams."""

        # Position (from ODOMETRY)
        async def position_loop():
            async for pos in self.drone.telemetry.position_velocity_ned():
                self.state.pos = [
                    pos.position.north_m,
                    pos.position.east_m,
                    pos.position.down_m
                ]
                self.state.vel = [
                    pos.velocity.north_m_s,
                    pos.velocity.east_m_s,
                    pos.velocity.down_m_s
                ]
                self.state.timestamp = time.time()
                self.logger.log("state", {
                    "pos": self.state.pos,
                    "vel": self.state.vel,
                    "att": self.state.att,
                })

        # Attitude
        async def attitude_loop():
            async for att in self.drone.telemetry.attitude_euler():
                self.state.att = [
                    math.radians(att.roll_deg),
                    math.radians(att.pitch_deg),
                    math.radians(att.yaw_deg)
                ]

        # Flight mode / armed status
        async def status_loop():
            async for armed in self.drone.telemetry.armed():
                self.state.armed = armed

        asyncio.ensure_future(position_loop())
        asyncio.ensure_future(attitude_loop())
        asyncio.ensure_future(status_loop())
        print("[CTRL] Telemetry streams active")

    async def arm_and_takeoff(self, altitude=2.0):
        """Arm the drone and take off to given altitude (meters)."""
        print(f"[CTRL] Arming...")
        await self.drone.action.arm()

        print(f"[CTRL] Taking off to {altitude}m...")
        try:
            await self.drone.action.set_takeoff_altitude(altitude)
        except Exception as e:
            print(f"[CTRL] set_takeoff_altitude skipped ({e})")
        await self.drone.action.takeoff()

        # Wait to reach altitude
        await asyncio.sleep(3)
        print(f"[CTRL] Airborne at ~{altitude}m")
        self.state.in_air = True

    async def start_offboard(self):
        """Switch to offboard mode for direct position/attitude control."""
        print(f"[CTRL] Starting offboard mode ({self.planner.mode})...")

        # Must send a setpoint before starting offboard
        if self.planner.mode == "velocity":
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        else:
            initial = PositionNedYaw(
                self.state.pos[0],
                self.state.pos[1],
                self.state.pos[2],
                0.0
            )
            await self.drone.offboard.set_position_ned(initial)

        try:
            await self.drone.offboard.start()
            print("[CTRL] Offboard mode active")
        except OffboardError as e:
            print(f"[CTRL] Offboard start failed: {e}")
            raise

    async def run_mission(self):
        """Main control loop — fly through all gates."""
        self.start_time = time.time()
        print(f"\n[CTRL] === MISSION START ===")
        print(f"[CTRL] Mode: {self.planner.mode} | Speed: {self.planner.cruise_speed} m/s")
        print(f"[CTRL] Gates: {self.sequencer.gate_count}")
        print(f"[CTRL] Command rate: {self.command_hz} Hz")
        print()

        dt = 1.0 / self.command_hz
        loop_count = 0

        while not self.sequencer.completed:
            loop_start = time.time()

            # Check for gate pass
            if self.sequencer.check_gate_pass(self.state.pos, threshold=2.5):
                gate_num = self.sequencer.current_idx  # already advanced
                elapsed = time.time() - self.start_time
                print(f"  [GATE] Passed gate {gate_num}/{self.sequencer.gate_count} "
                      f"at t={elapsed:.1f}s")
                self.logger.log("gate_pass", {
                    "gate": gate_num,
                    "elapsed": elapsed,
                    "pos": self.state.pos
                })
                # Notify planner of gate pass for phase tracking
                gate_speed = math.sqrt(sum(v**2 for v in self.state.vel[:2]))
                self.planner.on_gate_passed(gate_speed)

            # Plan next target
            target = self.planner.plan(
                self.state,
                self.sequencer.current_gate,
                self.sequencer.next_gate_after()
            )

            # Send command — velocity or position based on planner mode
            if isinstance(target, VelocityNedYaw):
                await self.drone.offboard.set_velocity_ned(target)
                self.logger.log("cmd", {
                    "mode": "velocity",
                    "vn": target.north_m_s,
                    "ve": target.east_m_s,
                    "vd": target.down_m_s,
                    "yaw": target.yaw_deg,
                    "gate_idx": self.sequencer.current_idx,
                })
            else:
                await self.drone.offboard.set_position_ned(target)
                self.logger.log("cmd", {
                    "mode": "position",
                    "target_n": target.north_m,
                    "target_e": target.east_m,
                    "target_d": target.down_m,
                    "target_yaw": target.yaw_deg,
                    "gate_idx": self.sequencer.current_idx,
                })

            # Status print every second
            loop_count += 1
            if loop_count % self.command_hz == 0:
                gate = self.sequencer.current_gate
                if gate:
                    dx = gate[0] - self.state.pos[0]
                    dy = gate[1] - self.state.pos[1]
                    dz = gate[2] - self.state.pos[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    speed = math.sqrt(sum(v**2 for v in self.state.vel))
                    print(f"  pos=({self.state.pos[0]:5.1f}, {self.state.pos[1]:5.1f}, {self.state.pos[2]:5.1f}) "
                          f"→ gate {self.sequencer.current_idx + 1} "
                          f"dist={dist:.1f}m speed={speed:.1f}m/s")

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Mission complete
        total_time = time.time() - self.start_time
        print(f"\n[CTRL] === MISSION COMPLETE ===")
        print(f"[CTRL] All {self.sequencer.gate_count} gates passed!")
        print(f"[CTRL] Total time: {total_time:.2f}s")

        self.logger.log("mission_complete", {
            "total_time": total_time,
            "gates": self.sequencer.gate_count,
        })

    async def land(self):
        """Land the drone."""
        print("[CTRL] Landing...")
        await self.drone.action.land()
        await asyncio.sleep(3)
        print("[CTRL] Landed")

    async def run(self):
        """Full flight sequence."""
        try:
            await self.connect()
            await self.start_telemetry()
            await asyncio.sleep(1)  # let telemetry settle

            await self.arm_and_takeoff(altitude=2.0)
            await self.start_offboard()
            await self.run_mission()
            await self.land()

        except Exception as e:
            print(f"[CTRL] ERROR: {e}")
            raise
        finally:
            self.logger.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  AI Grand Prix — Control Skeleton")
    print("  V5: Phase-Aware PX4 Planner")
    print("=" * 60)
    print()

    controller = Controller(connection_string="udpin://0.0.0.0:14540")
    await controller.run()


if __name__ == '__main__':
    asyncio.run(main())
