"""
Mock MAVLink Simulator for AI Grand Prix Development
=====================================================
Lightweight MAVLink-speaking sim that MAVSDK can connect to.
Sends HEARTBEAT, ATTITUDE, ODOMETRY at realistic rates.
Accepts SET_POSITION_TARGET_LOCAL_NED and SET_ATTITUDE_TARGET.
Simulates simple quadrotor dynamics (point mass + drag).

Usage:
    python3.13 mock_sim.py

MAVSDK connects to: udp://:14540
"""

import time
import math
import struct
import socket
import threading
import numpy as np
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2


class MockDrone:
    """Simple point-mass drone with drag model."""

    def __init__(self):
        # State in local NED frame
        self.pos = np.array([0.0, 0.0, -1.0])  # x, y, z (NED, so -1 = 1m up)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.att = np.array([0.0, 0.0, 0.0])   # roll, pitch, yaw (rad)
        self.att_rate = np.array([0.0, 0.0, 0.0])

        # Physics params
        self.mass = 0.5  # kg
        self.drag = 0.3  # drag coefficient
        self.max_thrust = 15.0  # N
        self.max_speed = 10.0   # m/s
        self.max_tilt = math.radians(45)  # max roll/pitch

        # Control targets
        self.pos_target = None   # (x, y, z) or None
        self.vel_target = None   # (vx, vy, vz) or None
        self.att_target = None   # (roll, pitch, yaw, thrust) or None
        self.control_mode = "idle"  # "position", "attitude", "idle"

        # Gate tracking
        self.gates_passed = 0
        self.start_time = time.time()

    def set_position_target(self, x, y, z, vx=0, vy=0, vz=0, type_mask=0):
        """Handle SET_POSITION_TARGET_LOCAL_NED."""
        # type_mask bits: position bits 0-2, velocity bits 3-5
        use_pos = not (type_mask & 0x07 == 0x07)  # position not ignored
        use_vel = not (type_mask & 0x38 == 0x38)  # velocity not ignored

        if use_pos:
            self.pos_target = np.array([x, y, z])
        else:
            self.pos_target = None
        if use_vel:
            self.vel_target = np.array([vx, vy, vz])
        else:
            self.vel_target = None

        # Pick control mode based on what's commanded
        if use_pos:
            self.control_mode = "position"
        elif use_vel:
            self.control_mode = "velocity"

    def set_attitude_target(self, roll, pitch, yaw, thrust):
        """Handle SET_ATTITUDE_TARGET."""
        self.att_target = (roll, pitch, yaw, thrust)
        self.control_mode = "attitude"

    def step(self, dt):
        """Advance physics by dt seconds."""
        if self.control_mode == "velocity" and self.vel_target is not None:
            # Direct velocity tracking — for racing
            vel_cmd = self.vel_target.copy()
            speed = np.linalg.norm(vel_cmd)
            if speed > self.max_speed:
                vel_cmd = vel_cmd / speed * self.max_speed

            # Fast velocity tracking (higher gain than position mode)
            vel_err = vel_cmd - self.vel
            accel = 12.0 * vel_err  # aggressive velocity P gain
            accel -= self.drag * self.vel

            self.vel += accel * dt
            self.pos += self.vel * dt

            # Derive attitude from velocity
            if speed > 0.5:
                self.att[2] = math.atan2(self.vel[1], self.vel[0])
            self.att[0] = np.clip(-self.vel[1] * 0.1, -self.max_tilt, self.max_tilt)
            self.att[1] = np.clip(self.vel[0] * 0.1, -self.max_tilt, self.max_tilt)

        elif self.control_mode == "position" and self.pos_target is not None:
            # Simple PD controller to follow position target
            pos_err = self.pos_target - self.pos
            vel_cmd = 3.0 * pos_err  # P gain
            if self.vel_target is not None:
                vel_cmd += self.vel_target

            # Clamp speed
            speed = np.linalg.norm(vel_cmd)
            if speed > self.max_speed:
                vel_cmd = vel_cmd / speed * self.max_speed

            # Acceleration toward commanded velocity
            vel_err = vel_cmd - self.vel
            accel = 5.0 * vel_err  # velocity P gain

            # Drag
            accel -= self.drag * self.vel

            self.vel += accel * dt
            self.pos += self.vel * dt

            # Derive attitude from velocity direction (simplified)
            if speed > 0.5:
                self.att[2] = math.atan2(self.vel[1], self.vel[0])  # yaw
            self.att[0] = np.clip(-self.vel[1] * 0.1, -self.max_tilt, self.max_tilt)  # roll
            self.att[1] = np.clip(self.vel[0] * 0.1, -self.max_tilt, self.max_tilt)   # pitch

        elif self.control_mode == "attitude" and self.att_target is not None:
            roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = self.att_target

            # Attitude tracking
            self.att[0] += 5.0 * (roll_cmd - self.att[0]) * dt
            self.att[1] += 5.0 * (pitch_cmd - self.att[1]) * dt
            self.att[2] += 2.0 * (yaw_cmd - self.att[2]) * dt

            # Thrust -> acceleration in body frame, then to NED
            thrust_force = thrust_cmd * self.max_thrust
            # Simplified: thrust along body z, projected to NED
            ax = thrust_force * math.sin(self.att[1]) / self.mass
            ay = -thrust_force * math.sin(self.att[0]) / self.mass
            az = -thrust_force * math.cos(self.att[0]) * math.cos(self.att[1]) / self.mass + 9.81

            accel = np.array([ax, ay, az]) - self.drag * self.vel
            self.vel += accel * dt
            self.pos += self.vel * dt

        else:
            # Idle: hover with slow drift down
            hover_accel = np.array([0, 0, 0]) - self.drag * self.vel
            self.vel += hover_accel * dt
            self.pos += self.vel * dt

        # Ground clamp (NED: z=0 is ground, negative is up)
        if self.pos[2] > 0:
            self.pos[2] = 0
            self.vel[2] = 0


class MockSimulator:
    """MAVLink simulator that MAVSDK can connect to."""

    def __init__(self, port=14540):
        self.port = port
        self.drone = MockDrone()
        self.running = False
        self.boot_time_ms = int(time.time() * 1000)

        self.__init_params()

        # MAVLink connection — single udpout socket for both directions.
        # Sends telemetry TO MAVSDK on port 14540.
        # MAVSDK sees the source port and sends commands back to it.
        # pymavlink udpout sockets can also recv on their ephemeral port.
        self.mav = mavutil.mavlink_connection(
            f'udpout:127.0.0.1:{port}',
            source_system=1,
            source_component=1,
            dialect='common'
        )

        # Physics rate
        self.physics_hz = 120
        self.telemetry_hz = 50
        self.heartbeat_hz = 2

    def get_time_boot_ms(self):
        return int(time.time() * 1000) - self.boot_time_ms

    def send_heartbeat(self):
        """Send HEARTBEAT message."""
        self.mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_PX4,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,  # custom mode
            mavutil.mavlink.MAV_STATE_ACTIVE
        )

    def send_attitude(self):
        """Send ATTITUDE message."""
        self.mav.mav.attitude_send(
            self.get_time_boot_ms(),
            self.drone.att[0],  # roll
            self.drone.att[1],  # pitch
            self.drone.att[2],  # yaw
            self.drone.att_rate[0],  # rollspeed
            self.drone.att_rate[1],  # pitchspeed
            self.drone.att_rate[2],  # yawspeed
        )

    def send_highres_imu(self):
        """Send HIGHRES_IMU message."""
        self.mav.mav.highres_imu_send(
            int(time.time() * 1e6),  # time_usec
            self.drone.vel[0] * 0.1 + np.random.normal(0, 0.01),  # xacc
            self.drone.vel[1] * 0.1 + np.random.normal(0, 0.01),  # yacc
            -9.81 + np.random.normal(0, 0.01),  # zacc
            self.drone.att_rate[0] + np.random.normal(0, 0.001),  # xgyro
            self.drone.att_rate[1] + np.random.normal(0, 0.001),  # ygyro
            self.drone.att_rate[2] + np.random.normal(0, 0.001),  # zgyro
            0, 0, 0,  # xmag, ymag, zmag
            0,  # abs_pressure
            0,  # diff_pressure
            0,  # pressure_alt
            0,  # temperature
            0xFFFF,  # fields_updated (all)
            0,  # id
        )

    def send_local_position_ned(self):
        """Send LOCAL_POSITION_NED — position + velocity in local frame.
        This is what MAVSDK's position_velocity_ned() reads."""
        self.mav.mav.local_position_ned_send(
            self.get_time_boot_ms(),
            self.drone.pos[0],  # x (north)
            self.drone.pos[1],  # y (east)
            self.drone.pos[2],  # z (down)
            self.drone.vel[0],  # vx
            self.drone.vel[1],  # vy
            self.drone.vel[2],  # vz
        )

    def __init_params(self):
        """Initialize fake PX4 parameters that MAVSDK expects."""
        self.params = {
            'MIS_TAKEOFF_ALT': 2.0,
            'MPC_TKO_SPEED': 1.5,
            'COM_DISARM_LAND': 2.0,
            'NAV_DLL_ACT': 0.0,
            'NAV_RCL_ACT': 0.0,
            'COM_RC_IN_MODE': 1.0,
            'COM_FLTMODE1': 0.0,
        }

    def process_commands(self):
        """Check for incoming commands from MAVSDK."""
        msg = self.mav.recv_match(blocking=False)
        if msg is None:
            return

        msg_type = msg.get_type()

        if msg_type == 'SET_POSITION_TARGET_LOCAL_NED':
            self.drone.set_position_target(
                msg.x, msg.y, msg.z,
                msg.vx, msg.vy, msg.vz,
                msg.type_mask
            )

        elif msg_type == 'SET_ATTITUDE_TARGET':
            # Extract quaternion to euler (simplified)
            q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
            roll = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
            pitch = math.asin(max(-1, min(1, 2*(q[0]*q[2] - q[3]*q[1]))))
            yaw = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
            self.drone.set_attitude_target(roll, pitch, yaw, msg.thrust)

        elif msg_type == 'HEARTBEAT':
            pass  # Client heartbeat

        elif msg_type == 'PARAM_REQUEST_READ':
            # MAVSDK reads params — respond with stored value
            param_id = msg.param_id.rstrip('\x00')
            value = self.params.get(param_id, 0.0)
            self.mav.mav.param_value_send(
                param_id.encode('utf-8').ljust(16, b'\x00'),
                value,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                len(self.params),
                list(self.params.keys()).index(param_id) if param_id in self.params else 0
            )

        elif msg_type == 'PARAM_SET':
            # MAVSDK sets params — store and echo back
            param_id = msg.param_id.rstrip('\x00')
            self.params[param_id] = msg.param_value
            self.mav.mav.param_value_send(
                param_id.encode('utf-8').ljust(16, b'\x00'),
                msg.param_value,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                len(self.params),
                list(self.params.keys()).index(param_id) if param_id in self.params else 0
            )

        elif msg_type == 'PARAM_REQUEST_LIST':
            # Send all params
            for i, (pid, val) in enumerate(self.params.items()):
                self.mav.mav.param_value_send(
                    pid.encode('utf-8').ljust(16, b'\x00'),
                    val,
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                    len(self.params),
                    i
                )

        elif msg_type == 'COMMAND_LONG':
            # Handle arm/disarm, mode changes etc
            if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                print(f"  [SIM] Arm command: {'ARM' if msg.param1 == 1 else 'DISARM'}")
            elif msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                alt = msg.param7 if not math.isnan(msg.param7) else 2.0
                print(f"  [SIM] Takeoff to {alt}m")
                self.drone.set_position_target(
                    self.drone.pos[0], self.drone.pos[1], -abs(alt)
                )
            else:
                pass  # Log nothing for routine commands
            # ACK all COMMAND_LONGs
            self.mav.mav.command_ack_send(
                msg.command,
                mavutil.mavlink.MAV_RESULT_ACCEPTED
            )

        elif msg_type == 'COMMAND_INT':
            # ACK command_int too
            self.mav.mav.command_ack_send(
                msg.command,
                mavutil.mavlink.MAV_RESULT_ACCEPTED
            )

    def run(self):
        """Main simulation loop."""
        self.running = True
        print(f"[SIM] Mock simulator running")
        print(f"[SIM] MAVSDK connect to: udp://:14540")
        print(f"[SIM] Sending telemetry: HEARTBEAT, ATTITUDE, HIGHRES_IMU, LOCAL_POSITION_NED")
        print(f"[SIM] Physics: {self.physics_hz} Hz | Telemetry: {self.telemetry_hz} Hz")
        print(f"[SIM] Drone at position: {self.drone.pos}")
        print()

        physics_dt = 1.0 / self.physics_hz
        telem_interval = 1.0 / self.telemetry_hz
        hb_interval = 1.0 / self.heartbeat_hz

        last_telem = 0
        last_hb = 0
        step_count = 0

        try:
            while self.running:
                now = time.time()

                # Physics step
                self.drone.step(physics_dt)

                # Process incoming commands
                self.process_commands()

                # Send heartbeat
                if now - last_hb >= hb_interval:
                    self.send_heartbeat()
                    last_hb = now

                # Send telemetry
                if now - last_telem >= telem_interval:
                    self.send_attitude()
                    self.send_highres_imu()
                    self.send_local_position_ned()
                    last_telem = now

                    # Print status every 2 seconds
                    step_count += 1
                    if step_count % (self.telemetry_hz * 2) == 0:
                        d = self.drone
                        print(f"  pos=({d.pos[0]:6.2f}, {d.pos[1]:6.2f}, {d.pos[2]:6.2f}) "
                              f"vel=({d.vel[0]:5.2f}, {d.vel[1]:5.2f}, {d.vel[2]:5.2f}) "
                              f"mode={d.control_mode}")

                # Sleep to maintain physics rate
                elapsed = time.time() - now
                sleep_time = physics_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[SIM] Shutting down")
            self.running = False


if __name__ == '__main__':
    sim = MockSimulator(port=14540)
    sim.run()
