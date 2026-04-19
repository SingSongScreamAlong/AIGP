"""
PX4 SITL Benchmark V2: Competition-scale courses + response logging
===================================================================
3 courses:
  Sprint   - long straights, gentle turns (tests top speed)
  Technical - tight corners, short spacing (tests cornering)
  Mixed    - realistic race layout (tests everything)

Logs commanded vs achieved velocity, tracking lag, peak accel.
"""
import asyncio
import time
import math
import json
import subprocess
import os
from collections import defaultdict

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

# ─────────────────────────────────────────────
# Course Definitions (NED: z negative = up)
# ─────────────────────────────────────────────
COURSES = {
    'sprint': {
        'description': 'Long straights, gentle turns - tests sustained cruise',
        'altitude': -3.0,
        'gates': [
            (30.0,   0.0, -3.0),   # 30m straight
            (50.0,  15.0, -3.0),   # angled right (~27deg)
            (80.0,  15.0, -3.0),   # 30m straight
            (100.0,  0.0, -3.0),   # angled left back
            (130.0,  0.0, -3.0),   # 30m straight
            (130.0, 20.0, -3.0),   # 90deg right turn
            (100.0, 20.0, -3.0),   # 30m straight back
            (70.0,  30.0, -3.0),   # angled
            (30.0,  30.0, -3.0),   # 40m straight
            (0.0,   15.0, -3.0),   # angled back to start area
        ],
    },
    'technical': {
        'description': 'Tight corners, short spacing - tests cornering',
        'altitude': -2.5,
        'gates': [
            (8.0,   0.0, -2.5),   # short straight
            (12.0,   6.0, -2.5),   # 45deg right
            (8.0,  12.0, -2.5),   # sharp left (135deg turn)
            (0.0,  12.0, -2.5),   # short straight
            (-4.0,   6.0, -2.5),  # sharp right
            (0.0,   0.0, -2.5),   # back toward start
            (6.0,  -4.0, -2.5),   # new direction
            (14.0,  -4.0, -2.5),  # straight
            (18.0,   0.0, -2.5),  # turn
            (14.0,   4.0, -2.5),  # tight S
            (8.0,   4.0, -2.5),   # straight
            (4.0,   0.0, -3.0),   # descending finish
        ],
    },
    'mixed': {
        'description': 'Realistic race layout - mixed straights and corners',
        'altitude': -3.0,
        'gates': [
            (20.0,   0.0, -3.0),  # 20m opening straight
            (35.0,  10.0, -3.0),  # sweeping right
            (50.0,  10.0, -3.0),  # 15m straight
            (55.0,  20.0, -3.5),  # climbing right turn
            (40.0,  25.0, -3.5),  # short connecting
            (25.0,  30.0, -3.0),  # descending left
            (10.0,  30.0, -3.0),  # 15m straight
            (0.0,   20.0, -2.5),  # tight hairpin entry
            (-5.0,  10.0, -2.5),  # hairpin apex
            (5.0,    5.0, -3.0),  # hairpin exit + accel
            (15.0,  -5.0, -3.0),  # cross-track
            (25.0,   0.0, -3.0),  # back to start zone
        ],
    },
}


# ─────────────────────────────────────────────
# V3 Planner (identical to controller)
# ─────────────────────────────────────────────
class V3Planner:
    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5):
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.base_blend = base_blend

    def plan_velocity(self, pos, vel, target, next_gate=None):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]
        dist_xy = math.sqrt(dx*dx + dy*dy)
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist_3d < 0.1:
            return VelocityNedYaw(0, 0, 0, 0)
        ux = dx / dist_xy if dist_xy > 0.01 else 0
        uy = dy / dist_xy if dist_xy > 0.01 else 0

        turn_angle = 0.0
        if next_gate is not None:
            app_x, app_y = target[0]-pos[0], target[1]-pos[1]
            dep_x, dep_y = next_gate[0]-target[0], next_gate[1]-target[1]
            app_mag = math.sqrt(app_x**2 + app_y**2)
            dep_mag = math.sqrt(dep_x**2 + dep_y**2)
            if app_mag > 0.01 and dep_mag > 0.01:
                dot = (app_x*dep_x + app_y*dep_y) / (app_mag * dep_mag)
                dot = max(-1.0, min(1.0, dot))
                turn_angle = math.acos(dot)

        current_speed = math.sqrt(vel[0]**2 + vel[1]**2)
        dynamic_blend = self.base_blend + 0.2*current_speed + (turn_angle/math.pi)*1.5
        adjusted_cruise = self.cruise_speed * (0.4 + 0.6*math.cos(turn_angle/2.0))

        if dist_xy < dynamic_blend:
            blend = 1.0 - (dist_xy / dynamic_blend)
            apex_factor = 1.0 - 0.3*math.sin(blend*math.pi)
            speed = adjusted_cruise * apex_factor
        elif dist_xy < dynamic_blend * 1.5:
            ratio = (dist_xy - dynamic_blend) / (dynamic_blend * 0.5)
            speed = adjusted_cruise + (self.cruise_speed - adjusted_cruise) * ratio
        else:
            speed = self.cruise_speed

        speed = min(speed, self.max_speed)
        vx, vy = ux*speed, uy*speed
        vz = (target[2] - pos[2]) * 3.0
        yaw = math.degrees(math.atan2(dy, dx))
        return VelocityNedYaw(vx, vy, vz, yaw)


# ─────────────────────────────────────────────
# Response Logger
# ─────────────────────────────────────────────
class ResponseLogger:
    """Logs commanded vs achieved for PX4 tracking analysis."""
    def __init__(self):
        self.entries = []
        self.gate_events = []

    def log(self, t, cmd_vn, cmd_ve, cmd_vd, cmd_yaw,
            ach_vn, ach_ve, ach_vd, ach_yaw,
            pos_n, pos_e, pos_d, gate_idx):
        cmd_spd = math.sqrt(cmd_vn**2 + cmd_ve**2)
        ach_spd = math.sqrt(ach_vn**2 + ach_ve**2)
        vel_err = math.sqrt((cmd_vn-ach_vn)**2 + (cmd_ve-ach_ve)**2)

        # Compute achieved acceleration from last entry
        accel = 0.0
        if len(self.entries) > 0:
            prev = self.entries[-1]
            dt = t - prev['t']
            if dt > 0.001:
                dvn = ach_vn - prev['ach_vn']
                dve = ach_ve - prev['ach_ve']
                accel = math.sqrt(dvn**2 + dve**2) / dt

        self.entries.append({
            't': t,
            'cmd_vn': cmd_vn, 'cmd_ve': cmd_ve, 'cmd_vd': cmd_vd,
            'cmd_spd': cmd_spd, 'cmd_yaw': cmd_yaw,
            'ach_vn': ach_vn, 'ach_ve': ach_ve, 'ach_vd': ach_vd,
            'ach_spd': ach_spd, 'ach_yaw': ach_yaw,
            'vel_err': vel_err, 'accel': accel,
            'pos_n': pos_n, 'pos_e': pos_e, 'pos_d': pos_d,
            'gate_idx': gate_idx,
        })

    def log_gate(self, gate_num, t, pos, vel):
        self.gate_events.append({
            'gate': gate_num,
            't': t,
            'pos': pos[:],
            'vel': vel[:],
            'speed': math.sqrt(vel[0]**2 + vel[1]**2),
        })

    def analyze(self):
        """Compute tracking statistics."""
        if not self.entries:
            return {}

        vel_errors = [e['vel_err'] for e in self.entries]
        cmd_speeds = [e['cmd_spd'] for e in self.entries]
        ach_speeds = [e['ach_spd'] for e in self.entries]
        accels = [e['accel'] for e in self.entries[1:]]  # skip first

        # Per-gate-leg analysis
        leg_stats = []
        for i in range(len(self.gate_events)):
            ge = self.gate_events[i]
            t_start = self.gate_events[i-1]['t'] if i > 0 else self.entries[0]['t']
            t_end = ge['t']
            leg_entries = [e for e in self.entries if t_start <= e['t'] <= t_end]
            if leg_entries:
                leg_cmd = [e['cmd_spd'] for e in leg_entries]
                leg_ach = [e['ach_spd'] for e in leg_entries]
                leg_err = [e['vel_err'] for e in leg_entries]
                leg_acc = [e['accel'] for e in leg_entries[1:]] if len(leg_entries) > 1 else [0]
                leg_stats.append({
                    'gate': ge['gate'],
                    'duration': round(t_end - t_start, 3),
                    'entry_speed': round(ge['speed'], 2),
                    'avg_cmd_speed': round(sum(leg_cmd)/len(leg_cmd), 2),
                    'avg_ach_speed': round(sum(leg_ach)/len(leg_ach), 2),
                    'max_ach_speed': round(max(leg_ach), 2),
                    'avg_vel_error': round(sum(leg_err)/len(leg_err), 3),
                    'max_vel_error': round(max(leg_err), 3),
                    'peak_accel': round(max(leg_acc) if leg_acc else 0, 2),
                    'speed_utilization': round(sum(leg_ach)/max(sum(leg_cmd),0.01), 3),
                })

        return {
            'overall': {
                'avg_vel_error': round(sum(vel_errors)/len(vel_errors), 3),
                'max_vel_error': round(max(vel_errors), 3),
                'p95_vel_error': round(sorted(vel_errors)[int(0.95*len(vel_errors))], 3),
                'avg_cmd_speed': round(sum(cmd_speeds)/len(cmd_speeds), 2),
                'avg_ach_speed': round(sum(ach_speeds)/len(ach_speeds), 2),
                'max_ach_speed': round(max(ach_speeds), 2),
                'peak_accel': round(max(accels) if accels else 0, 2),
                'speed_utilization': round(sum(ach_speeds)/max(sum(cmd_speeds),0.01), 3),
                'samples': len(self.entries),
            },
            'legs': leg_stats,
        }


# ─────────────────────────────────────────────
# PX4 Race Tuning
# ─────────────────────────────────────────────
RACE_PARAMS = {
    'MPC_ACC_HOR': 10.0,
    'MPC_ACC_HOR_MAX': 10.0,
    'MPC_JERK_AUTO': 30.0,
    'MPC_JERK_MAX': 50.0,
    'MPC_TILTMAX_AIR': 70.0,
    'MPC_XY_VEL_MAX': 15.0,
    'MPC_Z_VEL_MAX_UP': 5.0,
}

async def tune_racing(drone):
    for name, value in RACE_PARAMS.items():
        try:
            await drone.param.set_param_float(name, value)
        except:
            pass


def restart_px4():
    os.system('pkill -f mavsdk_server 2>/dev/null')
    os.system('pkill -9 -f "bin/px4" 2>/dev/null')
    time.sleep(3)
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters.bson 2>/dev/null')
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters_backup.bson 2>/dev/null')
    subprocess.Popen(
        ['/bin/bash', '/tmp/run_px4_sih.sh'],
        stdout=open('/tmp/px4_sih_out.log', 'w'),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    time.sleep(7)


# ─────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────
PROFILES = {
    'aggressive': {'max_speed': 12.0, 'cruise_speed': 10.0, 'base_blend': 2.5, 'threshold': 2.5},
    'balanced':   {'max_speed': 11.0, 'cruise_speed': 9.0,  'base_blend': 1.8, 'threshold': 2.5},
    'safe':       {'max_speed': 10.0, 'cruise_speed': 8.0,  'base_blend': 1.2, 'threshold': 2.0},
}


# ─────────────────────────────────────────────
# Trial Runner
# ─────────────────────────────────────────────
async def run_trial(course_name, profile_name, params, race_tune=True):
    course = COURSES[course_name]
    gates = course['gates']
    takeoff_alt = abs(course['altitude'])

    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    if race_tune:
        await tune_racing(drone)

    pos = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]
    att_yaw = 0.0

    async def pos_loop():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]

    async def att_loop():
        nonlocal att_yaw
        async for att in drone.telemetry.attitude_euler():
            att_yaw = att.yaw_deg

    asyncio.ensure_future(pos_loop())
    asyncio.ensure_future(att_loop())
    await asyncio.sleep(0.5)

    await drone.action.arm()
    await drone.action.set_takeoff_altitude(takeoff_alt)
    await drone.action.takeoff()
    await asyncio.sleep(4)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    planner = V3Planner(
        max_speed=params['max_speed'],
        cruise_speed=params['cruise_speed'],
        base_blend=params['base_blend']
    )
    threshold = params['threshold']
    logger = ResponseLogger()

    gate_idx = 0
    start_time = time.time()
    dt = 1.0 / 50
    timeout = 120  # 2 minute timeout

    while gate_idx < len(gates):
        if time.time() - start_time > timeout:
            print(f'    TIMEOUT at gate {gate_idx+1}/{len(gates)}')
            break

        gate = gates[gate_idx]
        dx, dy, dz = gate[0]-pos[0], gate[1]-pos[1], gate[2]-pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < threshold:
            elapsed = time.time() - start_time
            logger.log_gate(gate_idx+1, elapsed, pos, vel)
            gate_idx += 1
            continue

        next_g = gates[gate_idx+1] if gate_idx+1 < len(gates) else None
        cmd = planner.plan_velocity(pos, vel, gate, next_g)

        # Log before sending
        logger.log(
            time.time() - start_time,
            cmd.north_m_s, cmd.east_m_s, cmd.down_m_s, cmd.yaw_deg,
            vel[0], vel[1], vel[2], att_yaw,
            pos[0], pos[1], pos[2], gate_idx
        )

        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - start_time
    completed = gate_idx >= len(gates)

    try:
        await drone.offboard.stop()
    except:
        pass
    await drone.action.land()
    await asyncio.sleep(2)

    analysis = logger.analyze()

    return {
        'course': course_name,
        'profile': profile_name,
        'race_tuned': race_tune,
        'completed': completed,
        'gates_passed': gate_idx,
        'total_gates': len(gates),
        'total_time': round(total, 3),
        'gate_times': [g['t'] for g in logger.gate_events],
        'splits': [round(logger.gate_events[i]['t'] - (logger.gate_events[i-1]['t'] if i>0 else 0), 3) for i in range(len(logger.gate_events))],
        'tracking': analysis,
        'params': params,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main():
    all_results = []
    courses_to_run = ['sprint', 'technical', 'mixed']
    profiles_to_run = ['aggressive', 'balanced', 'safe']

    for course_name in courses_to_run:
        course = COURSES[course_name]
        print(f'\n{"="*60}')
        print(f'COURSE: {course_name.upper()} ({len(course["gates"])} gates)')
        print(f'  {course["description"]}')
        print(f'{"="*60}')

        for prof_name in profiles_to_run:
            params = PROFILES[prof_name]
            print(f'\n  --- {prof_name} ---')

            result = await run_trial(course_name, prof_name, params, race_tune=True)
            all_results.append(result)

            t = result['tracking']
            overall = t.get('overall', {})
            status = 'COMPLETE' if result['completed'] else f'FAILED at gate {result["gates_passed"]}/{result["total_gates"]}'
            print(f'    {status} | {result["total_time"]}s')
            print(f'    Max speed: {overall.get("max_ach_speed", 0)} m/s | Peak accel: {overall.get("peak_accel", 0)} m/s2')
            print(f'    Avg vel error: {overall.get("avg_vel_error", 0)} m/s | Speed util: {overall.get("speed_utilization", 0):.1%}')
            print(f'    Splits: {result["splits"]}')

            # Restart PX4
            restart_px4()
            await asyncio.sleep(2)

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    print(f'\n\n{"="*80}')
    print('FULL BENCHMARK SUMMARY')
    print(f'{"="*80}')

    for course_name in courses_to_run:
        cr = [r for r in all_results if r['course'] == course_name]
        print(f'\n  {course_name.upper()} ({COURSES[course_name]["description"]})')
        print(f'  {"Profile":<15} {"Status":<10} {"Total":>7} {"MaxSpd":>7} {"AvgErr":>7} {"PkAccl":>7} {"SpdUtil":>8}')
        print(f'  {"-"*62}')
        for r in cr:
            o = r['tracking'].get('overall', {})
            st = 'OK' if r['completed'] else 'FAIL'
            print(f'  {r["profile"]:<15} {st:<10} {r["total_time"]:>6.2f}s {o.get("max_ach_speed",0):>5.1f} {o.get("avg_vel_error",0):>6.3f} {o.get("peak_accel",0):>6.1f} {o.get("speed_utilization",0):>7.1%}')

    # Per-leg detail for best run per course
    print(f'\n\n{"="*80}')
    print('LEG-BY-LEG DETAIL (aggressive profile)')
    print(f'{"="*80}')
    for course_name in courses_to_run:
        cr = [r for r in all_results if r['course'] == course_name and r['profile'] == 'aggressive']
        if cr:
            r = cr[0]
            legs = r['tracking'].get('legs', [])
            print(f'\n  {course_name.upper()}:')
            print(f'  {"Leg":<5} {"Time":>6} {"EntSpd":>7} {"AvgCmd":>7} {"AvgAch":>7} {"MaxAch":>7} {"AvgErr":>7} {"PkAcc":>6} {"Util":>6}')
            print(f'  {"-"*62}')
            for leg in legs:
                print(f'  {leg["gate"]:>3}   {leg["duration"]:>5.2f}s {leg["entry_speed"]:>5.1f} {leg["avg_cmd_speed"]:>6.1f} {leg["avg_ach_speed"]:>6.1f} {leg["max_ach_speed"]:>6.1f} {leg["avg_vel_error"]:>6.3f} {leg["peak_accel"]:>5.1f} {leg["speed_utilization"]:>5.1%}')

    # Save
    out_path = '/Users/conradweeden/ai-grand-prix/logs/px4_benchmark_v2.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')

asyncio.run(main())
