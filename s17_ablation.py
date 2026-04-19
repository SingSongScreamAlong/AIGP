"""Session 17 — Ablation harness.

Runs controlled ablations to isolate the vision-nav perception tax.
No new features. Just isolation experiments.

Ablations:
  A1: Vision path, zero noise (architecture cost only)
  A2a: Noisy bearing, perfect range
  A2b: Perfect bearing, noisy range
  A3: Forced periodic detection dropout (0.5s gaps every 3s)
  A4: Search mode disabled (hold last bearing on loss)
  A5: Speed clamped to 70% of V5.1 cruise
  GT: Ground truth baseline (control)
  S16_clean: S16 clean profile (reference)
  S16_mild: S16 mild profile (reference)
  S16_harsh: S16 harsh profile (reference)

Metrics per run: lap time, completion, % track/coast/search,
detection loss count, avg reacquisition time, heading error vs gate,
velocity oscillation (std dev).
"""

import asyncio, time, math, random, json, os, sys
from statistics import median, mean, stdev
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from px4_v51_baseline import V51Planner, COURSES

from vision_nav import (
    VirtualCamera, GateTracker, VisionNav, TrackerState,
    GateDetection, NOISE_PROFILES, RACE_PARAMS,
)

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 's17_ablation.json')
COURSE = 'technical'
GATES = COURSES[COURSE]
N_TRIALS = 3
B = os.path.expanduser('~/PX4-Autopilot')

# ── Ablation noise profiles ──────────────────────────────────────
ABLATION_PROFILES = {
    'A1_zero_noise': {
        'bearing_sigma_deg': 0.0, 'range_sigma_frac': 0.0,
        'miss_prob': 0.0, 'fov_h_deg': 120.0, 'fov_v_deg': 90.0,
        'max_detect_range': 60.0, 'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
    },
    'A2a_noisy_bearing': {
        'bearing_sigma_deg': 2.0, 'range_sigma_frac': 0.0,
        'miss_prob': 0.0, 'fov_h_deg': 120.0, 'fov_v_deg': 90.0,
        'max_detect_range': 60.0, 'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
    },
    'A2b_noisy_range': {
        'bearing_sigma_deg': 0.0, 'range_sigma_frac': 0.08,
        'miss_prob': 0.0, 'fov_h_deg': 120.0, 'fov_v_deg': 90.0,
        'max_detect_range': 60.0, 'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
    },
    'A3_periodic_dropout': {
        'bearing_sigma_deg': 0.5, 'range_sigma_frac': 0.02,
        'miss_prob': 0.0,  # dropout handled externally
        'fov_h_deg': 120.0, 'fov_v_deg': 90.0,
        'max_detect_range': 60.0, 'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
        '_forced_dropout_on_s': 0.5,   # blackout duration
        '_forced_dropout_period_s': 3.0,  # every N seconds
    },
}

# Add S16 profiles for reference
for name, profile in NOISE_PROFILES.items():
    ABLATION_PROFILES[f'S16_{name}'] = dict(profile)

# ── Conditions list ──────────────────────────────────────────────
CONDITIONS = [
    'groundtruth',
    'A1_zero_noise',
    'A2a_noisy_bearing',
    'A2b_noisy_range',
    'A3_periodic_dropout',
    'A4_no_search',      # uses clean noise but disables search
    'A5_speed_clamp',    # uses clean noise but clamps speed to 70%
    'S16_clean',
    'S16_mild',
    'S16_harsh',
]


# ── Enhanced camera with forced dropout support ──────────────────
class AblationCamera(VirtualCamera):
    """VirtualCamera with optional forced periodic dropout."""

    def __init__(self, gates, noise_profile_dict, t0=None):
        # Manually init to use a dict directly
        self.gates = gates
        self.np = noise_profile_dict
        self.rng = random.Random(42)
        self.t0 = t0 or time.time()
        self._dropout_on = noise_profile_dict.get('_forced_dropout_on_s', 0)
        self._dropout_period = noise_profile_dict.get('_forced_dropout_period_s', 0)

    def observe(self, pos, vel, yaw_deg):
        # Check forced dropout
        if self._dropout_period > 0:
            elapsed = time.time() - self.t0
            cycle_pos = elapsed % self._dropout_period
            if cycle_pos < self._dropout_on:
                return []  # forced blackout

        return super().observe(pos, vel, yaw_deg)


# ── Enhanced run_trial with full metrics ─────────────────────────
async def run_trial_ablation(gates, condition, trial_num):
    """Run one trial with full ablation metrics."""
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    for n, v in RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0, 0, 0]; vel = [0, 0, 0]; yaw_deg = 0.0

    async def telem_loop():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]

    async def heading_loop():
        nonlocal yaw_deg
        async for h in drone.telemetry.heading():
            yaw_deg = h.heading_deg

    asyncio.ensure_future(telem_loop())
    asyncio.ensure_future(heading_loop())
    await asyncio.sleep(0.5)

    # Takeoff
    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    _wait_start = time.time()
    while True:
        if (time.time() - _wait_start) >= 10.0: break
        if abs(pos[2]) >= 0.95 * alt and abs(vel[2]) < 0.3: break
        await asyncio.sleep(0.05)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    # ── Setup vision stack based on condition ──
    is_groundtruth = (condition == 'groundtruth')
    is_A4 = (condition == 'A4_no_search')
    is_A5 = (condition == 'A5_speed_clamp')

    if is_groundtruth:
        planner = V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
        planner.px4_speed_ceiling = 9.5
        camera = None; tracker = None; nav = None
    else:
        if is_A4:
            noise_dict = dict(NOISE_PROFILES['clean'])
        elif is_A5:
            noise_dict = dict(NOISE_PROFILES['clean'])
        elif condition in ABLATION_PROFILES:
            noise_dict = dict(ABLATION_PROFILES[condition])
        else:
            noise_dict = dict(NOISE_PROFILES.get(condition, NOISE_PROFILES['clean']))

        t0_flight = time.time()
        camera = AblationCamera(gates, noise_dict, t0=t0_flight)
        tracker = GateTracker(len(gates))
        if is_A4:
            tracker.SEARCH_TIMEOUT_FRAMES = 999999  # never enter search
        nav = VisionNav()
        if is_A5:
            nav.cruise_speed = 9.0 * 0.7  # 70% clamp
            nav.max_speed = 12.0 * 0.7

    # ── Flight loop with full metrics ──
    t0 = time.time()
    dt = 1 / 50
    max_spd = 0
    cmds = []; achs = []; heading_errors = []; vel_norms = []
    frames_tracking = 0; frames_coast = 0; frames_search = 0
    detection_losses = 0; was_detected = True
    reacq_times = []; loss_start = None
    gi = 0  # for groundtruth

    while True:
        if time.time() - t0 > 90:
            break

        if is_groundtruth:
            if gi >= len(gates): break
            g = gates[gi]
            d = math.sqrt((g[0]-pos[0])**2 + (g[1]-pos[1])**2 + (g[2]-pos[2])**2)
            if d < 2.5:
                gspd = math.sqrt(vel[0]**2 + vel[1]**2)
                planner.on_gate_passed(gspd)
                gi += 1; continue
            ng = gates[gi+1] if gi+1 < len(gates) else None
            cmd = planner.plan(pos, vel, g, ng)
            frames_tracking += 1

            # Heading error for GT: angle between velocity and gate direction
            dx, dy = g[0]-pos[0], g[1]-pos[1]
            gate_bearing = math.atan2(dy, dx)
            vel_bearing = math.atan2(vel[1], vel[0])
            herr = abs(math.degrees(gate_bearing - vel_bearing))
            if herr > 180: herr = 360 - herr
            heading_errors.append(herr)
        else:
            if tracker.current_target >= len(gates): break
            detections = camera.observe(pos, vel, yaw_deg)
            state = tracker.update(detections, dt)
            cmd = nav.plan(state, pos, vel, yaw_deg, dt)

            # Metrics
            if state.detected:
                frames_tracking += 1
                if not was_detected and loss_start is not None:
                    reacq_times.append(time.time() - loss_start)
                    loss_start = None
                was_detected = True
            elif state.search_mode:
                frames_search += 1
                if was_detected:
                    detection_losses += 1
                    loss_start = time.time()
                was_detected = False
            else:
                frames_coast += 1
                if was_detected:
                    detection_losses += 1
                    loss_start = time.time()
                was_detected = False

            # Heading error: angle between velocity and tracker bearing
            if state.detected:
                target_yaw = yaw_deg + math.degrees(state.bearing_h)
                vel_bearing = math.degrees(math.atan2(vel[1], vel[0]))
                herr = abs(target_yaw - vel_bearing)
                if herr > 180: herr = 360 - herr
                heading_errors.append(herr)

        cspd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        aspd = math.sqrt(vel[0]**2 + vel[1]**2)
        cmds.append(cspd); achs.append(aspd)
        vel_norms.append(aspd)
        if aspd > max_spd: max_spd = aspd

        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0

    # Cleanup
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land(); await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    await asyncio.sleep(2)
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    await asyncio.sleep(2)

    total_frames = frames_tracking + frames_coast + frames_search
    gates_passed = gi if is_groundtruth else tracker.gates_passed
    completed = gates_passed >= len(gates)

    vel_std = stdev(vel_norms) if len(vel_norms) > 1 else 0
    avg_herr = mean(heading_errors) if heading_errors else 0
    avg_reacq = mean(reacq_times) if reacq_times else 0

    return {
        'condition': condition, 'trial': trial_num,
        'completed': completed, 'time': round(total, 3),
        'gates_passed': gates_passed,
        'max_spd': round(max_spd, 2),
        'avg_cmd': round(mean(cmds), 2) if cmds else 0,
        'avg_ach': round(mean(achs), 2) if achs else 0,
        'total_frames': total_frames,
        'pct_tracking': round(100 * frames_tracking / max(total_frames, 1), 1),
        'pct_coast': round(100 * frames_coast / max(total_frames, 1), 1),
        'pct_search': round(100 * frames_search / max(total_frames, 1), 1),
        'detection_losses': detection_losses,
        'avg_reacq_time': round(avg_reacq, 3),
        'avg_heading_error': round(avg_herr, 1),
        'vel_oscillation': round(vel_std, 3),
        'bench_failure': False,
    }


async def run_one(condition, trial_num, results):
    tag = f'[{condition} t{trial_num}]'
    print(f'{tag} hardened restart...', flush=True)
    try:
        bench.hardened_restart(B)
    except Exception as e:
        print(f'{tag} BENCH FAIL: {e}', flush=True)
        results.append({
            'condition': condition, 'trial': trial_num,
            'completed': False, 'error': f'restart: {e}',
            'bench_failure': True,
        })
        bench.atomic_write_json(RESULT, results)
        return

    print(f'{tag} flying...', flush=True)
    try:
        r = await run_trial_ablation(GATES, condition, trial_num)
        print(f'{tag} -> lap={r["time"]} gates={r["gates_passed"]}/{len(GATES)} '
              f'done={r["completed"]} trk={r["pct_tracking"]}% '
              f'coast={r["pct_coast"]}% srch={r["pct_search"]}% '
              f'losses={r["detection_losses"]} herr={r["avg_heading_error"]} '
              f'vel_osc={r["vel_oscillation"]}', flush=True)
    except Exception as e:
        r = {'condition': condition, 'trial': trial_num,
             'completed': False, 'error': str(e), 'bench_failure': False}
        print(f'{tag} ERROR: {e}', flush=True)

    results.append(r)
    bench.atomic_write_json(RESULT, results)


async def main():
    bench.acquire_singleton('abl17')
    print(f'[init] Session 17 — Ablation Matrix', flush=True)
    print(f'[init] Course: {COURSE} ({len(GATES)} gates)', flush=True)
    print(f'[init] Conditions: {CONDITIONS}', flush=True)
    print(f'[init] Trials per condition: {N_TRIALS}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    for cond in CONDITIONS:
        print(f'\n=== {cond.upper()} ({N_TRIALS} trials) ===', flush=True)
        for t in range(1, N_TRIALS + 1):
            await run_one(cond, t, results)

    # Summary
    print(f'\n=== SUMMARY ===', flush=True)
    print(f'{"condition":25s} {"done":5s} {"median":8s} {"trk%":6s} {"cst%":6s} '
          f'{"src%":6s} {"losses":7s} {"reacq":7s} {"herr":6s} {"v_osc":7s}', flush=True)
    for cond in CONDITIONS:
        done = [r for r in results if r['condition'] == cond and r.get('completed')]
        fail = [r for r in results if r['condition'] == cond and not r.get('completed')]
        if done:
            times = [r['time'] for r in done]
            print(f'{cond:25s} {len(done)}/{len(done)+len(fail):3s}  '
                  f'{median(times):7.3f}  '
                  f'{mean([r.get("pct_tracking",0) for r in done]):5.1f}  '
                  f'{mean([r.get("pct_coast",0) for r in done]):5.1f}  '
                  f'{mean([r.get("pct_search",0) for r in done]):5.1f}  '
                  f'{mean([r.get("detection_losses",0) for r in done]):6.1f}  '
                  f'{mean([r.get("avg_reacq_time",0) for r in done]):6.3f}  '
                  f'{mean([r.get("avg_heading_error",0) for r in done]):5.1f}  '
                  f'{mean([r.get("vel_oscillation",0) for r in done]):6.3f}', flush=True)
        else:
            print(f'{cond:25s} 0/{len(fail)}    DNF', flush=True)

    bench.atomic_write_json(RESULT, results)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
