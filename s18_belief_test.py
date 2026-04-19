"""Session 18 — Belief model A/B test.

Compares S16 VisionNav (control) vs BeliefNav (treatment) on the conditions
that matter: clean, mild, harsh, and A3_periodic_dropout.

4 conditions × 2 navigators × 3 trials = 24 flights.

Pass criteria:
  clean ≤ +2.5s over GT (≤17.9s)
  mild ≤ +3.0s over GT (≤18.4s)
  harsh ≥ 2/3 completion
  search time reduced ≥ 50% vs S17 baseline
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
from gate_belief import GateBelief, BeliefNav

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 's18_belief.json')
COURSE = 'technical'
GATES = COURSES[COURSE]
N_TRIALS = 3
B = os.path.expanduser('~/PX4-Autopilot')

# ── Noise configs (same as S17) ──────────────────────────────────
NOISE_CONFIGS = {
    'clean': dict(NOISE_PROFILES['clean']),
    'mild': dict(NOISE_PROFILES['mild']),
    'harsh': dict(NOISE_PROFILES['harsh']),
    'A3_periodic_dropout': {
        'bearing_sigma_deg': 0.5, 'range_sigma_frac': 0.02,
        'miss_prob': 0.0, 'fov_h_deg': 120.0, 'fov_v_deg': 90.0,
        'max_detect_range': 60.0, 'min_detect_range': 0.5,
        'bearing_bias_deg': 0.0,
        '_forced_dropout_on_s': 0.5, '_forced_dropout_period_s': 3.0,
    },
}

# ── Conditions: (noise_key, nav_type) ────────────────────────────
CONDITIONS = [
    ('clean', 'vision'),      # S16 VisionNav control
    ('clean', 'belief'),      # BeliefNav treatment
    ('mild', 'vision'),
    ('mild', 'belief'),
    ('harsh', 'vision'),
    ('harsh', 'belief'),
    ('A3_periodic_dropout', 'vision'),
    ('A3_periodic_dropout', 'belief'),
]


# ── AblationCamera with forced dropout (from S17) ────────────────
class AblationCamera(VirtualCamera):
    """VirtualCamera with optional forced periodic dropout."""

    def __init__(self, gates, noise_profile_dict, t0=None):
        self.gates = gates
        self.np = noise_profile_dict
        self.rng = random.Random(42)
        self.t0 = t0 or time.time()
        self._dropout_on = noise_profile_dict.get('_forced_dropout_on_s', 0)
        self._dropout_period = noise_profile_dict.get('_forced_dropout_period_s', 0)

    def observe(self, pos, vel, yaw_deg):
        if self._dropout_period > 0:
            elapsed = time.time() - self.t0
            cycle_pos = elapsed % self._dropout_period
            if cycle_pos < self._dropout_on:
                return []  # forced blackout
        return super().observe(pos, vel, yaw_deg)


# ── Trial runner ──────────────────────────────────────────────────
async def run_trial(gates, noise_key, nav_type, trial_num):
    """Run one trial with specified noise config and navigator type."""
    noise_dict = dict(NOISE_CONFIGS[noise_key])

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

    # ── Build vision + nav stack ──
    t0_flight = time.time()
    camera = AblationCamera(gates, noise_dict, t0=t0_flight)
    tracker = GateTracker(len(gates))

    if nav_type == 'belief':
        nav = BeliefNav()
    else:
        nav = VisionNav()

    # ── Flight loop with metrics ──
    t0 = time.time()
    dt = 1 / 50
    max_spd = 0
    cmds = []; achs = []; heading_errors = []; vel_norms = []
    frames_tracking = 0; frames_coast = 0; frames_search = 0
    detection_losses = 0; was_detected = True
    reacq_times = []; loss_start = None

    while True:
        if time.time() - t0 > 90:
            break

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

        # Heading error
        if state.detected:
            target_yaw = math.radians(yaw_deg) + state.bearing_h
            vel_bearing = math.atan2(vel[1], vel[0])
            herr = abs(math.degrees(target_yaw - vel_bearing))
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
    gates_passed = tracker.gates_passed
    completed = gates_passed >= len(gates)

    vel_std = stdev(vel_norms) if len(vel_norms) > 1 else 0
    avg_herr = mean(heading_errors) if heading_errors else 0
    avg_reacq = mean(reacq_times) if reacq_times else 0

    return {
        'noise': noise_key, 'nav': nav_type, 'trial': trial_num,
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


async def run_one(noise_key, nav_type, trial_num, results):
    tag = f'[{noise_key}/{nav_type} t{trial_num}]'
    print(f'{tag} hardened restart...', flush=True)
    try:
        bench.hardened_restart(B)
    except Exception as e:
        print(f'{tag} BENCH FAIL: {e}', flush=True)
        results.append({
            'noise': noise_key, 'nav': nav_type, 'trial': trial_num,
            'completed': False, 'error': f'restart: {e}',
            'bench_failure': True,
        })
        bench.atomic_write_json(RESULT, results)
        return

    print(f'{tag} flying...', flush=True)
    try:
        r = await run_trial(GATES, noise_key, nav_type, trial_num)
        print(f'{tag} -> lap={r["time"]} gates={r["gates_passed"]}/{len(GATES)} '
              f'done={r["completed"]} trk={r["pct_tracking"]}% '
              f'coast={r["pct_coast"]}% srch={r["pct_search"]}% '
              f'losses={r["detection_losses"]} herr={r["avg_heading_error"]} '
              f'v_osc={r["vel_oscillation"]}', flush=True)
    except Exception as e:
        r = {'noise': noise_key, 'nav': nav_type, 'trial': trial_num,
             'completed': False, 'error': str(e), 'bench_failure': False}
        print(f'{tag} ERROR: {e}', flush=True)

    results.append(r)
    bench.atomic_write_json(RESULT, results)


async def main():
    bench.acquire_singleton('s18')
    print(f'[init] Session 18 — Belief Model A/B Test', flush=True)
    print(f'[init] Course: {COURSE} ({len(GATES)} gates)', flush=True)
    print(f'[init] Conditions: {len(CONDITIONS)}', flush=True)
    print(f'[init] Trials per condition: {N_TRIALS}', flush=True)
    print(f'[init] Total flights: {len(CONDITIONS) * N_TRIALS}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    for noise_key, nav_type in CONDITIONS:
        print(f'\n=== {noise_key}/{nav_type} ({N_TRIALS} trials) ===', flush=True)
        for t in range(1, N_TRIALS + 1):
            await run_one(noise_key, nav_type, t, results)

    # Summary
    print(f'\n=== SUMMARY ===', flush=True)
    print(f'{"condition":30s} {"done":5s} {"median":8s} {"trk%":6s} {"cst%":6s} '
          f'{"src%":6s} {"losses":7s} {"reacq":7s} {"herr":6s} {"v_osc":7s}', flush=True)
    for noise_key, nav_type in CONDITIONS:
        label = f'{noise_key}/{nav_type}'
        done = [r for r in results if r.get('noise') == noise_key and r.get('nav') == nav_type and r.get('completed')]
        fail = [r for r in results if r.get('noise') == noise_key and r.get('nav') == nav_type and not r.get('completed')]
        if done:
            times = [r['time'] for r in done]
            print(f'{label:30s} {len(done)}/{len(done)+len(fail):3s}  '
                  f'{median(times):7.3f}  '
                  f'{mean([r.get("pct_tracking",0) for r in done]):5.1f}  '
                  f'{mean([r.get("pct_coast",0) for r in done]):5.1f}  '
                  f'{mean([r.get("pct_search",0) for r in done]):5.1f}  '
                  f'{mean([r.get("detection_losses",0) for r in done]):6.1f}  '
                  f'{mean([r.get("avg_reacq_time",0) for r in done]):6.3f}  '
                  f'{mean([r.get("avg_heading_error",0) for r in done]):5.1f}  '
                  f'{mean([r.get("vel_oscillation",0) for r in done]):6.3f}', flush=True)
        else:
            print(f'{label:30s} 0/{len(fail)}    DNF', flush=True)

    bench.atomic_write_json(RESULT, results)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
