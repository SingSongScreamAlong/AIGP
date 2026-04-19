# s14_px4_sweep.py — Session 14: PX4 Controller Tracking Floor Investigation
# Sweeps key PX4 controller params one-at-a-time against locked planner baseline.
# For each config: run 1 technical lap, measure tracking quality metrics.
# Goal: identify which PX4 params have leverage on the cmd→ach gap.

import asyncio, time, math, json, os, sys
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
import bench
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from statistics import median, mean

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
RESULT = os.path.join(LOGS, 's14_px4_sweep.json')
TRIAL_TIMEOUT = 90.0
POST_RESTART_WAIT = 5.0

# ---- Monkey-patch for phase tracing (same as px4_trace_lap.py) ----
_orig_phase = B.V51Planner._phase
def _phase_trace(self, dxy, cs, ta, db):
    ph = _orig_phase(self, dxy, cs, ta, db)
    self._trace_phase = ph
    return ph
B.V51Planner._phase = _phase_trace

# ---- Sweep configs ----
# Baseline config (current locked tune)
BASELINE = {
    'MPC_XY_VEL_P_ACC': 6.0,
    'MPC_ACC_HOR': 10.0,
    'MPC_ACC_HOR_MAX': 10.0,
    'MPC_JERK_AUTO': 30.0,
    'MPC_JERK_MAX': 50.0,
    'MPC_TILTMAX_AIR': 70.0,
    'MPC_XY_VEL_MAX': 15.0,
    'MPC_Z_VEL_MAX_UP': 5.0,
}

# Sweep: one param varies, rest at baseline
SWEEPS = [
    # Velocity P gain — higher = more aggressive speed tracking
    {'name': 'baseline',         'params': {}},
    {'name': 'vel_p_4.0',       'params': {'MPC_XY_VEL_P_ACC': 4.0}},
    {'name': 'vel_p_8.0',       'params': {'MPC_XY_VEL_P_ACC': 8.0}},
    {'name': 'vel_p_10.0',      'params': {'MPC_XY_VEL_P_ACC': 10.0}},
    # Horizontal acceleration limit — higher = faster speed changes
    {'name': 'acc_hor_12.0',    'params': {'MPC_ACC_HOR': 12.0, 'MPC_ACC_HOR_MAX': 12.0}},
    {'name': 'acc_hor_15.0',    'params': {'MPC_ACC_HOR': 15.0, 'MPC_ACC_HOR_MAX': 15.0}},
    # Jerk limit — higher = snappier acceleration onset
    {'name': 'jerk_45',         'params': {'MPC_JERK_AUTO': 45.0, 'MPC_JERK_MAX': 75.0}},
    {'name': 'jerk_60',         'params': {'MPC_JERK_AUTO': 60.0, 'MPC_JERK_MAX': 100.0}},
    # Max tilt — higher = more aggressive attitude for speed
    {'name': 'tilt_80',         'params': {'MPC_TILTMAX_AIR': 80.0}},
    {'name': 'tilt_85',         'params': {'MPC_TILTMAX_AIR': 85.0}},
    # Max velocity — raise ceiling PX4 will target
    {'name': 'vel_max_18',      'params': {'MPC_XY_VEL_MAX': 18.0}},
    {'name': 'vel_max_20',      'params': {'MPC_XY_VEL_MAX': 20.0}},
    # Combo: aggressive tracking stack
    {'name': 'aggressive_combo', 'params': {
        'MPC_XY_VEL_P_ACC': 8.0,
        'MPC_ACC_HOR': 15.0,
        'MPC_ACC_HOR_MAX': 15.0,
        'MPC_JERK_AUTO': 45.0,
        'MPC_JERK_MAX': 75.0,
        'MPC_TILTMAX_AIR': 80.0,
    }},
]

def angle_deg(ax, ay, bx, by):
    ma = math.sqrt(ax*ax + ay*ay)
    mb = math.sqrt(bx*bx + by*by)
    if ma < 0.01 or mb < 0.01:
        return 0.0
    dot = (ax*bx + ay*by) / (ma * mb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

async def run_one_lap(config_name, param_overrides):
    """Run a single technical lap with given PX4 param overrides. Returns metrics dict."""
    gates = B.COURSES['technical']

    # Build full param set: baseline + overrides
    params = dict(BASELINE)
    params.update(param_overrides)

    # Hardened restart
    print(f'  [{config_name}] hardened restart...', flush=True)
    try:
        bench.hardened_restart(B)
    except Exception as e:
        return {'config': config_name, 'error': f'restart_failed: {e}', 'completed': False}

    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    # Set PX4 params
    for n, v in params.items():
        try:
            await drone.param.set_param_float(n, v)
        except Exception as e:
            print(f'    param {n}={v} failed: {e}', flush=True)

    await asyncio.sleep(POST_RESTART_WAIT)

    # Telemetry
    pos = [0, 0, 0]; vel = [0, 0, 0]
    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    # Planner — same locked baseline
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)

    # Takeoff
    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    gate_alt_frac = 0.95; gate_vz_max = 0.3; gate_timeout = 10.0
    wait_start = time.time()
    while True:
        if (time.time() - wait_start) >= gate_timeout: break
        if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break
        await asyncio.sleep(0.05)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    # Fly lap, collect samples
    samples = []
    gate_events = []
    gi = 0; t0 = time.time(); dt = 1/50; threshold = 2.5

    while gi < len(gates):
        if time.time() - t0 > TRIAL_TIMEOUT:
            break
        g = gates[gi]
        dxg = g[0]-pos[0]; dyg = g[1]-pos[1]; dzg = g[2]-pos[2]
        d = math.sqrt(dxg*dxg + dyg*dyg + dzg*dzg)
        if d < threshold:
            tg = time.time() - t0
            gspd = math.sqrt(vel[0]**2 + vel[1]**2)
            gate_events.append({'gate_idx': gi, 'time': round(tg, 4), 'ach_speed': round(gspd, 3)})
            planner.on_gate_passed(gspd)
            gi += 1
            continue
        ng = gates[gi+1] if gi+1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        cvn, cve, cvd = cmd.north_m_s, cmd.east_m_s, cmd.down_m_s
        cmd_spd = math.sqrt(cvn*cvn + cve*cve)
        ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)
        samples.append({
            't': round(time.time()-t0, 4),
            'leg_idx': gi,
            'phase': getattr(planner, '_trace_phase', None),
            'vel_n': round(vel[0], 3), 'vel_e': round(vel[1], 3),
            'cmd_vn': round(cvn, 3), 'cmd_ve': round(cve, 3),
            'cmd_spd': round(cmd_spd, 3), 'ach_spd': round(ach_spd, 3),
        })
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0
    completed = gi >= len(gates)

    try: await drone.offboard.stop()
    except: pass
    await drone.action.land()
    await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    # Compute metrics
    if not samples:
        return {'config': config_name, 'error': 'no_samples', 'completed': False}

    all_gaps = [s['cmd_spd'] - s['ach_spd'] for s in samples]
    all_hdg = [angle_deg(s['cmd_vn'], s['cmd_ve'], s['vel_n'], s['vel_e']) for s in samples]
    all_ach = [s['ach_spd'] for s in samples]
    all_cmd = [s['cmd_spd'] for s in samples]

    # SUSTAIN-only metrics (most informative for tracking quality)
    sus_samples = [s for s in samples if s.get('phase') == 'SUSTAIN']
    sus_gaps = [s['cmd_spd'] - s['ach_spd'] for s in sus_samples] if sus_samples else []
    sus_hdg = [angle_deg(s['cmd_vn'], s['cmd_ve'], s['vel_n'], s['vel_e']) for s in sus_samples] if sus_samples else []
    sus_ach = [s['ach_spd'] for s in sus_samples] if sus_samples else []

    within_1 = sum(1 for g in all_gaps if abs(g) < 1.0) / max(len(all_gaps), 1) * 100
    within_2 = sum(1 for g in all_gaps if abs(g) < 2.0) / max(len(all_gaps), 1) * 100

    result = {
        'config': config_name,
        'params': param_overrides,
        'completed': completed,
        'lap_time': round(total, 3),
        'gates_passed': gi,
        'n_samples': len(samples),
        'mean_cmd_spd': round(mean(all_cmd), 3),
        'mean_ach_spd': round(mean(all_ach), 3),
        'median_spd_gap': round(median(all_gaps), 3),
        'mean_spd_gap': round(mean(all_gaps), 3),
        'p90_spd_gap': round(sorted(all_gaps)[int(0.9*len(all_gaps))], 3),
        'within_1ms': round(within_1, 1),
        'within_2ms': round(within_2, 1),
        'mean_hdg_err': round(mean(all_hdg), 1),
        'p90_hdg_err': round(sorted(all_hdg)[int(0.9*len(all_hdg))], 1),
        'sus_mean_ach': round(mean(sus_ach), 3) if sus_ach else None,
        'sus_mean_gap': round(mean(sus_gaps), 3) if sus_gaps else None,
        'sus_mean_hdg': round(mean(sus_hdg), 1) if sus_hdg else None,
    }
    return result

async def main():
    bench.acquire_singleton('px4sweep14')
    print(f'[init] Session 14 — PX4 Controller Tracking Floor Sweep', flush=True)
    print(f'[init] {len(SWEEPS)} configs to test', flush=True)
    print(f'[init] result file: {RESULT}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    for i, sweep in enumerate(SWEEPS):
        print(f'\n[{i+1}/{len(SWEEPS)}] config={sweep["name"]}  overrides={sweep["params"]}', flush=True)
        r = await run_one_lap(sweep['name'], sweep['params'])
        results.append(r)
        bench.atomic_write_json(RESULT, results)

        if r.get('completed'):
            print(f'  -> lap={r["lap_time"]}s  ach={r["mean_ach_spd"]}  gap={r["mean_spd_gap"]}  '
                  f'hdg={r["mean_hdg_err"]}d  within1={r["within_1ms"]}%  '
                  f'sus_ach={r.get("sus_mean_ach")}  sus_gap={r.get("sus_mean_gap")}', flush=True)
        else:
            print(f'  -> FAILED: {r.get("error", "timeout/incomplete")}', flush=True)

    # ---- Summary table ----
    print('\n' + '=' * 110)
    print('SESSION 14 — PX4 CONTROLLER SWEEP RESULTS')
    print('=' * 110)
    print(f'{"config":<20s} {"lap":>6s} {"ach_spd":>7s} {"gap":>6s} {"hdg":>5s} {"w1%":>5s} {"w2%":>5s} '
          f'{"sus_ach":>7s} {"sus_gap":>7s} {"sus_hdg":>7s}')
    print('-' * 110)
    baseline_lap = None
    for r in results:
        if not r.get('completed'):
            print(f'{r["config"]:<20s}  FAILED ({r.get("error", "timeout")})')
            continue
        if r['config'] == 'baseline':
            baseline_lap = r['lap_time']
        delta = f'{r["lap_time"] - baseline_lap:+.3f}' if baseline_lap else 'n/a'
        print(f'{r["config"]:<20s} {r["lap_time"]:>6.3f} {r["mean_ach_spd"]:>7.3f} {r["mean_spd_gap"]:>6.3f} '
              f'{r["mean_hdg_err"]:>5.1f} {r["within_1ms"]:>5.1f} {r["within_2ms"]:>5.1f} '
              f'{r.get("sus_mean_ach","n/a"):>7} {r.get("sus_mean_gap","n/a"):>7} {r.get("sus_mean_hdg","n/a"):>7}  '
              f'd={delta}')

    print('\n--- END SESSION 14 ---')
    bench.atomic_write_json(RESULT, results)

if __name__ == '__main__':
    asyncio.run(main())
