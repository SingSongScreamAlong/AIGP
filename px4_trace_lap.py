# Single-lap sample-level trace on the NEW locked baseline.
#   - gated takeoff (use_takeoff_gate=True, gate_alt_frac=0.95, gate_vz_max=0.3, gate_timeout=10.0)
#   - VEL_P_ACC=6.0, ACC_HOR=10, JERK_AUTO=30
#   - V5.1 planner (max_speed=12, cruise_speed=9, base_blend=1.5, px4_speed_ceiling=9.5) [Session 7 promoted]
#   - 5s post-restart cooldown (proven necessary for mixed)
#
# Runs ONE lap on technical and ONE lap on mixed. Dumps per-sample JSON + CSV.
# Captures: t, leg_idx, phase (from monkeypatched V51Planner._phase),
# pos/vel/cmd, cmd_spd, ach_spd, err_spd, dxy, d3, ta, db, desired, cmd_pre_smooth.
import asyncio, time, math, json, os, sys, csv
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS              = '/Users/conradweeden/ai-grand-prix/logs'
TRIAL_TIMEOUT     = 200.0
POST_RESTART_WAIT = 5.0

# ---- Monkey-patch V51Planner._phase and _px4_cmd to stash trace context ----
_orig_phase = B.V51Planner._phase
def _phase_trace(self, dxy, cs, ta, db):
    ph = _orig_phase(self, dxy, cs, ta, db)
    self._trace_phase = ph
    self._trace_dxy   = dxy
    self._trace_cs_in = cs
    self._trace_ta    = ta
    self._trace_db    = db
    return ph
B.V51Planner._phase = _phase_trace

_orig_px4_cmd = B.V51Planner._px4_cmd
def _px4_cmd_trace(self, desired, ta_for_util):
    out = _orig_px4_cmd(self, desired, ta_for_util)
    self._trace_desired        = desired
    self._trace_cmd_pre_smooth = out
    return out
B.V51Planner._px4_cmd = _px4_cmd_trace


async def run_trace(course_name):
    gates = B.COURSES[course_name]
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break
    for n, v in B.RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0, 0, 0]; vel = [0, 0, 0]

    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    # Gated takeoff (matches the promoted baseline)
    gate_alt_frac = 0.95
    gate_vz_max   = 0.3
    gate_timeout  = 10.0
    wait_start = time.time()
    while True:
        if (time.time() - wait_start) >= gate_timeout: break
        if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break
        await asyncio.sleep(0.05)
    t_offboard_wait = round(time.time() - wait_start, 3)
    alt_at_offboard = round(abs(pos[2]), 3)
    vz_at_offboard  = round(vel[2], 3)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    await drone.offboard.start()

    samples     = []
    gate_events = []
    gi  = 0
    t0  = time.time()
    dt  = 1/50
    threshold = 2.5

    while gi < len(gates):
        if time.time() - t0 > 90:
            print(f'    TIMEOUT@gate{gi+1}')
            break
        g = gates[gi]
        dxg = g[0] - pos[0]; dyg = g[1] - pos[1]; dzg = g[2] - pos[2]
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

        cmd_vx, cmd_vy, cmd_vz = cmd.north_m_s, cmd.east_m_s, cmd.down_m_s
        cmd_spd = math.sqrt(cmd_vx*cmd_vx + cmd_vy*cmd_vy)
        ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)

        samples.append({
            't':                  round(time.time() - t0, 4),
            'leg_idx':            gi,
            'phase':              getattr(planner, '_trace_phase', None),
            'pos_n':              round(pos[0], 3),
            'pos_e':              round(pos[1], 3),
            'pos_d':              round(pos[2], 3),
            'vel_n':              round(vel[0], 3),
            'vel_e':              round(vel[1], 3),
            'vel_d':              round(vel[2], 3),
            'cmd_vn':             round(cmd_vx, 3),
            'cmd_ve':             round(cmd_vy, 3),
            'cmd_vd':             round(cmd_vz, 3),
            'cmd_spd':            round(cmd_spd, 3),
            'ach_spd':            round(ach_spd, 3),
            'err_spd':            round(cmd_spd - ach_spd, 3),
            'dxy_to_gate':        round(getattr(planner, '_trace_dxy', 0.0), 3),
            'dist_to_gate':       round(d, 3),
            'turn_angle_rad':     round(getattr(planner, '_trace_ta', 0.0), 4),
            'blend_radius':       round(getattr(planner, '_trace_db', 0.0), 3),
            'desired_pre_ceiling':round(getattr(planner, '_trace_desired', 0.0), 3),
            'cmd_pre_smooth':     round(getattr(planner, '_trace_cmd_pre_smooth', 0.0), 3),
            'yaw_cmd_deg':        round(cmd.yaw_deg, 2),
        })

        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total     = time.time() - t0
    completed = gi >= len(gates)

    try: await drone.offboard.stop()
    except: pass
    await drone.action.land(); await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    return {
        'course':           course_name,
        'completed':        completed,
        'total_time':       round(total, 3),
        'gates_passed':     gi,
        'n_samples':        len(samples),
        'gate_events':      gate_events,
        'samples':          samples,
        't_offboard_wait':  t_offboard_wait,
        'alt_at_offboard':  alt_at_offboard,
        'vz_at_offboard':   vz_at_offboard,
        'locked_tune':      dict(B.RACE_PARAMS),
        'planner':          {'max_speed': 12.0, 'cruise_speed': 9.0, 'base_blend': 1.5, 'px4_speed_ceiling': 9.5},
        'gates':            B.COURSES[course_name],
    }


def _write_outputs(result):
    course = result['course']
    os.makedirs(LOGS, exist_ok=True)
    jpath = os.path.join(LOGS, f'trace_{course}.json')
    cpath = os.path.join(LOGS, f'trace_{course}.csv')
    with open(jpath, 'w') as f:
        json.dump(result, f)
    if result['samples']:
        with open(cpath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(result['samples'][0].keys()))
            w.writeheader()
            for row in result['samples']:
                w.writerow(row)
    return jpath, cpath


def _print_summary(result):
    from collections import defaultdict
    course = result['course']
    print(f'\n=== {course.upper()} TRACE SUMMARY ===')
    st = 'OK' if result['completed'] else f'FAIL@{result["gates_passed"]}'
    print(f'{st}  total={result["total_time"]}s  n_samples={result["n_samples"]}  '
          f't_off={result["t_offboard_wait"]} alt@off={result["alt_at_offboard"]}')

    phase_stat = defaultdict(lambda: {'n': 0, 't_sum': 0, 'cmd': 0.0, 'ach': 0.0, 'err': 0.0})
    dt = 1/50
    for s in result['samples']:
        p = s['phase'] or 'UNKNOWN'
        phase_stat[p]['n']     += 1
        phase_stat[p]['t_sum'] += dt
        phase_stat[p]['cmd']   += s['cmd_spd']
        phase_stat[p]['ach']   += s['ach_spd']
        phase_stat[p]['err']   += s['err_spd']
    print('PHASE BREAKDOWN:')
    print(f'{"phase":<10} {"n":>5} {"dur":>6} {"frac":>6} {"cmd":>6} {"ach":>6} {"err":>6}')
    total_n = sum(v['n'] for v in phase_stat.values())
    for p, v in sorted(phase_stat.items(), key=lambda kv: -kv[1]['n']):
        n = v['n']; frac = n / max(total_n, 1); dur = v['t_sum']
        vcmd = v['cmd']/n; vach = v['ach']/n; verr = v['err']/n
        print(f'{p:<10} {n:>5} {dur:>5.2f}s {frac:>5.1%} {vcmd:>6.2f} {vach:>6.2f} {verr:>6.2f}')

    leg_stat = defaultdict(lambda: {'n': 0, 'cmd': 0.0, 'ach': 0.0, 'err': 0.0, 't_start': None, 't_end': None, 'ta_max': 0.0})
    for s in result['samples']:
        L = s['leg_idx']
        ls = leg_stat[L]
        ls['n']   += 1
        ls['cmd'] += s['cmd_spd']
        ls['ach'] += s['ach_spd']
        ls['err'] += s['err_spd']
        if ls['t_start'] is None: ls['t_start'] = s['t']
        ls['t_end'] = s['t']
        if s['turn_angle_rad'] > ls['ta_max']: ls['ta_max'] = s['turn_angle_rad']
    print('PER-LEG BREAKDOWN:')
    print(f'{"leg":>3} {"dur":>6} {"n":>4} {"cmd":>6} {"ach":>6} {"err":>6} {"util":>6} {"ta_max":>7}')
    for L in sorted(leg_stat.keys()):
        v = leg_stat[L]
        n = v['n']; dur = (v['t_end'] or 0) - (v['t_start'] or 0)
        cmd = v['cmd']/n; ach = v['ach']/n; err = v['err']/n
        util = ach/cmd if cmd > 0 else 0
        print(f'{L:>3} {dur:>5.2f}s {n:>4} {cmd:>6.2f} {ach:>6.2f} {err:>6.2f} {util:>5.1%} {v["ta_max"]:>6.2f}')


async def main():
    # Ensure race params lock
    for k, v in (('MPC_XY_VEL_P_ACC', 6.0), ('MPC_ACC_HOR', 10.0),
                 ('MPC_ACC_HOR_MAX', 10.0), ('MPC_JERK_AUTO', 30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)

    for course in ('technical', 'mixed'):
        print(f'\n[init] restart PX4 for {course}...', flush=True)
        os.system('pkill -9 -f mavsdk_server 2>/dev/null')
        os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
        os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
        time.sleep(1)
        B.restart_px4()
        await asyncio.sleep(POST_RESTART_WAIT)
        print(f'[trace] single lap on {course}...', flush=True)
        try:
            result = await asyncio.wait_for(run_trace(course), timeout=TRIAL_TIMEOUT)
        except asyncio.TimeoutError:
            print(f'[trace] TIMEOUT on {course}', flush=True)
            os.system('pkill -9 -f mavsdk_server 2>/dev/null')
            os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
            continue
        jpath, cpath = _write_outputs(result)
        print(f'[trace] wrote {jpath} / {cpath}', flush=True)
        _print_summary(result)


if __name__ == '__main__':
    asyncio.run(main())
