# S8 z_gate A/B
# Arm A (control)  : z_gate_alt_frac = 0.0   (gate disabled)
# Arm B (treatment): z_gate_alt_frac = 0.92  (vz_band=0.25, ramp_ms=200)
#
# Plan: 10 mixed pairs + 4 technical pairs (28 trials), interleaved A then B per trial,
#       fresh PX4 restart per trial, MIXED FIRST then TECHNICAL.
#       (Mixed is the target surface; technical is the regression guard.)
#
# Decision rule (Conrad-locked):
#   ADOPT z_gate iff:
#     - mixed completion: treatment >= control AND treatment >= 8/10
#     - mixed lap median: treatment - control <= 0.000
#     - technical lap regression: treatment - control <= +0.150
# Watch (NOT gating):
#     - mixed leg0 median delta should not exceed +0.200

import asyncio, time, math, os, sys, json
from statistics import median, stdev
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
POST_RESTART_WAIT = 5.0
TRIAL_TIMEOUT = 200.0
N_MIXED = 10
N_TECH  = 4

VZ_SETTLE = 0.3
UNDER_CMD = 5.0
TRACK_TOL = 0.85

ARMS = [
    {'name': 'control_zgate_off', 'frac': 0.0},
    {'name': 'treatment_zgate',   'frac': 0.92},
]
ZG_VZ_BAND = 0.25
ZG_RAMP_MS = 200

async def run_one(course, arm):
    gates = B.COURSES[course]
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break
    for n, v in B.RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass
    pos = [0,0,0]; vel = [0,0,0]
    async def pl():
        nonlocal pos, vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
            vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    # Locked promoted baseline (Session 7): max=12, cruise=9, blend=1.5, ceiling=9.5.
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
    planner.cold_ramp_seed = 0.0  # prefill rejected S8
    planner.z_gate_alt_frac = arm['frac']
    planner.z_gate_vz_band  = ZG_VZ_BAND
    planner.z_gate_ramp_ms  = ZG_RAMP_MS

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()
    ws = time.time()
    while True:
        if time.time()-ws >= 10.0: break
        if abs(pos[2]) >= 0.95*alt and abs(vel[2]) < 0.3: break
        await asyncio.sleep(0.05)
    t_offboard = time.time() - ws
    alt_off = abs(pos[2]); vz_off = vel[2]

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
    await drone.offboard.start()

    leg0_t = None
    max_ach = 0.0
    leg0_buckets = {'vertical_settle':0.0,'horiz_undercommand':0.0,
                    'ramp_gap':0.0,'tracking_ok':0.0}
    leg0_samples = []
    t_first_motion = None

    gi = 0; t0 = time.time(); dt = 1/50
    while gi < len(gates):
        if time.time()-t0 > 90: break
        g = gates[gi]
        d = math.sqrt((g[0]-pos[0])**2+(g[1]-pos[1])**2+(g[2]-pos[2])**2)
        if d < 2.5:
            if gi == 0 and leg0_t is None:
                leg0_t = time.time()-t0
            gspd = math.sqrt(vel[0]**2 + vel[1]**2)
            planner.on_gate_passed(gspd)
            gi += 1
            continue
        ng = gates[gi+1] if gi+1 < len(gates) else None
        cmd = planner.plan(pos, vel, g, ng)
        cmd_spd = math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2)
        ach_spd = math.sqrt(vel[0]**2 + vel[1]**2)
        vz_now = abs(vel[2])
        if ach_spd > max_ach: max_ach = ach_spd

        if gi == 0:
            t_rel = time.time() - t0
            leg0_samples.append((t_rel, cmd_spd, ach_spd, vz_now))
            if t_first_motion is None and ach_spd > 1.0:
                t_first_motion = t_rel
            if vz_now > VZ_SETTLE:
                b = 'vertical_settle'
            elif cmd_spd < UNDER_CMD:
                b = 'horiz_undercommand'
            elif ach_spd < cmd_spd * TRACK_TOL:
                b = 'ramp_gap'
            else:
                b = 'tracking_ok'
            leg0_buckets[b] += dt

        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total = time.time() - t0
    completed = gi >= len(gates)

    # snapshot z_gate state BEFORE teardown
    zg_open  = bool(planner.z_gate_open)
    zg_t     = (planner.z_gate_open_time - (t0)) if planner.z_gate_open_time else None
    zg_alt   = planner.z_gate_open_alt
    zg_vz    = planner.z_gate_open_vz

    try: await drone.offboard.stop()
    except: pass
    try: await drone.action.land()
    except: pass
    await asyncio.sleep(2)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(2)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    await asyncio.sleep(2)

    grid = []
    if leg0_samples:
        t_start = leg0_samples[0][0]
        for k in range(11):
            ts = t_start + k*0.1
            best = min(leg0_samples, key=lambda r: abs(r[0]-ts))
            grid.append({'t': round(best[0]-t_start,3),
                         'cmd': round(best[1],3),
                         'ach': round(best[2],3),
                         'vz':  round(best[3],3)})

    return {
        'arm': arm['name'], 'frac': arm['frac'], 'course': course,
        'completed': completed, 'lap_time': round(total,3),
        'leg0_time': round(leg0_t,3) if leg0_t is not None else None,
        't_offboard': round(t_offboard,3),
        'alt_off': round(alt_off,3), 'vz_off': round(vz_off,3),
        't_first_motion': round(t_first_motion,3) if t_first_motion else None,
        'b_vert_settle': round(leg0_buckets['vertical_settle'],3),
        'b_horiz_under': round(leg0_buckets['horiz_undercommand'],3),
        'b_ramp_gap':    round(leg0_buckets['ramp_gap'],3),
        'b_tracking_ok': round(leg0_buckets['tracking_ok'],3),
        'max_spd': round(max_ach,3),
        'first_1s_grid': grid,
        # z_gate-specific telemetry
        'zg_open':   zg_open,
        'zg_t_open': round(zg_t,3) if zg_t is not None else None,
        'zg_alt_open': round(zg_alt,3) if zg_alt is not None else None,
        'zg_vz_open':  round(zg_vz,3) if zg_vz is not None else None,
    }

async def trial(course, trial_idx, arm, results):
    tag = f'{course} t{trial_idx} {arm["name"]}'
    print(f'[{tag}] restart...', flush=True)
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('lsof -ti :14540 | xargs kill -9 2>/dev/null')
    os.system('lsof -ti :50051 | xargs kill -9 2>/dev/null')
    time.sleep(1)
    B.restart_px4()
    await asyncio.sleep(POST_RESTART_WAIT)
    try:
        r = await asyncio.wait_for(run_one(course, arm), timeout=TRIAL_TIMEOUT)
    except asyncio.TimeoutError:
        r = {'arm':arm['name'],'frac':arm['frac'],'course':course,
             'completed':False,'lap_time':None,'error':'timeout'}
        os.system('pkill -9 -f mavsdk_server 2>/dev/null')
        os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
    r['trial'] = trial_idx
    results.append(r)
    print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
          f'b_horiz={r.get("b_horiz_under")} b_ramp={r.get("b_ramp_gap")} '
          f'b_settle={r.get("b_vert_settle")} ok={r.get("b_tracking_ok")} '
          f'max={r.get("max_spd")} zg_open={r.get("zg_open")} '
          f'zg_t={r.get("zg_t_open")} done={r.get("completed")}', flush=True)
    with open(os.path.join(LOGS,'ab_zgate.json'),'w') as f:
        json.dump(results, f, indent=2)

async def main():
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ARMS = {ARMS}', flush=True)
    print(f'[init] z_gate defaults: alt_frac=0.92 vz_band={ZG_VZ_BAND} ramp_ms={ZG_RAMP_MS}', flush=True)
    print(f'[init] plan: {N_MIXED} mixed pairs + {N_TECH} technical pairs (mixed first)', flush=True)

    results = []
    # MIXED first (target surface, primary metric)
    for t in range(N_MIXED):
        for arm in ARMS:
            await trial('mixed', t+1, arm, results)
    # TECHNICAL second (regression guard)
    for t in range(N_TECH):
        for arm in ARMS:
            await trial('technical', t+1, arm, results)

    # === RESULTS ===
    print('\n=== RESULTS ===')
    for course in ('mixed','technical'):
        n_per_arm = N_MIXED if course=='mixed' else N_TECH
        print(f'\n--- {course.upper()} ---')
        print(f'  {"arm":<20} {"n":>5} {"lap_med":>9} {"lap_sd":>8} '
              f'{"leg0":>7} {"horiz":>7} {"ramp":>7} {"settle":>7} '
              f'{"ok":>6} {"max":>6} {"zg_t":>6}')
        for arm in ARMS:
            rs = [r for r in results if r['course']==course and r['arm']==arm['name']]
            done = [r for r in rs if r.get('completed')]
            if not done:
                print(f'  {arm["name"]:<20} 0/{n_per_arm} FAIL'); continue
            laps = [r['lap_time'] for r in done]
            l0 = [r['leg0_time'] for r in rs if r.get('leg0_time') is not None]
            bh = [r['b_horiz_under'] for r in rs]
            br = [r['b_ramp_gap']    for r in rs]
            bs = [r['b_vert_settle'] for r in rs]
            bo = [r['b_tracking_ok'] for r in rs]
            ms = [r['max_spd']       for r in rs]
            zt = [r['zg_t_open']     for r in rs if r.get('zg_t_open') is not None]
            lm = median(laps); lsd = stdev(laps) if len(laps)>1 else 0
            print(f'  {arm["name"]:<20} {len(done)}/{n_per_arm} {lm:9.3f} {lsd:8.3f} '
                  f'{median(l0) if l0 else 0:7.3f} {median(bh):7.3f} {median(br):7.3f} '
                  f'{median(bs):7.3f} {median(bo):6.3f} {median(ms):6.3f} '
                  f'{(median(zt) if zt else 0):6.3f}')

    # === DECISION RULE ===
    def laps(course, name): return [r['lap_time'] for r in results
                                    if r['course']==course and r['arm']==name and r.get('completed')]
    def medkey(course, name, key):
        vs = [r[key] for r in results if r['course']==course and r['arm']==name and r.get(key) is not None]
        return median(vs) if vs else None
    def n_done(course, name): return sum(1 for r in results
                                         if r['course']==course and r['arm']==name and r.get('completed'))

    mc = laps('mixed','control_zgate_off'); mt = laps('mixed','treatment_zgate')
    tc = laps('technical','control_zgate_off'); tt = laps('technical','treatment_zgate')
    print('\n=== DECISION RULE ===')
    if not (mc and mt and tc and tt):
        print('  insufficient data'); return
    n_mc = n_done('mixed','control_zgate_off'); n_mt = n_done('mixed','treatment_zgate')
    mix_ctl = median(mc); mix_tre = median(mt)
    tec_ctl = median(tc); tec_tre = median(tt)
    d_mix = mix_tre - mix_ctl
    d_tec = tec_tre - tec_ctl
    leg0_ctl = medkey('mixed','control_zgate_off','leg0_time')
    leg0_tre = medkey('mixed','treatment_zgate','leg0_time')
    d_leg0 = (leg0_tre - leg0_ctl) if (leg0_ctl is not None and leg0_tre is not None) else None
    print(f'  mixed completion : control {n_mc}/{N_MIXED}  treatment {n_mt}/{N_MIXED}')
    print(f'  mixed lap median : control {mix_ctl:.3f}  treatment {mix_tre:.3f}  delta {d_mix:+.3f} (need <= 0.000)')
    print(f'  tech  lap median : control {tec_ctl:.3f}  treatment {tec_tre:.3f}  delta {d_tec:+.3f} (need <= +0.150)')
    if d_leg0 is not None:
        warn = '' if d_leg0 <= 0.20 else '  [WARN > +0.20]'
        print(f'  [watch] mixed leg0 delta: {d_leg0:+.3f}  (NOT gating){warn}')
    rule_a = n_mt >= n_mc and n_mt >= 8
    rule_b = d_mix <= 0.0
    rule_c = d_tec <= 0.15
    print(f'\n  rule_a (mixed done >= ctl AND >= 8/10): {rule_a}')
    print(f'  rule_b (mixed lap delta <= 0)         : {rule_b}')
    print(f'  rule_c (tech regression <= +0.15)     : {rule_c}')
    verdict = 'ADOPT z_gate (alt_frac=0.92)' if (rule_a and rule_b and rule_c) else 'REJECT z_gate (alt_frac=0.92)'
    print(f'  --> {verdict}')

if __name__ == '__main__':
    asyncio.run(main())
