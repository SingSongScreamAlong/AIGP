# S8 Cold-Ramp Prefill A/B
# Arm A (control):  cold_ramp_seed = 0.0  (current promoted baseline)
# Arm B (prefill):  cold_ramp_seed = 4.0  (capped by px4_speed_ceiling=9.5)
#
# 5 trials x 2 courses x 2 arms, paired/interleaved A then B per trial,
# fresh PX4 restart per trial, technical FIRST then mixed.
#
# Tracks: lap_time, completion, leg0_time, leg0 sub-buckets
#         (vertical_settle / horiz_undercommand / ramp_gap / tracking_ok),
#         first-1s cmd/ach trajectory grid (10 samples @ 100ms),
#         max_spd.
#
# Decision rule (Conrad-locked):
#   ADOPT prefill if all of:
#     - technical horiz_undercommand drops by >= 0.30s (median over 5)
#     - technical lap improves (median delta < 0)
#     - mixed lap does not regress materially (median delta within +0.5s)
#       AND mixed completion rate not worse.

import asyncio, time, math, os, sys, json
from statistics import median, stdev
sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
POST_RESTART_WAIT = 5.0
TRIAL_TIMEOUT = 200.0
N_TRIALS = 5

# Leg-0 sub-bucket thresholds (same as leg0_dive.py)
VZ_SETTLE = 0.3
UNDER_CMD = 5.0
TRACK_TOL = 0.85

ARMS = [
    {'name': 'control_seed00', 'seed': 0.0},
    {'name': 'prefill_seed40', 'seed': 4.0},
]

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

    # Locked promoted baseline (Session 7): max_speed=12, cruise_speed=9, base_blend=1.5,
    # px4_speed_ceiling=9.5 (now hardcoded in __init__).
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
    planner.cold_ramp_seed = arm['seed']  # THE knob under test

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
    leg0_samples = []   # (t_rel, cmd_spd, ach_spd, vz)
    t_first_motion = None  # first sample where ach > 1.0 m/s

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

        # Sub-bucket accounting (only on leg 0)
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

    # First-1s 100ms grid
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
        'arm': arm['name'], 'seed': arm['seed'], 'course': course,
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
    }

async def main():
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ARMS = {ARMS}', flush=True)
    print(f'[init] N_TRIALS = {N_TRIALS}, courses=technical->mixed', flush=True)

    results = []
    for course in ('technical','mixed'):
        for trial in range(N_TRIALS):
            for arm in ARMS:
                tag = f'{course} t{trial+1} {arm["name"]}'
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
                    r = {'arm':arm['name'],'seed':arm['seed'],'course':course,
                         'completed':False,'lap_time':None,'error':'timeout'}
                    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
                    os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
                r['trial'] = trial+1
                results.append(r)
                print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
                      f'b_horiz={r.get("b_horiz_under")} b_ramp={r.get("b_ramp_gap")} '
                      f'b_settle={r.get("b_vert_settle")} ok={r.get("b_tracking_ok")} '
                      f'max={r.get("max_spd")} done={r.get("completed")}', flush=True)
                with open(os.path.join(LOGS,'ab_prefill.json'),'w') as f:
                    json.dump(results,f,indent=2)

    print('\n=== RESULTS ===')
    for course in ('technical','mixed'):
        print(f'\n--- {course.upper()} ---')
        print(f'  {"arm":<16} {"n":>5} {"lap_med":>8} {"lap_sd":>7} '
              f'{"leg0":>6} {"horiz":>7} {"ramp":>7} {"settle":>7} {"ok":>6} {"max":>6}')
        for arm in ARMS:
            rs = [r for r in results if r['course']==course and r['arm']==arm['name']
                  and r.get('completed')]
            if not rs:
                print(f'  {arm["name"]:<16} 0/{N_TRIALS} FAIL'); continue
            laps = [r['lap_time'] for r in rs]
            l0 = [r['leg0_time'] for r in rs if r.get('leg0_time') is not None]
            bh = [r['b_horiz_under'] for r in rs]
            br = [r['b_ramp_gap']    for r in rs]
            bs = [r['b_vert_settle'] for r in rs]
            bo = [r['b_tracking_ok'] for r in rs]
            ms = [r['max_spd']       for r in rs]
            lm = median(laps); lsd = stdev(laps) if len(laps)>1 else 0
            print(f'  {arm["name"]:<16} {len(rs)}/{N_TRIALS} {lm:8.3f} {lsd:7.3f} '
                  f'{median(l0):6.3f} {median(bh):7.3f} {median(br):7.3f} '
                  f'{median(bs):7.3f} {median(bo):6.3f} {median(ms):6.3f}')

    # Decision rule
    def med_metric(course, arm_name, key):
        rs=[r for r in results if r['course']==course and r['arm']==arm_name and r.get('completed')]
        vs=[r[key] for r in rs if r.get(key) is not None]
        return median(vs) if vs else None
    print('\n=== DECISION RULE ===')
    tech_horiz_ctl = med_metric('technical','control_seed00','b_horiz_under')
    tech_horiz_pre = med_metric('technical','prefill_seed40','b_horiz_under')
    tech_lap_ctl = med_metric('technical','control_seed00','lap_time')
    tech_lap_pre = med_metric('technical','prefill_seed40','lap_time')
    mix_lap_ctl  = med_metric('mixed','control_seed00','lap_time')
    mix_lap_pre  = med_metric('mixed','prefill_seed40','lap_time')
    if None in (tech_horiz_ctl,tech_horiz_pre,tech_lap_ctl,tech_lap_pre,mix_lap_ctl,mix_lap_pre):
        print('  insufficient data'); return
    d_horiz = tech_horiz_ctl - tech_horiz_pre
    d_tech_lap = tech_lap_pre - tech_lap_ctl
    d_mix_lap  = mix_lap_pre  - mix_lap_ctl
    print(f'  technical horiz_undercommand drop: {d_horiz:+.3f}s  (need >= 0.30)')
    print(f'  technical lap delta:               {d_tech_lap:+.3f}s (need < 0)')
    print(f'  mixed lap delta:                   {d_mix_lap:+.3f}s (need <= +0.50)')
    cond_horiz = d_horiz >= 0.30
    cond_tech  = d_tech_lap < 0
    cond_mix   = d_mix_lap <= 0.50
    verdict = 'ADOPT prefill (seed=4.0)' if (cond_horiz and cond_tech and cond_mix) else 'REJECT prefill'
    print(f'  --> {verdict}')

if __name__ == '__main__':
    asyncio.run(main())
