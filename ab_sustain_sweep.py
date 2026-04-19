# SUSTAIN achievability sweep -- JOINT px4_speed_ceiling + max_speed pivot.
#   arm1: px4_speed_ceiling=8.5,  max_speed=11.0  (current baseline)
#   arm2: px4_speed_ceiling=9.5,  max_speed=12.0
#   arm3: px4_speed_ceiling=10.5, max_speed=13.0
# 5 trials each, paired/interleaved, technical FIRST then mixed.
# Tracks lap time, leg0 time, sustain cmd/ach/gap/util, exit leg util,
# max_spd (peak achieved ground speed), completion.
# Pivot rationale: cruise_speed is dead in SUSTAIN because
#   desired=cruise_speed -> clamped to px4_speed_ceiling=8.5 -> _px4_cmd inflates by util.
# The real levers for SUSTAIN cmd are the ceiling and the max_speed cap.
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

_orig_phase = B.V51Planner._phase
def _phase_trace(self, dxy, cs, ta, db):
    ph = _orig_phase(self, dxy, cs, ta, db)
    self._trace_phase = ph
    return ph
B.V51Planner._phase = _phase_trace

ARMS = [
    {'name': 'ceil85_max11',  'ceiling':  8.5, 'max_speed': 11.0},
    {'name': 'ceil95_max12',  'ceiling':  9.5, 'max_speed': 12.0},
    {'name': 'ceil105_max13', 'ceiling': 10.5, 'max_speed': 13.0},
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

    # cruise_speed=9.0 kept fixed; the effective levers are ceiling + max_speed.
    planner = B.V51Planner(max_speed=arm['max_speed'],
                            cruise_speed=9.0,
                            base_blend=1.5)
    # POST-construction override: ceiling is hardcoded in base __init__.
    planner.px4_speed_ceiling = arm['ceiling']

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()

    ws = time.time()
    while True:
        if time.time()-ws >= 10.0: break
        if abs(pos[2]) >= 0.95*alt and abs(vel[2]) < 0.3: break
        await asyncio.sleep(0.05)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
    await drone.offboard.start()

    sustain_cmd_sum = 0.0; sustain_ach_sum = 0.0; sustain_n = 0
    last_leg_cmd = 0.0;    last_leg_ach = 0.0;    last_leg_n = 0
    leg0_t = None
    max_ach = 0.0
    gi = 0; t0 = time.time(); dt = 1/50
    last_leg_idx = len(gates) - 1

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
        if ach_spd > max_ach:
            max_ach = ach_spd
        ph = getattr(planner, '_trace_phase', None)
        if ph == 'SUSTAIN' and gi >= 2 and gi != last_leg_idx:
            sustain_cmd_sum += cmd_spd
            sustain_ach_sum += ach_spd
            sustain_n += 1
        if gi == last_leg_idx:
            last_leg_cmd += cmd_spd
            last_leg_ach += ach_spd
            last_leg_n += 1
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

    sus_cmd  = sustain_cmd_sum/max(sustain_n,1)
    sus_ach  = sustain_ach_sum/max(sustain_n,1)
    sus_util = sus_ach/sus_cmd if sus_cmd>0 else 0
    el_util  = last_leg_ach/last_leg_cmd if last_leg_cmd>0 else 0
    return {
        'arm': arm['name'], 'course': course,
        'ceiling': arm['ceiling'], 'max_speed': arm['max_speed'],
        'completed': completed, 'lap_time': round(total,3),
        'leg0_time': round(leg0_t,3) if leg0_t is not None else None,
        'sustain_n': sustain_n,
        'sustain_cmd': round(sus_cmd,3), 'sustain_ach': round(sus_ach,3),
        'sustain_gap': round(sus_cmd-sus_ach,3), 'sustain_util': round(sus_util,4),
        'exit_leg_util': round(el_util,4), 'exit_leg_n': last_leg_n,
        'max_spd': round(max_ach,3),
    }


async def main():
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] ARMS = {ARMS}', flush=True)

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
                    r = {'arm':arm['name'],'course':course,'completed':False,
                         'lap_time':None,'error':'timeout',
                         'ceiling':arm['ceiling'],'max_speed':arm['max_speed']}
                    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
                    os.system("pkill -9 -f 'bin/px4' 2>/dev/null")
                r['trial'] = trial+1
                results.append(r)
                print(f'  -> {r}', flush=True)
                with open(os.path.join(LOGS,'ab_sustain_sweep.json'),'w') as f:
                    json.dump(results,f,indent=2)

    print('\n=== RESULTS ===')
    for course in ('technical','mixed'):
        print(f'\n--- {course.upper()} ---')
        print(f'  {"arm":<14} {"n":>5} {"lap_med":>8} {"lap_sd":>7} '
              f'{"sus_cmd":>7} {"sus_ach":>7} {"sus_gap":>7} {"sus_u":>6} '
              f'{"exit_u":>6} {"max_sp":>6} {"leg0":>6}')
        for arm in ARMS:
            rs = [r for r in results if r['course']==course and r['arm']==arm['name']
                  and r.get('completed')]
            if not rs:
                print(f'  {arm["name"]:<14} 0/{N_TRIALS} FAIL'); continue
            laps = [r['lap_time']  for r in rs]
            sc   = [r['sustain_cmd'] for r in rs]
            sa   = [r['sustain_ach'] for r in rs]
            sg   = [r['sustain_gap'] for r in rs]
            su   = [r['sustain_util'] for r in rs]
            eu   = [r['exit_leg_util'] for r in rs]
            ms   = [r['max_spd'] for r in rs]
            l0   = [r['leg0_time'] for r in rs if r.get('leg0_time') is not None]
            lm   = median(laps); lsd = stdev(laps) if len(laps)>1 else 0
            print(f'  {arm["name"]:<14} {len(rs)}/{N_TRIALS} {lm:8.3f} {lsd:7.3f} '
                  f'{median(sc):7.3f} {median(sa):7.3f} {median(sg):7.3f} '
                  f'{median(su):6.1%} {median(eu):6.1%} '
                  f'{median(ms):6.3f} {median(l0):6.3f}')

if __name__ == '__main__':
    asyncio.run(main())
