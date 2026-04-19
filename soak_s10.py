# soak_s10.py — Session 10 baseline soak.
# 20-trial run on the new default (V5.1 + ceil95_max12 + transition_blend).
# tb_enabled=True (adopted Session 9). Purpose: establish new noise-floor baseline.
import asyncio, os, sys, time, math, json
from statistics import median, stdev

sys.path.insert(0, '/Users/conradweeden/ai-grand-prix')
import px4_v51_baseline as B
import bench

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

LOGS = '/Users/conradweeden/ai-grand-prix/logs'
RESULT = os.path.join(LOGS, 'soak_s10.json')
N_SOAK = 20
COURSE = 'mixed'
TRIAL_TIMEOUT = 200.0
POST_RESTART_WAIT = 2.0

async def run_one(course):
    """Run a single trial on the new default config (tb_enabled=True)."""
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')

    diag = await bench.wait_healthy(drone, timeout=15.0)
    if not all([diag['connected'], diag['telemetry'], diag['armable']]):
        return {'completed': False, 'error': 'unhealthy', 'health': diag}

    for n, v in B.RACE_PARAMS.items():
        try: await drone.param.set_param_float(n, v)
        except: pass

    pos = [0,0,0]; vel = [0,0,0]
    stop_pl = asyncio.Event()
    async def pl():
        nonlocal pos, vel
        try:
            async for pv in drone.telemetry.position_velocity_ned():
                pos = [pv.position.north_m, pv.position.east_m, pv.position.down_m]
                vel = [pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
                if stop_pl.is_set(): return
        except asyncio.CancelledError:
            return
    pl_task = asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)

    gates = B.COURSES[course]
    planner = B.V51Planner(max_speed=12.0, cruise_speed=9.0, base_blend=1.5)
    # New default: transition_blend ON (adopted Session 9)
    planner.cold_ramp_seed = 0.0
    planner.z_gate_alt_frac = 0.0
    planner.tb_enabled = True
    planner.tb_vz_thresh = 1.0

    try:
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

        await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
        await drone.offboard.start()

        leg0_t = None
        max_ach = 0.0
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
            ach = math.sqrt(vel[0]**2 + vel[1]**2)
            if ach > max_ach: max_ach = ach
            await drone.offboard.set_velocity_ned(cmd)
            await asyncio.sleep(dt)
        total = time.time() - t0
        completed = gi >= len(gates)

        try: await drone.offboard.stop()
        except: pass
        try: await drone.action.land()
        except: pass

        return {
            'completed': completed,
            'lap_time': round(total, 3),
            'leg0_time': round(leg0_t, 3) if leg0_t is not None else None,
            't_offboard': round(t_offboard, 3),
            'max_spd': round(max_ach, 3),
            'health': diag,
        }
    finally:
        stop_pl.set()
        pl_task.cancel()
        try: await pl_task
        except (asyncio.CancelledError, Exception): pass

async def trial(idx, results):
    tag = f'soak t{idx}/{N_SOAK}'
    print(f'[{tag}] hardened restart...', flush=True)
    restart_ok = False
    for attempt in range(2):
        try:
            bench.hardened_restart(B)
            restart_ok = True
            break
        except Exception as e:
            if attempt == 0:
                print(f'  -> RESTART FAILED (attempt 1), retrying: {e}', flush=True)
                bench.kill_stack()
                await asyncio.sleep(3.0)
            else:
                r = {'trial': idx, 'completed': False,
                     'error': f'restart_failed_2x: {e}', 'restart_retried': True}
                results.append(r)
                bench.atomic_write_json(RESULT, results)
                print(f'  -> RESTART FAILED (attempt 2): {e}', flush=True)
                return
    await asyncio.sleep(POST_RESTART_WAIT)

    try:
        r = await asyncio.wait_for(run_one(COURSE), timeout=TRIAL_TIMEOUT)
    except asyncio.TimeoutError:
        r = {'completed': False, 'error': 'timeout'}
        bench.kill_stack()
    except Exception as e:
        r = {'completed': False, 'error': f'exception: {type(e).__name__}: {e}'}
        bench.kill_stack()

    r['trial'] = idx
    results.append(r)
    bench.atomic_write_json(RESULT, results)
    print(f'  -> lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
          f'done={r.get("completed")} err={r.get("error")}', flush=True)

async def main():
    bench.acquire_singleton('soak')
    for k, v in (('MPC_XY_VEL_P_ACC',6.0),('MPC_ACC_HOR',10.0),
                 ('MPC_ACC_HOR_MAX',10.0),('MPC_JERK_AUTO',30.0)):
        B.RACE_PARAMS[k] = v
    print(f'[init] Session 10 soak — new default (tb_enabled=True)', flush=True)
    print(f'[init] locked tune -> {B.RACE_PARAMS}', flush=True)
    print(f'[init] N_SOAK={N_SOAK}, COURSE={COURSE}', flush=True)
    print(f'[init] result file: {RESULT}', flush=True)

    results = []
    bench.atomic_write_json(RESULT, results)

    # PREFLIGHT: 1 trial sanity check
    print('[preflight] 1 trial sanity check', flush=True)
    await trial(0, results)
    pre = results[0]
    if not pre.get('completed'):
        print(f'[preflight] FAILED: {pre.get("error")}; aborting soak', flush=True)
        return
    print(f'[preflight] OK lap={pre.get("lap_time")}', flush=True)

    # SOAK: 20 trials
    for t in range(1, N_SOAK+1):
        await trial(t, results)

    # === REPORT ===
    print('\n=== SESSION 10 SOAK RESULTS ===', flush=True)
    print(f'Config: V5.1 + ceil95_max12 + transition_blend (tb_enabled=True, vz_thresh=1.0)', flush=True)
    soak = [r for r in results if r['trial'] >= 1]
    done = [r for r in soak if r.get('completed')]
    laps = [r['lap_time'] for r in done if r.get('lap_time') is not None]
    leg0s = [r['leg0_time'] for r in done if r.get('leg0_time') is not None]
    print(f'  trials run         : {len(soak)}/{N_SOAK}')
    print(f'  completed          : {len(done)}/{len(soak)}')
    if laps:
        print(f'  lap median         : {median(laps):.3f}s')
        print(f'  lap stdev          : {stdev(laps) if len(laps)>1 else 0:.3f}s')
        print(f'  lap min/max        : {min(laps):.3f} / {max(laps):.3f}')
    if leg0s:
        print(f'  leg0 median        : {median(leg0s):.3f}s')
        print(f'  leg0 stdev         : {stdev(leg0s) if len(leg0s)>1 else 0:.3f}s')
    errors = [r.get('error') for r in soak if r.get('error')]
    if errors:
        print(f'  errors             : {errors}')

    # Comparison to S9 pre-transition_blend baseline
    S9_LAP = 22.879
    S9_STD = 0.101
    if laps:
        delta = median(laps) - S9_LAP
        print(f'\n  vs S9 baseline     : {delta:+.3f}s (S9 was {S9_LAP:.3f} +/- {S9_STD:.3f})')

    print('\n  per-trial:')
    for r in soak:
        print(f'    t{r["trial"]:>2} lap={r.get("lap_time")} leg0={r.get("leg0_time")} '
              f'done={r.get("completed")} err={r.get("error")}')

if __name__ == '__main__':
    asyncio.run(main())
