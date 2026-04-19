"""PX4 Benchmark - Mixed course only (fixed hairpin)"""
import asyncio
import time
import math
import json
import subprocess
import os

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

# Fixed mixed course - wider hairpin, more achievable geometry
GATES = [
    (20.0,   0.0, -3.0),  # 1: 20m opening straight
    (35.0,  10.0, -3.0),  # 2: sweeping right
    (50.0,  10.0, -3.0),  # 3: 15m straight
    (55.0,  20.0, -3.0),  # 4: right turn
    (40.0,  28.0, -3.0),  # 5: connecting
    (20.0,  28.0, -3.0),  # 6: 20m straight
    (8.0,   20.0, -3.0),  # 7: sweeping left
    (8.0,   10.0, -3.0),  # 8: straight down (was hairpin, now wider)
    (15.0,   2.0, -3.0),  # 9: exit turn
    (25.0,  -2.0, -3.0),  # 10: cross-track
    (35.0,   2.0, -3.0),  # 11: back to finish area
    (20.0,   5.0, -3.0),  # 12: finish
]

RACE_PARAMS = {
    'MPC_ACC_HOR': 10.0, 'MPC_ACC_HOR_MAX': 10.0,
    'MPC_JERK_AUTO': 30.0, 'MPC_JERK_MAX': 50.0,
    'MPC_TILTMAX_AIR': 70.0, 'MPC_XY_VEL_MAX': 15.0,
    'MPC_Z_VEL_MAX_UP': 5.0,
}

PROFILES = [
    ('aggressive', {'max_speed': 12.0, 'cruise_speed': 10.0, 'base_blend': 2.5, 'threshold': 2.5}),
    ('balanced',   {'max_speed': 11.0, 'cruise_speed': 9.0,  'base_blend': 1.8, 'threshold': 2.5}),
    ('safe',       {'max_speed': 10.0, 'cruise_speed': 8.0,  'base_blend': 1.2, 'threshold': 2.0}),
]

class V3Planner:
    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5):
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.base_blend = base_blend

    def plan_velocity(self, pos, vel, target, next_gate=None):
        dx, dy, dz = target[0]-pos[0], target[1]-pos[1], target[2]-pos[2]
        dist_xy = math.sqrt(dx*dx + dy*dy)
        if dist_xy < 0.1:
            return VelocityNedYaw(0, 0, 0, 0)
        ux, uy = dx/dist_xy, dy/dist_xy
        turn_angle = 0.0
        if next_gate:
            ax, ay = target[0]-pos[0], target[1]-pos[1]
            bx, by = next_gate[0]-target[0], next_gate[1]-target[1]
            am, bm = math.sqrt(ax**2+ay**2), math.sqrt(bx**2+by**2)
            if am > 0.01 and bm > 0.01:
                d = max(-1, min(1, (ax*bx+ay*by)/(am*bm)))
                turn_angle = math.acos(d)
        cs = math.sqrt(vel[0]**2+vel[1]**2)
        db = self.base_blend + 0.2*cs + (turn_angle/math.pi)*1.5
        ac = self.cruise_speed*(0.4+0.6*math.cos(turn_angle/2))
        if dist_xy < db:
            b = 1-(dist_xy/db)
            speed = ac*(1-0.3*math.sin(b*math.pi))
        elif dist_xy < db*1.5:
            r = (dist_xy-db)/(db*0.5)
            speed = ac+(self.cruise_speed-ac)*r
        else:
            speed = self.cruise_speed
        speed = min(speed, self.max_speed)
        vz = (target[2]-pos[2])*3.0
        yaw = math.degrees(math.atan2(dy, dx))
        return VelocityNedYaw(ux*speed, uy*speed, vz, yaw)

class Logger:
    def __init__(self):
        self.entries = []
        self.gate_events = []
    def log(self, t, cmd, vel, pos, gate_idx):
        cs = math.sqrt(cmd.north_m_s**2+cmd.east_m_s**2)
        avs = math.sqrt(vel[0]**2+vel[1]**2)
        ve = math.sqrt((cmd.north_m_s-vel[0])**2+(cmd.east_m_s-vel[1])**2)
        acc = 0
        if self.entries:
            p = self.entries[-1]
            dt = t-p['t']
            if dt > 0.001:
                acc = math.sqrt((vel[0]-p['av'][0])**2+(vel[1]-p['av'][1])**2)/dt
        self.entries.append({'t':t,'cs':cs,'as':avs,'ve':ve,'acc':acc,'av':vel[:],'gi':gate_idx})
    def log_gate(self, n, t, pos, vel):
        self.gate_events.append({'gate':n,'t':t,'speed':math.sqrt(vel[0]**2+vel[1]**2)})
    def analyze(self):
        if not self.entries: return {}
        ve = [e['ve'] for e in self.entries]
        cs = [e['cs'] for e in self.entries]
        avs = [e['as'] for e in self.entries]
        acc = [e['acc'] for e in self.entries[1:]]
        legs = []
        for i,ge in enumerate(self.gate_events):
            ts = self.gate_events[i-1]['t'] if i>0 else self.entries[0]['t']
            le = [e for e in self.entries if ts<=e['t']<=ge['t']]
            if le:
                lc = [e['cs'] for e in le]
                la = [e['as'] for e in le]
                lv = [e['ve'] for e in le]
                lac = [e['acc'] for e in le[1:]] or [0]
                legs.append({
                    'gate':ge['gate'],'dur':round(ge['t']-ts,3),
                    'entry_spd':round(ge['speed'],2),
                    'avg_cmd':round(sum(lc)/len(lc),2),'avg_ach':round(sum(la)/len(la),2),
                    'max_ach':round(max(la),2),'avg_err':round(sum(lv)/len(lv),3),
                    'pk_acc':round(max(lac),2),
                    'util':round(sum(la)/max(sum(lc),0.01),3),
                })
        return {
            'overall': {
                'avg_err':round(sum(ve)/len(ve),3),'max_err':round(max(ve),3),
                'avg_cmd':round(sum(cs)/len(cs),2),'avg_ach':round(sum(avs)/len(avs),2),
                'max_ach':round(max(avs),2),'pk_acc':round(max(acc) if acc else 0,2),
                'util':round(sum(avs)/max(sum(cs),0.01),3),
            },
            'legs': legs,
        }

def restart_px4():
    os.system('pkill -f mavsdk_server 2>/dev/null')
    os.system('pkill -9 -f "bin/px4" 2>/dev/null')
    time.sleep(3)
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters.bson 2>/dev/null')
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters_backup.bson 2>/dev/null')
    subprocess.Popen(['/bin/bash','/tmp/run_px4_sih.sh'],stdout=open('/tmp/px4_sih_out.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(7)

async def run_trial(prof_name, params):
    drone = System()
    await drone.connect(system_address='udpin://0.0.0.0:14540')
    async for s in drone.core.connection_state():
        if s.is_connected: break
    for n,v in RACE_PARAMS.items():
        try: await drone.param.set_param_float(n,v)
        except: pass
    pos=[0,0,0]; vel=[0,0,0]
    async def pl():
        nonlocal pos,vel
        async for pv in drone.telemetry.position_velocity_ned():
            pos=[pv.position.north_m,pv.position.east_m,pv.position.down_m]
            vel=[pv.velocity.north_m_s,pv.velocity.east_m_s,pv.velocity.down_m_s]
    asyncio.ensure_future(pl())
    await asyncio.sleep(0.5)
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(3.0)
    await drone.action.takeoff()
    await asyncio.sleep(4)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
    await drone.offboard.start()
    planner=V3Planner(params['max_speed'],params['cruise_speed'],params['base_blend'])
    logger=Logger()
    gi=0; t0=time.time(); dt=1/50
    while gi<len(GATES):
        if time.time()-t0>90:
            print(f'    TIMEOUT at gate {gi+1}/{len(GATES)}')
            break
        g=GATES[gi]
        d=math.sqrt((g[0]-pos[0])**2+(g[1]-pos[1])**2+(g[2]-pos[2])**2)
        if d<params['threshold']:
            logger.log_gate(gi+1,time.time()-t0,pos,vel)
            gi+=1; continue
        ng=GATES[gi+1] if gi+1<len(GATES) else None
        cmd=planner.plan_velocity(pos,vel,g,ng)
        logger.log(time.time()-t0,cmd,vel,pos,gi)
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)
    total=time.time()-t0
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land()
    await asyncio.sleep(2)
    a=logger.analyze()
    return {
        'profile':prof_name,'completed':gi>=len(GATES),'gates':gi,'total':len(GATES),
        'time':round(total,3),
        'gate_times':[g['t'] for g in logger.gate_events],
        'splits':[round(logger.gate_events[i]['t']-(logger.gate_events[i-1]['t'] if i>0 else 0),3) for i in range(len(logger.gate_events))],
        'tracking':a,'params':params,
    }

async def main():
    results=[]
    for pn,pp in PROFILES:
        print(f'\n--- {pn} ---')
        r=await run_trial(pn,pp)
        results.append(r)
        o=r['tracking'].get('overall',{})
        st='OK' if r['completed'] else f'FAIL@{r["gates"]}'
        print(f'  {st} | {r["time"]}s | MaxSpd:{o.get("max_ach",0)} | AvgErr:{o.get("avg_err",0)} | Util:{o.get("util",0):.1%}')
        print(f'  Splits: {r["splits"]}')
        if pn != PROFILES[-1][0]:
            restart_px4()
            await asyncio.sleep(2)

    print('\nMIXED COURSE SUMMARY:')
    print(f'{"Profile":<15} {"St":<6} {"Time":>7} {"MaxSpd":>7} {"AvgErr":>7} {"Util":>7}')
    for r in results:
        o=r['tracking'].get('overall',{})
        st='OK' if r['completed'] else 'FAIL'
        print(f'{r["profile"]:<15} {st:<6} {r["time"]:>6.2f}s {o.get("max_ach",0):>5.1f} {o.get("avg_err",0):>6.3f} {o.get("util",0):>6.1%}')

    # Leg detail for aggressive
    if results[0]['tracking'].get('legs'):
        print(f'\nAGGRESSIVE LEG DETAIL:')
        print(f'{"Leg":<5} {"Dur":>6} {"EntSpd":>7} {"AvgCmd":>7} {"AvgAch":>7} {"MaxAch":>7} {"Err":>7} {"PkAcc":>7} {"Util":>6}')
        for l in results[0]['tracking']['legs']:
            print(f'{l["gate"]:>3}   {l["dur"]:>5.2f}s {l["entry_spd"]:>5.1f} {l["avg_cmd"]:>6.1f} {l["avg_ach"]:>6.1f} {l["max_ach"]:>6.1f} {l["avg_err"]:>6.3f} {l["pk_acc"]:>6.1f} {l["util"]:>5.1%}')

    with open('/Users/conradweeden/ai-grand-prix/logs/px4_bench_mixed.json','w') as f:
        json.dump(results,f,indent=2)
    print(f'\nSaved to logs/px4_bench_mixed.json')

asyncio.run(main())
