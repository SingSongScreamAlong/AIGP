"""
Mock-sim A/B: EXM1 vs V5.1, in the same domain the execution_model curves were fit from.

This harness spawns mock_sim.py + an in-process MAVSDK controller per trial, and runs
EXM1 (V5.1 + Curve 1 per-leg ceiling + Curve 2 leg0 launch profile) against V5.1.

Phase bucketing, launch delivery, sustain underdelivery, and consistency are computed
per-trial so we can see which loss buckets the empirical curves actually move.

Run:
  python3.13 mock_ab_exm1_vs_v51.py
"""
import asyncio, time, math, json, subprocess, os, sys, statistics
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

BASE_DIR     = "/Users/conradweeden/ai-grand-prix"
SIM_PATH     = os.path.join(BASE_DIR, "src/sim/mock_sim.py")
LOG_PATH     = os.path.join(BASE_DIR, "logs/mock_ab_exm1_vs_v51.json")
TRIALS_PER_COMBO = 3
MAVSDK_PORT  = 14540
PYTHON       = "/opt/homebrew/bin/python3.13"

# Courses identical to px4 harness (leg-length distribution matters for Curve 1/2).
COURSES = {
    'sprint': [
        (30,0,-3),(50,15,-3),(80,15,-3),(100,0,-3),(130,0,-3),
        (130,20,-3),(100,20,-3),(70,30,-3),(30,30,-3),(0,15,-3),
    ],
    'technical': [
        (8,0,-2.5),(12,6,-2.5),(8,12,-2.5),(0,12,-2.5),(-4,6,-2.5),
        (0,0,-2.5),(6,-4,-2.5),(14,-4,-2.5),(18,0,-2.5),(14,4,-2.5),
        (8,4,-2.5),(4,0,-3),
    ],
    'mixed': [
        (20,0,-3),(35,10,-3),(50,10,-3),(55,20,-3),(40,28,-3),
        (20,28,-3),(8,20,-3),(8,10,-3),(15,2,-3),(25,-2,-3),
        (35,2,-3),(20,5,-3),
    ],
}


# =====================================================================
# V5.1 Planner — ported from control_skeleton.py Planner class.
# Interface matches plan(state, target, next_gate) where state is the
# controller's DroneState with .pos and .vel attributes.
# =====================================================================
class V51Planner:
    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5):
        self.max_speed=max_speed; self.cruise_speed=cruise_speed; self.base_blend=base_blend
        self.blend_radius=base_blend; self.mode='velocity'
        self.px4_max_accel=6.0; self.px4_max_decel=4.0; self.px4_speed_ceiling=8.5
        self.px4_util_straight=0.78; self.px4_util_turn=0.55
        self.px4_hover_ramp_time=2.0
        self.last_cmd_speed=0.0; self.mission_start_time=None
        self.gates_passed=0; self.is_first_leg=True; self.prev_gate_speed=0.0
        self.last_phase='NONE'

    def _turn_entry_speed(self, ta):
        return min(self.cruise_speed*(0.4+0.6*math.cos(ta/2)), self.px4_speed_ceiling)

    def _decel_distance(self, cs, ta):
        if ta<0.3: return 0.0
        ts=self._turn_entry_speed(ta); sd=max(0,cs-ts)
        if sd<=0: return 0.0
        t=sd/self.px4_max_decel; d=cs*t-0.5*self.px4_max_decel*t*t
        return max(d*0.8, self.base_blend)

    def _phase(self, dxy, cs, ta, db):
        if self.is_first_leg and self.mission_start_time:
            if time.time()-self.mission_start_time<self.px4_hover_ramp_time: return 'LAUNCH'
        if dxy<=db: return 'TURN'
        dd=self._decel_distance(cs,ta)
        if dxy<=dd and ta>0.3: return 'PRE_TURN'
        if dxy<6.0 and ta>0.8: return 'SHORT'
        return 'SUSTAIN'

    def _px4_cmd(self, desired, ta_for_util):
        tr=ta_for_util/math.pi
        util=self.px4_util_straight*(1-tr)+self.px4_util_turn*tr
        return min(desired/max(util,0.3), self.max_speed)

    def _smooth(self, target, phase, dt=0.02):
        rates={'LAUNCH':6.0,'PRE_TURN':12.0,'TURN':6.0,'SHORT':10.0,'SUSTAIN':12.0}
        mr=rates.get(phase,8.0); md=mr*dt
        if target>self.last_cmd_speed: s=min(target,self.last_cmd_speed+md)
        else: s=max(target,self.last_cmd_speed-md*1.5)
        self.last_cmd_speed=s; return s

    def on_gate_passed(self, speed):
        self.gates_passed+=1; self.is_first_leg=False; self.prev_gate_speed=speed

    def plan(self, state, target, next_gate=None):
        if self.mission_start_time is None: self.mission_start_time=time.time()
        pos=state.pos; vel=state.vel
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dxy=math.sqrt(dx*dx+dy*dy); d3=math.sqrt(dx*dx+dy*dy+dz*dz)
        yaw=math.degrees(math.atan2(dy,dx))
        if d3<0.1:
            self.last_phase='STOP'
            return VelocityNedYaw(0,0,0,yaw)
        ux,uy,uz=dx/d3,dy/d3,dz/d3
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        ta=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny)
            if nd>0.1 and dxy>0.1:
                ax,ay=dx/dxy,dy/dxy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        db=self.base_blend+0.25*cs+(ta/math.pi)*2.0
        self.blend_radius=db
        phase=self._phase(dxy,cs,ta,db)
        self.last_phase=phase
        if phase=='LAUNCH':
            el=time.time()-self.mission_start_time
            ramp=min(1.0,el/self.px4_hover_ramp_time); ramp=ramp*ramp
            desired=max(self.cruise_speed*ramp,2.5)
        elif phase=='SHORT':
            te=self._turn_entry_speed(ta); desired=min(max(cs+2.0,te*0.8),te)
        elif phase=='SUSTAIN':
            desired=self.cruise_speed
        elif phase=='PRE_TURN':
            te=self._turn_entry_speed(ta); dd=self._decel_distance(cs,ta)
            if dd>0.1:
                prog=1.0-(dxy-db)/max(dd-db,0.1); prog=max(0,min(1,prog))
                desired=cs+(te-cs)*prog
            else: desired=te
        elif phase=='TURN':
            te=self._turn_entry_speed(ta)
            bl=max(0,min(1,1-(dxy/db)))
            ad=0.2+0.2*(ta/math.pi); af=1.0-ad*math.sin(bl*math.pi)
            desired=te*af
        else: desired=self.cruise_speed
        desired=min(desired, self._ceiling(state, target, dxy))
        cmd_spd=self._px4_cmd(desired, ta if phase in ('PRE_TURN','TURN','SHORT') else 0.0)
        sp=self._smooth(cmd_spd, phase)
        vx,vy,vz=ux*sp,uy*sp,uz*sp
        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s; vy*=s; vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)

    # Hook V5.1 uses scalar ceiling; EXM1 overrides this to use per-leg bins.
    def _ceiling(self, state, target, dxy):
        return self.px4_speed_ceiling


# =====================================================================
# EXM1 Planner — V5.1 + execution_model_v1 curves (Path 1 lite).
#   - Curve 1 per-leg-length speed ceiling (at v_cmd ~10.75 slice)
#   - Curve 2 leg0 empirical launch profile (gate_time + max_ach)
# =====================================================================
class EXM1Planner(V51Planner):
    _CEILING_BINS = [
        (0,   8,  7.21),
        (8,  12,  7.81),
        (12, 16,  8.34),
        (16, 20,  7.65),   # real dip in data
        (20, 24,  7.77),
        (24, 30,  9.06),
        (30, 999, 9.40),
    ]
    _LEG0_BINS = [
        #  lo   hi  gate_time  max_ach
        (  0,  16,    2.72,      6.14),
        ( 16,  28,    4.52,      8.26),
        ( 28, 999,    5.99,      8.27),
    ]

    def __init__(self, max_speed=11.0, cruise_speed=9.0, base_blend=1.5):
        super().__init__(max_speed=max_speed, cruise_speed=cruise_speed, base_blend=base_blend)
        self.leg0_length=None
        self.current_leg_length=None
        self._new_leg=True

    def _bin_ceiling(self, L):
        if L is None: return self.px4_speed_ceiling
        for lo,hi,v in self._CEILING_BINS:
            if lo<=L<hi: return v
        return self.px4_speed_ceiling

    def _leg0_params(self, L):
        if L is None: return (self.px4_hover_ramp_time, self.px4_speed_ceiling)
        for lo,hi,gt,ma in self._LEG0_BINS:
            if lo<=L<hi: return (gt, ma)
        return (self.px4_hover_ramp_time, self.px4_speed_ceiling)

    def _ceiling(self, state, target, dxy):
        return self._bin_ceiling(self.current_leg_length)

    def _phase(self, dxy, cs, ta, db):
        if self.is_first_leg and self.mission_start_time:
            gt,_ = self._leg0_params(self.leg0_length)
            if time.time()-self.mission_start_time<gt: return 'LAUNCH'
        if dxy<=db: return 'TURN'
        dd=self._decel_distance(cs,ta)
        if dxy<=dd and ta>0.3: return 'PRE_TURN'
        if dxy<6.0 and ta>0.8: return 'SHORT'
        return 'SUSTAIN'

    def on_gate_passed(self, speed):
        super().on_gate_passed(speed)
        self._new_leg=True

    def plan(self, state, target, next_gate=None):
        # Record leg length at the start of each new leg.
        if self._new_leg:
            dx=target[0]-state.pos[0]; dy=target[1]-state.pos[1]
            L=math.sqrt(dx*dx+dy*dy)
            self.current_leg_length=L
            if self.is_first_leg and self.leg0_length is None:
                self.leg0_length=L
            self._new_leg=False

        # Override LAUNCH speed to use empirical max_ach target instead of cruise.
        # Simplest way: temporarily rebind the LAUNCH branch by calling parent plan()
        # first and then patching the returned velocity if we're in LAUNCH.
        # But parent plan() reads cruise_speed in LAUNCH, so we intercept by
        # replicating just the LAUNCH branch here, else delegate.
        if self.mission_start_time is None: self.mission_start_time=time.time()
        pos=state.pos; vel=state.vel
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dxy=math.sqrt(dx*dx+dy*dy); d3=math.sqrt(dx*dx+dy*dy+dz*dz)
        if d3<0.1: return super().plan(state, target, next_gate)
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        ta=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny)
            if nd>0.1 and dxy>0.1:
                ax,ay=dx/dxy,dy/dxy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        db=self.base_blend+0.25*cs+(ta/math.pi)*2.0
        phase=self._phase(dxy,cs,ta,db)
        if phase=='LAUNCH':
            # Linear ramp to empirical max_ach for this leg0 length.
            gt, ma = self._leg0_params(self.leg0_length)
            el=time.time()-self.mission_start_time
            ramp=min(1.0, el/max(gt,0.1))
            desired=max(ma*ramp, 2.5)
            self.blend_radius=db
            self.last_phase=phase
            yaw=math.degrees(math.atan2(dy,dx))
            cmd_spd=self._px4_cmd(desired, 0.0)
            # Apply per-leg ceiling too (usually leg0 is bounded by max_ach not ceiling)
            cmd_spd=min(cmd_spd, self._bin_ceiling(self.current_leg_length))
            sp=self._smooth(cmd_spd, phase)
            ux,uy,uz=dx/d3,dy/d3,dz/d3
            vx,vy,vz=ux*sp,uy*sp,uz*sp
            vz=(target[2]-pos[2])*3.0
            st=math.sqrt(vx*vx+vy*vy+vz*vz)
            if st>self.max_speed: s=self.max_speed/st; vx*=s; vy*=s; vz*=s
            return VelocityNedYaw(vx,vy,vz,yaw)
        # Non-LAUNCH: delegate to parent (which calls self._ceiling, now bin-based).
        return super().plan(state, target, next_gate)


# =====================================================================
# DroneState stub — just enough for Planner.plan()
# =====================================================================
class DroneState:
    def __init__(self):
        self.pos=[0.0,0.0,0.0]; self.vel=[0.0,0.0,0.0]


# =====================================================================
# Mock sim lifecycle
# =====================================================================
def kill_stragglers():
    os.system("pkill -9 -f mavsdk_server 2>/dev/null")
    os.system("pkill -9 -f 'src/sim/mock_sim.py' 2>/dev/null")
    os.system(f"lsof -ti :{MAVSDK_PORT} | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(0.5)


def start_mock_sim():
    kill_stragglers()
    proc = subprocess.Popen(
        [PYTHON, "-u", SIM_PATH],
        stdout=open("/tmp/mock_sim_out.log","w"),
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        start_new_session=True,
    )
    time.sleep(1.5)  # let sim bind
    return proc


async def run_trial(planner, gates, threshold=2.5, timeout=90):
    drone = System()
    await drone.connect(system_address=f'udpin://0.0.0.0:{MAVSDK_PORT}')
    async for s in drone.core.connection_state():
        if s.is_connected: break

    state = DroneState()
    async def telem():
        async for pv in drone.telemetry.position_velocity_ned():
            state.pos=[pv.position.north_m, pv.position.east_m, pv.position.down_m]
            state.vel=[pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
    asyncio.ensure_future(telem())
    await asyncio.sleep(0.5)

    await drone.action.arm()
    alt = abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()
    await asyncio.sleep(3)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
    await drone.offboard.start()

    gi=0; gate_times=[]; t0=time.time(); dt=1/50
    max_spd=0.0
    phase_time={}
    cmds=[]; achs=[]; phases=[]
    # sample series for leg-level analysis
    leg_series=[]   # list of dicts per leg: {'cmds':[], 'achs':[], 'length':L}
    current_leg={'cmds':[], 'achs':[], 'length':None}

    while gi<len(gates):
        if time.time()-t0>timeout:
            print(f'      TIMEOUT@gate{gi+1}'); break
        g=gates[gi]
        d=math.sqrt((g[0]-state.pos[0])**2+(g[1]-state.pos[1])**2+(g[2]-state.pos[2])**2)
        if d<threshold:
            gate_times.append(time.time()-t0)
            gspd=math.sqrt(state.vel[0]**2+state.vel[1]**2)
            planner.on_gate_passed(gspd)
            leg_series.append(current_leg)
            current_leg={'cmds':[], 'achs':[], 'length':None}
            gi+=1; continue
        ng=gates[gi+1] if gi+1<len(gates) else None
        cmd=planner.plan(state, g, ng)
        cspd=math.sqrt(cmd.north_m_s**2+cmd.east_m_s**2)
        aspd=math.sqrt(state.vel[0]**2+state.vel[1]**2)
        cmds.append(cspd); achs.append(aspd)
        phases.append(getattr(planner,'last_phase','NONE'))
        if current_leg['length'] is None:
            # first sample of a leg: record its starting distance
            dx=g[0]-state.pos[0]; dy=g[1]-state.pos[1]
            current_leg['length']=math.sqrt(dx*dx+dy*dy)
        current_leg['cmds'].append(cspd); current_leg['achs'].append(aspd)
        if aspd>max_spd: max_spd=aspd
        phase_time[phases[-1]]=phase_time.get(phases[-1],0)+dt
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)

    total=time.time()-t0
    try: await drone.offboard.stop()
    except: pass
    try: await drone.action.land()
    except: pass
    await asyncio.sleep(1)
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(0.5)

    # Metrics
    completed = (gi>=len(gates))
    avg_cmd = sum(cmds)/len(cmds) if cmds else 0
    avg_ach = sum(achs)/len(achs) if achs else 0
    avg_err = sum(abs(c-a) for c,a in zip(cmds,achs))/len(cmds) if cmds else 0
    util = avg_ach/avg_cmd if avg_cmd>0 else 0
    splits = [round(gate_times[i]-(gate_times[i-1] if i>0 else 0),3) for i in range(len(gate_times))]

    # Per-phase underdelivery (avg_cmd - avg_ach during that phase)
    phase_under = {}
    phase_counts = {}
    for i,ph in enumerate(phases):
        phase_under[ph] = phase_under.get(ph,0) + max(0, cmds[i]-achs[i])
        phase_counts[ph] = phase_counts.get(ph,0)+1
    phase_underdelivery = {
        ph: round(phase_under[ph]/phase_counts[ph], 3) for ph in phase_under
    }

    # Launch loss: time from arm to first leg completion minus ideal
    launch_time = gate_times[0] if gate_times else None
    leg0_length = leg_series[0]['length'] if leg_series else None
    # ideal time = leg0_length / cruise_speed (no accel budget)
    ideal_launch = (leg0_length / planner.cruise_speed) if leg0_length else None
    launch_overhead = (launch_time - ideal_launch) if (launch_time and ideal_launch) else None

    # Exit rebuild: avg rate of aspd growth in the first 0.5s of each non-leg0 leg
    exit_rebuilds=[]
    for leg in leg_series[1:]:
        if len(leg['achs'])>=25:  # first 0.5s at 50Hz
            achs25=leg['achs'][:25]
            if achs25[-1]>achs25[0]:
                exit_rebuilds.append((achs25[-1]-achs25[0])/0.5)  # m/s per second
    exit_rebuild_mean = (sum(exit_rebuilds)/len(exit_rebuilds)) if exit_rebuilds else 0

    return {
        'completed': completed,
        'time': round(total,3),
        'gates_passed': gi,
        'max_spd': round(max_spd,2),
        'avg_cmd': round(avg_cmd,2),
        'avg_ach': round(avg_ach,2),
        'avg_err': round(avg_err,2),
        'util':    round(util,3),
        'splits':  splits,
        'phase_time': {k: round(v,2) for k,v in phase_time.items()},
        'phase_underdelivery': phase_underdelivery,
        'launch_time': round(launch_time,3) if launch_time else None,
        'launch_overhead': round(launch_overhead,3) if launch_overhead else None,
        'exit_rebuild_mps2': round(exit_rebuild_mean,3),
    }


async def main():
    kill_stragglers()
    results=[]
    configs=[
        ('V5.1', V51Planner,  dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)),
        ('EXM1', EXM1Planner, dict(max_speed=11.0, cruise_speed=9.0, base_blend=1.5)),
    ]
    for cn in ['mixed','sprint','technical']:
        gates=COURSES[cn]
        print(f'\n{"="*60}')
        print(f'COURSE: {cn.upper()} ({len(gates)} gates)')
        print(f'{"="*60}')
        for trial in range(TRIALS_PER_COMBO):
            for vname, Cls, kw in configs:
                print(f'  [trial {trial+1}/{TRIALS_PER_COMBO}] {vname}...', end=' ', flush=True)
                sim_proc = start_mock_sim()
                try:
                    p=Cls(**kw)
                    r=await run_trial(p, gates)
                except Exception as e:
                    r={'completed':False,'time':0,'gates_passed':0,'error':str(e),
                       'max_spd':0,'avg_cmd':0,'avg_ach':0,'avg_err':0,'util':0,
                       'splits':[],'phase_time':{},'phase_underdelivery':{},
                       'launch_time':None,'launch_overhead':None,'exit_rebuild_mps2':0}
                    print(f'ERR {e}', end=' ')
                finally:
                    try: sim_proc.terminate()
                    except: pass
                    kill_stragglers()
                    await asyncio.sleep(0.8)
                r['version']=vname; r['course']=cn; r['trial']=trial
                results.append(r)
                st='OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
                print(f'{st} {r["time"]}s | Max:{r["max_spd"]} Util:{r["util"]:.1%}')

    # ---- Summary ----
    print(f'\n\n{"="*80}')
    print('EXM1 vs V5.1 HEAD-TO-HEAD on MOCK SIM  (median over trials, OK only)')
    print(f'{"="*80}')
    print(f'{"Course":<10}{"Ver":<5}{"N":>3} {"Time p50":>9} {"MaxSpd":>7} {"AvgAch":>7} {"Util":>7} {"LchOvh":>8} {"ExitRb":>8}')
    print('-'*80)
    summary={}
    for cn in ['mixed','sprint','technical']:
        for vname,_,_ in configs:
            rs=[r for r in results if r['course']==cn and r['version']==vname and r['completed']]
            if not rs:
                print(f'{cn:<10}{vname:<5}{0:>3} no OK trials')
                continue
            rs_sorted=sorted(rs, key=lambda x:x['time'])
            med=rs_sorted[len(rs_sorted)//2]
            avg_ms=sum(r['max_spd'] for r in rs)/len(rs)
            avg_ach=sum(r['avg_ach'] for r in rs)/len(rs)
            avg_util=sum(r['util'] for r in rs)/len(rs)
            lcho=[r['launch_overhead'] for r in rs if r['launch_overhead'] is not None]
            exrb=[r['exit_rebuild_mps2'] for r in rs]
            avg_lcho=sum(lcho)/len(lcho) if lcho else 0
            avg_exrb=sum(exrb)/len(exrb) if exrb else 0
            summary[(cn,vname)]=dict(
                n=len(rs), time_p50=med['time'], max_spd=round(avg_ms,2),
                avg_ach=round(avg_ach,2), util=round(avg_util,3),
                launch_overhead=round(avg_lcho,3), exit_rebuild=round(avg_exrb,3),
                time_std=round(statistics.stdev([r['time'] for r in rs]),3) if len(rs)>1 else 0.0,
            )
            print(f'{cn:<10}{vname:<5}{len(rs):>3} {med["time"]:>8.2f}s {avg_ms:>6.2f} {avg_ach:>6.2f} {avg_util:>6.1%} {avg_lcho:>7.2f} {avg_exrb:>7.2f}')
        a=summary.get((cn,'V5.1')); b=summary.get((cn,'EXM1'))
        if a and b:
            d=a['time_p50']-b['time_p50']; pct=d/a['time_p50']*100
            marker='>>' if d>0 else '<<'
            print(f'  {marker} EXM1 delta vs V5.1: {-d:+.2f}s ({-pct:+.1f}%)   [negative = EXM1 faster]')
            # consistency
            print(f'     time std  V5.1={a["time_std"]:.2f}s  EXM1={b["time_std"]:.2f}s')
            # launch overhead delta
            if a['launch_overhead'] and b['launch_overhead']:
                print(f'     launch overhead  V5.1={a["launch_overhead"]:.2f}s  EXM1={b["launch_overhead"]:.2f}s')
        print()

    out={
        'trials': results,
        'summary': {f'{k[0]}_{k[1]}': v for k,v in summary.items()},
    }
    with open(LOG_PATH,'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {LOG_PATH}')


if __name__=='__main__':
    asyncio.run(main())
