"""
EXM1 vs V5.1 Head-to-Head on PX4 SITL

EXM1 = V5.1 + empirical response curves from execution_model_v1 (Path 1 lite):
  - per-leg-length speed ceiling lookup (Curve 1 at cruise ~10.75)
  - empirical leg0 launch profile (Curve 2, n=289, R^2>=0.95)
  - leg0 max_ach cap (Curve 2)

Changes NOT included (deferred to v_cmd sweep + preturn instrumentation):
  - v_achieved(v_cmd, .) as function of v_cmd (Curve 1 v_cmd axis was sparse)
  - decel envelope (Curve 3 had no preturn_onset signal in source jsonl)

Baseline V5.1 is copied verbatim from px4_v51_vs_v4.py so A/B is apples-to-apples.
Runs N trials per (planner, course) to average out run-to-run noise.
"""
import asyncio, time, math, json, subprocess, os
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

TRIALS_PER_COMBO = 3  # bump if you want tighter confidence intervals

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
RACE_PARAMS = {
    'MPC_ACC_HOR':10.0,'MPC_ACC_HOR_MAX':10.0,'MPC_JERK_AUTO':30.0,
    'MPC_JERK_MAX':50.0,'MPC_TILTMAX_AIR':70.0,'MPC_XY_VEL_MAX':15.0,
    'MPC_Z_VEL_MAX_UP':5.0,
}


# =====================================================================
# V51 Planner - baseline, verbatim from px4_v51_vs_v4.py
# =====================================================================
class V51Planner:
    def __init__(self, max_speed=12.0, cruise_speed=10.0, base_blend=2.5):
        self.max_speed=max_speed; self.cruise_speed=cruise_speed; self.base_blend=base_blend
        self.blend_radius=base_blend; self.mode='velocity'
        self.px4_max_accel=6.0; self.px4_max_decel=4.0; self.px4_speed_ceiling=8.5
        self.px4_util_straight=0.78; self.px4_util_turn=0.55
        self.px4_hover_ramp_time=2.0
        self.last_cmd_speed=0.0; self.mission_start_time=None
        self.gates_passed=0; self.is_first_leg=True; self.prev_gate_speed=0.0
    def _turn_entry_speed(self, ta):
        return min(self.cruise_speed*(0.4+0.6*math.cos(ta/2)), self.px4_speed_ceiling)
    def _decel_distance(self, cs, ta):
        if ta<0.2: return 0.0
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
    def plan(self, pos, vel, target, next_gate=None):
        if self.mission_start_time is None: self.mission_start_time=time.time()
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dxy=math.sqrt(dx*dx+dy*dy); d3=math.sqrt(dx*dx+dy*dy+dz*dz)
        yaw=math.degrees(math.atan2(dy,dx))
        if d3<0.1: return VelocityNedYaw(0,0,0,yaw)
        ux,uy,uz=dx/d3,dy/d3,dz/d3
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        ta=0.0; nsl=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny); nsl=nd
            if nd>0.1 and dxy>0.1:
                ax,ay=dx/dxy,dy/dxy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        db=self.base_blend+0.25*cs+(ta/math.pi)*2.0
        self.blend_radius=db
        phase=self._phase(dxy,cs,ta,db)
        if phase=='LAUNCH':
            el=time.time()-self.mission_start_time
            ramp=min(1.0,el/self.px4_hover_ramp_time); ramp=ramp*ramp; desired=max(self.cruise_speed*ramp,2.5)
        elif phase=='SHORT':
            te=self._turn_entry_speed(ta); desired=min(max(cs+2.0,te*0.8),te)
        elif phase=='SUSTAIN': desired=self.cruise_speed
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
        desired=min(desired,self.px4_speed_ceiling)
        cmd_spd=self._px4_cmd(desired, ta if phase in ('PRE_TURN','TURN','SHORT') else 0.0)
        if phase=='TURN' and next_gate and dxy<db:
            bl=max(0,min(1,1-(dxy/db)))
            nx2,ny2,nz2=next_gate[0]-target[0],next_gate[1]-target[1],next_gate[2]-target[2]
            nd2=math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2)
            if nd2>0.1: nux,nuy,nuz=nx2/nd2,ny2/nd2,nz2/nd2
            else: nux,nuy,nuz=ux,uy,uz
            bxd=ux*(1-bl)+nux*bl;byd=uy*(1-bl)+nuy*bl;bzd=uz*(1-bl)+nuz*bl
            bm=math.sqrt(bxd*bxd+byd*byd+bzd*bzd)
            if bm>0.01: bxd,byd,bzd=bxd/bm,byd/bm,bzd/bm
            sp=self._smooth(cmd_spd,phase)
            vx,vy,vz=bxd*sp,byd*sp,bzd*sp; yaw=math.degrees(math.atan2(byd,bxd))
        else:
            sp=self._smooth(cmd_spd,phase)
            vx,vy,vz=ux*sp,uy*sp,uz*sp
        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)


# =====================================================================
# EXM1 Planner - V5.1 + execution_model_v1 empirical curves
# =====================================================================
class EXM1Planner(V51Planner):
    """V5.1 with empirical response curves (Curve 1 ceiling + Curve 2 launch)."""

    # Curve 1: avg_ach p50 per leg_length bin at v_cmd in [10.5, 11),
    # non-leg0 TRUTH cohort, n=2748. Used as a per-leg speed ceiling.
    _CEILING_BINS = [
        (0,   8,  7.21),
        (8,  12,  7.81),
        (12, 16,  8.34),
        (16, 20,  7.65),  # real dip in data
        (20, 24,  7.77),
        (24, 30,  9.06),
        (30, 999, 9.40),
    ]

    # Curve 2: leg0 empirical profile.
    # gate_time_by_leglen p50  (time from arm to passing first gate, seconds)
    # max_ach_by_leglen p50    (peak achieved speed on leg 0, m/s)
    _LEG0_BINS = [
        #  lo   hi  gate_time  max_ach
        (  0,  16,    2.72,      6.14),
        ( 16,  28,    4.52,      8.26),
        ( 28, 999,    5.99,      8.27),
    ]

    def __init__(self, max_speed=12.0, cruise_speed=10.0, base_blend=2.5):
        super().__init__(max_speed=max_speed, cruise_speed=cruise_speed, base_blend=base_blend)
        # Per-leg state additions
        self.leg0_length = None       # length of leg 0 (set on first plan call)
        self.current_leg_length = None # length of current leg (set at start of each leg)
        self._new_leg = True           # flag: next plan() call starts a fresh leg

    # ---- empirical lookups ----
    def _ceiling_for_length(self, L):
        if L is None:
            return self.px4_speed_ceiling  # fall back to baseline
        for lo, hi, v in self._CEILING_BINS:
            if lo <= L < hi:
                return v
        return self.px4_speed_ceiling

    def _leg0_params(self, L):
        """Return (gate_time, max_ach) for leg 0 of length L."""
        if L is None:
            return (self.px4_hover_ramp_time, self.px4_speed_ceiling)
        for lo, hi, gt, ma in self._LEG0_BINS:
            if lo <= L < hi:
                return (gt, ma)
        return (self.px4_hover_ramp_time, self.px4_speed_ceiling)

    def _phase(self, dxy, cs, ta, db):
        # Override LAUNCH check to use empirical leg0 gate_time instead of fixed 2.0s
        if self.is_first_leg and self.mission_start_time:
            gt, _ = self._leg0_params(self.leg0_length)
            if time.time()-self.mission_start_time < gt:
                return 'LAUNCH'
        if dxy<=db: return 'TURN'
        dd=self._decel_distance(cs,ta)
        if dxy<=dd and ta>0.3: return 'PRE_TURN'
        if dxy<6.0 and ta>0.8: return 'SHORT'
        return 'SUSTAIN'

    def on_gate_passed(self, speed):
        super().on_gate_passed(speed)
        self._new_leg = True

    def plan(self, pos, vel, target, next_gate=None):
        if self.mission_start_time is None: self.mission_start_time=time.time()
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dxy=math.sqrt(dx*dx+dy*dy); d3=math.sqrt(dx*dx+dy*dy+dz*dz)
        yaw=math.degrees(math.atan2(dy,dx))
        if d3<0.1: return VelocityNedYaw(0,0,0,yaw)

        # First plan() call of each new leg: record current leg length (= distance from
        # where we currently are to the target gate). On leg 0, also record as leg0_length.
        if self._new_leg:
            self.current_leg_length = dxy
            if self.is_first_leg and self.leg0_length is None:
                self.leg0_length = dxy
            self._new_leg = False

        ux,uy,uz=dx/d3,dy/d3,dz/d3
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        ta=0.0; nsl=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny); nsl=nd
            if nd>0.1 and dxy>0.1:
                ax,ay=dx/dxy,dy/dxy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        db=self.base_blend+0.25*cs+(ta/math.pi)*2.0
        self.blend_radius=db
        phase=self._phase(dxy,cs,ta,db)

        if phase=='LAUNCH':
            gt, ma = self._leg0_params(self.leg0_length)
            el=time.time()-self.mission_start_time
            # Linear ramp up to empirical max achievable speed on this leg0 length.
            # Empirical launch is closer to linear than quadratic; we end at ma, not cruise.
            ramp=min(1.0, el/max(gt,0.1))
            desired=max(ma*ramp, 2.5)
        elif phase=='SHORT':
            te=self._turn_entry_speed(ta); desired=min(max(cs+2.0,te*0.8),te)
        elif phase=='SUSTAIN': desired=self.cruise_speed
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

        # *** Key EXM1 change: per-leg-length empirical ceiling instead of scalar 8.5
        ceil = self._ceiling_for_length(self.current_leg_length)
        desired=min(desired, ceil)

        cmd_spd=self._px4_cmd(desired, ta if phase in ('PRE_TURN','TURN','SHORT') else 0.0)
        if phase=='TURN' and next_gate and dxy<db:
            bl=max(0,min(1,1-(dxy/db)))
            nx2,ny2,nz2=next_gate[0]-target[0],next_gate[1]-target[1],next_gate[2]-target[2]
            nd2=math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2)
            if nd2>0.1: nux,nuy,nuz=nx2/nd2,ny2/nd2,nz2/nd2
            else: nux,nuy,nuz=ux,uy,uz
            bxd=ux*(1-bl)+nux*bl;byd=uy*(1-bl)+nuy*bl;bzd=uz*(1-bl)+nuz*bl
            bm=math.sqrt(bxd*bxd+byd*byd+bzd*bzd)
            if bm>0.01: bxd,byd,bzd=bxd/bm,byd/bm,bzd/bm
            sp=self._smooth(cmd_spd,phase)
            vx,vy,vz=bxd*sp,byd*sp,bzd*sp; yaw=math.degrees(math.atan2(byd,bxd))
        else:
            sp=self._smooth(cmd_spd,phase)
            vx,vy,vz=ux*sp,uy*sp,uz*sp
        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)


def restart_px4():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system('pkill -9 -f "bin/px4" 2>/dev/null')
    time.sleep(2)
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(2)
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters.bson 2>/dev/null')
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters_backup.bson 2>/dev/null')
    subprocess.Popen(['/bin/bash','/tmp/run_px4_sih.sh'],stdout=open('/tmp/px4_sih_out.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(10)


async def run_trial(planner, gates, threshold=2.5):
    drone=System()
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
    alt=abs(gates[0][2])
    await drone.action.set_takeoff_altitude(alt)
    await drone.action.takeoff()
    await asyncio.sleep(4)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))
    await drone.offboard.start()
    gi=0; gt=[]; t0=time.time(); dt=1/50; max_spd=0
    cmds=[]; achs=[]
    while gi<len(gates):
        if time.time()-t0>90:
            print(f'      TIMEOUT@gate{gi+1}'); break
        g=gates[gi]
        d=math.sqrt((g[0]-pos[0])**2+(g[1]-pos[1])**2+(g[2]-pos[2])**2)
        if d<threshold:
            gt.append(time.time()-t0)
            gspd=math.sqrt(vel[0]**2+vel[1]**2)
            planner.on_gate_passed(gspd)
            gi+=1; continue
        ng=gates[gi+1] if gi+1<len(gates) else None
        cmd=planner.plan(pos,vel,g,ng)
        cspd=math.sqrt(cmd.north_m_s**2+cmd.east_m_s**2)
        aspd=math.sqrt(vel[0]**2+vel[1]**2)
        cmds.append(cspd); achs.append(aspd)
        if aspd>max_spd: max_spd=aspd
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)
    total=time.time()-t0
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
    splits=[round(gt[i]-(gt[i-1] if i>0 else 0),3) for i in range(len(gt))]
    avg_cmd=sum(cmds)/len(cmds) if cmds else 0
    avg_ach=sum(achs)/len(achs) if achs else 0
    errs=[abs(c-a) for c,a in zip(cmds,achs)]
    avg_err=sum(errs)/len(errs) if errs else 0
    util=avg_ach/avg_cmd if avg_cmd>0 else 0
    return {
        'completed':gi>=len(gates),'time':round(total,3),
        'max_spd':round(max_spd,2),'avg_cmd':round(avg_cmd,2),
        'avg_ach':round(avg_ach,2),'avg_err':round(avg_err,2),
        'util':round(util,3),'splits':splits,'gates_passed':gi,
    }


async def main():
    os.system('pkill -9 -f mavsdk_server 2>/dev/null')
    os.system("lsof -ti :14540 | xargs kill -9 2>/dev/null")
    os.system("lsof -ti :50051 | xargs kill -9 2>/dev/null")
    time.sleep(1)
    results=[]
    configs=[
        ('V5.1', V51Planner, dict(max_speed=11.0,cruise_speed=9.0,base_blend=1.5)),
        ('EXM1', EXM1Planner, dict(max_speed=11.0,cruise_speed=9.0,base_blend=1.5)),
    ]
    for cn in ['mixed','sprint','technical']:
        gates=COURSES[cn]
        print(f'\n{"="*60}')
        print(f'COURSE: {cn.upper()} ({len(gates)} gates)')
        print(f'{"="*60}')
        for trial in range(TRIALS_PER_COMBO):
            for vname, PlannerCls, kw in configs:
                print(f'  [trial {trial+1}/{TRIALS_PER_COMBO}] {vname}...',end=' ',flush=True)
                p=PlannerCls(**kw)
                r=await run_trial(p,gates)
                r['version']=vname; r['course']=cn; r['trial']=trial
                results.append(r)
                st='OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
                print(f'{st} {r["time"]}s | MaxSpd:{r["max_spd"]} AvgErr:{r["avg_err"]} Util:{r["util"]:.1%}')
                restart_px4(); await asyncio.sleep(2)

    # Aggregate per (course, version)
    print(f'\n\n{"="*72}')
    print('EXM1 vs V5.1 HEAD-TO-HEAD  (median over trials, OK only)')
    print(f'{"="*72}')
    print(f'{"Course":<12} {"Ver":<5} {"N":>3} {"Time p50":>9} {"MaxSpd":>7} {"AvgAch":>7} {"Util":>7}')
    print('-'*72)
    summary={}
    for cn in ['mixed','sprint','technical']:
        for vname,_,_ in configs:
            rs=[r for r in results if r['course']==cn and r['version']==vname and r['completed']]
            if not rs:
                print(f'{cn:<12} {vname:<5} {0:>3} {"no trials":>9}')
                continue
            rs_sorted=sorted(rs,key=lambda x:x['time'])
            med=rs_sorted[len(rs_sorted)//2]
            avg_ms=sum(r['max_spd'] for r in rs)/len(rs)
            avg_ach=sum(r['avg_ach'] for r in rs)/len(rs)
            avg_util=sum(r['util'] for r in rs)/len(rs)
            summary[(cn,vname)]=dict(
                n=len(rs),time_p50=med['time'],max_spd=round(avg_ms,2),
                avg_ach=round(avg_ach,2),util=round(avg_util,3),
            )
            print(f'{cn:<12} {vname:<5} {len(rs):>3} {med["time"]:>8.2f}s {avg_ms:>6.2f} {avg_ach:>6.2f} {avg_util:>6.1%}')
        # delta row
        a=summary.get((cn,'V5.1')); b=summary.get((cn,'EXM1'))
        if a and b:
            d=a['time_p50']-b['time_p50']; pct=d/a['time_p50']*100
            marker='>>' if d>0 else '<<'
            print(f'  {marker} EXM1 delta vs V5.1: {-d:+.2f}s ({-pct:+.1f}%)   [negative = EXM1 faster]')
        print()

    out={'trials':results,'summary':{f'{k[0]}_{k[1]}':v for k,v in summary.items()}}
    with open('/Users/conradweeden/ai-grand-prix/logs/exm1_vs_v51.json','w') as f:
        json.dump(out,f,indent=2)
    print('Saved to logs/exm1_vs_v51.json')


asyncio.run(main())
