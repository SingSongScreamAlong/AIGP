"""V4 vs V3 Head-to-Head on PX4 SITL - All 3 courses"""
import asyncio, time, math, json, subprocess, os
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

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

class V3Planner:
    def __init__(self, max_speed=12.0, cruise_speed=10.0, base_blend=2.5):
        self.max_speed=max_speed; self.cruise_speed=cruise_speed; self.base_blend=base_blend
    def plan(self, pos, vel, target, next_gate=None):
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dist_xy=math.sqrt(dx*dx+dy*dy); dist_3d=math.sqrt(dx*dx+dy*dy+dz*dz)
        yaw=math.degrees(math.atan2(dy,dx))
        if dist_3d<0.1: return VelocityNedYaw(0,0,0,yaw)
        ux,uy,uz=dx/dist_3d,dy/dist_3d,dz/dist_3d
        ta=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny)
            if nd>0.1 and dist_xy>0.1:
                ax,ay=dx/dist_xy,dy/dist_xy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        db=self.base_blend+0.2*cs+(ta/math.pi)*1.5
        ac=self.cruise_speed*(0.4+0.6*math.cos(ta/2))
        if next_gate and dist_xy<db:
            bl=max(0,min(1,1-(dist_xy/db)))
            nx2,ny2,nz2=next_gate[0]-target[0],next_gate[1]-target[1],next_gate[2]-target[2]
            nd2=math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2)
            if nd2>0.1: nux,nuy,nuz=nx2/nd2,ny2/nd2,nz2/nd2
            else: nux,nuy,nuz=ux,uy,uz
            bxd=ux*(1-bl)+nux*bl;byd=uy*(1-bl)+nuy*bl;bzd=uz*(1-bl)+nuz*bl
            bm=math.sqrt(bxd*bxd+byd*byd+bzd*bzd)
            if bm>0.01: bxd,byd,bzd=bxd/bm,byd/bm,bzd/bm
            af=1.0-0.3*math.sin(bl*math.pi); sp=ac*af
            vx,vy,vz=bxd*sp,byd*sp,bzd*sp
            yaw=math.degrees(math.atan2(byd,bxd))
        else:
            if dist_xy<db*1.5:
                af=dist_xy/(db*1.5); sp=ac*(0.7+0.3*af)
            else:
                sp=min(ac,max(1.8,dist_xy*2.4))
            vx,vy,vz=ux*sp,uy*sp,uz*sp
        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)

class V4Planner:
    def __init__(self, max_speed=12.0, cruise_speed=10.0, base_blend=2.5):
        self.max_speed=max_speed; self.cruise_speed=cruise_speed; self.base_blend=base_blend
        self.px4_max_accel=6.0; self.px4_speed_ceiling=8.0
        self.px4_util_straight=0.75; self.px4_util_turn=0.55
        self.px4_hover_ramp_time=3.0
        self.last_cmd_speed=0.0; self.mission_start_time=None
    def _px4_speed(self, raw, ta, seg_len, cur_spd):
        tr=ta/math.pi
        util=self.px4_util_straight*(1-tr)+self.px4_util_turn*tr
        if seg_len>0.1:
            ma=math.sqrt(cur_spd**2+2*self.px4_max_accel*seg_len)
            raw=min(raw,min(ma,self.px4_speed_ceiling))
        return min(raw/max(util,0.3),self.max_speed)
    def _smooth(self, target, dt=0.02):
        mr=8.0; md=mr*dt
        if target>self.last_cmd_speed: s=min(target,self.last_cmd_speed+md)
        else: s=max(target,self.last_cmd_speed-md*2)
        self.last_cmd_speed=s; return s
    def plan(self, pos, vel, target, next_gate=None):
        if self.mission_start_time is None: self.mission_start_time=time.time()
        dx,dy,dz=target[0]-pos[0],target[1]-pos[1],target[2]-pos[2]
        dist_xy=math.sqrt(dx*dx+dy*dy); dist_3d=math.sqrt(dx*dx+dy*dy+dz*dz)
        yaw=math.degrees(math.atan2(dy,dx))
        if dist_3d<0.1: return VelocityNedYaw(0,0,0,yaw)
        ux,uy,uz=dx/dist_3d,dy/dist_3d,dz/dist_3d
        cs=math.sqrt(vel[0]**2+vel[1]**2)
        ta=0.0
        if next_gate:
            nx,ny=next_gate[0]-target[0],next_gate[1]-target[1]
            nd=math.sqrt(nx*nx+ny*ny)
            if nd>0.1 and dist_xy>0.1:
                ax,ay=dx/dist_xy,dy/dist_xy; bx,by=nx/nd,ny/nd
                d=max(-1,min(1,ax*bx+ay*by)); ta=math.acos(d)
        db=self.base_blend+0.25*cs+(ta/math.pi)*2.0
        base_target=self.cruise_speed*(0.4+0.6*math.cos(ta/2))
        me=time.time()-self.mission_start_time
        if me<self.px4_hover_ramp_time:
            r=(me/self.px4_hover_ramp_time)**2; base_target=max(base_target*r,2.0)
        if dist_xy<8.0 and ta>0.5:
            base_target*=(0.5+0.5*(dist_xy/8.0))
        cmd_spd=self._px4_speed(base_target,ta,dist_xy,cs)
        if next_gate and dist_xy<db:
            bl=max(0,min(1,1-(dist_xy/db)))
            nx2,ny2,nz2=next_gate[0]-target[0],next_gate[1]-target[1],next_gate[2]-target[2]
            nd2=math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2)
            if nd2>0.1: nux,nuy,nuz=nx2/nd2,ny2/nd2,nz2/nd2
            else: nux,nuy,nuz=ux,uy,uz
            bxd=ux*(1-bl)+nux*bl;byd=uy*(1-bl)+nuy*bl;bzd=uz*(1-bl)+nuz*bl
            bm=math.sqrt(bxd*bxd+byd*byd+bzd*bzd)
            if bm>0.01: bxd,byd,bzd=bxd/bm,byd/bm,bzd/bm
            ad=0.2+0.2*(ta/math.pi); af=1.0-ad*math.sin(bl*math.pi)
            sp=self._smooth(cmd_spd*af)
            vx,vy,vz=bxd*sp,byd*sp,bzd*sp; yaw=math.degrees(math.atan2(byd,bxd))
        else:
            if dist_xy<db*2.0 and ta>0.3:
                ar=dist_xy/(db*2.0); sp=self._smooth(cmd_spd*(0.6+0.4*ar))
            else:
                sp=self._smooth(cmd_spd)
            vx,vy,vz=ux*sp,uy*sp,uz*sp
        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)

def restart_px4():
    os.system('pkill -f mavsdk_server 2>/dev/null')
    os.system('pkill -9 -f "bin/px4" 2>/dev/null')
    time.sleep(3)
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters.bson 2>/dev/null')
    os.system('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters_backup.bson 2>/dev/null')
    subprocess.Popen(['/bin/bash','/tmp/run_px4_sih.sh'],stdout=open('/tmp/px4_sih_out.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(7)

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
            gt.append(time.time()-t0); gi+=1; continue
        ng=gates[gi+1] if gi+1<len(gates) else None
        cmd=planner.plan(pos,vel,g,ng)
        cs=math.sqrt(cmd.north_m_s**2+cmd.east_m_s**2)
        avs=math.sqrt(vel[0]**2+vel[1]**2)
        cmds.append(cs); achs.append(avs)
        if avs>max_spd: max_spd=avs
        await drone.offboard.set_velocity_ned(cmd)
        await asyncio.sleep(dt)
    total=time.time()-t0
    try: await drone.offboard.stop()
    except: pass
    await drone.action.land(); await asyncio.sleep(2)
    # Kill mavsdk_server to release port before PX4 restart
    import signal
    try: drone._stop_mavsdk_server()
    except: pass
    await asyncio.sleep(1)
    os.system('pkill -f mavsdk_server 2>/dev/null')
    await asyncio.sleep(1)
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
    results=[]
    for cn in ['sprint','technical','mixed']:
        gates=COURSES[cn]
        print(f'\n{"="*60}')
        print(f'COURSE: {cn.upper()} ({len(gates)} gates)')
        print(f'{"="*60}')
        for vname,PlannerCls in [('V3',V3Planner),('V4',V4Planner)]:
            print(f'  {vname}...',end=' ',flush=True)
            p=PlannerCls(max_speed=12.0,cruise_speed=10.0,base_blend=2.5)
            r=await run_trial(p,gates)
            r['version']=vname; r['course']=cn
            results.append(r)
            st='OK' if r['completed'] else f'FAIL@{r["gates_passed"]}'
            print(f'{st} {r["time"]}s | MaxSpd:{r["max_spd"]} AvgErr:{r["avg_err"]} Util:{r["util"]:.1%}')
            os.system('pkill -f mavsdk_server 2>/dev/null')
            await asyncio.sleep(1)
            restart_px4(); await asyncio.sleep(2)

    print(f'\n\n{"="*70}')
    print('V4 vs V3 HEAD-TO-HEAD')
    print(f'{"="*70}')
    print(f'{"Course":<12} {"Ver":<5} {"Time":>7} {"MaxSpd":>7} {"AvgCmd":>7} {"AvgAch":>7} {"AvgErr":>7} {"Util":>7}')
    print('-'*70)
    for cn in ['sprint','technical','mixed']:
        cr=[r for r in results if r['course']==cn]
        for r in cr:
            print(f'{cn:<12} {r["version"]:<5} {r["time"]:>6.1f}s {r["max_spd"]:>5.1f} {r["avg_cmd"]:>6.1f} {r["avg_ach"]:>6.1f} {r["avg_err"]:>6.2f} {r["util"]:>6.1%}')
        if len(cr)==2:
            v3t=cr[0]['time']; v4t=cr[1]['time']
            delta=v3t-v4t; pct=delta/v3t*100
            print(f'  >> V4 delta: {delta:+.1f}s ({pct:+.1f}%)')
        print()

    with open('/Users/conradweeden/ai-grand-prix/logs/v4_vs_v3.json','w') as f:
        json.dump(results,f,indent=2)
    print('Saved to logs/v4_vs_v3.json')

asyncio.run(main())
