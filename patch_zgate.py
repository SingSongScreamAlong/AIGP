#!/usr/bin/env python3
"""Patch px4_v51_baseline.py to add z_gate to V51Planner ONLY (not EXM1)."""
import sys, re
P='/Users/conradweeden/ai-grand-prix/px4_v51_baseline.py'
src=open(P).read()

# 1) Extend __init__ state with z_gate fields. Anchor on the cold_ramp_seed line.
old_init="self.last_cmd_speed=0.0; self.mission_start_time=None; self.cold_ramp_seed=0.0  # S8 prefill seed"
new_init=("self.last_cmd_speed=0.0; self.mission_start_time=None; self.cold_ramp_seed=0.0  # S8 prefill seed\n"
          "        # S8 z_gate: latched horizontal-authority gate during LAUNCH\n"
          "        self.z_gate_alt_frac=0.0; self.z_gate_vz_band=0.25; self.z_gate_ramp_ms=200\n"
          "        self.z_gate_open=False; self.z_gate_open_time=None\n"
          "        self.z_gate_open_alt=None; self.z_gate_open_vz=None")
assert old_init in src, "init anchor missing"
src=src.replace(old_init,new_init,1)

# 2) on_gate_passed: keep z_gate state as-is (latched). No change needed.

# 3) plan(): scale horizontal mag during LAUNCH if z_gate enabled.
#    Insert just before the final `return VelocityNedYaw(vx,vy,vz,yaw)` of V51Planner.plan().
#    The cleanest insertion point is right after `vz=(target[2]-pos[2])*3.0` and before
#    the `st=math.sqrt(vx*vx+vy*vy+vz*vz)` clamp. We must affect ONLY V51Planner, not the
#    EXM1Planner override. EXM1Planner has its own plan() override below.
#
#    Strategy: anchor on V51Planner's specific surrounding context that doesn't appear in EXM1.
old_block="""        vz=(target[2]-pos[2])*3.0
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)"""
new_block="""        vz=(target[2]-pos[2])*3.0
        # --- S8 z_gate: latched horizontal attenuation during LAUNCH only ---
        if self.is_first_leg and phase=='LAUNCH' and self.z_gate_alt_frac>0:
            alt_now=-pos[2]; alt_tgt=-target[2]
            vz_now=abs(vel[2]) if len(vel)>2 else 0.0
            if not self.z_gate_open:
                if alt_tgt>0.1 and (alt_now/alt_tgt)>=self.z_gate_alt_frac and vz_now<=self.z_gate_vz_band:
                    self.z_gate_open=True
                    self.z_gate_open_time=time.time()
                    self.z_gate_open_alt=alt_now
                    self.z_gate_open_vz=vz_now
            if not self.z_gate_open:
                vx=0.0; vy=0.0
            else:
                el_ms=(time.time()-self.z_gate_open_time)*1000.0
                g=min(1.0, el_ms/max(self.z_gate_ramp_ms,1.0))
                vx*=g; vy*=g
        # --- end z_gate ---
        st=math.sqrt(vx*vx+vy*vy+vz*vz)
        if st>self.max_speed: s=self.max_speed/st; vx*=s;vy*=s;vz*=s
        return VelocityNedYaw(vx,vy,vz,yaw)"""
# Both V51Planner.plan() and EXM1Planner.plan() end with the same tail.
# Replace ONLY the FIRST occurrence (which is V51Planner since it's defined first).
n=src.count(old_block)
assert n==2, f"expected 2 matches (V51 and EXM1), got {n}"
# Sanity: confirm V51Planner appears before the first match
v51_pos=src.find("class V51Planner")
exm1_pos=src.find("class EXM1Planner")
first_match=src.find(old_block)
assert v51_pos < first_match < exm1_pos, "first match not inside V51Planner"
src=src.replace(old_block,new_block,1)  # only first

open(P,'w').write(src)
print("OK")
