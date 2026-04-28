"""Standalone attitude controller tests — no pytest needed."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from control.attitude_controller import AttitudeController, _wrap_180

DT = 0.02
passed = 0
failed = 0

def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1

print("=== Attitude Controller Tests ===\n")

# 1. Hover hold
ctrl = AttitudeController()
cmd = ctrl.convert(0,0,0,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("hover throttle", abs(cmd.throttle - 0.5) < 0.05, f"{cmd.throttle}")
check("hover pitch", abs(cmd.pitch_deg) < 1.0, f"{cmd.pitch_deg}")
check("hover roll", abs(cmd.roll_deg) < 1.0, f"{cmd.roll_deg}")

# 2. Forward flight
ctrl.reset()
cmd = ctrl.convert(5,0,0,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("forward→pitch<0", cmd.pitch_deg < 0, f"{cmd.pitch_deg}")
check("forward→roll≈0", abs(cmd.roll_deg) < 1.0, f"{cmd.roll_deg}")

# 3. Lateral flight
ctrl.reset()
cmd = ctrl.convert(0,5,0,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("east→roll>0", cmd.roll_deg > 0, f"{cmd.roll_deg}")
check("east→pitch≈0", abs(cmd.pitch_deg) < 1.0, f"{cmd.pitch_deg}")

# 4. Altitude
ctrl.reset()
cmd = ctrl.convert(0,0,-1,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("climb→throttle up", cmd.throttle > 0.5, f"{cmd.throttle}")
ctrl.reset()
cmd = ctrl.convert(0,0,1,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("descend→throttle down", cmd.throttle < 0.5, f"{cmd.throttle}")

# 5. Yaw tracking
ctrl.reset()
cmd = ctrl.convert(0,0,0,90, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("yaw→toward 90", cmd.yaw_deg > 0, f"{cmd.yaw_deg}")

# 6. Yaw shortest path (350 from 10 = turn left)
ctrl.reset()
cmd = ctrl.convert(0,0,0,350, current_vel_ned=(0,0,0), current_yaw_deg=10, dt=DT)
check("yaw shortest path", cmd.yaw_deg < 10, f"{cmd.yaw_deg}")

# 7. Body-frame rotation (yaw=90, east demand = body forward)
ctrl.reset()
cmd = ctrl.convert(0,5,0,90, current_vel_ned=(0,0,0), current_yaw_deg=90, dt=DT)
check("yaw90+east→pitch<0", cmd.pitch_deg < -1.0, f"{cmd.pitch_deg}")
check("yaw90+east→roll≈0", abs(cmd.roll_deg) < 2.0, f"{cmd.roll_deg}")

# 8. Body-frame rotation (yaw=90, north demand = body left)
ctrl.reset()
cmd = ctrl.convert(5,0,0,90, current_vel_ned=(0,0,0), current_yaw_deg=90, dt=DT)
check("yaw90+north→roll<0", cmd.roll_deg < -1.0, f"{cmd.roll_deg}")

# 9. Saturation
ctrl2 = AttitudeController(max_roll_deg=30.0, max_pitch_deg=25.0)
cmd = ctrl2.convert(100,100,0,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("roll saturated", abs(cmd.roll_deg) <= 30.01, f"{cmd.roll_deg}")
check("pitch saturated", abs(cmd.pitch_deg) <= 25.01, f"{cmd.pitch_deg}")

# 10. Throttle clamp
ctrl.reset()
cmd = ctrl.convert(0,0,-100,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("throttle max clamp", cmd.throttle <= 1.0, f"{cmd.throttle}")
cmd = ctrl.convert(0,0,100,0, current_vel_ned=(0,0,0), current_yaw_deg=0, dt=DT)
check("throttle min clamp", cmd.throttle >= 0.0, f"{cmd.throttle}")

# 11. Reset clears state
ctrl.reset()
check("reset integral vn", ctrl._vn_state.integral == 0.0)
check("reset integral ve", ctrl._ve_state.integral == 0.0)

# 12. Convergence (closed-loop with simple plant)
ctrl.reset()
g = 9.81
vel_n = 0.0
target = 3.0
for _ in range(200):
    cmd = ctrl.convert(target,0,0,0, current_vel_ned=(vel_n,0,0), current_yaw_deg=0, dt=DT)
    # Negative pitch tilts thrust forward → positive north accel
    vel_n += g * math.tan(math.radians(-cmd.pitch_deg)) * DT
check("convergence", abs(vel_n - target) < 1.0, f"vel={vel_n:.2f} target={target}")

# 13. wrap_180
check("wrap 270→-90", _wrap_180(270) == -90.0)
check("wrap -270→90", _wrap_180(-270) == 90.0)

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
