"""Closed-loop tests: AttitudeController + MockDCLAdapter attitude dynamics.

Validates the full pipeline: velocity demand → attitude controller → T/R/P/Y
→ MockDCLAdapter physics → position/velocity change.
"""
import sys, os, math, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sim.mock_dcl import MockDCLAdapter
from control.attitude_controller import AttitudeController

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


async def run_tests():
    print("=== MockDCLAdapter + AttitudeController Tests ===\n")

    # ── 1. Adapter basics ──────────────────────────────────
    print("[Adapter basics]")
    from sim.adapter import SimCapability
    adapter = MockDCLAdapter()

    check("has ATTITUDE cap",
          SimCapability.ATTITUDE in adapter.capabilities)
    check("no VELOCITY_NED cap",
          SimCapability.VELOCITY_NED not in adapter.capabilities)
    check("has CAMERA_RGB",
          SimCapability.CAMERA_RGB in adapter.capabilities)

    # send_velocity_ned should raise
    try:
        await adapter.send_velocity_ned(1, 0, 0, 0)
        check("vel_ned raises", False, "did not raise")
    except NotImplementedError:
        check("vel_ned raises", True)

    # send_attitude should work
    await adapter.connect()
    await adapter.send_attitude(0, 0, 0, 0.5)
    state = await adapter.get_state()
    check("send_attitude works", state.connected)

    # ── 2. Hover hold ──────────────────────────────────────
    print("\n[Hover hold]")
    adapter = MockDCLAdapter(initial_altitude_m=5.0)
    await adapter.connect()

    # Find hover throttle: at hover, thrust = gravity
    # hover_throttle = g / max_thrust = 9.81 / 20.0 ≈ 0.49
    hover_thr = 9.81 / 20.0
    for _ in range(100):  # 2 seconds at 50Hz
        await adapter.send_attitude(0, 0, 0, hover_thr)

    state = await adapter.get_state()
    alt = -state.pos_ned[2]  # NED: down is positive, so altitude = -z
    check("hover altitude stable", abs(alt - 5.0) < 1.0,
          f"alt={alt:.2f}, expected ~5.0")
    check("hover vel_down small", abs(state.vel_ned[2]) < 1.0,
          f"vd={state.vel_ned[2]:.2f}")

    # ── 3. Forward flight via attitude controller ──────────
    print("\n[Forward flight closed-loop]")
    adapter = MockDCLAdapter(initial_altitude_m=5.0)
    await adapter.connect()
    ctrl = AttitudeController(hover_throttle=hover_thr)

    target_vn = 3.0
    for i in range(200):  # 4 seconds
        state = await adapter.get_state()
        cmd = ctrl.convert(
            desired_vn=target_vn, desired_ve=0, desired_vd=0,
            desired_yaw_deg=0,
            current_vel_ned=state.vel_ned,
            current_yaw_deg=math.degrees(state.att_rad[2]),
            dt=1.0/50,
        )
        await adapter.send_attitude(cmd.roll_deg, cmd.pitch_deg, cmd.yaw_deg, cmd.throttle)

    state = await adapter.get_state()
    check("forward vel converged", abs(state.vel_ned[0] - target_vn) < 2.0,
          f"vn={state.vel_ned[0]:.2f}, target={target_vn}")
    check("moved north", state.pos_ned[0] > 2.0,
          f"pos_n={state.pos_ned[0]:.2f}")
    check("lateral drift small", abs(state.pos_ned[1]) < 2.0,
          f"pos_e={state.pos_ned[1]:.2f}")

    # ── 4. Lateral flight ──────────────────────────────────
    print("\n[Lateral flight closed-loop]")
    adapter = MockDCLAdapter(initial_altitude_m=5.0)
    await adapter.connect()
    ctrl = AttitudeController(hover_throttle=hover_thr)

    target_ve = 3.0
    for i in range(200):
        state = await adapter.get_state()
        cmd = ctrl.convert(
            desired_vn=0, desired_ve=target_ve, desired_vd=0,
            desired_yaw_deg=0,
            current_vel_ned=state.vel_ned,
            current_yaw_deg=math.degrees(state.att_rad[2]),
            dt=1.0/50,
        )
        await adapter.send_attitude(cmd.roll_deg, cmd.pitch_deg, cmd.yaw_deg, cmd.throttle)

    state = await adapter.get_state()
    check("east vel converged", abs(state.vel_ned[1] - target_ve) < 2.0,
          f"ve={state.vel_ned[1]:.2f}, target={target_ve}")
    check("moved east", state.pos_ned[1] > 2.0,
          f"pos_e={state.pos_ned[1]:.2f}")

    # ── 5. Yaw turn ───────────────────────────────────────
    print("\n[Yaw turn]")
    adapter = MockDCLAdapter(initial_altitude_m=5.0)
    await adapter.connect()
    ctrl = AttitudeController(hover_throttle=hover_thr)

    for i in range(100):
        state = await adapter.get_state()
        cmd = ctrl.convert(
            desired_vn=0, desired_ve=0, desired_vd=0,
            desired_yaw_deg=90.0,
            current_vel_ned=state.vel_ned,
            current_yaw_deg=math.degrees(state.att_rad[2]),
            dt=1.0/50,
        )
        await adapter.send_attitude(cmd.roll_deg, cmd.pitch_deg, cmd.yaw_deg, cmd.throttle)

    state = await adapter.get_state()
    yaw_deg = math.degrees(state.att_rad[2])
    check("yaw reached ~90", abs(yaw_deg - 90) < 15,
          f"yaw={yaw_deg:.1f}")

    # ── 6. Camera frame ───────────────────────────────────
    print("\n[Camera frame]")
    frame = await adapter.get_camera_frame()
    check("frame not None", frame is not None)
    check("frame shape", frame.shape == (480, 640, 3), f"{frame.shape}")

    # ── 7. IMU exists after stepping ──────────────────────
    print("\n[IMU]")
    imu = await adapter.get_imu()
    check("IMU not None", imu is not None)
    check("IMU has accel", hasattr(imu, 'accel_body'))
    check("IMU has gyro", hasattr(imu, 'gyro_body'))

    # ── 8. Reset ──────────────────────────────────────────
    print("\n[Reset]")
    await adapter.reset()
    state = await adapter.get_state()
    check("reset pos north", abs(state.pos_ned[0]) < 0.1, f"{state.pos_ned[0]}")
    check("reset pos east", abs(state.pos_ned[1]) < 0.1, f"{state.pos_ned[1]}")
    check("reset altitude", abs(-state.pos_ned[2] - 5.0) < 0.1,
          f"alt={-state.pos_ned[2]:.2f}")

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


asyncio.run(run_tests())
