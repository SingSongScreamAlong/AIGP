
P = "/Users/conradweeden/ai-grand-prix/px4_v51_baseline.py"
s = open(P).read()

old_sig = "async def run_trial(planner, gates, threshold=2.5):"
new_sig = "async def run_trial(planner, gates, threshold=2.5, use_takeoff_gate=True, gate_alt_frac=0.95, gate_vz_max=0.3, gate_timeout=10.0):"
assert s.count(old_sig) == 1, "sig count " + str(s.count(old_sig))
s = s.replace(old_sig, new_sig)

old_sleep = "    await drone.action.takeoff()\n    await asyncio.sleep(4)\n    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))"
assert s.count(old_sleep) == 1, "sleep count " + str(s.count(old_sleep))
new_sleep = (
    "    await drone.action.takeoff()\n"
    "    _wait_start = time.time()\n"
    "    if use_takeoff_gate:\n"
    "        while True:\n"
    "            if (time.time() - _wait_start) >= gate_timeout: break\n"
    "            if abs(pos[2]) >= gate_alt_frac * alt and abs(vel[2]) < gate_vz_max: break\n"
    "            await asyncio.sleep(0.05)\n"
    "    else:\n"
    "        await asyncio.sleep(4)\n"
    "    t_offboard_wait = round(time.time() - _wait_start, 3)\n"
    "    alt_at_offboard = round(abs(pos[2]), 3)\n"
    "    vz_at_offboard = round(vel[2], 3)\n"
    "    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,0))"
)
s = s.replace(old_sleep, new_sleep)

old_ret = "'util':round(util,3),'splits':splits,'gates_passed':gi,"
assert s.count(old_ret) == 1, "ret count " + str(s.count(old_ret))
new_ret = "'util':round(util,3),'splits':splits,'gates_passed':gi,\n        't_offboard_wait':t_offboard_wait,'alt_at_offboard':alt_at_offboard,'vz_at_offboard':vz_at_offboard,"
s = s.replace(old_ret, new_ret)

open(P, "w").write(s)
print("PATCH_OK", len(s))
