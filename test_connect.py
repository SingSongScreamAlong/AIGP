import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("  Connected to drone!")
            break

    print("Waiting for GPS lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("  GPS OK")
            break

    print("Arming...")
    await drone.action.arm()
    print("  Armed!")

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)

    async for position in drone.telemetry.position():
        print(f"  Altitude: {position.relative_altitude_m:.1f}m")
        break

    print("Landing...")
    await drone.action.land()
    await asyncio.sleep(10)

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("  Landed safely!")
            break

    print("Disarming...")
    await drone.action.disarm()
    print("  Done! First autonomous flight!")

asyncio.run(run())
