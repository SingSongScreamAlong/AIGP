# bench.py — Session 9 hardening helpers.
# Drop-in. No edits to px4_v51_baseline.py or ab_zgate.py.
import asyncio, os, time, json, subprocess, sys, fcntl, atexit, errno

PIDFILE = '/tmp/agp_soak.pid'

def acquire_singleton(name='soak'):
    """Refuse to start if another instance is already running. Atomic via O_EXCL."""
    p = f'/tmp/agp_{name}.pid'
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            existing = open(p).read().strip()
        except Exception:
            existing = '?'
        raise RuntimeError(f'singleton lock {p} already exists (pid={existing}); refusing to start')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(lambda: (os.path.exists(p) and os.unlink(p)))
    return p

def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def port_free(port, timeout=10.0):
    """Wait until nothing is listening on `port` (or held by a TCP/UDP socket)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _run(f'lsof -ti :{port} 2>/dev/null')
        if not r.stdout.strip():
            return True
        time.sleep(0.25)
    return False

def kill_stack():
    """Aggressively tear down PX4 + mavsdk_server + bound ports. Idempotent."""
    _run('pkill -9 -f mavsdk_server 2>/dev/null')
    _run('pkill -9 -f "bin/px4" 2>/dev/null')
    _run('pkill -9 -f "px4_sitl_default" 2>/dev/null')
    time.sleep(1.5)
    for p in (14540, 50051):
        r = _run(f'lsof -ti :{p} 2>/dev/null')
        if r.stdout.strip():
            for pid in r.stdout.split():
                _run(f'kill -9 {pid} 2>/dev/null')
    time.sleep(0.5)

def hardened_restart(B):
    """Replacement for B.restart_px4 — guarantees ports free before returning."""
    kill_stack()
    if not port_free(14540, timeout=8.0):
        raise RuntimeError('hardened_restart: port 14540 still bound after kill_stack')
    if not port_free(50051, timeout=8.0):
        raise RuntimeError('hardened_restart: port 50051 still bound after kill_stack')
    # delegate the actual PX4 launch to the existing baseline (it has the right paths)
    # but first wipe stale param files (mirrors restart_px4)
    _run('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters.bson 2>/dev/null')
    _run('rm -f /Users/conradweeden/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters_backup.bson 2>/dev/null')
    subprocess.Popen(
        ['/bin/bash','/tmp/run_px4_sih.sh'],
        stdout=open('/tmp/px4_sih_out.log','w'),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # wait for PX4 to be ready by checking the log for the takeoff-ready marker
    deadline = time.time() + 25.0
    ready = False
    while time.time() < deadline:
        try:
            txt = open('/tmp/px4_sih_out.log').read()
            if 'Ready for takeoff' in txt or 'INFO  [commander] Ready for takeoff' in txt:
                ready = True
                break
        except FileNotFoundError:
            pass
        time.sleep(0.25)
    if not ready:
        raise RuntimeError('hardened_restart: PX4 did not reach "Ready for takeoff" within 25s')
    # small extra settle for MAVLink endpoint
    time.sleep(1.5)

async def wait_healthy(drone, timeout=15.0):
    """Verify drone is actually responsive: connected, telemetry flowing, EKF position ok, armable.
       Returns dict with diagnostic flags. Caller decides whether to abort."""
    diag = {'connected': False, 'telemetry': False, 'position_ok': False, 'armable': False}
    deadline = time.time() + timeout
    # connection
    try:
        async for s in drone.core.connection_state():
            if s.is_connected:
                diag['connected'] = True
                break
            if time.time() > deadline: break
    except Exception:
        pass
    if not diag['connected']:
        return diag
    # telemetry stream actually producing data
    try:
        async def _one():
            async for _ in drone.telemetry.position_velocity_ned():
                return True
            return False
        diag['telemetry'] = await asyncio.wait_for(_one(), timeout=max(1.0, deadline-time.time()))
    except Exception:
        diag['telemetry'] = False
    # EKF / local position health
    try:
        async def _hpos():
            async for h in drone.telemetry.health():
                return bool(h.is_local_position_ok and h.is_global_position_ok or h.is_local_position_ok)
            return False
        diag['position_ok'] = await asyncio.wait_for(_hpos(), timeout=max(1.0, deadline-time.time()))
    except Exception:
        diag['position_ok'] = False
    # armable
    try:
        async def _arm():
            async for h in drone.telemetry.health():
                return bool(h.is_armable)
            return False
        diag['armable'] = await asyncio.wait_for(_arm(), timeout=max(1.0, deadline-time.time()))
    except Exception:
        diag['armable'] = False
    return diag

def atomic_write_json(path, data):
    """Write JSON atomically: temp file + fsync + rename. No partial states ever visible."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
