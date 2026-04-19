#!/usr/bin/env python3.13
"""
Parameter Sweep — AI Grand Prix Planner Envelope Test
=====================================================
Runs the mock sim + controller with different planner params,
captures lap time and gate pass count for each combo.
Outputs a results table sorted by lap time.

Usage:
    python3.13 sweep.py
"""

import subprocess
import time
import os
import signal
import json
import sys
import re
import itertools

# ── Sweep grid ──────────────────────────────────────────
CRUISE_SPEEDS = [6.0, 7.0, 8.0, 9.0, 10.0]
BLEND_RADII = [0.8, 1.2, 1.8, 2.5]
GATE_THRESHOLDS = [1.5, 2.0, 2.5]

# Fixed params
MAX_SPEED_OVERHEAD = 2.0  # max_speed = cruise + this
TIMEOUT = 20  # seconds per run

BASE_DIR = os.path.expanduser("~/ai-grand-prix")
SIM_PATH = os.path.join(BASE_DIR, "src/sim/mock_sim.py")
CTRL_PATH = os.path.join(BASE_DIR, "src/control/control_skeleton.py")
CTRL_BACKUP = CTRL_PATH + ".backup"
SIM_BACKUP = SIM_PATH + ".backup"


def patch_controller(cruise_speed, max_speed, blend_radius, threshold):
    """Patch controller params in-place."""
    with open(CTRL_PATH) as f:
        code = f.read()

    # Replace Planner constructor
    code = re.sub(
        r'Planner\(max_speed=[\d.]+, cruise_speed=[\d.]+, base_blend=[\d.]+\)',
        f'Planner(max_speed={max_speed}, cruise_speed={cruise_speed}, base_blend={blend_radius})',
        code
    )

    # Replace gate threshold
    code = re.sub(
        r'threshold=[\d.]+\)',
        f'threshold={threshold})',
        code
    )

    # Replace speed ramp — scale with cruise speed
    ramp_mult = max(2.0, cruise_speed * 0.4)
    min_speed = max(1.5, cruise_speed * 0.3)
    code = re.sub(
        r"max\([\d.]+, dist_xy \* [\d.]+\)",
        f"max({min_speed:.1f}, dist_xy * {ramp_mult:.1f})",
        code
    )

    with open(CTRL_PATH, 'w') as f:
        f.write(code)


def run_trial(cruise_speed, blend_radius, threshold):
    """Run one sim+controller trial, return (gates_passed, lap_time, success)."""
    max_speed = cruise_speed + MAX_SPEED_OVERHEAD

    # Patch controller
    patch_controller(cruise_speed, max_speed, blend_radius, threshold)

    # Kill any existing processes
    subprocess.run(["pkill", "-9", "-f", "mock_sim.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "control_skeleton.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "mavsdk_server"], capture_output=True)
    time.sleep(2.0)  # longer wait for UDP port release

    # Wait for port 14540 to be free
    for _ in range(10):
        r = subprocess.run(["lsof", "-i", ":14540"], capture_output=True)
        if r.returncode != 0:  # port free
            break
        time.sleep(0.5)

    # Clear logs
    ctrl_log = "/tmp/sweep_ctrl.log"
    sim_log = "/tmp/sweep_sim.log"
    for f in [ctrl_log, sim_log]:
        if os.path.exists(f):
            os.remove(f)

    # Start sim
    sim_proc = subprocess.Popen(
        ["python3.13", "-u", SIM_PATH],
        stdout=open(sim_log, 'w'),
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        preexec_fn=os.setsid
    )

    time.sleep(2.0)  # let sim initialize and bind port

    # Start controller
    ctrl_proc = subprocess.Popen(
        ["python3.13", "-u", CTRL_PATH],
        stdout=open(ctrl_log, 'w'),
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        preexec_fn=os.setsid
    )

    # Wait for completion or timeout
    start = time.time()
    gates_passed = 0
    lap_time = None
    success = False

    while time.time() - start < TIMEOUT:
        time.sleep(0.5)
        if os.path.exists(ctrl_log):
            try:
                with open(ctrl_log) as f:
                    log = f.read()
                if "MISSION COMPLETE" in log:
                    # Parse results
                    m = re.search(r'Total time: ([\d.]+)s', log)
                    if m:
                        lap_time = float(m.group(1))
                    gate_matches = re.findall(r'Passed gate (\d+)/(\d+)', log)
                    if gate_matches:
                        gates_passed = int(gate_matches[-1][0])
                    success = True
                    break
                # Check for errors
                if "ERROR" in log or "Traceback" in log:
                    break
            except:
                pass

    # If no mission complete, check partial progress
    if not success and os.path.exists(ctrl_log):
        try:
            with open(ctrl_log) as f:
                log = f.read()
            gate_matches = re.findall(r'Passed gate (\d+)/(\d+)', log)
            if gate_matches:
                gates_passed = int(gate_matches[-1][0])
        except:
            pass

    # Kill processes
    try:
        os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
    except:
        pass
    try:
        os.killpg(os.getpgid(ctrl_proc.pid), signal.SIGKILL)
    except:
        pass

    time.sleep(1.0)  # let ports fully release
    return gates_passed, lap_time, success


def main():
    # Backup originals
    subprocess.run(["cp", CTRL_PATH, CTRL_BACKUP])
    subprocess.run(["cp", SIM_PATH, SIM_BACKUP])

    results = []
    total = len(CRUISE_SPEEDS) * len(BLEND_RADII) * len(GATE_THRESHOLDS)
    run_num = 0

    print(f"{'='*70}")
    print(f"  AI Grand Prix — Parameter Sweep ({total} trials)")
    print(f"{'='*70}")
    print(f"  Speeds: {CRUISE_SPEEDS}")
    print(f"  Blends: {BLEND_RADII}")
    print(f"  Thresholds: {GATE_THRESHOLDS}")
    print(f"{'='*70}\n")

    try:
        for cruise in CRUISE_SPEEDS:
            for blend in BLEND_RADII:
                for thresh in GATE_THRESHOLDS:
                    run_num += 1
                    tag = f"[{run_num}/{total}]"
                    sys.stdout.write(f"  {tag} v={cruise:.0f} b={blend:.1f} t={thresh:.1f} ... ")
                    sys.stdout.flush()

                    gates, lap, ok = run_trial(cruise, blend, thresh)

                    if ok and lap:
                        print(f"✓ {lap:.2f}s ({gates}/4 gates)")
                    elif gates > 0:
                        print(f"✗ TIMEOUT ({gates}/4 gates)")
                    else:
                        print(f"✗ FAIL (0 gates)")

                    results.append({
                        "cruise_speed": cruise,
                        "blend_radius": blend,
                        "threshold": thresh,
                        "max_speed": cruise + MAX_SPEED_OVERHEAD,
                        "gates_passed": gates,
                        "lap_time": lap,
                        "success": ok,
                    })

    except KeyboardInterrupt:
        print("\n\n  [SWEEP] Interrupted!")

    finally:
        # Restore originals
        subprocess.run(["cp", CTRL_BACKUP, CTRL_PATH])
        subprocess.run(["cp", SIM_BACKUP, SIM_PATH])

        # Kill any remaining
        subprocess.run(["pkill", "-9", "-f", "mock_sim.py"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "control_skeleton.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "mavsdk_server"], capture_output=True)

    # ── Results ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS — {len(results)} trials")
    print(f"{'='*70}\n")

    # Sort by success then lap time
    successful = [r for r in results if r["success"] and r["lap_time"]]
    failed = [r for r in results if not r["success"]]

    successful.sort(key=lambda r: r["lap_time"])

    print(f"  {'Speed':>5} {'Blend':>5} {'Thresh':>6} {'Time':>6} {'Gates':>5}")
    print(f"  {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*5}")

    for r in successful[:20]:
        print(f"  {r['cruise_speed']:5.1f} {r['blend_radius']:5.1f} {r['threshold']:6.1f} {r['lap_time']:5.2f}s {r['gates_passed']:>5}/4")

    if failed:
        print(f"\n  Failed/Timeout: {len(failed)}/{len(results)}")
        for r in failed[:10]:
            print(f"    v={r['cruise_speed']:.0f} b={r['blend_radius']:.1f} t={r['threshold']:.1f} → {r['gates_passed']}/4 gates")

    # Save full results
    results_path = os.path.join(BASE_DIR, "logs/sweep_results_v3.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results: {results_path}")

    # Best result
    if successful:
        best = successful[0]
        print(f"\n  ★ BEST: {best['lap_time']:.2f}s @ v={best['cruise_speed']} b={best['blend_radius']} t={best['threshold']}")

    print()


if __name__ == "__main__":
    main()
