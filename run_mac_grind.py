#!/usr/bin/env python3
"""Unattended Mac grind — generates synthetic data + trains YOLO on CPU.

Run this and walk away. It will:
  Phase A: Generate 5000+ synthetic gate images (MuJoCo 3D renders)
  Phase B: Train YOLOv8n on CPU (slow but steady — ~2-3 hours per cycle)
  Phase C: Run massive PID parameter sweep (10,000+ configs)
  Phase D: Push results to git so PC can pull trained models

Each phase produces real artifacts:
  - datasets/auto_synth/  → training data ready for PC GPU training
  - models/gate_detector_cpu.pt → trained model (can bootstrap PC training)
  - mac_grind_results/ → detailed logs + metrics

Usage:
    source .venv/bin/activate
    python run_mac_grind.py                  # run all phases
    python run_mac_grind.py --phase data     # just generate data
    python run_mac_grind.py --phase train    # just train
    python run_mac_grind.py --phase sweep    # just PID sweep
    python run_mac_grind.py --loop           # loop all phases forever
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_OUT = _REPO / "mac_grind_results" / _TS
_DATASET_DIR = _REPO / "datasets" / "auto_synth"
_MODELS_DIR = _REPO / "models"

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "vision" / "gate_yolo"))


def _log(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(f"  {msg}")


# ══════════════════════════════════════════════════════════════
# PHASE A: Synthetic Data Generation
# ══════════════════════════════════════════════════════════════

def phase_data(num_images: int = 5000, use_keypoints: bool = True) -> str | None:
    """Generate massive synthetic gate dataset using MuJoCo."""
    log = _OUT / "data_gen.log"
    _log(log, f"Starting data generation: {num_images} images")
    _log(log, f"Keypoints: {use_keypoints}")

    _DATASET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if use_keypoints:
            from generate_data_keypoints import SyntheticKeypointGenerator
            gen = SyntheticKeypointGenerator(img_w=640, img_h=480, fov=90.0)
            yaml_path = gen.generate_dataset(
                str(_DATASET_DIR), num_samples=num_images, train_split=0.85,
            )
        else:
            from generate_data import SyntheticDataGenerator
            gen = SyntheticDataGenerator(width=640, height=480)
            yaml_path = gen.generate_dataset(
                str(_DATASET_DIR), num_images=num_images,
                train_split=0.85, randomize=True,
            )

        _log(log, f"Generated {num_images} images → {yaml_path}")
        return yaml_path

    except Exception as e:
        _log(log, f"MuJoCo failed: {e}")
        _log(log, "Falling back to OpenCV generator...")

        # Import the autonomous script's OpenCV fallback
        sys.path.insert(0, str(_REPO))
        from run_pc_autonomous import _generate_opencv_gates
        yaml_path = _generate_opencv_gates(num_images, _DATASET_DIR)
        _log(log, f"OpenCV fallback generated {num_images} images → {yaml_path}")
        return yaml_path


# ══════════════════════════════════════════════════════════════
# PHASE B: CPU YOLO Training
# ══════════════════════════════════════════════════════════════

def phase_train(
    data_yaml: str | None = None,
    epochs: int = 30,
    batch_size: int = 8,
    base_model: str = "yolov8n.pt",
) -> dict:
    """Train YOLOv8n on CPU. Slow but produces a real model."""
    log = _OUT / "train.log"

    if data_yaml is None:
        data_yaml = str(_DATASET_DIR / "data.yaml")
    if not Path(data_yaml).exists():
        _log(log, f"No dataset at {data_yaml}. Run --phase data first.")
        return {"error": "no dataset"}

    # Use MPS (Apple Silicon GPU) if available, else CPU
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        est = f"{epochs * 0.5}-{epochs * 1} minutes"
    else:
        device = "cpu"
        est = f"{epochs * 2}-{epochs * 4} minutes"

    _log(log, f"Training YOLOv8n on {device.upper()}")
    _log(log, f"  Data: {data_yaml}")
    _log(log, f"  Epochs: {epochs}, Batch: {batch_size}")
    _log(log, f"  Estimated time: {est} on M3 Pro")

    from ultralytics import YOLO

    model = YOLO(base_model)
    t0 = time.time()

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=str(_OUT),
        name="yolo_cpu",
        exist_ok=True,
        patience=10,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
    )

    wall = time.time() - t0
    _log(log, f"Training done in {wall/60:.1f} minutes")

    # Evaluate
    best_path = _OUT / "yolo_cpu" / "weights" / "best.pt"
    metrics = {"wall_s": wall, "best_path": None}

    if best_path.exists():
        best_model = YOLO(str(best_path))
        val_results = best_model.val(data=data_yaml)
        metrics.update({
            "best_path": str(best_path),
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        })

        # Copy to models/
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dst = _MODELS_DIR / "gate_detector_cpu.pt"
        shutil.copy2(best_path, dst)
        metrics["saved_to"] = str(dst)

        _log(log, f"  mAP50={metrics['mAP50']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
        _log(log, f"  Saved: {dst}")

    # Save metrics
    with open(_OUT / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# ══════════════════════════════════════════════════════════════
# PHASE C: Massive PID Sweep
# ══════════════════════════════════════════════════════════════

def phase_sweep(num_configs: int = 2000) -> dict:
    """Run fine-grained PID sweep across all courses and noise."""
    log = _OUT / "sweep.log"
    _log(log, f"Starting PID sweep: {num_configs} configurations")

    import asyncio
    import itertools
    import numpy as np
    from sim.mock_dcl import MockDCLAdapter
    from control.attitude_controller import AttitudeController, PIDGains
    from courses import get_course, list_courses
    from race_loop import RaceLoop

    courses = list_courses()
    noise_profiles = ["clean", "mild"]

    # Fine grid
    kp_range = np.linspace(2.0, 12.0, 15)
    ki_range = np.linspace(0.0, 1.5, 8)
    kd_range = np.linspace(0.0, 2.0, 8)
    speed_range = np.linspace(4.0, 14.0, 8)

    # Generate configs — sample if too many
    all_combos = list(itertools.product(kp_range, ki_range, kd_range))
    if len(all_combos) > num_configs:
        import random
        random.shuffle(all_combos)
        all_combos = all_combos[:num_configs]

    _log(log, f"  Grid: {len(all_combos)} PID combos × {len(courses)} courses × {len(noise_profiles)} noise")
    total_experiments = len(all_combos) * len(courses) * len(noise_profiles)
    _log(log, f"  Total experiments: {total_experiments}")

    results = []
    best_time = float("inf")
    best_config = None
    completed = 0
    t0 = time.time()

    async def run_one(kp, ki, kd, course_name, noise):
        gates = get_course(course_name)
        adapter = MockDCLAdapter()
        gains = PIDGains(kp=kp, ki=ki, kd=kd)
        ctrl = AttitudeController(
            vel_xy_gains=gains,
            vel_z_gains=PIDGains(kp=3.0, ki=0.5, kd=0.5),
        )
        from vision.detector import VirtualDetector
        det = VirtualDetector(gates, noise_profile=noise)
        loop = RaceLoop(
            adapter=adapter, detector=det, gates_ned=gates,
            attitude_controller=ctrl, dt=0.05,
        )
        try:
            result = await loop.run(timeout_s=90.0, realtime=False)
            return {
                "gates_passed": result.gates_passed,
                "total_gates": result.total_gates,
                "time_s": result.elapsed_s,
                "completed": result.gates_passed == result.total_gates,
            }
        except Exception as e:
            return {"error": str(e), "completed": False}

    for idx, (kp, ki, kd) in enumerate(all_combos):
        for course_name in courses:
            for noise in noise_profiles:
                r = asyncio.run(run_one(kp, ki, kd, course_name, noise))
                r.update({
                    "kp": float(kp), "ki": float(ki), "kd": float(kd),
                    "course": course_name, "noise": noise,
                })
                results.append(r)
                completed += 1

                if r.get("completed") and r["time_s"] < best_time:
                    best_time = r["time_s"]
                    best_config = (kp, ki, kd, course_name, noise)

                if completed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (total_experiments - completed) / rate
                    _log(log, f"  [{completed}/{total_experiments}] "
                         f"rate={rate:.0f}/s, ETA={eta:.0f}s, "
                         f"best={best_time:.1f}s")

    # Save results
    sweep_file = _OUT / "sweep_results.json"
    with open(sweep_file, "w") as f:
        json.dump(results, f, default=str)

    # Find top configs per course
    summary = {"total": total_experiments, "completed": completed}
    summary["best_overall"] = {
        "time_s": best_time,
        "config": best_config,
    }

    # Top 10 per course
    for course_name in courses:
        course_results = [r for r in results
                         if r.get("course") == course_name and r.get("completed")]
        course_results.sort(key=lambda r: r["time_s"])
        summary[f"top5_{course_name}"] = course_results[:5]

    with open(_OUT / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    _log(log, f"Sweep done: {completed} experiments in {elapsed:.0f}s")
    _log(log, f"Best: kp={best_config[0]:.1f} ki={best_config[1]:.1f} kd={best_config[2]:.1f} "
         f"→ {best_time:.1f}s on {best_config[3]}/{best_config[4]}")

    return summary


# ══════════════════════════════════════════════════════════════
# PHASE D: Git push results
# ══════════════════════════════════════════════════════════════

def phase_push():
    """Commit and push trained models + datasets so PC can pull."""
    log = _OUT / "push.log"

    # Only push the model file, not the massive dataset or results
    model_path = _MODELS_DIR / "gate_detector_cpu.pt"
    if not model_path.exists():
        _log(log, "No model to push yet.")
        return

    _log(log, "Pushing trained model to git...")
    try:
        subprocess.run(
            ["git", "add", str(model_path)],
            cwd=str(_REPO), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"[mac-grind] CPU-trained gate model {_TS}"],
            cwd=str(_REPO), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(_REPO), check=True, capture_output=True,
        )
        _log(log, "Pushed model to origin/main")
    except subprocess.CalledProcessError as e:
        _log(log, f"Git push failed: {e.stderr.decode() if e.stderr else e}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Unattended Mac grind: synth data + CPU training + PID sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["data", "train", "sweep", "all"],
                   default="all", help="Which phase to run.")
    p.add_argument("--loop", action="store_true",
                   help="Loop all phases forever.")
    p.add_argument("--images", type=int, default=5000,
                   help="Synthetic images to generate.")
    p.add_argument("--epochs", type=int, default=30,
                   help="YOLO training epochs.")
    p.add_argument("--sweep-configs", type=int, default=2000,
                   help="Number of PID configs to test.")
    p.add_argument("--no-push", action="store_true",
                   help="Skip git push.")
    args = p.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MAC GRIND — Unattended Progress Generator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Phase: {args.phase}")
    print(f"  Output: {_OUT}")
    print("=" * 60)

    cycle = 0
    try:
        while True:
            cycle += 1
            t0 = time.time()

            if args.phase in ("data", "all"):
                print(f"\n{'─'*40}")
                print(f"  PHASE A: Generate {args.images} synthetic images")
                print(f"{'─'*40}")
                phase_data(num_images=args.images)

            if args.phase in ("train", "all"):
                print(f"\n{'─'*40}")
                print(f"  PHASE B: Train YOLOv8n on CPU ({args.epochs} epochs)")
                print(f"{'─'*40}")
                phase_train(epochs=args.epochs)

            if args.phase in ("sweep", "all"):
                print(f"\n{'─'*40}")
                print(f"  PHASE C: PID Sweep ({args.sweep_configs} configs)")
                print(f"{'─'*40}")
                phase_sweep(num_configs=args.sweep_configs)

            if not args.no_push:
                phase_push()

            wall = time.time() - t0
            print(f"\n  Cycle {cycle} done in {wall/3600:.1f} hours")

            if not args.loop:
                break

            print(f"\n  Starting next cycle...")

    except KeyboardInterrupt:
        print("\n\nStopped.")

    print(f"\n{'='*60}")
    print(f"  MAC GRIND COMPLETE")
    print(f"  Cycles: {cycle}")
    print(f"  Results: {_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
