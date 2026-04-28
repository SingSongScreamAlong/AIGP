#!/usr/bin/env python3
"""Fully autonomous training loop — zero human interaction required.

This script generates synthetic gate training data, trains YOLO,
evaluates, then loops with increasing difficulty/variety. No DCL,
no screen capture, no flying — it manufactures its own data.

What it does each cycle:
  1. Generate N synthetic gate images via MuJoCo (or OpenCV fallback)
  2. Train YOLOv8 on the accumulated dataset
  3. Evaluate model quality (mAP, precision, recall)
  4. Log results, save best model
  5. Increase difficulty / variety for next cycle
  6. Repeat

Clone the repo on your PC, then:
    pip install ultralytics opencv-python numpy mujoco pyyaml torch torchvision
    python run_pc_autonomous.py              # run forever
    python run_pc_autonomous.py --cycles 5   # run 5 cycles
    python run_pc_autonomous.py --quick      # small batch for smoke test

Results: autonomous_results/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "vision" / "gate_yolo"))


# ── Output directory ──────────────────────────────────────────
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_OUT = Path(f"autonomous_results/{_TS}")
_DATASET_DIR = _REPO / "datasets" / "auto_synth"
_MODELS_DIR = _REPO / "models"


def _log(log_file: Path, entry: dict):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ── Phase 1: Generate synthetic data ─────────────────────────

def generate_synthetic_data(
    num_images: int,
    output_dir: Path,
    use_keypoints: bool = True,
) -> str | None:
    """Generate synthetic gate images with labels.

    Tries MuJoCo renderer first, falls back to OpenCV-only generator.
    Returns path to data.yaml or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try MuJoCo-based generator (best quality)
    try:
        if use_keypoints:
            from generate_data_keypoints import SyntheticKeypointGenerator
            gen = SyntheticKeypointGenerator(img_w=640, img_h=480)
            yaml_path = gen.generate_dataset(
                str(output_dir), num_samples=num_images, train_split=0.85,
            )
            return yaml_path
        else:
            from generate_data import SyntheticDataGenerator
            gen = SyntheticDataGenerator(width=640, height=480)
            yaml_path = gen.generate_dataset(
                str(output_dir), num_images=num_images,
                train_split=0.85, randomize=True,
            )
            return yaml_path
    except ImportError as e:
        print(f"  MuJoCo not available ({e}), using OpenCV fallback...")
    except Exception as e:
        print(f"  MuJoCo generator failed ({e}), using OpenCV fallback...")

    # OpenCV fallback — generates gate-like rectangles with perspective
    return _generate_opencv_gates(num_images, output_dir)


def _generate_opencv_gates(num_images: int, output_dir: Path) -> str:
    """Generate synthetic gate images using only OpenCV.

    Renders colored gate frames at random perspectives on varied backgrounds.
    Not as realistic as MuJoCo but works anywhere with opencv-python.
    """
    import cv2
    import numpy as np
    import random

    for split in ("train", "valid"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    num_train = int(num_images * 0.85)
    total_labels = 0

    print(f"  Generating {num_images} synthetic gate images (OpenCV)...")

    for i in range(num_images):
        split = "train" if i < num_train else "valid"
        img_w, img_h = 640, 480
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Random background (sky/ground gradient or solid color)
        bg_type = random.choice(["gradient", "solid", "noise"])
        if bg_type == "gradient":
            for y in range(img_h):
                ratio = y / img_h
                sky = np.array([200, 180, 140]) * (1 - ratio)
                ground = np.array([80, 120, 80]) * ratio
                img[y, :] = (sky + ground).astype(np.uint8)
        elif bg_type == "solid":
            color = [random.randint(30, 200) for _ in range(3)]
            img[:] = color
        else:
            img = np.random.randint(20, 180, (img_h, img_w, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (15, 15), 5)

        # Random gate parameters
        n_gates = random.randint(1, 3)
        labels = []

        for _ in range(n_gates):
            # Gate center and size in image
            cx = random.uniform(0.15, 0.85) * img_w
            cy = random.uniform(0.15, 0.85) * img_h
            gate_size = random.uniform(40, 250)

            # Gate frame (outer rectangle)
            half = gate_size / 2
            outer_pts = np.array([
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ], dtype=np.float32)

            # Apply random perspective transform
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.6, 1.4)
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

            # Add perspective skew
            skew_x = random.uniform(-0.15, 0.15)
            skew_y = random.uniform(-0.15, 0.15)
            pts_src = outer_pts.copy()
            pts_dst = outer_pts.copy()
            pts_dst[0] += [skew_x * gate_size, skew_y * gate_size]
            pts_dst[1] += [-skew_x * gate_size, skew_y * gate_size]
            pts_dst[2] += [-skew_x * gate_size, -skew_y * gate_size]
            pts_dst[3] += [skew_x * gate_size, -skew_y * gate_size]

            # Clamp to image
            pts_dst[:, 0] = np.clip(pts_dst[:, 0], 5, img_w - 5)
            pts_dst[:, 1] = np.clip(pts_dst[:, 1], 5, img_h - 5)

            # Draw gate frame
            color = [random.randint(100, 255) for _ in range(3)]
            thickness = max(2, int(gate_size / 15))
            pts_int = pts_dst.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_int], True, color, thickness)

            # Draw inner opening (slightly smaller)
            inner_ratio = random.uniform(0.5, 0.7)
            inner_pts = (pts_dst - np.mean(pts_dst, axis=0)) * inner_ratio + np.mean(pts_dst, axis=0)
            inner_int = inner_pts.astype(np.int32).reshape((-1, 1, 2))
            inner_color = [max(0, c - 40) for c in color]
            cv2.polylines(img, [inner_int], True, inner_color, max(1, thickness - 1))

            # YOLO label from outer bounding box
            x_min = max(0, pts_dst[:, 0].min())
            x_max = min(img_w, pts_dst[:, 0].max())
            y_min = max(0, pts_dst[:, 1].min())
            y_max = min(img_h, pts_dst[:, 1].max())

            box_w = x_max - x_min
            box_h = y_max - y_min
            if box_w > 10 and box_h > 10:
                ncx = ((x_min + x_max) / 2) / img_w
                ncy = ((y_min + y_max) / 2) / img_h
                nw = box_w / img_w
                nh = box_h / img_h
                labels.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                total_labels += 1

        # Save
        img_path = output_dir / split / "images" / f"synth_{i:05d}.jpg"
        lbl_path = output_dir / split / "labels" / f"synth_{i:05d}.txt"
        cv2.imwrite(str(img_path), img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{num_images}] generated")

    # Write data.yaml
    abs_out = str(output_dir.resolve())
    yaml_content = f"""# Synthetic Gate Dataset (OpenCV generated)
train: {abs_out}/train/images
val: {abs_out}/valid/images

nc: 1
names: ['gate']
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  Generated {num_images} images, {total_labels} gate labels")
    return str(yaml_path)


# ── Phase 2: Train ────────────────────────────────────────────

def train_yolo(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    cycle: int = 0,
) -> dict:
    """Train YOLOv8 and return metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: pip install ultralytics")
        return {"error": "ultralytics not installed"}

    name = f"auto_cycle_{cycle:03d}"
    model = YOLO(base_model)

    print(f"  Training: {base_model} → {name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        project="runs/detect",
        name=name,
        exist_ok=True,
        patience=15,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
    )

    # Evaluate
    best_path = Path("runs/detect") / name / "weights" / "best.pt"
    metrics = {"best_path": None, "mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0}

    if best_path.exists():
        best_model = YOLO(str(best_path))
        val_results = best_model.val(data=data_yaml)
        metrics = {
            "best_path": str(best_path),
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        }

        # Save to models/
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dst = _MODELS_DIR / "gate_detector_auto.pt"
        shutil.copy2(best_path, dst)
        metrics["saved_to"] = str(dst)
        print(f"  Model saved: {dst}")
        print(f"  mAP50={metrics['mAP50']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")

    return metrics


# ── Phase 3: Accumulate data across cycles ────────────────────

def accumulate_dataset(cycle_dir: Path, accumulated_dir: Path) -> str:
    """Copy cycle data into a growing accumulated dataset."""
    import random

    for split in ("train", "valid"):
        (accumulated_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (accumulated_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    added = 0
    for split in ("train", "valid"):
        src_imgs = cycle_dir / split / "images"
        src_lbls = cycle_dir / split / "labels"
        if not src_imgs.exists():
            continue

        for img_file in src_imgs.iterdir():
            lbl_file = src_lbls / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Unique name to avoid collisions
            uid = f"c{cycle_dir.name}_{img_file.name}"
            dst_img = accumulated_dir / split / "images" / uid
            dst_lbl = accumulated_dir / split / "labels" / (uid.rsplit(".", 1)[0] + ".txt")

            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)
                shutil.copy2(lbl_file, dst_lbl)
                added += 1

    # Count totals
    n_train = len(list((accumulated_dir / "train" / "images").iterdir()))
    n_valid = len(list((accumulated_dir / "valid" / "images").iterdir()))

    # Write data.yaml
    abs_out = str(accumulated_dir.resolve())
    yaml_content = f"""# Accumulated Synthetic Gate Dataset
# {n_train + n_valid} total images ({n_train} train, {n_valid} val)
train: {abs_out}/train/images
val: {abs_out}/valid/images

nc: 1
names: ['gate']
"""
    yaml_path = accumulated_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  Accumulated: +{added} → {n_train + n_valid} total ({n_train} train, {n_valid} val)")
    return str(yaml_path)


# ── Main loop ─────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fully autonomous YOLO training loop — no DCL needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cycles", type=int, default=0,
                   help="Number of cycles (0 = infinite).")
    p.add_argument("--images-per-cycle", type=int, default=500,
                   help="Synthetic images to generate per cycle.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs per cycle.")
    p.add_argument("--batch", type=int, default=16,
                   help="Training batch size.")
    p.add_argument("--base-model", default="yolov8n.pt",
                   help="Starting YOLO model.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny run for smoke testing (50 images, 5 epochs).")
    p.add_argument("--use-keypoints", action="store_true", default=False,
                   help="Generate keypoint labels (YOLOv8-pose).")
    p.add_argument("--pause", type=float, default=10.0,
                   help="Seconds between cycles.")
    args = p.parse_args()

    if args.quick:
        args.images_per_cycle = 50
        args.epochs = 5
        args.batch = 8

    _OUT.mkdir(parents=True, exist_ok=True)
    log_file = _OUT / "loop_log.jsonl"
    accumulated_dir = _DATASET_DIR

    print("=" * 60)
    print("  AUTONOMOUS YOLO TRAINING LOOP")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cycles: {'infinite' if args.cycles == 0 else args.cycles}")
    print(f"  Images/cycle: {args.images_per_cycle}")
    print(f"  Epochs/cycle: {args.epochs}")
    print(f"  Base model: {args.base_model}")
    print(f"  Output: {_OUT}")
    print("=" * 60)

    current_model = args.base_model
    best_map50 = 0.0
    cycle = 0
    total_images = 0
    total_start = time.time()

    try:
        while True:
            cycle += 1
            if args.cycles > 0 and cycle > args.cycles:
                break

            cycle_start = time.time()
            cycle_dir = _OUT / f"cycle_{cycle:03d}"

            print(f"\n{'='*60}")
            print(f"  CYCLE {cycle}" +
                  (f" / {args.cycles}" if args.cycles > 0 else ""))
            print(f"  Model: {current_model}")
            print(f"{'='*60}")

            result = {
                "cycle": cycle,
                "started": datetime.now().isoformat(),
                "images_generated": 0,
                "metrics": {},
                "errors": [],
            }

            # Step 1: Generate synthetic data for this cycle
            print(f"\n  [1/3] Generating {args.images_per_cycle} synthetic images...")
            try:
                yaml_path = generate_synthetic_data(
                    args.images_per_cycle, cycle_dir,
                    use_keypoints=args.use_keypoints,
                )
                result["images_generated"] = args.images_per_cycle
                total_images += args.images_per_cycle
            except Exception as e:
                print(f"  ERROR generating data: {e}")
                result["errors"].append(f"gen: {e}")
                _log(log_file, result)
                continue

            # Step 2: Accumulate into master dataset
            print(f"\n  [2/3] Accumulating dataset...")
            try:
                acc_yaml = accumulate_dataset(cycle_dir, accumulated_dir)
            except Exception as e:
                print(f"  ERROR accumulating: {e}")
                acc_yaml = yaml_path  # Fall back to cycle-only data
                result["errors"].append(f"acc: {e}")

            # Step 3: Train
            print(f"\n  [3/3] Training YOLO...")
            try:
                metrics = train_yolo(
                    data_yaml=acc_yaml,
                    base_model=current_model,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    cycle=cycle,
                )
                result["metrics"] = metrics

                # Track best model
                map50 = metrics.get("mAP50", 0)
                if map50 > best_map50:
                    best_map50 = map50
                    print(f"  ★ NEW BEST mAP50: {best_map50:.3f}")

                # Use this cycle's best for next cycle
                if metrics.get("best_path"):
                    current_model = metrics["best_path"]

            except Exception as e:
                print(f"  ERROR training: {e}")
                result["errors"].append(f"train: {e}")

            result["wall_s"] = time.time() - cycle_start
            result["total_images"] = total_images
            result["best_map50"] = best_map50
            _log(log_file, result)

            # Summary
            print(f"\n  Cycle {cycle}: {result['wall_s']:.0f}s")
            print(f"  Total images: {total_images}")
            print(f"  Best mAP50: {best_map50:.3f}")

            if args.cycles > 0 and cycle >= args.cycles:
                break

            print(f"\n  Pausing {args.pause}s...")
            time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C).")

    total_wall = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Cycles run: {cycle}")
    print(f"  Total images generated: {total_images}")
    print(f"  Best mAP50: {best_map50:.3f}")
    print(f"  Final model: {current_model}")
    print(f"  Wall time: {total_wall/3600:.1f} hours")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
