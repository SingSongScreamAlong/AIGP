"""
Autonomous Training Loop
=========================
Continuously generates data and retrains, getting better each cycle.

Each cycle:
  1. Generate N new images (4 parallel batches)
  2. Merge everything into one big dataset
  3. Train YOLOv8 on the full dataset
  4. Log results
  5. Repeat

Usage:
    python auto_train_loop.py                    # Default: 500 imgs/batch, 4 batches, 30 epochs
    python auto_train_loop.py --cycles 10        # Run 10 cycles
    python auto_train_loop.py --imgs 1000        # 1000 images per batch
    python auto_train_loop.py --epochs 50        # 50 training epochs per cycle

Press Ctrl+C to stop gracefully after current cycle.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Process


def generate_batch(batch_id, num_images, output_dir):
    """Generate a batch of synthetic data (runs in subprocess)."""
    from generate_data import SyntheticDataGenerator
    gen = SyntheticDataGenerator(width=640, height=480)
    gen.generate_dataset(
        output_dir=output_dir,
        num_images=num_images,
        train_split=0.8,
        randomize=True,
    )


def run_generation_parallel(cycle, batches_per_cycle, imgs_per_batch, base_dir):
    """Run multiple data generation processes in parallel."""
    processes = []
    batch_dirs = []

    for b in range(batches_per_cycle):
        batch_dir = os.path.join(base_dir, f"cycle{cycle}_batch{b}")
        batch_dirs.append(batch_dir)

        p = Process(
            target=generate_batch,
            args=(b, imgs_per_batch, batch_dir),
        )
        p.start()
        processes.append(p)
        print(f"  Started generator {b+1}/{batches_per_cycle} -> {batch_dir}")

    # Wait for all to finish
    for p in processes:
        p.join()

    return batch_dirs


def merge_all_datasets(base_dir, output_dir):
    """Merge ALL batch directories into one unified dataset."""
    # Find all batch dirs
    base = Path(base_dir)
    batch_dirs = sorted([str(d) for d in base.iterdir()
                        if d.is_dir() and (d.name.startswith("cycle") or d.name.startswith("gates"))])

    if not batch_dirs:
        print("No batch directories found!")
        return None

    # Clean and rebuild merged dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    img_count = 0
    for batch_idx, batch_dir in enumerate(batch_dirs):
        for split in ["train", "valid"]:
            img_dir = os.path.join(batch_dir, split, "images")
            lbl_dir = os.path.join(batch_dir, split, "labels")
            if not os.path.exists(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if not fname.endswith((".jpg", ".png")):
                    continue
                stem = Path(fname).stem
                suffix = Path(fname).suffix
                new_name = f"b{batch_idx}_{stem}"
                shutil.copy2(
                    os.path.join(img_dir, fname),
                    os.path.join(output_dir, split, "images", new_name + suffix),
                )
                lbl_file = os.path.join(lbl_dir, stem + ".txt")
                if os.path.exists(lbl_file):
                    shutil.copy2(
                        lbl_file,
                        os.path.join(output_dir, split, "labels", new_name + ".txt"),
                    )
                img_count += 1

    abs_out = os.path.abspath(output_dir)
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {abs_out}/train/images\nval: {abs_out}/valid/images\nnc: 1\nnames: ['gate']\n")

    train_n = len(os.listdir(os.path.join(output_dir, "train", "images")))
    val_n = len(os.listdir(os.path.join(output_dir, "valid", "images")))
    print(f"  Merged {len(batch_dirs)} batches: {train_n} train + {val_n} val = {img_count} total")
    return yaml_path


def train_model(data_yaml, epochs, cycle, prev_model=None):
    """Train YOLOv8 and return metrics."""
    from ultralytics import YOLO

    # Start from pretrained or continue from previous best
    if prev_model and os.path.exists(prev_model):
        print(f"  Continuing from previous best: {prev_model}")
        model = YOLO(prev_model)
    else:
        print(f"  Starting from pretrained yolov8n.pt")
        model = YOLO("yolov8n.pt")

    name = f"gate_detector_cycle{cycle}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name=name,
        exist_ok=True,
        patience=15,
        device="cpu",
        workers=0,
        verbose=True,
    )

    # Find the best weights (handle nested path issue)
    best_path = None
    for root, dirs, files in os.walk("runs/detect"):
        if "best.pt" in files:
            candidate = os.path.join(root, "best.pt")
            if name in candidate:
                best_path = candidate
                break

    # Also check home directory
    if not best_path:
        home_path = os.path.expanduser(f"~/runs/detect/runs/detect/{name}/weights/best.pt")
        if os.path.exists(home_path):
            best_path = home_path

    if not best_path:
        home_path2 = os.path.expanduser(f"~/runs/detect/{name}/weights/best.pt")
        if os.path.exists(home_path2):
            best_path = home_path2

    # Validate
    metrics = None
    if best_path and os.path.exists(best_path):
        print(f"  Best model: {best_path}")
        val_model = YOLO(best_path)
        metrics = val_model.val(data=data_yaml)

        # Copy to a stable location
        stable_path = f"models/gate_detector_cycle{cycle}.pt"
        os.makedirs("models", exist_ok=True)
        shutil.copy2(best_path, stable_path)
        shutil.copy2(best_path, "models/gate_detector_latest.pt")
        print(f"  Saved to: {stable_path}")
    else:
        print(f"  WARNING: Could not find best.pt for {name}")

    return best_path, metrics


def log_cycle(cycle, total_images, metrics, elapsed, log_file="training_history.json"):
    """Append cycle results to a JSON log."""
    entry = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "elapsed_seconds": round(elapsed, 1),
    }

    if metrics:
        entry["mAP50"] = round(float(metrics.box.map50), 4)
        entry["mAP50_95"] = round(float(metrics.box.map), 4)
        entry["precision"] = round(float(metrics.box.mp), 4)
        entry["recall"] = round(float(metrics.box.mr), 4)

    # Load existing log or create new
    history = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            history = json.load(f)

    history.append(entry)

    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    return entry


def print_banner(cycle, entry):
    """Print a nice summary banner."""
    print()
    print("=" * 60)
    print(f"  CYCLE {cycle} COMPLETE")
    print("=" * 60)
    print(f"  Images:    {entry['total_images']}")
    print(f"  Time:      {entry['elapsed_seconds']:.0f}s")
    if 'mAP50' in entry:
        print(f"  mAP50:     {entry['mAP50']:.4f}")
        print(f"  mAP50-95:  {entry['mAP50_95']:.4f}")
        print(f"  Precision: {entry['precision']:.4f}")
        print(f"  Recall:    {entry['recall']:.4f}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Autonomous gate detection training loop")
    parser.add_argument("--cycles", type=int, default=999, help="Number of cycles to run (default: until stopped)")
    parser.add_argument("--batches", type=int, default=4, help="Parallel data generation batches per cycle")
    parser.add_argument("--imgs", type=int, default=500, help="Images per batch")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per cycle")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (delete all previous data)")
    args = parser.parse_args()

    data_dir = "datasets/auto"
    merged_dir = "datasets/auto_merged"

    if args.fresh and os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("Cleared previous data.")

    os.makedirs(data_dir, exist_ok=True)

    imgs_per_cycle = args.batches * args.imgs
    print()
    print("========================================")
    print("  AUTONOMOUS TRAINING LOOP")
    print("========================================")
    print(f"  Cycles:          {args.cycles}")
    print(f"  Batches/cycle:   {args.batches}")
    print(f"  Images/batch:    {args.imgs}")
    print(f"  Images/cycle:    {imgs_per_cycle}")
    print(f"  Epochs/cycle:    {args.epochs}")
    print(f"  Press Ctrl+C to stop after current cycle")
    print("========================================")
    print()

    prev_model = "models/gate_detector_latest.pt"
    if not os.path.exists(prev_model):
        prev_model = None

    for cycle in range(1, args.cycles + 1):
        t0 = time.time()
        print(f"\n{'#' * 60}")
        print(f"# CYCLE {cycle}")
        print(f"{'#' * 60}")

        # Step 1: Generate data
        print(f"\n[1/3] Generating {imgs_per_cycle} images ({args.batches} x {args.imgs})...")
        new_batches = run_generation_parallel(cycle, args.batches, args.imgs, data_dir)

        # Step 2: Merge everything
        print(f"\n[2/3] Merging all data...")
        data_yaml = merge_all_datasets(data_dir, merged_dir)
        if not data_yaml:
            print("ERROR: No data to train on!")
            break

        total_images = len(os.listdir(os.path.join(merged_dir, "train", "images"))) + \
                       len(os.listdir(os.path.join(merged_dir, "valid", "images")))

        # Step 3: Train
        print(f"\n[3/3] Training ({args.epochs} epochs on {total_images} images)...")
        best_path, metrics = train_model(data_yaml, args.epochs, cycle, prev_model)

        if best_path:
            prev_model = "models/gate_detector_latest.pt"

        elapsed = time.time() - t0

        # Log and display
        entry = log_cycle(cycle, total_images, metrics, elapsed)
        print_banner(cycle, entry)

        # Check if we should stop
        try:
            pass  # Continue to next cycle
        except KeyboardInterrupt:
            print("\nStopping after this cycle. Final model: models/gate_detector_latest.pt")
            break

    print("\nTraining loop finished.")
    print(f"Final model: models/gate_detector_latest.pt")
    print(f"History:     training_history.json")


if __name__ == "__main__":
    main()
