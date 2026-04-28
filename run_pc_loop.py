#!/usr/bin/env python3
"""Autonomous 24/7 PC loop: capture DCL → extract → label → train → repeat.

Run this on your Windows PC with GPU while DCL The Game is open and flying.
It will continuously:
  1. Record DCL gameplay (screen capture via ffmpeg)
  2. Extract & deduplicate frames
  3. Pre-label gates with existing YOLO model
  4. Build YOLO dataset
  5. Retrain the YOLO model on accumulated data
  6. Sleep, then loop back to step 1

The model improves every cycle. Each cycle's results are logged.

Setup (one-time on PC):
    pip install ultralytics opencv-python numpy torch torchvision
    # Make sure ffmpeg is on PATH (choco install ffmpeg, or download)
    # Have DCL running in windowed mode

Usage:
    python run_pc_loop.py                          # default: 2min capture, loop forever
    python run_pc_loop.py --capture-duration 300   # 5 min capture per cycle
    python run_pc_loop.py --cycles 5               # run 5 cycles then stop
    python run_pc_loop.py --skip-capture            # label+train only (frames already exist)
    python run_pc_loop.py --train-only              # just retrain on existing dataset

Results: pc_loop_results/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_DATA_RAW = _REPO / "data" / "dcl_raw"
_DATA_FRAMES = _REPO / "data" / "dcl_frames"
_DATASET_DIR = _REPO / "datasets" / "dcl_real"
_MODELS_DIR = _REPO / "models"

sys.path.insert(0, str(_REPO / "src"))


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(log_file: Path, entry: dict):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ── Step 1: Capture ──────────────────────────────────────────────

def capture_dcl(duration_s: float, fps: int, window_title: str | None) -> Path | None:
    """Record DCL gameplay via ffmpeg."""
    _DATA_RAW.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    output = _DATA_RAW / f"dcl_{ts}.mp4"

    is_windows = platform.system() == "Windows"

    if is_windows:
        cmd = [
            "ffmpeg", "-y",
            "-f", "gdigrab",
            "-framerate", str(fps),
        ]
        if window_title:
            cmd += ["-i", f"title={window_title}"]
        else:
            cmd += ["-i", "desktop"]
    else:
        # macOS fallback
        cmd = [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-framerate", str(fps),
            "-i", "3:",
        ]

    cmd += [
        "-t", str(duration_s),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    print(f"  Recording {duration_s}s at {fps} fps → {output.name}")
    try:
        subprocess.run(cmd, timeout=duration_s + 60, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install it: choco install ffmpeg")
        return None

    if output.exists() and output.stat().st_size > 10000:
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"  Captured: {output.name} ({size_mb:.1f} MB)")
        return output
    else:
        print(f"  WARNING: Capture failed or too small")
        return None


# ── Step 2: Extract frames ───────────────────────────────────────

def extract_frames(video_path: Path, fps: float = 2.0) -> Path | None:
    """Extract and deduplicate frames from a video."""
    from vision.dcl_capture.extract_frames import extract
    out_dir = _DATA_FRAMES / video_path.stem
    n = extract(
        video_path, output_dir=out_dir,
        extract_fps=fps, dedup_threshold=0.92,
        target_size=(640, 480),
    )
    if n > 0:
        return out_dir
    return None


# ── Step 3: Label with YOLO ──────────────────────────────────────

def label_frames(frames_dir: Path, model_path: str, conf: float = 0.25) -> int:
    """Run existing YOLO model on frames for pre-labeling."""
    from vision.dcl_capture.label_assist import prelabel_with_model
    return prelabel_with_model(
        frames_dir, model_path=model_path,
        conf_threshold=conf, save_viz=True,
    )


# ── Step 4: Build dataset ────────────────────────────────────────

def build_dataset(frames_dir: Path) -> Path | None:
    """Build YOLO dataset from labeled frames."""
    from vision.dcl_capture.label_assist import build_dataset as _build
    _build(frames_dir, output_dir=_DATASET_DIR, train_split=0.85)
    yaml_path = _DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    return None


def accumulate_to_dataset(frames_dir: Path) -> int:
    """Add newly labeled frames to the existing accumulated dataset.
    
    Instead of rebuilding from scratch each cycle, this appends
    new frames to the existing train/valid splits.
    """
    import random

    labels_dir = frames_dir / "labels"
    if not labels_dir.exists():
        return 0

    img_exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in frames_dir.iterdir() if f.suffix.lower() in img_exts]
    pairs = []
    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            pairs.append((img, lbl))

    if not pairs:
        return 0

    # Ensure dataset dirs exist
    for split in ("train", "valid"):
        (_DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (_DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    added = 0
    for img_path, lbl_path in pairs:
        split = "train" if random.random() < 0.85 else "valid"
        # Use unique name to avoid collisions
        uid = f"{frames_dir.name}_{img_path.stem}"
        dst_img = _DATASET_DIR / split / "images" / (uid + img_path.suffix)
        dst_lbl = _DATASET_DIR / split / "labels" / (uid + ".txt")

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            added += 1

    # Write / update data.yaml
    abs_out = str(_DATASET_DIR.resolve())
    n_train = len(list((_DATASET_DIR / "train" / "images").iterdir()))
    n_valid = len(list((_DATASET_DIR / "valid" / "images").iterdir()))
    yaml_content = f"""# DCL Real Gate Detection Dataset (accumulated)
# Auto-built by run_pc_loop.py
# {n_train + n_valid} images ({n_train} train, {n_valid} val)

train: {abs_out}/train/images
val: {abs_out}/valid/images

nc: 1
names: ['gate']
"""
    with open(_DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"  Accumulated: +{added} new frames ({n_train} train, {n_valid} val total)")
    return added


# ── Step 5: Train YOLO ───────────────────────────────────────────

def train_yolo(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = "runs/detect",
    name: str = "dcl_gate",
) -> str | None:
    """Train YOLOv8 on the accumulated dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed. Run: pip install ultralytics")
        return None

    print(f"  Training YOLOv8 on {data_yaml}")
    print(f"  Base: {base_model}, Epochs: {epochs}, Batch: {batch_size}")

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project,
        name=name,
        exist_ok=True,
        patience=15,
        save=True,
        val=True,
        plots=True,
    )

    # Find best weights
    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        # Copy to models/
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dst = _MODELS_DIR / "gate_detector_dcl.pt"
        shutil.copy2(best_path, dst)
        print(f"  Best model saved: {dst}")
        return str(dst)
    else:
        print(f"  WARNING: best.pt not found at {best_path}")
        return None


# ── Main loop ────────────────────────────────────────────────────

def run_cycle(
    cycle_num: int,
    capture_duration: float,
    capture_fps: int,
    window_title: str | None,
    model_path: str,
    label_conf: float,
    train_epochs: int,
    train_batch: int,
    skip_capture: bool,
    train_only: bool,
    log_file: Path,
) -> dict:
    """Run one complete capture→label→train cycle."""
    result = {
        "cycle": cycle_num,
        "started": _timestamp(),
        "capture": None,
        "frames_extracted": 0,
        "frames_labeled": 0,
        "frames_accumulated": 0,
        "model_path": None,
        "errors": [],
    }

    t0 = time.time()

    if not train_only:
        # Step 1: Capture
        video = None
        if not skip_capture:
            print(f"\n  [1/4] Capturing DCL gameplay ({capture_duration}s)...")
            video = capture_dcl(capture_duration, capture_fps, window_title)
            if video:
                result["capture"] = str(video)
            else:
                result["errors"].append("capture_failed")
                return result

        # Step 2: Extract frames
        print(f"  [2/4] Extracting frames...")
        if video:
            frames_dir = extract_frames(video, fps=2.0)
        else:
            # Find most recent frames dir
            if _DATA_FRAMES.exists():
                dirs = sorted(_DATA_FRAMES.iterdir())
                frames_dir = dirs[-1] if dirs else None
            else:
                frames_dir = None

        if frames_dir:
            n_frames = len([f for f in frames_dir.iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            result["frames_extracted"] = n_frames
            print(f"  Extracted: {n_frames} frames")
        else:
            result["errors"].append("no_frames")
            return result

        # Step 3: Label
        print(f"  [3/4] Labeling with {model_path}...")
        n_labeled = label_frames(frames_dir, model_path, label_conf)
        result["frames_labeled"] = n_labeled

        # Step 3b: Accumulate into dataset
        n_accum = accumulate_to_dataset(frames_dir)
        result["frames_accumulated"] = n_accum
    else:
        print(f"  Skipping capture/extract/label (--train-only)")

    # Step 4: Train
    data_yaml = _DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"  [4/4] Training YOLO...")
        new_model = train_yolo(
            str(data_yaml),
            base_model=model_path,
            epochs=train_epochs,
            batch_size=train_batch,
            name=f"dcl_gate_cycle{cycle_num:03d}",
        )
        if new_model:
            result["model_path"] = new_model
    else:
        print(f"  Skipping training — no dataset yet")
        result["errors"].append("no_dataset")

    result["wall_s"] = time.time() - t0
    result["finished"] = _timestamp()
    _log(log_file, result)
    return result


def main():
    p = argparse.ArgumentParser(
        description="Autonomous 24/7 DCL capture → train loop for Windows PC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cycles", type=int, default=0,
                   help="Number of cycles (0 = infinite).")
    p.add_argument("--capture-duration", type=float, default=120.0,
                   help="Seconds of gameplay to record per cycle.")
    p.add_argument("--capture-fps", type=int, default=15,
                   help="Capture frame rate.")
    p.add_argument("--window-title", default=None,
                   help="DCL window title for targeted capture.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Starting YOLO model for labeling. Updated each cycle.")
    p.add_argument("--label-conf", type=float, default=0.25,
                   help="YOLO confidence threshold for pre-labeling.")
    p.add_argument("--train-epochs", type=int, default=50,
                   help="Training epochs per cycle.")
    p.add_argument("--train-batch", type=int, default=16,
                   help="Training batch size.")
    p.add_argument("--pause", type=float, default=30.0,
                   help="Seconds to pause between cycles.")
    p.add_argument("--skip-capture", action="store_true",
                   help="Skip recording, use existing frames.")
    p.add_argument("--train-only", action="store_true",
                   help="Only retrain on existing dataset.")
    args = p.parse_args()

    # Output dir
    out_dir = _REPO / "pc_loop_results" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "loop_log.jsonl"

    print("=" * 60)
    print("  AUTONOMOUS DCL → YOLO TRAINING LOOP")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cycles: {'infinite' if args.cycles == 0 else args.cycles}")
    print(f"  Capture: {args.capture_duration}s at {args.capture_fps} fps")
    print(f"  Model: {args.model}")
    print(f"  Log: {log_file}")
    print("=" * 60)

    current_model = args.model
    cycle = 0
    total_frames = 0
    total_start = time.time()

    try:
        while True:
            cycle += 1
            if args.cycles > 0 and cycle > args.cycles:
                break

            print(f"\n{'='*60}")
            print(f"  CYCLE {cycle}" +
                  (f" / {args.cycles}" if args.cycles > 0 else ""))
            print(f"  Model: {current_model}")
            print(f"{'='*60}")

            result = run_cycle(
                cycle_num=cycle,
                capture_duration=args.capture_duration,
                capture_fps=args.capture_fps,
                window_title=args.window_title,
                model_path=current_model,
                label_conf=args.label_conf,
                train_epochs=args.train_epochs,
                train_batch=args.train_batch,
                skip_capture=args.skip_capture,
                train_only=args.train_only,
                log_file=log_file,
            )

            # Use newly trained model for next cycle's labeling
            if result.get("model_path"):
                current_model = result["model_path"]
                print(f"\n  Next cycle will use: {current_model}")

            total_frames += result.get("frames_accumulated", 0)
            wall = result.get("wall_s", 0)
            errors = result.get("errors", [])

            print(f"\n  Cycle {cycle} done in {wall:.0f}s")
            print(f"  Total accumulated frames: {total_frames}")
            if errors:
                print(f"  Errors: {errors}")

            if args.cycles > 0 and cycle >= args.cycles:
                break

            print(f"\n  Pausing {args.pause}s before next cycle...")
            time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    total_wall = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  LOOP COMPLETE")
    print(f"  Cycles: {cycle}")
    print(f"  Total frames accumulated: {total_frames}")
    print(f"  Total wall time: {total_wall/3600:.1f} hours")
    print(f"  Final model: {current_model}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
