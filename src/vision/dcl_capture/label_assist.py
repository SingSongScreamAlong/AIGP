"""Semi-automatic gate labeling for DCL captured frames.

Two modes:
  1. **Model-assist** (--model): Runs an existing YOLO model on each
     frame and writes predicted bounding boxes as YOLO-format labels.
     You then review/correct in a labeling tool (e.g. Label Studio,
     CVAT, or the built-in review mode).

  2. **Manual review** (--review): Opens each frame with its
     pre-generated label overlay so you can accept, reject, or
     skip frames interactively.

Usage:
    # Pre-label frames using existing YOLO model:
    python label_assist.py data/dcl_frames/dcl_20260428/ \\
        --model models/gate_detector_latest.pt

    # Pre-label with keypoint model:
    python label_assist.py data/dcl_frames/dcl_20260428/ \\
        --model src/vision/gate_yolo/yolov8n.pt --conf 0.3

    # Interactive review of pre-labels:
    python label_assist.py data/dcl_frames/dcl_20260428/ --review

    # Convert pre-labels into a YOLO dataset:
    python label_assist.py data/dcl_frames/dcl_20260428/ --build-dataset

Prerequisites:
    - ultralytics (for --model mode)
    - opencv-python, numpy (always)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]


# ── YOLO format helpers ──────────────────────────────────────────

def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int,
                 img_w: int, img_h: int, class_id: int = 0) -> str:
    """Convert (x1, y1, x2, y2) pixel box to YOLO format string."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def yolo_to_xyxy(line: str, img_w: int, img_h: int) -> Tuple[int, int, int, int, int]:
    """Parse a YOLO label line back to (class_id, x1, y1, x2, y2)."""
    parts = line.strip().split()
    cls = int(parts[0])
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return cls, x1, y1, x2, y2


# ── Model-assist labeling ────────────────────────────────────────

def prelabel_with_model(
    frames_dir: Path,
    model_path: str,
    conf_threshold: float = 0.3,
    save_viz: bool = True,
) -> int:
    """Run YOLO on all frames and write pre-labels.

    Creates a `labels/` directory alongside `images/` in frames_dir
    with one .txt per frame in YOLO format.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run:")
        print("  pip install ultralytics")
        print("Or transfer frames to your GPU machine for labeling.")
        return 0

    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")

    # Find all images
    img_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in img_exts
    )
    if not images:
        print(f"No images found in {frames_dir}")
        return 0

    labels_dir = frames_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    viz_dir = frames_dir / "viz" if save_viz else None
    if viz_dir:
        viz_dir.mkdir(exist_ok=True)

    labeled = 0
    total_boxes = 0

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf_threshold, verbose=False)

        lines = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                yolo_line = xyxy_to_yolo(x1, y1, x2, y2, w, h, class_id=0)
                lines.append(yolo_line)
                total_boxes += 1

        # Write label file
        label_path = labels_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        # Optional visualization
        if viz_dir and lines:
            viz = frame.copy()
            for line in lines:
                _, bx1, by1, bx2, by2 = yolo_to_xyxy(line, w, h)
                cv2.rectangle(viz, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(viz, "gate", (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(viz_dir / img_path.name), viz)

        labeled += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(images)}] {total_boxes} boxes so far")

    print(f"\nPre-labeled {labeled} frames with {total_boxes} gate boxes.")
    print(f"Labels: {labels_dir}")
    if viz_dir:
        print(f"Visualizations: {viz_dir}")
    return labeled


# ── Interactive review ────────────────────────────────────────────

def review_labels(frames_dir: Path) -> None:
    """Interactive OpenCV review of pre-labeled frames.

    Controls:
      [a] Accept — keep this frame + label
      [r] Reject — delete label (bad detection)
      [s] Skip — move on without changing
      [q] Quit
    """
    labels_dir = frames_dir / "labels"
    if not labels_dir.exists():
        print(f"No labels directory at {labels_dir}. Run --model first.")
        return

    accepted_dir = frames_dir / "accepted"
    rejected_dir = frames_dir / "rejected"
    accepted_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in img_exts
    )

    print(f"Review mode: {len(images)} frames")
    print("  [a] Accept  [r] Reject  [s] Skip  [q] Quit")
    print()

    accepted = 0
    rejected = 0

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]

        # Draw labels if they exist
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    _, x1, y1, x2, y2 = yolo_to_xyxy(line, w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Review", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):
            # Accept: move to accepted/
            shutil.copy2(img_path, accepted_dir / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path,
                             accepted_dir / label_path.name)
            accepted += 1
        elif key == ord("r"):
            # Reject: move label to rejected, keep image
            if label_path.exists():
                shutil.move(str(label_path),
                            str(rejected_dir / label_path.name))
            rejected += 1
        elif key == ord("q"):
            break
        # 's' = skip, do nothing

    cv2.destroyAllWindows()
    print(f"\nReview done: {accepted} accepted, {rejected} rejected")


# ── Dataset builder ───────────────────────────────────────────────

def build_dataset(
    frames_dir: Path,
    output_dir: Path | None = None,
    train_split: float = 0.8,
) -> None:
    """Build a YOLO dataset from accepted/labeled frames.

    Creates the standard YOLO directory structure:
        output_dir/
        ├── data.yaml
        ├── train/images/  train/labels/
        └── valid/images/  valid/labels/
    """
    # Look for accepted/ subdir first, fall back to frames_dir itself
    src_dir = frames_dir / "accepted"
    if not src_dir.exists():
        src_dir = frames_dir

    labels_dir = src_dir / "labels" if (src_dir / "labels").exists() else frames_dir / "labels"

    img_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in img_exts)

    # Filter to images that have a non-empty label file
    pairs = []
    for img_path in images:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            pairs.append((img_path, lbl_path))

    if not pairs:
        print("No labeled images found. Run --model first, then optionally --review.")
        return

    if output_dir is None:
        output_dir = _REPO / "datasets" / "dcl_real"
    output_dir = Path(output_dir)

    # Shuffle and split
    random.shuffle(pairs)
    n_train = int(len(pairs) * train_split)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    for split, split_pairs in [("train", train_pairs), ("valid", val_pairs)]:
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / (img_path.stem + ".txt"))

    # Write data.yaml
    abs_out = str(output_dir.resolve())
    yaml_content = f"""# DCL The Game - Real Gate Detection Dataset
# Captured from DCL gameplay, labeled with YOLO model-assist
# {len(pairs)} images ({len(train_pairs)} train, {len(val_pairs)} val)

train: {abs_out}/train/images
val: {abs_out}/valid/images

nc: 1
names: ['gate']
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset built: {output_dir}")
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Valid: {len(val_pairs)} images")
    print(f"  data.yaml: {yaml_path}")
    print(f"\nTo train: merge with synthetic data using merge_datasets.py")
    print(f"  or train directly: python 02_train.py --data {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Semi-auto gate labeling for DCL captured frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("frames_dir",
                   help="Directory containing extracted frames.")
    p.add_argument("--model", default=None,
                   help="Path to YOLO model for pre-labeling.")
    p.add_argument("--conf", type=float, default=0.3,
                   help="YOLO confidence threshold for pre-labeling.")
    p.add_argument("--review", action="store_true",
                   help="Interactive review of pre-labels.")
    p.add_argument("--build-dataset", action="store_true",
                   dest="build_dataset",
                   help="Build YOLO dataset from accepted labels.")
    p.add_argument("--output", default=None,
                   help="Output directory for --build-dataset.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip visualization images during pre-labeling.")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"ERROR: Directory not found: {frames_dir}")
        sys.exit(1)

    if args.model:
        prelabel_with_model(
            frames_dir,
            model_path=args.model,
            conf_threshold=args.conf,
            save_viz=not args.no_viz,
        )
    elif args.review:
        review_labels(frames_dir)
    elif args.build_dataset:
        output = Path(args.output) if args.output else None
        build_dataset(frames_dir, output_dir=output)
    else:
        p.print_help()
        print("\nSpecify --model, --review, or --build-dataset.")


if __name__ == "__main__":
    main()
