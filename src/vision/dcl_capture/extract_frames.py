"""Extract training frames from DCL gameplay recordings.

Takes an MP4 from capture_gameplay.py and pulls individual frames,
optionally deduplicating near-identical frames (e.g. menus, loading
screens, hover states) to keep only useful training data.

Usage:
    # Extract frames at 2 fps (1 frame every 0.5s):
    python extract_frames.py data/dcl_raw/dcl_20260428_153000.mp4

    # Extract at 5 fps with stricter dedup:
    python extract_frames.py video.mp4 --fps 5 --dedup-threshold 0.98

    # Process all recordings in data/dcl_raw/:
    python extract_frames.py --all

Output goes to data/dcl_frames/<video_stem>/ as numbered JPEGs.

Prerequisites:
    - pip install opencv-python numpy  (already available)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_RAW_DIR = _REPO / "data" / "dcl_raw"
_FRAMES_DIR = _REPO / "data" / "dcl_frames"


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalised correlation between two frames (0..1).
    
    Uses downscaled grayscale for speed. A value of 1.0 means
    identical frames; 0.95+ means nearly identical (menu, hover).
    """
    # Downsample for speed
    small_a = cv2.resize(a, (64, 48), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, (64, 48), interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY).astype(float).ravel()
    gb = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY).astype(float).ravel()
    ga -= ga.mean()
    gb -= gb.mean()
    denom = np.linalg.norm(ga) * np.linalg.norm(gb)
    if denom < 1e-9:
        return 1.0  # both blank
    return float(np.dot(ga, gb) / denom)


def extract(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    extract_fps: float = 2.0,
    dedup_threshold: float = 0.95,
    target_size: tuple[int, int] | None = (640, 480),
    max_frames: int = 10000,
) -> int:
    """Extract and deduplicate frames from a video.

    Args:
        video_path: Path to the MP4 file.
        output_dir: Where to save frames. Auto-generated if None.
        extract_fps: How many frames per second of video to extract.
        dedup_threshold: Skip frame if similarity to previous > this.
        target_size: Resize frames to (w, h). None = keep original.
        max_frames: Safety cap on extracted frames.

    Returns:
        Number of frames saved.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 0

    if output_dir is None:
        output_dir = _FRAMES_DIR / video_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    # Frame interval — extract every Nth frame
    frame_interval = max(1, int(video_fps / extract_fps))

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.1f}, Duration: {duration_s:.1f}s, "
          f"Frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames "
          f"(~{extract_fps:.1f} fps)")
    print(f"  Dedup threshold: {dedup_threshold}")
    if target_size:
        print(f"  Resizing to: {target_size[0]}x{target_size[1]}")
    print(f"  Output: {output_dir}")
    print()

    saved = 0
    skipped_dedup = 0
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        # Resize if requested
        if target_size:
            frame = cv2.resize(frame, target_size,
                               interpolation=cv2.INTER_AREA)

        # Deduplication
        if prev_frame is not None and dedup_threshold < 1.0:
            sim = frame_similarity(frame, prev_frame)
            if sim > dedup_threshold:
                skipped_dedup += 1
                frame_idx += 1
                continue

        # Save
        fname = f"dcl_{saved:05d}.jpg"
        cv2.imwrite(str(output_dir / fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        prev_frame = frame.copy()

        if saved % 50 == 0:
            print(f"  [{saved}] extracted, {skipped_dedup} deduped...")

        if saved >= max_frames:
            print(f"  Hit max_frames cap ({max_frames})")
            break

        frame_idx += 1

    cap.release()

    print(f"\nDone! Saved {saved} frames, skipped {skipped_dedup} "
          f"(dedup), from {frame_idx} total.")
    print(f"Output: {output_dir}")
    return saved


def main():
    p = argparse.ArgumentParser(
        description="Extract training frames from DCL gameplay recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", nargs="?", default=None,
                   help="Path to MP4 video. Use --all to process all in "
                        "data/dcl_raw/.")
    p.add_argument("--all", action="store_true",
                   help="Process all MP4s in data/dcl_raw/.")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Frames per second to extract.")
    p.add_argument("--dedup", type=float, default=0.95,
                   dest="dedup_threshold",
                   help="Similarity threshold for dedup (1.0 = no dedup).")
    p.add_argument("--size", default="640x480",
                   help="Resize to WxH. 'none' to keep original size.")
    p.add_argument("--max-frames", type=int, default=10000,
                   help="Max frames to extract per video.")
    p.add_argument("--output", default=None,
                   help="Output directory (auto-generated if not set).")
    args = p.parse_args()

    # Parse size
    target_size = None
    if args.size.lower() != "none":
        w, h = args.size.split("x")
        target_size = (int(w), int(h))

    if args.all:
        if not _RAW_DIR.exists():
            print(f"No recordings directory: {_RAW_DIR}")
            return
        videos = sorted(_RAW_DIR.glob("*.mp4"))
        if not videos:
            print(f"No MP4 files in {_RAW_DIR}")
            return
        total = 0
        for v in videos:
            n = extract(v, extract_fps=args.fps,
                        dedup_threshold=args.dedup_threshold,
                        target_size=target_size,
                        max_frames=args.max_frames)
            total += n
        print(f"\nAll done! {total} total frames from {len(videos)} videos.")
    elif args.video:
        extract(args.video, output_dir=args.output,
                extract_fps=args.fps,
                dedup_threshold=args.dedup_threshold,
                target_size=target_size,
                max_frames=args.max_frames)
    else:
        p.print_help()
        print("\nProvide a video path or use --all.")


if __name__ == "__main__":
    main()
