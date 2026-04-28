"""Screen-capture DCL The Game footage for YOLO retraining.

Records a region of the screen (or full screen) while you fly in
DCL The Game, producing an MP4 file. Uses ffmpeg's AVFoundation
input on macOS — no additional dependencies required.

Usage:
    # Record full screen (display 1) at 30 fps for 60 seconds:
    python capture_gameplay.py --duration 60

    # Record at lower fps (saves disk, still plenty for training):
    python capture_gameplay.py --duration 120 --fps 10

    # Record a specific screen region (x,y,w,h):
    python capture_gameplay.py --duration 60 --crop 100,50,1280,720

    # Just list available capture devices:
    python capture_gameplay.py --list-devices

The output goes to data/dcl_raw/<timestamp>.mp4.
After recording, run extract_frames.py to pull training frames.

Prerequisites:
    - ffmpeg (brew install ffmpeg)
    - DCL The Game running in a window or fullscreen
    - macOS screen recording permission granted to Terminal
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # ai-grand-prix root
_RAW_DIR = _REPO / "data" / "dcl_raw"


def list_avfoundation_devices():
    """Print available AVFoundation capture devices."""
    cmd = [
        "ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "",
    ]
    # ffmpeg prints device list to stderr, exits with error code
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)


def record(
    duration_s: float = 60.0,
    fps: int = 30,
    device: str = "3",  # AVFoundation index; "3" = Capture screen 0 on this Mac
    output: str | None = None,
    crop: str | None = None,
) -> Path:
    """Record screen to MP4.

    Args:
        duration_s: Recording duration in seconds.
        fps: Capture frame rate.
        device: AVFoundation device index. Run --list-devices to find yours.
        output: Output path. Auto-generated if None.
        crop: Optional "x,y,w,h" crop region.

    Returns:
        Path to the recorded MP4 file.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(_RAW_DIR / f"dcl_{ts}.mp4")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                           # overwrite
        "-f", "avfoundation",
        "-framerate", str(fps),
        "-i", f"{device}:",             # screen device
        "-t", str(duration_s),
        "-c:v", "libx264",
        "-preset", "ultrafast",         # fast encoding, larger file
        "-crf", "18",                   # high quality
        "-pix_fmt", "yuv420p",
    ]

    # Optional crop filter
    if crop:
        parts = crop.split(",")
        if len(parts) != 4:
            print(f"ERROR: --crop must be x,y,w,h (got {crop!r})")
            sys.exit(1)
        x, y, w, h = parts
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]

    cmd.append(output)

    print(f"Recording screen for {duration_s}s at {fps} fps...")
    print(f"Output: {output}")
    print(f"Press Ctrl+C to stop early.\n")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        proc = subprocess.run(cmd, timeout=duration_s + 10)
        if proc.returncode != 0:
            print(f"ffmpeg exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except subprocess.TimeoutExpired:
        print("\nRecording timed out (ffmpeg may still be writing).")

    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nWARNING: Output file not found at {out_path}")

    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Record DCL The Game screen for YOLO training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=60.0,
                   help="Recording duration (seconds). Ctrl+C to stop early.")
    p.add_argument("--fps", type=int, default=30,
                   help="Capture frame rate. 10-15 is enough for training data.")
    p.add_argument("--device", default="3",
                   help="AVFoundation device index. Run --list-devices to find yours.")
    p.add_argument("--output", default=None,
                   help="Output file path. Auto-generated if not set.")
    p.add_argument("--crop", default=None,
                   help="Crop region as x,y,w,h (pixels). Omit for full screen.")
    p.add_argument("--list-devices", action="store_true",
                   help="List available capture devices and exit.")
    args = p.parse_args()

    if args.list_devices:
        list_avfoundation_devices()
        return

    record(
        duration_s=args.duration,
        fps=args.fps,
        device=args.device,
        output=args.output,
        crop=args.crop,
    )


if __name__ == "__main__":
    main()
