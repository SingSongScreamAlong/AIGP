"""Screen-capture DCL The Game on Windows for YOLO retraining.

Uses ffmpeg's GDI or dshow screen capture on Windows.
Drop-in replacement for capture_gameplay.py (macOS version).

Usage:
    python capture_gameplay_win.py --duration 120 --fps 15
    python capture_gameplay_win.py --list-devices

Output: data/dcl_raw/<timestamp>.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RAW_DIR = _REPO / "data" / "dcl_raw"


def list_devices():
    """Print available capture devices (Windows dshow)."""
    cmd = ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)


def record(
    duration_s: float = 120.0,
    fps: int = 15,
    output: str | None = None,
    title: str | None = None,
) -> Path:
    """Record screen to MP4 using ffmpeg gdigrab.

    Args:
        duration_s: Recording duration in seconds.
        fps: Capture frame rate.
        output: Output path. Auto-generated if None.
        title: Window title to capture. None = full desktop.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(_RAW_DIR / f"dcl_{ts}.mp4")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command for Windows screen capture
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "gdigrab",
        "-framerate", str(fps),
    ]

    if title:
        cmd += ["-i", f"title={title}"]
    else:
        cmd += ["-i", "desktop"]

    cmd += [
        "-t", str(duration_s),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output,
    ]

    print(f"Recording {'window: ' + title if title else 'full desktop'}")
    print(f"Duration: {duration_s}s at {fps} fps")
    print(f"Output: {output}")
    print(f"Press Ctrl+C to stop early.\n")

    try:
        proc = subprocess.run(cmd, timeout=duration_s + 30)
        if proc.returncode != 0:
            print(f"ffmpeg exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except subprocess.TimeoutExpired:
        print("\nRecording timed out.")

    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")

    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Record DCL gameplay on Windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=120.0,
                   help="Recording duration (seconds).")
    p.add_argument("--fps", type=int, default=15,
                   help="Capture frame rate.")
    p.add_argument("--title", default=None,
                   help="Window title to capture (e.g. 'DCL - The Game'). "
                        "Omit for full desktop.")
    p.add_argument("--output", default=None,
                   help="Output file path.")
    p.add_argument("--list-devices", action="store_true",
                   help="List capture devices and exit.")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    record(
        duration_s=args.duration,
        fps=args.fps,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
