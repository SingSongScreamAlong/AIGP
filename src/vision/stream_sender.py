"""Fake VADR-TS-002 §4.6 vision stream sender for local testing.

Emits spec-compliant UDP packets so we can validate the receiver and the
downstream YOLO pipeline without the actual DCL simulator. The frames are
either supplied by the caller or synthesized (color bars + frame counter).
"""
from __future__ import annotations

import socket
import time
from typing import Iterable, Optional

import numpy as np

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from .stream_receiver import build_packets, DEFAULT_PORT
except ImportError:
    from stream_receiver import build_packets, DEFAULT_PORT  # type: ignore


def _synthetic_frame(idx: int, w: int = 640, h: int = 360) -> np.ndarray:
    """Generate a deterministic test frame: color bars + frame index text."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    bands = [
        (255, 255, 255),  # white
        (0, 255, 255),    # yellow
        (255, 255, 0),    # cyan
        (0, 255, 0),      # green
        (255, 0, 255),    # magenta
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
        (0, 0, 0),        # black
    ]
    bw = w // len(bands)
    for i, c in enumerate(bands):
        img[:, i * bw:(i + 1) * bw] = c
    if HAS_CV2:
        cv2.putText(
            img, f"frame {idx}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
        )
    return img


def stream_frames(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    frames: Optional[Iterable[np.ndarray]] = None,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 80,
    max_payload_size: int = 1400,
    start_time_ns: Optional[int] = None,
):
    """Stream frames to (host, port) as spec-compliant UDP packets.

    If `frames` is None, generates synthetic 640×360 color-bar frames.
    """
    if not HAS_CV2:
        raise RuntimeError("cv2 required for stream_sender")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / fps if fps > 0 else 0.0
    t0_ns = start_time_ns if start_time_ns is not None else time.time_ns()
    t0_wall = time.monotonic()

    if frames is None:
        def _gen():
            i = 0
            while max_frames is None or i < max_frames:
                yield _synthetic_frame(i)
                i += 1
        frames = _gen()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    sent = 0
    try:
        for idx, img in enumerate(frames):
            if max_frames is not None and idx >= max_frames:
                break
            ok, buf = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                continue
            jpeg = buf.tobytes()
            sim_time_ns = t0_ns + int((time.monotonic() - t0_wall) * 1e9)
            packets = build_packets(
                frame_id=idx & 0xFFFFFFFF,
                jpeg_bytes=jpeg,
                sim_time_ns=sim_time_ns,
                max_payload_size=max_payload_size,
            )
            for p in packets:
                sock.sendto(p, (host, port))
            sent += 1
            if period > 0:
                target = t0_wall + (idx + 1) * period
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        sock.close()
    return sent


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fake VADR-TS-002 §4.6 vision sender")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--frames", type=int, default=300, help="Total frames to send")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--mtu", type=int, default=1400,
                   help="Max JPEG bytes per UDP packet (excludes 24B header)")
    args = p.parse_args()

    print(f"Streaming {args.frames} synthetic frames @ {args.fps:.1f} Hz "
          f"to {args.host}:{args.port}")
    n = stream_frames(
        host=args.host, port=args.port, fps=args.fps,
        max_frames=args.frames, jpeg_quality=args.quality,
        max_payload_size=args.mtu,
    )
    print(f"Sent {n} frames.")
