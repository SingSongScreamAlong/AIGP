"""VADR-TS-002 §4.6 Vision Stream Receiver.

Receives the simulator's JPEG video stream over UDP and reassembles frames
from the chunked little-endian packet protocol defined in the spec.

Packet layout (24-byte header, little-endian, followed by JPEG payload):
    frame_id      uint32  4B   Unique sequence ID for the image frame
    chunk_id      uint16  2B   Index of this packet within the frame
    total_chunks  uint16  2B   Total packets required to complete the frame
    jpeg_size     uint32  4B   Size of the full reconstructed JPEG in bytes
    payload_size  uint32  4B   Size of the JPEG slice in this packet
    sim_time_ns   uint64  8B   Simulation epoch timestamp (nanoseconds)

Stream parameters per spec:
    Transport     : UDP
    Default port  : 5600
    Byte order    : little-endian
    Frame rate    : 30 Hz
    Resolution    : 640 × 360

Usage:
    rx = VisionStreamReceiver()
    rx.start()
    while running:
        frame = rx.latest_frame(timeout_s=1.0)
        if frame is not None:
            jpeg_bytes, sim_time_ns, frame_id = frame
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8),
                               cv2.IMREAD_COLOR)
    rx.stop()
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# VADR-TS-002 §4.6
HEADER_SIZE = 24
HEADER_FORMAT = "<IHHIIQ"  # little-endian: u32 u16 u16 u32 u32 u64
DEFAULT_PORT = 5600
DEFAULT_BIND = "0.0.0.0"
# Max UDP payload we'll accept per packet (header + JPEG slice). 65507 is the
# IPv4 UDP theoretical max; in practice slices are usually <~1400 bytes (MTU).
MAX_PACKET = 65535


@dataclass
class _FrameBuffer:
    """In-progress frame being reassembled from chunks."""
    frame_id: int
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    chunks: dict = field(default_factory=dict)  # chunk_id -> bytes
    first_seen_s: float = field(default_factory=time.monotonic)

    def is_complete(self) -> bool:
        if len(self.chunks) != self.total_chunks:
            return False
        total = sum(len(b) for b in self.chunks.values())
        return total == self.jpeg_size

    def assemble(self) -> bytes:
        return b"".join(self.chunks[i] for i in range(self.total_chunks))


@dataclass
class StreamStats:
    packets_received: int = 0
    packets_dropped_malformed: int = 0
    frames_completed: int = 0
    frames_dropped_incomplete: int = 0
    frames_dropped_stale: int = 0
    bytes_received: int = 0


class VisionStreamReceiver:
    """Background-threaded UDP receiver for the spec vision stream.

    The receiver decodes the 24-byte little-endian header on every packet,
    accumulates JPEG slices per frame_id, and exposes the latest fully
    reassembled JPEG to the consumer. Incomplete frames older than
    `frame_timeout_s` are dropped.

    The class is intentionally agnostic about whether the consumer wants
    raw JPEG bytes or a decoded numpy frame; `latest_frame()` returns
    `(jpeg_bytes, sim_time_ns, frame_id)` and a convenience method
    `latest_decoded()` returns the decoded BGR numpy image.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        bind: str = DEFAULT_BIND,
        frame_timeout_s: float = 0.5,
        socket_timeout_s: float = 0.25,
        recv_buffer_bytes: int = 1 << 20,
    ):
        self._port = port
        self._bind = bind
        self._frame_timeout_s = frame_timeout_s
        self._socket_timeout_s = socket_timeout_s
        self._recv_buffer_bytes = recv_buffer_bytes

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Lock-protected state shared with consumer thread
        self._lock = threading.Lock()
        self._pending: "OrderedDict[int, _FrameBuffer]" = OrderedDict()
        self._latest: Optional[Tuple[bytes, int, int]] = None  # (jpeg, t_ns, fid)
        self._new_frame = threading.Event()
        self.stats = StreamStats()

    # ────────────────────── lifecycle ──────────────────────

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buffer_bytes
            )
        except OSError:
            pass  # not fatal if the kernel rejects an oversized buffer
        self._sock.settimeout(self._socket_timeout_s)
        self._sock.bind((self._bind, self._port))

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, name="VisionStreamReceiver", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ────────────────────── consumer API ──────────────────────

    def latest_frame(self, timeout_s: float = 0.0) -> Optional[Tuple[bytes, int, int]]:
        """Return the most recent reassembled (jpeg_bytes, sim_time_ns, frame_id).

        If `timeout_s > 0`, block up to that long waiting for a new frame.
        Returns None if no frame is available.
        """
        if timeout_s > 0:
            got = self._new_frame.wait(timeout=timeout_s)
            if not got:
                return None
        with self._lock:
            self._new_frame.clear()
            return self._latest

    def latest_decoded(self, timeout_s: float = 0.0) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return (BGR numpy image, sim_time_ns, frame_id) or None."""
        if not HAS_CV2:
            raise RuntimeError("cv2 required for latest_decoded()")
        item = self.latest_frame(timeout_s=timeout_s)
        if item is None:
            return None
        jpeg, t_ns, fid = item
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img, t_ns, fid

    # ────────────────────── receive loop ──────────────────────

    def _run(self):
        assert self._sock is not None
        while not self._stop_flag.is_set():
            try:
                pkt, _addr = self._sock.recvfrom(MAX_PACKET)
            except socket.timeout:
                self._purge_stale_frames()
                continue
            except OSError:
                break

            self.stats.packets_received += 1
            self.stats.bytes_received += len(pkt)

            if len(pkt) < HEADER_SIZE:
                self.stats.packets_dropped_malformed += 1
                continue

            try:
                frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = (
                    struct.unpack_from(HEADER_FORMAT, pkt, 0)
                )
            except struct.error:
                self.stats.packets_dropped_malformed += 1
                continue

            payload = pkt[HEADER_SIZE:HEADER_SIZE + payload_size]
            if len(payload) != payload_size:
                self.stats.packets_dropped_malformed += 1
                continue

            self._accept_chunk(
                frame_id, chunk_id, total_chunks, jpeg_size, sim_time_ns, payload
            )

    def _accept_chunk(self, frame_id, chunk_id, total_chunks, jpeg_size,
                      sim_time_ns, payload):
        with self._lock:
            fb = self._pending.get(frame_id)
            if fb is None:
                fb = _FrameBuffer(
                    frame_id=frame_id,
                    total_chunks=total_chunks,
                    jpeg_size=jpeg_size,
                    sim_time_ns=sim_time_ns,
                )
                self._pending[frame_id] = fb
            # Ignore duplicate chunks; first one wins.
            if chunk_id not in fb.chunks:
                fb.chunks[chunk_id] = payload

            if fb.is_complete():
                jpeg = fb.assemble()
                self._latest = (jpeg, fb.sim_time_ns, fb.frame_id)
                self._new_frame.set()
                self.stats.frames_completed += 1
                # Remove this frame and any older partials, which are now stale.
                stale_ids = [fid for fid in self._pending if fid <= frame_id]
                for fid in stale_ids:
                    if fid != frame_id:
                        self.stats.frames_dropped_incomplete += 1
                    del self._pending[fid]

    def _purge_stale_frames(self):
        now = time.monotonic()
        with self._lock:
            stale = [
                fid for fid, fb in self._pending.items()
                if (now - fb.first_seen_s) > self._frame_timeout_s
            ]
            for fid in stale:
                del self._pending[fid]
                self.stats.frames_dropped_stale += 1


# ──────────────────────────── packet builder ────────────────────────────
# Useful for tests, simulators, and the local fake sender.

def build_packets(
    frame_id: int,
    jpeg_bytes: bytes,
    sim_time_ns: int,
    max_payload_size: int = 1400,
) -> list[bytes]:
    """Split a JPEG into spec-compliant UDP packets.

    `max_payload_size` is the JPEG slice per packet (excludes the 24-byte
    header). 1400 keeps us under the typical 1500-byte Ethernet MTU.
    """
    if max_payload_size <= 0:
        raise ValueError("max_payload_size must be > 0")
    if not (0 <= frame_id <= 0xFFFFFFFF):
        raise ValueError("frame_id out of range for uint32")

    jpeg_size = len(jpeg_bytes)
    total_chunks = max(1, (jpeg_size + max_payload_size - 1) // max_payload_size)
    if total_chunks > 0xFFFF:
        raise ValueError("JPEG too large to split into uint16 chunks")

    packets = []
    for i in range(total_chunks):
        start = i * max_payload_size
        end = min(start + max_payload_size, jpeg_size)
        payload = jpeg_bytes[start:end]
        header = struct.pack(
            HEADER_FORMAT,
            frame_id,
            i,
            total_chunks,
            jpeg_size,
            len(payload),
            sim_time_ns,
        )
        packets.append(header + payload)
    return packets
