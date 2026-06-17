"""Stub: record a webcam clip to data/ as .mp4. Implemented in P2.

Usage (P2):
    python scripts/record_clip.py --out data/apple_demo.mp4 --seconds 15
"""

from __future__ import annotations

import argparse
from pathlib import Path


def record(out_path: str | Path, seconds: float = 15.0, fps: int = 30) -> Path:
    """Capture ``seconds`` of webcam video to ``out_path`` (.mp4 under data/).

    P2: implement with ``cv2.VideoCapture(0)`` + ``cv2.VideoWriter``.
    """
    raise NotImplementedError("scripts.record_clip.record is a P2 stub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a webcam clip into data/.")
    parser.add_argument("--out", default="data/clip.mp4")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    record(args.out, args.seconds, args.fps)
