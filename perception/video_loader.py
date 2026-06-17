"""Stub: load an egocentric .mp4 into frames. Implemented in P2."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def load_frames(video_path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield RGB frames (H, W, 3, uint8) from an .mp4 egocentric clip.

    Args:
        video_path: Path to the source .mp4.
        stride: Keep every ``stride``-th frame (1 = every frame).

    Yields:
        RGB frames as ``np.uint8`` arrays of shape ``(H, W, 3)``.

    P2: implement with OpenCV (``cv2.VideoCapture``), converting BGR -> RGB.
    """
    raise NotImplementedError("video_loader.load_frames is a P2 stub.")
