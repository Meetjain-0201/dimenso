"""Stub: frames -> 21-keypoint hand landmarks -> .npz. Implemented in P2."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def extract_hand_landmarks(frames: Iterable[np.ndarray], out_path: str | Path) -> Path:
    """Run MediaPipe Hands over frames and save landmarks to ``out_path`` (.npz).

    Args:
        frames: Iterable of RGB frames (H, W, 3, uint8).
        out_path: Destination .npz path under ``data/``.

    Returns:
        The path the landmarks were written to.

    The saved .npz is expected to contain at least:
        ``landmarks``: (T, num_hands, 21, 3) normalized image-space keypoints
        ``handedness``: (T, num_hands) Left/Right labels
        ``frame_idx``:  (T,) source frame indices

    P2: implement with ``mediapipe.python.solutions.hands`` (or the Tasks API and
    the on-disk ``hand_landmarker.task`` model), two hands, 30 fps.
    """
    raise NotImplementedError("mediapipe_hands.extract_hand_landmarks is a P2 stub.")
