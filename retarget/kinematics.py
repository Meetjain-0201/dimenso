"""Stub: human wrist/grasp -> G1 EE target + grasp state. Implemented in P2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EETarget:
    """A retargeted end-effector target for one G1 arm.

    Frame matches the Isaac Lab task: absolute EE pose in the robot **pelvis** frame.
    """

    position: np.ndarray  # (3,) xyz in pelvis frame, metres
    orientation: np.ndarray  # (4,) wxyz quaternion in pelvis frame
    grasp: float  # 0.0 = open, 1.0 = fully closed


def retarget_hands(landmarks: np.ndarray, handedness: np.ndarray) -> dict[str, EETarget]:
    """Map per-frame human hand landmarks to G1 left/right EE targets + grasp.

    Args:
        landmarks: (num_hands, 21, 3) normalized hand keypoints for one frame.
        handedness: (num_hands,) Left/Right labels.

    Returns:
        ``{"left_wrist": EETarget, "right_wrist": EETarget}``.

    P2: derive wrist position/orientation from landmarks, map image space -> pelvis
    frame (workspace bounds + scaling), and a pinch metric (thumb-index distance) ->
    grasp scalar. Mirrors the bimanual mapping from isaac-hand-teleop.
    """
    raise NotImplementedError("kinematics.retarget_hands is a P2 stub.")
