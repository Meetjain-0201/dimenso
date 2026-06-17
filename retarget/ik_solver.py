"""Stub: G1 EE target -> upper-body joint positions via Pink IK. Implemented in P2.

The installed Isaac Lab ships the real machinery we will wire up here:
``isaaclab.controllers.pink_ik`` + ``PinkInverseKinematicsActionCfg``. The upstream
task ``Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`` configures it with absolute
EE pose tasks (LocalFrameTask) for both wrist_yaw links in the pelvis frame plus a
null-space posture task. See CLAUDE.md for the full extracted interface.
"""

from __future__ import annotations

import numpy as np

from .kinematics import EETarget


def solve_upper_body_ik(targets: dict[str, EETarget]) -> np.ndarray:
    """Solve G1 upper-body joint positions for the given EE targets.

    Args:
        targets: ``{"left_wrist": EETarget, "right_wrist": EETarget}``.

    Returns:
        Joint position targets for the Pink-controlled upper-body joints
        (shoulders, elbows, wrists, waist) plus the 14 hand/grasp joints.

    P2: drive this through Isaac Lab's Pink IK controller rather than re-solving by
    hand. In P1 we do not control the robot at all (it holds its default pose).
    """
    raise NotImplementedError("ik_solver.solve_upper_body_ik is a P2 stub.")
