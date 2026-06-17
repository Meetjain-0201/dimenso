# Copyright (c) 2026, dimenso project.
#
# dimenso-owned copy of the upstream G1 Pink-IK controller + action config
# (from isaaclab_tasks .../locomanipulation/pick_place/configs/pink_controller_cfg.py).
# Pink IK = Pinocchio forward-kinematics + QP solver, posture-regularized — the correct,
# stable IK for the redundant 7-DOF G1 arm (the generic DifferentialIKController jitters).
#
# Action = [left_wrist_pose(7: pos+quat), right_wrist_pose(7), hand_joints(14)] = 28,
# poses given in the world/env-origin frame (the action transforms them to the pelvis
# frame internally). Frame names with the `g1_29dof_with_hand_rev_1_0_` prefix are the
# Pinocchio/URDF link names; `target_eef_link_names` maps tasks -> Isaac Sim link names.

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

# 14 Dex3 hand joints (both hands), in the order the action expects.
HAND_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
]
# indices of the RIGHT-hand joints within HAND_JOINT_NAMES (for grasp control)
RIGHT_HAND_IDX_IN_HANDS = [i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("right_")]

G1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=14,
    show_ik_warnings=False,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=[
                "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            ],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
)

G1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
        "waist_.*_joint",
    ],
    hand_joint_names=HAND_JOINT_NAMES,
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    asset_name="robot",
    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)

# Nucleus URDF the Pink controller loads (set on the cfg in env __post_init__).
G1_URDF_NUCLEUS_PATH = (
    "{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
    "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
)
