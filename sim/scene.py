# Copyright (c) 2026, dimenso project.
#
# Dimenso-owned G1 fixed-base pick-and-place scene + task `Dimenso-AppleBasket-G1-v0`.
#
# Action = upstream G1 Pink-IK (Pinocchio + QP, posture-regularized) — the correct stable
# solver for the redundant 7-DOF arm. Action vector (28): [left_wrist_pose(7),
# right_wrist_pose(7), Dex3 hand joints(14)], poses in world/env-origin frame.
# Grasp is REAL physics (friction/force) — no welding, no fixed joints.

"""Dimenso G1 fixed-base cube-in-basket scene and `Dimenso-AppleBasket-G1-v0` task."""

import gymnasium as gym

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

# Right arm (7-DOF) + right Dex3 hand (7 joints); IK EE frame = the palm.
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]
EE_BODY = "right_hand_palm_link"
# Waist (3-DOF) joins the IK chain — matches the upstream G1 pick-place task. The fixed-base
# right arm alone maxes out its joints at every reachable grasp pose (diagnostics/PROBE.md);
# letting the torso turn toward the table opens the workspace and relieves the arm joints.
WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
IK_JOINTS = WAIST_JOINTS + RIGHT_ARM_JOINTS

# Moderate DLS damping (default ~0.05): enough to suppress elbow-flare blow-ups on the
# redundant 7-DOF arm (with fine interpolation) without too much tracking lag to reach.
DLS_LAMBDA = 0.1

# Object / target poses (right arm's reach; table top z≈0.70). World frame.
OBJECT_KIND = "cube"   # "cube" (default) or "ball" — swap here to revert
OBJECT_SIZE = 0.05     # cube edge length, or sphere diameter
OBJECT_MASS = 0.05     # kg
# Probe-chosen sweet spots (diagnostics/PROBE.md): tilt-0.5 grasp is reachable + in-limits
# in the near-arm region. Cube and basket both placed there, diagonally separated.
# NO riser. The whole tabletop is raised so a 5cm cube resting on it puts the grasp
# centre at z=0.96 — the ONE clean pose (reachable + all arm joints >0.15 rad clear +
# non-singular) found by diagnostics/probe_height.py once the waist DoF joined the IK.
TABLE_TOP = 0.78                       # tabletop height; cube centroid lands at 0.805
# Cube sits AT the grasp-centre's low reach floor (~0.806, from iter9 telemetry) and aligned in y
# to where the grasp centre actually lands (0.22). The arm cannot push the grasp centre below
# ~0.806, so a lower cube gets clipped on top and knocked off; this height centres the grasp.
OBJECT_POS = (0.13, 0.22, TABLE_TOP + OBJECT_SIZE / 2)   # grasp centre lands here: (0.13,0.22,0.805)
BASKET_CENTER = (0.04, 0.34)           # reachable place pose; gap to cube = 0.150 m
BASKET_FLOOR_Z = TABLE_TOP + 0.01      # basket floor box centre, resting on the tabletop
BASKET_WALL_H = 0.04                   # shallow walls -> low rim (the arm can barely lift past ~0.85)
BASKET_INNER = 0.14

_HIGH_FRICTION = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.8, dynamic_friction=1.6, restitution=0.0
)

# module-level (NOT a scene field) — shared camera lens spawn cfg
_CAM_SPAWN = sim_utils.PinholeCameraCfg(
    focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
    clipping_range=(0.02, 20.0),
)


def _object_spawn():
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1.0,
            linear_damping=0.2, angular_damping=0.2,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_HIGH_FRICTION,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10)),
    )
    if OBJECT_KIND == "ball":
        return sim_utils.SphereCfg(radius=OBJECT_SIZE / 2, **common)
    return sim_utils.CuboidCfg(size=(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE), **common)


def _static_box(size, color=(0.80, 0.65, 0.35), friction=True):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_HIGH_FRICTION if friction else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


@configclass
class DimensoAppleBasketSceneCfg(InteractiveSceneCfg):
    """G1 fixed-base scene: table + cube + basket (right-arm reachable) + 3-camera rig."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # raised tabletop (top at TABLE_TOP); cube + basket both rest directly on it — no riser
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.45, TABLE_TOP - 0.02)),
        spawn=_static_box((0.80, 0.60, 0.04), color=(0.55, 0.40, 0.25)),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=_object_spawn(),
    )

    # basket centred on BASKET_CENTER=(0.04,0.34); walls rest on the tabletop (TABLE_TOP)
    basket_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.04, 0.34, BASKET_FLOOR_Z)),
        spawn=_static_box((BASKET_INNER, BASKET_INNER, 0.02)),
    )
    basket_wall_px = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.11, 0.34, TABLE_TOP + BASKET_WALL_H / 2)),
        spawn=_static_box((0.02, BASKET_INNER, BASKET_WALL_H)),
    )
    basket_wall_nx = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.03, 0.34, TABLE_TOP + BASKET_WALL_H / 2)),
        spawn=_static_box((0.02, BASKET_INNER, BASKET_WALL_H)),
    )
    basket_wall_py = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.04, 0.41, TABLE_TOP + BASKET_WALL_H / 2)),
        spawn=_static_box((BASKET_INNER, 0.02, BASKET_WALL_H)),
    )
    basket_wall_ny = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.04, 0.27, TABLE_TOP + BASKET_WALL_H / 2)),
        spawn=_static_box((BASKET_INNER, 0.02, BASKET_WALL_H)),
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- 3-camera rig (GR00T-style). Disabled unless env.enable_cameras=True. ---
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_cam",
        update_period=0.0, height=240, width=320, data_types=["rgb"],
        spawn=_CAM_SPAWN,
        # forward (+Y) and down over the workspace, from above the torso
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.20, 0.45), rot=(0.0, 0.259, 0.0, 0.966), convention="ros"),
    )
    right_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_wrist_cam",
        update_period=0.0, height=200, width=200, data_types=["rgb"],
        spawn=_CAM_SPAWN,
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    left_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_wrist_cam",
        update_period=0.0, height=200, width=200, data_types=["rgb"],
        spawn=_CAM_SPAWN,
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.fix_root_link = True
        # stiffen the Dex3 fingers so they actually close on the cube (default 20 was too soft)
        self.robot.actuators["hands"].stiffness = 200.0
        self.robot.actuators["hands"].damping = 10.0


@configclass
class ActionsCfg:
    """Differential IK on the right arm (abs EE pose, damped LS) + Dex3 finger joints."""

    arm_ik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=IK_JOINTS,
        body_name=EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": DLS_LAMBDA},
        ),
        scale=1.0,
    )
    hand = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_HAND_JOINTS, use_default_offset=False
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _EmptyManagerCfg:
    """Empty manager cfg (no terms) — avoids None-manager PLAY-callback crashes."""


@configclass
class DimensoAppleBasketG1EnvCfg(ManagerBasedRLEnvCfg):
    """G1 fixed-base cube-in-basket environment (single env, Pink IK + Dex3 grasp)."""

    enable_cameras: bool = False  # set True (and launch --enable_cameras) for the rig

    scene: DimensoAppleBasketSceneCfg = DimensoAppleBasketSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: _EmptyManagerCfg = _EmptyManagerCfg()
    rewards: _EmptyManagerCfg = _EmptyManagerCfg()
    curriculum: _EmptyManagerCfg = _EmptyManagerCfg()
    events: _EmptyManagerCfg = _EmptyManagerCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        # disable the camera rig unless explicitly enabled (keeps the headless test fast)
        if not self.enable_cameras:
            self.scene.head_cam = None
            self.scene.right_wrist_cam = None
            self.scene.left_wrist_cam = None


gym.register(
    id="Dimenso-AppleBasket-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": DimensoAppleBasketG1EnvCfg},
    disable_env_checker=True,
)
