# Copyright (c) 2026, dimenso project.
#
# Dimenso-owned G1 fixed-base ball-in-basket scene + task `Dimenso-AppleBasket-G1-v0`.
# Adapted from Isaac Lab's `Isaac-PickPlace-FixedBaseUpperBodyIK-G1` (we own this copy;
# the installed isaaclab/isaacsim libs stay as deps).
#
# Action interface (real, used by the pipeline):
#   * arm_ik : built-in differential IK on the RIGHT arm — absolute EE pose (7: pos+quat)
#              targeting `right_hand_palm_link`, in the robot base (pelvis) frame.
#   * hand   : joint-position targets for the 7 RIGHT Dex3 finger joints (grasp/release).
# Action vector = [arm_ik(7), hand(7)] = 14.
#
# Grasp is REAL physics (friction + finger force/position) — no welding, no fixed joints.

"""Dimenso G1 fixed-base ball-in-basket scene and `Dimenso-AppleBasket-G1-v0` task."""

import gymnasium as gym

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

# Right arm (7-DOF) and right Dex3 hand (7 joints) — verified from the loaded articulation.
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]
EE_BODY = "right_hand_palm_link"

# Object / target poses (right arm's reach; table top z≈0.70). World frame.
BALL_POS = (0.15, 0.45, 0.725)
BALL_RADIUS = 0.025
BASKET_CENTER = (0.32, 0.45)
BASKET_FLOOR_Z = 0.71
BASKET_WALL_H = 0.09
BASKET_INNER = 0.16  # full width


@configclass
class DimensoAppleBasketSceneCfg(InteractiveSceneCfg):
    """G1 fixed-base manipulation scene: table + ball + basket (right-arm reachable)."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table — static cuboid slab, top ≈ 0.70 m.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.45, 0.68)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.60, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25)),
        ),
    )

    # Ball — light, small, high-friction (graspable by the Dex3). Dynamic rigid body.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                linear_damping=0.2, angular_damping=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.6, dynamic_friction=1.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10)),
        ),
    )

    # Basket — static open-topped bin (floor + 4 walls) centered at BASKET_CENTER.
    basket_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.45, BASKET_FLOOR_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(BASKET_INNER, BASKET_INNER, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_px = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.40, 0.45, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, BASKET_INNER, BASKET_WALL_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_nx = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.24, 0.45, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, BASKET_INNER, BASKET_WALL_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_py = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.53, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(BASKET_INNER, 0.02, BASKET_WALL_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_ny = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.37, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(BASKET_INNER, 0.02, BASKET_WALL_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Built-in differential IK on the right arm (abs EE pose) + Dex3 finger positions."""

    arm_ik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        body_name=EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
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
        ball_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
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
    """G1 fixed-base ball-in-basket environment (single env, IK + Dex3 grasp)."""

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
        self.decimation = 2
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation


gym.register(
    id="Dimenso-AppleBasket-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": DimensoAppleBasketG1EnvCfg},
    disable_env_checker=True,
)
