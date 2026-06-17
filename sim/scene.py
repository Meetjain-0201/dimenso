# Copyright (c) 2026, dimenso project.
#
# Adapted from Isaac Lab's `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`
# (isaaclab_tasks.manager_based.locomanipulation.pick_place.fixed_base_upper_body_ik_g1_env_cfg).
# dimenso owns this editable copy on purpose: we deliberately do NOT import the task
# code live from ~/projects/IsaacLab at runtime. The installed `isaaclab` / `isaacsim`
# libraries remain plain dependencies.
#
# Differences from the upstream env (intentional, for the P1 scaffold):
#   * Scene props are primitive shapes spawned on-disk (no Nucleus asset downloads):
#       - table  = static cuboid slab
#       - apple  = dynamic red sphere (the graspable object)
#       - basket = static open-topped bin built from cuboid walls (the target)
#   * Action = simple JointPositionAction on the arm joints. The real upstream env
#     uses a Pink IK action (absolute EE pose + grasp DoF). We do NOT implement IK in
#     P1 (see retarget/ik_solver.py). The full Pink-IK interface is documented in
#     CLAUDE.md and is the P2 target. With a zero action the robot holds its default
#     pose, i.e. it stays idle — which is all run_headless.py needs.
#   * No OpenXR / teleop devices, no Pink controller, no URDF retrieval — so importing
#     this module needs only `pxr` (i.e. a launched Isaac Sim app), nothing else.

"""Dimenso G1 fixed-base apple-in-basket scene and `Dimenso-AppleBasket-G1-v0` task."""

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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG


##
# Scene definition
##
@configclass
class DimensoAppleBasketSceneCfg(InteractiveSceneCfg):
    """G1 fixed-base manipulation scene: table + apple + basket.

    The G1 humanoid is pinned at the pelvis (fixed root link) so only the upper body
    can move — matching the upstream FixedBaseUpperBodyIK task. The apple is the
    graspable object; the basket is the place target.
    """

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table — static cuboid slab (primitive, no asset download). Top surface ~0.70 m.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.50, 0.68)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.60, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25)),
        ),
    )

    # Apple — dynamic red sphere, the graspable object. Rests on the table.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.50, 0.745), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10)),
        ),
    )

    # Basket — static open-topped bin (floor + 4 thin walls), the place target.
    basket_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.50, 0.71)),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_px = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.50, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.20, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_nx = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNx",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.10, 0.50, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.20, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_py = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallPy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.60, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.02, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )
    basket_wall_ny = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketWallNy",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.40, 0.755)),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.02, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.65, 0.35)),
        ),
    )

    # Unitree G1 (29 DoF) — assigned in __post_init__ so we can pin the root link.
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        # Fixed base: pin the pelvis so only the upper body moves.
        self.robot.spawn.articulation_props.fix_root_link = True


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specification.

    P1 placeholder: direct joint-position targets for the arm joints (shoulders,
    elbows, wrists). `use_default_offset=True` means a zero action holds the default
    pose, so the robot is idle. The real task replaces this with a Pink IK action
    (absolute EE pose + grasp DoF) — see retarget/ik_solver.py and CLAUDE.md.
    """

    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specification (minimal, low-dim, for the P1 scaffold)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        apple_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _EmptyManagerCfg:
    """An empty manager cfg (no terms).

    Used instead of ``None`` for the managers we don't use (commands/rewards/
    curriculum/events). A manager built from ``None`` still subscribes to the PLAY
    timeline event in ``ManagerBase.__init__`` and then crashes in
    ``_resolve_terms_callback`` (``None.__dict__``) when the sim starts playing —
    headless launchers hit this because the sim is not yet playing at env construction.
    An empty (but non-None) cfg keeps the callback happy with zero terms.
    """


@configclass
class DimensoAppleBasketG1EnvCfg(ManagerBasedRLEnvCfg):
    """G1 fixed-base apple-in-basket environment (single env, headless-friendly)."""

    scene: DimensoAppleBasketSceneCfg = DimensoAppleBasketSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers (no RL training in P1). Empty cfgs, not None — see _EmptyManagerCfg.
    commands: _EmptyManagerCfg = _EmptyManagerCfg()
    rewards: _EmptyManagerCfg = _EmptyManagerCfg()
    curriculum: _EmptyManagerCfg = _EmptyManagerCfg()
    events: _EmptyManagerCfg = _EmptyManagerCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 200.0  # 200 Hz physics
        self.sim.render_interval = self.decimation


##
# Gym registration — dimenso owns this task id.
##
gym.register(
    id="Dimenso-AppleBasket-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": DimensoAppleBasketG1EnvCfg},
    disable_env_checker=True,
)
