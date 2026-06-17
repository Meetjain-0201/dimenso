"""dimenso pipeline — the single entry point: text task -> G1 Dex3 pick-and-place.

`run(config)` is the ONE callable the test / CLI / control panel invoke. For now the VLM
(Gemini) layer is stubbed by `get_task_text()`; a code-as-policy sequencer turns the task
string into an ordered step list, which is executed in the `Dimenso-AppleBasket-G1-v0`
env via the env's built-in differential IK (absolute EE pose) + Dex3 finger control.

Real physics — the grasp is friction/force only (no welding, no fixed joints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    success: bool
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    steps: list = field(default_factory=list)


# ---- VLM stub + code-as-policy sequencer --------------------------------------------
def get_task_text() -> str:
    """Stubbed Gemini Robotics-ER output (P2 will replace with a real call)."""
    return "pick up the ball and place it in the basket"


PICKPLACE_STEPS = [
    "move_above_ball",
    "descend_to_ball",
    "grasp",
    "lift",
    "move_above_basket",
    "descend",
    "release",
]


def sequence_task(task_text: str) -> list[str]:
    """Map a task string to an ordered step list (code-as-policy)."""
    t = task_text.lower()
    if "ball" in t and "basket" in t and ("pick" in t or "place" in t):
        return list(PICKPLACE_STEPS)
    raise ValueError(f"no policy for task: {task_text!r}")


# ---- config -------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "headless": True,
    "enable_camera": False,
    "close_app": True,
    "keep_alive": False,  # if GUI: hold the window open after the sequence for viewing
    "task_text": None,
    "objects": {"ball": "object", "basket": "basket"},
    "basket_center": (0.32, 0.45),
    "basket_floor_z": 0.71,
    "basket_rim_z": 0.80,
    "basket_radius": 0.085,
    "approach_height": 0.12,
    "grasp_palm_above": 0.06,
    "lift_z": 0.95,
    "release_palm_z": 0.82,
    "steps_warmup": 90,
    "interp_waypoints": 40,     # EE target split into this many small increments per move
    "steps_per_waypoint": 6,    # physics steps held per increment (smooth -> stable IK)
    "steps_settle_move": 30,    # hold at the final target after a move
    "steps_grasp": 180,         # hold while fingers close/open
    "steps_settle": 120,
    # Dex3 right hand: [index_0,index_1,middle_0,middle_1,thumb_0,thumb_1,thumb_2]
    "hand_open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "hand_close": [0.9, 0.9, 0.9, 0.9, 0.6, 0.9, 0.6],
    "grasp_euler_w": None,  # None -> hold the EE's reset orientation
}


def _cfg(config) -> dict:
    out = dict(DEFAULTS)
    if isinstance(config, dict):
        out.update(config)
    return out


def run(config: dict | None = None) -> RunResult:
    """Run the full text -> G1 pick-and-place pipeline. Single entry point."""
    cfg = _cfg(config)

    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = bool(cfg["headless"])
    if cfg["enable_camera"]:
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        return _execute(cfg, simulation_app)
    finally:
        if cfg["close_app"]:
            simulation_app.close()


def _execute(cfg, simulation_app) -> RunResult:
    import math
    import sys
    from pathlib import Path

    import torch
    import gymnasium as gym
    from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    import scene  # registers Dimenso-AppleBasket-G1-v0

    task_text = cfg["task_text"] or get_task_text()
    steps = sequence_task(task_text)
    print(f"[ego2g1] task: {task_text!r}")
    print(f"[ego2g1] steps: {steps}")

    env_cfg = scene.DimensoAppleBasketG1EnvCfg()
    env_cfg.scene.num_envs = 1
    print("[ego2g1] building env ...", flush=True)
    env = gym.make("Dimenso-AppleBasket-G1-v0", cfg=env_cfg)
    u = env.unwrapped
    device = u.device
    robot = u.scene["robot"]
    ball = u.scene[cfg["objects"]["ball"]]
    print("[ego2g1] env built; resetting ...", flush=True)
    env.reset()
    print("[ego2g1] env reset done — starting sequence", flush=True)
    ee_idx = robot.body_names.index(scene.EE_BODY)

    def ee_pose_w():
        return robot.data.body_pos_w[0, ee_idx].clone(), robot.data.body_quat_w[0, ee_idx].clone()

    def ball_pos_w():
        return ball.data.root_pos_w[0].clone()

    hand_open = torch.tensor(cfg["hand_open"], device=device, dtype=torch.float32)
    hand_close = torch.tensor(cfg["hand_close"], device=device, dtype=torch.float32)

    _, ee_quat0 = ee_pose_w()
    if cfg["grasp_euler_w"] is not None:
        r, p, y = cfg["grasp_euler_w"]
        grasp_quat_w = quat_from_euler_xyz(
            torch.tensor([r], device=device), torch.tensor([p], device=device),
            torch.tensor([y], device=device),
        )[0]
    else:
        grasp_quat_w = ee_quat0.clone()

    def target_for(step, bpos):
        bx, by, bz = float(bpos[0]), float(bpos[1]), float(bpos[2])
        cx, cy = cfg["basket_center"]
        ah, ga = cfg["approach_height"], cfg["grasp_palm_above"]
        table = {
            "move_above_ball": ((bx, by, bz + ah), hand_open),
            "descend_to_ball": ((bx, by, bz + ga), hand_open),
            "grasp": ((bx, by, bz + ga), hand_close),
            "lift": ((bx, by, cfg["lift_z"]), hand_close),
            "move_above_basket": ((cx, cy, cfg["lift_z"]), hand_close),
            "descend": ((cx, cy, cfg["release_palm_z"]), hand_close),
            "release": ((cx, cy, cfg["release_palm_z"]), hand_open),
        }
        pos, hand = table[step]
        return pos, grasp_quat_w, hand

    def make_action(target_pos_w, target_quat_w, hand_targets):
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        tpos = torch.tensor(target_pos_w, device=device, dtype=torch.float32).unsqueeze(0)
        tquat = target_quat_w.unsqueeze(0)
        pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, tpos, tquat)
        return torch.cat([pos_b, quat_b, hand_targets.unsqueeze(0)], dim=-1)

    # warmup: hold the arm at its default pose with the hand open, let the ball settle,
    # and capture its rest pose. Also tells us if the ball stays on the table at all.
    ee_p0, _ = ee_pose_w()
    hold = make_action(list(ee_p0.tolist()), grasp_quat_w, hand_open)
    z_reset = float(ball_pos_w()[2])
    for k in range(cfg["steps_warmup"]):
        env.step(hold)
        if k % 30 == 0:
            print(f"[ego2g1]   ..warmup {k}/{cfg['steps_warmup']} ball_z={float(ball_pos_w()[2]):.3f}", flush=True)
    ball_rest = ball_pos_w()
    print(f"[ego2g1] warmup: ball z reset={z_reset:.3f} -> rest={float(ball_rest[2]):.3f} "
          f"pos={[round(float(v), 3) for v in ball_rest]}")

    arm_idx = [robot.joint_names.index(j) for j in scene.RIGHT_ARM_JOINTS]

    def arm_q():
        return [round(float(q), 2) for q in robot.data.joint_pos[0, arm_idx].tolist()]

    def respawn_if_fallen():
        bp = ball_pos_w()
        if float(bp[2]) < 0.55:  # below the table -> it rolled/fell off
            pose = torch.tensor([[*scene.BALL_POS, 1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
            ball.write_root_pose_to_sim(pose)
            ball.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            print("[ego2g1]   (ball had fallen off — respawned on table)", flush=True)

    def move_to(target_pos, hand, label):
        # interpolate the EE target in small increments so each IK correction is tiny
        # and the motion stays smooth/stable (avoids the redundant-arm oscillation).
        nwp, spw = cfg["interp_waypoints"], cfg["steps_per_waypoint"]
        cur = ee_pose_w()[0].tolist()
        for i in range(1, nwp + 1):
            f = i / nwp
            wp = [cur[j] + (target_pos[j] - cur[j]) * f for j in range(3)]
            a = make_action(wp, grasp_quat_w, hand)
            for _ in range(spw):
                env.step(a)
            if i % max(1, nwp // 4) == 0:
                ep = ee_pose_w()[0]
                print(f"[ego2g1]   ..{label} wp{i}/{nwp} ee={[round(float(v), 3) for v in ep]} "
                      f"arm_q={arm_q()}", flush=True)
        a = make_action(target_pos, grasp_quat_w, hand)
        for _ in range(cfg["steps_settle_move"]):
            env.step(a)

    log = []
    grasp_state = "open"
    for step in steps:
        respawn_if_fallen()
        tpos, tquat, hand = target_for(step, ball_rest)
        if step == "grasp":
            grasp_state = "closed"
        elif step == "release":
            grasp_state = "open"
        if step in ("grasp", "release"):
            # hold the EE pose; actuate the Dex3 fingers (close / open) over time
            a = make_action(tpos, grasp_quat_w, hand)
            for _ in range(cfg["steps_grasp"]):
                env.step(a)
        else:
            move_to(tpos, hand, step)
        ee_p, _ = ee_pose_w()
        bp = ball_pos_w()
        err = float(torch.norm(ee_p - torch.tensor(tpos, device=device)))
        rec = {
            "step": step, "grasp": grasp_state,
            "target": [round(x, 3) for x in tpos],
            "ee": [round(float(v), 3) for v in ee_p],
            "ee_err": round(err, 3),
            "ball": [round(float(v), 3) for v in bp],
            "arm_q": arm_q(),
        }
        log.append(rec)
        print(f"[ego2g1] {step:18s} grasp={grasp_state:6s} ee_err={err:.3f} "
              f"ee={rec['ee']} ball={rec['ball']} arm_q={rec['arm_q']}", flush=True)

    last = make_action(*target_for("release", ball_pos_w()))
    for _ in range(cfg["steps_settle"]):
        env.step(last)

    bp = ball_pos_w()
    cx, cy = cfg["basket_center"]
    xy_off = math.hypot(float(bp[0]) - cx, float(bp[1]) - cy)
    in_xy = xy_off <= cfg["basket_radius"]
    in_z = (cfg["basket_floor_z"] - 0.03) <= float(bp[2]) <= (cfg["basket_rim_z"] + 0.02)
    success = bool(in_xy and in_z)
    metrics = {
        "task_text": task_text,
        "ball_final_pos": [round(float(v), 4) for v in bp],
        "basket_center": [cx, cy, cfg["basket_floor_z"]],
        "basket_radius": cfg["basket_radius"],
        "basket_rim_z": cfg["basket_rim_z"],
        "basket_floor_z": cfg["basket_floor_z"],
        "xy_off": round(xy_off, 4),
        "in_xy": in_xy, "in_z": in_z,
        "final_grasp_state": grasp_state,
    }
    msg = "ball placed in basket" if success else f"ball not in basket (xy_off={xy_off:.3f}, z={float(bp[2]):.3f})"
    print(f"[ego2g1] RESULT success={success} | {msg}")
    result = RunResult(success=success, message=msg, metrics=metrics, steps=log)

    if cfg.get("keep_alive") and not cfg["headless"]:
        print("[ego2g1] view open — close the Isaac Sim window to exit.")
        hold = make_action(*target_for("release", ball_rest))
        while simulation_app.is_running():
            env.step(hold)

    env.close()
    return result
