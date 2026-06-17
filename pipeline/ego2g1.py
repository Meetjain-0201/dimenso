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
    "telemetry": None,    # path prefix -> dump per-step telemetry JSON+CSV (diagnostics)
    "respawn": True,      # respawn object if it falls; set False for honest diagnostics
    "task_text": None,
    "objects": {"ball": "object", "basket": "basket"},
    "basket_center": (0.04, 0.34),  # matches scene BASKET_CENTER (probe-chosen, gap 0.141 m)
    "basket_floor_z": 0.98,     # cube rest level inside basket (table 0.935 + floor box -> ~0.98)
    "basket_rim_z": 0.985,      # wall top = TABLE_TOP 0.935 + WALL_H 0.05
    "basket_radius": 0.085,
    "grasp_tilt": 0.5,          # clean grasp pose tilt (probe_height.py): reachable + joint-clear
    "grasp_center_frac": 0.75,  # control the finger grasp-center (this frac of palm->fingertip), not the palm
    "approach_height": 0.05,    # grasp-centre height above cube for the "above" waypoint (stay in reach)
    "grasp_palm_above": 0.0,    # descend grasp-centre TO the cube centroid (z=0.96)
    "lift_z": 1.01,             # lift grasp-centre just above the rim (0.985)
    "release_palm_z": 1.0,      # grasp-centre above rim, then open -> cube drops into basket
    "steps_warmup": 90,
    "interp_waypoints": 60,     # fine: small EE delta per waypoint -> small joint delta (no swing)
    "steps_per_waypoint": 8,    # physics steps held per increment (less tracking lag)
    "steps_settle_move": 60,    # hold at the final target after a move (converge to target)
    "steps_grasp": 180,         # hold while fingers close/open
    "steps_settle": 120,
    "arm_q_delta_thresh": 0.15, # max per-step joint change (rad) allowed on the carry (no jitter)
    "ee_err_thresh": 0.12,      # max EE tracking error (m) allowed through the trajectory
    # right Dex3 close targets, in RIGHT_HAND_JOINTS order: index0,index1,middle0,middle1,thumb0,thumb1,thumb2
    "hand_close_vals": [1.2, 1.5, 1.2, 1.5, 0.9, 1.0, 0.6],  # fuller curl to bridge the gap
    "grasp_euler_w": None,  # None -> hold the palm's reset orientation
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
    from isaaclab.utils.math import (
        quat_from_angle_axis, quat_from_euler_xyz, quat_mul, quat_rotate,
        quat_rotate_inverse, subtract_frame_transforms,
    )

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
    ee_idx = robot.body_names.index(scene.EE_BODY)  # right_hand_palm_link

    def ee_pose_w():
        return robot.data.body_pos_w[0, ee_idx].clone(), robot.data.body_quat_w[0, ee_idx].clone()

    def ball_pos_w():
        return ball.data.root_pos_w[0].clone()

    # right Dex3 hand command (7 joints, in RIGHT_HAND_JOINTS order)
    def hand_vec(closed):
        v = torch.zeros(7, device=device, dtype=torch.float32)
        if closed:
            v[:] = torch.as_tensor(cfg["hand_close_vals"], device=device, dtype=torch.float32)
        return v

    hand_open = hand_vec(False)
    hand_close = hand_vec(True)

    _, ee_quat0 = ee_pose_w()
    # auto fingers-down orientation: rotate the palm so its current finger direction
    # (palm -> fingertips) points world -Z, with minimal wrist twist.
    ftip_idx = [robot.body_names.index(n) for n in ("right_hand_middle_1_link", "right_hand_index_1_link")]

    def fingers_down_quat():
        palm = robot.data.body_pos_w[0, ee_idx]
        ftips = robot.data.body_pos_w[0, ftip_idx].mean(dim=0)
        v = ftips - palm
        n = float(v.norm())
        if n < 1e-6:
            return ee_quat0.clone()
        v = v / n
        down = torch.tensor([0.0, 0.0, -1.0], device=device)
        axis = torch.cross(v, down, dim=0)
        s = float(axis.norm())
        if s < 1e-6:
            return ee_quat0.clone()
        axis = axis / s
        angle = torch.acos(torch.clamp(torch.dot(v, down), -1.0, 1.0)) * float(cfg["grasp_tilt"])
        dq = quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0))[0]
        return quat_mul(dq.unsqueeze(0), ee_quat0.unsqueeze(0))[0]

    if cfg["grasp_euler_w"] is not None:
        r, p, y = cfg["grasp_euler_w"]
        grasp_quat_w = quat_from_euler_xyz(
            torch.tensor([r], device=device), torch.tensor([p], device=device),
            torch.tensor([y], device=device),
        )[0]
    else:
        grasp_quat_w = fingers_down_quat()
    print(f"[ego2g1] grasp orientation (fingers-down) quat={[round(float(q),3) for q in grasp_quat_w.tolist()]}", flush=True)

    # Grasp-center offset: IK targets the palm link, but the Dex3 closes ~5-6cm further out.
    # offset_local = palm->fingertip-mean in the palm's LOCAL frame (orientation-independent).
    # We treat all targets as GRASP-CENTER targets and convert to a palm target via this offset,
    # so the fingers cage the object. gc_actual_w() = the live grasp-center world position.
    _gc_tips = [robot.body_names.index(n) for n in
                ("right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link")]
    _palm0 = robot.data.body_pos_w[0, ee_idx]
    _ftm0 = robot.data.body_pos_w[0, _gc_tips].mean(dim=0)
    grasp_offset_local = (quat_rotate_inverse(ee_quat0.unsqueeze(0), (_ftm0 - _palm0).unsqueeze(0))[0]
                          * float(cfg["grasp_center_frac"]))
    print(f"[ego2g1] grasp-center local offset={[round(float(v), 3) for v in grasp_offset_local.tolist()]}", flush=True)

    def gc_actual_w():
        palm = robot.data.body_pos_w[0, ee_idx]
        return palm + quat_rotate(grasp_quat_w.unsqueeze(0), grasp_offset_local.unsqueeze(0))[0]

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

    def make_action(target_pos_w, target_quat_w, hand7):
        # target_pos_w is the desired GRASP-CENTER world pos; convert to the palm IK target
        # by subtracting the rotated grasp-center offset. DiffIK action (14): [pos_b, quat_b, hand].
        tquat = (target_quat_w if torch.is_tensor(target_quat_w)
                 else torch.as_tensor(target_quat_w, device=device, dtype=torch.float32))
        gc = torch.as_tensor(target_pos_w, device=device, dtype=torch.float32)
        world_off = quat_rotate(tquat.unsqueeze(0), grasp_offset_local.unsqueeze(0))[0]
        palm_target = (gc - world_off).unsqueeze(0)
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, palm_target, tquat.unsqueeze(0))
        return torch.cat([pos_b, quat_b, hand7.unsqueeze(0)], dim=-1)

    # warmup: hold the arm at its default pose with the hand open, let the ball settle,
    # and capture its rest pose. Also tells us if the ball stays on the table at all.
    ee_p0, _ = ee_pose_w()
    hold = make_action(list(gc_actual_w().tolist()), grasp_quat_w, hand_open)
    z_reset = float(ball_pos_w()[2])
    for k in range(cfg["steps_warmup"]):
        env.step(hold)
        if k % 30 == 0:
            print(f"[ego2g1]   ..warmup {k}/{cfg['steps_warmup']} ball_z={float(ball_pos_w()[2]):.3f}", flush=True)
    ball_rest = ball_pos_w()
    print(f"[ego2g1] warmup: ball z reset={z_reset:.3f} -> rest={float(ball_rest[2]):.3f} "
          f"pos={[round(float(v), 3) for v in ball_rest]}")

    arm_idx = [robot.joint_names.index(j) for j in scene.RIGHT_ARM_JOINTS]
    hand_dof_idx = [robot.joint_names.index(j) for j in scene.RIGHT_HAND_JOINTS]
    _jl = robot.data.soft_joint_pos_limits[0]  # (num_dof, 2)
    arm_limits = [(float(_jl[i, 0]), float(_jl[i, 1])) for i in arm_idx]

    def arm_q_t():
        return robot.data.joint_pos[0, arm_idx]

    def arm_q():
        return [round(float(q), 2) for q in arm_q_t().tolist()]

    def manip_cond():
        # manipulability sqrt(det(J Jᵀ)) and Jacobian condition for the EE wrt the 7 arm joints
        try:
            J = robot.root_physx_view.get_jacobians()
            jb = ee_idx - 1 if J.shape[1] == (robot.num_bodies - 1) else ee_idx
            Je = J[0, jb][:, arm_idx].float()  # 6x7
            s = torch.linalg.svdvals(Je)
            smin, smax = float(s.min()), float(s.max())
            return float(torch.prod(s)), (smax / smin if smin > 1e-9 else float("inf"))
        except Exception:
            return float("nan"), float("nan")

    def _quat_angle(q1, q2):
        d = float(torch.clamp((q1 * q2).sum().abs(), 0.0, 1.0))
        return 2.0 * math.acos(d)

    telem = []
    cur = {"phase": "warmup", "tpos": None, "fcmd": None, "fell": False}

    def record():
        if not cfg.get("telemetry"):
            return
        q = arm_q_t()
        ee_p = gc_actual_w()  # report the grasp center (the controlled point), not the palm
        ee_q = robot.data.body_quat_w[0, ee_idx]
        cp = ball.data.root_pos_w[0]
        manip, cond = manip_cond()
        perr = (float((ee_p - torch.as_tensor(cur["tpos"], device=device, dtype=torch.float32)).norm())
                if cur["tpos"] is not None else float("nan"))
        oerr = _quat_angle(ee_q, grasp_quat_w)
        lim = [round(min(float(q[i]) - arm_limits[i][0], arm_limits[i][1] - float(q[i])), 3) for i in range(7)]
        telem.append({
            "i": len(telem), "phase": cur["phase"],
            "cube_pos": [round(float(v), 4) for v in cp],
            "cube_vel": round(float(ball.data.root_lin_vel_w[0].norm()), 4),
            "ee_pos": [round(float(v), 4) for v in ee_p],
            "ee_pos_err": round(perr, 4), "ee_orient_err": round(oerr, 4),
            "arm_q": [round(float(x), 3) for x in q], "jlim": lim,
            "manip": round(manip, 5), "jac_cond": round(cond, 2),
            "fcmd": cur["fcmd"],
            "fact": [round(float(robot.data.joint_pos[0, h]), 3) for h in hand_dof_idx],
        })

    def respawn_if_fallen():
        bp = ball_pos_w()
        if float(bp[2]) < 0.55:  # below the table -> it rolled/fell off
            cur["fell"] = True
            if not cfg.get("respawn", True):
                return  # diagnostics: let it lie so we can see the real fall
            pose = torch.tensor([[*scene.OBJECT_POS, 1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
            ball.write_root_pose_to_sim(pose)
            ball.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            print("[ego2g1]   (object had fallen off — respawned [FAILURE EVENT])", flush=True)

    # stability accumulators (the no-jitter check): max per-step joint delta + max EE error,
    # tracked across the WHOLE trajectory (incl. carry).
    stab = {"max_dq": 0.0, "max_ee_err": 0.0}
    prev_q = arm_q_t().clone()

    def step_track(a, target_pos, hand=None):
        nonlocal prev_q
        cur["tpos"] = target_pos
        if hand is not None:
            cur["fcmd"] = [round(float(x), 3) for x in (hand.tolist() if torch.is_tensor(hand) else hand)]
        env.step(a)
        respawn_if_fallen()  # the instant the cube falls off, put it back (checked every step)
        q = arm_q_t()
        dq = float((q - prev_q).abs().max())
        prev_q = q.clone()
        err = float((gc_actual_w()
                     - torch.as_tensor(target_pos, device=device, dtype=torch.float32)).norm())
        stab["max_dq"] = max(stab["max_dq"], dq)
        stab["max_ee_err"] = max(stab["max_ee_err"], err)
        record()

    def move_to(target_pos, hand, label, track_z=None):
        # interpolate the GRASP-CENTER target in small increments (smooth, stable IK).
        # If track_z is set, re-acquire the LIVE cube each waypoint (closed-loop approach):
        # target = (live cube xy, live cube z + track_z) — so it follows the cube in real time.
        nwp, spw = cfg["interp_waypoints"], cfg["steps_per_waypoint"]
        start = gc_actual_w().tolist()
        for i in range(1, nwp + 1):
            f = i / nwp
            if track_z is not None:
                cb = ball_pos_w()
                target_pos = [float(cb[0]), float(cb[1]), float(cb[2]) + track_z]
            wp = [start[j] + (target_pos[j] - start[j]) * f for j in range(3)]
            a = make_action(wp, grasp_quat_w, hand)
            for _ in range(spw):
                step_track(a, wp, hand)
            if i % max(1, nwp // 4) == 0:
                print(f"[ego2g1]   ..{label} wp{i}/{nwp} gc={[round(float(v), 3) for v in gc_actual_w()]} "
                      f"cube={[round(float(v), 3) for v in ball_pos_w()]} max_dq={stab['max_dq']:.3f}", flush=True)
        a = make_action(target_pos, grasp_quat_w, hand)
        for _ in range(cfg["steps_settle_move"]):
            step_track(a, target_pos, hand)

    log = []
    grasp_state = "open"
    for step in steps:
        respawn_if_fallen()
        cur["phase"] = step
        tpos, tquat, hand = target_for(step, ball_pos_w())  # LIVE cube position (closed-loop)
        track_z = {"move_above_ball": cfg["approach_height"],
                   "descend_to_ball": cfg["grasp_palm_above"]}.get(step)
        if step == "grasp":
            grasp_state = "closed"
        elif step == "release":
            grasp_state = "open"
        if step in ("grasp", "release"):
            # hold the grasp-center over the live cube; actuate the Dex3 fingers over time
            a = make_action(tpos, grasp_quat_w, hand)
            for _ in range(cfg["steps_grasp"]):
                step_track(a, tpos, hand)
        else:
            move_to(tpos, hand, step, track_z=track_z)
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

    cur["phase"] = "settle"
    rel_pos, _, rel_hand = target_for("release", ball_rest)
    last = make_action(rel_pos, grasp_quat_w, rel_hand)
    for _ in range(cfg["steps_settle"]):
        step_track(last, rel_pos, rel_hand)

    bp = ball_pos_w()
    cx, cy = cfg["basket_center"]
    xy_off = math.hypot(float(bp[0]) - cx, float(bp[1]) - cy)
    in_xy = xy_off <= cfg["basket_radius"]
    in_z = (cfg["basket_floor_z"] - 0.03) <= float(bp[2]) <= (cfg["basket_rim_z"] + 0.02)
    stable = bool(stab["max_dq"] <= cfg["arm_q_delta_thresh"] and stab["max_ee_err"] <= cfg["ee_err_thresh"])
    start_z = float(scene.OBJECT_POS[2])
    # grasp-held: cube must rise AND stay locked to the grasp centre during lift/carry (not flung/left behind).
    _lr = [r for r in telem if r["phase"] in ("lift", "move_above_basket")]
    cube_z_lift = max((r["cube_pos"][2] for r in _lr), default=0.0)
    _near = sum(1 for r in _lr if math.dist(r["cube_pos"], r["ee_pos"]) < 0.08)
    grasp_held = bool(_lr and _near / len(_lr) > 0.7 and cube_z_lift > start_z + 0.04)
    # --- 4-flag success verifier (all four required) ---
    _pre = [r for r in telem if r["phase"] in ("move_above_ball", "descend_to_ball")]
    flag_on_table = bool(_pre and min(r["cube_pos"][2] for r in _pre) > start_z - 0.05)  # cube on table pre-grasp
    flag_held = grasp_held                                                                # locked to grasp centre
    flag_elevated = bool(cube_z_lift > start_z + 0.04)                                    # truly lifted during carry
    flag_in_basket = bool(in_xy and in_z and float(telem[-1]["cube_vel"]) < 0.05)         # inside bounds AND at rest
    flags = {"on_table": flag_on_table, "held": flag_held, "elevated": flag_elevated, "in_basket_at_rest": flag_in_basket}
    success = bool(flag_on_table and flag_held and flag_elevated and flag_in_basket and not cur["fell"])
    metrics = {
        "flags": flags,
        "grasp_held": grasp_held, "cube_z_max_lift": round(cube_z_lift, 4),
        "object_fell": bool(cur["fell"]),
        "task_text": task_text,
        "ball_final_pos": [round(float(v), 4) for v in bp],
        "basket_center": [cx, cy, cfg["basket_floor_z"]],
        "basket_radius": cfg["basket_radius"],
        "basket_rim_z": cfg["basket_rim_z"],
        "basket_floor_z": cfg["basket_floor_z"],
        "xy_off": round(xy_off, 4),
        "in_xy": in_xy, "in_z": in_z,
        "final_grasp_state": grasp_state,
        "max_arm_dq": round(stab["max_dq"], 4),
        "max_ee_err": round(stab["max_ee_err"], 4),
        "arm_q_delta_thresh": cfg["arm_q_delta_thresh"],
        "ee_err_thresh": cfg["ee_err_thresh"],
        "stable": stable,
    }
    msg = "cube placed in basket" if success else f"cube not in basket (xy_off={xy_off:.3f}, z={float(bp[2]):.3f})"
    print(f"[ego2g1] RESULT success={success} | flags={flags} | {msg}")
    result = RunResult(success=success, message=msg, metrics=metrics, steps=log)

    if cfg.get("telemetry"):
        import csv as _csv
        import json as _json
        import os as _os
        p = cfg["telemetry"]
        _os.makedirs(_os.path.dirname(p) or ".", exist_ok=True)
        basket = {"center": list(cfg["basket_center"]), "radius": cfg["basket_radius"],
                  "floor_z": cfg["basket_floor_z"], "rim_z": cfg["basket_rim_z"]}
        with open(p + ".json", "w") as f:
            _json.dump({"metrics": metrics, "basket": basket, "object_start": list(scene.OBJECT_POS),
                        "telemetry": telem}, f, indent=1)
        if telem:
            cols = ["i", "phase", "ee_pos_err", "ee_orient_err", "manip", "jac_cond", "cube_vel"]
            with open(p + ".csv", "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(cols + ["cube_x", "cube_y", "cube_z"])
                for r in telem:
                    w.writerow([r[c] for c in cols] + r["cube_pos"])
        print(f"[ego2g1] telemetry -> {p}.json / .csv ({len(telem)} rows)", flush=True)

    if cfg.get("keep_alive") and not cfg["headless"]:
        print("[ego2g1] view open — close the Isaac Sim window to exit.")
        hold = make_action(*target_for("release", ball_rest))
        while simulation_app.is_running():
            env.step(hold)

    env.close()
    return result
