"""Workspace probe — find the reachable fingers-down 'sweet spot' for the cube.

Sweeps candidate cube (x,y) positions in front of the right arm, commands the DiffIK to a
fingers-down palm pose just above each, lets it converge, and records EE position error +
manipulability + nearest joint-limit distance. Prints a ranked table and the best spot.

Usage: python diagnostics/probe.py
"""

import argparse
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a = _p.parse_args([])
_a.headless = True
_app = AppLauncher(_a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
import scene  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROBE.md")
TABLE_TOP = 0.70
GRASP_CLEAR = 0.10  # palm this far above the candidate (cube) z


def main():
    cfg = scene.DimensoAppleBasketG1EnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("Dimenso-AppleBasket-G1-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    robot = u.scene["robot"]
    env.reset()
    ee_idx = robot.body_names.index(scene.EE_BODY)
    ftip_idx = [robot.body_names.index(n) for n in ("right_hand_middle_1_link", "right_hand_index_1_link")]
    arm_idx = [robot.joint_names.index(j) for j in scene.RIGHT_ARM_JOINTS]
    hand_dof = [robot.joint_names.index(j) for j in scene.RIGHT_HAND_JOINTS]
    jl = robot.data.soft_joint_pos_limits[0]
    arm_lim = [(float(jl[i, 0]), float(jl[i, 1])) for i in arm_idx]

    # fingers-down quat from the current pose
    palm = robot.data.body_pos_w[0, ee_idx]
    ftips = robot.data.body_pos_w[0, ftip_idx].mean(dim=0)
    v = ftips - palm
    v = v / v.norm()
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    axis = torch.cross(v, down, dim=0)
    axis = axis / axis.norm()
    angle = torch.acos(torch.clamp(torch.dot(v, down), -1.0, 1.0))
    q0 = robot.data.body_quat_w[0, ee_idx]
    grasp_q = quat_mul(quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0))[0].unsqueeze(0), q0.unsqueeze(0))[0]

    def manip():
        try:
            J = robot.root_physx_view.get_jacobians()
            jb = ee_idx - 1 if J.shape[1] == (robot.num_bodies - 1) else ee_idx
            s = torch.linalg.svdvals(J[0, jb][:, arm_idx].float())
            return float(torch.prod(s))
        except Exception:
            return float("nan")

    hand_hold = torch.zeros((1, len(hand_dof)), device=dev)

    def quat_for_tilt(f):
        # f=1 -> full fingers-down; f<1 -> tilted toward down from the default orientation
        dq = quat_from_angle_axis((angle * f).unsqueeze(0), axis.unsqueeze(0))[0]
        return quat_mul(dq.unsqueeze(0), q0.unsqueeze(0))[0]

    def goto(pos, gq, n=160):
        for _ in range(n):
            rp, rq = robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1]
            tp = torch.tensor([pos], device=dev, dtype=torch.float32)
            pb, qb = subtract_frame_transforms(rp, rq, tp, gq.unsqueeze(0))
            env.step(torch.cat([pb, qb, hand_hold], dim=-1))

    # sweep grasp tilt (1.0=fully down ... 0.0=default) x a few near-arm positions
    results = []
    tilts = [1.0, 0.75, 0.5, 0.25, 0.0]
    cands = [(0.08, 0.20), (0.16, 0.20), (0.08, 0.28), (0.16, 0.28), (0.0, 0.20)]
    z = TABLE_TOP + GRASP_CLEAR
    for f in tilts:
        gq = quat_for_tilt(torch.tensor(f, device=dev))
        for (x, y) in cands:
            env.reset()
            goto((x, y, z), gq)
            ee = robot.data.body_pos_w[0, ee_idx]
            perr = float((ee - torch.tensor([x, y, z], device=dev)).norm())
            q = robot.data.joint_pos[0, arm_idx]
            jdist = min(min(float(q[i]) - arm_lim[i][0], arm_lim[i][1] - float(q[i])) for i in range(7))
            mp = manip()
            results.append((round(perr, 3), round(mp, 4), round(jdist, 3), f, x, y))
            print(f"  tilt={f:.2f} ({x:.2f},{y:.2f}) err={perr:.3f} manip={mp:.4f} jlim={jdist:.3f}", flush=True)

    # reachable = in-limits + low error; among those prefer the MOST downward tilt, then manip
    reachable = [r for r in results if r[0] < 0.04 and r[2] > 0.05]
    ranked = sorted(reachable or results, key=lambda r: (-r[3], r[0], -r[1]))  # max tilt, low err
    best = ranked[0]
    feasible = bool(reachable)
    with open(OUT, "w") as f_:
        f_.write("# Workspace probe — grasp tilt x position reachability (palm 0.10 m above candidate)\n\n")
        f_.write("tilt: 1.0=fingers fully down, 0.0=default orientation. err=EE pos error (m); "
                 "manip=sqrt(det(JJᵀ)); jlim=nearest joint-limit dist (rad, <0 = past limit).\n\n")
        f_.write("| tilt | x | y | err | manip | jlim |\n|---|---|---|---|---|---|\n")
        for perr, mp, jd, fl, x, y in sorted(results, key=lambda r: (-r[3], r[0])):
            f_.write(f"| {fl:.2f} | {x:.2f} | {y:.2f} | {perr:.3f} | {mp:.4f} | {jd:.3f} |\n")
        f_.write(f"\n**Reachable fingers-down feasible: {feasible}**\n\n")
        f_.write(f"**Chosen: tilt={best[3]:.2f}, x={best[4]:.2f}, y={best[5]:.2f}** "
                 f"(err={best[0]} m, manip={best[1]}, jlim={best[2]} rad) — "
                 f"{'most-downward tilt that stays in-limits and reachable' if feasible else 'NO in-limit reachable pose found; least-bad'}.\n")
    print(f"PROBE BEST tilt={best[3]:.2f} x={best[4]:.2f} y={best[5]:.2f} err={best[0]} manip={best[1]} jlim={best[2]} feasible={feasible}")
    print(f"PROBE -> {OUT}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
