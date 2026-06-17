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
JLIM_OK = 0.15      # require every arm joint at least this far (rad) from its limit
ERR_OK = 0.03       # EE position error must be under this (m) to count as reachable


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

    def probe_pose(x, y, z, f):
        gq = quat_for_tilt(torch.tensor(f, device=dev))
        env.reset()
        goto((x, y, z), gq)
        ee = robot.data.body_pos_w[0, ee_idx]
        perr = float((ee - torch.tensor([x, y, z], device=dev)).norm())
        q = robot.data.joint_pos[0, arm_idx]
        per_j = [min(float(q[i]) - arm_lim[i][0], arm_lim[i][1] - float(q[i])) for i in range(7)]
        jdist = min(per_j)
        return round(perr, 3), round(manip(), 4), round(jdist, 3), per_j.index(jdist)

    # --- GRASP sweep: cube sits at z = scene grasp height; find the arm's clean region. ---
    gz = scene.OBJECT_POS[2]
    results = []
    tilts = [0.6, 0.5, 0.4]
    xs = [0.00, 0.08, 0.16, 0.24]
    ys = [0.18, 0.22, 0.26, 0.30, 0.34]
    for f in tilts:
        for x in xs:
            for y in ys:
                perr, mp, jd, jidx = probe_pose(x, y, gz, f)
                results.append((perr, mp, jd, jidx, f, x, y))
                print(f"  grasp tilt={f:.2f} ({x:.2f},{y:.2f},{gz:.3f}) err={perr:.3f} manip={mp:.4f} minjlim={jd:.3f}@j{jidx}", flush=True)

    # clean = reachable (low err) AND every joint clear of its limit by >= JLIM_OK
    clean = [r for r in results if r[0] < ERR_OK and r[2] > JLIM_OK]
    ranked = sorted(clean or results, key=lambda r: (-r[2], r[0], -r[1]))  # most joint clearance, low err, high manip
    best = ranked[0]

    # --- BASKET place reachability: same clean test at carry/place height over basket candidates. ---
    pz = scene.TABLE_TOP + scene.BASKET_WALL_H + 0.04  # grasp-centre just above the rim
    basket_cands = [(0.05, 0.34), (0.00, 0.34), (0.08, 0.34), (0.05, 0.30), (0.05, 0.38), (-0.02, 0.32)]
    bresults = []
    for (x, y) in basket_cands:
        perr, mp, jd, jidx = probe_pose(x, y, pz, best[4])
        bresults.append((perr, mp, jd, jidx, x, y))
        print(f"  place ({x:.2f},{y:.2f},{pz:.3f}) err={perr:.3f} manip={mp:.4f} minjlim={jd:.3f}@j{jidx}", flush=True)
    bclean = [r for r in bresults if r[0] < ERR_OK and r[2] > JLIM_OK]
    bbest = sorted(bclean or bresults, key=lambda r: (-r[2], r[0]))[0]

    feasible = bool(clean)
    with open(OUT, "w") as f_:
        f_.write(f"# Workspace probe — joint-limit-clear grasp + place poses (grasp z={gz:.3f}, place z={pz:.3f})\n\n")
        f_.write(f"Clean = EE err < {ERR_OK} m AND every arm joint > {JLIM_OK} rad from its limit. "
                 "minjlim<0 = a joint pinned past its limit (target unreachable).\n\n")
        f_.write("## Grasp poses (cube)\n\n| tilt | x | y | err | manip | minjlim | @joint | clean |\n|---|---|---|---|---|---|---|---|\n")
        for perr, mp, jd, jidx, fl, x, y in sorted(results, key=lambda r: (-r[2], r[0])):
            f_.write(f"| {fl:.2f} | {x:.2f} | {y:.2f} | {perr:.3f} | {mp:.4f} | {jd:.3f} | j{jidx} | {'YES' if perr<ERR_OK and jd>JLIM_OK else ''} |\n")
        f_.write(f"\n**Clean grasp pose exists: {feasible}**\n")
        f_.write(f"**Chosen cube pose: tilt={best[4]:.2f}, x={best[5]:.2f}, y={best[6]:.2f}** "
                 f"(err={best[0]} m, manip={best[1]}, minjlim={best[2]} rad @j{best[3]}).\n\n")
        f_.write("## Place poses (basket centre)\n\n| x | y | err | manip | minjlim | @joint | clean |\n|---|---|---|---|---|---|---|\n")
        for perr, mp, jd, jidx, x, y in sorted(bresults, key=lambda r: (-r[2], r[0])):
            f_.write(f"| {x:.2f} | {y:.2f} | {perr:.3f} | {mp:.4f} | {jd:.3f} | j{jidx} | {'YES' if perr<ERR_OK and jd>JLIM_OK else ''} |\n")
        f_.write(f"\n**Chosen basket centre: x={bbest[4]:.2f}, y={bbest[5]:.2f}** "
                 f"(err={bbest[0]} m, manip={bbest[1]}, minjlim={bbest[2]} rad @j{bbest[3]}).\n")
    print(f"PROBE GRASP best tilt={best[4]:.2f} x={best[5]:.2f} y={best[6]:.2f} err={best[0]} minjlim={best[2]}@j{best[3]} clean={feasible}")
    print(f"PROBE PLACE best x={bbest[4]:.2f} y={bbest[5]:.2f} err={bbest[0]} minjlim={bbest[2]}@j{bbest[3]}")
    print(f"PROBE -> {OUT}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
