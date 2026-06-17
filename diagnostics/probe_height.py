"""Height probe — sweep grasp-centre Z (table height) to find a band where the fixed-base
G1 right arm reaches the cube AccuratELY *and* with joint-limit margin and decent manipulability.

Warm-converged (no reset between poses, 300 steps) so it reflects what the pipeline can hold,
not a cold-start transient. Usage: python diagnostics/probe_height.py
"""
import os, sys
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher
import argparse
_p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(_p)
_a = _p.parse_args([]); _a.headless = True
_app = AppLauncher(_a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, subtract_frame_transforms  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
import scene  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROBE_HEIGHT.md")
JLIM_OK, ERR_OK, MANIP_OK = 0.15, 0.03, 0.004


def main():
    cfg = scene.DimensoAppleBasketG1EnvCfg(); cfg.scene.num_envs = 1
    env = gym.make("Dimenso-AppleBasket-G1-v0", cfg=cfg)
    u = env.unwrapped; dev = u.device; robot = u.scene["robot"]; env.reset()
    ee_idx = robot.body_names.index(scene.EE_BODY)
    ftip_idx = [robot.body_names.index(n) for n in ("right_hand_middle_1_link", "right_hand_index_1_link")]
    arm_idx = [robot.joint_names.index(j) for j in scene.RIGHT_ARM_JOINTS]
    hand_dof = [robot.joint_names.index(j) for j in scene.RIGHT_HAND_JOINTS]
    jl = robot.data.soft_joint_pos_limits[0]
    arm_lim = [(float(jl[i, 0]), float(jl[i, 1])) for i in arm_idx]
    palm = robot.data.body_pos_w[0, ee_idx]
    ftips = robot.data.body_pos_w[0, ftip_idx].mean(dim=0)
    v = ftips - palm; v = v / v.norm()
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    axis = torch.cross(v, down, dim=0); axis = axis / axis.norm()
    angle = torch.acos(torch.clamp(torch.dot(v, down), -1.0, 1.0))
    q0 = robot.data.body_quat_w[0, ee_idx]
    hand_hold = torch.zeros((1, len(hand_dof)), device=dev)

    def manip():
        try:
            J = robot.root_physx_view.get_jacobians()
            jb = ee_idx - 1 if J.shape[1] == (robot.num_bodies - 1) else ee_idx
            s = torch.linalg.svdvals(J[0, jb][:, arm_idx].float())
            return float(torch.prod(s))
        except Exception:
            return float("nan")

    def quat_for_tilt(f):
        dq = quat_from_angle_axis((angle * f).unsqueeze(0), axis.unsqueeze(0))[0]
        return quat_mul(dq.unsqueeze(0), q0.unsqueeze(0))[0]

    def probe(pos, gq, n=300):
        for _ in range(n):
            rp, rq = robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1]
            tp = torch.tensor([pos], device=dev, dtype=torch.float32)
            pb, qb = subtract_frame_transforms(rp, rq, tp, gq.unsqueeze(0))
            env.step(torch.cat([pb, qb, hand_hold], dim=-1))
        ee = robot.data.body_pos_w[0, ee_idx]
        perr = float((ee - torch.tensor(pos, device=dev)).norm())
        q = robot.data.joint_pos[0, arm_idx]
        per_j = [min(float(q[i]) - arm_lim[i][0], arm_lim[i][1] - float(q[i])) for i in range(7)]
        jd = min(per_j)
        return round(perr, 3), round(manip(), 4), round(jd, 3), per_j.index(jd)

    # near-arm xy candidates (right arm, in front), sweep grasp z (=> table height) and tilt
    cands = [(0.10, 0.22), (0.14, 0.24), (0.10, 0.26), (0.16, 0.22), (0.06, 0.24)]
    zs = [0.84, 0.88, 0.92, 0.96]
    tilts = [0.5, 0.35]
    rows = []
    gq_cache = {f: quat_for_tilt(torch.tensor(f, device=dev)) for f in tilts}
    for z in zs:
        for f in tilts:
            for (x, y) in cands:
                perr, mp, jd, jidx = probe((x, y, z), gq_cache[f])
                clean = perr < ERR_OK and jd > JLIM_OK and mp > MANIP_OK
                rows.append((z, f, x, y, perr, mp, jd, jidx, clean))
                print(f"  z={z:.2f} tilt={f:.2f} ({x:.2f},{y:.2f}) err={perr:.3f} manip={mp:.4f} minjlim={jd:.3f}@j{jidx} {'CLEAN' if clean else ''}", flush=True)

    clean = [r for r in rows if r[8]]
    ranked = sorted(clean or rows, key=lambda r: (-int(r[8]), r[4], -r[6]))
    b = ranked[0]
    with open(OUT, "w") as fo:
        fo.write(f"# Height probe — clean grasp band (clean = err<{ERR_OK} & minjlim>{JLIM_OK} & manip>{MANIP_OK})\n\n")
        fo.write("| z | tilt | x | y | err | manip | minjlim | @j | clean |\n|---|---|---|---|---|---|---|---|---|\n")
        for z, f, x, y, perr, mp, jd, jidx, cl in sorted(rows, key=lambda r: (-int(r[8]), r[4])):
            fo.write(f"| {z:.2f} | {f:.2f} | {x:.2f} | {y:.2f} | {perr:.3f} | {mp:.4f} | {jd:.3f} | j{jidx} | {'YES' if cl else ''} |\n")
        fo.write(f"\n**Clean band exists: {bool(clean)}**\n")
        fo.write(f"**Chosen: z={b[0]:.2f} tilt={b[1]:.2f} x={b[2]:.2f} y={b[3]:.2f}** "
                 f"(err={b[4]} manip={b[5]} minjlim={b[6]}@j{b[7]})\n")
    print(f"HEIGHT BEST z={b[0]:.2f} tilt={b[1]:.2f} x={b[2]:.2f} y={b[3]:.2f} err={b[4]} manip={b[5]} minjlim={b[6]}@j{b[7]} cleanband={bool(clean)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
