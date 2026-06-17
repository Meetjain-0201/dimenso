"""Place probe — with waist DoF, find a basket position >=10cm from the clean grasp
pose (0.14,0.24) that the arm can reach at carry/place height. Warm-converged."""
import os, sys, math
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

CUBE_XY = (0.14, 0.24)


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
    dq = quat_from_angle_axis((angle * 0.5).unsqueeze(0), axis.unsqueeze(0))[0]
    gq = quat_mul(dq.unsqueeze(0), q0.unsqueeze(0))[0]

    def probe(pos, n=300):
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
        return round(perr, 3), round(jd, 3), per_j.index(jd)

    cands = [(0.04, 0.24), (0.04, 0.30), (0.04, 0.18), (0.14, 0.34), (0.04, 0.34),
             (0.24, 0.24), (0.14, 0.14), (0.06, 0.32), (0.00, 0.28)]
    z = 1.00  # carry/place height (grasp z 0.96 + a little, above a short basket rim)
    rows = []
    for (x, y) in cands:
        gap = math.dist((x, y), CUBE_XY)
        if gap < 0.10:
            continue
        env.reset()
        perr, jd, jidx = probe((x, y, z))
        rows.append((perr, jd, jidx, x, y, round(gap, 3)))
        print(f"  place ({x:.2f},{y:.2f},{z:.2f}) gap={gap:.3f} err={perr:.3f} minjlim={jd:.3f}@j{jidx}", flush=True)
    # reachable place: low err; prefer joint margin and bigger gap
    ok = [r for r in rows if r[0] < 0.04]
    best = sorted(ok or rows, key=lambda r: (r[0], -r[1]))[0]
    print(f"PLACE BEST x={best[3]:.2f} y={best[4]:.2f} gap={best[5]} err={best[0]} minjlim={best[1]}@j{best[2]} reachable={best[0]<0.04}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
