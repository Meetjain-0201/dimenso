"""Measure where the GRASP CENTRE lands when the PALM is driven to its clean pose
(0.14,0.24,0.96, tilt 0.5). The cube must be placed there (not at the palm) so the
fingers close around it. Reports palm pos, grasp-centre pos, and their offset."""
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

PALM_TARGET = (0.14, 0.24, 0.96)   # the clean palm pose from probe_height.py
TILT = 0.5
GC_TIPS = ("right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link")


def main():
    cfg = scene.DimensoAppleBasketG1EnvCfg(); cfg.scene.num_envs = 1
    env = gym.make("Dimenso-AppleBasket-G1-v0", cfg=cfg)
    u = env.unwrapped; dev = u.device; robot = u.scene["robot"]; env.reset()
    ee_idx = robot.body_names.index(scene.EE_BODY)
    gc_idx = [robot.body_names.index(n) for n in GC_TIPS]
    ftip_idx = [robot.body_names.index(n) for n in ("right_hand_middle_1_link", "right_hand_index_1_link")]
    hand_dof = [robot.joint_names.index(j) for j in scene.RIGHT_HAND_JOINTS]
    palm = robot.data.body_pos_w[0, ee_idx]
    ftips = robot.data.body_pos_w[0, ftip_idx].mean(dim=0)
    v = ftips - palm; v = v / v.norm()
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    axis = torch.cross(v, down, dim=0); axis = axis / axis.norm()
    angle = torch.acos(torch.clamp(torch.dot(v, down), -1.0, 1.0))
    q0 = robot.data.body_quat_w[0, ee_idx]
    dq = quat_from_angle_axis((angle * TILT).unsqueeze(0), axis.unsqueeze(0))[0]
    gq = quat_mul(dq.unsqueeze(0), q0.unsqueeze(0))[0]
    hand_hold = torch.zeros((1, len(hand_dof)), device=dev)
    for _ in range(350):
        rp, rq = robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1]
        tp = torch.tensor([PALM_TARGET], device=dev, dtype=torch.float32)
        pb, qb = subtract_frame_transforms(rp, rq, tp, gq.unsqueeze(0))
        env.step(torch.cat([pb, qb, hand_hold], dim=-1))
    palm_w = robot.data.body_pos_w[0, ee_idx].tolist()
    gc_w = robot.data.body_pos_w[0, gc_idx].mean(dim=0).tolist()
    perr = float((torch.tensor(palm_w, device=dev) - torch.tensor(PALM_TARGET, device=dev)).norm())
    print(f"GC_PROBE palm={[round(x,3) for x in palm_w]} (target {PALM_TARGET}, err={perr:.3f})")
    print(f"GC_PROBE grasp_centre={[round(x,3) for x in gc_w]}  <-- PLACE THE CUBE HERE")
    print(f"GC_PROBE palm->gc offset={[round(gc_w[i]-palm_w[i],3) for i in range(3)]}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
