# Copyright (c) 2026, dimenso project.

"""Launch the dimenso G1 apple-in-basket scene headless, hold it idle, exit cleanly.

This is the smoke test for the P1 scaffold: it proves the scene loads and physics
steps on the 8 GB GPU. The robot receives a zero action every step, so it holds its
default pose (idle) — no IK, no teleop. Real control arrives in P2/P3.

Usage:
    conda activate isaac_sim
    cd ~/projects/dimenso
    python sim/run_headless.py                # 10 s, headless
    python sim/run_headless.py --seconds 5    # shorter
"""

import argparse
import os
import sys
from pathlib import Path

# Isaac Sim refuses to start without explicit EULA acceptance.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher  # noqa: E402

# --- CLI / app launch -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run the dimenso G1 scene headless.")
parser.add_argument("--seconds", type=float, default=10.0, help="Seconds to hold the sim.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force the hardware ceiling: headless, single environment.
args.headless = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Everything below needs a launched app (pxr / isaaclab.sim) -------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# Import the scene module so `Dimenso-AppleBasket-G1-v0` gets registered. dimenso
# owns this task; we never import the env from ~/projects/IsaacLab at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402


def main() -> int:
    # Isaac Lab does not auto-resolve `env_cfg_entry_point` into `cfg`; instantiate
    # our cfg and pass it explicitly (num_envs forced to 1 for the 8 GB ceiling).
    env_cfg = scene.DimensoAppleBasketG1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("Dimenso-AppleBasket-G1-v0", cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        step_dt = unwrapped.step_dt  # control dt = sim.dt * decimation
        num_steps = max(1, int(args.seconds / step_dt))
        action_dim = unwrapped.action_manager.total_action_dim
        zero_action = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)

        print(
            f"[dimenso] scene loaded: G1 fixed-base + table + apple + basket | "
            f"num_envs={unwrapped.num_envs} action_dim={action_dim} step_dt={step_dt:.4f}s "
            f"-> holding idle for {args.seconds:.1f}s ({num_steps} steps)"
        )
        env.reset()
        for i in range(num_steps):
            env.step(zero_action)
            if (i + 1) % 50 == 0:
                print(f"[dimenso]   stepped {i + 1}/{num_steps}")
        print("[dimenso] idle hold complete — scene ran cleanly.")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
