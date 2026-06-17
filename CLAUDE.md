# CLAUDE.md — dimenso

Persistent project brain. Read this first.

## Goal

Egocentric (first-person) video of a hand doing a pick-and-place → retarget the motion
onto a **fixed-base Unitree G1** humanoid → reproduce the **apple-in-basket** task in
**Isaac Lab**, driven from a **local control panel**.

## Architecture — 4 layers (top calls down)

```
frontend/ (control panel)  →  server/ (FastAPI POST /run)  →  pipeline/ego2g1.run(config)  →  sim/ (Isaac Lab G1 scene)
```

- **frontend/index.html** — static page: app dropdown, video dropdown, Run, log area.
- **server/app.py** — FastAPI, `POST /run` → calls the pipeline. (P4 stub)
- **pipeline/ego2g1.py** — `run(config)`. See the rule below.
- **sim/** — the dimenso-owned Isaac Lab scene + task `Dimenso-AppleBasket-G1-v0`.

### THE RULE
`pipeline.ego2g1.run(config)` is the **single entry point**. The server, the CLI, and
the control panel all call exactly this one function. Keep its signature stable. Inside
it: `video → perception → retarget → IK → sim scene → RunResult`.

## Environment

- **conda env: `isaac_sim`** (Python 3.11). Activate with:
  ```bash
  conda activate isaac_sim
  ```
- Already present (do NOT touch): Isaac Sim 5.1.0, Isaac Lab 0.54.3 (editable, from
  `~/projects/IsaacLab`), torch 2.12.0.dev+cu128, numpy 1.26.0, warp-lang 1.12.0.
- **Added directly to `isaac_sim` for dimenso** (NOT a clone — the clone attempt filled
  the disk and corrupted `~/.claude.json`, so we install in place):
  `mediapipe==0.10.21`, `mujoco==3.9.0`, `opencv-python==4.11.0.86`.
  - Side effect: `protobuf 7.34.0 → 4.25.9` (mediapipe needs `protobuf<5`). Verified
    safe — `onnx` needs only `>=4.25.1`, and `import isaaclab`/`import onnx` still work.
  - **Rollback snapshot:** `~/dimenso_isaac_sim_freeze_backup.txt`
    (`pip install -r` it into `isaac_sim` to restore the pre-dimenso state).

## The G1 task action / observation interface

Extracted from the upstream env we copied,
`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`
(`isaaclab_tasks/.../locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py`
+ its `configs/pink_controller_cfg.py`). **This is the P2 target** — the real control
interface dimenso must produce from retargeted hand motion.

### Action (real upstream task) — Pink IK, ABSOLUTE end-effector pose
`PinkInverseKinematicsActionCfg`. The action is **not** joint torques/positions — it is
target EE poses solved by Pink IK each step:

- **EE pose targets (absolute, in the G1 `pelvis` frame):**
  - left  wrist → EE link `left_wrist_yaw_link`  → 7 numbers: position xyz (3) + quaternion wxyz (4)
  - right wrist → EE link `right_wrist_yaw_link` → 7 numbers: position xyz (3) + quaternion wxyz (4)
  - frame: `base_link_name="pelvis"`; LocalFrameTask base frame `g1_29dof..._pelvis`.
- **Grasp DoF — 14 hand joints** (`num_hand_joints=14`, Inspire tri-hand), driven directly:
  `{left,right}_hand_{index_0, middle_0, thumb_0, index_1, middle_1, thumb_1}` + `{left,right}_hand_thumb_2`.
- **IK-controlled joints** (solved to reach the EE targets): `.*_shoulder_{pitch,roll,yaw}_joint`,
  `.*_elbow_joint`, `.*_wrist_{pitch,roll,yaw}_joint`, `waist_.*_joint`.
- **Action vector ≈ 28** = left pose (7) + right pose (7) + 14 hand joints.

### Observation (real upstream task) — dict group (`concatenate_terms=False`)
`actions` (last action), `robot_joint_pos`, `robot_root_pos`, `robot_root_rot`,
`object_pos`, `object_rot`, `robot_links_state` (all links),
`left_eef_pos`/`left_eef_quat` (`left_wrist_yaw_link`),
`right_eef_pos`/`right_eef_quat` (`right_wrist_yaw_link`),
`hand_joint_state` (`.*_hand.*`), `head_joint_state` (empty),
`object` (object pose relative to both EEFs).

### What the P1 scaffold scene actually uses (placeholder)
We do **not** implement IK in P1. `sim/scene.py` swaps the Pink IK action for a simple
`JointPositionAction` on the arm joints (`.*_shoulder_*`, `.*_elbow_joint`, `.*_wrist_*`
→ 14-dim) with `use_default_offset=True`, so a **zero action holds the default pose
(robot idle)** — enough to prove the scene loads and steps. Obs is reduced to
`joint_pos_rel`, `joint_vel_rel`, apple `root_pos_w`, `last_action`. P2 restores the
real Pink IK action/obs above.

## Assets used (P1 scene)

- **Robot:** `G1_29DOF_CFG` from `isaaclab_assets.robots.unitree`
  (USD `{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd`), `fix_root_link=True` (fixed base).
- **Table:** static cuboid slab `0.80×0.60×0.04 m` (primitive, no download).
- **Apple (graspable object):** dynamic **red sphere**, radius `0.035 m`, mass `0.15 kg`.
- **Basket (target):** static open-topped bin = floor + 4 thin cuboid walls (primitive).
- Only the G1 USD comes from Nucleus; everything else is a primitive, on purpose.

## Hardware ceiling

Lenovo Legion 7i · **RTX 5060 Laptop GPU (Blackwell, 8 GB VRAM)** · Ubuntu 22.04.
→ **headless only, `num_envs=1`, no parallel RL, no GUI.** The disk also runs near-full;
the 20 GB env clone is what broke things — never clone this env.

## DON'Ts

- ❌ **Don't upgrade torch** (the nightly `cu128` build is required for Blackwell sm_120;
  stable torch loses GPU support).
- ❌ **Don't launch the GUI** / non-headless sim (8 GB can't spare it; headless only).
- ❌ **Don't clone or reinstall the `isaac_sim` env** (filled the disk, corrupted
  `~/.claude.json`). Install additions in place; snapshot first.
- ❌ **Don't import task code live from `~/projects/IsaacLab`** at runtime — dimenso owns
  its copy in `sim/scene.py`. The installed `isaaclab`/`isaacsim` libs stay as deps.
- ❌ **Don't import `sim/scene.py` before launching the app** (needs `pxr`).

## Phase plan

- **P1 — scaffold + runnable scene** ✓ (`feat/scaffold`)
- **P2 — perception + retarget** — MediaPipe hands → landmarks; human → G1 EE/grasp
  mapping; wire Isaac Lab Pink IK (restore the real action/obs above).
- **P3 — end-to-end pipeline** — implement `pipeline.ego2g1.run(config)` through all layers.
- **P4 — control panel** — FastAPI `POST /run` + `frontend/index.html`.

## Run

```bash
conda activate isaac_sim
cd ~/projects/dimenso
python sim/run_headless.py            # headless, num_envs=1, ~10 s idle, clean exit
```
