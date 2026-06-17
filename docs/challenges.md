# Technical Challenges & Notes

Seeded from the sibling project `~/projects/isaac-hand-teleop` (its `docs/setup_notes.md`
and README "Key Technical Challenges"). These are the gotchas most likely to bite
dimenso too, plus dimenso-specific ones discovered during scaffold.

## Inherited from isaac-hand-teleop

- **RTX 5060 Blackwell (sm_120) needs the torch nightly cu128 build.** Stable torch
  does not support this GPU. The `isaac_sim` env already has `torch 2.12.0.dev+cu128`.
  Do **not** "upgrade" torch — it will break GPU support.
- **Blackwell rendering bug:** if/when rendering with a GUI, use `--renderer PathTracing`
  to avoid a blurry-output bug. (dimenso runs headless, so this rarely applies.)
- **EULA:** Isaac Sim refuses to start without `OMNI_KIT_ACCEPT_EULA=YES`.
  `run_headless.py` sets it automatically.
- **8 GB VRAM ceiling:** the teleop project could only run a single robot instance.
  Same here — `num_envs=1`, headless, no parallel RL.
- **Python version split (teleop):** ROS2 hand tracking ran on system Py 3.10 while
  Isaac ran on conda Py 3.11, bridged over UDP. dimenso avoids this by adding MediaPipe
  directly into the Py 3.11 `isaac_sim` env (one interpreter, no bridge).
- **IK instability / oscillation, depth-from-image limits, bimanual control mapping:**
  documented at length in
  `~/projects/isaac-hand-teleop/docs/Technical Challenges & Interview Prep — Isaac Hand Teleop.pdf`.
  Revisit before implementing the P2 retarget layer.

## dimenso-specific

- **`pxr` import order.** Any module that pulls `isaaclab.sim` / `isaaclab_assets`
  (including `sim/scene.py`) imports `pxr`, which only exists **after** `AppLauncher`
  starts the Kit runtime. Always launch the app first, then import the scene. This is
  why `sim/__init__.py` does NOT auto-import `scene`.
- **protobuf downgrade.** Installing `mediapipe==0.10.21` forced `protobuf 7.34.0 ->
  4.25.9` (mediapipe requires `protobuf<5`). Verified safe: the only dependent is
  `onnx 1.20.1` (`protobuf>=4.25.1`), and `import isaaclab` / `import onnx` both still
  work. Rollback snapshot: `~/dimenso_isaac_sim_freeze_backup.txt`.
- **Primitive scene props.** Table/apple/basket are primitive shapes (cuboid/sphere),
  not Nucleus USD assets, to avoid asset downloads and keep the scene self-contained.
  The G1 robot USD itself still loads from Nucleus (unavoidable; standard G1 asset).
- **Real action is Pink IK, not joint position.** The P1 scene uses a placeholder
  JointPositionAction so the robot can hold idle. The real task action (absolute EE
  pose + grasp DoF via Pink IK) is documented in `CLAUDE.md` and is the P2 target.
