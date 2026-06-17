# dimenso

**Egocentric video → Unitree G1 pick-and-place in Isaac Lab.**

> Watch a first-person video of a hand picking an apple and dropping it in a basket;
> retarget that motion onto a fixed-base Unitree G1 humanoid and reproduce the
> pick-and-place in NVIDIA Isaac Lab. A local control panel drives it all.

> **Status: P1 scaffold.** Perception, retargeting, and IK are stubs
> (`NotImplementedError`). The one real, runnable piece is the Isaac Lab scene.

---

## Architecture (4 layers)

```
[ frontend/  ]  local control panel (static page: pick app + video, Run)
       │  HTTP
[ server/    ]  FastAPI: POST /run
       │
[ pipeline/  ]  ego2g1.run(config)  ← the ONE entry point everything calls
       │
[ sim/       ]  Isaac Lab scene: Dimenso-AppleBasket-G1-v0 (G1 + table + apple + basket)
```

Data flow inside `pipeline.ego2g1.run`:
`video → perception (MediaPipe hands) → retarget (kinematics + IK) → G1 scene → result`.

## Layout

| Path | Role | Status |
|---|---|---|
| `sim/scene.py` | G1 fixed-base + table + apple + basket; registers `Dimenso-AppleBasket-G1-v0` | **real** |
| `sim/run_headless.py` | launch headless, hold idle ~10 s, exit | **real** |
| `perception/` | video → 21-kpt hand landmarks (.npz) | stub (P2) |
| `retarget/` | landmarks → G1 EE target + grasp → joints | stub (P2) |
| `pipeline/ego2g1.py` | single `run(config)` entry point | stub (P3) |
| `server/app.py` | FastAPI `POST /run` | stub (P4) |
| `frontend/index.html` | control panel | stub (P4) |
| `scripts/record_clip.py` | webcam → mp4 | stub (P2) |
| `configs/apple_in_basket.yaml` | app config | done |

## Environment

Uses the existing **`isaac_sim`** conda env (Isaac Sim 5.1 + Isaac Lab 0.54.3 +
torch 2.12 nightly cu128). MediaPipe / MuJoCo / OpenCV were added directly to it.
**Do not clone or reinstall this env.** See `CLAUDE.md` for the full rationale,
rollback snapshot, and hardware constraints.

```bash
conda activate isaac_sim
```

## Run the scene (P1 smoke test)

```bash
conda activate isaac_sim
cd ~/projects/dimenso
python sim/run_headless.py            # headless, num_envs=1, ~10 s idle, clean exit
```

## Phase plan

- **P1 — scaffold + runnable scene** ✓ (this branch)
- **P2 — perception + retarget** (MediaPipe hands, human→G1 mapping, Pink IK)
- **P3 — end-to-end pipeline** (`ego2g1.run` wired through all layers)
- **P4 — local control panel** (FastAPI server + frontend)

## Hardware

Lenovo Legion 7i · RTX 5060 Laptop GPU (Blackwell, **8 GB**) · Ubuntu 22.04.
8 GB → **headless, `num_envs=1`, no parallel RL, no GUI.**
