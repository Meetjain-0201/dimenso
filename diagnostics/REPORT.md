# Overnight diagnostic-driven pick fix — REPORT

**Branch:** `feat/integrate-pipeline` · native DiffIK only (no Pink/pinocchio/cuRobo/GraspGen/mink — all blocked).
**Loop:** instrument → analyze → root-cause → targeted fix → rerun. **Stopped at iteration 4/20**
(graceful — single well-characterized remaining cause; no thrashing; no blocked solvers touched).

## Result: NOT green yet — but the carry/approach/stability are SOLVED and the remaining failure is isolated to one cause with a clean fix.

| metric (iter4, best) | value | status |
|---|---|---|
| stability (`max_arm_dq`) | 0.127 (< 0.15) | ✅ no jitter |
| stability (`max_ee_err`) | 0.076 m (< 0.12) | ✅ tracks |
| `stable` | **True** | ✅ |
| near-singular | manip 0.0099, jac_cond ~50, joints in-limits | ✅ resolved |
| cube knocked on approach | not fired | ✅ resolved |
| fling | not fired (object_fell=False) | ✅ resolved |
| **grasp_held** | **False** | ❌ remaining |
| cube in basket | xy_off 0.176 m | ❌ (consequence of no grasp) |

## Root cause (from `ROOTCAUSE.md`, ground-truth telemetry)
Baseline (`diag0`): the auto **fingers-down grasp was kinematically infeasible** — at the
grasp target the right wrist was driven **0.197 rad past its soft limit**, **manipulability ≈ 7e-5
(singular)**, `jac_cond=3710`; DiffIK couldn't track it (`palm_short 0.189 m`, position-dominated)
so the contorted/singular arm **flung** the cube (to z=0.94 then 4.3 m away).

## What each fix did (data-checked)
1. **Tilt the grasp (probe-driven).** `diagnostics/probe.py` swept grasp-tilt × position:
   full fingers-down is singular **everywhere** (manip≈1e-4); **tilt=0.5 at (0.16, 0.20) is
   reachable & in-limits** (err 0.024 m, manip 0.0086, joint-limit margin 0.66 rad). → set
   `grasp_tilt=0.5`. **Killed near_singular + palm_short.**
2. **Reachable placement.** Cube → (0.16, 0.20), basket → (0.04, 0.32) — both in the probed
   reachable region.
3. **Vertical descent.** Approach goes to a waypoint 0.18 m directly above the cube then
   descends with xy fixed. **Killed cube_knocked.**
4. **Raise the cube (riser).** The decisive kinematic finding: the fixed-base G1 right arm can
   only reach a (tilted) grasp pose down to palm-z ≈ **0.80–0.82 m**; below that it re-enters
   singularity (iter2: manip 2e-4). The cube on the table top (z 0.726, top 0.75) left a
   **~10 cm palm-to-cube gap the short Dex3 fingers can't bridge**. Putting the cube on a small
   **riser (top 0.78 m)** lifts it into the reachable band → at grasp the palm is now **5.6 cm**
   from the cube and the whole motion is **stable** (no fling, no jitter).

## The one remaining failure: grasp closure (`grasp_empty`)
At grasp, palm 5.6 cm from the cube, fingers curl, but at lift the cube stays at 0.805 m while
the palm rises (cube not caged). **Best hypothesis (data-backed):** the IK controls the
**palm link**, but the Dex3's actual **grasp center is ~5–6 cm further out** along the finger
direction — so commanding the palm onto the cube leaves the cube *just outside* the closing
fingers. The fingers close in front of/beside the cube, not around it.

## Options for the morning (pick one — all native DiffIK)
1. **(Recommended, ~1–2 iters) Add an IK `body_offset`** to `DifferentialInverseKinematicsActionCfg`
   so the controlled point is the **finger-grasp center** (palm + ~0.05–0.06 m along the local
   finger/approach axis), not the palm link. Then commanding it onto the cube puts the fingers
   *around* the cube. Re-probe reach for the offset point. This directly closes the 5.6 cm gap.
2. **Tune palm pose + finger timing for closure**: drop `grasp_palm_above` 0.04 → ~0.01–0.02 and
   close fingers *during* the last descent cm (pre-shape), so the fingers cage the cube as the
   palm arrives. A few iterations of fine tuning.
3. **Easier-to-cage object / pinch primitive**: a smaller object (≤3.5 cm) or a 2-finger pinch
   (thumb vs index+middle) closing on a thin object — the 3-finger cage of a 5 cm cube is the
   finickiest case.

## Notes / constraints honored
- **Grasp detection = kinematic** (cube rises *and stays near the palm*), not hardware contact
  sensors — deliberately, to avoid mutating the robot/cube spawn and risking the env (per the
  no-risky-changes constraint). It correctly distinguishes empty/fling/held.
- **Riser caveat:** it elevates the cube off the flat table. If a flat-table grasp is required,
  Option 1 may still not reach low enough — that would confirm the fixed-base G1 right arm
  *cannot* top-down grasp a flat-table object without going singular, and the proper solution is
  the upstream **wrist-target + posture-regularized Pink IK** (currently blocked by Isaac Lab
  **#4090** on this Blackwell/Sim 5.1 stack) or a lowered/mobile base. Flagging, not installing.

## Tooling left in place (`diagnostics/`)
`instrument.py` (per-step telemetry JSON+CSV) · `analyze.py` (event detectors) · `probe.py`
(reachability sweep) · `ROOTCAUSE.md` · `PROBE.md` · `ITERATION_LOG.md` · `telemetry/` (diag0,
iter1–4) · `replays/` (camera renders).
