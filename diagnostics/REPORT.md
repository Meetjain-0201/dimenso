# dimenso pick-and-place — diagnostic report (native DifferentialIKController only)

## Headline
The pick-and-place blocker for this whole effort was **kinematic, not perception or tuning**:
the fixed-base Unitree G1 **right arm alone cannot grasp a tabletop cube** — every pose it can
reach pins a joint at its limit. **Fix (within constraints): add the waist DoF to the IK chain.**
With that, **grasp + lift + carry are solved** (iter10: `held`+`elevated`, no drop); the
remaining work is the final place-into-basket leg.

## Root-cause chain (data-driven, `diagnostics/`)
1. `PROBE.md` / `PROBE_HEIGHT.md` (arm-only IK, 7 joints): swept grasp pose × height. **No clean
   operating point at any height** — wherever the arm reaches accurately (err<3 cm) a joint is
   *past* its soft limit (minjlim −0.16 rad); wherever joints are clear, the arm is unreachable
   (err 7–19 cm) and near-singular (manip≈0.0003). A degrees-of-freedom wall.
2. **Fix:** `IK_JOINTS = waist_yaw/pitch/roll + right arm` in `sim/scene.py` (matches the upstream
   G1 pick-place task). The torso turns toward the table so the arm no longer stretches to its
   limits. Re-probe: clean grasp reach err **6 cm → 2 mm**, elbow margin **−0.16 → +0.18 rad**,
   manip 0.005 (non-singular). Native `DifferentialIKController` only — no Pink/cuRobo/GraspGen.
3. **Probe warm-start caveat (`probe_gc.py`):** `env.reset()` does not re-home the arm, so the
   sweep's "clean at z=0.96" was a warm-start artifact. The realistic (cold/trajectory) grasp
   centre **floors at ~0.806 m**. Cube placed AT that floor, y-aligned to where the grasp centre
   lands → the closing Dex3 fingers cage it.

## Scene geometry (final, no riser)
- **Table top z = 0.78 m** (raised; riser deleted — cube rests directly on the tabletop).
- **Cube** at `(0.13, 0.22, 0.805)` — the grasp-centre reach floor, y-aligned to the grasp centre.
- **Basket** at `(0.04, 0.34)`, shallow walls (0.04 m, rim 0.82) since the arm's lift ceiling is
  ~0.86 m. **Cube↔basket gap = 0.150 m** (≥0.10 ✓).
- IK chain = waist (3) + right arm (7); EE = `right_hand_palm_link`; grasp tilt 0.5 (full
  fingers-down is singular everywhere).

## 4-flag verifier (`pipeline/ego2g1.py`)
`success = on_table ∧ held ∧ elevated ∧ in_basket_at_rest ∧ ¬fell`:
- **on_table** — cube on the table pre-grasp (didn't fall during approach)
- **held** — cube locked within 8 cm of the grasp centre through lift+carry
- **elevated** — cube rose >4 cm above its start during carry
- **in_basket_at_rest** — final cube inside basket xy-radius + z-band AND speed <0.05 m/s

## Results
| iter | on_table | held | elevated | in_basket | note |
|------|----------|------|----------|-----------|------|
| 8 | ✓ | ✗ | ✗ | ✗ | gc 18 cm low (frame error: targeted palm-height as grasp-centre) |
| 9 | ✓ | ✗ | ✗ | ✗ | cube below gc floor → clipped on top, knocked off |
| 10 | ✓ | **✓** | **✓** | ✗ | **grasp+lift+carry SOLVED; slipped on final basket reach** |
| 11 | — | — | — | — | firmer grip (stiffness 300, tighter curl) — see ITERATION_LOG / will append |

## Remaining
The place leg: the basket pose twists the wrist (j4) past its soft limit, prying the grip loose
during the last ~14 cm of carry (iter10 slipped at y≈0.20). Levers tried/next: firmer grip
(iter11); a place pose with less wrist twist; slower carry. The strict "every joint >0.15 rad
clear" at BOTH grasp and place is not simultaneously achievable for this arm — the proper fix
(joint-limit-aware nullspace / whole-body IK, 6-DoF grasp planning) is the blocked dep set
(Pink IK #4090, cuRobo/GraspGen Blackwell CUDA). Grasp itself is solved with native DiffIK.

## Constraints honored
Native DifferentialIKController only; no Pink/pinocchio/cuRobo/GraspGen/mink; no env-breaking
installs; GUI kept open at head-POV throughout; CLAUDE.md <200 lines.
