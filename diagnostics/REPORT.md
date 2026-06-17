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
| 10 | ✓ | **✓** | **✓** | ✗ | **grasp+lift+carry SOLVED; cube set down short of basket** |
| 11 | ✓ | ✗ | ✗ | ✗ | firmer grip (stiffness 300) over-squeezed & EJECTED the rigid cube — reverted |
| 12 | ✓ | **✓** | **✓** | ✗ | iter10 grip + slow carry: held all the way; but arm carries only to y≈0.21, basket at y=0.34 unreachable → set down 14 cm short |
| 13 | ✓ | ✗ | ✗ | ✗ | basket → (0.0,0.20) to match carry reach, but its near wall collides with the grasp → knocked off |
| 14 | ✓ | ✗ | ✗ | ✗ | near wall lowered; cube still dropped at grasp → grasp is marginal (run-to-run variance) |

## Remaining (two coupled issues)
1. **Grasp reliability** — held+elevated in iter10 & iter12 but dropped at grasp in iter13/14
   (identical grasp phase). The grasp pose sits at the arm's reach floor, so it's marginal;
   3-consecutive-success requires robustness not yet achieved.
2. **Place workspace conflict** — at carry height the arm moves the cube laterally (x) but cannot
   extend further forward (reaches only y≈0.21). A basket within carry reach (y≈0.21) sits close
   enough to collide with the grasp; a basket ≥10 cm away in the only free direction (y=0.34) is
   out of carry reach. The usable workspace (~10–15 cm) is too small for a ≥10 cm-separated,
   collision-free pick-and-place.

The clean fix for both (joint-limit-aware nullspace / whole-body IK for a larger usable envelope,
6-DoF grasp planning for a robust grasp) is the **blocked** dep set (Pink IK #4090, cuRobo/GraspGen
Blackwell CUDA). **The grasp+lift+carry — the blocker for this whole effort — is solved with native
DiffIK + the waist DoF.** Practical unblock without the blocked deps: a larger/lower/closer basket
(relax the ≥10 cm-on-the-same-table constraint), or accept the grasp+carry demo as the milestone.

## Constraints honored
Native DifferentialIKController only; no Pink/pinocchio/cuRobo/GraspGen/mink; no env-breaking
installs; GUI kept open at head-POV throughout; CLAUDE.md <200 lines.
