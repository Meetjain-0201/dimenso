# ROOTCAUSE — diagnostic-driven (native DiffIK)

Ground truth from `diagnostics/instrument.py` (3180 telemetry rows, respawn OFF) →
`diagnostics/analyze.py`. Run: `diag0`. Every fix below must be checked against these.

## Detectors that fired (with governing numbers)

| Detector | Numbers | Meaning |
|---|---|---|
| **near_singular** | `manip≈7e-5` (≈0, singular), `jac_cond=3710`, **`right_wrist_roll` (arm_joint[4]) = 0.197 rad PAST its soft limit** | At the fingers-down grasp target the arm is driven into a singular, joint-limit-violating config. **This is the root cause.** |
| **palm_short** | `ee_pos_err = 0.189 m`, orientation_err = 0.157 rad → **position-dominated** | The palm physically can't reach the cube — it's a *reach* problem, not orientation. Direct consequence of the singular/limited config above. |
| **cube_knocked** | `move_above_ball` step 426, cube xy jumped `0.052 m` (cube_vel 0.36) | The approach grazes the cube (~5 cm). Secondary. |
| **cube_fell** | `move_above_basket` step 2239, cube z = `0.656 m` | Cube lost during carry. |
| **slip / fling** | cube rose to `0.941 m` then flew to final `xy_off = 4.33 m`, `ball_final=[-3.83,1.52,0.025]`, object_fell=True | NOT a real grasp — the contorted/singular arm **flung** the cube. The `grasp_held` heuristic was fooled by the fling. |
| **release_miss** | final `xy_off = 4.33 m` ≫ basket radius 0.085 m | Cube nowhere near the basket. |

## Conclusion
The auto **fingers-down orientation at the cube position is kinematically infeasible**
for the G1 right arm with native DiffIK: it forces `right_wrist_roll` past its soft
limit and the Jacobian to a singularity (`manip≈0`), so DiffIK can't track it
(`palm_short 0.19 m`, position-dominated) and the arm flails/flings the cube.

Stability metrics: `max_arm_dq=0.61`, `max_ee_err=0.24`, `stable=False` — note the
*carry* itself was smooth earlier; the swing here is the singular pick, not the carry.

## What the fixes must achieve (checked against the detectors)
1. **near_singular / palm_short → reachable placement.** Probe the workspace for a cube
   position where a fingers-down (or feasibly-tilted) pose is reachable with `manip` well
   above 0, all joints inside limits, and `ee_pos_err < 0.03 m`. (`diagnostics/probe.py`)
   - **If NO position yields a reachable full fingers-down pose**, the wrist can't achieve
     it → switch to an **angled/tilted grasp** (per HAT: vertical grasp is the hard case;
     a tilted approach is reachable). Decide from probe data, not guessing.
2. **cube_knocked → vertical descent.** Pre-orient at a safe height, then descend with
   only z changing (no lateral motion near the cube). Verify `cube_knocked` never fires.
3. **slip/fling → real grasp.** Once reachable, tune descent depth + Dex3 close + friction
   until the cube rises *with the palm* (cube-follows-palm), not flung, and is held through
   carry. (Replace the fling-foolable heuristic with a cube-near-palm check.)
