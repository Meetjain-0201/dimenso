# Iteration log — diagnostic-driven pick fix (native DiffIK, cap=20)

Baseline (diag0): near_singular (full fingers-down infeasible: manip≈0, right_wrist_roll past limit),
palm_short 0.19m, cube flung. Probe → tilt=0.5 @ (0.16,0.20) reachable (manip 0.0086, jlim 0.66).

| iter | failing detector(s) | change applied |
|------|---------------------|----------------|
| 1 | grasp_empty (reach/singular/knock all fixed) | tilt=0.5, cube→(0.16,0.20), basket→(0.04,0.32), approach 0.18, vertical descent |
| 2 | grasp_empty + near_singular at low clearance | dropped clearance to 0.04 -> singular; fingers under-curled |
| 3 | grasp_empty (kinematic wall: palm reachable ~0.82, cube top 0.75 -> ~10cm gap; Dex3 too short; lower=singular) | fuller finger curl + clearance 0.08 (singular gone, manip 0.0099) but gap remains |
| 4 | grasp_empty (closure) | cube raised on riser to reachable band -> stable=True, palm 5.6cm from cube, no fling; but Dex3 still doesn't cage cube (IK targets palm link, grasp center is ~5-6cm further out) |

**Stopped at iteration 4 (graceful, well under the 20 cap): remaining failure is a single,
well-characterized cause — grasp closure / control-point offset — with a clean fix (below).
Did NOT thrash; did NOT touch any blocked solver.**
| 5 | grasp_empty + cube knocked off riser (open-loop, respawn off) | grasp-center offset + stiff Dex3 fingers; cube fell, robot oblivious |
| 6 | grasp_empty (gc 5.8cm off, elbow j3 at limit) | live cube tracking + every-step respawn; cube recovered to riser but grasp gc still off + near-singular |
| 7 | near_singular: joints PAST limits (minjlim -0.20) | removed riser, raised table to 0.78, pulled cube to (0.18,0.16) — WRONG direction, joints worse |
| — | RE-PROBE (PROBE.md/PROBE_HEIGHT.md): arm-only DiffIK has NO clean pose at any height — reach⟺joint-limit tradeoff, manip≈0 where joints clear | diagnosis: degrees-of-freedom wall |
| — | FIX: add **waist_yaw/pitch/roll to IK chain** (IK_JOINTS, matches upstream) — torso turns to table | clean grasp unlocked: cube reach err 6cm→2mm, elbow −0.16→+0.18 rad |
| 8 | held/elevated False; ee_err 0.178 (gc 18cm low) | warm-start probe height (0.96) was a reset artifact; gc floors at ~0.806 on real trajectory |
| 9 | cube knocked off at grasp (gc 2.6cm above cube) | cube at 0.78 was below gc floor → fingers clipped top; |
| 10 | **GRASP SOLVED: held=True, elevated=True, no respawn** — slipped on final basket reach (in_basket False) | cube → (0.13,0.22,0.805) at gc floor, y-aligned; carried to basket but wrist-twist (j4 past limit) pried grip loose at y≈0.20 |
| 11 | (place leg) | firmer grip: hand stiffness 300, tighter curl/thumb opposition to survive carry |

**Architecture finding (iter7→8): native arm-only DiffIK cannot grasp a tabletop cube on the
fixed-base G1 — every reachable pose pins a joint at its limit. Solved WITHIN constraints by
adding the waist DoF to the IK (no Pink/cuRobo/GraspGen). Grasp+lift+carry solved at iter10.**
