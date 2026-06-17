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
