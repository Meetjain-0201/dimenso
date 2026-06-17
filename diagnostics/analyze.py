"""Root-cause analyzer — reads a telemetry log and fires explicit event detectors.

Usage:
    python diagnostics/analyze.py [telemetry_prefix]   # default: diagnostics/telemetry/run
"""

import json
import math
import sys

# detector thresholds
TH = {
    "knock_xy": 0.05,    # cube xy drift (m) while not grasped -> knocked
    "table_z": 0.69,     # below this z -> fell off the table (top ~0.70)
    "palm_short": 0.03,  # EE position error (m) at bottom of descent -> can't reach
    "manip": 0.002,   # below this = singular (0.005 w cond~50 is healthy)
    "jlim": 0.15,        # joint within this (rad) of a limit -> near-singular / contorted
}


def load(path):
    if not path.endswith(".json"):
        path += ".json"
    with open(path) as f:
        return json.load(f)


def analyze(d):
    """Return list of (detector_name, detail_string) for fired detectors."""
    t = d["telemetry"]
    m = d["metrics"]
    b = d["basket"]
    start = d["object_start"]
    fired = []

    # cube_knocked: cube xy moves while not yet grasped (approach phases)
    for r in (x for x in t if x["phase"] in ("warmup", "move_above_ball", "descend_to_ball")):
        dxy = math.hypot(r["cube_pos"][0] - start[0], r["cube_pos"][1] - start[1])
        if dxy > TH["knock_xy"]:
            fired.append(("cube_knocked",
                          f"phase={r['phase']} step={r['i']}: cube xy jumped {dxy:.3f} m (cube_vel={r['cube_vel']})"))
            break

    # cube_fell: cube below table height at any point
    fell = next((r for r in t if r["cube_pos"][2] < TH["table_z"]), None)
    if fell:
        fired.append(("cube_fell", f"phase={fell['phase']} step={fell['i']}: cube z={fell['cube_pos'][2]:.3f} m"))

    # palm_short: EE position error at the bottom of descent (split pos vs orient)
    desc = [r for r in t if r["phase"] in ("descend_to_ball", "grasp")]
    if desc:
        last = desc[-1]
        if last["ee_pos_err"] > TH["palm_short"]:
            fired.append(("palm_short",
                          f"ee_pos_err={last['ee_pos_err']:.3f} m  (orientation_err={last['ee_orient_err']:.3f} rad "
                          f"-> {'orientation-dominated' if last['ee_orient_err'] > 0.3 else 'position-dominated'})"))

    # near_singular: low manipulability or a joint near its limit at the grasp target
    grasp_rows = [r for r in t if r["phase"] == "grasp"]
    if grasp_rows:
        gr = grasp_rows[len(grasp_rows) // 2]
        wj = min(range(7), key=lambda i: gr["jlim"][i])
        if gr["manip"] < TH["manip"] or gr["jlim"][wj] < TH["jlim"]:
            fired.append(("near_singular",
                          f"manip={gr['manip']} jac_cond={gr['jac_cond']} arm_joint[{wj}] limit_dist={gr['jlim'][wj]:.3f} rad"))

    # grasp_empty: cube never rose with the palm
    if not m.get("grasp_held", False):
        fired.append(("grasp_empty",
                      f"cube_z_max_lift={m.get('cube_z_max_lift')} m (cube stayed ~table; fingers closed on nothing)"))
    elif m.get("object_fell") or m.get("xy_off", 0) > b["radius"]:
        # slip: was grasped (rose) but lost during lift/carry
        fired.append(("slip",
                      f"grasped (rose to {m.get('cube_z_max_lift')} m) but lost: object_fell={m.get('object_fell')} "
                      f"final xy_off={m.get('xy_off')} m"))

    # release_miss: final cube xy outside basket
    if m.get("xy_off", 1.0) > b["radius"]:
        fired.append(("release_miss", f"final cube xy_off={m.get('xy_off')} m > basket radius {b['radius']} m"))

    return fired


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/telemetry/run"
    d = load(path)
    fired = analyze(d)
    print("=== ROOT-CAUSE DETECTORS ===")
    if not fired:
        print("  (none fired — clean run)")
    for name, detail in fired:
        print(f"  [{name}] {detail}")
    print("=== key metrics ===")
    m = d["metrics"]
    for k in ("success", "grasp_held", "stable", "max_arm_dq", "max_ee_err", "xy_off",
              "ball_final_pos", "cube_z_max_lift", "object_fell"):
        if k in m:
            print(f"  {k} = {m[k]}")
    return fired


if __name__ == "__main__":
    main()
