"""Regression test: G1 Dex3 picks the ball and places it in the basket (real physics).

Runs the single pipeline entry point headless and asserts the ball ends up inside the
basket bounds and the episode reports success. No welding / fixed joints — the ball's
final pose is whatever physics produced.

Run directly (Isaac Sim must launch before pytest-style import machinery, so this is a
runnable script rather than a pytest test):

    conda activate isaac_sim
    python tests/test_pickplace.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `pipeline`

from pipeline.ego2g1 import run


def check(result):
    m = result.metrics
    bx, by, bz = m["ball_final_pos"]
    cx, cy, _ = m["basket_center"]
    xy = math.hypot(bx - cx, by - cy)

    assert result.success, f"pipeline reported failure: {result.message}"
    assert xy <= m["basket_radius"], (
        f"ball xy offset {xy:.3f} m exceeds basket radius {m['basket_radius']:.3f} m"
    )
    assert bz <= m["basket_rim_z"] + 0.02, (
        f"ball z {bz:.3f} m is above basket rim {m['basket_rim_z']:.3f} m (not in basket)"
    )
    assert bz >= m["basket_floor_z"] - 0.03, (
        f"ball z {bz:.3f} m is below basket floor {m['basket_floor_z']:.3f} m"
    )
    # (c) stability / no-jitter check across the whole trajectory (incl. carry)
    assert m["max_arm_dq"] <= m["arm_q_delta_thresh"], (
        f"jitter: max per-step arm_q delta {m['max_arm_dq']:.3f} rad > {m['arm_q_delta_thresh']} (swinging)"
    )
    assert m["max_ee_err"] <= m["ee_err_thresh"], (
        f"IK tracking error {m['max_ee_err']:.3f} m > {m['ee_err_thresh']} m"
    )
    return xy


if __name__ == "__main__":
    result = run({"headless": True})
    try:
        xy = check(result)
        print(f"RESULT: PASS | ball_in_basket xy_off={xy:.3f}m metrics={result.metrics}")
        sys.exit(0)
    except AssertionError as e:
        print(f"RESULT: FAIL | {e} | metrics={result.metrics}")
        sys.exit(1)
