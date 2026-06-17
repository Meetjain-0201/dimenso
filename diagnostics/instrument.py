"""Instrumented pipeline run — dumps per-step telemetry (JSON + CSV) for diagnosis.

Runs headless with respawn OFF so failures (knock/fall) are NOT masked.

Usage:
    conda activate isaac_sim
    python diagnostics/instrument.py [out_prefix]   # default: diagnostics/telemetry/run
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ego2g1 import run


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/telemetry/run"
    result = run({"headless": True, "telemetry": out, "respawn": False})
    print(f"INSTRUMENT DONE success={result.success}")
    print(f"metrics={result.metrics}")


if __name__ == "__main__":
    main()
