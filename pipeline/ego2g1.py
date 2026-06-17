"""Stub: the single end-to-end entry point. Implemented in P3.

`run(config)` is the ONE callable everything else (server, CLI, control panel) calls.
Keep this signature stable — it is the public contract of the whole project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Outcome of a single ego2g1 run."""

    success: bool
    message: str
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path
    metrics: dict[str, Any] = field(default_factory=dict)


def run(config: dict | str | Path) -> RunResult:
    """Run the full egocentric-video -> G1 pick-and-place pipeline.

    Args:
        config: A config dict, or a path to a YAML app config (see
            ``configs/apple_in_basket.yaml``). Specifies the object, basket, target
            zone, and source video path.

    Returns:
        A :class:`RunResult` describing success and produced artifacts.

    P3 wiring (each step is a P2 stub today):
        1. perception.video_loader.load_frames(video_path)
        2. perception.mediapipe_hands.extract_hand_landmarks(...)
        3. retarget.kinematics.retarget_hands(...) -> EE targets + grasp
        4. retarget.ik_solver.solve_upper_body_ik(...) -> joint targets
        5. drive `Dimenso-AppleBasket-G1-v0` headless, roll out, score success.
    """
    raise NotImplementedError("ego2g1.run is a P3 stub — the single pipeline entry point.")
