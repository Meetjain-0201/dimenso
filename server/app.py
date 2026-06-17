"""Stub: minimal local FastAPI exposing POST /run -> pipeline.ego2g1.run(). P4.

Intentionally not wired up yet. The shape below is the P4 target; uncomment and
implement once FastAPI is added to the env. Kept import-light so the module can be
inspected without FastAPI installed.

Planned contract:
    POST /run   body: {"config": "configs/apple_in_basket.yaml"}  (or inline dict)
                -> 200 {"success": bool, "message": str, "artifacts": {...}}
    GET  /apps   -> list of available app configs (configs/*.yaml)
    GET  /videos -> list of available clips (data/*.mp4)

Run (P4):
    conda activate isaac_sim
    uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations


def create_app():
    """Build and return the FastAPI app. Implemented in P4."""
    raise NotImplementedError(
        "server.app.create_app is a P4 stub. It will expose POST /run that calls "
        "pipeline.ego2g1.run(config) — the single pipeline entry point."
    )
