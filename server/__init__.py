"""Server layer: a minimal local FastAPI wrapper around the pipeline.

P4. Exposes ``POST /run`` which calls ``pipeline.ego2g1.run(config)``. The local
control-panel frontend (``frontend/index.html``) talks to this.
"""
