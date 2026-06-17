"""Pipeline layer: the single orchestration entry point.

`pipeline.ego2g1.run(config)` is the ONE callable that the server, the CLI, and any
future control panel invoke. Everything flows through it:
    video -> perception -> retarget -> Isaac Lab scene -> result.
"""
