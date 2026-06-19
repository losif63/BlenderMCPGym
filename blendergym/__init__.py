"""BlenderGym — Suite A Scene-Edit Benchmark harness.

Run agent ablations (model, and later tool) over the Blender scene-edit tasks.
A *run* is identified by ``run_id = "{tool}__{model}"`` and writes all of its
outputs under ``data/runs/<run_id>/``, never into the task definitions.

See PLAN.md for the full architecture and migration plan.
"""

from .config import RunConfig, MODELS, DEFAULT_TOOL

__all__ = ["RunConfig", "MODELS", "DEFAULT_TOOL"]
