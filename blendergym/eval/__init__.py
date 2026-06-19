"""Evaluation: score a run's outputs against the task goals.

Per-view image metrics (edit render vs goal render), lower is better:
  - photometric_loss  (image MSE)
  - n_clip            (1 - CLIP similarity)

Reads  data/runs/<run_id>/<task>/  and  data/tasks/<task>/  (goal),
writes data/results/<run_id>/.
"""

from .metrics import photometric_loss, clip_similarity, negative_clip

__all__ = ["photometric_loss", "clip_similarity", "negative_clip"]
