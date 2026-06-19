"""Filesystem layout for the three-tree data model (see PLAN.md).

  tasks   -> immutable task definitions (read-only; runs never write here)
  runs    -> per-run agent outputs, keyed by run_id
  results -> derived scoring/analysis, keyed by run_id

NOTE: the task definitions still physically live in ``bench_data/`` and are
renamed to ``data/tasks/`` in Stage 3. TASKS_DIR points at the current location
so the rest of the code is already written against the final abstraction.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TASKS_DIR = REPO_ROOT / "bench_data"      # -> data/tasks in Stage 3
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
RESULTS_DIR = DATA_DIR / "results"
CONFIGS_DIR = REPO_ROOT / "configs"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def task_run_dir(run_id: str, task_name: str) -> Path:
    return RUNS_DIR / run_id / task_name
