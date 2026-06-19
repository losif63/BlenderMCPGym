"""Sweep the benchmark over a set of models (the ablation matrix).

For each model, runs every task (optionally filtered by type), skipping tasks
that already have outputs under data/runs/<run_id>/. Runs are sequential because
BlenderMCP binds a fixed port.

Examples:
    python -m blendergym.run.run_benchmark --models opus-4.8 sonnet-4.6 haiku-4.5
    python -m blendergym.run.run_benchmark --task_type blendshape          # default models from config
"""

from argparse import ArgumentParser
import json

from ..config import RunConfig, MODELS
from ..paths import CONFIGS_DIR
from ..tasks import iter_tasks
from ..eval.evaluate import evaluate_run
from .. import runs
from .single_task import run_task


def load_default_models() -> list[str]:
    cfg = CONFIGS_DIR / "models.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f)["models"]
    return list(MODELS)


def main(args) -> None:
    models = args.models or load_default_models()
    tasks = iter_tasks(args.task_type)
    print(f"Models: {models}")
    print(f"Running {len(tasks)} tasks x {len(models)} models "
          f"(tool={args.tool}, trial={args.trial})")

    for model in models:
        config = RunConfig(model=model, tool=args.tool, trial=args.trial)
        print(f"\n{'#'*60}\n# Run: {config.run_id}\n{'#'*60}")
        for task in tasks:
            if not args.force and runs.is_done(config, task.name):
                print(f"[{config.run_id}/{task.name}] Skipping — outputs already exist.")
                continue
            print(f"\n{'='*50}\n{config.run_id}  ::  {task.name}\n{'='*50}")
            try:
                run_task(task, config, virtual_display=args.virtual)
            except Exception as e:
                print(f"ERROR on {config.run_id}/{task.name}: {e}")
                continue

        if args.eval:
            print(f"\n{'#'*60}\n# Eval: {config.run_id}\n{'#'*60}")
            try:
                evaluate_run(config, task_type_filter=args.task_type, verbose=False)
            except Exception as e:
                print(f"ERROR evaluating {config.run_id}: {e}")

    print("\nAll runs complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the scene-edit benchmark across models.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model slugs to sweep (default: configs/models.json)")
    parser.add_argument("--task_type", type=str, default=None,
                        help="Filter tasks by prefix, e.g. blendshape / material / placement")
    parser.add_argument("--tool", type=str, default="claudecode", help="Agent tool")
    parser.add_argument("--trial", type=int, default=0, help="Trial index (repeat runs)")
    parser.add_argument("--virtual", action="store_true", help="Set DISPLAY=:99 for headless Blender")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even if outputs exist")
    parser.add_argument("--eval", action="store_true",
                        help="Score each model with photometric loss after its sweep finishes")
    main(parser.parse_args())
