"""Score a run against the task goals → data/results/<run_id>/.

Per shared render view, compares the agent's edit render with the goal render:
  photometric_loss : image MSE
  n_clip           : 1 - CLIP similarity
Both are averaged over the views a task's edit and goal renders share. Lower is
better for each.

Reads  data/runs/<run_id>/<task>/renders/edit/  and  data/tasks/<task>/renders/goal/.

Examples:
    python -m blendergym.eval.evaluate --model haiku-4.5
    python -m blendergym.eval.evaluate --model opus-4.8 --task_type geometry
"""

from argparse import ArgumentParser
import json
import re

from PIL import Image

from ..config import RunConfig
from ..paths import TASKS_DIR, RESULTS_DIR, task_run_dir
from ..tasks import iter_tasks
from .metrics import photometric_loss, negative_clip

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def task_type(name: str) -> str:
    m = re.match(r"^([a-z]+)\d+$", name)
    return m.group(1) if m else name


def metrics_for_task(run_id: str, task_name: str, *, clip: bool = True) -> dict:
    """Mean photometric loss and n_clip over the views shared by edit and goal."""
    edit_dir = task_run_dir(run_id, task_name) / "renders" / "edit"
    goal_dir = TASKS_DIR / task_name / "renders" / "goal"
    if not edit_dir.is_dir() or not goal_dir.is_dir():
        return {"photometric_loss": None, "n_clip": None, "n_views": 0, "note": "missing renders"}

    goal_imgs = {p.name: p for p in goal_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    per_view = {}
    for name, gpath in sorted(goal_imgs.items()):
        epath = edit_dir / name
        if not epath.exists():
            continue
        edit_img, goal_img = Image.open(epath), Image.open(gpath)
        v = {"photometric_loss": photometric_loss(edit_img, goal_img)}
        if clip:
            v["n_clip"] = negative_clip(edit_img, goal_img)
        per_view[name] = v

    if not per_view:
        return {"photometric_loss": None, "n_clip": None, "n_views": 0, "note": "no shared views"}

    def avg(key):
        xs = [v[key] for v in per_view.values() if key in v]
        return sum(xs) / len(xs) if xs else None

    return {
        "photometric_loss": avg("photometric_loss"),
        "n_clip": avg("n_clip") if clip else None,
        "n_views": len(per_view),
        "per_view": per_view,
    }


def evaluate_run(config: RunConfig, *, task_type_filter: str | None = None,
                 clip: bool = True, verbose: bool = True, **_ignored) -> dict:
    """Score one run; write data/results/<run_id>/{scores,summary}.json. Returns summary."""
    run_id = config.run_id
    tasks = iter_tasks(task_type_filter)
    if not task_run_dir(run_id, tasks[0].name).parent.exists():
        raise FileNotFoundError(f"No run outputs found for {run_id} (run the benchmark first).")

    print(f"Evaluating {run_id} over {len(tasks)} tasks (photometric"
          f"{' + n_clip' if clip else ''})")

    scores: dict[str, dict] = {}
    for task in tasks:
        entry = metrics_for_task(run_id, task.name, clip=clip)
        entry["task_type"] = task_type(task.name)
        scores[task.name] = entry
        if verbose and entry["n_views"]:
            print(f"  {task.name:<14} pl={entry['photometric_loss']!s:<10} "
                  f"n_clip={entry['n_clip']!s:<10} ({entry['n_views']} views)")

    summary = summarize(scores)

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir/'scores.json'} and summary.json")
    for tt, s in summary.items():
        print(f"  {tt:<12} n={s['count']:>3}  pl={s['avg_photometric_loss']}  n_clip={s['avg_n_clip']}")
    return summary


def summarize(scores: dict) -> dict:
    """Per-task-type averages of photometric loss and n_clip (ignoring nulls)."""
    by_type: dict[str, dict] = {}
    for entry in scores.values():
        b = by_type.setdefault(entry["task_type"], {"pl": [], "nc": []})
        if entry.get("photometric_loss") is not None:
            b["pl"].append(entry["photometric_loss"])
        if entry.get("n_clip") is not None:
            b["nc"].append(entry["n_clip"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        tt: {
            "count": len(b["pl"]),
            "avg_photometric_loss": avg(b["pl"]),
            "avg_n_clip": avg(b["nc"]),
        }
        for tt, b in sorted(by_type.items())
    }


def main(args) -> None:
    config = RunConfig(model=args.model, tool=args.tool, trial=args.trial)
    evaluate_run(config, task_type_filter=args.task_type, clip=not args.no_clip)


if __name__ == "__main__":
    parser = ArgumentParser(description="Score a run with photometric loss + n_clip.")
    parser.add_argument("--model", type=str, required=True, help="Model slug of the run to evaluate")
    parser.add_argument("--tool", type=str, default="claudecode")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--task_type", type=str, default=None, help="Filter by task type prefix")
    parser.add_argument("--no_clip", action="store_true", help="Skip CLIP; photometric loss only")
    main(parser.parse_args())
