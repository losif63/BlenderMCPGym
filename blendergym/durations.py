"""Sum duration_seconds across a chosen subset of run tasks.

Reads every data/runs/<run_id>/<task>/metadata.json and sums its
``duration_seconds`` over the tasks matching the filters. All filters AND
together; an unspecified filter matches everything.

  --models      one or more model slugs   (e.g. opus-4.8 opus-4.1)
  --task_type   one or more task types     (e.g. blendshape geometry lighting)
  --tasks       explicit task names        (e.g. blendshape31 geometry2)
  --tool/--trial   narrow to a tool / trial index

Examples:
    python -m blendergym.durations --models opus-4.8
    python -m blendergym.durations --task_type lighting material
    python -m blendergym.durations --models opus-4.8 --tasks blendshape31 blendshape2
"""

from argparse import ArgumentParser
import json
import re
from collections import defaultdict

from .paths import RUNS_DIR


def task_type(name: str) -> str:
    m = re.match(r"^([a-z]+)\d+$", name)
    return m.group(1) if m else name


def format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def collect(models=None, task_types=None, tasks=None, tool=None, trial=None) -> list[dict]:
    """Per-task metadata records matching the filters (with duration present)."""
    records = []
    for meta_path in RUNS_DIR.glob("*/*/metadata.json"):
        try:
            with open(meta_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = m.get("task") or meta_path.parent.name
        m.setdefault("task", name)
        m["task_type"] = task_type(name)

        if models and m.get("model") not in models:
            continue
        if task_types and m["task_type"] not in task_types:
            continue
        if tasks and name not in tasks:
            continue
        if tool and m.get("tool") != tool:
            continue
        if trial is not None and m.get("trial") != trial:
            continue
        records.append(m)
    return records


def main(args) -> None:
    records = collect(models=args.models, task_types=args.task_type,
                      tasks=args.tasks, tool=args.tool, trial=args.trial)

    print(f"Filters: models={args.models or 'ALL'}  task_type={args.task_type or 'ALL'}  "
          f"tasks={args.tasks or 'ALL'}  tool={args.tool or 'ALL'}  trial={args.trial if args.trial is not None else 'ALL'}")

    with_dur = [r for r in records if isinstance(r.get("duration_seconds"), (int, float))]
    missing = len(records) - len(with_dur)
    if not records:
        print("No matching tasks found.")
        return

    by_model = defaultdict(lambda: [0.0, 0])
    by_type = defaultdict(lambda: [0.0, 0])
    total = 0.0
    for r in with_dur:
        d = r["duration_seconds"]
        total += d
        by_model[r.get("model", "?")][0] += d
        by_model[r.get("model", "?")][1] += 1
        by_type[r["task_type"]][0] += d
        by_type[r["task_type"]][1] += 1

    print(f"\nMatched {len(records)} task(s)"
          + (f" ({missing} without duration)" if missing else ""))

    if len(by_model) > 1:
        print("\nBy model:")
        for model in sorted(by_model):
            secs, n = by_model[model]
            print(f"  {model:<12} {n:>4} tasks   {secs:>10.1f}s   ({format_hms(secs)})")
    if len(by_type) > 1:
        print("\nBy task type:")
        for tt in sorted(by_type):
            secs, n = by_type[tt]
            print(f"  {tt:<12} {n:>4} tasks   {secs:>10.1f}s   ({format_hms(secs)})")

    print(f"\nTOTAL: {total:.1f} s   ({format_hms(total)})   over {len(with_dur)} task(s)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sum duration_seconds over a subset of run tasks.")
    parser.add_argument("--models", nargs="+", default=None, help="Model slug(s) to include")
    parser.add_argument("--task_type", nargs="+", default=None, help="Task type(s) to include")
    parser.add_argument("--tasks", nargs="+", default=None, help="Explicit task name(s) to include")
    parser.add_argument("--tool", type=str, default=None, help="Restrict to a tool (e.g. claudecode)")
    parser.add_argument("--trial", type=int, default=None, help="Restrict to a trial index")
    main(parser.parse_args())
