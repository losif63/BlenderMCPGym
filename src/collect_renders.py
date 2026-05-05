#!/usr/bin/env python3
"""Collect rendered edit/goal images from each task into a flat renders/ tree.

For every task directory under bench_data/, copies:
  bench_data/{task}/renders/edit_ver3/*  -> renders/{task}/edit/
  bench_data/{task}/renders/goal/*       -> renders/{task}/goal/
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "bench_data"


def copy_tree(src: Path, dst: Path) -> int:
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "renders",
        help="Output root directory (default: <repo>/renders)",
    )
    args = parser.parse_args()
    out_root: Path = args.out

    if not BENCH_DIR.is_dir():
        print(f"error: {BENCH_DIR} not found", file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    tasks = sorted(p for p in BENCH_DIR.iterdir() if p.is_dir())
    total_edit = total_goal = task_count = 0
    skipped = []

    for task_dir in tasks:
        task = task_dir.name
        edit_src = task_dir / "renders" / "edit_ver3"
        goal_src = task_dir / "renders" / "goal"

        if not edit_src.is_dir() and not goal_src.is_dir():
            skipped.append(task)
            continue

        n_edit = copy_tree(edit_src, out_root / task / "edit")
        n_goal = copy_tree(goal_src, out_root / task / "goal")

        if n_edit == 0 and n_goal == 0:
            skipped.append(task)
            continue

        total_edit += n_edit
        total_goal += n_goal
        task_count += 1
        print(f"{task}: edit={n_edit} goal={n_goal}")

    print(
        f"\ndone: {task_count} tasks, "
        f"{total_edit} edit images, {total_goal} goal images -> {out_root}"
    )
    if skipped:
        print(f"skipped {len(skipped)} dirs without renders: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
