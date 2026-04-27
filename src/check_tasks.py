"""
Sanity-check each task directory in bench_data/ for required files.

Checks per task:
  - blender_file.blend
  - start.py
  - goal.py
  - description.txt
  - detailed_instruction.txt
  - renders/start/   (at least one image)
  - renders/goal/    (at least one image)

Usage:
    python check_tasks.py
    python check_tasks.py --task_type blendshape material
    python check_tasks.py --verbose      # show per-file status for every task
"""

import os
import re
from argparse import ArgumentParser
from pathlib import Path


BENCH_DATA_DIR = f"{os.getcwd()}/bench_data"
SKIP_ENTRIES = {"blender_files"}
TASK_TYPES = ["blendshape", "geometry", "lighting", "material", "placement"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(f.suffix.lower() in IMAGE_EXTS for f in directory.iterdir())


def check_task(task_dir: Path) -> dict[str, bool]:
    return {
        "blender_file.blend":       (task_dir / "blender_file.blend").is_file(),
        "start.py":                 (task_dir / "start.py").is_file(),
        "goal.py":                  (task_dir / "goal.py").is_file(),
        "description.txt":          (task_dir / "description.txt").is_file(),
        "detailed_instruction.txt": (task_dir / "detailed_instruction.txt").is_file(),
        "renders/start/ (images)":  has_images(task_dir / "renders" / "start"),
        "renders/goal/  (images)":  has_images(task_dir / "renders" / "goal"),
    }


def main(args):
    task_filter = set(args.task_type) if args.task_type else set(TASK_TYPES)

    task_dirs = sorted(
        e for e in BENCH_DATA_DIR.iterdir()
        if e.is_dir()
        and e.name not in SKIP_ENTRIES
        and (m := re.match(r'^([a-z]+)\d+$', e.name)) is not None
        and m.group(1) in task_filter
    )

    if not task_dirs:
        print("No matching task directories found.")
        return

    total = len(task_dirs)
    passed = 0
    failed_tasks = []

    for task_dir in task_dirs:
        results = check_task(task_dir)
        ok = all(results.values())

        if ok:
            passed += 1
            if args.verbose:
                print(f"  [OK] {task_dir.name}")
        else:
            failed_tasks.append((task_dir.name, results))
            missing = [name for name, present in results.items() if not present]
            print(f"  [FAIL] {task_dir.name}: missing {', '.join(missing)}")

        if args.verbose and not ok:
            for name, present in results.items():
                mark = "+" if present else "-"
                print(f"         [{mark}] {name}")

    print(f"\n{passed}/{total} tasks passed all checks.")
    if failed_tasks:
        print(f"{total - passed} task(s) incomplete — see above.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sanity-check bench_data task directories.")
    parser.add_argument(
        "--task_type", nargs="+", default=None, choices=TASK_TYPES,
        help="Limit check to specific task types (default: all)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-file status for every task, not just failures"
    )
    args = parser.parse_args()
    main(args)
