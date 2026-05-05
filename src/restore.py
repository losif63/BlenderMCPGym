from argparse import ArgumentParser
import os
import re
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DATA_DIR = os.path.join(PROJECT_ROOT, "bench_data")
SKIP_ENTRIES = {"blender_files"}
TASK_TYPES = ["blendshape", "geometry", "lighting", "material", "placement"]


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def restore_task(task_dir, keep_prev=False):
    task_name = os.path.basename(task_dir)
    prev_dir = os.path.join(task_dir, "prev")
    if not os.path.isdir(prev_dir):
        return False

    targets = {}
    for ver in (1, 2, 3):
        ver_tag = f"ver{ver}"
        targets[f"edit_{task_name}_{ver_tag}.blend"] = os.path.join(task_dir, f"edit_{task_name}_{ver_tag}.blend")
        targets[f"edit_{ver_tag}"]                   = os.path.join(task_dir, "renders", f"edit_{ver_tag}")
        targets[f"metadata_{ver_tag}.json"]          = os.path.join(task_dir, f"metadata_{ver_tag}.json")
        targets[ver_tag]                             = os.path.join(task_dir, ver_tag)
    targets["temp"] = os.path.join(task_dir, "temp")

    moved = False
    for name, dst in targets.items():
        src = os.path.join(prev_dir, name)
        if not os.path.exists(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)
        print(f"  prev/{name} -> {os.path.relpath(dst, task_dir)}")
        moved = True

    if not keep_prev and os.path.isdir(prev_dir) and not os.listdir(prev_dir):
        os.rmdir(prev_dir)
        print(f"  removed empty prev/")

    return moved


def main(args):
    task_dirs = get_task_dirs()

    if args.task_name:
        task_dirs = [t for t in task_dirs if os.path.basename(t) == args.task_name]
        if not task_dirs:
            print(f"ERROR: Task '{args.task_name}' not found.")
            return

    if args.task_type:
        type_filter = set(args.task_type)
        filtered = []
        for t in task_dirs:
            m = re.match(r'^([a-z]+)\d+$', os.path.basename(t))
            if m and m.group(1) in type_filter:
                filtered.append(t)
        task_dirs = filtered
        if not task_dirs:
            print(f"ERROR: No tasks found for task_type={sorted(type_filter)}.")
            return

    restored = 0
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        if restore_task(task_dir, keep_prev=args.keep_prev):
            print(f"[{task_name}] Restored.")
            restored += 1

    print(f"\nDone. {restored} task(s) restored.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, default=None,
                        help="Restore a single task. If omitted, restores all tasks.")
    parser.add_argument("--task_type", nargs="+", choices=TASK_TYPES,
                        help="Limit restore to these task types (e.g., blendshape placement).")
    parser.add_argument("--keep-prev", action="store_true",
                        help="Do not delete the prev/ directory after restoring.")
    args = parser.parse_args()

    main(args)
