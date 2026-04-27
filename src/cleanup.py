from argparse import ArgumentParser
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DATA_DIR = os.path.join(PROJECT_ROOT, "bench_data")
SKIP_ENTRIES = {"blender_files"}


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def cleanup_task(task_dir, force_delete=False):
    task_name = os.path.basename(task_dir)
    targets = {}
    for ver in (1, 2, 3):
        ver_tag = f"ver{ver}"
        targets[f"edit_{task_name}_{ver_tag}.blend"] = os.path.join(task_dir, f"edit_{task_name}_{ver_tag}.blend")
        targets[f"renders/edit_{ver_tag}"]           = os.path.join(task_dir, "renders", f"edit_{ver_tag}")
        targets[f"metadata_{ver_tag}.json"]          = os.path.join(task_dir, f"metadata_{ver_tag}.json")
        targets[ver_tag]                             = os.path.join(task_dir, ver_tag)
    targets["temp"] = os.path.join(task_dir, "temp")

    blend1_files = [f for f in os.listdir(task_dir) if f.endswith(".blend1")]
    for f in blend1_files:
        os.remove(os.path.join(task_dir, f))
        print(f"  deleted {f}")

    prev_dir = os.path.join(task_dir, "prev")

    if force_delete and os.path.exists(prev_dir):
        shutil.rmtree(prev_dir)
        print(f"  deleted prev/")

    if not blend1_files and not any(os.path.exists(p) for p in targets.values()):
        return False

    os.makedirs(prev_dir, exist_ok=True)

    for name, src in targets.items():
        if not os.path.exists(src):
            continue
        dst = os.path.join(prev_dir, os.path.basename(src))
        if os.path.exists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)
        print(f"  {name} -> prev/")

    return True


def main(args):
    task_dirs = get_task_dirs()

    if args.task_name:
        task_dirs = [t for t in task_dirs if os.path.basename(t) == args.task_name]
        if not task_dirs:
            print(f"ERROR: Task '{args.task_name}' not found.")
            return

    cleaned = 0
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        if cleanup_task(task_dir, force_delete=args.force_delete):
            print(f"[{task_name}] Cleaned up.")
            cleaned += 1

    print(f"\nDone. {cleaned} task(s) cleaned up.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, default=None,
                        help="Clean up a single task. If omitted, cleans all tasks.")
    parser.add_argument("--force-delete", action="store_true",
                        help="Also delete the prev/ subdirectory.")
    args = parser.parse_args()

    main(args)
