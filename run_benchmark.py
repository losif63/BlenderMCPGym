from argparse import ArgumentParser
import os

from single_task import run_task


BENCH_DATA_DIR = f"{os.getcwd()}/bench_data"
SKIP_ENTRIES = {"blender_files"}


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def already_done(task_dir, version):
    task_name = os.path.basename(task_dir)
    ver_tag = f"ver{version}"
    edit_file = os.path.join(task_dir, f"edit_{task_name}_{ver_tag}.blend")
    edit_renders = os.path.join(task_dir, "renders", f"edit_{ver_tag}")
    has_renders = os.path.isdir(edit_renders) and any(
        f.lower().endswith((".png", ".jpg", ".jpeg"))
        for f in os.listdir(edit_renders)
    )
    return os.path.isfile(edit_file) and has_renders


def main(args):
    task_dirs = get_task_dirs()

    if args.task_type:
        task_dirs = [t for t in task_dirs if os.path.basename(t).startswith(args.task_type)]

    print(f"Running {len(task_dirs)} tasks (version {args.version})...")
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        if already_done(task_dir, args.version):
            print(f"[{task_name}] Skipping — edit file and renders already exist.")
            continue
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        try:
            run_task(task_dir, version=args.version)
        except Exception as e:
            print(f"ERROR on {task_name}: {e}")
            continue

    print("\nAll tasks complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default=None,
                        help="Filter by task type prefix, e.g. 'blendshape', 'material', 'placement'")
    parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                        help="Experiment version: 1 (default, provides start.py) or 2 (no start.py)")
    args = parser.parse_args()

    main(args)
