from argparse import ArgumentParser
import os

from single_task import run_task


BENCH_DATA_DIR = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data")
SKIP_ENTRIES = {"blender_files"}


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def main(args):
    task_dirs = get_task_dirs()

    if args.task_type:
        task_dirs = [t for t in task_dirs if os.path.basename(t).startswith(args.task_type)]

    print(f"Running {len(task_dirs)} tasks...")
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        try:
            run_task(task_dir)
        except Exception as e:
            print(f"ERROR on {task_name}: {e}")
            continue

    print("\nAll tasks complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default=None,
                        help="Filter by task type prefix, e.g. 'blendshape', 'material', 'placement'")
    args = parser.parse_args()

    main(args)
