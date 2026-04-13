from argparse import ArgumentParser
import os
from multiprocessing import Pool

from single_task import run_task


BENCH_DATA_DIR = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data")
SKIP_ENTRIES = {"blender_files"}
BASE_PORT = 9876
MAX_WORKERS = 5


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def run_task_with_logging(args):
    task_index, task_dir, version = args
    port = BASE_PORT + (task_index % MAX_WORKERS)
    task_name = os.path.basename(task_dir)
    print(f"\n[{task_name}] Starting on port {port} (version {version})...")
    try:
        run_task(task_dir, port=port, version=version)
    except Exception as e:
        print(f"[{task_name}] ERROR: {e}")
        return
    print(f"[{task_name}] Done.")


def main(args):
    task_dirs = get_task_dirs()

    if args.task_type:
        task_dirs = [t for t in task_dirs if os.path.basename(t).startswith(args.task_type)]

    indexed_tasks = [(i, d, args.version) for i, d in enumerate(task_dirs)]
    print(f"Running {len(indexed_tasks)} tasks with up to {MAX_WORKERS} workers (version {args.version})...")

    with Pool(processes=MAX_WORKERS) as pool:
        pool.map(run_task_with_logging, indexed_tasks)

    print("\nAll tasks complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default=None,
                        help="Filter by task type prefix, e.g. 'blendshape', 'material', 'placement'")
    parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                        help="Experiment version: 1 (default, provides start.py) or 2 (no start.py)")
    args = parser.parse_args()

    main(args)
