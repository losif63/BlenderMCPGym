from argparse import ArgumentParser
import json
import os
import time
from multiprocessing import Pool

from single_task import run_task


BENCH_DATA_DIR = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data")
SKIP_ENTRIES = {"blender_files"}
BASE_PORT = 9876
MAX_WORKERS = 10


def get_task_dirs():
    entries = sorted(os.listdir(BENCH_DATA_DIR))
    return [
        os.path.join(BENCH_DATA_DIR, e)
        for e in entries
        if e not in SKIP_ENTRIES and os.path.isdir(os.path.join(BENCH_DATA_DIR, e))
    ]


def run_task_with_logging(args):
    task_index, task_dir = args
    port = BASE_PORT + (task_index % MAX_WORKERS)
    task_name = os.path.basename(task_dir)
    print(f"\n[{task_name}] Starting on port {port}...")
    try:
        start_time = time.time()
        run_task(task_dir, port=port)
        duration = time.time() - start_time
    except Exception as e:
        print(f"[{task_name}] ERROR: {e}")
        return

    metadata_path = os.path.join(task_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    metadata["duration_seconds"] = round(duration, 2)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[{task_name}] Done in {duration:.2f}s")


def main(args):
    task_dirs = get_task_dirs()

    if args.task_type:
        task_dirs = [t for t in task_dirs if os.path.basename(t).startswith(args.task_type)]

    indexed_tasks = list(enumerate(task_dirs))
    print(f"Running {len(indexed_tasks)} tasks with up to {MAX_WORKERS} workers...")

    with Pool(processes=MAX_WORKERS) as pool:
        pool.map(run_task_with_logging, indexed_tasks)

    print("\nAll tasks complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default=None,
                        help="Filter by task type prefix, e.g. 'blendshape', 'material', 'placement'")
    args = parser.parse_args()

    main(args)
