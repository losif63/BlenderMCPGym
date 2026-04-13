import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt


BENCH_DATA_DIR = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data")
SKIP_ENTRIES = {"blender_files"}
TASK_TYPES = ["blendshape", "geometry", "lighting", "material", "placement"]


def collect_durations(version):
    durations = {t: [] for t in TASK_TYPES}
    metadata_filename = f"metadata_ver{version}.json"
    for entry in sorted(os.listdir(BENCH_DATA_DIR)):
        if entry in SKIP_ENTRIES:
            continue
        task_dir = os.path.join(BENCH_DATA_DIR, entry)
        if not os.path.isdir(task_dir):
            continue
        metadata_path = os.path.join(task_dir, metadata_filename)
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        duration = metadata.get("duration_seconds")
        if duration is None:
            continue
        for task_type in TASK_TYPES:
            if entry.startswith(task_type):
                durations[task_type].append(duration)
                break
    return durations


def main():
    parser = ArgumentParser()
    parser.add_argument("--version", type=int, default=1, choices=[1, 2],
                        help="Experiment version to plot durations for (default: 1)")
    args = parser.parse_args()

    durations = collect_durations(args.version)

    all_durations = [d for vals in durations.values() for d in vals]
    if not all_durations:
        print("No metadata.json files with duration_seconds found.")
        return

    n_types = sum(1 for vals in durations.values() if vals)
    fig, axes = plt.subplots(1, n_types + 1, figsize=(5 * (n_types + 1), 4))
    fig.suptitle("Task Duration Distribution (seconds)", fontsize=14)

    # Overall histogram
    axes[0].hist(all_durations, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("All Tasks")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")

    # Per-type histograms
    ax_idx = 1
    colors = ["#e07b54", "#6abf69", "#f0c040", "#7f8dc9", "#c97fc9"]
    for (task_type, vals), color in zip(durations.items(), colors):
        if not vals:
            continue
        axes[ax_idx].hist(vals, bins=10, color=color, edgecolor="white")
        axes[ax_idx].set_title(task_type.capitalize())
        axes[ax_idx].set_xlabel("Duration (s)")
        ax_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
