from argparse import ArgumentParser
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BENCH_DATA_DIR = f"{os.getcwd()}/bench_data"
SKIP_ENTRIES = {"blender_files"}
TASK_TYPES = ["blendshape", "geometry", "lighting", "material", "placement"]


def load_images(directory):
    paths = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    return [(p, mpimg.imread(p)) for p in paths]


def compare_task(task_name, version, comparisons_dir, show):
    ver_tag = f"ver{version}"
    task_dir = os.path.join(BENCH_DATA_DIR, task_name)
    start_renders_dir = os.path.join(task_dir, "renders", "start")
    goal_renders_dir = os.path.join(task_dir, "renders", "goal")
    edit_renders_dir = os.path.join(task_dir, "renders", f"edit_{ver_tag}")
    description_path = os.path.join(task_dir, "detailed_instruction.txt")
    edit_file = os.path.join(task_dir, f"edit_{task_name}_{ver_tag}.blend")

    if not os.path.exists(edit_file):
        print(f"  [{task_name}] Skipping — {edit_file} not found.")
        return False
    if not os.path.isdir(edit_renders_dir):
        print(f"  [{task_name}] Skipping — {edit_renders_dir} not found.")
        return False

    with open(description_path, "r") as f:
        content = f.read()
    ind = content.find("INSTRUCTION:")
    instruction = content[ind:].strip() if ind != -1 else content.strip()

    start_images = load_images(start_renders_dir)
    goal_images = load_images(goal_renders_dir)
    edit_images = load_images(edit_renders_dir)

    n_cols = max(len(start_images), len(goal_images), len(edit_images))

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 17))
    fig.suptitle(f"Task: {task_name}\n\n{instruction}", fontsize=11, wrap=True)
    fig.subplots_adjust(top=0.84)

    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (label, images) in enumerate(zip(["Start", "Goal", "Edit"], [start_images, goal_images, edit_images])):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(images):
                _, img = images[col_idx]
                ax.imshow(img)
                ax.set_title(f"{label} — render{col_idx + 1}")
            ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(comparisons_dir, f"{task_name}_{ver_tag}.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  [{task_name}] Saved to {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return True


def get_all_task_names(task_type_filter):
    names = []
    for entry in sorted(os.listdir(BENCH_DATA_DIR)):
        if entry in SKIP_ENTRIES:
            continue
        if not os.path.isdir(os.path.join(BENCH_DATA_DIR, entry)):
            continue
        m = re.match(r'^([a-z]+)\d+$', entry)
        if not m:
            continue
        if task_type_filter and m.group(1) not in task_type_filter:
            continue
        names.append(entry)
    return names


def main(args):
    comparisons_dir = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)

    if args.all:
        # Headless-safe: don't try to open display windows when iterating
        matplotlib.use("Agg")
        task_type_filter = set(args.task_type) if args.task_type else None
        task_names = get_all_task_names(task_type_filter)
        print(f"Generating comparisons for {len(task_names)} tasks (version {args.version})...")
        done = sum(compare_task(name, args.version, comparisons_dir, show=False) for name in task_names)
        print(f"\nDone. {done}/{len(task_names)} comparisons saved to {comparisons_dir}")
    else:
        if not args.task_name:
            print("ERROR: provide --task_name or use --all.")
            return
        compare_task(args.task_name, args.version, comparisons_dir, show=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_name", type=str, help="Single task to compare")
    group.add_argument("--all", action="store_true", help="Generate comparisons for all completed tasks")
    parser.add_argument("--task_type", nargs="+", choices=TASK_TYPES,
                        help="With --all, limit to these task types")
    parser.add_argument("--version", type=int, default=1, choices=[1, 2],
                        help="Experiment version to compare (default: 1)")
    args = parser.parse_args()

    main(args)
