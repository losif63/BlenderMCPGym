from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BENCH_DATA_DIR = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data")


def load_images(directory):
    paths = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    return [(p, mpimg.imread(p)) for p in paths]


def main(args):
    task_dir = os.path.join(BENCH_DATA_DIR, args.task_name)
    start_renders_dir = os.path.join(task_dir, "renders", "start")
    goal_renders_dir = os.path.join(task_dir, "renders", "goal")
    edit_renders_dir = os.path.join(task_dir, "renders", "edit")
    description_path = os.path.join(task_dir, "description.txt")
    edit_file = os.path.join(task_dir, f"edit_{args.task_name}.blend")

    if not os.path.exists(edit_file):
        print(f"ERROR: {edit_file} does not exist. Run single_task.py first.")
        return
    if not os.path.exists(edit_renders_dir):
        print(f"ERROR: {edit_renders_dir} does not exist. Run single_task.py first.")
        return

    with open(description_path, "r") as f:
        content = f.read()
    ind = content.find("INSTRUCTION:")
    instruction = content[ind:].strip() if ind != -1 else content.strip()

    start_images = load_images(start_renders_dir)
    goal_images = load_images(goal_renders_dir)
    edit_images = load_images(edit_renders_dir)

    n_cols = max(len(start_images), len(goal_images), len(edit_images))
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows + 2))
    fig.suptitle(f"Task: {args.task_name}\n\n{instruction}", fontsize=11, wrap=True)
    fig.subplots_adjust(top=0.84)

    # Ensure axes is always 2D
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    row_labels = ["Start", "Goal", "Edit"]
    for row_idx, (label, images) in enumerate(zip(row_labels, [start_images, goal_images, edit_images])):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(images):
                _, img = images[col_idx]
                ax.imshow(img)
                ax.set_title(f"{label} — render{col_idx + 1}")
            else:
                ax.axis("off")
            ax.axis("off")

    plt.tight_layout()

    comparisons_dir = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    output_path = os.path.join(comparisons_dir, f"{args.task_name}.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved comparison to {output_path}")

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    main(args)
