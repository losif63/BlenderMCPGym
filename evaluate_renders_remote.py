import os
import json
from PIL import Image
from utils import photometric_loss, clip_similarity

RENDERS_DIR = "renders_remote"
OUTPUT_PATH = "renders_remote/scores.json"


def evaluate_task(task_dir):
    edit_dir = os.path.join(task_dir, "edit")
    goal_dir = os.path.join(task_dir, "goal")

    if not os.path.isdir(edit_dir) or not os.path.isdir(goal_dir):
        return None

    edit_images = set(os.listdir(edit_dir))
    goal_images = set(os.listdir(goal_dir))
    shared = sorted(edit_images & goal_images)

    if not shared:
        return None

    per_image = {}
    pl_values = []
    n_clip_values = []

    for fname in shared:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        try:
            edit_img = Image.open(os.path.join(edit_dir, fname))
            goal_img = Image.open(os.path.join(goal_dir, fname))
        except Exception as e:
            print(f"  Warning: could not open {fname}: {e}")
            continue

        pl = float(photometric_loss(edit_img, goal_img))
        n_clip = float(1 - clip_similarity(edit_img, goal_img))

        per_image[fname] = {"photometric_loss": pl, "n_clip": n_clip}
        pl_values.append(pl)
        n_clip_values.append(n_clip)

    if not per_image:
        return None

    return {
        "per_image": per_image,
        "avg_photometric_loss": sum(pl_values) / len(pl_values),
        "avg_n_clip": sum(n_clip_values) / len(n_clip_values),
    }


def main():
    results = {}

    task_dirs = sorted(
        d for d in os.listdir(RENDERS_DIR)
        if os.path.isdir(os.path.join(RENDERS_DIR, d))
    )

    for task in task_dirs:
        task_dir = os.path.join(RENDERS_DIR, task)
        print(f"Evaluating {task}...")
        task_result = evaluate_task(task_dir)
        if task_result is None:
            print(f"  Skipped (missing edit/ or goal/ or no matching images).")
            continue
        results[task] = task_result
        print(f"  avg_pl={task_result['avg_photometric_loss']:.4f}  avg_n_clip={task_result['avg_n_clip']:.4f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved scores for {len(results)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
