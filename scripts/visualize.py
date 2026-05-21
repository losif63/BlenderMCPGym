"""
Generate visualization/{task_name}.png for each task.

Layout:
  Row 0  – images : Original | Haiku 4.5 | Sonnet 4.6 | Opus 4.6
  Rows 1+ – one row per metric, showing metric label + score per model
"""

import json
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
VIZ_DIR = ROOT / "visualization"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

# (json_key, dir_name, display_label)
MODELS = [
    ("haiku4.5",  "haiku-4.5",  "Haiku 4.5"),
    ("sonnet4.6", "sonnet-4.6", "Sonnet 4.6"),
    ("opus4.6",   "opus-4.6",   "Opus 4.6"),
]

# (data_key, display_label, json_filename)
METRICS = [
    ("dinov2",   "DINOv2-Large",    "dinov2_large.json"),
    ("dinov3",   "DINOv3-ViT-B/16", "dinov3_vitb16.json"),
    ("dreamsim", "DreamSim",        "dreamsim.json"),
    ("clip",     "CLIP ViT-B/32",   "clip_vitb32.json"),
    ("lpips",    "LPIPS (Alex)",    "lpips_alex.json"),
]

ROW_COLORS = ["#f5f5f5", "#e8e8e8"]


def load_metrics() -> dict:
    data = {}
    for key, _, filename in METRICS:
        path = ROOT / filename
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
        else:
            print(f"[WARN] {filename} not found — metric '{key}' will show N/A")
            data[key] = {}
    return data


def find_reference_image(task_name: str) -> Path | None:
    matches = glob.glob(str(IMAGES_DIR / f"{task_name}.*"))
    return Path(matches[0]) if matches else None


def open_image(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def create_visualization(task: str, metrics_data: dict):
    ref_img = open_image(find_reference_image(task))

    # (display_label, PIL image or None)
    renders = []
    for model_key, model_dir, model_label in MODELS:
        render_path = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
        renders.append((model_label, open_image(render_path)))

    n_metrics = len(METRICS)
    n_cols = 4

    img_height = 3.5
    metric_height = 0.42
    fig_height = img_height + n_metrics * metric_height + 0.5

    fig = plt.figure(figsize=(14, fig_height))
    gs = gridspec.GridSpec(
        1 + n_metrics, n_cols,
        figure=fig,
        height_ratios=[img_height] + [metric_height] * n_metrics,
        hspace=0.0,
        wspace=0.03,
    )

    # --- Image row ---
    col_labels = ["Original"] + [label for label, _ in renders]
    col_images = [ref_img] + [img for _, img in renders]

    for col, (label, img) in enumerate(zip(col_labels, col_images)):
        ax = fig.add_subplot(gs[0, col])
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "Missing", ha="center", va="center",
                    fontsize=11, color="#666666", transform=ax.transAxes)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Metric rows ---
    for row_idx, (metric_key, metric_label, _) in enumerate(METRICS):
        row = row_idx + 1
        bg = ROW_COLORS[row_idx % 2]
        task_scores = metrics_data.get(metric_key, {}).get(task, {})

        # Metric label (col 0)
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(bg)
        ax0.text(0.5, 0.5, metric_label, ha="center", va="center",
                 fontsize=9, fontweight="bold", transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_linewidth(0.5)

        # Score cells (cols 1–3)
        for col_idx, (model_key, _, _) in enumerate(MODELS):
            ax = fig.add_subplot(gs[row, col_idx + 1])
            ax.set_facecolor(bg)
            score = task_scores.get(model_key)
            text = f"{score:.4f}" if score is not None else "N/A"
            ax.text(0.5, 0.5, text, ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    fig.suptitle(task, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)

    out_path = VIZ_DIR / f"{task}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    VIZ_DIR.mkdir(exist_ok=True)
    metrics_data = load_metrics()

    for task in TASKS:
        print(f"Rendering {task}...")
        create_visualization(task, metrics_data)

    print(f"\nDone. Visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
