"""
Side-by-side comparison of v1 (claudecode) and v2 (claudecode_v2) recreations.

For each task, lays out:
  Row 0  – Reference | {model} v1 | {model} v2 | ... (one v1/v2 pair per model)
  Rows 1+ – one row per metric, showing the score per (model, version) cell

v2 results are partial:
  - render.png is not produced by run_claudecode_v2.py. We fall back to
    `process/render_final.png`, then to the highest-numbered
    `process/render_vNN.png` if neither exists.
  - The existing eval JSONs (dinov3_vitl16.json, etc.) only cover v1 because
    those scripts hardcode the "claudecode" platform dir. For v2 we look up:
      1. v2-specific JSONs (e.g. `dinov3_vitl16_v2.json`) keyed identically.
      2. For the DINOv3 row only, the last `dinov3_cosine_distance` entry in
         `recreation/{task}/claudecode_v2/{model}/score_log.jsonl` — that is
         the score of the v2 agent's final render, computed by the same DINOv3
         model the v1 eval uses.
    Anything still missing renders as "pending".

Defaults to comparing haiku-4.5 and opus-4.7 (the models the v2 sweep has been
run on). Override with --models.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
VIZ_DIR = ROOT / "visualization" / "v1_vs_v2"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

# Full registry of (json_key, dir_name, display_label). The CLI selects a
# subset of these to actually plot.
MODEL_REGISTRY = {
    "haiku-4.5":  ("haiku4.5",  "Haiku 4.5"),
    "sonnet-4.6": ("sonnet4.6", "Sonnet 4.6"),
    "opus-4.6":   ("opus4.6",   "Opus 4.6"),
    "opus-4.7":   ("opus4.7",   "Opus 4.7"),
}

DEFAULT_MODEL_DIRS = ["haiku-4.5", "opus-4.7"]

# (data_key, display_label, v1_json, v2_json)
# The v2 json is optional; if absent we fall through to score_log.jsonl for
# DINOv3, or display as "pending" for everything else.
METRICS = [
    ("dinov2",   "DINOv2-Large",    "dinov2_large.json",  "dinov2_large_v2.json"),
    ("dinov3",   "DINOv3-ViT-L/16", "dinov3_vitl16.json", "dinov3_vitl16_v2.json"),
    ("dreamsim", "DreamSim",        "dreamsim.json",      "dreamsim_v2.json"),
    ("clip",     "CLIP ViT-B/32",   "clip_vitb32.json",   "clip_vitb32_v2.json"),
    ("lpips",    "LPIPS (Alex)",    "lpips_alex.json",    "lpips_alex_v2.json"),
]

V1_BG = "#eef3fb"  # cool blue tint
V2_BG = "#fff1e6"  # warm orange tint
BEST_BG = "#d4edda"  # green tint for the best score in a row


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_metrics() -> dict:
    """Returns {metric_key: {"v1": {...}, "v2": {...}}}."""
    data = {}
    for key, _, v1_file, v2_file in METRICS:
        data[key] = {
            "v1": load_json_if_exists(ROOT / v1_file),
            "v2": load_json_if_exists(ROOT / v2_file),
        }
    return data


def find_reference_image(task_name: str) -> Path | None:
    matches = glob.glob(str(IMAGES_DIR / f"{task_name}.*"))
    return Path(matches[0]) if matches else None


def find_v2_render(task: str, model_dir: str) -> Path | None:
    """Resolve a v2 render image, preferring the canonical render.png."""
    base = RECREATION_DIR / task / "claudecode_v2" / model_dir
    if not base.exists():
        return None
    canonical = base / "render.png"
    if canonical.exists():
        return canonical
    final = base / "process" / "render_final.png"
    if final.exists():
        return final
    # Fallback: highest-numbered iteration render
    iters = sorted(base.glob("process/render_v*.png"))
    return iters[-1] if iters else None


def find_v1_render(task: str, model_dir: str) -> Path | None:
    p = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
    return p if p.exists() else None


def last_score_log_dinov3(task: str, model_dir: str) -> float | None:
    log = RECREATION_DIR / task / "claudecode_v2" / model_dir / "score_log.jsonl"
    if not log.exists():
        return None
    last_dist = None
    with open(log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = rec.get("dinov3_cosine_distance")
            if isinstance(d, (int, float)):
                last_dist = float(d)
    return last_dist


def open_image(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def metric_value(
    metric_key: str,
    version: str | None,
    task: str,
    model_dir: str | None,
    model_json_key: str | None,
    metrics_data: dict,
) -> float | None:
    if version is None or model_dir is None or model_json_key is None:
        return None
    bucket = metrics_data.get(metric_key, {}).get(version, {})
    score = bucket.get(task, {}).get(model_json_key)
    if score is not None:
        return score
    # Special case: pull the v2 DINOv3 score from the per-run score_log
    if version == "v2" and metric_key == "dinov3":
        return last_score_log_dinov3(task, model_dir)
    return None


def create_visualization(task: str, model_dirs: list[str], metrics_data: dict):
    ref_img = open_image(find_reference_image(task))

    # Build column spec: ("Reference", image, version=None, model_dir=None, model_json_key=None)
    # followed by one cell per (model, version) pair.
    columns: list[tuple[str, Image.Image | None, str | None, str | None, str | None]] = [
        ("Reference", ref_img, None, None, None),
    ]
    for model_dir in model_dirs:
        model_json_key, label = MODEL_REGISTRY[model_dir]
        v1_img = open_image(find_v1_render(task, model_dir))
        v2_path = find_v2_render(task, model_dir)
        v2_img = open_image(v2_path)
        columns.append((f"{label}\n(v1)", v1_img, "v1", model_dir, model_json_key))
        columns.append((f"{label}\n(v2)", v2_img, "v2", model_dir, model_json_key))

    n_metrics = len(METRICS)
    n_cols = len(columns)

    img_height = 3.5
    metric_height = 0.42
    fig_height = img_height + n_metrics * metric_height + 0.6
    fig_width = 3.5 * n_cols

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        1 + n_metrics, n_cols,
        figure=fig,
        height_ratios=[img_height] + [metric_height] * n_metrics,
        hspace=0.0,
        wspace=0.03,
    )

    # --- Image row ---
    for col, (label, img, version, _, _) in enumerate(columns):
        ax = fig.add_subplot(gs[0, col])
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#cccccc")
            msg = "Not run" if version == "v2" else "Missing"
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    fontsize=11, color="#666666", transform=ax.transAxes)
        # Color the title by version for at-a-glance grouping
        title_color = "#1f4e79" if version == "v1" else ("#a04000" if version == "v2" else "#000000")
        ax.set_title(label, fontsize=11, fontweight="bold", color=title_color, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Metric rows ---
    for row_idx, (metric_key, metric_label, _, _) in enumerate(METRICS):
        row = row_idx + 1

        # Compute every cell value first so we can find the best (min) per row.
        cell_scores: list[float | None] = [None]  # reference col has no score
        for _, _, version, model_dir, model_json_key in columns[1:]:
            cell_scores.append(metric_value(
                metric_key, version, task, model_dir, model_json_key, metrics_data
            ))
        valid = [s for s in cell_scores if s is not None]
        best = min(valid) if valid else None

        # Metric label (col 0)
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor("#e8e8e8")
        ax0.text(0.5, 0.5, metric_label, ha="center", va="center",
                 fontsize=9, fontweight="bold", transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_linewidth(0.5)

        # Score cells (cols 1..n)
        for col_idx, (_, _, version, _, _) in enumerate(columns[1:], start=1):
            ax = fig.add_subplot(gs[row, col_idx])
            score = cell_scores[col_idx]
            is_best = best is not None and score == best
            base_bg = V1_BG if version == "v1" else V2_BG
            ax.set_facecolor(BEST_BG if is_best else base_bg)
            if score is None:
                text = "pending" if version == "v2" else "N/A"
                color = "#888888"
            else:
                text = f"{score:.4f}"
                color = "#000000"
            ax.text(0.5, 0.5, text, ha="center", va="center",
                    fontsize=9, fontweight="bold" if is_best else "normal",
                    color=color, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    fig.suptitle(f"{task}  —  v1 vs v2", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)

    out_path = VIZ_DIR / f"{task}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_DIRS,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model dir names to compare (one v1/v2 pair per model).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names to render (default: all 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    metrics_data = load_metrics()
    tasks = args.tasks if args.tasks else TASKS

    for task in tasks:
        print(f"Rendering {task}...")
        create_visualization(task, args.models, metrics_data)

    print(f"\nDone. Visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
