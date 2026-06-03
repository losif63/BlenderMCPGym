#!/usr/bin/env python3
"""
Build a per-task comparison montage: the reference image alongside the latest
process/ render from each opus-4.7 condition, side by side with large labels and
the FED-metric distances (DINOv3, SigLIP2, ConvNeXt2, DreamSim) beneath each
render.

Panels (left to right):
  Reference            images/{task}.{ext}
  Old (v2)             recreation_old/{task}/claudecode_v2/{model}/process/<latest>
  Avg (ensemble_avg)   recreation/{task}/claudecode_v2_avg/{model}/process/<latest>
  Vector (ens_vector)  recreation/{task}/claudecode_v2_vector/{model}/process/<latest>

Scores are computed through src/variance/metrics.py (same checkpoints/devices as
the in-run signal and the *_eval.py scripts), so they're directly comparable.
Pass --no-scores to skip model loading and render images only.

Missing panels render as a gray "missing" placeholder so the grid stays aligned.
Output: montage/{task}.png
"""

import argparse
import glob
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from eval_common import latest_process_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "variance"))
from metrics import LABELS, MetricEnsemble  # noqa: E402

IMAGES_DIR = ROOT / "images"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

# FED metrics shown beneath each render, in display order.
SCORE_ORDER = ["dinov3", "siglip2", "convnextv2", "dreamsim"]

PANEL_H = 384            # all panels normalized to this height
TITLE_FONT_SIZE = 34
SCORE_FONT_SIZE = 24
TITLE_H = TITLE_FONT_SIZE + 18
SCORE_LINE_H = SCORE_FONT_SIZE + 8
PAD = 8
BG = (255, 255, 255)
PLACEHOLDER_BG = (60, 60, 60)
INK = (0, 0, 0)
MUTED = (120, 120, 120)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "DejaVuSans.ttf",
]


@lru_cache(maxsize=None)
def get_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def find_reference_image(task: str) -> Path | None:
    matches = glob.glob(str(IMAGES_DIR / f"{task}.*"))
    return Path(matches[0]) if matches else None


def load_panel(img_path: Path | None) -> Image.Image:
    """A PANEL_H-tall RGB image; gray placeholder if img_path is None/unreadable."""
    if img_path is not None:
        try:
            img = Image.open(img_path).convert("RGB")
            w = max(1, round(img.width * PANEL_H / img.height))
            return img.resize((w, PANEL_H), Image.LANCZOS)
        except OSError:
            pass
    ph = Image.new("RGB", (round(PANEL_H * 4 / 3), PANEL_H), PLACEHOLDER_BG)
    ImageDraw.Draw(ph).text(
        (ph.width // 2 - 36, PANEL_H // 2), "missing",
        fill=(200, 200, 200), font=get_font(SCORE_FONT_SIZE),
    )
    return ph


def build_column(label: str, img_path: Path | None, scores: dict | None,
                 is_reference: bool, scored: bool) -> Image.Image:
    panel = load_panel(img_path)
    title_font = get_font(TITLE_FONT_SIZE)
    score_font = get_font(SCORE_FONT_SIZE)

    score_h = (len(SCORE_ORDER) * SCORE_LINE_H + 16) if scored else 0

    # Column wide enough for the panel and the score text.
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_w = max(
        (measure.textlength(f"{LABELS[k]}:", font=score_font) for k in SCORE_ORDER),
        default=0,
    )
    value_x = 8 + label_w + 12
    min_w = int(value_x + measure.textlength("0.000", font=score_font) + 8)
    col_w = max(panel.width, min_w)
    total_h = TITLE_H + panel.height + score_h

    col = Image.new("RGB", (col_w, total_h), BG)
    d = ImageDraw.Draw(col)

    # Title.
    d.text((8, 6), label, fill=INK, font=title_font)
    # Image (centered if the column was widened for text).
    col.paste(panel, ((col_w - panel.width) // 2, TITLE_H))

    # Scores.
    if scored:
        y = TITLE_H + panel.height + 8
        if scores:
            for k in SCORE_ORDER:
                d.text((8, y), f"{LABELS[k]}:", fill=INK, font=score_font)
                d.text((value_x, y), f"{scores[k]:.3f}", fill=INK, font=score_font)
                y += SCORE_LINE_H
        else:
            d.text((8, y), "(target)" if is_reference else "(no score)",
                   fill=MUTED, font=score_font)
    return col


def build_task_montage(task: str, model: str, rec_new: Path, rec_old: Path,
                       ensemble: MetricEnsemble | None) -> Image.Image:
    ref_path = find_reference_image(task)
    conditions = [
        ("Reference", ref_path, True),
        ("Old (v2)", latest_process_image(rec_old / task / "claudecode_v2" / model), False),
        ("Avg", latest_process_image(rec_new / task / "claudecode_v2_avg" / model), False),
        ("Vector", latest_process_image(rec_new / task / "claudecode_v2_vector" / model), False),
    ]

    columns = []
    for label, img_path, is_ref in conditions:
        scores = None
        if ensemble is not None and not is_ref and ref_path is not None and img_path is not None:
            try:
                scores = ensemble.compute_all(str(ref_path), str(img_path))
            except Exception as e:
                print(f"    [score fail] {task}/{label}: {e}")
        columns.append(
            build_column(label, img_path, scores, is_ref, scored=ensemble is not None)
        )

    total_w = sum(c.width for c in columns) + PAD * (len(columns) + 1)
    height = max(c.height for c in columns) + PAD * 2
    canvas = Image.new("RGB", (total_w, height), BG)
    x = PAD
    for c in columns:
        canvas.paste(c, (x, PAD))
        x += c.width + PAD
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Build per-task comparison montages.")
    parser.add_argument("--model", default="opus-4.7", help="Model dir name (default: opus-4.7).")
    parser.add_argument("--recreation-dir", default="recreation",
                        help="Root for avg/vector runs (default: recreation).")
    parser.add_argument("--recreation-old-dir", default="recreation_old",
                        help="Root for the old claudecode_v2 runs (default: recreation_old).")
    parser.add_argument("--task", nargs="+", default=None, help="Task(s) (default: all).")
    parser.add_argument("--out", default="montage", help="Output dir (default: montage/).")
    parser.add_argument("--no-scores", action="store_true",
                        help="Skip metric model loading; render images and titles only.")
    args = parser.parse_args()

    rec_new = Path(args.recreation_dir)
    rec_new = rec_new if rec_new.is_absolute() else ROOT / rec_new
    rec_old = Path(args.recreation_old_dir)
    rec_old = rec_old if rec_old.is_absolute() else ROOT / rec_old
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble = None
    if not args.no_scores:
        print(f"Loading FED metrics {SCORE_ORDER} for score overlay...")
        ensemble = MetricEnsemble(SCORE_ORDER, [])

    tasks = args.task if args.task else TASKS
    made = 0
    for task in tasks:
        montage = build_task_montage(task, args.model, rec_new, rec_old, ensemble)
        out_path = out_dir / f"{task}.png"
        montage.save(out_path)
        print(f"  {task} -> {out_path.relative_to(ROOT)}")
        made += 1

    print(f"\nDone. {made} montages in {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
