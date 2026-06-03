"""
Compute ConvNeXt-v2 cosine distances between reference images and
model-reconstructed renders. Results saved to convnextv2_atto.json (or
convnextv2_atto_v2.json when --platform claudecode_v2 is passed).

The distance is computed through the shared in-run metric module
(src/variance/metrics.py), so the checkpoint, preprocessing, device handling,
and cosine-distance definition are byte-for-byte identical to the ConvNeXt-v2
signal the agent receives during a v2 run. In particular this metric runs on CPU
by default, because its MPS kernels are non-deterministic (see
metrics.DEFAULT_DEVICE_OVERRIDES). Override the checkpoint with --checkpoint to
match a swapped in-run variant.

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/{platform}/{model_dir}/render.png
"""

import argparse
import json
import glob
import sys
from pathlib import Path
from eval_common import latest_process_image, output_suffix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "variance"))
from metrics import build_metric  # noqa: E402

IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_STEM = "convnextv2_atto"
METRIC = "convnextv2"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

MODELS = {
    "haiku4.5": "haiku-4.5",
    "sonnet4.6": "sonnet-4.6",
    "opus4.6": "opus-4.6",
    "opus4.7": "opus-4.7"
}


def find_reference_image(task_name: str) -> Path:
    matches = glob.glob(str(IMAGES_DIR / f"{task_name}.*"))
    if not matches:
        raise FileNotFoundError(f"No reference image found for task '{task_name}'")
    return Path(matches[0])


def main(args):
    recreation_dir = Path(args.recreation_dir)
    if not recreation_dir.is_absolute():
        recreation_dir = ROOT / recreation_dir
    suffix = output_suffix(args.platform, recreation_dir.name)
    output_file = ROOT / f"{OUTPUT_STEM}{suffix}.json"
    models = {k: v for k, v in MODELS.items()
              if not args.model or v in args.model or k in args.model}

    metric = build_metric(METRIC, checkpoint=args.checkpoint)
    print(f"Loaded {METRIC} ({metric.checkpoint}) on {metric.device}")

    results = {}

    for task in TASKS:
        results[task] = {}
        try:
            ref_path = find_reference_image(task)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        for model_key, model_dir in models.items():
            mdir = recreation_dir / task / args.platform / model_dir
            render_path = latest_process_image(mdir)
            if render_path is None:
                print(f"  [MISSING] no process/*.png: {mdir}")
                results[task][model_key] = None
                continue

            dist = metric.distance(str(ref_path), str(render_path))
            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", default="claudecode",
        choices=["claudecode", "claudecode_v2", "claudecode_v2_avg", "claudecode_v2_vector"],
        help="Recreation platform dir to score. Output suffix mirrors the platform (e.g. _v2, _v2_avg, _v2_vector).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override the ConvNeXt-v2 timm checkpoint (default: metric module default).",
    )
    parser.add_argument(
        "--recreation-dir", default="recreation",
        help="Root with {task}/{platform}/{model}/process/ (e.g. recreation or recreation_old).",
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Limit to these model dir names (e.g. opus-4.7). Default: all.",
    )
    main(parser.parse_args())
