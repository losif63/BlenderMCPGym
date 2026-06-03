"""
Compute DINOv2-large cosine distances between reference images and
model-reconstructed renders. Results saved to dinov2_large.json (or
dinov2_large_v2.json when --platform claudecode_v2 is passed).

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/{platform}/{model_dir}/render.png
"""

import argparse
import json
import glob
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from eval_common import latest_process_image, output_suffix
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_STEM = "dinov2_large"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

# Map user-facing model keys to directory names
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


def extract_embedding(image: Image.Image, processor, model, device) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze(0)  # shape: (hidden_size,)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return (1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main(args):
    recreation_dir = Path(args.recreation_dir)
    if not recreation_dir.is_absolute():
        recreation_dir = ROOT / recreation_dir
    suffix = output_suffix(args.platform, recreation_dir.name)
    output_file = ROOT / f"{OUTPUT_STEM}{suffix}.json"
    models = {k: v for k, v in MODELS.items()
              if not args.model or v in args.model or k in args.model}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2-large on {device}...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
    model.eval()

    results = {}

    for task in TASKS:
        results[task] = {}
        try:
            ref_path = find_reference_image(task)
            ref_image = Image.open(ref_path).convert("RGB")
            ref_emb = extract_embedding(ref_image, processor, model, device)
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

            render_image = Image.open(render_path).convert("RGB")
            render_emb = extract_embedding(render_image, processor, model, device)
            dist = cosine_distance(ref_emb, render_emb)
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
        "--recreation-dir", default="recreation",
        help="Root with {task}/{platform}/{model}/process/ (e.g. recreation or recreation_old).",
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Limit to these model dir names (e.g. opus-4.7). Default: all.",
    )
    main(parser.parse_args())
