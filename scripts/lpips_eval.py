"""
Compute LPIPS (AlexNet) distances between reference images and
model-reconstructed renders. Results saved to lpips_alex.json (or
lpips_alex_v2.json when --platform claudecode_v2 is passed).

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/{platform}/{model_dir}/render.png
"""

import argparse
import json
import glob
import lpips
import torch
import torchvision.transforms as T
from pathlib import Path
from eval_common import latest_process_image, output_suffix
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_STEM = "lpips_alex"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

MODELS = {
    "haiku4.5": "haiku-4.5",
    "sonnet4.6": "sonnet-4.6",
    "opus4.6": "opus-4.6",
    "opus4.7": "opus-4.7"
}

# LPIPS expects tensors in [-1, 1] with shape (1, 3, H, W)
to_tensor = T.Compose([
    T.ToTensor(),           # [0, 1]
    T.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
])


def find_reference_image(task_name: str) -> Path:
    matches = glob.glob(str(IMAGES_DIR / f"{task_name}.*"))
    if not matches:
        raise FileNotFoundError(f"No reference image found for task '{task_name}'")
    return Path(matches[0])


def load_tensor(image_path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return to_tensor(img).unsqueeze(0)  # (1, 3, H, W)


def main(args):
    recreation_dir = Path(args.recreation_dir)
    if not recreation_dir.is_absolute():
        recreation_dir = ROOT / recreation_dir
    suffix = output_suffix(args.platform, recreation_dir.name)
    output_file = ROOT / f"{OUTPUT_STEM}{suffix}.json"
    models = {k: v for k, v in MODELS.items()
              if not args.model or v in args.model or k in args.model}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LPIPS (AlexNet) on {device}...")
    loss_fn = lpips.LPIPS(net="alex").to(device)

    results = {}

    for task in TASKS:
        results[task] = {}
        try:
            ref_path = find_reference_image(task)
            ref_tensor = load_tensor(ref_path).to(device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        ref_size = (ref_tensor.shape[3], ref_tensor.shape[2])  # (W, H) for PIL

        for model_key, model_dir in models.items():
            mdir = recreation_dir / task / args.platform / model_dir
            render_path = latest_process_image(mdir)
            if render_path is None:
                print(f"  [MISSING] no process/*.png: {mdir}")
                results[task][model_key] = None
                continue

            # Resize render to match reference so LPIPS can compare them
            render_tensor = load_tensor(render_path, size=ref_size).to(device)

            with torch.no_grad():
                dist = loss_fn(ref_tensor, render_tensor).item()

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
