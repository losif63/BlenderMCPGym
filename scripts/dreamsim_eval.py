"""
Compute DreamSim distances between reference images and
model-reconstructed renders. Results saved to dreamsim.json (or
dreamsim_v2.json when --platform claudecode_v2 is passed).

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/{platform}/{model_dir}/render.png
"""

import argparse
import json
import glob
from pathlib import Path
from PIL import Image
from dreamsim import dreamsim

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_STEM = "dreamsim"

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
    suffix = "_v2" if args.platform == "claudecode_v2" else ""
    output_file = ROOT / f"{OUTPUT_STEM}{suffix}.json"

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Loading DreamSim on {device}...")
    model, preprocess = dreamsim(pretrained=True, device=device)

    results = {}

    for task in TASKS:
        results[task] = {}
        try:
            ref_path = find_reference_image(task)
            ref_tensor = preprocess(Image.open(ref_path).convert("RGB")).to(device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        for model_key, model_dir in MODELS.items():
            render_path = RECREATION_DIR / task / args.platform / model_dir / "render.png"
            if not render_path.exists():
                print(f"  [MISSING] {task}/{model_key}: {render_path}")
                results[task][model_key] = None
                continue

            render_tensor = preprocess(Image.open(render_path).convert("RGB")).to(device)
            dist = model(ref_tensor, render_tensor).item()
            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", default="claudecode",
        choices=["claudecode", "claudecode_v2"],
        help="Recreation platform dir to score. Writes *_v2.json for claudecode_v2.",
    )
    main(parser.parse_args())
