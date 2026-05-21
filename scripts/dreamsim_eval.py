"""
Compute DreamSim distances between reference images and
model-reconstructed renders. Results saved to dreamsim.json.

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/claudecode/{model_dir}/render.png
"""

import json
import glob
from pathlib import Path
from PIL import Image
from dreamsim import dreamsim

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_FILE = ROOT / "dreamsim.json"

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]

MODELS = {
    "haiku4.5": "haiku-4.5",
    "sonnet4.6": "sonnet-4.6",
    "opus4.6": "opus-4.6",
}


def find_reference_image(task_name: str) -> Path:
    matches = glob.glob(str(IMAGES_DIR / f"{task_name}.*"))
    if not matches:
        raise FileNotFoundError(f"No reference image found for task '{task_name}'")
    return Path(matches[0])


def main():
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
            render_path = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
            if not render_path.exists():
                print(f"  [MISSING] {task}/{model_key}: {render_path}")
                results[task][model_key] = None
                continue

            render_tensor = preprocess(Image.open(render_path).convert("RGB")).to(device)
            dist = model(ref_tensor, render_tensor).item()
            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
