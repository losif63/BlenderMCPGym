"""
Compute LPIPS (AlexNet) distances between reference images and
model-reconstructed renders. Results saved to lpips_alex.json.

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/claudecode/{model_dir}/render.png
"""

import json
import glob
import lpips
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_FILE = ROOT / "lpips_alex.json"

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


def main():
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

        for model_key, model_dir in MODELS.items():
            render_path = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
            if not render_path.exists():
                print(f"  [MISSING] {task}/{model_key}: {render_path}")
                results[task][model_key] = None
                continue

            # Resize render to match reference so LPIPS can compare them
            render_tensor = load_tensor(render_path, size=ref_size).to(device)

            with torch.no_grad():
                dist = loss_fn(ref_tensor, render_tensor).item()

            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
