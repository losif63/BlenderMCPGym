"""
Compute DINOv3 (vitb16) cosine distances between reference images and
model-reconstructed renders. Results saved to dinov3_vitb16.json.

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/claudecode/{model_dir}/render.png
"""

import json
import glob
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_FILE = ROOT / "dinov3_vitl16.json"

PRETRAINED_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

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


def extract_embedding(image_path: str, processor, model) -> torch.Tensor:
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze(0)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return (1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    print(f"Loading {PRETRAINED_MODEL}...")
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model = AutoModel.from_pretrained(PRETRAINED_MODEL, device_map="auto")
    model.eval()

    results = {}

    for task in TASKS:
        results[task] = {}
        try:
            ref_path = find_reference_image(task)
            ref_emb = extract_embedding(str(ref_path), processor, model)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        for model_key, model_dir in MODELS.items():
            render_path = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
            if not render_path.exists():
                print(f"  [MISSING] {task}/{model_key}: {render_path}")
                results[task][model_key] = None
                continue

            render_emb = extract_embedding(str(render_path), processor, model)
            dist = cosine_distance(ref_emb, render_emb)
            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
