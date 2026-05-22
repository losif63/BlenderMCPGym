"""
Compute CLIP (ViT-B/32) cosine distances between reference images and
model-reconstructed renders. Results saved to clip_vitb32.json.

Structure:
  Reference: images/{task_name}.{ext}
  Renders:   recreation/{task_name}/claudecode/{model_dir}/render.png
"""

import json
import glob
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
RECREATION_DIR = ROOT / "recreation"
OUTPUT_FILE = ROOT / "clip_vitb32.json"

PRETRAINED_MODEL = "openai/clip-vit-base-patch32"

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


def extract_embedding(image: Image.Image, processor, model, device) -> torch.Tensor:
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features.squeeze(0)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return (1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {PRETRAINED_MODEL} on {device}...")
    processor = CLIPProcessor.from_pretrained(PRETRAINED_MODEL)
    model = CLIPModel.from_pretrained(PRETRAINED_MODEL).to(device)
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

        for model_key, model_dir in MODELS.items():
            render_path = RECREATION_DIR / task / "claudecode" / model_dir / "render.png"
            if not render_path.exists():
                print(f"  [MISSING] {task}/{model_key}: {render_path}")
                results[task][model_key] = None
                continue

            render_image = Image.open(render_path).convert("RGB")
            render_emb = extract_embedding(render_image, processor, model, device)
            dist = cosine_distance(ref_emb, render_emb)
            results[task][model_key] = dist
            print(f"  {task} / {model_key}: {dist:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
