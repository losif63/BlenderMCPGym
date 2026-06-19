"""Metric definitions: photometric loss and negative CLIP score.

Both compare two rendered images (edit vs goal). Lower is better for each.
The CLIP model is loaded once and reused across calls.
"""

import numpy as np
from PIL import Image

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP = None  # lazily-loaded (model, processor, torch)


def photometric_loss(image1: Image.Image, image2: Image.Image) -> float:
    """Mean-squared error between two RGB images in [0,1]. Lower is better.

    Ported verbatim from the original utils.photometric_loss.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    a = np.array(image1)[:, :, :3].astype(np.float32) / 255.0
    b = np.array(image2)[:, :, :3].astype(np.float32) / 255.0
    return float(np.mean(np.square(a - b)))


def _get_clip(model_name: str = CLIP_MODEL_NAME):
    global _CLIP
    if _CLIP is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP = (model, processor, torch)
    return _CLIP


def clip_similarity(image1: Image.Image, image2: Image.Image,
                    model_name: str = CLIP_MODEL_NAME) -> float:
    """Cosine similarity of the two images' CLIP embeddings, in [-1, 1]."""
    model, processor, torch = _get_clip(model_name)
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    inputs = processor(
        images=[image1.convert("RGB"), image2.convert("RGB")],
        return_tensors="pt",
    )
    with torch.no_grad():
        # vision pooler_output -> visual_projection gives the (N, 512) CLIP image
        # embeddings, stable across transformers versions (get_image_features
        # returns a model-output object in transformers 5.x).
        pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        embeds = model.visual_projection(pooled)
    sim = torch.nn.functional.cosine_similarity(embeds[0], embeds[1], dim=-1)
    return float(sim.item())


def negative_clip(image1: Image.Image, image2: Image.Image,
                  model_name: str = CLIP_MODEL_NAME) -> float:
    """n_clip = 1 - CLIP similarity. Lower is better (matches the original eval)."""
    return 1.0 - clip_similarity(image1, image2, model_name)
