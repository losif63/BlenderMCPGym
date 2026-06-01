"""
MCP server that scores rendered images against a fixed reference image using DINOv3.

Launched by `run_claudecode_v2.py` as part of a per-task MCP config. Exposes a single
tool, `score_render(render_path)`, that the agent calls after each render to receive
the DINOv3 cosine distance between its render and the reference image (lower = closer).

Configuration (env vars):
  REFERENCE_IMAGE  Absolute path to the reference image (required).
  DINOV3_MODEL     HuggingFace model id (default: facebook/dinov3-vitl16-pretrain-lvd1689m).
  SCORE_LOG_PATH   If set, append one JSONL record per score_render call to this path.
                   Lets you inspect the agent's render trajectory after the run.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from mcp.server.fastmcp import FastMCP
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

PRETRAINED_MODEL = os.environ.get(
    "DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m"
)

REFERENCE_IMAGE = os.environ.get("REFERENCE_IMAGE")
if not REFERENCE_IMAGE:
    print("ERROR: REFERENCE_IMAGE env var must be set", file=sys.stderr)
    sys.exit(1)
if not Path(REFERENCE_IMAGE).exists():
    print(f"ERROR: reference image not found: {REFERENCE_IMAGE}", file=sys.stderr)
    sys.exit(1)

SCORE_LOG_PATH = os.environ.get("SCORE_LOG_PATH")
if SCORE_LOG_PATH:
    Path(SCORE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"[feedback_mcp_server] logging scores to {SCORE_LOG_PATH}", file=sys.stderr)

print(f"[feedback_mcp_server] loading {PRETRAINED_MODEL}...", file=sys.stderr)
processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
model = AutoModel.from_pretrained(PRETRAINED_MODEL, device_map="auto")
model.eval()


def extract_embedding(image_path: str) -> torch.Tensor:
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze(0)


print(
    f"[feedback_mcp_server] embedding reference image {REFERENCE_IMAGE}...",
    file=sys.stderr,
)
ref_embedding = extract_embedding(REFERENCE_IMAGE)
print("[feedback_mcp_server] ready.", file=sys.stderr)

mcp = FastMCP("render_feedback")


@mcp.tool()
def score_render(render_path: str) -> dict:
    """Score how close a rendered image is to the target reference image you are
    trying to reproduce.

    THIS IS THE ONLY SANCTIONED WAY TO SCORE A RENDER. Call this tool after every
    render you make. Do not write a Python script, do not import from
    feedback_mcp_server.py, do not search the filesystem for scoring utilities —
    just call this tool with the render's absolute path.

    Returns the cosine distance between the rendered image and the reference image,
    computed using the DINOv3 model. Lower distance = closer to the reference;
    0.0 = identical embeddings. This is the same metric your final output will be
    evaluated by.

    Args:
        render_path: Absolute path to the rendered PNG you want scored.

    Returns:
        Dict with the DINOv3 cosine distance and the paths used.
    """
    rp = Path(render_path).expanduser()
    if not rp.is_absolute():
        rp = Path.cwd() / rp
    rp = rp.resolve()
    if not rp.exists():
        return {"error": f"file not found: {rp}"}
    try:
        render_embedding = extract_embedding(str(rp))
    except Exception as e:
        return {"error": f"failed to embed render: {e}"}
    cos_sim = F.cosine_similarity(
        ref_embedding.unsqueeze(0), render_embedding.unsqueeze(0)
    ).item()
    distance = 1.0 - cos_sim

    if SCORE_LOG_PATH:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "render_path": str(rp),
            "reference_image": REFERENCE_IMAGE,
            "dinov3_cosine_distance": distance,
        }
        try:
            with open(SCORE_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(
                f"[feedback_mcp_server] failed to write {SCORE_LOG_PATH}: {e}",
                file=sys.stderr,
            )

    return {
        "dinov3_cosine_distance": distance,
        "reference_image": REFERENCE_IMAGE,
        "render_image": str(rp),
        "note": "Lower is closer to the reference; 0.0 means identical embeddings.",
    }


if __name__ == "__main__":
    mcp.run()
