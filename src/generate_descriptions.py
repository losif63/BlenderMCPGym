"""
Generate textual descriptions of before-and-after changes for each BlenderGym task instance.

For each bench_data/{task}{N}/ instance, sends all start and goal render images to
the Gemini API and writes the generated description to bench_data/{task}{N}/description.txt.

Usage:
    python generate_descriptions.py
    python generate_descriptions.py --api_key YOUR_KEY
    python generate_descriptions.py --task placement geometry   # run only specific task types
    python generate_descriptions.py --overwrite                 # regenerate existing descriptions
"""

import os
import re
import argparse
import base64
import time
from pathlib import Path
from glob import glob

import google.generativeai as genai
from PIL import Image

# ── Task types present in bench_data/ ────────────────────────────────────────
TASK_TYPES = ['blendshape', 'geometry', 'lighting', 'material', 'placement']

PROMPT = """
You are analyzing a 3D scene editing task in Blender. I will show you rendered images of a
scene BEFORE and AFTER a single programmatic edit was applied.

The BEFORE images are the starting scene rendered from one or more camera angles.
The AFTER images are the goal scene rendered from the same camera angles.

First, describe precisely what changed between the before and after scenes. Focus on:
- What object or property was modified (e.g. a light, a material, an object's position, a shape)
- The direction and magnitude of the change where applicable (e.g. brighter, moved left, scaled up)
- Any color, texture, geometry, or placement differences

Be concise and specific. Write 1-3 sentences. Do not describe what stayed the same.

Next, write a concise instruction (1-3 sentences) that tells an AI agent exactly how to edit the
BEFORE scene to produce the AFTER scene, based on your observations. Use imperative language (e.g. "Move...", "Change...",
"Increase...", "Apply..."). Be specific about what to modify and in which direction or to what
value where apparent. Do not describe what stays the same. Begin your instruction with the delimiter INSTRUCTION:
"""

DETAILED_PROMPT = """
You are an expert 3D scene analyst. I will show you rendered images of a Blender scene BEFORE
and AFTER a single programmatic edit was applied, from one or more camera angles.

Your job is to scrutinize every minor detail and find every little difference between the two
scenes — no matter how subtle. Examine colors, shading, shadows, reflections, object positions,
rotations, scales, geometry, textures, materials, lighting intensity and color, and any other
visual properties you can observe. Compare the images pixel by pixel in your mind.

Then write an extremely detailed instruction (at minimum one full paragraph) that tells an AI
agent exactly how to reproduce the change in Blender. The instruction must:
- Identify the exact object(s) or property/properties affected
- Describe every observable difference with as much numerical or directional specificity as possible
  (e.g. approximate axis of movement, relative scale change, hue shift, brightness delta)
- Note subtle secondary effects such as changes in shadows, reflections, or indirect lighting
- Use imperative language (e.g. "Move...", "Set...", "Increase...", "Apply...")
- Be thorough enough that an agent with no prior knowledge of the scene could reproduce the edit
  precisely from your description alone

Do not mention what stayed the same. Begin your instruction with the delimiter INSTRUCTION:
"""


def load_api_key(api_key_arg: str) -> str:
    if api_key_arg:
        return api_key_arg
    cred_path = Path("system/credentials/gemini_api.txt")
    if cred_path.exists():
        key = cred_path.read_text().strip()
        if key and key != "your-api-here":
            return key
    raise ValueError(
        "No Gemini API key found. Pass --api_key or fill in system/credentials/gemini_api.txt"
    )


def collect_instances(bench_data_dir: Path, task_filter: list) -> list:
    """Return sorted list of instance dirs matching the given task types."""
    instances = []
    for entry in sorted(bench_data_dir.iterdir()):
        if not entry.is_dir():
            continue
        # match e.g. 'placement1', 'geometry23'
        m = re.match(r'^([a-z]+)(\d+)$', entry.name)
        if not m:
            continue
        task_type = m.group(1)
        if task_type not in task_filter:
            continue
        instances.append(entry)
    return instances


def load_images_as_pil(render_dir: Path) -> list:
    """Return sorted list of PIL Images from a renders/{start,goal}/ directory."""
    paths = sorted(render_dir.glob("*.png"))
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return images


def describe_instance(model, instance_dir: Path) -> tuple[str, str]:
    start_dir = instance_dir / "renders" / "start"
    goal_dir = instance_dir / "renders" / "goal"

    start_images = load_images_as_pil(start_dir)
    goal_images = load_images_as_pil(goal_dir)

    if not start_images:
        raise FileNotFoundError(f"No start renders found in {start_dir}")
    if not goal_images:
        raise FileNotFoundError(f"No goal renders found in {goal_dir}")

    # Build shared image block: label + images for before, then after
    image_block = ["BEFORE (start scene):"] + start_images + ["AFTER (goal scene):"] + goal_images

    response = model.generate_content([PROMPT] + image_block)
    description = response.text.strip()

    detailed_response = model.generate_content([DETAILED_PROMPT] + image_block)
    detailed_instruction = detailed_response.text.strip()

    return description, detailed_instruction


def main():
    parser = argparse.ArgumentParser(description="Generate textual descriptions for BlenderGym instances")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--task", nargs="+", default=TASK_TYPES,
                        choices=TASK_TYPES, help="Task types to process (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing description.txt files")
    parser.add_argument("--bench_data", type=str, default="bench_data",
                        help="Path to bench_data directory")
    parser.add_argument("--model", type=str, default="models/gemma-4-31b-it",
                        help="Gemini model to use")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between API calls (avoid rate limits)")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    bench_data_dir = Path(args.bench_data)
    if not bench_data_dir.exists():
        raise FileNotFoundError(f"bench_data directory not found: {bench_data_dir}")

    instances = collect_instances(bench_data_dir, args.task)
    print(f"Found {len(instances)} instances across tasks: {args.task}")

    success = 0
    skipped = 0
    failed = 0

    for i, instance_dir in enumerate(instances):
        desc_path = instance_dir / "description.txt"
        detailed_path = instance_dir / "detailed_instruction.txt"

        if desc_path.exists() and detailed_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(instances)}] {instance_dir.name}: skipping (already exists)")
            skipped += 1
            continue

        print(f"[{i+1}/{len(instances)}] {instance_dir.name}: generating...", end=" ", flush=True)
        try:
            description, detailed_instruction = describe_instance(model, instance_dir)
            desc_path.write_text(description + "\n")
            detailed_path.write_text(detailed_instruction + "\n")
            print(f"done.")
            print(f"  → {description}")
            print(f"  → (detailed) {detailed_instruction[:120]}...")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        if i < len(instances) - 1:
            time.sleep(args.delay)

    print(f"\nDone. {success} generated, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
