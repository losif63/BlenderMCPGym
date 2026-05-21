#!/usr/bin/env python3
"""
Render render.png for every haiku-4.5 / sonnet-4.6 / opus-4.6 blend file
found under recreation/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BLENDER = Path(__file__).resolve().parent.parent / "infinigen/Blender.app/Contents/MacOS/Blender"
RECREATION_DIR = Path(__file__).resolve().parent.parent / "recreation"
# ALL_MODELS = ["haiku-4.5", "sonnet-4.6", "opus-4.6"]
ALL_MODELS = ["opus-4.7"]

RENDER_SCRIPT = """
import bpy, sys

blend_path = sys.argv[sys.argv.index("--") + 1]
out_path    = sys.argv[sys.argv.index("--") + 2]

bpy.ops.wm.open_mainfile(filepath=blend_path)

scene = bpy.context.scene

# Find the first camera in the scene
camera = next(
    (obj for obj in scene.objects if obj.type == "CAMERA"),
    None,
)
if camera is None:
    print("ERROR: no camera found in", blend_path, file=sys.stderr)
    sys.exit(1)

scene.camera = camera
scene.render.filepath = out_path
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("Rendered:", out_path)
"""


def render(blend_file: Path, out_file: Path) -> bool:
    cmd = [
        str(BLENDER),
        "--background",
        "--python-expr", RENDER_SCRIPT,
        "--",
        str(blend_file),
        str(out_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {blend_file.relative_to(RECREATION_DIR)}")
        print(result.stderr[-2000:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Render blend files under recreation/")
    parser.add_argument("--model", nargs="+", default=ALL_MODELS,
                        help="Model dir name(s) to render (default: all)")
    parser.add_argument("--task", nargs="+", default=None,
                        help="Task name(s) to render (default: all)")
    args = parser.parse_args()

    models = args.model
    tasks = set(args.task) if args.task else None

    if not BLENDER.exists():
        sys.exit(f"Blender not found at {BLENDER}")

    jobs: list[tuple[Path, Path]] = []

    for task_dir in sorted(RECREATION_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        if tasks and task_dir.name not in tasks:
            continue
        for platform_dir in sorted(task_dir.iterdir()):
            if not platform_dir.is_dir():
                continue
            for model in models:
                model_dir = platform_dir / model
                blend_file = model_dir / "blender_file.blend"
                if blend_file.exists():
                    jobs.append((blend_file, model_dir / "render.png"))

    print(f"Found {len(jobs)} blend files to render.\n")

    success, failed = 0, 0
    for i, (blend_file, out_file) in enumerate(jobs, 1):
        label = f"{blend_file.parts[-4]}/{blend_file.parts[-3]}/{blend_file.parts[-2]}"
        print(f"[{i}/{len(jobs)}] {label} ...", end=" ", flush=True)
        if render(blend_file, out_file):
            print("OK")
            success += 1
        else:
            failed += 1

    print(f"\nDone. {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
