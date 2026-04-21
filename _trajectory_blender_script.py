"""
Inner Blender-side script for trajectory rendering.
Invoked by render_trajectory.py via:
  blender --background trajectory_logger.blend --python _trajectory_blender_script.py -- <config.json>

Reads a JSON config, executes start.py to initialise the scene, then for every
code file (in ascending seq order) executes it and renders all available cameras.
Output images: trajectory_dir/seq_{num:03d}_cam{i}.png
"""

import bpy
import json
import os
import sys


def setup_rendering() -> None:
    """Configure Cycles renderer with the best available compute device."""
    bpy.context.scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("METAL", "CUDA", "OPTIX", "NONE"):
        try:
            prefs.compute_device_type = device_type
            break
        except TypeError:
            continue
    prefs.get_devices()
    for device in prefs.devices:
        if device.type != "CPU":
            device.use = True
    bpy.context.scene.cycles.device = (
        "GPU" if prefs.compute_device_type != "NONE" else "CPU"
    )
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.render.image_settings.color_mode = "RGB"
    bpy.context.scene.render.image_settings.file_format = "PNG"


def render_all_cameras(trajectory_dir: str, seq_num: int) -> None:
    """Render from every CameraX present in the scene."""
    for cam_idx in range(1, 6):
        cam_name = f"Camera{cam_idx}"
        if cam_name not in bpy.data.objects:
            continue
        bpy.context.scene.camera = bpy.data.objects[cam_name]
        out_path = os.path.join(trajectory_dir, f"seq_{seq_num:03d}_cam{cam_idx}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[trajectory] rendered {out_path}")


def exec_file(path: str, label: str) -> None:
    """Execute a Python file in the current Blender Python context."""
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        exec(compile(source, path, "exec"), {"__file__": path})
        print(f"[trajectory] executed {label}: OK")
    except Exception as exc:
        print(f"[trajectory] executed {label}: ERROR — {exc}")


def main() -> None:
    # Blender passes everything after '--' as extra argv
    argv = sys.argv
    try:
        sep_idx = argv.index("--")
    except ValueError:
        print("[trajectory] ERROR: no '--' separator found in argv")
        sys.exit(1)

    extra_args = argv[sep_idx + 1:]
    if not extra_args:
        print("[trajectory] ERROR: config path not provided after '--'")
        sys.exit(1)

    config_path = extra_args[0]
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    start_py    = cfg["start_py"]
    code_dir    = cfg["code_dir"]
    trajectory_dir = cfg["trajectory_dir"]

    os.makedirs(trajectory_dir, exist_ok=True)
    setup_rendering()

    # Initialise the scene to its starting state
    exec_file(start_py, "start.py")

    # Collect code files and sort by numeric seq
    code_files = [f for f in os.listdir(code_dir) if f.endswith(".py")]
    code_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    for fname in code_files:
        seq_num = int(os.path.splitext(fname)[0])
        code_path = os.path.join(code_dir, fname)
        exec_file(code_path, fname)
        render_all_cameras(trajectory_dir, seq_num)


main()
