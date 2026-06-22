"""Blender-side: deterministic replay of a logged run for validation.

Run by Blender, not imported:
    blender --background <copy of blender_file.blend> --python validate_script.py -- <config.json>

Replays one task from a logged session to check the run is authentic:
  1. exec start.py to put the scene in its starting state
  2. (optional) render the start state as seq0
  3. for each execute_blender_code step in tool_calls.jsonl (in seq order),
     exec the logged code, then render every camera

Each step's code is exec'd in a fresh namespace, matching how the BlenderMCP
addon runs each execute_blender_code call independently. Output images go to
out_dir as seq{seq}_camera{cam}.png, where seq is the log's sequence number.

Config JSON (passed after '--'):
  { start_py, tool_calls, out_dir, engine ("eevee"|"cycles"),
    samples, resolution, render_start }
"""

import json
import os
import sys

import bpy

RENDER_DEVICE_ENV = "BLENDERGYM_RENDER_DEVICE"


def eevee_engine_id() -> str:
    """The Eevee engine enum id for this Blender (renamed to *_NEXT in 4.2+)."""
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if cand in items:
            return cand
    return "BLENDER_EEVEE"


def configure_cycles_device() -> None:
    """Select the Cycles compute device, honoring BLENDERGYM_RENDER_DEVICE."""
    mode = os.environ.get(RENDER_DEVICE_ENV, "AUTO").upper()
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if mode == "CPU":
        prefs.compute_device_type = "NONE"
        bpy.context.scene.cycles.device = "CPU"
        return
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
    bpy.context.scene.cycles.device = "GPU" if prefs.compute_device_type != "NONE" else "CPU"


def configure_engine(engine: str, samples: int, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "PNG"
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        configure_cycles_device()
        scene.cycles.samples = samples
    else:
        scene.render.engine = eevee_engine_id()
        try:
            scene.eevee.taa_render_samples = samples
        except Exception:
            pass


def flush_shading() -> None:
    """Force pending material/lighting edits into the depsgraph before rendering.

    Replayed steps set values via the Python API, which Blender evaluates lazily;
    without this flush the first render after a shading edit can be stale.
    """
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            mat.node_tree.update_tag()
    for light in bpy.data.lights:
        light.update_tag()
    bpy.context.view_layer.update()


def render_all_cameras(out_dir: str, seq: int) -> list:
    flush_shading()
    scene = bpy.context.scene
    written = []
    for i in range(1, 6):
        cam_name = f"Camera{i}"
        if cam_name not in bpy.data.objects:
            continue
        scene.camera = bpy.data.objects[cam_name]
        out_path = os.path.join(out_dir, f"seq{seq}_camera{i}.png")
        scene.render.filepath = out_path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            print(f"VALIDATE_RENDER_FAIL seq{seq} {cam_name}: {exc}")
            continue
        if os.path.exists(out_path):
            written.append(out_path)
            print(f"VALIDATE_RENDER_OK seq{seq} {cam_name}")
    return written


def exec_step(code: str, label: str) -> bool:
    """Exec one logged code step in a fresh namespace (as BlenderMCP does)."""
    try:
        exec(compile(code, label, "exec"), {})
        print(f"VALIDATE_EXEC {label}: OK")
        return True
    except Exception as exc:
        print(f"VALIDATE_EXEC {label}: ERROR — {exc}")
        return False


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0]) as f:
        cfg = json.load(f)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    configure_engine(cfg["engine"], cfg["samples"], cfg["resolution"])

    # 1. start state
    with open(cfg["start_py"]) as f:
        exec(compile(f.read(), cfg["start_py"], "exec"), {})
    print("VALIDATE_EXEC start.py: OK")
    if cfg.get("render_start", True):
        render_all_cameras(out_dir, 0)

    # 2. replay logged code steps in seq order
    steps = []
    with open(cfg["tool_calls"]) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("tool") == "execute_blender_code":
                steps.append((rec.get("seq"), rec.get("params", {}).get("code", "")))
    steps.sort(key=lambda s: s[0])

    print(f"VALIDATE_PLAN {len(steps)} code steps, engine={cfg['engine']}")
    for seq, code in steps:
        exec_step(code, f"seq{seq}")
        render_all_cameras(out_dir, seq)

    print(f"VALIDATE_DONE {len(steps)} steps -> {out_dir}")


if __name__ == "__main__":
    main()
