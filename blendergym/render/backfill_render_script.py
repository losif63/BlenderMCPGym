"""Backfill missing edit renders by reopening a saved .blend.

Standalone Blender script:

    blender --background <edit.blend> --python backfill_render_script.py \
        -- <out_dir> [--overwrite]

Reopening the saved file is faithful for geometry / blendshape / placement edits
(which persist through save), so this fills in renders a partial in-session pass
left behind, without re-running the agent. It is deliberately NOT used for
lighting / material, whose edits can be dropped on reopen — those must be
rendered in the live session (see blendergym.run.blender.render_in_session).

Renders one image per camera *currently in the scene* into out_dir, using the
``Camera{N} -> render{N}.png`` convention so the views line up with the goal
renders for evaluation; any other-named camera is written to a sanitized
``render_{name}.png``. A camera whose render already exists is skipped (so this
only fills gaps) unless ``--overwrite`` is passed.
"""

import importlib.util
import os
import re
import sys

import bpy

# Reuse the device-selection logic from the in-session render script next door
# rather than duplicating the GPU/CPU handling. ``__name__`` is the loader name,
# not "__main__", so that script's CLI block does not fire on import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "blendergym_edit_render", os.path.join(_HERE, "edit_render_script.py"))
_ers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ers)
configure_cycles_device = _ers.configure_cycles_device


def render_filename(cam_name):
    """render{N}.png for a ``Camera{N}`` camera; render_{name}.png otherwise.

    The numbered form matches the goal renders so evaluation pairs the views;
    the sanitized fallback keeps any extra camera from colliding with them.
    """
    m = re.fullmatch(r'Camera(\d+)', cam_name)
    if m:
        return f'render{m.group(1)}.png'
    return f'render_{re.sub(r"[^A-Za-z0-9_-]+", "_", cam_name)}.png'


def _camera_sort_key(obj):
    # Numbered cameras first, in numeric order (Camera2 before Camera10); other
    # cameras after, by name. Fixed-length tuples so the keys stay comparable.
    m = re.fullmatch(r'Camera(\d+)', obj.name)
    return (0, int(m.group(1)), '') if m else (1, 0, obj.name)


def render_missing_cameras(rendering_dir, overwrite=False):
    """Render every camera in the active scene into rendering_dir.

    Skips a camera whose target file already exists unless ``overwrite`` is set,
    so repeated runs only fill gaps. Each camera is rendered independently: a
    failure is reported and the loop continues. Returns (written, skipped).
    """
    os.makedirs(rendering_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'

    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']
    if not cameras:
        print("WARNING: no cameras in scene; nothing to backfill")
        return [], []

    written, skipped = [], []
    for obj in sorted(cameras, key=_camera_sort_key):
        fname = render_filename(obj.name)
        out_path = os.path.join(rendering_dir, fname)
        if os.path.exists(out_path) and not overwrite:
            skipped.append(fname)
            print(f"SKIP {obj.name}: {fname} already exists")
            continue
        scene.camera = obj
        scene.render.filepath = out_path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:  # report and move on to the next camera
            print(f"RENDER_FAIL {obj.name}: {exc}")
            continue
        if os.path.exists(out_path):
            written.append(fname)
            print(f"RENDER_OK {obj.name} -> {fname}")
        else:
            print(f"RENDER_MISSING {obj.name}: no file written to {out_path}")

    print(f"BACKFILL_SUMMARY written={written} skipped={skipped}")
    return written, skipped


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    if not argv:
        raise SystemExit("usage: ... -- <out_dir> [--overwrite]")
    out_dir = argv[0]
    overwrite = "--overwrite" in argv[1:]
    configure_cycles_device()
    render_missing_cameras(out_dir, overwrite=overwrite)
