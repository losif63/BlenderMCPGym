"""Render the edited scene from each named camera to render{N}.png.

Two entry points, same logic:
  - Standalone (fresh reopen): blender --background edit.blend --python
    edit_render_script.py -- <out_dir>
  - In-session: ``render_all_cameras`` is imported and called inside the live
    agent Blender (see blendergym.run.blender.render_in_session) so that edits
    which only exist in memory until the scene is rendered — lighting and
    material changes especially — are captured. A fresh reopen renders the
    saved .blend, which can omit those, so the live-session path is preferred.
"""

import bpy
import os
import sys

# Host-level render device, shared with blendergym.run.blender.RENDER_DEVICE_ENV.
# Kept as a literal here because this script is run by Blender, not imported.
RENDER_DEVICE_ENV = "BLENDERGYM_RENDER_DEVICE"


def configure_cycles_device():
    """Select the Cycles compute device per the BLENDERGYM_RENDER_DEVICE env var.

    CPU forces CPU and disables GPU compute entirely. GPU/AUTO prefer Metal/CUDA/
    OPTIX where available and fall back to CPU when no GPU backend is present.
    """
    bpy.context.scene.render.engine = 'CYCLES'
    mode = os.environ.get(RENDER_DEVICE_ENV, "AUTO").upper()
    prefs = bpy.context.preferences.addons['cycles'].preferences

    if mode == "CPU":
        prefs.compute_device_type = 'NONE'
        bpy.context.scene.cycles.device = 'CPU'
        return

    for device_type in ('METAL', 'CUDA', 'OPTIX', 'NONE'):
        try:
            prefs.compute_device_type = device_type
            break
        except TypeError:
            continue
    prefs.get_devices()
    for device in prefs.devices:
        if device.type != 'CPU':
            device.use = True
    bpy.context.scene.cycles.device = 'GPU' if prefs.compute_device_type != 'NONE' else 'CPU'


def render_all_cameras(rendering_dir):
    """Render render{N}.png from each existing Camera1..Camera5 into rendering_dir.

    Assumes the active scene already holds the edits to render. Callers are
    responsible for selecting the device first (configure_cycles_device()).
    """
    os.makedirs(rendering_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'

    for cam_name in ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5']:
        if cam_name in bpy.data.objects:
            scene.camera = bpy.data.objects[cam_name]
            idx = cam_name[-1]
            scene.render.filepath = os.path.join(rendering_dir, f'render{idx}.png')
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    rendering_dir = sys.argv[sys.argv.index("--") + 1]
    configure_cycles_device()
    render_all_cameras(rendering_dir)
