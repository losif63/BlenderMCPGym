import bpy
import os
import sys

if __name__ == "__main__":
    rendering_dir = sys.argv[sys.argv.index("--") + 1]
    os.makedirs(rendering_dir, exist_ok=True)

    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
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

    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    for cam_name in ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5']:
        if cam_name in bpy.data.objects:
            bpy.context.scene.camera = bpy.data.objects[cam_name]
            idx = cam_name[-1]
            bpy.context.scene.render.filepath = os.path.join(rendering_dir, f'render{idx}.png')
            bpy.ops.render.render(write_still=True)
