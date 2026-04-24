# Project Instructions

## BlenderMCP
### execute_blender_python_code
- When rendering images using blender python code, save all rendered images to the directory given by the `BLENDER_RENDER_DIR` environment variable (read it with `os.environ["BLENDER_RENDER_DIR"]`). This directory already exists — do not recreate it.
- This can be done by setting the `bpy.context.scene.render.filepath` variable to a path inside `BLENDER_RENDER_DIR`, e.g. `os.path.join(os.environ["BLENDER_RENDER_DIR"], "render.png")`
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render