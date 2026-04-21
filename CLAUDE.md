# Project Instructions

## BlenderMCP
### execute_blender_python_code
- When rendering images using blender python code, save all rendered images to `bench_data/{task_name}/temp/renders/` (create the directory if it doesn't exist)
- This can be done by setting the bpy.context.scene.render.filepath variable appropriately
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render