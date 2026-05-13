# Project Instructions

## BlenderMCP
### Blender Scene Edit benchmarking
- When rendering images using blender python code, save all rendered images to `bench_data/{task_name}/ver{version}/{session_name}/temp/renders/` (create the directory if it doesn't exist)
- This can be done by setting the bpy.context.scene.render.filepath variable appropriately
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render

### Recreating 2D Images with Blender
- When rendering images using blender python code, save all rendered images to `recreation/{image_name}/{platform}/{model_name}/process/` (create the directory if it doesn't exist)
- This can be done by setting the bpy.context.scene.render.filepath variable appropriately
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render