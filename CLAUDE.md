# Project Instructions

## BlenderMCP
### Blender Scene Edit benchmarking
- `bench_data/` holds immutable task definitions and is **read-only** — never write renders or any other output there.
- When rendering images using blender python code, save them to the session renders directory given in the task prompt (the absolute path under its "## Saving Renders" section). If the prompt does not provide one, fall back to `data/runs/{tool}__{model}/{task_name}/session/renders/` — where `{tool}__{model}` is the run id, the agent runner joined to the model slug, e.g. `claudecode__opus-4.8`. Create the directory if it doesn't exist.
- This can be done by setting the bpy.context.scene.render.filepath variable appropriately
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render

### Recreating 2D Images with Blender
- When rendering images using blender python code, save all rendered images to `recreation/{task_name}/{platform}/{model_dir}/process/` (create the directory if it doesn't exist)
- This can be done by setting the bpy.context.scene.render.filepath variable appropriately
- Do not write to `/tmp/` or other system directories
- Prioritizing rendering using blender python code rather than extracting viewport screenshot, as viewport screenshot is often vastly different from the actual render