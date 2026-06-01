"""
Helpers for the v2 image-recreation experiment.

Differences vs `util.py` (v1):
  * `setup_blendermcp` enables Polyhaven asset downloads in BlenderMCP.
  * `build_prompt` uses prompts that (a) tell the agent it can use Polyhaven, and
    (b) tell it to call the `mcp__render_feedback__score_render` tool after each
    render to receive DINOv3 cosine-distance feedback against the reference image.

Everything else (Blender executable resolution, MCP socket helpers, prompt-type
inference) is reused from `util.py` to avoid drift.
"""

import os
import subprocess

from util import (
    get_blender_executable,
    get_blender_env,
    infer_prompt_type,
    save_blender_file,
    wait_for_blendermcp,
)

SYSTEM_PROMPT_BEGINNER = """
You are an expert 3D artist working in Blender through the BlenderMCP interface.
Your task is to model the object in the provided reference image as a 3D object in Blender, as faithfully as the available tools allow.
In the blender scene, create a single camera only. Position and orient the camera so that the resulting render reproduces the viewpoint of the input image as closely as possible.

You may use free 3D assets from PolyHaven when they help you faithfully reproduce the reference. PolyHaven access is already enabled in BlenderMCP — search assets with `mcp__blender__search_polyhaven_assets`, list categories with `mcp__blender__get_polyhaven_categories`, and bring an asset into the current scene with `mcp__blender__download_polyhaven_asset`. You can call `mcp__blender__get_polyhaven_status` first to confirm it is enabled.
"""

SYSTEM_PROMPT_ADVANCED = """
You are an expert 3D artist working in Blender through the BlenderMCP interface.
You are provided a reference image. This reference image was created and rendered entirely by blender.
Your task is to reconstruct the scene in the provided reference image as a 3D scene in Blender, as faithfully as the available tools allow.
In the blender scene, create a single camera only. Position and orient the camera so that the resulting render reproduces the viewpoint of the input image as closely as possible.

You may use free 3D assets from PolyHaven when they help you faithfully reproduce the reference. PolyHaven access is already enabled in BlenderMCP — search assets with `mcp__blender__search_polyhaven_assets`, list categories with `mcp__blender__get_polyhaven_categories`, and bring an asset into the current scene with `mcp__blender__download_polyhaven_asset`. You can call `mcp__blender__get_polyhaven_status` first to confirm it is enabled.
"""

SCORING_PROTOCOL = """
## Scoring protocol (MANDATORY)

After every PNG you render to the process/ directory, your VERY NEXT tool call MUST be `mcp__render_feedback__score_render` with the absolute path of that PNG as the `render_path` argument. The tool is already registered and available in your tool palette — you do not need to discover it or set anything up.

The tool returns the cosine distance between your render and the reference image, computed with the DINOv3 model. Lower is closer to the reference; 0.0 means identical embeddings. This is the exact metric your final output is evaluated by, so treat each returned score as objective signal for whether your last edit moved the scene closer to the target.

Hard rules — do not violate these:
  - Do NOT write your own Python script to compute the distance. The dependencies are not available in your shell environment.
  - Do NOT import anything from `feedback_mcp_server.py`. It is an MCP server, not a library.
  - Do NOT search the filesystem (`find`, `ls`, `grep`) for "scoring utilities". The tool you need is `mcp__render_feedback__score_render` — call it directly.
  - Do NOT skip scoring just because the render "looks fine". Render then score, every time.

Example sequence for one iteration:
  1. `mcp__blender__execute_blender_code` — apply edits and render to `…/process/iter_N.png`
  2. `mcp__render_feedback__score_render` with `render_path="…/process/iter_N.png"` — get the DINOv3 distance
  3. Use the returned distance to decide your next edit
"""

PROJECT_DIR = os.getcwd()

SYSTEM_PROMPTS = {
    "beginner": SYSTEM_PROMPT_BEGINNER,
    "advanced": SYSTEM_PROMPT_ADVANCED,
}


def build_prompt(image_name, plat, model_name, prompt_type=None):
    if prompt_type is None:
        prompt_type = infer_prompt_type(image_name)
    if prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown prompt_type '{prompt_type}'. Must be one of: {list(SYSTEM_PROMPTS)}"
        )
    system_prompt = SYSTEM_PROMPTS[prompt_type]
    image_stem = os.path.splitext(image_name)[0]
    prompt = (
        f"{system_prompt}\n\n"
        "## Reference Image\n"
        "The following is the path to the reference image. Read the file before proceeding:\n"
        f"{PROJECT_DIR}/images/{image_name}\n\n"
        f"As mentioned in CLAUDE.md, save all rendered images under recreation/{image_stem}/{plat}/{model_name}/process/.\n"
        "The platform and model names are as follows:\n"
        f"Platform: {plat}\n"
        f"Model name: {model_name}\n"
        f"{SCORING_PROTOCOL}"
    )
    return prompt


def setup_blendermcp(virtual_display=False, use_polyhaven=True):
    """Launch Blender with the BlenderMCP server listening on port 9876.

    If `use_polyhaven` is True, also enables the PolyHaven asset toggle on the
    scene before the server starts so the polyhaven MCP tools become usable.
    """
    polyhaven_expr = (
        "bpy.context.scene.blendermcp_use_polyhaven = True; " if use_polyhaven else ""
    )
    python_expr = (
        "import bpy; "
        "bpy.context.scene.blendermcp_port = 9876; "
        f"{polyhaven_expr}"
        "bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    )
    blender_proc = subprocess.Popen(
        args=[get_blender_executable(), "--python-expr", python_expr],
        env=get_blender_env(virtual_display),
    )
    if not wait_for_blendermcp():
        print("Error - Blendermcp did not start within timeout limit.")
        blender_proc.terminate()
        return None
    return blender_proc


__all__ = [
    "SYSTEM_PROMPTS",
    "build_prompt",
    "infer_prompt_type",
    "save_blender_file",
    "setup_blendermcp",
]
