import os
import subprocess
import platform
import time
import socket
import json

SYSTEM_PROMPT_BEGINNER = """
You are an expert 3D artist working in Blender through the BlenderMCP interface.
Your task is to model the object in the provided reference image as a 3D object in Blender, as faithfully as the available tools allow.
In the blender scene, create a single camera only. Position and orient the camera so that the resulting render reproduces the viewpoint of the input image as closely as possible..
"""

SYSTEM_PROMPT_ADVANCED = """
You are an expert 3D artist working in Blender through the BlenderMCP interface.
You are provided a reference image. This reference image was created and rendered entirely by blender.
Your task is to reconstruct the scene in the provided reference image as a 3D scene in Blender, as faithfully as the available tools allow.
In the blender scene, create a single camera only. Position and orient the camera so that the resulting render reproduces the viewpoint of the input image as closely as possible..
"""

PROJECT_DIR = os.getcwd()

SYSTEM_PROMPTS = {
    "beginner": SYSTEM_PROMPT_BEGINNER,
    "advanced": SYSTEM_PROMPT_ADVANCED,
}

def infer_prompt_type(image_name):
    stem = os.path.splitext(image_name)[0]
    if stem.startswith("beginner"):
        return "beginner"
    if stem.startswith("advanced"):
        return "advanced"
    return None

def build_prompt(image_name, plat, model_name, prompt_type=None):
    if prompt_type is None:
        prompt_type = infer_prompt_type(image_name)
    if prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'. Must be one of: {list(SYSTEM_PROMPTS)}")
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
    f"Model name: {model_name}"
    )
    return prompt

def get_blender_executable():
    """Return the Blender executable path.

    Resolution order:
      1. BLENDER_PATH environment variable (any OS)
      2. Platform default: './infinigen/blender/blender' on Linux,
         the standard .app path on macOS
    """
    if "BLENDER_PATH" in os.environ:
        return os.environ["BLENDER_PATH"]
    project_root = os.getcwd()
    if platform.system() == "Darwin":
        return os.path.join(project_root, "infinigen", "Blender.app", "Contents", "MacOS", "Blender")
    return "./infinigen/blender/blender"

def get_blender_env(virtual_display=False):
    """Return the environment dict for launching Blender.

    If virtual_display is True, sets DISPLAY=:99 for use with a headless
    virtual display (e.g. Xvfb running on :99).
    """
    env = os.environ.copy()
    if virtual_display:
        env["DISPLAY"] = ":99"
    return env

def wait_for_blendermcp(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("localhost", 9876))
            return True
        except OSError:
            time.sleep(1.0)
    return False

def save_blender_file(save_path):
    command = json.dumps({"type": "execute_code", "params": {"code": f"bpy.ops.wm.save_mainfile(filepath='{save_path}', check_existing=False)"}})
    with socket.create_connection(("localhost", 9876), timeout=10) as s:
        s.sendall(command.encode("utf-8"))
        response = json.loads(s.recv(8192).decode("utf-8"))
    if response.get("status") != "success":
        print(f"WARNING: save_mainfile returned unexpected response: {response}")


def setup_blendermcp(virtual_display = False):
    blender_proc = subprocess.Popen(
    args=[
        get_blender_executable(),
        "--python-expr",
        f"import bpy; bpy.context.scene.blendermcp_port = 9876; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ],
    env=get_blender_env(virtual_display)
    )
    if not wait_for_blendermcp():
        print("Error - Blendermcp did not start within timeout limit.")
        blender_proc.terminate()
        return None
    
    return blender_proc
    

    
    