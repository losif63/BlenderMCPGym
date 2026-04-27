from argparse import ArgumentParser
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time


BLENDERMCP_PORT = 9876
MAX_TOOL_CALLS = 100


def get_blender_executable():
    """Return the Blender executable path.

    Resolution order:
      1. BLENDER_PATH environment variable (any OS)
      2. Platform default: './infinigen/blender/blender' on Linux,
         the standard .app path on macOS
    """
    if "BLENDER_PATH" in os.environ:
        return os.environ["BLENDER_PATH"]
    if platform.system() == "Darwin":
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "infinigen", "Blender.app", "Contents", "MacOS", "Blender")
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

# Version 1: agent is given start.py and told to use only its variables/methods.
SYSTEM_PROMPT_V1 = (
    "Your task is to edit a Blender scene using BlenderMCP tools. "
    "You will be given rendered images of the current scene, a textual instruction "
    "describing the intended edit, rendered images of the target scene, and the Blender "
    "Python initialization script (start.py) that was used to set up the scene. "
    "Use the BlenderMCP tools to edit the Blender scene according to the instruction "
    "so that the scene matches the target rendered images as closely as possible. "
    "IMPORTANT: Refer to the provided start.py code to understand what variables, objects, "
    "and methods are already defined in the scene. Make your edits by building on those "
    "existing variables and methods only — do not create new objects or use approaches "
    "that are inconsistent with how the scene was initialized. "
    "When you are done with all edits, stop. Do not close Blender or stop the server."
)

# Version 2: agent receives no start.py context and is free to use any approach.
SYSTEM_PROMPT_V2 = (
    "Your task is to edit a Blender scene using BlenderMCP tools. "
    "You will be given rendered images of the current scene, a textual instruction "
    "describing the intended edit, and rendered images of the target scene. "
    "Use the BlenderMCP tools to edit the Blender scene according to the instruction "
    "so that the scene matches the target rendered images as closely as possible. "
    "When you are done with all edits, stop. Do not close Blender or stop the server."
)

# Version 3: agent receives start.py and rendered images but no textual instruction.
SYSTEM_PROMPT_V3 = (
    "Your task is to edit a Blender scene using BlenderMCP tools. "
    "You will be given the Blender Python initialization script (start.py) used to set up "
    "the scene, rendered images of the current scene, and rendered images of the target scene. "
    "Study the start.py code to understand the scene's objects, variables, and methods, then "
    "use the BlenderMCP tools to modify the scene so that its rendered result matches the "
    "target images as closely as possible. "
    "IMPORTANT: Make your edits by building on the existing variables and methods defined in "
    "start.py only — do not create new objects or use approaches inconsistent with how the "
    "scene was initialized. "
    "When you are done with all edits, stop. Do not close Blender or stop the server."
)


def wait_for_blendermcp(port=BLENDERMCP_PORT, timeout=60, interval=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


def get_session_dirs(log_dir):
    """Return the set of session subdirectory names currently under log_dir."""
    try:
        return {
            d for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d))
        }
    except OSError:
        return set()


def count_tool_calls_in_session(session_dir):
    """Count lines in the tool_calls.jsonl file of a single session directory."""
    path = os.path.join(session_dir, "tool_calls.jsonl")
    try:
        with open(path, "r") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def monitor_tool_calls(log_dir, existing_sessions, proc, max_calls, check_interval=2.0):
    """
    Wait for a new session directory to appear under log_dir (one not in
    existing_sessions), then watch only that session's tool_calls.jsonl and
    terminate proc once the line count reaches max_calls.
    """
    # Wait for the new session dir to be created by Claude Code
    new_session_dir = None
    while proc.poll() is None and new_session_dir is None:
        current = get_session_dirs(log_dir)
        new_dirs = current - existing_sessions
        if new_dirs:
            new_session_dir = os.path.join(log_dir, sorted(new_dirs)[-1])
            print(f"  [watchdog] Monitoring session: {new_session_dir}")
        time.sleep(check_interval)

    if new_session_dir is None:
        return  # proc already finished before a new session appeared

    while proc.poll() is None:
        if count_tool_calls_in_session(new_session_dir) >= max_calls:
            print(f"  [watchdog] Tool call limit ({max_calls}) reached — terminating claude.")
            proc.terminate()
            return
        time.sleep(check_interval)


def save_blender_file(port=BLENDERMCP_PORT):
    command = json.dumps({"type": "execute_code", "params": {"code": "bpy.ops.wm.save_mainfile()"}})
    with socket.create_connection(("localhost", port), timeout=10) as s:
        s.sendall(command.encode("utf-8"))
        response = json.loads(s.recv(8192).decode("utf-8"))
    if response.get("status") != "success":
        print(f"WARNING: save_mainfile returned unexpected response: {response}")


def build_prompt(instruction, start_renders, goal_renders, version=1, start_code=None):
    start_images = sorted([
        os.path.join(start_renders, f)
        for f in os.listdir(start_renders)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    goal_images = sorted([
        os.path.join(goal_renders, f)
        for f in os.listdir(goal_renders)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    start_paths = "\n".join(f"  - {p}" for p in start_images)
    goal_paths = "\n".join(f"  - {p}" for p in goal_images)

    if version == 1:
        system_prompt = SYSTEM_PROMPT_V1
    elif version == 3:
        system_prompt = SYSTEM_PROMPT_V3
    else:
        system_prompt = SYSTEM_PROMPT_V2

    if version == 3:
        prompt = (
            f"{system_prompt}\n\n"
            f"## Initialization Script (start.py)\n"
            f"The following Blender Python script was used to initialize the scene. "
            f"Use only the variables, objects, and methods defined here when making your edits:\n"
            f"```python\n{start_code}\n```\n\n"
            f"## Starting Scene\n"
            f"The following are rendered images of the current Blender scene. Read each file before proceeding:\n"
            f"{start_paths}\n\n"
            f"## Target Scene\n"
            f"The following are rendered images of the target scene showing the desired result. Read each file before proceeding:\n"
            f"{goal_paths}"
        )
    else:
        prompt = (
            f"{system_prompt}\n\n"
            f"## Starting Scene\n"
            f"The following are rendered images of the current Blender scene. Read each file before proceeding:\n"
            f"{start_paths}\n\n"
            f"## Instruction\n"
            f"{instruction}\n\n"
            f"## Target Scene\n"
            f"The following are rendered images of the target scene showing the desired result. Read each file before proceeding:\n"
            f"{goal_paths}"
        )

        if version == 1 and start_code is not None:
            prompt += (
                f"\n\n## Initialization Script (start.py)\n"
                f"The following Blender Python script was used to initialize the scene. "
                f"Use only the variables, objects, and methods defined here when making your edits:\n"
                f"```python\n{start_code}\n```"
            )

    return prompt


def run_task(task_dir, port=BLENDERMCP_PORT, version=1, virtual_display=False):
    start_time = time.time()
    task_name = os.path.basename(task_dir)
    ver_tag = f"ver{version}"
    blend_file = f"{task_dir}/blender_file.blend"
    edit_file = f"{task_dir}/edit_{task_name}_{ver_tag}.blend"
    start_renders = f"{task_dir}/renders/start"
    goal_renders = f"{task_dir}/renders/goal"
    log_dir = f"{task_dir}/{ver_tag}"

    instruction = None
    if version in (1, 2):
        with open(f"{task_dir}/detailed_instruction.txt", "r") as f:
            content = f.read()
        ind = content.find("INSTRUCTION:")
        instruction = content[ind:].strip()

    start_script = f"{task_dir}/start.py"
    start_code = None
    if version in (1, 3):
        with open(start_script, "r") as f:
            start_code = f.read()

    prompt = build_prompt(instruction, start_renders, goal_renders, version=version, start_code=start_code)

    shutil.copy2(blend_file, edit_file)
    blender_proc = subprocess.Popen([
        get_blender_executable(),
        edit_file,
        "--python", start_script,
        "--python-expr",
        f"import bpy; bpy.context.scene.blendermcp_port = {port}; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ], env=get_blender_env(virtual_display))

    print(f"[{task_dir}] Waiting for BlenderMCP server on port {port}...")
    if not wait_for_blendermcp(port=port):
        print(f"[{task_dir}] ERROR: BlenderMCP server did not start within timeout.")
        blender_proc.terminate()
        return

    os.makedirs(log_dir, exist_ok=True)
    existing_sessions = get_session_dirs(log_dir)
    print(f"[{task_dir}] BlenderMCP server is ready. Launching Claude Code (version {version}, limit {MAX_TOOL_CALLS} tool calls)...")
    claude_proc = subprocess.Popen(
        ["claude", "-p", prompt, "--dangerously-skip-permissions"],
        cwd=os.getcwd(),
        env={**os.environ, "BLENDER_MCP_LOG_DIR": log_dir},
    )
    watchdog = threading.Thread(
        target=monitor_tool_calls,
        args=(log_dir, existing_sessions, claude_proc, MAX_TOOL_CALLS),
        daemon=True,
    )
    watchdog.start()
    claude_proc.wait()
    watchdog.join(timeout=5)

    print(f"[{task_dir}] Claude Code finished. Saving Blender file...")
    save_blender_file(port=port)
    print(f"[{task_dir}] Saved. Closing Blender...")
    blender_proc.terminate()

    edit_renders_dir = f"{task_dir}/renders/edit_{ver_tag}"
    os.makedirs(edit_renders_dir, exist_ok=True)
    print(f"[{task_dir}] Rendering edited scene...")
    edit_render_script = f"{os.getcwd()}/bench_data/edit_render_script.py"
    subprocess.run([
        get_blender_executable(),
        "--background", edit_file,
        "--python", edit_render_script,
        "--", edit_renders_dir,
    ], env=get_blender_env(virtual_display), check=True)
    print(f"[{task_dir}] Renders saved to {edit_renders_dir}")

    metadata_path = f"{task_dir}/metadata_{ver_tag}.json"
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    metadata["version"] = version
    metadata["duration_seconds"] = round(time.time() - start_time, 2)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[{task_dir}] Logged duration: {metadata['duration_seconds']}s")


def main(args):
    task_dir = f"{os.getcwd()}/bench_data/{args.task_name}"
    run_task(task_dir, version=args.version, virtual_display=args.virtual)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default="blendshape1")
    parser.add_argument(
        '--virtual', action='store_true',
        help="Set DISPLAY=:99 when launching Blender (for headless servers with a virtual display)."
    )
    parser.add_argument(
        '--version', type=int, default=1, choices=[1, 2, 3],
        help=(
            "Experiment version. "
            "1 (default): provides textual instruction, rendered images, and start.py. "
            "2: provides textual instruction and rendered images only (no start.py). "
            "3: provides start.py and rendered images only (no textual instruction)."
        )
    )
    args = parser.parse_args()

    main(args)
