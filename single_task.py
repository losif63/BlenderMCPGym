from argparse import ArgumentParser
import json
import os
import shutil
import socket
import subprocess
import time


BLENDERMCP_PORT = 9876

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


def wait_for_blendermcp(port=BLENDERMCP_PORT, timeout=60, interval=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


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

    system_prompt = SYSTEM_PROMPT_V1 if version == 1 else SYSTEM_PROMPT_V2

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


def run_task(task_dir, port=BLENDERMCP_PORT, version=1):
    start_time = time.time()
    task_name = os.path.basename(task_dir)
    ver_tag = f"ver{version}"
    blend_file = f"{task_dir}/blender_file.blend"
    edit_file = f"{task_dir}/edit_{task_name}_{ver_tag}.blend"
    start_renders = f"{task_dir}/renders/start"
    goal_renders = f"{task_dir}/renders/goal"
    log_dir = f"{task_dir}/{ver_tag}"

    with open(f"{task_dir}/detailed_instruction.txt", "r") as f:
        content = f.read()
    ind = content.find("INSTRUCTION:")
    instruction = content[ind:].strip()

    start_script = f"{task_dir}/start.py"
    start_code = None
    if version == 1:
        with open(start_script, "r") as f:
            start_code = f.read()

    prompt = build_prompt(instruction, start_renders, goal_renders, version=version, start_code=start_code)

    shutil.copy2(blend_file, edit_file)
    blender_proc = subprocess.Popen([
        "/Applications/Blender.app/Contents/MacOS/blender",
        edit_file,
        "--python", start_script,
        "--python-expr",
        f"import bpy; bpy.context.scene.blendermcp_port = {port}; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ])

    print(f"[{task_dir}] Waiting for BlenderMCP server on port {port}...")
    if not wait_for_blendermcp(port=port):
        print(f"[{task_dir}] ERROR: BlenderMCP server did not start within timeout.")
        blender_proc.terminate()
        return

    os.makedirs(log_dir, exist_ok=True)
    print(f"[{task_dir}] BlenderMCP server is ready. Launching Claude Code (version {version})...")
    subprocess.run(
        ["claude", "-p", prompt, "--dangerously-skip-permissions"],
        cwd=os.path.expanduser("~/Desktop/Research/BlenderMCPGym"),
        env={**os.environ, "BLENDER_MCP_LOG_DIR": log_dir},
    )

    print(f"[{task_dir}] Claude Code finished. Saving Blender file...")
    save_blender_file(port=port)
    print(f"[{task_dir}] Saved. Closing Blender...")
    blender_proc.terminate()

    edit_renders_dir = f"{task_dir}/renders/edit_{ver_tag}"
    os.makedirs(edit_renders_dir, exist_ok=True)
    print(f"[{task_dir}] Rendering edited scene...")
    edit_render_script = os.path.expanduser("~/Desktop/Research/BlenderMCPGym/bench_data/edit_render_script.py")
    subprocess.run([
        "/Applications/Blender.app/Contents/MacOS/blender",
        "--background", edit_file,
        "--python", edit_render_script,
        "--", edit_renders_dir,
    ], check=True)
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
    task_dir = os.path.expanduser(f"~/Desktop/Research/BlenderMCPGym/bench_data/{args.task_name}")
    run_task(task_dir, version=args.version)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default="blendshape1")
    parser.add_argument(
        '--version', type=int, default=1, choices=[1, 2],
        help=(
            "Experiment version. "
            "1 (default): provides start.py as context and instructs the agent to use only "
            "its variables and methods. "
            "2: no start.py context; agent is free to use any approach."
        )
    )
    args = parser.parse_args()

    main(args)
