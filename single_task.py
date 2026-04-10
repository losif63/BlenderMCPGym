from argparse import ArgumentParser
import os
import shutil
import socket
import subprocess
import time


BLENDERMCP_PORT = 9876
SYSTEM_PROMPT = (
    "Your task is to edit a Blender scene using BlenderMCP tools. "
    "You will be given rendered images of the current scene, a textual instruction "
    "describing the intended edit, and rendered images of the target scene. "
    "Use the BlenderMCP tools to edit the Blender scene according to the instruction "
    "so that the scene matches the target rendered images as closely as possible. "
    "When you are done, do not close Blender or stop the server."
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


def build_prompt(task_dir, instruction, start_renders, goal_renders):
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

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Starting Scene\n"
        f"The following are rendered images of the current Blender scene. Read each file before proceeding:\n"
        f"{start_paths}\n\n"
        f"## Instruction\n"
        f"{instruction}\n\n"
        f"## Target Scene\n"
        f"The following are rendered images of the target scene showing the desired result. Read each file before proceeding:\n"
        f"{goal_paths}"
    )


def main(args):
    task_dir = os.path.expanduser(f"~/Desktop/Research/BlenderMCPGym/bench_data/{args.task_name}")
    blend_file = f"{task_dir}/blender_file.blend"
    edit_file = f"{task_dir}/edit.blend"
    start_renders = f"{task_dir}/renders/start"
    goal_renders = f"{task_dir}/renders/goal"

    with open(f"{task_dir}/description.txt", "r") as f:
        content = f.read()
    ind = content.find("INSTRUCTION:")
    instruction = content[ind:].strip()

    prompt = build_prompt(task_dir, instruction, start_renders, goal_renders)

    shutil.copy2(blend_file, edit_file)
    blender_proc = subprocess.Popen([
        "/Applications/Blender.app/Contents/MacOS/blender",
        edit_file,
        "--python-expr",
        "import bpy; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ])

    print(f"Waiting for BlenderMCP server on port {BLENDERMCP_PORT}...")
    if not wait_for_blendermcp():
        print("ERROR: BlenderMCP server did not start within timeout.")
        blender_proc.terminate()
        return

    print("BlenderMCP server is ready. Launching Claude Code...")
    subprocess.run(
        ["claude", "-p", prompt, "--dangerously-skip-permissions"],
        cwd=os.path.expanduser("~/Desktop/Research/BlenderMCPGym")
    )

    print("Claude Code finished. Closing Blender...")
    blender_proc.terminate()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default="blendshape1")
    args = parser.parse_args()

    main(args)
