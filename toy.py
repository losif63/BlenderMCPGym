import os
import socket
import subprocess
import time


BLENDERMCP_PORT = 9876
PROMPT = "Use the BlenderMCP tools to build a detailed castle in the Blender scene."


def wait_for_blendermcp(port=BLENDERMCP_PORT, timeout=60, interval=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


def main():
    subprocess.Popen([
        "/Applications/Blender.app/Contents/MacOS/blender",
        "--python-expr",
        "import bpy; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ])

    print(f"Waiting for BlenderMCP server on port {BLENDERMCP_PORT}...")
    if not wait_for_blendermcp():
        print("ERROR: BlenderMCP server did not start within timeout.")
        return

    print("BlenderMCP server is ready. Launching Claude Code...")
    subprocess.run(
        ["claude", "-p", PROMPT, "--dangerously-skip-permissions"],
        cwd=os.path.expanduser("~/Desktop/Research/BlenderMCPGym")
    )


if __name__ == '__main__':
    main()
