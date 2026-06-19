"""Blender process helpers shared by the run orchestration."""

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

from ..paths import REPO_ROOT

BLENDERMCP_PORT = 9876

# Host-level render device selection. This is a property of the machine, not of
# a benchmark run, so it lives in the environment rather than in RunConfig.
#   CPU   -> force CPU; disables Metal/CUDA entirely so the agent cannot re-enable
#            the GPU (use on GPU-limited machines, e.g. Apple M1 where Metal can
#            destabilize the system).
#   GPU   -> prefer the GPU (Metal/CUDA/OPTIX) where available.
#   AUTO  -> default; leave Blender/.blend defaults untouched in the live session.
RENDER_DEVICE_ENV = "BLENDERGYM_RENDER_DEVICE"


def render_device_mode() -> str:
    """Resolve the configured render device: 'CPU', 'GPU', or 'AUTO' (default)."""
    mode = os.environ.get(RENDER_DEVICE_ENV, "AUTO").upper()
    return mode if mode in ("CPU", "GPU", "AUTO") else "AUTO"


def render_device_setup_code() -> str:
    """Blender-Python snippet that pins the render device for the live session.

    Returned as source so it can be prepended to ``--python-expr``. When the mode
    is CPU it disables GPU compute at the *preferences* level: even if the agent
    later sets ``scene.cycles.device = 'GPU'`` there are no enabled GPU devices,
    so Cycles falls back to CPU. AUTO/GPU leave the launch untouched (the agent
    and the final render script configure the device themselves).
    """
    if render_device_mode() != "CPU":
        return ""
    return (
        "_p = bpy.context.preferences.addons['cycles'].preferences; "
        "_p.compute_device_type = 'NONE'; "
        "[setattr(_s.cycles, 'device', 'CPU') for _s in bpy.data.scenes]; "
    )


def get_blender_executable() -> str:
    """Resolve the Blender executable.

    1. BLENDER_PATH env var (any OS)
    2. Platform default: bundled infinigen Blender.
    """
    if "BLENDER_PATH" in os.environ:
        return os.environ["BLENDER_PATH"]
    if platform.system() == "Darwin":
        return str(REPO_ROOT / "infinigen" / "Blender.app" / "Contents" / "MacOS" / "Blender")
    return "./infinigen/blender/blender"


def get_blender_env(virtual_display: bool = False) -> dict:
    """Environment for launching Blender; DISPLAY=:99 for headless Xvfb."""
    env = os.environ.copy()
    if virtual_display:
        env["DISPLAY"] = ":99"
    return env


def wait_for_blendermcp(port: int = BLENDERMCP_PORT, timeout: float = 60, interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


def _port_in_use(port: int) -> bool:
    """True if something is accepting connections on the port (i.e. not free)."""
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


def terminate_blender(proc, port: int = BLENDERMCP_PORT, grace: float = 10.0,
                      kill_timeout: float = 10.0, port_free_timeout: float = 10.0) -> None:
    """Terminate a launched Blender subprocess and confirm it is really gone.

    ``proc.terminate()`` only sends SIGTERM and returns immediately; a Blender
    that is mid-render or hung can ignore it and linger, leaving its BlenderMCP
    server bound to ``port``. The next task then launches its own Blender (which
    fails to bind the port) while ``wait_for_blendermcp`` connects to the stale
    server — and leaked Blenders pile up. This escalates SIGTERM -> SIGKILL,
    waits for the process to actually exit, then confirms the port is released
    before returning so the next task starts from a clean slate.
    """
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            print(f"Blender (pid {proc.pid}) ignored SIGTERM after {grace:.0f}s; sending SIGKILL.")
            proc.kill()
            try:
                proc.wait(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                print(f"WARNING: Blender (pid {proc.pid}) still alive after SIGKILL.")
                return

    # Process is gone; confirm the MCP port is released before the next task binds it.
    deadline = time.time() + port_free_timeout
    while time.time() < deadline:
        if not _port_in_use(port):
            return
        time.sleep(0.5)
    print(f"WARNING: port {port} still in use after terminating Blender (pid {proc.pid}); "
          f"a stale Blender from an earlier run may still be holding it.")


def save_blender_file(port: int = BLENDERMCP_PORT) -> None:
    command = json.dumps({"type": "execute_code", "params": {"code": "bpy.ops.wm.save_mainfile()"}})
    with socket.create_connection(("localhost", port), timeout=10) as s:
        s.sendall(command.encode("utf-8"))
        response = json.loads(s.recv(8192).decode("utf-8"))
    if response.get("status") != "success":
        print(f"WARNING: save_mainfile returned unexpected response: {response}")


def edit_render_script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "render" / "edit_render_script.py"


def render_in_session(rendering_dir, port: int = BLENDERMCP_PORT, timeout: float = 900.0) -> None:
    """Render the edited scene *inside the live agent Blender*, over the MCP socket.

    Rendering in the running session (rather than reopening the saved .blend in a
    fresh process) is required for edits that only materialize in memory until the
    scene is rendered — lighting and material changes in particular. A reopen would
    render the saved file, which can omit those, yielding meaningless renders.

    The render logic is the same code path as the standalone script: we load
    edit_render_script.py by file path inside Blender (bypassing the package
    __init__, which Blender's bundled Python can't necessarily import) and call its
    configure_cycles_device() + render_all_cameras(). ``timeout`` bounds the socket
    read, which must cover the full multi-camera Cycles render at 512 samples.
    """
    script_path = str(edit_render_script_path())
    out_dir = str(rendering_dir)
    code = (
        "import importlib.util as _ilu\n"
        f"_spec = _ilu.spec_from_file_location('blendergym_edit_render', {script_path!r})\n"
        "_mod = _ilu.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        "_mod.configure_cycles_device()\n"
        f"_mod.render_all_cameras({out_dir!r})\n"
    )
    command = json.dumps({"type": "execute_code", "params": {"code": code}})

    response = None
    with socket.create_connection(("localhost", port), timeout=10) as s:
        s.settimeout(timeout)
        s.sendall(command.encode("utf-8"))
        buf = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf += chunk
            try:
                response = json.loads(buf.decode("utf-8"))
                break
            except json.JSONDecodeError:
                continue  # partial JSON; keep reading
    if not response or response.get("status") != "success":
        print(f"WARNING: in-session render returned unexpected response: {response}")
