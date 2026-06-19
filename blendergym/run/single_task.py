"""Run one task with one RunConfig.

Reads the task definition (read-only) and writes everything under
``data/runs/<run_id>/<task>/``:

    edit.blend                 the agent's edited scene
    renders/edit/render*.png   rendered result
    session/                   BlenderMCP tool_calls.jsonl + screenshots
    agent.log                  streamed agent stdout
    metadata.json              status, duration, model, git sha
"""

from argparse import ArgumentParser
import shutil
import subprocess
import time

from ..agents import get_runner
from ..config import RunConfig
from ..paths import REPO_ROOT, task_run_dir
from ..tasks import Task, build_prompt
from .. import runs
from .blender import (
    BLENDERMCP_PORT,
    get_blender_env,
    get_blender_executable,
    render_device_setup_code,
    render_in_session,
    save_blender_file,
    terminate_blender,
    wait_for_blendermcp,
)


def run_task(task: Task, config: RunConfig, *, port: int = BLENDERMCP_PORT, virtual_display: bool = False) -> None:
    start_time = time.time()
    out_dir = task_run_dir(config.run_id, task.name)
    edit_file = out_dir / "edit.blend"
    session_dir = out_dir / "session"
    renders_dir = out_dir / "renders" / "edit"
    log_path = out_dir / "agent.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(task, session_renders_dir=session_dir / "renders")
    shutil.copy2(task.blender_file, edit_file)

    blender_proc = subprocess.Popen([
        get_blender_executable(),
        str(edit_file),
        "--python", str(task.start_script),
        "--python-expr",
        f"import bpy; {render_device_setup_code()}"
        f"bpy.context.scene.blendermcp_port = {port}; "
        f"bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)",
    ], env=get_blender_env(virtual_display))

    # Always confirm Blender is terminated, even if the agent / render / save raises —
    # otherwise a leaked Blender keeps the MCP port and the next task piles up behind it.
    try:
        print(f"[{task.name}] Waiting for BlenderMCP server on port {port}...")
        if not wait_for_blendermcp(port=port):
            print(f"[{task.name}] ERROR: BlenderMCP server did not start within timeout.")
            _finish(config, task, start_time, status="blender_timeout")
            return

        print(f"[{task.name}] Server ready. Launching {config.tool} (model={config.model}).")
        runner = get_runner(config.tool)
        result = runner.run(
            prompt,
            cwd=REPO_ROOT,
            session_dir=session_dir,
            model_cli=config.model_cli,
            log_path=log_path,
        )

        # Render in the live agent session BEFORE saving/closing. Lighting and material
        # edits only materialize in memory until rendered; reopening the saved .blend in
        # a fresh process can drop them, so render here, then save, then close.
        renders_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{task.name}] Agent finished. Rendering edited scene (in session)...")
        render_in_session(renders_dir, port=port)

        print(f"[{task.name}] Render complete. Saving Blender file...")
        save_blender_file(port=port)

        status = "ok" if result.ok else "agent_error"
        meta = _finish(config, task, start_time, status=status, session_id=result.session_id)
        print(f"[{task.name}] Done ({status}) in {meta['duration_seconds']}s "
              f"(session {result.session_id}) -> {out_dir}")
    finally:
        print(f"[{task.name}] Terminating Blender and confirming exit...")
        terminate_blender(blender_proc, port=port)


def _finish(config: RunConfig, task: Task, start_time: float, *, status: str,
            session_id: str | None = None) -> dict:
    duration = time.time() - start_time
    meta = runs.write_task_metadata(config, task.name, duration_seconds=duration,
                                    status=status, session_id=session_id)
    runs.update_registry(config, task.name, status=status, duration_seconds=duration,
                         session_id=session_id)
    return meta


def main(args) -> None:
    config = RunConfig(model=args.model, tool=args.tool, trial=args.trial)
    task = Task(args.task)
    if not task.exists():
        raise SystemExit(f"Task {args.task!r} not found at {task.dir}")
    run_task(task, config, virtual_display=args.virtual)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run one scene-edit task with one model.")
    parser.add_argument("--task", type=str, default="blendshape1", help="Task name, e.g. blendshape1")
    parser.add_argument("--model", type=str, default="opus-4.8", help="Model slug (see blendergym.config.MODELS)")
    parser.add_argument("--tool", type=str, default="claudecode", help="Agent tool")
    parser.add_argument("--trial", type=int, default=0, help="Trial index (repeat runs)")
    parser.add_argument("--virtual", action="store_true", help="Set DISPLAY=:99 for headless Blender")
    main(parser.parse_args())
