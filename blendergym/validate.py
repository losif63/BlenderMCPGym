"""Validate a run by deterministically replaying its logged Blender code.

For each task in a run, starts from the task's blender_file.blend + start.py,
replays the execute_blender_code steps from session/tool_calls.jsonl, and renders
every camera after each step into data/runs/<run_id>/<task>/validation/ as
seq{n}_camera{m}.png. If replaying the log reproduces the run's edit renders, the
run is authentic.

Renderer is configurable:
  eevee   fast; fine for geometry / blendshape / placement
  cycles  needed for faithful lighting / material
  auto    (default) cycles for lighting/material, eevee otherwise

Examples:
    python -m blendergym.validate --model opus-4.8 --task_type blendshape
    python -m blendergym.validate --model opus-4.8 --task_type lighting --engine cycles
"""

from argparse import ArgumentParser
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import RunConfig
from .paths import TASKS_DIR, task_run_dir
from .tasks import iter_tasks, Task
from .run.blender import get_blender_executable, get_blender_env

VALIDATE_SCRIPT = Path(__file__).resolve().parent / "render" / "validate_script.py"

# Task types whose appearance depends on shading -> need Cycles under engine=auto.
SHADING_TYPES = ("lighting", "material")


def resolve_engine(engine: str, task_name: str) -> str:
    if engine != "auto":
        return engine
    return "cycles" if task_name.startswith(SHADING_TYPES) else "eevee"


def validate_task(run_id: str, task_name: str, *, engine: str, samples: int,
                  resolution: int, render_start: bool, virtual_display: bool) -> bool:
    task = Task(task_name, TASKS_DIR)
    tool_calls = task_run_dir(run_id, task_name) / "session" / "tool_calls.jsonl"
    if not tool_calls.exists():
        print(f"[{run_id}/{task_name}] skip — no session/tool_calls.jsonl")
        return False
    if not task.blender_file.exists() or not task.start_script.exists():
        print(f"[{run_id}/{task_name}] skip — missing blender_file.blend or start.py")
        return False

    out_dir = task_run_dir(run_id, task_name) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = resolve_engine(engine, task_name)

    # Copy the .blend so bench_data/ is never opened/modified in place.
    work_fd, work_blend = tempfile.mkstemp(suffix=".blend")
    os.close(work_fd)
    shutil.copy2(task.blender_file, work_blend)

    cfg = {
        "start_py": str(task.start_script),
        "tool_calls": str(tool_calls),
        "out_dir": str(out_dir),
        "engine": eng,
        "samples": samples,
        "resolution": resolution,
        "render_start": render_start,
    }
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(cfg_fd, "w") as f:
        json.dump(cfg, f)

    print(f"[{run_id}/{task_name}] replaying (engine={eng}) -> {out_dir}")
    try:
        subprocess.run(
            [get_blender_executable(), "--background", work_blend,
             "--python", str(VALIDATE_SCRIPT), "--", cfg_path],
            env=get_blender_env(virtual_display), check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{run_id}/{task_name}] ERROR: Blender exited {e.returncode}")
        return False
    finally:
        os.remove(work_blend)
        os.remove(cfg_path)


def main(args) -> None:
    config = RunConfig(model=args.model, tool=args.tool, trial=args.trial)
    if args.task:
        tasks = [Task(args.task, TASKS_DIR)]
    else:
        tasks = iter_tasks(args.task_type)
    print(f"Validating {config.run_id}: {len(tasks)} candidate task(s) "
          f"(engine={args.engine})")
    done = 0
    for task in tasks:
        if validate_task(config.run_id, task.name, engine=args.engine,
                         samples=args.samples, resolution=args.resolution,
                         render_start=not args.no_start, virtual_display=args.virtual):
            done += 1
    print(f"\nValidated {done}/{len(tasks)} tasks for {config.run_id}.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Replay & validate a run's logged Blender code.")
    parser.add_argument("--model", type=str, required=True, help="Model slug of the run")
    parser.add_argument("--tool", type=str, default="claudecode")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--task", type=str, default=None,
                        help="Validate a single task by name (e.g. blendshape31); overrides --task_type")
    parser.add_argument("--task_type", type=str, default=None, help="Filter by task type prefix")
    parser.add_argument("--engine", choices=["auto", "eevee", "cycles"], default="auto",
                        help="Renderer: auto (cycles for lighting/material, else eevee)")
    parser.add_argument("--samples", type=int, default=64, help="Render samples (cycles or eevee TAA)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--no_start", action="store_true", help="Skip the seq0 start-state render")
    parser.add_argument("--virtual", action="store_true", help="DISPLAY=:99 for headless Blender")
    main(parser.parse_args())
