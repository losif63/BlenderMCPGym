#!/usr/bin/env python3
"""
Render step-by-step trajectories for bench_data tasks.

Usage:
    python render_trajectory.py               # all tasks
    python render_trajectory.py blendshape1   # single task
    python render_trajectory.py --skip-existing  # skip tasks that already have trajectory/

For each task's most recent session under <task>/ver1/ the script will:
  1. Copy <task>/blender_file.blend  →  <task>/trajectory_logger.blend
  2. Open it headlessly in Blender
  3. Execute start.py to initialise the scene
  4. For every code/{seq}.py (ascending order):
       - Execute the code
       - Render from all cameras → ver1/<session>/trajectory/seq_NNN_camX.png
  5. Delete trajectory_logger.blend
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

BENCH_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data")
BLENDER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infinigen", "Blender.app", "Contents", "MacOS", "Blender")
INNER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_trajectory_blender_script.py")
SKIP_DIRS    = {"blender_files"}


def get_latest_session(ver1_dir: str) -> str | None:
    sessions = [
        s for s in os.listdir(ver1_dir)
        if s.startswith("session_") and os.path.isdir(os.path.join(ver1_dir, s))
    ]
    return sorted(sessions)[-1] if sessions else None


def render_task(task_name: str, skip_existing: bool) -> str:
    """
    Render trajectory for one task.
    Returns a status string: 'ok', 'skipped', or 'error: <reason>'.
    """
    task_dir = os.path.join(BENCH_DIR, task_name)

    blend_src = os.path.join(task_dir, "blender_file.blend")
    if not os.path.exists(blend_src):
        return "error: blender_file.blend not found"

    start_py = os.path.join(task_dir, "start.py")
    if not os.path.exists(start_py):
        return "error: start.py not found"

    ver1_dir = os.path.join(task_dir, "ver1")
    if not os.path.isdir(ver1_dir):
        return "error: ver1/ not found"

    session_name = get_latest_session(ver1_dir)
    if session_name is None:
        return "error: no session directories found"

    session_dir    = os.path.join(ver1_dir, session_name)
    code_dir       = os.path.join(session_dir, "code")
    trajectory_dir = os.path.join(session_dir, "trajectory")

    if not os.path.isdir(code_dir):
        return "error: code/ not found — run extract_trajectory.py first"

    code_files = [f for f in os.listdir(code_dir) if f.endswith(".py")]
    if not code_files:
        return "error: no .py files in code/"

    if skip_existing and os.path.isdir(trajectory_dir) and os.listdir(trajectory_dir):
        return "skipped"

    blend_copy = os.path.join(task_dir, "trajectory_logger.blend")
    shutil.copy2(blend_src, blend_copy)

    config = {
        "start_py":       start_py,
        "code_dir":       code_dir,
        "trajectory_dir": trajectory_dir,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(config, tmp, indent=2)
        config_path = tmp.name

    cmd = [
        BLENDER,
        "--background", blend_copy,
        "--python", INNER_SCRIPT,
        "--",
        config_path,
    ]
    try:
        result = subprocess.run(cmd, check=False)
        status = "ok" if result.returncode == 0 else f"error: blender exit {result.returncode}"
    except Exception as exc:
        status = f"error: {exc}"
    finally:
        os.unlink(config_path)
        if os.path.exists(blend_copy):
            os.remove(blend_copy)

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trajectories for bench_data tasks.")
    parser.add_argument(
        "task", nargs="?", default=None,
        help="Single task name (e.g. blendshape1). Omit to process all tasks.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tasks whose trajectory/ directory is already populated.",
    )
    args = parser.parse_args()

    if args.task:
        task_names = [args.task]
        if not os.path.isdir(os.path.join(BENCH_DIR, args.task)):
            sys.exit(f"ERROR: task directory not found: {args.task}")
    else:
        task_names = sorted(
            d for d in os.listdir(BENCH_DIR)
            if os.path.isdir(os.path.join(BENCH_DIR, d)) and d not in SKIP_DIRS
        )

    total   = len(task_names)
    results = {"ok": [], "skipped": [], "error": []}

    for i, task_name in enumerate(task_names, 1):
        print(f"\n[{i}/{total}] {task_name}")
        status = render_task(task_name, args.skip_existing)
        print(f"  → {status}")
        key = "error" if status.startswith("error") else status
        results[key].append(task_name)

    print(f"\n{'='*50}")
    print(f"Done.  ok={len(results['ok'])}  skipped={len(results['skipped'])}  errors={len(results['error'])}")
    if results["error"]:
        print("Errored tasks:")
        for t in results["error"]:
            print(f"  {t}")


if __name__ == "__main__":
    main()
