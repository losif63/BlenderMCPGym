"""Backfill missing edit renders for completed runs.

A partial in-session render can leave a task with fewer ``renders/edit/`` images
than it has cameras. For task types where reopening the saved ``edit.blend`` is
faithful — geometry, blendshape, placement — this reopens each task's blend in a
fresh background Blender and renders any camera whose image is missing, without
re-running the agent.

Lighting and material are excluded by default: their edits can be lost on reopen,
so a backfilled render would be wrong. They must be (re)rendered in the live
session instead. ``--include-unsafe`` overrides this, at your own risk.

Examples:
    python -m blendergym.run.backfill_renders --model opus-4.8
    python -m blendergym.run.backfill_renders --model opus-4.8 --task_type placement
    python -m blendergym.run.backfill_renders --model opus-4.8 --tasks geometry1 placement22
    python -m blendergym.run.backfill_renders --model opus-4.8 --overwrite
"""

from argparse import ArgumentParser
import subprocess

from ..config import RunConfig
from ..paths import REPO_ROOT, TASKS_DIR, task_run_dir
from ..tasks import iter_tasks
from .blender import get_blender_env, get_blender_executable

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Task types whose edits survive a save + reopen, so a fresh-process render of
# edit.blend is faithful. Lighting/material are excluded (see module docstring).
REOPEN_SAFE_TYPES = ("geometry", "blendshape", "placement")

BACKFILL_SCRIPT = REPO_ROOT / "blendergym" / "render" / "backfill_render_script.py"


def _pngs(directory):
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def _task_type(name: str) -> str:
    return "".join(c for c in name if not c.isdigit())


def backfill_task(edit_blend, edit_dir, *, overwrite: bool,
                  virtual_display: bool) -> None:
    """Reopen edit_blend in a fresh background Blender and render its cameras
    into edit_dir (skipping existing files unless overwrite)."""
    edit_dir.mkdir(parents=True, exist_ok=True)
    argv = [str(edit_dir)] + (["--overwrite"] if overwrite else [])
    subprocess.run(
        [
            get_blender_executable(),
            "--background", str(edit_blend),
            "--python", str(BACKFILL_SCRIPT),
            "--", *argv,
        ],
        env=get_blender_env(virtual_display),
        check=True,
    )


def main(args) -> None:
    config = RunConfig(model=args.model, tool=args.tool, trial=args.trial)

    if args.tasks:
        names = list(args.tasks)
    else:
        names = [t.name for t in iter_tasks(args.task_type)]

    safe_types = REOPEN_SAFE_TYPES + (("lighting", "material") if args.include_unsafe else ())

    print(f"Backfilling renders for run {config.run_id} "
          f"(safe types: {', '.join(safe_types)})")

    n_filled = n_complete = n_skipped = n_failed = 0
    for name in names:
        ttype = _task_type(name)
        if ttype not in safe_types:
            print(f"[{name}] skip — '{ttype}' edits are not faithful on reopen "
                  f"(use --include-unsafe to force)")
            n_skipped += 1
            continue

        tdir = task_run_dir(config.run_id, name)
        edit_blend = tdir / "edit.blend"
        if not edit_blend.is_file():
            print(f"[{name}] skip — no edit.blend at {edit_blend}")
            n_skipped += 1
            continue

        edit_dir = tdir / "renders" / "edit"
        # Expected views come from the goal renders (named the same way); fill any
        # the edit dir is missing. Opening the blend would reveal extra cameras
        # too, but the goal set is what evaluation needs, so it gates the launch.
        expected = _pngs(TASKS_DIR / name / "renders" / "goal")
        missing = expected - _pngs(edit_dir)
        if not missing and not args.overwrite:
            print(f"[{name}] complete — {len(expected)} render(s) present")
            n_complete += 1
            continue

        action = "overwrite" if args.overwrite else f"missing {sorted(missing)}"
        print(f"[{name}] backfilling ({action}) -> {edit_dir}")
        try:
            backfill_task(edit_blend, edit_dir,
                          overwrite=args.overwrite, virtual_display=args.virtual)
            still_missing = expected - _pngs(edit_dir)
            if still_missing:
                print(f"[{name}] WARNING: still missing {sorted(still_missing)} after backfill")
                n_failed += 1
            else:
                n_filled += 1
        except subprocess.CalledProcessError as e:
            print(f"[{name}] ERROR: Blender exited {e.returncode}")
            n_failed += 1

    print(f"\nDone. filled={n_filled} already_complete={n_complete} "
          f"skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Backfill missing edit renders by reopening edit.blend.")
    parser.add_argument("--model", type=str, required=True, help="Model slug (see blendergym.config.MODELS)")
    parser.add_argument("--tool", type=str, default="claudecode", help="Agent tool")
    parser.add_argument("--trial", type=int, default=0, help="Trial index")
    parser.add_argument("--task_type", type=str, default=None,
                        help="Filter tasks by prefix (e.g. geometry); ignored if --tasks is given")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Explicit task names to backfill, e.g. geometry1 placement22")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render every camera even if its image already exists")
    parser.add_argument("--include-unsafe", action="store_true",
                        help="Also backfill lighting/material (reopen may drop their edits)")
    parser.add_argument("--virtual", action="store_true", help="Set DISPLAY=:99 for headless Blender")
    main(parser.parse_args())
