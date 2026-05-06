#!/usr/bin/env python3
"""Render every task's edit_{task}_ver3.blend into renders_v2/{task}/.

Mirrors the GPU/resolution/samples/camera logic from
bench_data/all_render_script.py, minus the exec(code) step (the edit blend
already contains the applied edits).

Usage (driver, system Python):
    python3 src/render_edits_v2.py [--out renders_v2] [--blender blender]
                                   [--only task1 task2 ...] [--skip-existing]

Internally this re-invokes itself inside Blender:
    blender -b <edit.blend> -P render_edits_v2.py -- <out_dir>
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Worker path (runs inside Blender). Detect via importable bpy.
# ---------------------------------------------------------------------------
try:
    import bpy  # type: ignore

    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False


def render_current_scene(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    cprefs = bpy.context.preferences.addons["cycles"].preferences
    cprefs.compute_device_type = "CUDA"
    cprefs.get_devices()
    for device in cprefs.devices:
        if device.type == "GPU" and not device.use:
            device.use = True
    scene.cycles.device = "GPU"

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "PNG"

    for i in range(1, 6):
        cam_name = f"Camera{i}"
        if cam_name in bpy.data.objects:
            scene.camera = bpy.data.objects[cam_name]
            scene.render.filepath = str(out_dir / f"render{i}.png")
            bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Driver path (runs under system Python).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "bench_data"
DEFAULT_BLENDER = REPO_ROOT / "infinigen" / "blender" / "blender"


def discover_tasks() -> list[str]:
    tasks = []
    for p in sorted(BENCH_DIR.iterdir()):
        if not p.is_dir() or p.name == "blender_files":
            continue
        if (p / f"edit_{p.name}_ver3.blend").is_file():
            tasks.append(p.name)
    return tasks


def has_renders(out_dir: Path) -> bool:
    return out_dir.is_dir() and any(out_dir.glob("render*.png"))


def render_task(task: str, blender: str, out_root: Path) -> int:
    blend = BENCH_DIR / task / f"edit_{task}_ver3.blend"
    out_dir = out_root / task
    cmd = [
        blender,
        "-b",
        str(blend),
        "-P",
        str(Path(__file__).resolve()),
        "--",
        str(out_dir),
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def main_driver() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "renders_v2")
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument(
        "--only",
        nargs="*",
        help="Render only these task names (default: all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output dir already contains render*.png.",
    )
    args = parser.parse_args()

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks()
    if args.only:
        wanted = set(args.only)
        tasks = [t for t in tasks if t in wanted]
        missing = wanted - set(tasks)
        if missing:
            print(f"warning: unknown tasks: {sorted(missing)}", file=sys.stderr)

    print(f"rendering {len(tasks)} tasks -> {out_root}")
    failed = []
    for i, task in enumerate(tasks, 1):
        out_dir = out_root / task
        if args.skip_existing and has_renders(out_dir):
            print(f"[{i}/{len(tasks)}] {task}: skip (already rendered)")
            continue
        print(f"[{i}/{len(tasks)}] {task}: rendering...")
        rc = render_task(task, args.blender, out_root)
        if rc != 0:
            print(f"  failed (exit {rc})", file=sys.stderr)
            failed.append(task)

    print(f"\ndone. {len(tasks) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


def main_worker() -> int:
    # When Blender invokes us, argv after `--` has the output dir.
    if "--" in sys.argv:
        extra = sys.argv[sys.argv.index("--") + 1 :]
    else:
        extra = []
    if not extra:
        print("worker: missing output dir argument", file=sys.stderr)
        return 2
    out_dir = Path(extra[0])
    render_current_scene(out_dir)
    return 0


if __name__ == "__main__":
    if INSIDE_BLENDER:
        sys.exit(main_worker())
    else:
        sys.exit(main_driver())
