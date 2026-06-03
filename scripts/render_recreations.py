#!/usr/bin/env python3
"""
Populate render.png for every run by copying the most-recent image the agent
wrote under that run's process/ directory (selected by modification time).

This replaces the previous behaviour of re-rendering each blender_file.blend in
headless Blender: instead of producing a fresh render, we adopt the agent's last
process/ render as the canonical render.png that the *_eval.py and montage
scripts read. Pass --recreation-dir to also process recreation_old (the script
auto-discovers every {task}/{platform}/{model} under each root).
"""

import argparse
import shutil
import sys
from pathlib import Path

from eval_common import latest_process_image

ROOT = Path(__file__).resolve().parent.parent
# ALL_MODELS = ["haiku-4.5", "sonnet-4.6", "opus-4.6", "opus-4.7"]
ALL_MODELS = ["opus-4.7"]


def main():
    parser = argparse.ArgumentParser(
        description="Copy each run's latest process/ image to render.png."
    )
    parser.add_argument("--recreation-dir", nargs="+", default=["recreation"],
                        help="Recreation root(s) to process (e.g. recreation recreation_old).")
    parser.add_argument("--model", nargs="+", default=ALL_MODELS,
                        help="Model dir name(s) to process (default: opus-4.7).")
    parser.add_argument("--task", nargs="+", default=None,
                        help="Task name(s) to process (default: all).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite render.png even if it already exists.")
    args = parser.parse_args()

    models = args.model
    tasks = set(args.task) if args.task else None

    jobs: list[tuple[Path, Path]] = []  # (src process image, dst render.png)
    skipped = 0
    no_process = 0

    for rec_name in args.recreation_dir:
        rec_dir = Path(rec_name)
        if not rec_dir.is_absolute():
            rec_dir = ROOT / rec_dir
        if not rec_dir.is_dir():
            print(f"[WARN] recreation dir not found: {rec_dir}")
            continue

        for task_dir in sorted(rec_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            if tasks and task_dir.name not in tasks:
                continue
            for platform_dir in sorted(task_dir.iterdir()):
                if not platform_dir.is_dir():
                    continue
                for model in models:
                    model_dir = platform_dir / model
                    if not model_dir.is_dir():
                        continue
                    src = latest_process_image(model_dir)
                    if src is None:
                        no_process += 1
                        continue
                    dst = model_dir / "render.png"
                    if dst.exists() and not args.force:
                        skipped += 1
                        continue
                    jobs.append((src, dst))

    print(f"Found {len(jobs)} render.png to write "
          f"({skipped} skipped — already present, pass --force to overwrite; "
          f"{no_process} runs had no process/*.png).\n")

    copied, failed = 0, 0
    for i, (src, dst) in enumerate(jobs, 1):
        label = "/".join(dst.parts[-4:-1])
        print(f"[{i}/{len(jobs)}] {label}  <-  {src.name} ...", end=" ", flush=True)
        try:
            shutil.copy2(src, dst)
            print("OK")
            copied += 1
        except OSError as e:
            print(f"FAIL ({e})")
            failed += 1

    print(f"\nDone. {copied} copied, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
