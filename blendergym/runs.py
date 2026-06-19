"""Bookkeeping for run outputs: per-task metadata, the run registry, and
"already done" checks. Keeps single_task / run_benchmark free of JSON plumbing.
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .config import RunConfig
from .paths import RUNS_DIR, REPO_ROOT, task_run_dir

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def claude_transcript_path(session_id: str | None, cwd: Path = REPO_ROOT) -> str | None:
    """Resolve a claudecode session id to its ~/.claude transcript .jsonl.

    Claude Code stores transcripts under a project slug derived from the cwd by
    replacing every non-alphanumeric character with '-'.
    """
    if not session_id:
        return None
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    return str(Path.home() / ".claude" / "projects" / slug / f"{session_id}.jsonl")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def edit_renders_dir(config: RunConfig, task_name: str) -> Path:
    return task_run_dir(config.run_id, task_name) / "renders" / "edit"


def is_done(config: RunConfig, task_name: str) -> bool:
    """True if this task already has an edited blend + at least one render."""
    tdir = task_run_dir(config.run_id, task_name)
    edit = tdir / "edit.blend"
    renders = edit_renders_dir(config, task_name)
    has_render = renders.is_dir() and any(
        p.suffix.lower() in IMAGE_EXTS for p in renders.iterdir()
    )
    return edit.is_file() and has_render


def write_task_metadata(config: RunConfig, task_name: str, *, duration_seconds: float,
                        status: str, session_id: str | None = None) -> dict:
    meta = {
        "run_id": config.run_id,
        "tool": config.tool,
        "model": config.model,
        "model_cli": config.model_cli,
        "trial": config.trial,
        "task": task_name,
        "status": status,
        "duration_seconds": round(duration_seconds, 2),
        "session_id": session_id,
        "transcript_path": claude_transcript_path(session_id),
        "git_sha": git_sha(),
        "timestamp": _now(),
    }
    path = task_run_dir(config.run_id, task_name) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def update_registry(config: RunConfig, task_name: str, *, status: str, duration_seconds: float,
                    session_id: str | None = None) -> None:
    """Maintain data/runs/<run_id>/registry.json with per-task status."""
    path = RUNS_DIR / config.run_id / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            reg = json.load(f)
    else:
        reg = {
            "run_id": config.run_id,
            "tool": config.tool,
            "model": config.model,
            "model_cli": config.model_cli,
            "trial": config.trial,
            "created": _now(),
            "git_sha": git_sha(),
            "tasks": {},
        }
    reg["tasks"][task_name] = {
        "status": status,
        "duration_seconds": round(duration_seconds, 2),
        "session_id": session_id,
        "timestamp": _now(),
    }
    reg["updated"] = _now()
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)
