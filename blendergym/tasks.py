"""Task definitions and v3 prompt construction.

A Task is an immutable benchmark instance under TASKS_DIR. v3 is the only
condition: the agent is given start.py plus rendered start/goal images and no
textual instruction.
"""

import re
from pathlib import Path

from .paths import TASKS_DIR

TASK_RE = re.compile(r"^(blendshape|geometry|lighting|material|placement)\d+$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# v3: start.py + rendered start/goal images, no textual instruction.
SYSTEM_PROMPT = (
    "Your task is to edit a Blender scene using BlenderMCP tools. "
    "You will be given the Blender Python initialization script (start.py) used to set up "
    "the scene, rendered images of the current scene, and rendered images of the target scene. "
    "Study the start.py code to understand the scene's objects, variables, and methods, then "
    "use the BlenderMCP tools to modify the scene so that its rendered result matches the "
    "target images as closely as possible. "
    "IMPORTANT: Make your edits by building on the existing variables and methods defined in "
    "start.py only — do not create new objects or use approaches inconsistent with how the "
    "scene was initialized. "
    "When you are done with all edits, stop. Do not close Blender or stop the server."
)


class Task:
    def __init__(self, name: str, root: Path = TASKS_DIR):
        self.name = name
        self.dir = Path(root) / name

    @property
    def blender_file(self) -> Path:
        return self.dir / "blender_file.blend"

    @property
    def start_script(self) -> Path:
        return self.dir / "start.py"

    @property
    def start_renders(self) -> Path:
        return self.dir / "renders" / "start"

    @property
    def goal_renders(self) -> Path:
        return self.dir / "renders" / "goal"

    def exists(self) -> bool:
        return self.dir.is_dir()

    def __repr__(self) -> str:
        return f"Task({self.name!r})"


def iter_tasks(task_type: str | None = None, root: Path = TASKS_DIR) -> list[Task]:
    """All benchmark tasks under root, optionally filtered by type prefix."""
    root = Path(root)
    names = sorted(d.name for d in root.iterdir() if d.is_dir() and TASK_RE.match(d.name))
    if task_type:
        names = [n for n in names if n.startswith(task_type)]
    return [Task(n, root) for n in names]


def _images(d: Path) -> list[str]:
    d = Path(d)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_prompt(task: Task, session_renders_dir=None) -> str:
    """Assemble the prompt for one task.

    If ``session_renders_dir`` is given, its absolute path is embedded in the
    prompt so the agent saves intermediate renders to a known, run-scoped
    location instead of reconstructing a templated path (which it can't do
    reliably from inside the session — it doesn't know its own model slug).
    """
    start_code = Path(task.start_script).read_text()
    start_paths = "\n".join(f"  - {p}" for p in _images(task.start_renders))
    goal_paths = "\n".join(f"  - {p}" for p in _images(task.goal_renders))
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Initialization Script (start.py)\n"
        f"The following Blender Python script was used to initialize the scene. "
        f"Use only the variables, objects, and methods defined here when making your edits:\n"
        f"```python\n{start_code}\n```\n\n"
        f"## Starting Scene\n"
        f"The following are rendered images of the current Blender scene. Read each file before proceeding:\n"
        f"{start_paths}\n\n"
        f"## Target Scene\n"
        f"The following are rendered images of the target scene showing the desired result. Read each file before proceeding:\n"
        f"{goal_paths}"
    )
    if session_renders_dir is not None:
        prompt += (
            f"\n\n## Saving Renders\n"
            f"If you render images while working (e.g. to compare your edits against the target), "
            f"save them under this exact directory, creating it if it does not exist:\n"
            f"  {session_renders_dir}\n"
            f"Set bpy.context.scene.render.filepath accordingly. Do not write renders anywhere else — "
            f"in particular not under bench_data/ (read-only) or /tmp/."
        )
    return prompt
