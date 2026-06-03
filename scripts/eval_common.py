"""
Shared helpers for the recreation eval / comparison scripts.

The single source of truth for "which image represents a run" is the most-recent
PNG the agent wrote under that run's process/ directory (by modification time,
since iteration filenames like iter_5 / iter_46 don't sort lexicographically).
render_recreations.py copies that image to render.png; the eval and montage
scripts select it directly via latest_process_image() so they agree regardless
of whether render.png has been refreshed.
"""

from pathlib import Path


def latest_process_image(model_dir) -> Path | None:
    """Most-recently-modified *.png under {model_dir}/process/, or None if the
    process/ dir is missing or empty."""
    process = Path(model_dir) / "process"
    if not process.is_dir():
        return None
    pngs = list(process.glob("*.png"))
    if not pngs:
        return None
    return max(pngs, key=lambda p: p.stat().st_mtime)


def output_suffix(platform: str, recreation_dir_name: str) -> str:
    """Suffix for an eval output filename, mirroring both the platform and the
    recreation root so different conditions never collide:

        recreation     + claudecode            -> ""            (dinov3_vitl16.json)
        recreation     + claudecode_v2          -> "_v2"        (..._v2.json)
        recreation     + claudecode_v2_avg      -> "_v2_avg"
        recreation     + claudecode_v2_vector   -> "_v2_vector"
        recreation_old + claudecode_v2          -> "_v2_old"
    """
    if platform == "claudecode":
        plat = ""
    else:
        plat = "_" + platform[len("claudecode_"):]

    if recreation_dir_name == "recreation":
        root = ""
    elif recreation_dir_name.startswith("recreation_"):
        root = "_" + recreation_dir_name[len("recreation_"):]
    else:
        root = "_" + recreation_dir_name
    return plat + root
