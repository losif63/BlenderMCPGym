#!/usr/bin/env python3
"""
Extract blender code and info from tool_calls.jsonl for every task under bench_data/.
For each task's most recent session under ver1/:
  - Creates session_dir/code/{seq}.py  for every execute_blender_code call
  - Creates session_dir/info/{seq}.json for every get_scene_info / get_object_info call
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(PROJECT_ROOT, "bench_data")


def get_latest_session(ver1_dir: str) -> str | None:
    sessions = [
        s for s in os.listdir(ver1_dir)
        if s.startswith("session_") and os.path.isdir(os.path.join(ver1_dir, s))
    ]
    return sorted(sessions)[-1] if sessions else None


def process_task(task_dir: str) -> None:
    ver1_dir = os.path.join(task_dir, "ver1")
    if not os.path.isdir(ver1_dir):
        return

    session_name = get_latest_session(ver1_dir)
    if session_name is None:
        return

    session_dir = os.path.join(ver1_dir, session_name)
    jsonl_path = os.path.join(session_dir, "tool_calls.jsonl")
    if not os.path.exists(jsonl_path):
        return

    code_dir = os.path.join(session_dir, "code")
    info_dir = os.path.join(session_dir, "info")
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            tool = entry.get("tool")
            seq = entry.get("seq")

            if tool == "execute_blender_code":
                code = entry.get("params", {}).get("code", "")
                out_path = os.path.join(code_dir, f"{seq}.py")
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(code)
                    if "output" in entry:
                        out.write("\n# " + "-" * 60 + "\n")
                        for line in entry["output"].splitlines():
                            out.write(f"# {line}\n")
                    elif "error" in entry:
                        out.write("\n# " + "-" * 60 + "\n")
                        out.write("# ERROR:\n")
                        for line in entry["error"].splitlines():
                            out.write(f"# {line}\n")

            elif tool in ("get_scene_info", "get_object_info"):
                result = entry.get("result")
                out_path = os.path.join(info_dir, f"{seq}.json")
                with open(out_path, "w", encoding="utf-8") as out:
                    json.dump(result, out, indent=2, ensure_ascii=False)


def main() -> None:
    if not os.path.isdir(BENCH_DIR):
        raise FileNotFoundError(f"bench_data not found at {BENCH_DIR}")

    tasks = sorted(
        d for d in os.listdir(BENCH_DIR)
        if os.path.isdir(os.path.join(BENCH_DIR, d)) and d not in ("blender_files",)
    )

    processed = 0
    for task_name in tasks:
        task_dir = os.path.join(BENCH_DIR, task_name)
        process_task(task_dir)
        processed += 1

    print(f"Done. Processed {processed} tasks.")


if __name__ == "__main__":
    main()
