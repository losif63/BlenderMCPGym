"""
Analyze ver1 experiment results: average tool calls and duration per task type.
Uses the latest session in each task's ver1/ directory.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).parent / "bench_data"
TASK_TYPES = ["blendshape", "geometry", "material", "lighting", "placement"]
SKIP_DIRS = {"blender_files"}


def latest_session(ver1_dir: Path) -> Path | None:
    sessions = sorted(
        [d for d in ver1_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return sessions[-1] if sessions else None


def count_tool_calls(session_dir: Path) -> int:
    jsonl = session_dir / "tool_calls.jsonl"
    if not jsonl.exists():
        return 0
    with open(jsonl) as f:
        return sum(1 for line in f if line.strip())


def get_duration(task_dir: Path) -> float | None:
    meta = task_dir / "metadata_ver1.json"
    if not meta.exists():
        return None
    with open(meta) as f:
        data = json.load(f)
    return data.get("duration_seconds")


def collect_data():
    stats: dict[str, list[dict]] = defaultdict(list)

    for entry in sorted(BENCH_DIR.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue

        task_type = None
        for t in TASK_TYPES:
            if entry.name.startswith(t):
                task_type = t
                break
        if task_type is None:
            continue

        ver1_dir = entry / "ver1"
        if not ver1_dir.exists():
            continue

        session = latest_session(ver1_dir)
        if session is None:
            continue

        tool_calls = count_tool_calls(session)
        duration = get_duration(entry)

        if duration is not None:
            stats[task_type].append({
                "task": entry.name,
                "tool_calls": tool_calls,
                "duration_seconds": duration,
            })

    return stats


def summarize(stats):
    summary = {}
    for task_type, records in stats.items():
        tool_calls = [r["tool_calls"] for r in records]
        durations = [r["duration_seconds"] for r in records]
        summary[task_type] = {
            "count": len(records),
            "avg_tool_calls": np.mean(tool_calls),
            "std_tool_calls": np.std(tool_calls),
            "avg_duration_s": np.mean(durations),
            "std_duration_s": np.std(durations),
        }
    return summary


def plot(summary):
    types = TASK_TYPES
    present = [t for t in types if t in summary]

    avg_calls = [summary[t]["avg_tool_calls"] for t in present]
    avg_dur = [summary[t]["avg_duration_s"] / 60 for t in present]
    counts = [summary[t]["count"] for t in present]

    x = np.arange(len(present))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BlenderMCPGym Ver1 Results by Task Type", fontsize=14, fontweight="bold")

    bars1 = ax1.bar(x, avg_calls, width,
                    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}\n(n={counts[i]})" for i, t in enumerate(present)])
    ax1.set_ylabel("Tool Calls")
    ax1.set_title("Average Tool Calls per Task")
    ax1.bar_label(bars1, fmt="%.1f", padding=4)
    ax1.set_ylim(0, max(avg_calls) * 1.3)

    bars2 = ax2.bar(x, avg_dur, width,
                    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{t}\n(n={counts[i]})" for i, t in enumerate(present)])
    ax2.set_ylabel("Duration (minutes)")
    ax2.set_title("Average Duration per Task")
    ax2.bar_label(bars2, fmt="%.1f", padding=4)
    ax2.set_ylim(0, max(avg_dur) * 1.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "results_ver1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def main():
    stats = collect_data()
    summary = summarize(stats)

    print(f"\n{'Task Type':<12} {'N':>4}  {'Avg Calls':>10}  {'Std Calls':>10}  {'Avg Min':>8}  {'Std Min':>8}")
    print("-" * 62)
    for t in TASK_TYPES:
        if t not in summary:
            continue
        s = summary[t]
        print(
            f"{t:<12} {s['count']:>4}  {s['avg_tool_calls']:>10.2f}  "
            f"{s['std_tool_calls']:>10.2f}  {s['avg_duration_s']/60:>8.2f}  "
            f"{s['std_duration_s']/60:>8.2f}"
        )

    plot(summary)


if __name__ == "__main__":
    main()
