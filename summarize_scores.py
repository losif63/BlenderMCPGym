import json

SCORES_PATH = "renders_remote_v2/scores.json"
TASK_TYPES = ["blendshape", "geometry", "lighting", "material", "placement"]

with open(SCORES_PATH, "r") as f:
    scores = json.load(f)

summary = {}
for task_type in TASK_TYPES:
    pl_values = []
    n_clip_values = []

    for task_name, task_scores in scores.items():
        if task_name.startswith(task_type):
            pl_values.append(task_scores["avg_photometric_loss"])
            n_clip_values.append(task_scores["avg_n_clip"])

    if pl_values:
        summary[task_type] = {
            "count": len(pl_values),
            "avg_photometric_loss": sum(pl_values) / len(pl_values),
            "avg_n_clip": sum(n_clip_values) / len(n_clip_values),
        }
    else:
        summary[task_type] = {"count": 0}

print(f"{'Task Type':<12} {'Count':>6} {'Avg PL':>12} {'Avg N-CLIP':>12}")
print("-" * 46)
for task_type, stats in summary.items():
    if stats["count"] == 0:
        print(f"{task_type:<12} {'0':>6} {'N/A':>12} {'N/A':>12}")
    else:
        print(f"{task_type:<12} {stats['count']:>6} {stats['avg_photometric_loss']:>12.4f} {stats['avg_n_clip']:>12.4f}")

output_path = "renders_remote_v2/summary_scores.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nSaved summary to {output_path}")
