#!/usr/bin/env python3
"""
Compare per-task iteration counts between two feedback conditions, to test the
hypothesis that a single scalar signal (ensemble_avg) makes the agent iterate
more — oscillating without getting closer — while the per-metric vector
(ensemble_vector) suppresses that.

For each task it reads {recreation}/{task}/{platform}/{model}/score_log.jsonl
(one record per score_render call) and reports:

  iters        number of score_render calls (= render→score loop iterations)
  best         lowest "closeness" reached (mean of the FED-metric distances;
               lower = closer to the reference). This is exactly the avg
               condition's own fed signal, computed identically for both.
  final        closeness at the last iteration
  post_best    iterations AFTER the best was first reached (high => the agent
               kept going without improving on its best — the oscillation tell)

Outputs a per-task table, an aggregate summary, and (optionally) a JSON dump.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TASKS = [f"advanced{i}" for i in range(1, 11)] + [f"beginner{i}" for i in range(1, 11)]


def closeness(record, metric):
    """Lower = closer. 'mean_fed' averages the record's fed_metrics; otherwise a
    single named metric. Returns None if unavailable."""
    metrics = record.get("metrics") or {}
    if metric == "mean_fed":
        names = record.get("fed_metrics") or []
        vals = [metrics[n] for n in names if isinstance(metrics.get(n), (int, float))]
        return sum(vals) / len(vals) if vals else None
    v = metrics.get(metric)
    return v if isinstance(v, (int, float)) else None


def run_stats(log_path: Path, metric: str):
    """Parse one score_log.jsonl into per-run stats, or None if absent/empty."""
    if not log_path.exists():
        return None
    series = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            series.append(closeness(rec, metric))

    iters = len(series)
    if iters == 0:
        return None

    valid = [(i, c) for i, c in enumerate(series) if c is not None]
    if valid:
        best_idx, best = min(valid, key=lambda t: t[1])
        final = next((c for _, c in reversed(valid)), None)
        post_best = iters - (best_idx + 1)
    else:
        best = final = None
        post_best = None

    return {"iters": iters, "best": best, "final": final, "post_best": post_best}


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def main():
    parser = argparse.ArgumentParser(description="Compare iteration counts across feedback modes.")
    parser.add_argument("--recreation-dir", default="recreation")
    parser.add_argument("--model", default="opus-4.7")
    parser.add_argument("--platform-a", default="claudecode_v2_avg")
    parser.add_argument("--platform-b", default="claudecode_v2_vector")
    parser.add_argument("--metric", default="mean_fed",
                        help="Closeness metric: 'mean_fed' (default) or a single metric name (e.g. dinov3).")
    parser.add_argument("--task", nargs="+", default=None, help="Task(s) (default: all).")
    parser.add_argument("--save", default=None, help="Write the full comparison to this JSON path.")
    args = parser.parse_args()

    rec = Path(args.recreation_dir)
    rec = rec if rec.is_absolute() else ROOT / rec
    A, B = args.platform_a, args.platform_b
    tasks = args.task if args.task else TASKS

    def log(task, platform):
        return rec / task / platform / args.model / "score_log.jsonl"

    rows = []
    for task in tasks:
        a = run_stats(log(task, A), args.metric)
        b = run_stats(log(task, B), args.metric)
        if a is None and b is None:
            continue
        rows.append((task, a, b))

    # ---- per-task table ----
    short_a, short_b = A.replace("claudecode_v2_", ""), B.replace("claudecode_v2_", "")
    print(f"\nCloseness metric: {args.metric} (lower = closer)   model: {args.model}")
    print(f"A = {short_a}    B = {short_b}\n")
    header = (f"{'task':<12} {'A_iters':>7} {'B_iters':>7} {'Δiters':>7}   "
              f"{'A_best':>7} {'B_best':>7}   {'A_post':>6} {'B_post':>6}")
    print(header)
    print("-" * len(header))
    for task, a, b in rows:
        ai = a["iters"] if a else None
        bi = b["iters"] if b else None
        di = (ai - bi) if (ai is not None and bi is not None) else None
        print(f"{task:<12} "
              f"{(ai if ai is not None else '-'):>7} "
              f"{(bi if bi is not None else '-'):>7} "
              f"{(di if di is not None else '-'):>7}   "
              f"{fmt(a['best'] if a else None):>7} {fmt(b['best'] if b else None):>7}   "
              f"{(a['post_best'] if a and a['post_best'] is not None else '-'):>6} "
              f"{(b['post_best'] if b and b['post_best'] is not None else '-'):>6}")

    # ---- aggregate ----
    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    paired = [(a, b) for _, a, b in rows if a and b]
    a_iters = mean([a["iters"] for a, _ in paired])
    b_iters = mean([b["iters"] for _, b in paired])
    a_best = mean([a["best"] for a, _ in paired])
    b_best = mean([b["best"] for _, b in paired])
    a_post = mean([a["post_best"] for a, _ in paired])
    b_post = mean([b["post_best"] for _, b in paired])

    print("-" * len(header))
    print(f"{'MEAN(n=' + str(len(paired)) + ')':<12} "
          f"{fmt(a_iters,1):>7} {fmt(b_iters,1):>7} "
          f"{fmt((a_iters - b_iters) if (a_iters and b_iters) else None,1):>7}   "
          f"{fmt(a_best):>7} {fmt(b_best):>7}   {fmt(a_post,1):>6} {fmt(b_post,1):>6}")

    if a_iters and b_iters:
        print(f"\n{short_a} runs {a_iters / b_iters:.2f}x as many iterations as {short_b} "
              f"(mean {a_iters:.1f} vs {b_iters:.1f}).")
        if a_best is not None and b_best is not None:
            better = short_a if a_best < b_best else short_b
            print(f"Best closeness: {short_a} {a_best:.3f} vs {short_b} {b_best:.3f} "
                  f"(closer: {better}; Δ={abs(a_best - b_best):.3f}).")
        if a_post is not None and b_post is not None:
            print(f"Iterations after best: {short_a} {a_post:.1f} vs {short_b} {b_post:.1f} "
                  f"— higher = more no-improvement churn.")

    if args.save:
        out = Path(args.save)
        out = out if out.is_absolute() else ROOT / out
        payload = {
            "metric": args.metric, "model": args.model,
            "platform_a": A, "platform_b": B,
            "per_task": {t: {"a": a, "b": b} for t, a, b in rows},
            "aggregate": {
                "n": len(paired),
                "a_iters": a_iters, "b_iters": b_iters,
                "a_best": a_best, "b_best": b_best,
                "a_post_best": a_post, "b_post_best": b_post,
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
