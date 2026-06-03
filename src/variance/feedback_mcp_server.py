"""
MCP server that scores rendered images against a fixed reference image.

Launched by `run_claudecode_v2.py` as part of a per-task MCP config. Exposes a
single tool, `score_render(render_path)`, that the agent calls after each render
to receive a visual-similarity signal against the reference image (lower =
closer). The signal is one of three feedback variants, selected by FEEDBACK_MODE:

  dinov3_only      Single DINOv3 cosine distance (the original v2 control).
  ensemble_avg     Single scalar = unweighted mean of the raw FED-metric distances.
  ensemble_vector  Labeled per-FED-metric breakdown so the agent can reason about
                   which aspect is off.

Held-out metrics (Inception, LPIPS) are ALWAYS computed and logged but are NEVER
included in the signal sent to the agent — they are for offline evaluation only.

Opt-in boundary
---------------
If FEEDBACK_MODE is UNSET, the server runs the original lightweight path: DINOv3
only, original return dict, no new dependencies. This keeps reruns of old
mcp_config.json files byte-identical. Setting FEEDBACK_MODE (the new runner sets
it to "dinov3_only" by default) engages the full fed + held-out ensemble.

Configuration (env vars)
------------------------
  REFERENCE_IMAGE   Absolute path to the reference image (required).
  FEEDBACK_MODE     dinov3_only | ensemble_avg | ensemble_vector. Unset => legacy.
  FED_METRICS       Comma list fed to the agent. Default: dinov3,siglip2,convnextv2,dreamsim
  HELDOUT_METRICS   Comma list logged but never fed. Default: inception,lpips
  TASK_ID           Logged with each record. Default: reference image stem.
  SCORE_LOG_PATH    If set, append one JSONL record per score_render call.
  DINOV3_MODEL      Back-compat checkpoint override for dinov3.
  <METRIC>_CHECKPOINT  Per-metric checkpoint override, e.g. SIGLIP2_CHECKPOINT.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from metrics import LABELS, MetricEnsemble

REFERENCE_IMAGE = os.environ.get("REFERENCE_IMAGE")
if not REFERENCE_IMAGE:
    print("ERROR: REFERENCE_IMAGE env var must be set", file=sys.stderr)
    sys.exit(1)
if not Path(REFERENCE_IMAGE).exists():
    print(f"ERROR: reference image not found: {REFERENCE_IMAGE}", file=sys.stderr)
    sys.exit(1)

VALID_MODES = ("dinov3_only", "ensemble_avg", "ensemble_vector")

# Unset FEEDBACK_MODE => legacy path: DINOv3 only, no held-out, no new deps.
_raw_mode = os.environ.get("FEEDBACK_MODE")
LEGACY = _raw_mode is None
FEEDBACK_MODE = _raw_mode or "dinov3_only"
if FEEDBACK_MODE not in VALID_MODES:
    print(
        f"ERROR: FEEDBACK_MODE='{FEEDBACK_MODE}' invalid; must be one of {VALID_MODES}",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_list(env_value, default):
    if env_value is None:
        return list(default)
    return [x.strip() for x in env_value.split(",") if x.strip()]


if LEGACY:
    FED_METRICS = ["dinov3"]
    HELDOUT_METRICS = []
else:
    FED_METRICS = _parse_list(
        os.environ.get("FED_METRICS"),
        ["dinov3", "siglip2", "convnextv2", "dreamsim"],
    )
    HELDOUT_METRICS = _parse_list(os.environ.get("HELDOUT_METRICS"), ["inception", "lpips"])

if FEEDBACK_MODE == "dinov3_only" and "dinov3" not in FED_METRICS:
    print(
        "ERROR: feedback_mode 'dinov3_only' requires 'dinov3' in FED_METRICS",
        file=sys.stderr,
    )
    sys.exit(1)

TASK_ID = os.environ.get("TASK_ID") or Path(REFERENCE_IMAGE).stem

SCORE_LOG_PATH = os.environ.get("SCORE_LOG_PATH")
if SCORE_LOG_PATH:
    Path(SCORE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"[feedback_mcp_server] logging scores to {SCORE_LOG_PATH}", file=sys.stderr)


def _checkpoint_for(name):
    # Back-compat: DINOV3_MODEL kept working from the original server.
    if name == "dinov3" and os.environ.get("DINOV3_MODEL"):
        return os.environ["DINOV3_MODEL"]
    return os.environ.get(f"{name.upper()}_CHECKPOINT")


checkpoints = {n: _checkpoint_for(n) for n in FED_METRICS + HELDOUT_METRICS}

# Device selection: METRIC_DEVICE pins everything; <METRIC>_DEVICE overrides one
# metric; otherwise each metric uses its default (auto MPS/CPU, except metrics
# pinned to CPU in metrics.DEFAULT_DEVICE_OVERRIDES for reproducibility).
GLOBAL_DEVICE = os.environ.get("METRIC_DEVICE")
devices = {
    n: os.environ[f"{n.upper()}_DEVICE"]
    for n in FED_METRICS + HELDOUT_METRICS
    if os.environ.get(f"{n.upper()}_DEVICE")
}

print(
    f"[feedback_mcp_server] mode={FEEDBACK_MODE} "
    f"fed={FED_METRICS} heldout={HELDOUT_METRICS}; loading models...",
    file=sys.stderr,
)
ensemble = MetricEnsemble(
    FED_METRICS,
    HELDOUT_METRICS,
    checkpoints=checkpoints,
    devices=devices,
    device=GLOBAL_DEVICE,
)
print(
    f"[feedback_mcp_server] embedding reference image {REFERENCE_IMAGE} "
    f"on {ensemble.device}...",
    file=sys.stderr,
)
ensemble.warm(REFERENCE_IMAGE)
print("[feedback_mcp_server] ready.", file=sys.stderr)

mcp = FastMCP("render_feedback")

_step_index = 0


def _build_feedback(results):
    """Return (payload_dict, feedback_string) for the active mode.

    payload_dict is what the agent literally receives back from the tool;
    feedback_string is the human-readable signal, logged verbatim.
    """
    fed = ensemble.fed_values(results)
    if FEEDBACK_MODE == "dinov3_only":
        d = results["dinov3"]
        # Return dict kept identical to the original v2 server for the control.
        payload = {
            "dinov3_cosine_distance": d,
            "reference_image": REFERENCE_IMAGE,
            "render_image": None,  # filled by caller
            "note": "Lower is closer to the reference; 0.0 means identical embeddings.",
        }
        feedback_string = (
            f"DINOv3 cosine distance to target: {d:.6f} "
            f"(lower is closer; 0.0 = identical)."
        )
        return payload, feedback_string

    if FEEDBACK_MODE == "ensemble_avg":
        mean = sum(fed.values()) / len(fed)
        feedback_string = (
            f"Visual feedback (mean distance to target across "
            f"{len(fed)} metrics, lower = closer): {mean:.4f}"
        )
    else:  # ensemble_vector
        lines = ["Visual feedback (distance to target, lower = closer):"]
        for name in ensemble.fed_names:
            label = LABELS.get(name, name) + ":"
            lines.append(f"  {label:<11}{results[name]:.4f}")
        feedback_string = "\n".join(lines)

    payload = {
        "feedback": feedback_string,
        "reference_image": REFERENCE_IMAGE,
        "render_image": None,  # filled by caller
    }
    return payload, feedback_string


@mcp.tool()
def score_render(render_path: str) -> dict:
    """Score how close a rendered image is to the target reference image you are
    trying to reproduce.

    THIS IS THE ONLY SANCTIONED WAY TO SCORE A RENDER. Call this tool after every
    render you make. Do not write a Python script, do not import from
    feedback_mcp_server.py, do not search the filesystem for scoring utilities —
    just call this tool with the render's absolute path.

    Returns a visual-similarity signal (lower = closer to the reference). The
    exact form depends on the configured feedback mode.

    Args:
        render_path: Absolute path to the rendered PNG you want scored.

    Returns:
        Dict with the feedback signal and the paths used.
    """
    global _step_index

    rp = Path(render_path).expanduser()
    if not rp.is_absolute():
        rp = Path.cwd() / rp
    rp = rp.resolve()
    if not rp.exists():
        return {"error": f"file not found: {rp}"}

    try:
        results = ensemble.compute_all(REFERENCE_IMAGE, str(rp))
    except Exception as e:
        return {"error": f"failed to score render: {e}"}

    payload, feedback_string = _build_feedback(results)
    payload["render_image"] = str(rp)

    if SCORE_LOG_PATH:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": TASK_ID,
            "step_index": _step_index,
            "feedback_mode": FEEDBACK_MODE,
            "render_path": str(rp),
            "reference_image": REFERENCE_IMAGE,
            "fed_metrics": ensemble.fed_names,
            "heldout_metrics": ensemble.heldout_names,
            "metrics": results,
            "feedback_string": feedback_string,
            # Back-compat key for scripts/visualize_v1_v2.py and friends.
            "dinov3_cosine_distance": results.get("dinov3"),
        }
        try:
            with open(SCORE_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(
                f"[feedback_mcp_server] failed to write {SCORE_LOG_PATH}: {e}",
                file=sys.stderr,
            )

    _step_index += 1
    return payload


if __name__ == "__main__":
    mcp.run()
