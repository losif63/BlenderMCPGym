"""
v2 of the image-recreation experiment runner.

Differences vs `run_claudecode.py`:
  * Launches Blender with PolyHaven asset downloads enabled (via util_v2.setup_blendermcp).
  * Spins up a per-image MCP server (`feedback_mcp_server.py`) that exposes a
    `score_render(render_path)` tool returning the DINOv3 cosine distance between
    the agent's render and the reference image. The server is configured via the
    REFERENCE_IMAGE env var and wired in through Claude Code's `--mcp-config` flag.
  * Outputs land under `recreation/{image}/claudecode_v2/{model}/` so v1 and v2
    results live side-by-side. `render_recreations.py` will pick up the v2 blend
    files automatically; the eval scripts need their hardcoded "claudecode" path
    swapped to "claudecode_v2" to score v2 renders.
"""

import glob
import json
import os
import re
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from util_v2 import build_prompt, infer_prompt_type, save_blender_file, setup_blendermcp

MODEL_IDS = {
    "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6",
    "opus-4.5": "claude-opus-4-5",
    "opus-4.1": "claude-opus-4-1",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5",
    "haiku-4.5": "claude-haiku-4-5",
    "deepseek-v4-flash": "opus",
    "deepseek-v4-pro": "haiku",
}

GROUPED_IMAGE_PATTERN = re.compile(r'^(beginner|advanced)\d+\.')

PLATFORM = "claudecode_v2"

# Each feedback mode writes to its own platform dir so runs don't overwrite each
# other. dinov3_only keeps the bare "claudecode_v2" for back-compat with prior
# control runs and existing eval --platform values.
PLATFORM_BY_MODE = {
    "dinov3_only": "claudecode_v2",
    "ensemble_avg": "claudecode_v2_avg",
    "ensemble_vector": "claudecode_v2_vector",
}

FEEDBACK_SERVER = Path(__file__).resolve().parent / "feedback_mcp_server.py"

# The default ensemble pulls in timm (SigLIP2/ConvNeXt-v2/Inception), dreamsim,
# and lpips on top of the original torch/transformers/mcp set. Probing for them
# here surfaces a missing dep before Blender and the agent are launched.
FEEDBACK_DEPS = (
    "torch",
    "transformers",
    "mcp.server.fastmcp",
    "timm",
    "dreamsim",
    "lpips",
)


def resolve_feedback_python() -> str:
    """Find a Python interpreter that can run feedback_mcp_server.py.

    Preference order:
      1. $FEEDBACK_PYTHON (explicit override)
      2. sys.executable (the Python running this script)
      3. Any conda env Python under ~/anaconda3 or ~/miniconda3

    Probes each candidate by importing torch/transformers/mcp; first one that
    succeeds wins. Raises if none work, with instructions for the fix.
    """
    candidates: list[str] = []
    explicit = os.environ.get("FEEDBACK_PYTHON")
    if explicit:
        candidates.append(explicit)
    candidates.append(sys.executable)
    for pat in (
        "~/anaconda3/envs/*/bin/python",
        "~/miniconda3/envs/*/bin/python",
        "~/opt/anaconda3/envs/*/bin/python",
    ):
        candidates.extend(sorted(glob.glob(os.path.expanduser(pat))))

    seen: set[str] = set()
    probe_code = "import " + ", ".join(FEEDBACK_DEPS)
    for cand in candidates:
        if cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        result = subprocess.run(
            [cand, "-c", probe_code], capture_output=True, timeout=20
        )
        if result.returncode == 0:
            if cand != sys.executable:
                print(
                    f"[run_claudecode_v2] sys.executable ({sys.executable}) is "
                    f"missing one of {FEEDBACK_DEPS}; using {cand} for the "
                    f"feedback MCP server."
                )
            return cand

    raise RuntimeError(
        f"Could not find a Python interpreter with {FEEDBACK_DEPS} installed. "
        f"Activate the env that has torch/transformers/mcp before running this "
        f"script, or set FEEDBACK_PYTHON=/path/to/python."
    )


def write_mcp_config(
    config_path: str,
    reference_image_path: str,
    score_log_path: str,
    feedback_python: str,
    feedback_mode: str,
    task_id: str,
    fed_metrics=None,
    heldout_metrics=None,
) -> None:
    """Write a Claude Code --mcp-config JSON that registers the feedback server.

    Uses `feedback_python` so the server runs under the env that has
    torch/transformers/timm/dreamsim/lpips/mcp installed. Merged with the global
    MCP config by default (so the existing blender-mcp config is preserved).

    Env wired into the server:
      REFERENCE_IMAGE  reference to score against.
      SCORE_LOG_PATH   JSONL trajectory of every score_render call.
      FEEDBACK_MODE    dinov3_only | ensemble_avg | ensemble_vector (selects the
                       signal fed to the agent; held-out metrics are always
                       logged but never fed).
      TASK_ID          logged with each record.
      FED_METRICS / HELDOUT_METRICS  optional comma lists overriding the server
                       defaults (omitted here => server defaults apply).
    """
    env = {
        "REFERENCE_IMAGE": reference_image_path,
        "SCORE_LOG_PATH": score_log_path,
        "FEEDBACK_MODE": feedback_mode,
        "TASK_ID": task_id,
    }
    if fed_metrics:
        env["FED_METRICS"] = fed_metrics
    if heldout_metrics:
        env["HELDOUT_METRICS"] = heldout_metrics
    config = {
        "mcpServers": {
            "render_feedback": {
                "command": feedback_python,
                "args": [str(FEEDBACK_SERVER)],
                "env": env,
            }
        }
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main(args):
    if args.all:
        images = [
            f for f in os.listdir('images/') if GROUPED_IMAGE_PATTERN.match(f)
        ]
    else:
        images = [args.image]

    is_deepseek = "deepseek" in args.model
    model_id = MODEL_IDS[args.model]
    project_dir = os.getcwd()
    feedback_python = resolve_feedback_python()
    platform = PLATFORM_BY_MODE[args.feedback_mode]

    for image in images:
        prompt_type = args.prompt_type or infer_prompt_type(image)
        if prompt_type is None:
            print(f"WARNING: cannot determine prompt type for '{image}', skipping.")
            continue

        image_stem = Path(image).stem
        ref_image_path = os.path.join(project_dir, "images", image)
        base_path = f"recreation/{image_stem}/{platform}/{args.model}"
        blend_path = f"{base_path}/blender_file.blend"
        os.makedirs(base_path, exist_ok=True)

        mcp_config_path = f"{base_path}/mcp_config.json"
        score_log_path = os.path.join(project_dir, base_path, "score_log.jsonl")
        write_mcp_config(
            mcp_config_path,
            ref_image_path,
            score_log_path,
            feedback_python,
            feedback_mode=args.feedback_mode,
            task_id=image_stem,
            fed_metrics=args.fed_metrics,
            heldout_metrics=args.heldout_metrics,
        )

        blender_proc = setup_blendermcp(virtual_display=False, use_polyhaven=True)
        if blender_proc is None:
            continue

        prompt = build_prompt(
            image_name=image,
            plat=platform,
            model_name=args.model,
            prompt_type=prompt_type,
            feedback_mode=args.feedback_mode,
        )

        if is_deepseek:
            shell_cmd = (
                f"source ~/.zshrc && claude-ds --model {model_id} "
                f"--mcp-config {shlex.quote(mcp_config_path)} "
                f"--output-format json -p {shlex.quote(prompt)}"
            )
            popen_args = ["bash", "-c", shell_cmd]
        else:
            popen_args = [
                "claude",
                "--model", model_id,
                "--mcp-config", mcp_config_path,
                "--output-format", "json",
                "-p", prompt,
            ]

        claude_proc = subprocess.Popen(
            args=popen_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "BLENDER_MCP_SESSION_DIR": os.path.join(project_dir, base_path),
            },
        )
        stdout, _ = claude_proc.communicate()
        save_blender_file(blend_path)
        blender_proc.terminate()

        try:
            result = json.loads(stdout.decode())
            print(list(result.keys()))
            usage_path = f"{base_path}/usage.json"
            with open(usage_path, "w") as f:
                json.dump({"model": model_id, **result}, f, indent=2)
            session_id = result.get("session_id")
            if session_id:
                with open(f"{base_path}/session_id.txt", "w") as f:
                    f.write(session_id)
                print(f"Session ID: {session_id}")
                print(
                    f"Logs: ~/.claude/projects/{Path(os.getcwd()).name}/{session_id}.jsonl"
                )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"WARNING: could not parse claude output for token usage: {e}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='opus-4.7',
        choices=list(MODEL_IDS.keys()), help="Specify model to use",
    )
    parser.add_argument(
        '--all', default=False, required=False, action='store_true',
        help='Specify whether to experiment on all images',
    )
    parser.add_argument(
        '--image', type=str, default=None, required=False,
        help='Specify image to run on',
    )
    parser.add_argument(
        '--prompt-type', type=str, default=None,
        choices=['beginner', 'advanced'],
        help='System prompt to use (inferred from image name if omitted)',
    )
    parser.add_argument(
        '--feedback-mode', type=str, default='dinov3_only',
        choices=['dinov3_only', 'ensemble_avg', 'ensemble_vector'],
        help=(
            'Visual signal fed to the agent. dinov3_only = original control; '
            'ensemble_avg = mean of FED-metric distances; ensemble_vector = '
            'per-metric breakdown. Held-out metrics are always logged, never fed.'
        ),
    )
    parser.add_argument(
        '--fed-metrics', type=str, default=None,
        help='Override FED metric set, comma-separated (default: dinov3,siglip2,convnextv2,dreamsim).',
    )
    parser.add_argument(
        '--heldout-metrics', type=str, default=None,
        help='Override held-out metric set, comma-separated (default: inception,lpips).',
    )
    args = parser.parse_args()

    if not args.all and not args.image:
        print("Target unspecified. Please provide --all option or --image option.")
    else:
        main(args)
