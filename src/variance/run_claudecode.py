import json
import os
import re
import subprocess
from argparse import ArgumentParser
from util import setup_blendermcp, build_prompt, save_blender_file, infer_prompt_type
from pathlib import Path

MODEL_IDS = {
    "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6",
    "opus-4.5": "claude-opus-4-5",
    "opus-4.1": "claude-opus-4-1",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5",
    "haiku-4.5": "claude-haiku-4-5",
    "deepseek-v4-flash": "opus",
    "deepseek-v4-pro": "haiku"
}

GROUPED_IMAGE_PATTERN = re.compile(r'^(beginner|advanced)\d+\.')

def main(args):
    images = []
    if args.all:
        images = [f for f in os.listdir('images/') if GROUPED_IMAGE_PATTERN.match(f)]
    else:
        images = [args.image]

    claude = "claude" if "deepseek" not in args.model else "claude-ds"
    model_id = MODEL_IDS[args.model]

    for image in images:
        prompt_type = args.prompt_type or infer_prompt_type(image)
        if prompt_type is None:
            print(f"WARNING: cannot determine prompt type for '{image}', skipping.")
            continue
        blender_proc = setup_blendermcp(False)
        if blender_proc is None:
            continue
        prompt = build_prompt(image_name=image, plat="claudecode", model_name=args.model, prompt_type=prompt_type)
        image_name = Path(image).stem
        blend_path = f"recreation/{image_name}/claudecode/{args.model}/blender_file.blend"
        os.makedirs(os.path.dirname(blend_path), exist_ok=True)
        claude_proc = subprocess.Popen(
            args=[claude, "--model", model_id, "--output-format", "json", "-p", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "BLENDER_MCP_SESSION_DIR": os.path.join(os.getcwd(), os.path.dirname(blend_path))}
        )
        stdout, _ = claude_proc.communicate()
        save_blender_file(blend_path)
        blender_proc.terminate()

        try:
            result = json.loads(stdout.decode())
            print(list(result.keys()))
            base_path = f"recreation/{image_name}/claudecode/{args.model}"
            usage_path = f"{base_path}/usage.json"
            with open(usage_path, "w") as f:
                json.dump({
                    "model": model_id,
                    **result
                }, f, indent=2)
            session_id = result.get("session_id")
            if session_id:
                with open(f"{base_path}/session_id.txt", "w") as f:
                    f.write(session_id)
                print(f"Session ID: {session_id}")
                print(f"Logs: ~/.claude/projects/{Path(os.getcwd()).name}/{session_id}.jsonl")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"WARNING: could not parse claude output for token usage: {e}")

# Configurable variables:
# Model type
# Image type -> Specific image or all
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='opus-4.7', choices=list(MODEL_IDS.keys()), help="Specify model to use")
    parser.add_argument('--all', default=False, required=False, action='store_true', help='Specify whether to experiment on all images')
    parser.add_argument('--image', type=str, default=None, required=False, help='Specify image to run on')
    parser.add_argument('--prompt-type', type=str, default=None, choices=['beginner', 'advanced'], help='System prompt to use (inferred from image name if omitted)')
    args = parser.parse_args()
    
    if not args.all and not args.image:
        print("Target unspecified. Please provide --all option or --image option.")
    else:
        main(args)

