import os
import subprocess
from argparse import ArgumentParser
from util import setup_blendermcp, build_prompt, save_blender_file
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

def main(args):
    images = []
    if args.all:
        images = os.listdir('images/')
    else:
        images = [args.image]
    
    claude = "claude" if "deepseek" not in args.model else "claude-ds"
    model_id = MODEL_IDS[args.model]
    
    for image in images:
        blender_proc = setup_blendermcp(False)
        if blender_proc is None:
            continue 
        prompt = build_prompt(image_name=image, plat="claudecode", model_name=args.model)
        image_name = Path(image).stem
        blend_path = f"recreation/{image_name}/claudecode/{args.model}/blender_file.blend"
        os.makedirs(os.path.dirname(blend_path), exist_ok=True)
        claude_proc = subprocess.Popen(
            args=[claude, "--model", model_id, "-p", prompt],
            env={**os.environ, "BLENDER_MCP_SESSION_DIR": os.path.join(os.getcwd(), os.path.dirname(blend_path))}
        )
        claude_proc.wait()
        save_blender_file(blend_path)
        blender_proc.terminate()

# Configurable variables:
# Model type
# Image type -> Specific image or all
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='opus-4.7', choices=list(MODEL_IDS.keys()), help="Specify model to use")
    parser.add_argument('--all', default=False, required=False, action='store_true', help='Specify whether to experiment on all images')
    parser.add_argument('--image', type=str, default=None, required=False, help='Specify image to run on')
    args = parser.parse_args()
    
    if not args.all and not args.image:
        print("Target unspecified. Please provide --all option or --image option.")
    else:
        main(args)

