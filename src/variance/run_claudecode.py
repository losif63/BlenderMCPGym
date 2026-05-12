import os, subprocess
from argparse import ArgumentParser

SYSTEM_PROMPT = """
You are an expert 3D artist working in Blender through the BlenderMCP interface.
Your task is to reconstruct the scene in the provided reference image as a 3D scene in Blender, as faithfully as the available tools allow.
"""


def main(args):
    images = []
    if args.all:
        images = os.listdir('images/')
    else:
        images = [args.image]
    print(images)

# Configurable variables:
# Model type
# Image type -> Specific image or all
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='sonnet', choices=['sonnet', 'opus', 'deepseek'], help="Specify model to use")
    parser.add_argument('--all', default=False, required=False, action='store_true', help='Specify whether to experiment on all images')
    parser.add_argument('--image', type=str, default=None, required=False, help='Specify image to run on')
    args = parser.parse_args()
    
    if not args.all and not args.image:
        print("Target unspecified. Please provide --all option or --image option.")
    else:
        main(args)

