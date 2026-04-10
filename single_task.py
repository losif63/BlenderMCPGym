from argparse import ArgumentParser
import os
import shutil
import subprocess


def main(args):
    blend_file = os.path.expanduser(f"~/Desktop/Research/BlenderMCPGym/bench_data/{args.task_name}/blender_file.blend")
    edit_file = os.path.expanduser(f"~/Desktop/Research/BlenderMCPGym/bench_data/{args.task_name}/edit.blend")
    shutil.copy2(blend_file, edit_file)
    subprocess.Popen([
        "/Applications/Blender.app/Contents/MacOS/blender",
        edit_file,
        "--python-expr",
        "import bpy; bpy.app.timers.register(lambda: bpy.ops.blendermcp.start_server(), first_interval=1.0)"
    ])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default="blendshape1")
    args = parser.parse_args()

    main(args)

