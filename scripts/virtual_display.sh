Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &

DISPLAY=:99 ./infinigen/blender/blender --background --python-expr "
import bpy
bpy.ops.preferences.addon_enable(module='addon')
bpy.ops.wm.save_userpref()
"