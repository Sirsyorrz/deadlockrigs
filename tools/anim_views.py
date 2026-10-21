"""blender -b <hero>_rig.blend -P anim_views.py -- out_prefix clip frames(comma) [view] [physics 0/1] [loop 0/1]
Loads a game clip through the rig's operator, bakes physics, renders the chosen frames."""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
prefix, clip, frames = args[0], args[1], [int(x) for x in args[2].split(',')]
view = args[3] if len(args) > 3 else 'side'
phys = (args[4] if len(args) > 4 else '1') == '1'
loop = (args[5] if len(args) > 5 else '1') == '1'
t = bpy.data.texts['dlrig_ui.py']
exec(compile(t.as_string(), t.name, 'exec'), {'__name__': 'dlrig_ui'})
arm = next(o for o in bpy.data.objects if 'dlrig' in o)
bpy.context.view_layer.objects.active = arm
sc = bpy.context.scene
sc.dlrig_phys_auto = False
sc.dlrig_phys_loop = loop
bpy.ops.dlrig.game_anim(clip=clip, mode='NEW')
if phys:
    print(bpy.ops.dlrig.bake_physics())
for c in arm.data.collections_all:
    c.is_visible = False
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'TEXTURE'
sc.render.resolution_x, sc.render.resolution_y = 420, 560
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 50
dirs = {'hips': Vector((-0.6, -1, 0.0)), 'hair': Vector((-0.3, 1, 0.2)), 'rside': Vector((-1, 0, 0.05)), 'side': Vector((1, 0, 0.05)), 'front': Vector((0.2, -1, 0.1)), 'back': Vector((0.2, 1, 0.1)),
        'threeq': Vector((0.8, -0.8, 0.1))}
for f in frames:
    sc.frame_set(f)
    root = arm.matrix_world @ arm.pose.bones['pelvis'].head
    c = Vector((root.x, root.y, root.z + {'hips': -0.39, 'hair': 0.36}.get(view, -0.24)))
    cam.location = c + dirs[view].normalized() * {'hips': 2.0, 'hair': 1.6}.get(view, 3.4)
    cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'{prefix}_{f:03d}.png'
    bpy.ops.render.render(write_still=True)
