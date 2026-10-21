"""blender -b <hero>_rig.blend -P hand_views.py -- out_prefix side pose [pose...]
Renders one hand isolated (Mask modifier on hand/finger weights) from palm, back, thumb and pinky side."""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
prefix, side, poses = args[0], args[1], args[2:]
import os
sys.path.insert(0, os.path.dirname(__file__))
import importlib
import hands
arm = next(o for o in bpy.data.objects if 'dlrig' in o)
body0 = bpy.data.objects['Body']
POSES, REPORTS = hands.hand_poses(arm, body0)
for r in REPORTS:
    print('THUMB', r)
bpy.context.view_layer.objects.active = arm
body = bpy.data.objects['Body']
bpy.data.objects['Head'].hide_render = True

groups = [g for g in body.vertex_groups if g.name == f'hand_{side}' or (g.name.startswith('finger_') and g.name.endswith('_' + side))]
gi = {g.index for g in groups}
vg = body.vertex_groups.new(name='HANDMASK')
for v in body.data.vertices:
    w = sum(g.weight for g in v.groups if g.group in gi)
    if w > 0.3:
        vg.add([v.index], 1.0, 'REPLACE')
m = body.modifiers.new('mask', 'MASK')
m.vertex_group = 'HANDMASK'

sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x = sc.render.resolution_y = 360
w = bpy.data.worlds.new('w')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.35, 0.37, 0.4, 1)
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 4
sc.collection.objects.link(sun)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c'))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 0.24
pb = arm.pose.bones
sx = 1 if side == 'L' else -1
views = {'palm': Vector((-sx, 0, 0)), 'back': Vector((sx, 0, 0)), 'thumb': Vector((0, -1, 0)), 'pinky': Vector((0, 1, 0))}
for pose in poses:
    for b in pb:
        if b.name.startswith('finger_') and b.name.endswith('_' + side):
            b.rotation_quaternion = (1, 0, 0, 0)
    if pose != 'Rest':
        for b, q in POSES[pose][side].items():
            pb[b].rotation_mode = 'QUATERNION'
            pb[b].rotation_quaternion = q
    bpy.context.view_layer.update()
    c = pb['hand_' + side].head + (pb['finger_middle_1_' + side].head - pb['hand_' + side].head) * 0.6
    for vn, d in views.items():
        cam.location = c + d * 0.5
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sun.rotation_euler = cam.rotation_euler
        sc.render.filepath = f'{prefix}_{pose.replace(" ", "_")}_{vn}.png'
        bpy.ops.render.render(write_still=True)
