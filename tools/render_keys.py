"""blender -b -P render_keys.py -- in.blend|glb out_dir [key[=value],...] [side]

Renders a face close-up per shape key (plus neutral.png) and a contact sheet with ImageMagick if present.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
src, out_dir = argv[0], argv[1]
wanted = argv[2].split(',') if len(argv) > 2 and argv[2] else []
angle = float(argv[3]) if len(argv) > 3 else 0.0
focus = argv[4] if len(argv) > 4 else 'face'

if src.endswith('.blend'):
    bpy.ops.wm.open_mainfile(filepath=src)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
keyed = [o for o in scene.objects if o.type == 'MESH' and o.data.shape_keys]
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
eyes = (arm.matrix_world @ arm.pose.bones['eye_0_L'].head + arm.matrix_world @ arm.pose.bones['eye_0_R'].head) / 2
centre = Vector((0, eyes.y, eyes.z - 0.04))
dist = 0.6
if focus in ('L', 'R'):
    centre = arm.matrix_world @ arm.pose.bones['eye_0_' + focus].head
    dist = 0.2
elif focus == 'mouth':
    centre = Vector((0, eyes.y, eyes.z - 0.082))
    dist = 0.25
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 512
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.55, 0.6, 1)
scene.world = world
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3
sun.data.use_shadow = os.environ.get('SHADOW', '1') == '1'
sun.rotation_euler = (math.radians(60), 0, math.radians(20))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 85
scene.collection.objects.link(cam)
scene.camera = cam
a = math.radians(angle)
cam.location = centre + Vector((math.sin(a), -math.cos(a), 0.05)) * dist
cam.rotation_euler = (centre - cam.location).to_track_quat('-Z', 'Y').to_euler()
hair = [o for o in scene.objects if o.type == 'MESH' and o not in keyed]
for o in (hair if os.environ.get('HAIR') != '1' else []):
    o.hide_render = True

names = [kb.name for o in keyed for kb in o.data.shape_keys.key_blocks[1:]]
targets = [w.split('=') for w in wanted] if wanted else [[n] for n in names]
os.makedirs(out_dir, exist_ok=True)


def render(values: dict, file: str) -> None:
    for o in keyed:
        for kb in o.data.shape_keys.key_blocks[1:]:
            kb.value = values.get(kb.name, 0.0)
    scene.render.filepath = os.path.join(out_dir, file)
    bpy.ops.render.render(write_still=True)


render({}, 'neutral.png')
for t in targets:
    if '+' in t[0]:
        vals = {k: float(t[1]) if len(t) > 1 else 1.0 for k in t[0].split('+')}
        render(vals, '='.join(t) + '.png')
    else:
        render({t[0]: float(t[1]) if len(t) > 1 else 1.0}, '='.join(t) + '.png')
