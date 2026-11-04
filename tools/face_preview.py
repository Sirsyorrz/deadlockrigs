"""blender -b -P face_preview.py -- <hero> out_dir [keys] [views] [centre_mm] [size_mm] [regions]

Renders a hand-made face (handkeys.py) straight from data/keys.npz on the base build's head, without rebuilding:
one image per key (key or key=value, '+' joins keys into one image) per view, and a contact sheet (sheet.png).
views: side, front, 34 (three-quarter), below; centre/size frame the head in mm from the head bone.
regions (comma list): heat maps of data/regions.npz (handkeys.py --regions) instead of keys.
"""
import math
import os
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402

hero, rest = from_argv()
out_dir = rest[0]
wanted = rest[1].split(',') if len(rest) > 1 and rest[1] else []
views = rest[2].split(',') if len(rest) > 2 and rest[2] else ['34']
centre_mm = [float(v) for v in rest[3].split(',')] if len(rest) > 3 and rest[3] else [0, 0, 0]
size = float(rest[4]) / 1000 if len(rest) > 4 and rest[4] else 0.3
regions = rest[5].split(',') if len(rest) > 5 and rest[5] else []

bpy.ops.wm.open_mainfile(filepath=hero.base_blend)
scene = bpy.context.scene
head = bpy.data.objects[hero.face['mesh']]
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
origin = arm.matrix_world @ arm.data.bones[hero.face.get('bone', 'head')].head_local
centre = origin + Vector(centre_mm) / 1000
me = head.data
n = len(me.vertices)
base = np.empty(n * 3, np.float32)
(me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices).foreach_get('co', base)
base = base.reshape(-1, 3)
inv = np.array(head.matrix_world.inverted().to_3x3())
keys = np.load(os.path.join(hero.data, 'keys.npz'))
if not me.shape_keys:
    head.shape_key_add(name='Basis', from_mix=False)
for name in keys.files:
    kb = me.shape_keys.key_blocks.get(name) or head.shape_key_add(name=name, from_mix=False)
    kb.data.foreach_set('co', (base + keys[name][:n] @ inv.T).astype(np.float32).ravel())

scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 640
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.4, 0.43, 0.48, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.5
scene.world = world
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 2.5
sun.rotation_euler = (math.radians(50), 0, math.radians(25))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.type = 'ORTHO'
cam.data.ortho_scale = size
cam.data.clip_end = 100
scene.collection.objects.link(cam)
scene.camera = cam
VIEWS = {'front': (0, 0), 'side': (90, 0), '34': (40, 10), 'below': (25, -35), 'back34': (140, 10)}


def aim(view):
    yaw, pitch = (math.radians(a) for a in VIEWS[view])
    d = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    cam.location = centre + d * 2
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()


if regions:
    heat = bpy.data.materials.new('heat')
    heat.use_nodes = True
    nt = heat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'heat'
    bsdf = nt.nodes.new('ShaderNodeBsdfDiffuse')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(attr.outputs['Color'], bsdf.inputs['Color'])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    head.data.materials.clear()
    head.data.materials.append(heat)
    # REGIONS=facerig shows the bone face's weights (data/facerig.npz) instead
    regs = np.load(os.path.join(hero.data, os.environ.get('REGIONS', 'regions') + '.npz'))
    col = me.color_attributes.new('heat', 'FLOAT_COLOR', 'POINT')

os.makedirs(out_dir, exist_ok=True)
shots = []


def shoot(label, values=None, region=None):
    if region is not None:
        w = regs[region][:n]
        c = np.stack([w, 0.15 + 0 * w, 1 - w, np.ones_like(w)], 1).astype(np.float32)
        col.data.foreach_set('color', c.ravel())
        me.update()
    for kb in (me.shape_keys.key_blocks[1:] if me.shape_keys else []):
        kb.value = (values or {}).get(kb.name, 0.0)
    for v in views:
        aim(v)
        f = os.path.join(out_dir, f'{label}_{v}.png'.replace('=', '-'))
        scene.render.filepath = f
        bpy.ops.render.render(write_still=True)
        shots.append((label, f))


if regions:
    for r in regions:
        shoot(r, region=r)
else:
    shoot('neutral')
    for w in wanted or [k for k in keys.files if not k.endswith('_mid') and k.count('_') == 0]:
        name, _, val = w.partition('=')
        shoot(w, {k: float(val or 1) for k in name.split('+')})
try:
    cmd = ['magick', 'montage'] + [a for label, f in shots for a in ('-label', label, f)] + \
        ['-tile', f'{len(views) * (4 if len(views) == 1 else 2)}x', '-geometry', '320x320+2+2',
         os.path.join(out_dir, 'sheet.png')]
    subprocess.run(cmd, check=True)
except (OSError, subprocess.CalledProcessError) as e:
    print('no contact sheet:', e)
