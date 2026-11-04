"""blender -b -P control_views.py -- <hero> out.png [views] [centre_mm] [size_mm]

Renders the rig's face controls as markers on the head (bone widgets only draw in the viewport), to check where a
hand-made face (handkeys.py) put them. Labels go in out.txt next to the image.
"""
import math
import os
import subprocess
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402

hero, rest = from_argv()
out = rest[0]
views = rest[1].split(',') if len(rest) > 1 and rest[1] else ['34']
centre_mm = [float(v) for v in rest[2].split(',')] if len(rest) > 2 and rest[2] else [0, 0, 0]
size = float(rest[3]) / 1000 if len(rest) > 3 and rest[3] else 0.3

bpy.ops.wm.open_mainfile(filepath=hero.rig_blend)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE' and 'FACE' in o.pose.bones)
origin = arm.matrix_world @ arm.pose.bones['head'].head
centre = origin + Vector(centre_mm) / 1000
mat = bpy.data.materials.new('marker')
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
em = nt.nodes.new('ShaderNodeEmission')
em.inputs[0].default_value = (1, 0.1, 0.8, 1)
em.inputs[1].default_value = 4
o_ = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(em.outputs[0], o_.inputs[0])
fs = arm.get('face_scale', 1.0)
labels = []
for pb in arm.pose.bones:
    if not (pb.name.startswith('CTRL_') or pb.name == 'FACE'):
        continue
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.004 * fs, location=arm.matrix_world @ pb.head)
    m = bpy.context.object
    m.data.materials.append(mat)
    m.show_in_front = True
    labels.append(f'{pb.name} {tuple(round(v * 1000) for v in (arm.matrix_world @ pb.head - origin))}')

scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 800
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.4, 0.43, 0.48, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.5
scene.world = world
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.type = 'ORTHO'
cam.data.ortho_scale = size
cam.data.clip_end = 100
scene.collection.objects.link(cam)
scene.camera = cam
VIEWS = {'front': (0, 0), 'side': (90, 0), '34': (40, 10), 'below': (25, -35)}
files = []
for v in views:
    yaw, pitch = (math.radians(a) for a in VIEWS[v])
    d = Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    cam.location = centre + d * 3
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    f = out[:-4] + f'_{v}.png'
    scene.render.filepath = f
    bpy.ops.render.render(write_still=True)
    files.append(f)
subprocess.run(['magick'] + files + ['+append', out], check=False)
open(out[:-4] + '.txt', 'w').write('\n'.join(labels) + '\n')
