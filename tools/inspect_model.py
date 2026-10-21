"""blender -b -P inspect.py -- in.glb out_dir"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

for o in bpy.context.scene.objects:
    print('OBJ', o.name, o.type, o.parent.name if o.parent else None, tuple(round(v, 3) for v in o.matrix_world.translation),
          len(o.data.vertices) if o.type == 'MESH' else '', [m.name for m in o.data.materials] if o.type == 'MESH' else '')
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
for b in ('head', 'neck_0', 'jaw_0', 'eye_0_L', 'eye_0_R', 'beret'):
    pb = arm.pose.bones[b]
    print('BONE', b, tuple(round(v, 4) for v in arm.matrix_world @ pb.head), tuple(round(v, 4) for v in arm.matrix_world @ pb.tail),
          'parent', pb.parent.name if pb.parent else None)

for o in bpy.context.scene.objects:
    if o.type != 'MESH':
        continue
    groups = {g.index: g.name for g in o.vertex_groups}
    for name in ('head', 'jaw_0', 'eye_0_L', 'eye_0_R'):
        gi = next((i for i, n in groups.items() if n == name), None)
        if gi is None:
            continue
        vs = [v for v in o.data.vertices if any(g.group == gi and g.weight > 0.01 for g in v.groups)]
        if vs:
            ws = [o.matrix_world @ v.co for v in vs]
            lo = Vector(map(min, *ws)) if len(ws) > 1 else ws[0]
            hi = Vector(map(max, *ws)) if len(ws) > 1 else ws[0]
            print('GROUP', o.name, name, len(vs), tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi))
    for mi, m in enumerate(o.data.materials):
        polys = [p for p in o.data.polygons if p.material_index == mi]
        vi = {i for p in polys for i in p.vertices}
        if not vi:
            continue
        ws = [o.matrix_world @ o.data.vertices[i].co for i in vi]
        lo = Vector(map(min, *ws))
        hi = Vector(map(max, *ws))
        print('MAT', o.name, m.name, len(polys), len(vi), tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi))

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 768
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.55, 0.6, 1)
scene.world = world
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3
sun.rotation_euler = (math.radians(60), 0, math.radians(20))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 85
scene.collection.objects.link(cam)
scene.camera = cam
head = arm.matrix_world @ arm.pose.bones['head'].head
eyes = (arm.matrix_world @ arm.pose.bones['eye_0_L'].head + arm.matrix_world @ arm.pose.bones['eye_0_R'].head) / 2
fwd = eyes - head
fwd.z = 0
fwd.normalize()
print('FWD', tuple(fwd))
centre = Vector((eyes.x, eyes.y, eyes.z - 0.03))
for name, d, dist in (('front', fwd, 0.7), ('side', (fwd + fwd.cross(Vector((0, 0, 1))) * 1.2).normalized(), 0.7), ('body', fwd, 4.5)):
    c = centre if name != 'body' else Vector((eyes.x, eyes.y, eyes.z * 0.55))
    cam.location = c + d * dist
    cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = os.path.join(out, name + '.png')
    bpy.ops.render.render(write_still=True)
