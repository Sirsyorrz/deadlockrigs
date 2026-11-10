"""blender -b -P head_views.py -- in.blend out_dir [size_mm] [cx,cy,cz] [keys]

Orthographic front and side renders of a hero's head with a millimetre grid relative to the head bone (Blender
world axes: +x the hero's left, -y forward, +z up), plus a three-quarter view. For placing hand-made face handles.
size_mm is the frame width (default 260); cx,cy,cz the frame centre in head-relative mm (default 0,0,0).
keys: key[=value],... applied to every keyed mesh before rendering.
"""
import math
import re
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
src, out_dir = argv[0], argv[1]
size = float(argv[2]) / 1000 if len(argv) > 2 and argv[2] else None
off = Vector([float(v) / 1000 for v in argv[3].split(',')]) if len(argv) > 3 and argv[3] else None
keyvals = dict((kv.split('=') + ['1'])[:2] for kv in argv[4].split(',')) if len(argv) > 4 and argv[4] else {}

bpy.ops.wm.open_mainfile(filepath=src)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
head = arm.matrix_world @ arm.pose.bones[os.environ.get('HEAD', 'head')].head
if size is None or off is None:
    # frame the skin the head bone and its children carry
    names = {b.name for b in arm.pose.bones['head'].children_recursive
             if re.match(r'(jaw|eye|nose|lid|lip|brow|tongue|cheek|mouth|snout)', b.name)} | {'head'}
    pts = []
    for o in scene.objects:
        if o.type != 'MESH' or o.hide_render or o.parent != arm:
            continue
        gi = {g.index for g in o.vertex_groups if g.name in names}
        pts += [o.matrix_world @ v.co for v in o.data.vertices
                if sum(g.weight for g in v.groups if g.group in gi) > 0.5]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size = size or max(hi - lo) * 1.15
    off = off if off is not None else (lo + hi) / 2 - head
    print(f'head skin bbox (mm from head bone): {tuple(round(v * 1000) for v in lo - head)} '
          f'{tuple(round(v * 1000) for v in hi - head)}')
centre = head + off
if os.environ.get('ONLY'):
    # ONLY=Head: just that mesh, for faces under hats and hair (Wraith)
    for o in scene.objects:
        if o.type == 'MESH':
            # rig bodygroup/parts drivers would show it again
            o.driver_remove('hide_render')
            o.hide_render = o.name != os.environ['ONLY']
for o in scene.objects:
    if o.type == 'MESH' and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks[1:]:
            kb.value = float(keyvals.get(kb.name, 0))

scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 1024
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.35, 0.38, 0.42, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.5
scene.world = world
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 2.5
sun.rotation_euler = (math.radians(50), 0, math.radians(25))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
scene.collection.objects.link(cam)
scene.camera = cam

grid_mat = bpy.data.materials.new('grid')
grid_mat.use_nodes = True
nt = grid_mat.node_tree
nt.nodes.clear()
em = nt.nodes.new('ShaderNodeEmission')
em.inputs[0].default_value = (1, 0.2, 0.9, 1)
em.inputs[1].default_value = 1.5
outn = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(em.outputs[0], outn.inputs[0])


def grid(axes, depth_axis, depth, text_rot):
    """Lines every step in the plane of axes (two world axis indices) at depth along depth_axis, labelled in mm."""
    half = size / 2
    step = next(s for s in (0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2) if size / s <= 30)
    verts, faces = [], []
    objs = []
    lo = [centre[a] - half for a in axes]
    hi = [centre[a] + half for a in axes]

    def point(u, v):
        p = [0.0, 0.0, 0.0]
        p[axes[0]], p[axes[1]], p[depth_axis] = u, v, depth
        return p

    def rect(u0, v0, u1, v1):
        i = len(verts)
        verts.extend([point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)])
        faces.append((i, i + 1, i + 2, i + 3))

    def label(text, u, v):
        cu = bpy.data.curves.new('label', 'FONT')
        cu.body = text
        cu.size = size / 45
        t = bpy.data.objects.new('label', cu)
        t.data.materials.append(grid_mat)
        t.location = point(u, v)
        t.rotation_euler = text_rot
        objs.append(t)

    for i in (0, 1):
        a = axes[i]
        for k in range(math.ceil((lo[i] - head[a]) / step), math.floor((hi[i] - head[a]) / step) + 1):
            c = head[a] + k * step
            w = size / (700 if k % 5 == 0 else 2000)
            if i == 0:
                rect(c - w, lo[1], c + w, hi[1])
                if k % 2 == 0:
                    label(str(round(k * step * 1000)), c + size / 300, lo[1] + size / 100)
            else:
                rect(lo[0], c - w, hi[0], c + w)
                if k % 2 == 0:
                    label(str(round(k * step * 1000)), lo[0] + size / 100, c + size / 300)
    me = bpy.data.meshes.new('grid')
    me.from_pydata(verts, [], faces)
    g = bpy.data.objects.new('grid', me)
    g.data.materials.append(grid_mat)
    objs.append(g)
    for o in objs:
        scene.collection.objects.link(o)
    return objs


def shoot(name, loc, rot, ortho=True, with_grid=None):
    cam.location = loc
    cam.rotation_euler = rot
    cam.data.type = 'ORTHO' if ortho else 'PERSP'
    cam.data.ortho_scale = size
    cam.data.lens = 85
    cam.data.clip_end = 100
    objs = grid(*with_grid) if with_grid else []
    scene.render.filepath = os.path.join(out_dir, name + '.png')
    bpy.ops.render.render(write_still=True)
    for o in objs:
        bpy.data.objects.remove(o)


os.makedirs(out_dir, exist_ok=True)
facing_cam = (math.radians(90), 0, 0)
side_cam = (math.radians(90), 0, math.radians(90))
# the grids sit between the camera and the head, labels turned to the camera
shoot('front', centre + Vector((0, -1, 0)), facing_cam, with_grid=((0, 2), 1, centre.y - 0.25, facing_cam))
shoot('side', centre + Vector((1, 0, 0)), side_cam, with_grid=((1, 2), 0, centre.x + 0.25, side_cam))
d = Vector((math.sin(math.radians(35)), -math.cos(math.radians(35)), 0.1))
shoot('three_quarter', centre + d * size * 3.2, (-d).to_track_quat('-Z', 'Y').to_euler(), ortho=False)
