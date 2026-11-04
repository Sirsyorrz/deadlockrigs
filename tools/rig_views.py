"""blender -b -P rig_views.py -- <hero> out.png [views] [centre_mm] [size_mm] [collections] [pose.py] [hide]

Renders the rig as the viewport shows it: shaded meshes with the visible bone widgets drawn over them in their bone
colours (widgets only draw in the viewport, so they are rebuilt as tubes). For review sheets of the controls.
views: front, side, back, 34, below, top, rside, r34 (comma list, one image each, joined side by side)
centre_mm / size_mm: frame, mm from the head bone; collections: bone collections to draw (comma list, default the
ones shown); pose.py: a script run first with `arm` and `pb` set, to pose the rig, or defining POSES
(label -> function) for a row per pose (COLS=n lays the rows out n wide); hide: 1 to leave the widgets out.
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
arg = lambda i: rest[i] if len(rest) > i and rest[i] else None  # noqa: E731
views = (arg(1) or '34').split(',')
centre_mm = [float(v) for v in (arg(2) or '0,0,0').split(',')]
size = float(arg(3) or 300) / 1000
want = arg(4).split(',') if arg(4) else None
pose_script = arg(5)
no_widgets = arg(6) == '1'

bpy.ops.wm.open_mainfile(filepath=hero.rig_blend)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE' and 'dlrig' in o)
pb = arm.pose.bones
POSES = {'': None}
# the rig's switches (IK/FK, parents, bodygroups) as built, restored before each pose
SWITCHES = {k: arm[k] for k in arm.keys() if isinstance(arm[k], (int, float)) and not isinstance(arm[k], bool)}
if pose_script:
    ns = {'arm': arm, 'pb': pb, 'bpy': bpy, 'Vector': Vector, 'math': math}
    exec(open(pose_script).read(), ns)
    # a script with POSES (label -> function posing the rig) renders a row per pose
    POSES = ns.get('POSES', POSES)
bpy.context.view_layer.update()
origin = arm.matrix_world @ arm.data.bones['head'].head_local
centre = origin + Vector(centre_mm) / 1000

colls = arm.data.collections_all
shown = {c.name for c in colls if (c.name in want if want else c.is_visible)}
theme = bpy.context.preferences.themes[0].bone_color_sets
mats = {}


def material(rgb):
    key = tuple(round(c, 3) for c in rgb)
    if key not in mats:
        m = bpy.data.materials.new('wire')
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs[0].default_value = (*rgb, 1)
        em.inputs[1].default_value = 2.5
        o_ = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(em.outputs[0], o_.inputs[0])
        mats[key] = m
    return mats[key]


def bone_rgb(b):
    pal = b.color.palette
    if pal.startswith('THEME'):
        return tuple(theme[int(pal[5:]) - 1].normal)
    return (0.6, 0.6, 0.6)


tube = size / 900
wires = bpy.data.collections.new('wires')
scene.collection.children.link(wires)


def draw_widgets():
    for ob in list(wires.objects):
        bpy.data.objects.remove(ob)
    if no_widgets:
        return
    for p in pb:
        b = arm.data.bones[p.name]
        if p.custom_shape is None or not any(c.name in shown for c in b.collections):
            continue
        if getattr(p, 'hide', False) or b.hide:
            continue
        me = p.custom_shape.data
        m = arm.matrix_world @ p.matrix
        cu = bpy.data.curves.new('w', 'CURVE')
        cu.dimensions = '3D'
        cu.bevel_depth = tube
        for e in me.edges:
            sp = cu.splines.new('POLY')
            sp.points.add(1)
            for k, vi in enumerate(e.vertices):
                sp.points[k].co = (*(m @ me.vertices[vi].co), 1)
        ob = bpy.data.objects.new('w_' + p.name, cu)
        ob.data.materials.append(material(bone_rgb(p)))
        wires.objects.link(ob)


arm.hide_render = True

scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = int(os.environ.get('RES', 800))
w = bpy.data.worlds.new('w')
scene.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.2, 0.21, 0.23, 1)
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3.5
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = size
cam.data.clip_end = 50
DIRS = {'front': Vector((0, -1, 0)), 'side': Vector((1, 0, 0)), 'back': Vector((0, 1, 0)),
        '34': Vector((0.7, -1, 0.25)), 'below': Vector((0.3, -1, -0.8)), 'top': Vector((0.2, -0.6, 1)),
        'rside': Vector((-1, 0, 0)), 'r34': Vector((-0.7, -1, 0.25))}


def reset():
    for p in pb:
        p.location = (0, 0, 0)
        p.rotation_quaternion = (1, 0, 0, 0)
        p.rotation_euler = (0, 0, 0)
        p.scale = (1, 1, 1)
    for n, v in arm.get('prop_pose', {}).items():
        v = list(v)
        pb[n].location, pb[n].rotation_quaternion, pb[n].scale = v[0:3], v[3:7], v[7:10]
    if 'FACE' in pb:
        for k in pb['FACE'].keys():
            if isinstance(pb['FACE'][k], float):
                pb['FACE'][k] = 0.0
    for k, v in SWITCHES.items():
        arm[k] = v
    arm.update_tag()
    bpy.context.view_layer.update()


rows = []
for label, fn in POSES.items():
    if fn is not None:
        reset()
        fn()
    arm.update_tag()
    bpy.context.view_layer.update()
    draw_widgets()
    files = []
    for v in views:
        d = DIRS[v].normalized()
        cam.location = centre + d * 10
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        sun.rotation_euler = (-(d + Vector((0.3, 0, 0.6)))).to_track_quat('-Z', 'Y').to_euler()
        f = f'{out}_{len(rows)}_{v}.png'
        scene.render.filepath = f
        bpy.ops.render.render(write_still=True)
        files.append(f)
    row = f'{out}_{len(rows)}.png'
    cmd = ['magick'] + files + ['+append']
    if label:
        cmd += ['-gravity', 'NorthWest', '-pointsize', '28', '-fill', 'white', '-annotate', '+10+8', label]
    subprocess.run(cmd + [row])
    for f in files:
        os.remove(f)
    rows.append(row)
cols = int(os.environ.get('COLS', 1))
if cols > 1:
    subprocess.run(['magick', 'montage'] + rows + ['-tile', f'{cols}x', '-geometry', '+2+2', '-background', '#222',
                    out])
else:
    subprocess.run(['magick'] + rows + ['-append', out])
for r in rows:
    os.remove(r)
