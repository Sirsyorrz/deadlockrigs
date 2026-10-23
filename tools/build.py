"""blender -b -P build.py -- <hero>

Imports the S2V export, drops the bone display sphere, names the armature after the hero and the meshes after
their vmdl parts, hides meshes that are not in the default bodygroup choice, adds facial shape keys when the
hero has a face config, and writes build/<hero>.blend (the rig step's input). DLRIG_EXPORTS=1 also writes
out/<hero>.glb and .fbx.
"""
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
from bodygroups import mesh_parts, split_meshes  # noqa: E402
from collar import fix_collar  # noqa: E402
from hero import from_argv  # noqa: E402
from tongue_mesh import add_part, interior_material, tongue_material  # noqa: E402

hero, _ = from_argv()
os.makedirs(os.path.dirname(hero.base_blend), exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=hero.raw_glb)

# Source2Viewer writes the game's sRGB vertex colours (hair tints) unconverted, and glTF colours are linear, so they
# came out washed out (Pocket's dark brown hair as grey); the game's shader linearises them
for me in {o.data for o in bpy.data.objects if o.type == 'MESH'}:
    for ca in me.color_attributes:
        c = np.empty(len(ca.data) * 4, np.float32)
        ca.data.foreach_get('color', c)
        ca.data.foreach_set('color_srgb', c)

arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
for b in arm.pose.bones:
    b.custom_shape = None
arm.name = hero.name
arm.data.name = hero.name

parts = mesh_parts(open(hero.vmdl).read())
alt = bpy.data.collections.new('Alternate parts')
bpy.context.scene.collection.children.link(alt)
for o in list(bpy.data.objects):
    if o.type != 'MESH':
        continue
    if o.parent is None:
        bpy.data.objects.remove(o, do_unlink=True)
        continue
    part = next((p for p in sorted(parts, key=len, reverse=True) if o.name.endswith('.' + p)), None)
    if part is None:
        print('mesh without a vmdl part:', o.name)
        part = o.name.rsplit('.', 1)[-1]
    if part in hero.drop:
        bpy.data.objects.remove(o, do_unlink=True)
        continue
    o.name = hero.rename.get(part, part)
    o.data.name = o.name
    if not parts.get(part, True):
        for c in o.users_collection:
            c.objects.unlink(o)
        alt.objects.link(o)
        o.hide_set(True)
        o.hide_viewport = o.hide_render = True
if not alt.objects:
    bpy.data.collections.remove(alt)


split_meshes(hero)
print('meshes', sorted(o.name for o in bpy.data.objects if o.type == 'MESH' and not o.hide_viewport),
      'hidden', sorted(o.name for o in bpy.data.objects if o.type == 'MESH' and o.hide_viewport))

if hero.face:
    keys_path = os.path.join(hero.data, 'keys.npz')
    head = bpy.data.objects[hero.face['mesh']]
    me = head.data
    inv = np.array(head.matrix_world.inverted().to_3x3())
    n_head = len(me.vertices)
    # keys.py opened sculpted closed lips by dropping the faces joining them (dump.py's triangles are the polygons)
    cut_path = keys_path.replace('.npz', '_cut.npy')
    if os.path.exists(cut_path):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[int(i)] for i in np.load(cut_path)], context='FACES_ONLY')
        bm.to_mesh(me)
        bm.free()
        assert len(me.vertices) == n_head
    # or split them along the upper lip's edge: copies of those vertices for the lower lip's faces
    split_path = keys_path.replace('.npz', '_split.npz')
    if os.path.exists(split_path):
        sp = np.load(split_path)
        src = sp['src']
        me.vertices.add(len(src))
        wl = np.array(head.matrix_world.inverted())
        for i, (v, off) in enumerate(zip(src, sp['off'])):
            nv = me.vertices[n_head + i]
            nv.co = me.vertices[int(v)].co + Vector(off @ wl[:3, :3].T)
            for g in me.vertices[int(v)].groups:
                head.vertex_groups[g.group].add([nv.index], g.weight, 'REPLACE')
        for f, k, nv in sp['rewire']:
            me.loops[me.polygons[int(f)].loop_start + int(k)].vertex_index = int(nv)
        me.update(calc_edges=True)
        n_head = len(me.vertices)
        print('split the lips:', len(src), 'vertices')
    # generated parts after the head's own vertices, in the order keys.py's keys expect: mouth bag, then tongue
    extra = []
    for suffix, mat in (('_mouth.npz', interior_material), ('_tongue.npz', tongue_material)):
        part_path = keys_path.replace('.npz', suffix)
        if os.path.exists(part_path):
            extra.append(add_part(head, part_path, mat()))
    n = len(me.vertices)
    # game morphs and face bones come with the game's own skinning; only keys.py heads get the collar fix
    if hero.face.get('builder', 'keys.py') == 'keys.py':
        others = [o for o in bpy.data.objects if o.type == 'MESH' and o != head and not o.hide_viewport and o.vertex_groups]
        scale = arm.data.bones['head'].head_local.z / 2.094
        print('collar: head vertices covered', fix_collar(head, others, scale, n_head))
    base = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', base)
    keys = np.load(keys_path)
    if not me.shape_keys:
        head.shape_key_add(name='Basis', from_mix=False)
    # per-hero strength of a key (and its in-between): a sculpted open grin (Drifter) can't take a full pucker
    tune = hero.face.get('tune', {})
    for name in keys.files:
        d = np.concatenate([keys[name]] + [part[name] for part in extra])
        d = d * tune.get(name, tune.get(name[:-4], 1.0) if name.endswith('_mid') else 1.0)
        assert d.shape == (n, 3), (name, d.shape, n)
        # a game morph can share its ARKit name (jawOpen)
        kb = me.shape_keys.key_blocks.get(name) or head.shape_key_add(name=name, from_mix=False)
        kb.data.foreach_set('co', (base.reshape(-1, 3) + d @ inv.T).astype(np.float32).ravel())
    print('shape keys', len(keys.files))
    # props held in the mouth (Venator's cigar, heroes.json face "ride") move as one piece with the lips they touch
    head_world = base.reshape(-1, 3)[:n_head] @ np.array(head.matrix_world)[:3, :3].T + np.array(head.matrix_world)[:3, 3]
    for part in hero.face.get('ride', []):
        ob = bpy.data.objects.get(hero.rename.get(part, part))
        if ob is None:
            print('WARNING: no mesh', part, 'to ride the lips')
            continue
        q = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
        d = np.linalg.norm(head_world[:, None] - q[None], axis=2).min(1)
        touch = d < max(d.min() + 0.002, 0.003)
        oinv = np.array(ob.matrix_world.inverted().to_3x3())
        ob_base = np.array([v.co for v in ob.data.vertices], np.float32)
        if not ob.data.shape_keys:
            ob.shape_key_add(name='Basis', from_mix=False)
        for name in keys.files:
            off = keys[name][:n_head][touch].mean(0)
            kb = ob.data.shape_keys.key_blocks.get(name) or ob.shape_key_add(name=name, from_mix=False)
            kb.data.foreach_set('co', (ob_base + off @ oinv.T).astype(np.float32).ravel())
        print('riding the lips:', ob.name, int(touch.sum()), 'head vertices')
    # straps on the face (The Doorman's chinstrap, heroes.json face "follow": {mesh: [bones]}) take the keys of the
    # skin under them, fading out where they lift off it
    if hero.face.get('follow'):
        from mathutils.kdtree import KDTree
        tree = KDTree(n_head)
        for i, c in enumerate(head_world):
            tree.insert(c, i)
        tree.balance()
        scale = arm.data.bones['head'].head_local.z / 2.094
        near_d, far_d = 0.006 * scale, 0.015 * scale
    for part, bones in hero.face.get('follow', {}).items():
        ob = bpy.data.objects.get(hero.rename.get(part, part))
        if ob is None:
            print('WARNING: no mesh', part, 'to follow the face')
            continue
        groups = {g.index for g in ob.vertex_groups if g.name in bones}
        nv = len(ob.data.vertices)
        wts, nears = np.zeros(nv), np.zeros((nv, 6), int)
        for v in ob.data.vertices:
            if sum(g.weight for g in v.groups if g.group in groups) < 0.5:
                continue
            found = tree.find_n(ob.matrix_world @ v.co, 6)
            wts[v.index] = np.clip((far_d - found[0][2]) / (far_d - near_d), 0, 1)
            nears[v.index] = [f[1] for f in found]
        moving = np.where(wts > 0)[0]
        if not len(moving):
            print('WARNING:', part, 'does not touch the face')
            continue
        oinv = np.array(ob.matrix_world.inverted().to_3x3())
        ob_base = np.array([v.co for v in ob.data.vertices], np.float32)
        if not ob.data.shape_keys:
            ob.shape_key_add(name='Basis', from_mix=False)
        for name in keys.files:
            off = np.zeros((nv, 3))
            off[moving] = keys[name][:n_head][nears[moving]].mean(1) * wts[moving, None]
            kb = ob.data.shape_keys.key_blocks.get(name) or ob.shape_key_add(name=name, from_mix=False)
            kb.data.foreach_set('co', (ob_base + off @ oinv.T).astype(np.float32).ravel())
        print('following the face:', ob.name, len(moving), 'vertices')
    # bonekeys.py: keys on the other meshes the face bones move (eyes, teeth, tongue). The head gets every key too, as
    # the one rig.py drives; the others follow it
    for f in sorted(os.listdir(hero.data)):
        if not f.startswith('keysmesh_'):
            continue
        part = f[len('keysmesh_'):-len('.npz')]
        ob = bpy.data.objects.get(hero.rename.get(part, part))
        if ob is None:
            continue
        mk = np.load(os.path.join(hero.data, f))
        oinv = np.array(ob.matrix_world.inverted().to_3x3())
        ob_base = np.empty(len(ob.data.vertices) * 3, np.float32)
        ob.data.vertices.foreach_get('co', ob_base)
        if not ob.data.shape_keys:
            ob.shape_key_add(name='Basis', from_mix=False)
        for name in mk.files:
            kb = ob.data.shape_keys.key_blocks.get(name) or ob.shape_key_add(name=name, from_mix=False)
            kb.data.foreach_set('co', (ob_base.reshape(-1, 3) + mk[name] @ oinv.T).astype(np.float32).ravel())
            if name not in me.shape_keys.key_blocks:
                head.shape_key_add(name=name, from_mix=False)
        print('shape keys on', ob.name, len(mk.files))

# The mouth bag shares the skin texture, so an open mouth showed a flat skin-coloured wall. Per vertex:
# 0 outside the bag, 1 the lips' inner faces, 2 deeper in; faces reaching deeper in go dark. Game-morph faces have
# their own mouth interior.
bag_path = os.path.join(hero.data, 'keys_bag.npy')
if hero.face and os.path.exists(bag_path):
    bag = np.load(bag_path)
    mat = interior_material()
    if mat.name not in me.materials:
        me.materials.append(mat)
    mi = list(me.materials).index(mat)
    for poly in me.polygons:
        if poly.vertices[0] >= n_head:
            continue
        if all(bag[v] for v in poly.vertices) and any(bag[v] == 2 for v in poly.vertices):
            poly.material_index = mi

bpy.ops.wm.save_as_mainfile(filepath=hero.base_blend)
print('wrote', hero.base_blend)
if os.environ.get('DLRIG_EXPORTS') == '1':
    os.makedirs(hero.out, exist_ok=True)
    stem = os.path.join(hero.out, hero.slug)
    bpy.ops.export_scene.gltf(filepath=stem + '.glb', export_format='GLB', export_morph_normal=True,
                              export_apply=False, export_animations=False, use_visible=True)
    bpy.ops.export_scene.fbx(filepath=stem + '.fbx', path_mode='COPY', embed_textures=True,
                             add_leaf_bones=False, bake_anim=False, use_visible=True)
    print('wrote', stem + '.glb/.fbx')
