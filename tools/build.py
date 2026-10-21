"""blender -b -P build.py -- <hero>

Imports the S2V export, drops the bone display sphere, names the armature after the hero and the meshes after
their vmdl parts, hides meshes that are not in the default bodygroup choice, adds facial shape keys when the
hero has a face config, and writes out/<hero>.blend, .glb and .fbx.
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bodygroups import mesh_parts  # noqa: E402
from hero import from_argv  # noqa: E402
from tongue_mesh import add_tongue  # noqa: E402

hero, _ = from_argv()
os.makedirs(hero.out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=hero.raw_glb)

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
print('meshes', sorted(o.name for o in bpy.data.objects if o.type == 'MESH' and not o.hide_viewport),
      'hidden', sorted(o.name for o in bpy.data.objects if o.type == 'MESH' and o.hide_viewport))

if hero.face:
    keys_path = os.path.join(hero.data, 'keys.npz')
    head = bpy.data.objects[hero.face['mesh']]
    me = head.data
    inv = np.array(head.matrix_world.inverted().to_3x3())
    n_head = len(me.vertices)
    tongue_path = keys_path.replace('.npz', '_tongue.npz')
    tongue = add_tongue(head, tongue_path) if os.path.exists(tongue_path) else {}
    n = len(me.vertices)
    base = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', base)
    keys = np.load(keys_path)
    head.shape_key_add(name='Basis', from_mix=False)
    for name in keys.files:
        d = np.concatenate([keys[name], tongue[name]]) if tongue else keys[name]
        assert d.shape == (n, 3), (name, d.shape, n)
        kb = head.shape_key_add(name=name, from_mix=False)
        kb.data.foreach_set('co', (base.reshape(-1, 3) + d @ inv.T).astype(np.float32).ravel())
    print('shape keys', len(keys.files))

    # The mouth bag shares the skin texture, so an open mouth showed a flat skin-coloured wall. Per vertex:
    # 0 outside the bag, 1 the lips' inner faces, 2 deeper in; faces reaching deeper in go dark.
    bag = np.load(keys_path.replace('.npz', '_bag.npy'))
    mat = bpy.data.materials.new('mouth_interior')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.09, 0.025, 0.03, 1)
    bsdf.inputs['Roughness'].default_value = 0.6
    # flat, shallow bags (Apollo) otherwise just mirror the sky
    bsdf.inputs['Specular IOR Level'].default_value = 0
    me.materials.append(mat)
    mi = len(me.materials) - 1
    for poly in me.polygons:
        if poly.vertices[0] >= n_head:
            continue
        if all(bag[v] for v in poly.vertices) and any(bag[v] == 2 for v in poly.vertices):
            poly.material_index = mi

bpy.ops.wm.save_as_mainfile(filepath=hero.base_blend)
stem = os.path.join(hero.out, hero.slug)
bpy.ops.export_scene.gltf(filepath=stem + '.glb', export_format='GLB', export_morph_normal=True,
                          export_apply=False, export_animations=False, use_visible=True)
bpy.ops.export_scene.fbx(filepath=stem + '.fbx', path_mode='COPY', embed_textures=True,
                         add_leaf_bones=False, bake_anim=False, use_visible=True)
print('wrote', hero.out)
