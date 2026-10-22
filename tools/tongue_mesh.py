"""Adds keys.py's generated parts (data/keys_tongue.npz, data/keys_mouth.npz: a mouth bag for heads without one) to
the head mesh inside Blender, so the head's shape keys and drivers move them with no extra rigging."""
import bmesh
import bpy
import numpy as np
from mathutils import Matrix


def interior_material():
    """The dark mouth inside, shared by the bag faces build.py darkens and a generated bag."""
    mat = bpy.data.materials.get('mouth_interior')
    if mat is None:
        mat = bpy.data.materials.new('mouth_interior')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (0.09, 0.025, 0.03, 1)
        bsdf.inputs['Roughness'].default_value = 0.6
        # flat, shallow bags (Apollo) otherwise just mirror the sky
        bsdf.inputs['Specular IOR Level'].default_value = 0
    return mat


def tongue_material():
    mat = bpy.data.materials.new('tongue')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    # darkened by its own occlusion: in the mouth it otherwise catches the full sky and reads grey
    ao = mat.node_tree.nodes.new('ShaderNodeAmbientOcclusion')
    ao.inputs['Color'].default_value = (0.3, 0.05, 0.055, 1)
    ao.inputs['Distance'].default_value = 0.03
    mat.node_tree.links.new(ao.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.6
    # deep in the mouth there is nothing to reflect; full specular made it look grey
    bsdf.inputs['Specular IOR Level'].default_value = 0.08
    return mat


def add_part(head, path, mat):
    """Appends a generated part to head's mesh (before any shape keys exist). Returns {key: world offsets} for the
    new vertices, which come after the ones already there."""
    t = np.load(path)
    me = head.data
    n = len(me.vertices)
    base = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', base)
    mw = np.array(head.matrix_world)
    world = base.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    inv = np.array(Matrix(mw).inverted())
    local = t['co'] @ inv[:3, :3].T + inv[:3, 3]
    if mat.name not in me.materials:
        me.materials.append(mat)
    mi = list(me.materials).index(mat)

    bm = bmesh.new()
    bm.from_mesh(me)
    deform = bm.verts.layers.deform.verify()
    bm.verts.ensure_lookup_table()
    # skin weights of the nearest head vertex, so the part rides the same bones
    near = np.argmin(((world[None] - t['co'][:, None]) ** 2).sum(-1), axis=1)
    src = [dict(bm.verts[int(j)][deform].items()) for j in near]
    new = []
    for c, groups in zip(local, src):
        v = bm.verts.new(c)
        for g, wt in groups.items():
            v[deform][g] = wt
        new.append(v)
    for tri in t['tris']:
        f = bm.faces.new([new[i] for i in tri])
        f.material_index = mi
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    assert len(me.vertices) == n + len(local)
    return {k[4:]: t[k] for k in t.files if k.startswith('key_')}
