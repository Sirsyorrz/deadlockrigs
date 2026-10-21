"""Adds the generated tongue (keys.py -> data/keys_tongue.npz) to the head mesh inside Blender, so the head's
shape keys and drivers move it with no extra rigging."""
import bmesh
import bpy
import numpy as np
from mathutils import Matrix


def add_tongue(head, path):
    """Appends the tongue to head's mesh (before any shape keys exist). Returns {key: world offsets} for the
    new vertices, which come after the original ones."""
    t = np.load(path)
    me = head.data
    n = len(me.vertices)
    base = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', base)
    mw = np.array(head.matrix_world)
    world = base.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    inv = np.array(Matrix(mw).inverted())
    local = t['co'] @ inv[:3, :3].T + inv[:3, 3]

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
    me.materials.append(mat)
    mi = len(me.materials) - 1

    bm = bmesh.new()
    bm.from_mesh(me)
    deform = bm.verts.layers.deform.verify()
    bm.verts.ensure_lookup_table()
    # skin weights of the nearest head vertex, so the tongue rides the same bones
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
