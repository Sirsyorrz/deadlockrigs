"""Blender meshes as game models: write a vmdl_c (with its vmat_c and vtex_c) into the session overlay, which the
game loads when a prop names it. No compiler: the files are the game's own (a static prop's model, a pbr material)
with the mesh, material and texture swapped in. The game reads a loose model first written after it started, so a
mesh is in game seconds after it is exported, without a map reload."""

import copy
import hashlib
import os
import re
import struct

import numpy as np

from . import gamelaunch, s2res

UNIT = 0.0254
FOLDER = 'models/dlrigs/custom'
# A static prop and a pbr material without masks or translucency: the shapes our files copy.
MODEL_TEMPLATE = 'models/props/barrel_03.vmdl_c'
MATERIAL_TEMPLATE = 'models/props_city/materials/bench01a.vmat_c'
NORMAL_TEXTURE = 'materials/default/default_normal_tga_7be61377.vtex'
VERTEX_SIZE = 20  # position f32 x3, uv f16 x2, packed normal
VERTEX_FIELDS = {('POSITION', 0): 0, ('TEXCOORD', 0): 12, ('NORMAL', 0): 16}
MAX_TEXTURE = 2048

_templates = {}


class NoFaces(ValueError):
    pass


def overlay_root(game):
    return os.path.join(game, *gamelaunch.OVERLAY.split('/'))


def _template(game, path):
    vpk = os.path.join(game, 'citadel', 'pak01_dir.vpk')
    key = (vpk, os.path.getmtime(vpk), path)
    if key not in _templates:
        raw = gamelaunch.vpk_read(vpk, path)
        if raw is None:
            raise FileNotFoundError(f'{path} is not in the game files any more')
        _templates[key] = raw
    return s2res.read_resource(_templates[key])


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def pack_normals(n):
    """CompressedTangentFrame: octahedral normal in 10+10 bits, tangent angle 0, bitangent sign +."""
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    p = n / np.maximum(np.abs(n).sum(axis=1, keepdims=True), 1e-12)
    x, y, z = p[:, 0].copy(), p[:, 1].copy(), p[:, 2]
    neg = z < 0
    fx = (1 - np.abs(y)) * np.where(x >= 0, 1.0, -1.0)
    fy = (1 - np.abs(x)) * np.where(y >= 0, 1.0, -1.0)
    x, y = np.where(neg, fx, x), np.where(neg, fy, y)
    xb = np.clip(np.round((x * 0.5 + 0.5) * 1023), 0, 1023).astype(np.uint32)
    yb = np.clip(np.round((y * 0.5 + 0.5) * 1023), 0, 1023).astype(np.uint32)
    return (yb << 22) | (xb << 12) | np.uint32(1)


def mesh_arrays(obj, depsgraph, scale=(1.0, 1.0, 1.0)):
    """(positions in game units, normals, uvs, triangles, material index per triangle) of an object's evaluated mesh
    in its own space with scale applied, axes turned to the game's (x forward), one vertex per distinct corner."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        tris = mesh.loop_triangles
        nt = len(tris)
        if not nt:
            raise NoFaces(f'{obj.name} has no faces')
        loops = np.empty(nt * 3, np.int32)
        tris.foreach_get('loops', loops)
        mat = np.empty(nt, np.int32)
        tris.foreach_get('material_index', mat)
        vert_of_loop = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get('vertex_index', vert_of_loop)
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get('co', co)
        normals = np.empty(len(mesh.loops) * 3, np.float32)
        mesh.corner_normals.foreach_get('vector', normals)
        uv = np.zeros((len(mesh.loops), 2), np.float32)
        layer = mesh.uv_layers.active
        if layer is not None:
            flat = np.empty(len(mesh.loops) * 2, np.float32)
            layer.uv.foreach_get('vector', flat)
            uv = flat.reshape(-1, 2)
    finally:
        evaluated.to_mesh_clear()
    scale = np.maximum(np.array(scale, np.float32), 1e-6)
    co = co.reshape(-1, 3)[vert_of_loop[loops]] * scale
    normals = normals.reshape(-1, 3)[loops] / scale
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    uv = uv[loops] * [1, -1] + [0, 1]  # the game's v runs down from the top of the image
    # Blender +y is game +x (GAME_FROM_BLENDER turns 90 degrees about z).
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    pos = co @ turn.T / UNIT
    normals = normals @ turn.T
    # Corners alike in position, normal and uv share a vertex.
    rows = np.concatenate([np.round(pos, 4), np.round(normals, 4), np.round(uv, 5)], axis=1)
    _, first, index = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    return pos[first], normals[first], uv[first], index.reshape(-1).astype(np.uint32), mat


def material_inputs(material):
    """(base colour RGBA image as (h, w, 4) uint8 top row first, or None; base colour RGBA 0..1 in sRGB)."""
    color = (0.8, 0.8, 0.8, 1.0)
    if material is None:
        return None, color
    color = tuple(material.diffuse_color)
    tree = material.node_tree if material.use_nodes else None
    bsdf = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if tree else None
    if bsdf is None:
        return None, tuple(_linear_to_srgb(np.array(color[:3]))) + (color[3],)
    socket = bsdf.inputs['Base Color']
    if socket.is_linked:
        node = socket.links[0].from_node
        if node.type == 'TEX_IMAGE' and node.image is not None and node.image.size[0] > 0:
            return image_pixels(node.image), color
    color = tuple(socket.default_value)
    return None, tuple(_linear_to_srgb(np.array(color[:3]))) + (color[3],)


def image_pixels(image):
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[::-1]
    if image.is_float:
        px[..., :3] = _linear_to_srgb(px[..., :3])
    px = np.clip(np.round(px * 255), 0, 255).astype(np.uint8)
    while max(px.shape[:2]) > MAX_TEXTURE:
        px = np.clip(np.round(s2res._half_size(px.astype(np.float32))), 0, 255).astype(np.uint8)
    return px


def _short_hash(*parts):
    h = hashlib.sha1()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
    return h.hexdigest()[:10]


def _write(root, path, data):
    """Write a compiled file once: a path is named for its contents, so an existing one is already right."""
    full = os.path.join(root, *(path + '_c').split('/'))
    if not os.path.exists(full):
        os.makedirs(os.path.dirname(full), exist_ok=True)
        tmp = full + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, full)


def write_material(game, root, material):
    """A material's vmat_c (and texture), named for its contents: 'models/dlrigs/custom/mat_<hash>.vmat'."""
    pixels, color = material_inputs(material)
    if pixels is None:
        pixels = np.empty((4, 4, 4), np.uint8)
        pixels[:] = np.clip(np.round(np.array(color) * 255), 0, 255).astype(np.uint8)
    tex_path = f'{FOLDER}/tex_{_short_hash(pixels.shape, pixels.tobytes())}.vtex'
    _write(root, tex_path, s2res.texture(pixels))

    res = _template(game, MATERIAL_TEMPLATE)
    res.drop('RED2')
    doc = res.kv3('DATA')
    textures = {'g_tColor': tex_path, 'g_tNormalRoughness': NORMAL_TEXTURE}
    for param in doc.root['m_textureParams'].v:
        name = param['m_name'].v
        if name in textures:
            param['m_pValue'].v = textures[name]
    refs = [p['m_pValue'].v for p in doc.root['m_textureParams'].v]
    mat_path = f'{FOLDER}/mat_{_short_hash(MATERIAL_TEMPLATE, *refs)}.vmat'
    doc.root['m_materialName'].v = mat_path
    res.set_kv3('DATA', doc)
    res.block('RERL').data = s2res.reference_list(refs)
    _write(root, mat_path, s2res.write_resource(res))
    return mat_path


def _set_int(obj, key, value):
    obj[key] = s2res.integer(value)


def model_file(game, model_path, pos, normals, uv, indices, groups):
    """The vmdl_c bytes of a static, collision-free model. groups: [(material path, first index, index count)]."""
    res = _template(game, MODEL_TEMPLATE)
    res.drop('RED2')
    res.drop('PHYS')
    ctrl = res.kv3('CTRL')
    del ctrl.root['embedded_physics']
    mesh = ctrl.root['embedded_meshes'][0]
    if len(mesh['m_vertexBuffers'].v) != 1 or len(mesh['m_indexBuffers'].v) != 1:
        raise ValueError(f'{MODEL_TEMPLATE} changed shape')
    vb, ib = mesh['m_vertexBuffers'][0], mesh['m_indexBuffers'][0]
    fields = [f for f in vb['m_inputLayoutFields'].v
              if (f['m_pSemanticName'].v, f['m_nSemanticIndex'].v) in VERTEX_FIELDS]
    if len(fields) != len(VERTEX_FIELDS):
        raise ValueError(f'{MODEL_TEMPLATE} changed its vertex layout')
    for f in fields:
        _set_int(f, 'm_nOffset', VERTEX_FIELDS[(f['m_pSemanticName'].v, f['m_nSemanticIndex'].v)])
    vb['m_inputLayoutFields'].v = fields

    rows = np.zeros(len(pos), [('pos', '<f4', 3), ('uv', '<f2', 2), ('normal', '<u4')])
    rows['pos'], rows['uv'], rows['normal'] = pos, uv, pack_normals(normals)
    index_size = 2 if len(pos) <= 0xFFFF else 4
    vblock, iblock = vb['m_nBlockIndex'].v, ib['m_nBlockIndex'].v
    res.blocks[vblock].data = rows.tobytes()
    res.blocks[iblock].data = indices.astype(np.uint16 if index_size == 2 else np.uint32).tobytes()
    for buffer, count, size in ((vb, len(pos), VERTEX_SIZE), (ib, len(indices), index_size)):
        buffer['m_bMeshoptCompressed'] = s2res.boolean(False)
        buffer['m_bCompressedZSTD'] = s2res.boolean(False)
        buffer['m_bMeshoptIndexSequence'] = s2res.boolean(False)
        _set_int(buffer, 'm_nElementCount', count)
        _set_int(buffer, 'm_nElementSizeInBytes', size)
    res.set_kv3('CTRL', ctrl)

    mdat_kind = res.blocks[mesh['m_nDataBlock'].v].kind
    mdat = res.kv3(mdat_kind)
    scene_object = mdat.root['m_sceneObjects'][0]
    template_call = scene_object['m_drawCalls'][0]
    calls = []
    for material, first, count in groups:
        call = copy.deepcopy(template_call)
        _set_int(call, 'm_nBaseVertex', 0)
        _set_int(call, 'm_nVertexCount', len(pos))
        _set_int(call, 'm_nStartIndex', first)
        _set_int(call, 'm_nIndexCount', count)
        call['m_material'].v = material
        calls.append(call)
    scene_object['m_drawCalls'].v = calls
    for key, value in (('m_vMinBounds', pos.min(0)), ('m_vMaxBounds', pos.max(0))):
        for node, x in zip(scene_object[key].v, value):
            node.v = float(x)
    res.set_kv3(mdat_kind, mdat)
    res.block('RERL').data = s2res.reference_list([m for m, _, _ in groups])

    data = res.kv3('DATA')
    data.root['m_name'].v = model_path
    res.set_kv3('DATA', data)
    return s2res.write_resource(res)


def model_name(obj):
    return re.sub(r'[^a-z0-9_]+', '_', obj.name.lower()).strip('_') or 'mesh'


def export(obj, depsgraph, game, scale=(1.0, 1.0, 1.0)):
    """Write an object's mesh as a game model into the session overlay; its path ('models/dlrigs/custom/x.vmdl').
    '' when it has no faces. scale is baked in (an entity only scales uniformly). The path changes with the mesh or its materials: the game
    keeps a loaded model, so an edit needs a new name."""
    root = overlay_root(game)
    try:
        pos, normals, uv, indices, mat = mesh_arrays(obj, depsgraph, scale)
    except NoFaces:
        return ''
    tri_indices = indices.reshape(-1, 3)
    order = np.argsort(mat, kind='stable')
    tri_indices, mat = tri_indices[order], mat[order]
    slots = obj.material_slots
    groups = []
    for slot_index in np.unique(mat):
        first = int(np.searchsorted(mat, slot_index)) * 3
        count = int(np.count_nonzero(mat == slot_index)) * 3
        material = slots[slot_index].material if slot_index < len(slots) else None
        groups.append((write_material(game, root, material), first, count))
    indices = tri_indices.reshape(-1)
    model_path = f'{FOLDER}/{model_name(obj)}_{_short_hash(pos.tobytes(), normals.tobytes(), uv.tobytes(), indices.tobytes(), groups)}.vmdl'
    _write(root, model_path, model_file(game, model_path, pos, normals, uv, indices, groups))
    return model_path
