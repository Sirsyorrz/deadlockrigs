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


def _two_sided(obj, nslots):
    """Per material slot: drawn from both sides, as Blender does unless the material culls back faces."""
    slots = obj.material_slots
    return np.array([not (i < len(slots) and slots[i].material and slots[i].material.use_backface_culling)
                     for i in range(nslots)], bool)


def mesh_arrays(obj, depsgraph, scale=(1.0, 1.0, 1.0)):
    """(positions in game units, normals, uvs, triangles, material index per triangle) of an object's evaluated mesh
    in its own space with scale applied, axes turned to the game's (x forward), one vertex per distinct corner.
    A negative scale mirrors it. The game culls back faces: an open mesh (some edge with one face) gets each
    triangle twice, the copy facing the other way, unless its material has Backface Culling on."""
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
        edge_of_loop = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get('edge_index', edge_of_loop)
        is_open = bool(np.any(np.bincount(edge_of_loop, minlength=len(mesh.edges)) == 1))
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
    scale = np.array(scale, np.float32)
    scale = np.copysign(np.maximum(np.abs(scale), 1e-6), scale)
    if np.prod(scale) < 0:
        loops = loops.reshape(-1, 3)[:, ::-1].reshape(-1)  # mirrored: keep the faces facing out
    if is_open:
        both = np.repeat(_two_sided(obj, int(mat.max()) + 1)[mat], 3)
        flipped = loops.reshape(-1, 3)[:, ::-1].reshape(-1)[both]
        loops = np.concatenate([loops, flipped])
        mat = np.concatenate([mat, mat[both[::3]]])
    sign = np.ones(len(loops), np.float32)
    sign[nt * 3:] = -1
    co = co.reshape(-1, 3)[vert_of_loop[loops]] * scale
    normals = normals.reshape(-1, 3)[loops] / scale * sign[:, None]
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


def _image_floats(image):
    """An image's stored values as (h, w, 4) float32, top row first."""
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[::-1]


def _shrink(px):
    while max(px.shape[:2]) > MAX_TEXTURE:
        px = s2res._half_size(px)
    return px


def _linked_image(socket):
    """(image, channel) feeding a socket straight from an Image Texture, or through a Separate Color node;
    channel None means the colour (or the alpha for an Alpha output)."""
    if not socket.is_linked:
        return None
    link = socket.links[0]
    node, output = link.from_node, link.from_socket.name
    channel = None
    if node.type == 'SEPARATE_COLOR' and node.inputs[0].is_linked:
        channel = {'Red': 0, 'Green': 1, 'Blue': 2}.get(output)
        output = node.inputs[0].links[0].from_socket.name
        node = node.inputs[0].links[0].from_node
    if node.type != 'TEX_IMAGE' or node.image is None or node.image.size[0] <= 0:
        return None
    if channel is None and output == 'Alpha':
        channel = 3
    return node.image, channel


def _resample(px, shape):
    """Nearest-neighbour resize of an (h, w, ...) array to (h2, w2)."""
    h, w = px.shape[:2]
    rows = (np.arange(shape[0]) * h // shape[0]).clip(0, h - 1)
    cols = (np.arange(shape[1]) * w // shape[1]).clip(0, w - 1)
    return px[rows][:, cols]


def _scalar_map(bsdf, name):
    """A Principled input as one value, or as an (h, w) float image when an image drives it."""
    socket = bsdf.inputs[name]
    linked = _linked_image(socket)
    if linked is None:
        return float(socket.default_value)
    image, channel = linked
    px = _shrink(_image_floats(image))
    if channel is None:
        return px[..., :3].mean(axis=2)
    return px[..., channel]


def material_inputs(material):
    """(base colour RGBA 0..1 sRGB as an (h, w, 4) float image or a 4-tuple, roughness, metalness), each map a
    float or an (h, w) float image."""
    if material is None:
        return (0.8, 0.8, 0.8, 1.0), 0.5, 0.0
    color = tuple(material.diffuse_color)
    tree = material.node_tree if material.use_nodes else None
    bsdf = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if tree else None
    if bsdf is None:
        return (tuple(_linear_to_srgb(np.array(color[:3]))) + (color[3],), float(material.roughness),
                float(material.metallic))
    roughness, metalness = _scalar_map(bsdf, 'Roughness'), _scalar_map(bsdf, 'Metallic')
    linked = _linked_image(bsdf.inputs['Base Color'])
    if linked is not None and linked[1] is None:
        image = linked[0]
        px = _shrink(_image_floats(image))
        if image.is_float:
            px[..., :3] = _linear_to_srgb(px[..., :3])
        return px, roughness, metalness
    value = tuple(bsdf.inputs['Base Color'].default_value)
    return tuple(_linear_to_srgb(np.array(value[:3]))) + (value[3],), roughness, metalness


def _to_bytes(px):
    return np.clip(np.round(np.asarray(px, np.float32) * 255), 0, 255).astype(np.uint8)


def material_textures(material):
    """(colour RGBA, normal-roughness RGBA) uint8 images for pbr.vfx. Opaque pbr reads metalness from the colour
    alpha; g_tNormalRoughness is a hemi-octahedral normal in RG (flat: 0.5 0.5) and roughness in B."""
    color, roughness, metalness = material_inputs(material)
    maps = [m for m in (color, metalness) if not isinstance(m, (tuple, float))]
    shape = max((m.shape[:2] for m in maps), key=lambda s: s[0] * s[1], default=(4, 4))
    if isinstance(color, tuple):
        rgba = np.empty(shape + (4,), np.float32)
        rgba[:] = color
    else:
        rgba = _resample(color, shape).copy()
    rgba[..., 3] = metalness if isinstance(metalness, float) else _resample(metalness, shape)
    rough_shape = (4, 4) if isinstance(roughness, float) else roughness.shape[:2]
    normal = np.empty(rough_shape + (4,), np.float32)
    normal[:] = (0.5, 0.5, 0.5, 1.0)
    normal[..., 2] = roughness
    return _to_bytes(rgba), _to_bytes(normal)


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
    """A material's vmat_c (and textures), named for its contents: 'models/dlrigs/custom/mat_<hash>.vmat'."""
    textures = {}
    for param, pixels in zip(('g_tColor', 'g_tNormalRoughness'), material_textures(material)):
        textures[param] = f'{FOLDER}/tex_{_short_hash(pixels.shape, pixels.tobytes())}.vtex'
        _write(root, textures[param], s2res.texture(pixels))

    res = _template(game, MATERIAL_TEMPLATE)
    res.drop('RED2')
    doc = res.kv3('DATA')
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
    '' when it has no faces. scale (signed, as Matrix.to_scale gives it) is baked in: an entity only scales
    uniformly. The path changes with the mesh or its materials: the game keeps a loaded model, so an edit needs a
    new name."""
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
