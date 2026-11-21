"""Blender meshes as game models: write a vmdl_c (with its vmat_c and vtex_c) into the session overlay, which the
game loads when a prop names it. No compiler: the files are the game's own (a static prop's model, a pbr material)
with the mesh, material and texture swapped in. The game reads a loose model first written after it started, so a
mesh is in game seconds after it is exported, without a map reload."""

import copy
import hashlib
import os
import re
import struct
from dataclasses import dataclass

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


def _octahedral_bits(n):
    p = n / np.maximum(np.abs(n).sum(axis=1, keepdims=True), 1e-12)
    x, y, z = p[:, 0].copy(), p[:, 1].copy(), p[:, 2]
    neg = z < 0
    fx = (1 - np.abs(y)) * np.where(x >= 0, 1.0, -1.0)
    fy = (1 - np.abs(x)) * np.where(y >= 0, 1.0, -1.0)
    x, y = np.where(neg, fx, x), np.where(neg, fy, y)
    xb = np.clip(np.round((x * 0.5 + 0.5) * 1023), 0, 1023).astype(np.uint32)
    yb = np.clip(np.round((y * 0.5 + 0.5) * 1023), 0, 1023).astype(np.uint32)
    return xb, yb


def _unoctahedral(xb, yb):
    """The normal the game decodes from the bits (DecompressNormalTangent2)."""
    x, y = xb / 1023.0 * 2 - 1, yb / 1023.0 * 2 - 1
    z = 1 - np.abs(x) - np.abs(y)
    fold = np.clip(-z, 0, 1)
    x = x + np.where(x >= 0, -fold, fold)
    y = y + np.where(y >= 0, -fold, fold)
    n = np.stack([x, y, z], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def pack_frames(n, t, w):
    """CompressedTangentFrame: octahedral normal in 10+10 bits, then the tangent as an 11-bit turn about the normal
    from a reference the game derives from the normal alone, then the bitangent sign (bitangent = w cross(n, t))."""
    xb, yb = _octahedral_bits(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
    m = _unoctahedral(xb, yb)
    s = np.where(m[:, 2] >= 0, 1.0, -1.0)
    rcp = 1 / (s + m[:, 2])
    ref = np.stack([1 - s * m[:, 0] ** 2 * rcp, -s * m[:, 0] * m[:, 1] * rcp, -s * m[:, 0]], axis=1)
    angle = np.arctan2(np.einsum('ij,ij->i', t, np.cross(m, ref)), np.einsum('ij,ij->i', t, ref))
    tb = np.round(np.mod(angle, 2 * np.pi) / (2 * np.pi) * 2047).astype(np.uint32) % 2047
    return (yb << 22) | (xb << 12) | (tb << 1) | (w > 0).astype(np.uint32)


def _two_sided(obj, nslots):
    """Per material slot: drawn from both sides, as Blender does unless the material culls back faces."""
    slots = obj.material_slots
    return np.array([not (i < len(slots) and slots[i].material and slots[i].material.use_backface_culling)
                     for i in range(nslots)], bool)


def _has_ngons(mesh):
    totals = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get('loop_total', totals)
    return bool(np.any(totals > 4))


def _ngons_triangulated(mesh):
    """A copy of the mesh with its ngons cut into triangles (tangents are only computed for tris and quads)."""
    import bmesh
    import bpy
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        out = bpy.data.meshes.new('dlr_triangulated')
        bm.to_mesh(out)
    finally:
        bm.free()
    return out


def _corner_arrays(mesh):
    mesh.calc_loop_triangles()
    tris = mesh.loop_triangles
    nt = len(tris)
    loops = np.empty(nt * 3, np.int32)
    tris.foreach_get('loops', loops)
    mat = np.empty(nt, np.int32)
    tris.foreach_get('material_index', mat)
    nl = len(mesh.loops)
    edge_of_loop = np.empty(nl, np.int32)
    mesh.loops.foreach_get('edge_index', edge_of_loop)
    is_open = bool(np.any(np.bincount(edge_of_loop, minlength=len(mesh.edges)) == 1))
    vert_of_loop = np.empty(nl, np.int32)
    mesh.loops.foreach_get('vertex_index', vert_of_loop)
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get('co', co)
    normals = np.empty(nl * 3, np.float32)
    mesh.corner_normals.foreach_get('vector', normals)
    uv = np.zeros((nl, 2), np.float32)
    tangents = np.zeros((nl, 3), np.float32)
    bitangents = np.zeros((nl, 3), np.float32)
    layer = mesh.uv_layers.active
    if layer is not None and nt:
        flat = np.empty(nl * 2, np.float32)
        layer.uv.foreach_get('vector', flat)
        uv = flat.reshape(-1, 2)
        mesh.calc_tangents(uvmap=layer.name)
        mesh.loops.foreach_get('tangent', tangents.reshape(-1))
        mesh.loops.foreach_get('bitangent', bitangents.reshape(-1))
        mesh.free_tangents()
    return (loops, mat, is_open, vert_of_loop, co.reshape(-1, 3), normals.reshape(-1, 3), uv, tangents,
            bitangents)


def mesh_arrays(obj, depsgraph, scale=(1.0, 1.0, 1.0)):
    """(positions in game units, packed tangent frames, uvs, triangles, material index per triangle) of an object's
    evaluated mesh in its own space with scale applied, axes turned to the game's (x forward), one vertex per
    distinct corner. A negative scale mirrors it. The game culls back faces: an open mesh (some edge with one face)
    gets each triangle twice, the copy facing the other way, unless its material has Backface Culling on.
    Tangents (for normal maps) are Blender's, from the active uv map."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    triangulated = None
    try:
        if mesh.uv_layers.active is not None and _has_ngons(mesh):
            triangulated = _ngons_triangulated(mesh)
        loops, mat, is_open, vert_of_loop, co, normals, uv, tangents, bitangents = _corner_arrays(
            triangulated or mesh)
    finally:
        if triangulated is not None:
            import bpy
            bpy.data.meshes.remove(triangulated)
        evaluated.to_mesh_clear()
    nt = len(mat)
    if not nt:
        raise NoFaces(f'{obj.name} has no faces')
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
    co = co[vert_of_loop[loops]] * scale
    normals = normals[loops] / scale * sign[:, None]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # tangent and bitangent lie along the surface: they scale like positions
    tangents, bitangents = tangents[loops] * scale, bitangents[loops] * scale
    uv = uv[loops] * [1, -1] + [0, 1]  # the game's v runs down from the top of the image
    # Blender +y is game +x (GAME_FROM_BLENDER turns 90 degrees about z).
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    pos = co @ turn.T / UNIT
    normals, tangents, bitangents = normals @ turn.T, tangents @ turn.T, bitangents @ turn.T
    tangents -= normals * np.einsum('ij,ij->i', tangents, normals)[:, None]
    # The game's bitangent is w cross(n, t); w picks Blender's, so Blender's tangent-space normals carry over.
    w = np.where(np.einsum('ij,ij->i', np.cross(normals, tangents), bitangents) < 0, -1.0, 1.0)
    frames = pack_frames(normals, tangents, w)
    # Corners alike in position, frame and uv share a vertex.
    rows = np.concatenate([np.round(pos, 4), frames[:, None].astype(np.float64), np.round(uv, 5)], axis=1)
    _, first, index = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    return pos[first], frames[first], uv[first], index.reshape(-1).astype(np.uint32), mat


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


def _normal_map(bsdf):
    """The tangent-space normals (h, w, 3) of a Normal Map node (Tangent Space, strength applied) with an image
    feeding the Principled Normal, or None."""
    socket = bsdf.inputs['Normal']
    if not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type != 'NORMAL_MAP' or node.space != 'TANGENT':
        return None
    linked = _linked_image(node.inputs['Color'])
    if linked is None or linked[1] is not None:
        return None
    n = _shrink(_image_floats(linked[0]))[..., :3] * 2 - 1
    strength = max(float(node.inputs['Strength'].default_value), 0.0)
    n = n * strength + np.array([0, 0, 1 - strength], np.float32)
    return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)


@dataclass
class Surface:
    """A material as pbr.vfx takes it. Maps are a float or an (h, w) float image; color is sRGB."""
    color: object = (0.8, 0.8, 0.8)  # RGB tuple or (h, w, 3+) image
    roughness: object = 0.5
    metalness: object = 0.0
    alpha: object = 1.0
    blended: bool = False  # Blender's Blended render method: alpha always blends, never cuts
    normal: object = None  # (h, w, 3) tangent-space normals
    emission: tuple = (0.0, 0.0, 0.0)  # linear RGB times strength
    emission_mask: object = None  # (h, w) image: glows in the base colour where set


def surface(material):
    if material is None:
        return Surface()
    tree = material.node_tree if material.use_nodes else None
    bsdf = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if tree else None
    if bsdf is None:
        color = material.diffuse_color
        return Surface(tuple(_linear_to_srgb(np.array(color[:3]))), float(material.roughness),
                       float(material.metallic), float(color[3]))
    out = Surface(roughness=_scalar_map(bsdf, 'Roughness'), metalness=_scalar_map(bsdf, 'Metallic'),
                  alpha=_scalar_map(bsdf, 'Alpha'), normal=_normal_map(bsdf),
                  blended=getattr(material, 'surface_render_method', '') == 'BLENDED')
    linked = _linked_image(bsdf.inputs['Base Color'])
    if linked is not None and linked[1] is None:
        image = linked[0]
        px = _shrink(_image_floats(image))
        if image.is_float:
            px[..., :3] = _linear_to_srgb(px[..., :3])
        out.color = px
    else:
        out.color = tuple(_linear_to_srgb(np.array(bsdf.inputs['Base Color'].default_value[:3])))
    strength = bsdf.inputs['Emission Strength']
    strength = 1.0 if strength.is_linked else float(strength.default_value)
    emission = bsdf.inputs['Emission Color']
    linked = _linked_image(emission)
    if linked is not None and strength > 0:
        px = _shrink(_image_floats(linked[0]))
        mask = px[..., :3].max(axis=2) if linked[1] is None else px[..., linked[1]]
        out.emission, out.emission_mask = (strength,) * 3, mask
    elif not emission.is_linked:
        out.emission = tuple(float(c) * strength for c in emission.default_value[:3])
    return out


def _to_bytes(px):
    return np.clip(np.round(np.asarray(px, np.float32) * 255), 0, 255).astype(np.uint8)


def _largest(*maps, default=(4, 4)):
    shapes = [m.shape[:2] for m in maps if isinstance(m, np.ndarray)]
    return max(shapes, key=lambda s: s[0] * s[1], default=default)


def _fill(value, shape):
    return np.full(shape, value, np.float32) if not isinstance(value, np.ndarray) else _resample(value, shape)


def alpha_mode(surf):
    """'' (opaque), 'test' (cut out) or 'blend'. An image alpha that is all but fully on or off cuts."""
    alpha = surf.alpha
    if not isinstance(alpha, np.ndarray):
        return '' if alpha >= 0.999 else 'blend'
    if alpha.min() >= 0.999:
        return ''
    partial = np.count_nonzero((alpha > 0.02) & (alpha < 0.98)) / alpha.size
    return 'blend' if surf.blended or partial > 0.01 else 'test'


def material_textures(surf, mode):
    """{texture param: uint8 RGBA image} for pbr.vfx. Opaque pbr reads metalness from the colour alpha; cut-out and
    blended ones read opacity there and metalness from g_tMetalness. g_tNormalRoughness is a hemi-octahedral
    tangent-space normal in RG and roughness in B."""
    shape = _largest(surf.color, surf.metalness, surf.alpha)
    if isinstance(surf.color, np.ndarray):
        rgba = _resample(surf.color, shape)[..., :4].copy()
        if rgba.shape[2] == 3:
            rgba = np.concatenate([rgba, np.ones(shape + (1,), np.float32)], axis=2)
    else:
        rgba = np.empty(shape + (4,), np.float32)
        rgba[..., :3] = surf.color
    out = {}
    if mode:
        rgba[..., 3] = _fill(surf.alpha, shape)
        metal = _fill(surf.metalness, _largest(surf.metalness))
        out['g_tMetalness'] = np.repeat(metal[..., None], 4, axis=2)
    else:
        rgba[..., 3] = _fill(surf.metalness, shape)
    out['g_tColor'] = rgba
    shape = _largest(surf.normal, surf.roughness)
    nr = np.empty(shape + (4,), np.float32)
    nr[:] = (128 / 255, 128 / 255, 0.5, 1.0)
    if surf.normal is not None:
        nr[..., :2] = _hemi_octahedral(_resample(surf.normal, shape))
    nr[..., 2] = _fill(surf.roughness, shape)
    out['g_tNormalRoughness'] = nr
    if surf.emission_mask is not None:
        out['g_tSelfIllumMask'] = np.repeat(surf.emission_mask[..., None], 4, axis=2)
    return {name: _to_bytes(px) for name, px in out.items()}


def _hemi_octahedral(n):
    """RG of tangent-space normals for pbr's DecodeHemiOctahedronNormal, which flips y after decoding and centres
    on 128/255."""
    x, y, z = n[..., 0], -n[..., 1], np.maximum(n[..., 2], 0)
    total = np.maximum(np.abs(x) + np.abs(y) + z, 1e-12)
    px, py = x / total, y / total
    return np.stack([(px + py) / 2, (px - py) / 2], axis=-1) + 128 / 255


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


def _number(value):
    value = float(value)
    return s2res.Node(s2res.DOUBLE_ZERO if value == 0 else s2res.DOUBLE_ONE if value == 1 else s2res.DOUBLE, value)


PARAM_VALUE = {'m_intParams': 'm_nValue', 'm_floatParams': 'm_flValue', 'm_vectorParams': 'm_value',
               'm_textureParams': 'm_pValue'}


def _set_param(root, kind, name, value):
    """Set a material parameter, adding it (shaped like the list's first entry) when the template lacks it."""
    params = root[kind].v
    entry = next((p for p in params if p['m_name'].v == name), None)
    if entry is None:
        entry = copy.deepcopy(params[0])
        entry['m_name'].v = name
        params.append(entry)
        params.sort(key=lambda p: p['m_name'].v)
    key = PARAM_VALUE[kind]
    if kind == 'm_intParams':
        entry[key] = s2res.integer(int(value))
    elif kind == 'm_floatParams':
        entry[key] = _number(value)
    elif kind == 'm_vectorParams':
        for node, v in zip(entry[key].v, value):
            node.t, node.v = s2res.DOUBLE, float(v)
    else:
        entry[key].v = value


# Blender emission strength 1 to self illumination scale, matched by eye against EEVEE
EMISSION_SCALE = 1.0


def write_material(game, root, material):
    """A material's vmat_c (and textures), named for its contents: 'models/dlrigs/custom/mat_<hash>.vmat'."""
    surf = surface(material)
    mode = alpha_mode(surf)
    res = _template(game, MATERIAL_TEMPLATE)
    res.drop('RED2')
    doc = res.kv3('DATA')
    r = doc.root
    for param, pixels in material_textures(surf, mode).items():
        path = f'{FOLDER}/tex_{_short_hash(pixels.shape, pixels.tobytes())}.vtex'
        _write(root, path, s2res.texture(pixels))
        _set_param(r, 'm_textureParams', param, path)
    if mode == 'test':
        _set_param(r, 'm_intParams', 'F_ALPHA_TEST', 1)
        _set_param(r, 'm_floatParams', 'g_flAlphaTestReference1', 0.5)
    elif mode == 'blend':
        _set_param(r, 'm_intParams', 'F_TRANSLUCENT', 1)
        _set_param(r, 'm_floatParams', 'g_flOpacityScale1', 1.0)
    glow = max(surf.emission)
    if glow > 0:
        _set_param(r, 'm_intParams', 'F_SELF_ILLUM', 1)
        # the scale saturates where it blends the albedo to the tint, so the tint carries the colour at strength 1
        _set_param(r, 'm_vectorParams', 'g_vSelfIllumTint1', [c / glow for c in surf.emission] + [0.0])
        _set_param(r, 'm_floatParams', 'g_flSelfIllumScale1', glow * EMISSION_SCALE)
        _set_param(r, 'm_floatParams', 'g_flSelfIllumAlbedoFactor1', 1.0 if surf.emission_mask is not None else 0.0)
    refs = [p['m_pValue'].v for p in r['m_textureParams'].v]
    settings = [(p['m_name'].v, p[PARAM_VALUE[k]].py()) for k in PARAM_VALUE for p in r[k].v]
    mat_path = f'{FOLDER}/mat_{_short_hash(MATERIAL_TEMPLATE, settings)}.vmat'
    r['m_materialName'].v = mat_path
    res.set_kv3('DATA', doc)
    res.block('RERL').data = s2res.reference_list(refs)
    _write(root, mat_path, s2res.write_resource(res))
    return mat_path


def _set_int(obj, key, value):
    obj[key] = s2res.integer(value)


def model_file(game, model_path, pos, frames, uv, indices, groups):
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
    rows['pos'], rows['uv'], rows['normal'] = pos, uv, frames
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
        pos, frames, uv, indices, mat = mesh_arrays(obj, depsgraph, scale)
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
    model_path = f'{FOLDER}/{model_name(obj)}_{_short_hash(pos.tobytes(), frames.tobytes(), uv.tobytes(), indices.tobytes(), groups)}.vmdl'
    _write(root, model_path, model_file(game, model_path, pos, frames, uv, indices, groups))
    return model_path
