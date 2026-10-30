"""The map's baked lighting, fog and sky in Blender (used by map_import.py). The maths follows ValveResourceFormat's
renderer (Renderer/Shaders/common/lighting.slang, lighting.lpv.slang and fog.slang, MIT licence).

Baked light: lightmapped surfaces (lightmap version 8.3 to 8.5) store it as linear second order spherical harmonics:
an HDR DC texture and, per colour channel (8.3, 8.4) or once for the luminance (8.5), a texture with the packed
linear band. Everything else (props) is lit by light probe volumes, ambient cubes in a 3D atlas; those are sampled
per vertex at build time into the _PROBE_LIGHT attribute. Either way the material gets albedo * irradiance as
emission, added to its Principled BSDF so lights added in Blender still light it.

Fog: the map's cubemap fog (towards a blurred sky in the view direction) and gradient fog, by distance from the
camera and height, mixed over each map material for camera rays.
"""
import glob
import math
import os

import bpy
import numpy as np

GROUP = 'Deadlock baked light'
FOG_GROUP = 'Deadlock fog'
PROBE_ATTR = '_PROBE_LIGHT'
SQRT3 = math.sqrt(3)
ZONAL = 0.25 * math.sqrt(5 / (16 * math.pi))
UNIT = 0.0254
# Source2Viewer writes cubemaps as equirectangular images whose centre is Source +X, Blender -Y
SKY_ROTATION = (0.0, 0.0, -math.pi / 2)


class Graph:
    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links

    def put(self, socket, value):
        if hasattr(value, 'node'):
            self.links.new(value, socket)
        else:
            socket.default_value = value

    def node(self, kind, **props):
        n = self.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math(self, op, a, b=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def vec(self, op, a, b=None):
        n = self.node('ShaderNodeVectorMath', operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[3] if op == 'SCALE' else n.inputs[1], b)
        return n.outputs['Value' if op in ('DOT_PRODUCT', 'LENGTH') else 'Vector']

    def split(self, v):
        n = self.node('ShaderNodeSeparateXYZ')
        self.put(n.inputs[0], v)
        return n.outputs

    def combine(self, x, y, z):
        n = self.node('ShaderNodeCombineXYZ')
        for s, v in zip(n.inputs, (x, y, z)):
            self.put(s, v)
        return n.outputs[0]

    def mix(self, op, a, b, factor=1.0):
        n = self.node('ShaderNodeMix', data_type='RGBA', blend_type=op)
        self.put(n.inputs['Factor'], factor)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def image(self, img, vector):
        n = self.node('ShaderNodeTexImage', image=img, interpolation='Linear', extension='EXTEND')
        if vector is not None:
            self.put(n.inputs['Vector'], vector)
        return n.outputs['Color']

    def attribute(self, name, output='Vector'):
        return self.node('ShaderNodeAttribute', attribute_type='GEOMETRY', attribute_name=name).outputs[output]


def load_image(path):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = 'Non-Color'
    return img


def band_in_blender_axes(g, packed):
    """Unpacks a packed linear band to a direction in Blender's axes, without the DC and sqrt(3).

    The renderer turns the unpacked band u into Source's world axes as (-u.z, -u.x, u.y); the glTF import puts
    Source (x, y, z) at Blender (y, -x, z)."""
    u = g.split(g.vec('SUBTRACT', g.vec('SCALE', packed, 2.0), (1, 1, 1)))
    return g.combine(g.math('MULTIPLY', u[0], -1), u[2], u[1])


def sh_factor(g, w, normal):
    """EvaluateLightmapSH / DC for one channel whose band direction (over DC and sqrt 3) is w."""
    length = g.vec('LENGTH', w)
    axis = g.vec('SCALE', w, g.math('DIVIDE', 1.0, g.math('MAXIMUM', length, 1e-5)))
    ratio = g.math('MULTIPLY', length, SQRT3)
    linear = g.math('DIVIDE', g.math('ADD', 0.886227,
                                     g.math('MULTIPLY', g.vec('DOT_PRODUCT', w, normal), 1.023328 * SQRT3)), math.pi)
    cos = g.vec('DOT_PRODUCT', axis, normal)
    zonal = g.math('MULTIPLY', g.math('MULTIPLY', ratio, g.math('ADD', 0.08, g.math('MULTIPLY', ratio, 0.6))), ZONAL)
    zonal = g.math('MULTIPLY', zonal, g.math('SUBTRACT', g.math('MULTIPLY', g.math('MULTIPLY', cos, cos), 3), 1))
    return g.math('MAXIMUM', 0.0, g.math('ADD', linear, zonal))


def baked_light_group(dc_path=None, band_paths=()):
    """Node group: Base Color, Alpha, Normal -> Emission (albedo * the baked irradiance: the lightmap's on lightmapped
    surfaces, the light probes' elsewhere)."""
    tree = bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    for name, kind in (('Base Color', 'NodeSocketColor'), ('Alpha', 'NodeSocketFloat'),
                       ('Normal', 'NodeSocketVector')):
        tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
    tree.interface.new_socket(name='Emission', in_out='OUTPUT', socket_type='NodeSocketColor')
    g = Graph(tree)
    inp, out = g.node('NodeGroupInput'), g.node('NodeGroupOutput')

    # missing attributes read as zero: _LIGHTMAP_UV.z is 1 only on lightmapped surfaces, _PROBE_LIGHT only on the rest
    irradiance = g.attribute(PROBE_ATTR, 'Color')
    if dc_path:
        uvw = g.split(g.attribute('_LIGHTMAP_UV'))
        uv = g.combine(uvw[0], uvw[1], 0.0)
        normal = g.vec('NORMALIZE', inp.outputs['Normal'])
        dc = g.split(g.image(load_image(dc_path), uv))
        bands = [band_in_blender_axes(g, g.image(load_image(p), uv)) for p in band_paths]
        if len(bands) == 1:
            f = sh_factor(g, bands[0], normal)
            lightmap = g.combine(*(g.math('MULTIPLY', dc[i], f) for i in range(3)))
        else:
            # 8.3/8.4: one band per channel; the zonal axis is their luminance-weighted sum, which this
            # approximates with each channel's own axis
            lightmap = g.combine(*(g.math('MULTIPLY', dc[i], sh_factor(g, bands[i], normal)) for i in range(3)))
        irradiance = g.vec('ADD', irradiance, g.vec('SCALE', lightmap, uvw[2]))

    strength = g.node('ShaderNodeValue', label='Baked light strength', name='Strength')
    strength.outputs[0].default_value = 1.0
    scale = g.math('MULTIPLY', inp.outputs['Alpha'], strength.outputs[0])
    albedo_light = g.mix('MULTIPLY', inp.outputs['Base Color'], irradiance)
    g.links.new(g.vec('SCALE', albedo_light, scale), out.inputs['Emission'])
    return tree


def source_of(socket):
    return socket.links[0].from_socket if socket.is_linked else None


def material_output(mat):
    if not mat or not mat.use_nodes:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)


def surface_bsdf(mat):
    """The output, its surface socket and the Principled BSDF under it (first input of any Add Shaders above it),
    or Nones."""
    output = material_output(mat)
    surface = source_of(output.inputs['Surface']) if output else None
    node = surface.node if surface else None
    while node is not None and node.type == 'ADD_SHADER':
        node = source_of(node.inputs[0]).node if node.inputs[0].is_linked else None
    if node is None or node.type != 'BSDF_PRINCIPLED':
        return None, None, None
    return output, surface, node


def add_baked_light(mat, group):
    """Adds the group's emission to a material made of one Principled BSDF. False if it isn't."""
    output, surface, bsdf = surface_bsdf(mat)
    if bsdf is None:
        return False
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    node = nodes.new('ShaderNodeGroup')
    node.node_tree = group
    node.location = bsdf.location.x, bsdf.location.y - 700
    for name in ('Base Color', 'Alpha'):
        src = source_of(bsdf.inputs[name])
        if src:
            links.new(src, node.inputs[name])
        else:
            node.inputs[name].default_value = bsdf.inputs[name].default_value
    normal = source_of(bsdf.inputs['Normal'])
    if normal is None:
        normal = nodes.new('ShaderNodeNewGeometry').outputs['Normal']
    links.new(normal, node.inputs['Normal'])
    emission = nodes.new('ShaderNodeEmission')
    links.new(node.outputs['Emission'], emission.inputs['Color'])
    add = nodes.new('ShaderNodeAddShader')
    links.new(surface, add.inputs[0])
    links.new(emission.outputs[0], add.inputs[1])
    links.new(add.outputs[0], output.inputs['Surface'])
    emission.location = node.location.x + 250, node.location.y
    add.location = output.location.x, output.location.y - 200
    return True


def vmat_dict(mat):
    """The game material, from Source2Viewer's glTF extras (--gltf_export_extras)."""
    vmat = mat.get('vmat') if mat else None
    if vmat is None:
        return {}
    return vmat.to_dict() if hasattr(vmat, 'to_dict') else dict(vmat)


def vmat_params(mat):
    get = vmat_dict(mat)
    return dict(get.get('IntParams') or {}), dict(get.get('FloatParams') or {})


def is_environment(mat):
    return str(vmat_dict(mat).get('ShaderName', '')).startswith('environment_')


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def decode_paint(objects):
    """Source2Viewer writes the vertex paint as stored (sRGB bytes) and Blender reads it as linear; the game's
    shaders decode it, and Deadlock's environment_* shaders fade it to white by its alpha (unpainted vertices are
    all zero). Returns the number of meshes changed."""
    count = 0
    for ob in objects:
        attr = ob.data.color_attributes.get('Color')
        mats = [s.material for s in ob.material_slots if s.material]
        if attr is None or not mats:
            continue
        rgba = np.empty(len(attr.data) * 4, np.float32)
        attr.data.foreach_get('color', rgba)
        rgba = rgba.reshape(-1, 4)
        rgb = srgb_to_linear(np.clip(rgba[:, :3], 0.0, 1.0))
        if any(is_environment(m) for m in mats):
            rgb = 1.0 + (rgb - 1.0) * rgba[:, 3:4]
        rgba[:, :3] = rgb
        attr.data.foreach_set('color', rgba.reshape(-1))
        count += 1
    return count


def param(params, name, default):
    """Layered shaders number their parameters (g_flSelfIllumScale1); the first one found."""
    for key in (name, name + '1', name + '0'):
        if key in params:
            return float(params[key])
    return default


def paint_strength(mat, ints, floats):
    """How much the vertex paint tints the first layer's albedo. Deadlock's environment_* shaders always read the
    paint, unless the layer turns it off (g_bModelTint1, g_fVertexColorStrength1); pbr only with F_VERTEX_COLOR or
    F_PAINT_VERTEX_COLORS, otherwise the stream holds other data, often black."""
    if is_environment(mat):
        if not ints.get('g_bModelTint1', 1):
            return 0.0
    elif not (ints.get('F_VERTEX_COLOR') or ints.get('F_PAINT_VERTEX_COLORS')):
        return 0.0
    return float(floats.get('g_fVertexColorStrength1', 1.0))


def set_vertex_paint(mat, bsdf, strength):
    """The glTF import multiplies the base colour by COLOR_0 fully; the game by the paint strength."""
    mix = source_of(bsdf.inputs['Base Color'])
    while mix is not None and mix.node.type == 'MIX' and mix.node.blend_type == 'MULTIPLY':
        inputs = [s for s in mix.node.inputs if s.enabled and s.type == 'RGBA' and s.is_linked]
        if any(s.links[0].from_node.type in ('VERTEX_COLOR', 'ATTRIBUTE') for s in inputs):
            if not mix.node.inputs['Factor'].is_linked:
                mix.node.inputs['Factor'].default_value = min(max(strength, 0.0), 1.0)
            return
        # the object tint's multiply is over the paint's
        mix = source_of(inputs[0]) if inputs else None


def apply_vmat(mat, glass_masks=None):
    """What the glTF export loses of the game's shading: unlit and additive materials show their colour as it is,
    glass lets the scene through, and self-illumination is scaled by g_flSelfIllumScale. Returns 'unlit', 'glass',
    'self_illum' or None."""
    ints, floats = vmat_params(mat)
    output = material_output(mat)
    surface = source_of(output.inputs['Surface']) if output else None
    if surface is None or surface.node.type != 'BSDF_PRINCIPLED':
        return None
    bsdf = surface.node
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    set_vertex_paint(mat, bsdf, paint_strength(mat, ints, floats))
    if ints.get('F_UNLIT'):
        emission = nodes.new('ShaderNodeEmission')
        color = source_of(bsdf.inputs['Base Color'])
        if color:
            links.new(color, emission.inputs['Color'])
        else:
            emission.inputs['Color'].default_value = bsdf.inputs['Base Color'].default_value
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        alpha = source_of(bsdf.inputs['Alpha'])
        if ints.get('F_ADDITIVE_BLEND'):
            if alpha:
                links.new(alpha, emission.inputs['Strength'])
            mix = nodes.new('ShaderNodeAddShader')
            links.new(transparent.outputs[0], mix.inputs[0])
            links.new(emission.outputs[0], mix.inputs[1])
        else:
            mix = nodes.new('ShaderNodeMixShader')
            if alpha:
                links.new(alpha, mix.inputs[0])
            else:
                mix.inputs[0].default_value = bsdf.inputs['Alpha'].default_value
            links.new(transparent.outputs[0], mix.inputs[1])
            links.new(emission.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs['Surface'])
        emission.location = bsdf.location.x + 300, bsdf.location.y + 300
        mix.location = output.location.x, output.location.y + 200
        return 'unlit'
    if ints.get('F_GLASS'):
        mask = glass_mask(mat, glass_masks or {})
        if mask is not None:
            apply_glass(mat, bsdf, mask)
            return 'glass'
    if ints.get('F_SELF_ILLUM'):
        bsdf.inputs['Emission Strength'].default_value *= param(floats, 'g_flSelfIllumScale', 1.0)
        return 'self_illum'
    return None


def glass_mask(mat, glass_masks):
    """The glass mask (g_tGlass, red): the constant the material editor made the texture from, else the texture
    the build decompiled (glass_masks: texture path -> file). A float, a socket, or None."""
    vmat = vmat_dict(mat)
    constant = (vmat.get('VectorParams') or {}).get('TextureGlassMask1')
    if constant is not None:
        return float(constant[0])
    path = glass_masks.get((vmat.get('TextureParams') or {}).get('g_tGlass'))
    if not path:
        return None
    nodes = mat.node_tree.nodes
    uv = next((source_of(n.inputs['Vector']) for n in nodes if n.type == 'TEX_IMAGE' and n.inputs['Vector'].is_linked),
              None)
    g = Graph(mat.node_tree)
    color = g.image(load_image(path), uv)
    color.node.extension = 'REPEAT'
    return g.split(color)[0]


def apply_glass(mat, bsdf, mask):
    """F_GLASS (Citadel pbr): the diffuse gives way to transmission where the mask is, the specular stays. The
    scene behind shows through albedo ** (1 / N.V) (thicker at grazing angles), scaled by the mask, the
    non-metal part and what the specular (F0 0.04) leaves."""
    g = Graph(mat.node_tree)
    albedo = source_of(bsdf.inputs['Base Color']) or tuple(bsdf.inputs['Base Color'].default_value)[:3]
    g.links.new(g.vec('SCALE', albedo, g.math('SUBTRACT', 1.0, mask)), bsdf.inputs['Base Color'])
    geometry = g.node('ShaderNodeNewGeometry')
    facing = g.vec('DOT_PRODUCT', geometry.outputs['Normal'], geometry.outputs['Incoming'])
    exponent = g.math('DIVIDE', 1.0, g.math('MAXIMUM', g.math('ABSOLUTE', facing), 0.01))
    rgb = g.split(g.vec('MAXIMUM', albedo, (0.01, 0.01, 0.01)))
    through = g.combine(*(g.math('POWER', c, exponent) for c in rgb))
    metallic = source_of(bsdf.inputs['Metallic']) or bsdf.inputs['Metallic'].default_value
    amount = g.math('MULTIPLY', g.math('MULTIPLY', mask, g.math('SUBTRACT', 1.0, metallic)), 0.96)
    transparent = g.node('ShaderNodeBsdfTransparent')
    g.links.new(g.vec('SCALE', through, amount), transparent.inputs['Color'])
    output = material_output(mat)
    add = g.node('ShaderNodeAddShader')
    g.links.new(source_of(output.inputs['Surface']), add.inputs[0])
    g.links.new(transparent.outputs[0], add.inputs[1])
    g.links.new(add.outputs[0], output.inputs['Surface'])


def exclude_from_light(light_ob, objects, name):
    """Light linking: the light skips these objects."""
    coll = bpy.data.collections.new(name)
    for ob in objects:
        coll.objects.link(ob)
    light_ob.light_linking.receiver_collection = coll
    for item in coll.collection_objects:
        item.light_linking.link_state = 'EXCLUDE'
    return coll


# ---------------------------------------------------------------- light probes

def load_probe_atlas(folder):
    """The irradiance atlas, (depth, height, width, 3) with row 0 the texture's first row, from Source2Viewer's
    one image per depth slice. None if it isn't there."""
    files = glob.glob(os.path.join(folder, '**', 'env_light_probe_volume_atlas_z*.exr'), recursive=True)
    if not files:
        return None
    files.sort(key=lambda f: int(f.rsplit('_z', 1)[1].split('.')[0]))
    atlas = None
    for z, f in enumerate(files):
        img = bpy.data.images.load(f)
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        if atlas is None:
            atlas = np.empty((len(files), h, w, 3), np.float16)
        # Blender's rows go bottom up
        atlas[z] = px.reshape(h, w, 4)[::-1, :, :3]
    return atlas


def trilinear(atlas, x, y, z):
    """Samples the atlas at texel-centre coordinates (texel i's centre is i)."""
    d, h, w = atlas.shape[:3]
    out = np.zeros((len(x), 3), np.float32)
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = (x - x0)[:, None], (y - y0)[:, None], (z - z0)[:, None]
    x0, y0, z0 = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)
    for dz in (0, 1):
        wz = fz if dz else 1 - fz
        zi = np.clip(z0 + dz, 0, d - 1)
        for dy in (0, 1):
            wy = fy if dy else 1 - fy
            yi = np.clip(y0 + dy, 0, h - 1)
            for dx in (0, 1):
                wx = fx if dx else 1 - fx
                xi = np.clip(x0 + dx, 0, w - 1)
                out += atlas[zi, yi, xi].astype(np.float32) * (wx * wy * wz)
    return out


def sample_probes(atlas, volumes, points, normals):
    """Probe irradiance at Source-space points for Source-space unit normals (SampleProbeAmbientCube): each point
    takes the first volume, in priority order, that holds it, else the nearest one."""
    n = len(points)
    choice = np.full(n, -1, np.int64)
    locals_ = np.zeros((n, 3))
    nearest = np.full(n, np.inf)
    nearest_vol = np.zeros(n, np.int64)
    for i, v in enumerate(volumes):
        rot = np.array(v['rotation'])
        mins, maxs = np.array(v['mins']), np.array(v['maxs'])
        local = (points - np.array(v['origin'])) @ rot
        norm = (local - mins) / (maxs - mins)
        todo = choice < 0
        inside = todo & np.all((norm >= 0) & (norm <= 1), axis=1)
        choice[inside] = i
        locals_[inside] = norm[inside]
        dist = np.linalg.norm(np.maximum(mins - local, 0) + np.maximum(local - maxs, 0), axis=1)
        closer = todo & ~inside & (dist < nearest)
        nearest[closer] = dist[closer]
        nearest_vol[closer] = i
    rest = choice < 0
    for i in np.unique(nearest_vol[rest]):
        sel = rest & (nearest_vol == i)
        v = volumes[i]
        local = (points[sel] - np.array(v['origin'])) @ np.array(v['rotation'])
        locals_[sel] = np.clip((local - np.array(v['mins'])) / (np.array(v['maxs']) - np.array(v['mins'])), 0, 1)
        choice[sel] = i
    face_depth = atlas.shape[0] // 6
    out = np.zeros((n, 3), np.float32)
    for i in np.unique(choice):
        sel = choice == i
        v = volumes[i]
        size, offset = np.array(v['size'], float), np.array(v['offset'], float)
        # half a probe in from the edges, as the game's borders keep it, so filtering stays in this volume
        local = np.clip(locals_[sel], 0.5 / size, (size - 0.5) / size)
        texel = local * size + offset - 0.5
        rot = np.array(v['rotation'])
        nl = normals[sel] @ rot
        weights = nl * nl
        for axis in range(3):
            face = np.where(nl[:, axis] > 0, axis, axis + 3)
            sample = trilinear(atlas, texel[:, 0], texel[:, 1], texel[:, 2] + face * face_depth)
            out[sel] += sample * weights[:, axis:axis + 1]
    return out


def to_source(v):
    """Blender (x, y, z) to Source axes, which the glTF import put at Blender (y, -x, z)."""
    return np.column_stack([-v[:, 1], v[:, 0], v[:, 2]])


def bake_probes(objects, atlas, volumes):
    """Writes each mesh's per-vertex probe irradiance (for its vertex normals) into _PROBE_LIGHT."""
    meshes, points, normals = [], [], []
    for ob in objects:
        me = ob.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get('co', co)
        no = np.empty(len(me.vertices) * 3, np.float32)
        me.vertex_normals.foreach_get('vector', no)
        m = np.array(ob.matrix_world)
        points.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        normals.append(no.reshape(-1, 3) @ np.linalg.inv(m[:3, :3]))
        meshes.append(me)
    if not meshes:
        return 0
    pts = to_source(np.concatenate(points).astype(np.float64)) / UNIT
    nrm = to_source(np.concatenate(normals).astype(np.float64))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
    light = sample_probes(atlas, volumes, pts, nrm)
    start = 0
    for me in meshes:
        n = len(me.vertices)
        attr = me.attributes.new(PROBE_ATTR, 'FLOAT_COLOR', 'POINT')
        rgba = np.ones((n, 4), np.float32)
        rgba[:, :3] = light[start:start + n]
        attr.data.foreach_set('color', rgba.reshape(-1))
        start += n
    return start


# ---------------------------------------------------------------- sky and fog

def downsample(img, factor):
    """A box-filtered copy of an image, a stand-in for the cubemap's blurred mips."""
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h // factor, factor, w // factor, factor, 4).mean(axis=(1, 3))
    small = bpy.data.images.new(f'{img.name} mip{int(math.log2(factor))}', w // factor, h // factor, alpha=True,
                                float_buffer=True)
    small.pixels.foreach_set(px.reshape(-1))
    small.colorspace_settings.name = img.colorspace_settings.name
    small.pack()
    return small


def sky_vector(g, direction):
    mapping = g.node('ShaderNodeMapping', vector_type='POINT')
    mapping.inputs['Rotation'].default_value = SKY_ROTATION
    g.put(mapping.inputs['Vector'], direction)
    return mapping.outputs[0]


def env_image(g, img, vector):
    n = g.node('ShaderNodeTexEnvironment', image=img, interpolation='Linear')
    g.put(n.inputs['Vector'], vector)
    return n.outputs['Color']


def make_world(name, sky_img, strength):
    """The sky as the background the camera sees. It doesn't light the scene: the map's light is baked, and the game
    lights nothing from the sky at render time either."""
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    g = Graph(tree)
    coords = g.node('ShaderNodeTexCoord').outputs['Generated']
    bg = g.node('ShaderNodeBackground')
    g.put(bg.inputs['Color'], env_image(g, sky_img, sky_vector(g, coords)))
    bg.inputs['Strength'].default_value = strength
    black = g.node('ShaderNodeBackground')
    black.inputs['Color'].default_value = (0, 0, 0, 1)
    path = g.node('ShaderNodeLightPath')
    mix = g.node('ShaderNodeMixShader')
    g.links.new(path.outputs['Is Camera Ray'], mix.inputs[0])
    g.links.new(black.outputs[0], mix.inputs[1])
    g.links.new(bg.outputs[0], mix.inputs[2])
    out = g.node('ShaderNodeOutputWorld')
    g.links.new(mix.outputs[0], out.inputs['Surface'])
    return world


def ramp(g, x, start, end, exponent):
    """clamp((x - start) / (end - start), 1e-4, 1) ** exponent."""
    t = g.math('MULTIPLY', g.math('SUBTRACT', x, start), 1.0 / (end - start) if end != start else 0.0)
    t = g.math('MINIMUM', g.math('MAXIMUM', t, 1e-4), 1.0)
    return g.math('POWER', t, exponent)


def fog_group(cube=None, gradient=None, sky_mips=None, sky_strength=1.0, sky_face=1024):
    """Node group: Shader, Alpha -> Shader with the map's fog over it (camera rays only)."""
    tree = bpy.data.node_groups.new(FOG_GROUP, 'ShaderNodeTree')
    tree.interface.new_socket(name='Shader', in_out='INPUT', socket_type='NodeSocketShader')
    tree.interface.new_socket(name='Alpha', in_out='INPUT', socket_type='NodeSocketFloat')
    tree.interface.new_socket(name='Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    g = Graph(tree)
    inp, out = g.node('NodeGroupInput'), g.node('NodeGroupOutput')
    strength = g.node('ShaderNodeValue', label='Fog strength', name='Strength')
    strength.outputs[0].default_value = 1.0
    distance = g.node('ShaderNodeCameraData').outputs['View Distance']
    height = g.split(g.node('ShaderNodeNewGeometry').outputs['Position'])[2]
    camera = g.node('ShaderNodeLightPath').outputs['Is Camera Ray']
    gate = g.math('MULTIPLY', g.math('MULTIPLY', camera, inp.outputs['Alpha']), strength.outputs[0])
    shader = inp.outputs['Shader']

    def layer(shader, fog_color, amount):
        emission = g.node('ShaderNodeEmission')
        g.put(emission.inputs['Color'], fog_color)
        mix = g.node('ShaderNodeMixShader')
        g.put(mix.inputs[0], g.math('MULTIPLY', amount, gate, clamp=True))
        g.put(mix.inputs[1], shader)
        g.links.new(emission.outputs[0], mix.inputs[2])
        return mix.outputs[0]

    if gradient:
        f = gradient
        amount = ramp(g, distance, f['start'], f['end'], f['falloff'])
        if f['height_start'] != f['height_end']:
            amount = g.math('MULTIPLY', amount, ramp(g, height, f['height_end'], f['height_start'],
                                                     f['vertical_exponent']))
        shader = layer(shader, list(f['color']) + [1.0], g.math('MULTIPLY', amount, f['opacity']))

    if cube and sky_mips:
        f = cube
        amount = ramp(g, distance, f['start'], f['end'], f['falloff'])
        if f['height_width'] > 0:
            hs, he = f['height_start'], f['height_start'] + f['height_width']
            amount = g.math('MULTIPLY', amount, ramp(g, height, he, hs, f['height_exponent']))
        # the game blurs nearer fog more: mip (1 - amount * lod bias) * (log2(face size) - 3)
        lod = g.math('MULTIPLY', g.math('SUBTRACT', 1.0, g.math('MULTIPLY', amount, f['lod_bias']), clamp=True),
                     math.log2(sky_face) - 3)
        view = g.vec('SCALE', g.node('ShaderNodeNewGeometry').outputs['Incoming'], -1.0)
        vector = sky_vector(g, view)
        (sharp_lod, sharp), (blur_lod, blurred) = sky_mips
        t = g.math('MULTIPLY', g.math('SUBTRACT', lod, sharp_lod), 1.0 / (blur_lod - sharp_lod), clamp=True)
        color = g.mix('MIX', env_image(g, sharp, vector), env_image(g, blurred, vector), t)
        color = g.vec('SCALE', color, sky_strength)
        shader = layer(shader, color, g.math('MULTIPLY', amount, f['opacity']))

    g.links.new(shader, out.inputs['Shader'])
    return tree


def alpha_source(mat):
    bsdf = surface_bsdf(mat)[2]
    if bsdf is not None:
        return source_of(bsdf.inputs['Alpha']), bsdf.inputs['Alpha'].default_value
    return None, 1.0


def add_fog(mat, group):
    output = material_output(mat)
    surface = source_of(output.inputs['Surface']) if output else None
    if surface is None:
        return False
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    node = nodes.new('ShaderNodeGroup')
    node.node_tree = group
    node.location = output.location.x, output.location.y - 400
    links.new(surface, node.inputs['Shader'])
    alpha, default = alpha_source(mat)
    if alpha is not None:
        links.new(alpha, node.inputs['Alpha'])
    else:
        node.inputs['Alpha'].default_value = default
    links.new(node.outputs[0], output.inputs['Surface'])
    return True
