"""Deadlock's environment_parallax_layered windows in Blender (used by map_lighting.apply_vmat): a glass surface
over two interior layers (a shade or curtain, then the room) that sit g_flLayer2Offset and g_flLayer3Offset behind
it and glow. The glTF export only has the glass, so the windows came out dark.

Follows the game's pixel shader (shaders/vfx/environment_parallax_layered_vulkan_60_ps.vcs, decompiled with
Source2Viewer's --shader_combo): each interior layer's UV moves back along the view ray, its tangent-space
direction clamped to g_flParallaxViewAngleLimit, by offset / N.V (the offsets are in 1/1024 UV, as the material
editor divides them); g_tMask's red is the glass and g_tMask2's red the shade over the room; each layer's colour
alpha is its glow mask. The G-buffer gets the albedo, tinted towards each emissive tint where that layer glows, and
one self-illumination amount, which the lighting pass turns into albedo * amount. The emissive tints reach the
shader as 2 ** level * linear tint (the .vfx's Expression), so the level counts twice.
"""
import numpy as np

if __package__:
    from .map_lighting import (Graph, average_color, cached_image, color_correct_matrix, is_identity, node_group,
                               source_of, srgb_to_linear, transform_color, LUMINANCE_WEIGHTS)
else:
    from map_lighting import (Graph, average_color, cached_image, color_correct_matrix, is_identity, node_group,
                              source_of, srgb_to_linear, transform_color, LUMINANCE_WEIGHTS)

SHADER = 'environment_parallax_layered.vfx'
# the second UV set as a real UV map, so the Tangent node can give its tangents (map_import makes it)
UV_MAP = 'UV2'


def parallax_group():
    """uv - clamp(view in tangent space, -Limit, Limit) * Depth / N.V, Depth in UV units."""
    def build(g, uv, depth, limit):
        geo = g.node('ShaderNodeNewGeometry')
        normal, towards_eye = geo.outputs['Normal'], geo.outputs['Incoming']
        tangent = g.node('ShaderNodeTangent', direction_type='UV_MAP', uv_map=UV_MAP).outputs['Tangent']
        bitangent = g.vec('NORMALIZE', g.vec('CROSS_PRODUCT', normal, tangent))
        low = g.math('MULTIPLY', limit, -1.0)

        def clamped(axis):
            return g.math('MAXIMUM', g.math('MINIMUM', g.vec('DOT_PRODUCT', towards_eye, axis), limit), low)
        step = g.math('DIVIDE', depth, g.math('MAXIMUM', g.vec('DOT_PRODUCT', towards_eye, normal), 0.05))
        shift = g.combine(g.math('MULTIPLY', clamped(tangent), step), g.math('MULTIPLY', clamped(bitangent), step),
                          0.0)
        return [g.vec('SUBTRACT', uv, shift)]
    return node_group('Deadlock parallax', [('UV', 'Vector'), ('Depth', 'Float'), ('Limit', 'Float')],
                      [('UV', 'Vector')], build)


def emissive_tint(vectors, floats, n):
    """g_vEmissiveTint as the shader gets it: Expression(pow(2, g_flEmissiveLevel) * SrgbGammaToLinear(this))."""
    rgb = srgb_to_linear(np.asarray(list(vectors.get(f'g_vEmissiveTint{n}') or (1, 1, 1))[:3], float))
    return rgb * 2.0 ** float(floats.get(f'g_flEmissiveLevel{n}', 0.0))


def apply_parallax(mat, bsdf, ints, floats, vectors, textures):
    """Lays the interior layers under the glass. textures: texture path -> file (the build's layer textures).
    Returns whether it applied."""
    vmat = mat.get('vmat')
    paths = dict((vmat.to_dict() if hasattr(vmat, 'to_dict') else dict(vmat)).get('TextureParams') or {})

    def image(key, color):
        path = textures.get(paths.get(key))
        if not path:
            return None
        return cached_image(path, 'sRGB' if color else 'Non-Color', 'CHANNEL_PACKED')

    shade_img, room_img = image('g_tColor2', True), image('g_tColor3', True)
    if shade_img is None and room_img is None:
        return False
    g = Graph(mat.node_tree)
    uv1 = g.node('ShaderNodeUVMap').outputs['UV']
    uv2 = g.node('ShaderNodeUVMap', uv_map=UV_MAP).outputs['UV']
    paint = g.node('ShaderNodeVertexColor', layer_name='Color').outputs['Color']

    def sample(img, uv):
        n = g.node('ShaderNodeTexImage', image=img, interpolation='Linear', extension='REPEAT')
        g.put(n.inputs['Vector'], uv)
        return n.outputs['Color'], n.outputs['Alpha']

    uvs = {}

    def layer_uv(n):
        if n not in uvs:
            secondary = ints.get(f'g_bLayer{n}UseSecondaryUV', 1)
            uvs[n] = g.call(parallax_group(), uv2 if secondary else uv1,
                            float(floats.get(f'g_flLayer{n}Offset', 0.0)) / 1024.0,
                            float(floats.get(f'g_flParallaxViewAngleLimit{n}', 1.0)))[0]
        return uvs[n]

    def layer(img, n):
        if img is None:
            return (0.0, 0.0, 0.0, 1.0), 0.0
        color, alpha = sample(img, layer_uv(n))
        csb = list(vectors.get(f'g_vAlbedoContrastSaturationBrightness{n}') or (1, 1, 1))[:3]
        correct = color_correct_matrix(csb, average_color(img))
        if not is_identity(correct):
            color = transform_color(g, color, correct)
        strength = min(max(float(floats.get(f'g_fVertexColorStrength{n}', 0.0)), 0.0), 1.0)
        if strength:
            color = g.mix('MULTIPLY', color, paint, strength)
        return color, alpha

    shade, shade_alpha = layer(shade_img, 2)
    room, room_alpha = layer(room_img, 3)
    mask1_img, mask2_img = image('g_tMask', False), image('g_tMask2', False)
    glass = g.split(sample(mask1_img, uv1)[0])[0] if mask1_img else float((vectors.get('TextureMask1') or [0])[0])
    shade_cover = (g.split(sample(mask2_img, layer_uv(2))[0])[0] if mask2_img and shade_img
                   else 0.0 if shade_img is None else float((vectors.get('TextureMask2') or [1])[0]))
    through, room_through = g.math('SUBTRACT', 1.0, glass), g.math('SUBTRACT', 1.0, shade_cover)
    tint2, tint3 = emissive_tint(vectors, floats, 2), emissive_tint(vectors, floats, 3)
    glow2 = g.math('MULTIPLY', g.math('MULTIPLY', shade_alpha, through),
                   float(floats.get('g_flEmissiveLevel2', 0.0)) * float(tint2 @ LUMINANCE_WEIGHTS))
    glow3 = g.math('MULTIPLY', g.math('MULTIPLY', room_alpha, room_through),
                   float(floats.get('g_flEmissiveLevel3', 0.0)) * float(tint3 @ LUMINANCE_WEIGHTS))
    room_seen = g.math('MULTIPLY', through, room_through)

    albedo = source_of(bsdf.inputs['Base Color']) or tuple(bsdf.inputs['Base Color'].default_value)
    color = g.mix('MIX', g.mix('MIX', room, shade, shade_cover), albedo, glass)
    toward2 = g.math('MULTIPLY', g.math('SUBTRACT', 1.0, room_seen), g.math('MINIMUM', glow2, 1.0))
    color = g.mix('MIX', color, g.mix('MULTIPLY', color, tuple(tint2) + (1.0,)), toward2)
    color = g.mix('MIX', color, g.mix('MULTIPLY', color, tuple(tint3) + (1.0,)),
                  g.math('MULTIPLY', room_seen, g.math('MINIMUM', glow3, 1.0)))
    amount = g.mix('MIX', g.math('MULTIPLY', through, glow2), glow3, room_seen)
    amount = g.split(amount)[0]
    g.links.new(color, bsdf.inputs['Base Color'])
    g.links.new(color, bsdf.inputs['Emission Color'])
    g.links.new(amount, bsdf.inputs['Emission Strength'])
    return True


def add_uv_maps(objects):
    """The _UV2 attribute as a UV map on the meshes of parallax materials (see UV_MAP). Returns how many."""
    count = 0
    for ob in objects:
        me = ob.data
        attr = me.attributes.get('_UV2')
        if attr is None or UV_MAP in me.uv_layers or not any(
                s.material and (s.material.get('vmat') or {}).get('ShaderName') == SHADER for s in ob.material_slots):
            continue
        uvw = np.empty(len(attr.data) * 3, np.float32)
        attr.data.foreach_get('vector', uvw)
        index = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get('vertex_index', index)
        layer = me.uv_layers.new(name=UV_MAP, do_init=False)
        layer.data.foreach_set('uv', uvw.reshape(-1, 3)[index, :2].reshape(-1))
        count += 1
    return count
