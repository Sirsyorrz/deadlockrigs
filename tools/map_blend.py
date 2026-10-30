"""Deadlock's environment_blend albedo in Blender (used by map_lighting.apply_vmat). The glTF export only has the
first layer; this lays layers 2 and 3, their reveal masks, border tints, the world overlay and the UV overlay over
it, following ValveResourceFormat (Renderer/Shaders/citadel/environment_blend.frag.slang,
common/surface_environment_layers.slang and common/planar_projection.slang, MIT licence). Normals, roughness and
metalness stay layer 1's.

UVs are worked in the game's convention (v down) and flipped back for Blender's image nodes; world positions are
Source inches, from Blender metres and axes.
"""
import bpy

from map_lighting import (Graph, average_color, color_correct_matrix, color_tint_matrix, is_identity, srgb_to_linear,
                          transform_color, vmat_dict, UNIT)
import numpy as np

from map_environment import BLEND_TEXTURES as TEXTURES
SHADER = 'environment_blend.vfx'
NO_TRANSFORM = (1.0, 0.0, 0.0, 0.0)


class Blend:
    def __init__(self, mat, ints, floats, vectors, textures):
        self.mat, self.ints, self.floats, self.vectors = mat, ints, floats, vectors
        self.paths = dict(textures)
        self.g = Graph(mat.node_tree)
        self._uv, self._uv2, self._pos, self._normal, self._planar, self._paint = (None,) * 6

    def i(self, name, default=0):
        return int(self.ints.get(name, default))

    def f(self, name, default=0.0):
        return float(self.floats.get(name, default))

    def v(self, name, default):
        value = self.vectors.get(name)
        return list(value)[:len(default)] if value is not None else list(default)

    def image(self, key, color):
        path = self.paths.get(key)
        if not path:
            return None
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = 'sRGB' if color else 'Non-Color'
        img.alpha_mode = 'CHANNEL_PACKED'
        return img

    # coordinates

    def uv(self, secondary=False):
        g = self.g
        if secondary:
            if self._uv2 is None:
                a = g.split(g.attribute('_UV2'))
                self._uv2 = g.combine(a[0], g.math('SUBTRACT', 1.0, a[1]), 0.0)
            return self._uv2
        if self._uv is None:
            a = g.split(g.node('ShaderNodeUVMap').outputs['UV'])
            self._uv = g.combine(a[0], g.math('SUBTRACT', 1.0, a[1]), 0.0)
        return self._uv

    def transform(self, uv, srt):
        """ApplyUvTransform: scale * rotate(uv + translation / scale)."""
        scale, rotation, tu, tv = (list(srt) + [0, 0, 0, 0])[:4]
        if np.allclose((scale, rotation, tu, tv), NO_TRANSFORM):
            return uv
        g = self.g
        safe = scale or 1.0
        shifted = g.vec('ADD', uv, (tu / safe, tv / safe, 0.0))
        c, s = np.cos(np.radians(rotation)), np.sin(np.radians(rotation))
        x, y = g.split(shifted)[0], g.split(shifted)[1]
        rx = g.math('SUBTRACT', g.math('MULTIPLY', x, c), g.math('MULTIPLY', y, s))
        ry = g.math('ADD', g.math('MULTIPLY', x, s), g.math('MULTIPLY', y, c))
        return g.vec('SCALE', g.combine(rx, ry, 0.0), scale)

    def to_blender_uv(self, uv):
        g = self.g
        a = g.split(uv)
        return g.combine(a[0], g.math('SUBTRACT', 1.0, a[1]), 0.0)

    def position(self):
        """Source world position in inches."""
        if self._pos is None:
            g = self.g
            p = g.split(g.node('ShaderNodeNewGeometry').outputs['Position'])
            self._pos = [g.math('MULTIPLY', p[1], -1 / UNIT), g.math('MULTIPLY', p[0], 1 / UNIT),
                         g.math('MULTIPLY', p[2], 1 / UNIT)]
        return self._pos

    def planar(self):
        """GetPlanarAxes as per-axis weights: (major, median) indicators and the median plane's blend weight."""
        if self._planar is None:
            g = self.g
            n = g.split(g.node('ShaderNodeNewGeometry').outputs['True Normal'])
            a = [g.math('ABSOLUTE', n[1]), g.math('ABSOLUTE', n[0]), g.math('ABSOLUTE', n[2])]
            gt = lambda x, y: g.math('GREATER_THAN', x, y)
            lt = lambda x, y: g.math('LESS_THAN', x, y)
            both = lambda x, y: g.math('MULTIPLY', x, y)
            not_ = lambda x: g.math('SUBTRACT', 1.0, x)
            major0 = both(gt(a[0], a[1]), gt(a[0], a[2]))
            major1 = both(not_(major0), gt(a[1], a[2]))
            major = [major0, major1, not_(g.math('ADD', major0, major1))]
            minor0 = both(lt(a[0], a[1]), lt(a[0], a[2]))
            minor1 = both(not_(minor0), lt(a[1], a[2]))
            minor = [minor0, minor1, not_(g.math('ADD', minor0, minor1))]
            median = [not_(g.math('ADD', major[k], minor[k])) for k in range(3)]

            def pick(weights):
                return g.math('ADD', g.math('ADD', both(weights[0], a[0]), both(weights[1], a[1])),
                              both(weights[2], a[2]))

            def plane(x):
                return g.math('MULTIPLY', g.math('SUBTRACT', x, 0.5773), 2.3657441, clamp=True)
            wa, wb = plane(pick(major)), plane(pick(median))
            w_median = g.math('DIVIDE', wb, g.math('MAXIMUM', g.math('ADD', wa, wb), 1e-5))
            self._planar = major, median, w_median
        return self._planar

    def planar_uv(self, weights, tiling):
        """GetPlanarTexCoord for whichever axis the weights pick, divided by the tiling."""
        g = self.g
        x, y, z = self.position()
        neg = lambda s: g.math('MULTIPLY', s, -1.0)
        per_axis = [(y, neg(z)), (x, neg(z)), (x, neg(y))]
        u = v = 0.0
        for w, (pu, pv) in zip(weights, per_axis):
            u = g.math('ADD', u, g.math('MULTIPLY', w, pu))
            v = g.math('ADD', v, g.math('MULTIPLY', w, pv))
        inv = 1.0 / (tiling or 1.0)
        return g.combine(g.math('MULTIPLY', u, inv), g.math('MULTIPLY', v, inv), 0.0)

    def sample(self, img, game_uv):
        n = self.g.node('ShaderNodeTexImage', image=img, interpolation='Linear', extension='REPEAT')
        self.g.links.new(self.to_blender_uv(game_uv), n.inputs['Vector'])
        return n.outputs['Color'], n.outputs['Alpha']

    def paint(self):
        if self._paint is None:
            self._paint = self.g.node('ShaderNodeVertexColor', layer_name='Color').outputs['Color']
        return self._paint

    # layers

    def layer_coords(self, n):
        if self.i(f'F_LAYER{n}_BIPLANAR'):
            major, _median, _w = self.planar()
            return self.planar_uv(major, self.f(f'g_flBiPlanarTiling{n}', 196.0))
        return self.transform(self.uv(), self.v(f'g_vScaleRotationTranslation{n}', NO_TRANSFORM))

    def tint_mask(self, alpha, n):
        g = self.g
        contrast, brightness = self.v(f'g_vLayerTintMaskContrastBrightness{n}', (1.0, 0.0))
        return g.math('ADD', g.math('MULTIPLY', g.math('ADD', alpha, brightness - 0.5), contrast), 0.5, clamp=True)

    def layer(self, n):
        """SampleLayer for layer 2 or 3: albedo and tint mask sockets, or None without its colour texture."""
        img = self.image(f'g_tColor{n}', True)
        if img is None:
            return None
        g = self.g
        color, alpha = self.sample(img, self.layer_coords(n))
        mask = self.tint_mask(alpha, n)
        correct = color_correct_matrix(self.v(f'g_vAlbedoContrastSaturationBrightness{n}', (1, 1, 1)),
                                       average_color(img))
        albedo = color if is_identity(correct) else transform_color(g, color, correct)
        tint = color_tint_matrix(srgb_to_linear(np.asarray(self.v(f'g_vColorTint{n}', (1, 1, 1)), float)),
                                 self.i(f'g_nTextureColorTintMode{n}'))
        if not is_identity(tint):
            tinted = transform_color(g, albedo, tint)
            albedo = g.mix('MIX', albedo, tinted, mask if self.i(f'g_bMaskColorTint{n}', 1) else 1.0)
        if self.i(f'g_bModelTint{n}', 1):
            strength = self.f(f'g_fVertexColorStrength{n}', 1.0)
            amount = g.math('MULTIPLY', mask, strength) if self.i(f'g_bMaskVertexColorTint{n}', 1) else strength
            albedo = g.mix('MIX', albedo, g.mix('MULTIPLY', albedo, self.paint()), amount)
        return albedo, mask

    def reveal_threshold(self, n):
        img = self.image(f'g_tRevealMask{n}', False)
        if img is None:
            return 0.5
        g = self.g
        uv = self.transform(self.uv(bool(self.i(f'g_bRevealUseSecondaryUv{n}'))),
                            self.v(f'g_vRevealScaleRotationTranslation{n}', NO_TRANSFORM))
        reveal = g.split(self.sample(img, uv)[0])[0]
        if self.i(f'g_bRevealInvert{n}'):
            reveal = g.math('SUBTRACT', 1.0, reveal)
        half = self.f(f'g_flRevealOffset{n}') * 0.5
        mid = 0.5 + half
        softness = self.f(f'g_flRevealSoftness{n}', 0.125)
        low = g.math('MULTIPLY', g.math('MULTIPLY', reveal, 2.0, clamp=True), mid)
        high = g.math('ADD', g.math('MULTIPLY', g.math('MULTIPLY', g.math('SUBTRACT', reveal, 0.5), 2.0, clamp=True),
                                    0.5 - half), mid)
        remapped = self.select(g.math('LESS_THAN', reveal, 0.5), low, high)
        remapped = g.math('MINIMUM', g.math('MAXIMUM', remapped, 1e-4), 0.9999)
        return g.math('ADD', g.math('MULTIPLY', remapped, 1 - 2 * softness), softness)

    def select(self, cond, a, b):
        """cond ? a : b for a 0/1 condition."""
        g = self.g
        return g.math('ADD', g.math('MULTIPLY', cond, a), g.math('MULTIPLY', g.math('SUBTRACT', 1.0, cond), b))

    def smoothstep(self, e0, e1, x):
        g = self.g
        t = g.math('DIVIDE', g.math('SUBTRACT', x, e0), g.math('SUBTRACT', e1, e0), clamp=True)
        return g.math('MULTIPLY', g.math('MULTIPLY', t, t), g.math('SUBTRACT', 3.0, g.math('MULTIPLY', t, 2.0)))

    def weight(self, blend, threshold, softness):
        g = self.g
        soft = self.smoothstep(-softness, softness, g.math('SUBTRACT', blend, threshold))
        return self.select(g.math('GREATER_THAN', blend, 0.99999), 1.0, soft)

    def border_amount(self, blend, threshold, n):
        g = self.g
        width = self.v(f'g_vBorderWidth{n}', (0.125, 0.125))
        soft = self.v(f'g_vBorderSoftness{n}', (0.5, 0.5))
        below = g.math('LESS_THAN', blend, threshold)
        edge = self.select(below, soft[0], soft[1])
        outer = self.select(below, g.math('MAXIMUM', g.math('SUBTRACT', threshold, width[0]), 1e-4),
                            g.math('MINIMUM', g.math('ADD', threshold, width[1]), 0.9999))
        inner = self.select(below, g.math('MAXIMUM', g.math('SUBTRACT', threshold, 0.5 * width[0]), 1e-4),
                            g.math('MINIMUM', g.math('ADD', threshold, 0.5 * width[1]), 0.9999))
        lerp = lambda a, b, t: g.math('ADD', a, g.math('MULTIPLY', g.math('SUBTRACT', b, a), t))
        amount = self.smoothstep(lerp(inner, outer, edge), lerp(inner, threshold, edge), blend)
        # unpainted surfaces get no border
        return g.math('MULTIPLY', amount, g.math('GREATER_THAN', blend, 0.0))

    def border_tint(self, albedo, amount, n):
        g = self.g
        tint = self.v(f'g_vBorderTint{n}', (0.5, 0.5, 0.5, 1.0))
        mode = self.i(f'g_nBorderBlendMode{n}')
        weight = g.math('MULTIPLY', amount, tint[3])
        color = tuple(tint[:3]) if mode == 1 else tuple(srgb_to_linear(np.asarray(tint[:3], float)))
        if mode == 2:
            return g.mix('MIX', albedo, color + (1.0,), weight)
        scale = 1.9921569 if mode == 1 else 1.0
        factor = g.mix('MIX', (1.0, 1.0, 1.0, 1.0), tuple(c * scale for c in color) + (1.0,), weight)
        return g.mix('MULTIPLY', albedo, factor)

    def overlay_factor(self, overlay, dark, bright):
        """GetOverlayFactor on a colour socket."""
        g = self.g
        signed = g.vec('SUBTRACT', g.vec('SCALE', overlay, 2.0), (1.0, 1.0, 1.0))
        up = g.vec('SUBTRACT', (1.0, 1.0, 1.0), g.vec('MAXIMUM', signed, (0.0, 0.0, 0.0)))
        brighten = g.vec('SCALE', g.vec('SUBTRACT', (1.0, 1.0, 1.0),
                                        g.vec('POWER', g.vec('MAXIMUM', up, (1e-6,) * 3), (bright,) * 3)), bright)
        down = g.vec('ADD', (1.0, 1.0, 1.0), g.vec('MINIMUM', signed, (0.0, 0.0, 0.0)))
        darken = g.vec('SCALE', g.vec('SUBTRACT', g.vec('POWER', g.vec('MAXIMUM', down, (1e-6,) * 3), (dark,) * 3),
                                      (1.0, 1.0, 1.0)), dark)
        return g.vec('MAXIMUM', g.vec('ADD', g.vec('ADD', brighten, darken), (1.0, 1.0, 1.0)), (0.0, 0.0, 0.0))

    def overlay_layer(self, albedo, mask, factor, mode, strength):
        g = self.g
        amount = 1.0 if mode == 0 else (g.math('SUBTRACT', 1.0, mask) if mode == 2 else mask)
        amount = g.math('MULTIPLY', amount, strength) if not isinstance(amount, float) else strength
        return g.mix('MIX', albedo, g.mix('MULTIPLY', albedo, factor), amount)

    def world_overlays(self):
        """(factor, per-layer strengths) for each world overlay."""
        count = self.i('F_WORLD_OVERLAY')
        out = []
        for index in range(count):
            img = self.image('g_tColorWorldOverlay2' if index else 'g_tColorWorldOverlay', True)
            if img is None:
                continue
            if count == 2:
                tiling = self.v('g_vWorldOverlayTiling', (512.0, 512.0))[index]
                dark = self.v('g_vWorldOverlayDarknessContrast', (1.0, 1.0))[index]
                bright = self.v('g_vWorldOverlayBrightnessContrast', (1.0, 1.0))[index]
                strengths = [self.v(f'g_vWorldOverlayStrengths{n}', (1.0, 0.0))[index] for n in (1, 2, 3)]
            else:
                tiling = self.f('g_flWorldOverlayTiling', 512.0)
                dark = self.f('g_flWorldOverlayDarknessContrast', 1.0)
                bright = self.f('g_flWorldOverlayBrightnessContrast', 1.0)
                strengths = [self.f(f'g_flWorldOverlayStrength{n}', 1.0) for n in (1, 2, 3)]
            major, median, w_median = self.planar()
            a = self.sample(img, self.planar_uv(major, tiling))[0]
            b = self.sample(img, self.planar_uv(median, tiling))[0]
            overlay = self.g.mix('MIX', a, b, w_median)
            out.append((self.overlay_factor(overlay, dark, bright), strengths))
        return out

    def uv_overlay_coords(self):
        return self.transform(self.uv(bool(self.i('g_bUseSecondaryUvForOverlay', 1))),
                              self.v('g_vOverlayScaleRotationTranslation', NO_TRANSFORM))

    def build(self, layer1, mask1):
        """The composited albedo socket, from layer 1's (already corrected, tinted and painted) albedo."""
        g = self.g
        three = bool(self.i('F_LAYER_COUNT'))
        blend = g.split(g.attribute('_BLEND'))
        blend2, blend3 = blend[0], (blend[1] if three else 0.0)
        if three:
            blend2 = self.select(g.math('LESS_THAN', blend3, 1.0), blend2, 0.0)
            blend3 = self.select(g.math('GREATER_THAN', blend3, 1e-5), blend3, 0.0)
        blend2 = self.select(g.math('GREATER_THAN', blend2, 1e-5), blend2, 0.0)
        if self.i('F_UV_OVERLAY'):
            img = self.image('g_tRevealOverlay', False)
            if img is not None:
                reveal = g.split(self.sample(img, self.uv_overlay_coords())[0])[0]
                if self.i('g_bRevealOverlayInvert2'):
                    reveal = g.math('SUBTRACT', 1.0, reveal)
                blend2 = g.math('ADD', blend2, g.math('MULTIPLY', reveal, self.f('g_flRevealOverlayStrength2', 1.0)),
                                clamp=True)

        layers = [(layer1, mask1)]
        weights = []
        for n, b in ((2, blend2), (3, blend3)):
            if n == 3 and not three:
                break
            sampled = self.layer(n)
            if sampled is None:
                continue
            threshold = self.reveal_threshold(n)
            if self.i(f'g_bBorderEffects{n}'):
                amount = self.border_amount(b, threshold, n)
                amounts = list(self.v(f'g_vBorderLayerAmount{n}', (1.0, 1.0, 1.0)))
                if n == 2:
                    amounts = amounts[:2] + [0.0]
                touched = layers + [sampled]
                layers = [(self.border_tint(alb, g.math('MULTIPLY', amount, amounts[k]), n), m)
                          for k, (alb, m) in enumerate(touched)]
                sampled = layers.pop()
            layers.append(sampled)
            weights.append((len(layers) - 1, self.weight(b, threshold, self.f(f'g_flRevealSoftness{n}', 0.125))))

        for factor, strengths in self.world_overlays():
            layers = [(self.overlay_layer(alb, m, factor, self.i(f'g_nWorldOverlayTintMask{k + 1}', 1),
                                          strengths[k]), m) for k, (alb, m) in enumerate(layers)]
        if self.i('F_UV_OVERLAY'):
            img = self.image('g_tColorOverlay', True)
            if img is not None:
                factor = self.overlay_factor(self.sample(img, self.uv_overlay_coords())[0],
                                             self.f('g_flOverlayDarknessContrast', 1.0),
                                             self.f('g_flOverlayBrightnessContrast', 1.0))
                layers = [(self.overlay_layer(alb, m, factor, self.i(f'g_nOverlayTintMask{k + 1}'),
                                              self.f(f'g_flOverlayStrength{k + 1}', 1.0)), m)
                          for k, (alb, m) in enumerate(layers)]

        result = layers[0][0]
        for index, w in weights:
            result = g.mix('MIX', result, layers[index][0], w)
        return result


def layer1_coords(blend, mat):
    """Points layer 1's glTF textures at its biplanar or transformed UVs, which the export doesn't apply."""
    if blend.i('F_LAYER1_BIPLANAR'):
        major, _median, _w = blend.planar()
        coords = blend.planar_uv(major, blend.f('g_flBiPlanarTiling1', 196.0))
    else:
        srt = blend.v('g_vScaleRotationTranslation1', NO_TRANSFORM)
        if np.allclose(srt, NO_TRANSFORM):
            return
        coords = blend.transform(blend.uv(), srt)
    vector = blend.to_blender_uv(coords)
    for node in [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]:
        if not node.inputs['Vector'].is_linked or node.inputs['Vector'].links[0].from_node.type in ('UVMAP', 'TEX_COORD'):
            blend.g.links.new(vector, node.inputs['Vector'])


def upstream_image(socket):
    """The nearest image texture feeding a socket: layer 1's colour, under the correction and paint nodes."""
    queue, seen = [socket.node], set()
    while queue:
        node = queue.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == 'TEX_IMAGE':
            return node
        queue += [link.from_node for i in node.inputs for link in i.links]
    return None


applied = 0


def apply_blend(mat, bsdf, ints, floats, vectors, textures):
    """Replaces the base colour of an environment_blend material with its composited layers. textures maps the vmat
    texture paths to decompiled files. Returns whether it did."""
    source = bsdf.inputs['Base Color'].links[0].from_socket if bsdf.inputs['Base Color'].is_linked else None
    if source is None:
        return False
    paths = dict(vmat_dict(mat).get('TextureParams') or {})
    found = {key: textures[path] for key, path in paths.items() if key in TEXTURES and path in textures}
    blend = Blend(mat, ints, floats, vectors, found)
    layer1_coords(blend, mat)
    if not found and not ints.get('F_WORLD_OVERLAY'):
        return False
    image = upstream_image(source)
    mask1 = blend.tint_mask(image.outputs['Alpha'], 1) if image else 1.0
    result = blend.build(source, mask1)
    blend.g.links.new(result, bsdf.inputs['Base Color'])
    global applied
    applied += 1
    return True
