"""Renderers: how a system's particles are drawn. Sprites are cards around each particle (camera facing by default),
ropes a ribbon through the particles in order (a Catmull-Rom spline), trails a card per particle stretched along
its motion. All three draw through the spritecard shader: up to five texture layers (sheet animated, mixed by
channel and blend mode), the particle's colour and alpha, lit, scaled by an overbright factor, blended or added.

What can be worked out without a camera is done here (sheet frames, rope splines, trail lengths, colours); which way
cards and ribbons face is left to whoever draws them: ``Sprites.orientation`` and ``Ribbons.facing`` say how.
"""
import math

import numpy as np

from . import fields as F
from .params import Params, enum_names, fract, lerp, saturate

BLEND_MODES = enum_names("PARTICLE_OUTPUT_BLEND_MODE_ALPHA", "PARTICLE_OUTPUT_BLEND_MODE_ADD",
                         "PARTICLE_OUTPUT_BLEND_MODE_BLEND_ADD", "PARTICLE_OUTPUT_BLEND_MODE_HALF_BLEND_ADD",
                         "PARTICLE_OUTPUT_BLEND_MODE_NEG_HALF_BLEND_ADD", "PARTICLE_OUTPUT_BLEND_MODE_MOD2X",
                         "PARTICLE_OUTPUT_BLEND_MODE_LIGHTEN")
BLEND_ALPHA, BLEND_ADD, BLEND_MOD2X = 0, 1, 5

ORIENTATIONS = enum_names("PARTICLE_ORIENTATION_SCREEN_ALIGNED", "PARTICLE_ORIENTATION_SCREEN_Z_ALIGNED",
                          "PARTICLE_ORIENTATION_WORLD_Z_ALIGNED", "PARTICLE_ORIENTATION_ALIGN_TO_PARTICLE_NORMAL",
                          "PARTICLE_ORIENTATION_SCREENALIGN_TO_PARTICLE_NORMAL",
                          "PARTICLE_ORIENTATION_FULL_3AXIS_ROTATION")
SCREEN_ALIGNED, SCREEN_Z_ALIGNED, WORLD_Z_ALIGNED, ALIGN_TO_NORMAL, SCREEN_ALIGN_TO_NORMAL, FULL_ROTATION = range(6)

# How a drawer turns a card's base axes towards the camera (Sprites.orientation).
FACE_SCREEN, FACE_SCREEN_Z, FACE_STORED, FACE_SCREEN_NORMAL = range(4)

FEATHERING = enum_names("PARTICLE_DEPTH_FEATHERING_OFF", "PARTICLE_DEPTH_FEATHERING_ON_OPTIONAL",
                        "PARTICLE_DEPTH_FEATHERING_ON_REQUIRED")
FEATHER_OFF = 0

# Opacity scales matched by eye to in-game footage, for systems whose data drawn as read (and as VRF draws it) comes
# out far heavier than the game shows: the singularity's dust hid the ground in black clumps where the game has a
# translucent haze.
FOOTAGE_ALPHA = {
    "particles/abilities/dynamo/dynamo_singularity_dust.vpcf": 0.4,
}

ANIMATION = enum_names("ANIMATION_TYPE_FIXED_RATE", "ANIMATION_TYPE_FIT_LIFETIME", "ANIMATION_TYPE_MANUAL_FRAMES")
FIXED_RATE, FIT_LIFETIME, MANUAL_FRAMES = range(3)

TEXTURE_TYPES = enum_names("SPRITECARD_TEXTURE_DIFFUSE", "SPRITECARD_TEXTURE_ZOOM", "SPRITECARD_TEXTURE_1D_COLOR_LOOKUP",
                           "SPRITECARD_TEXTURE_UVDISTORTION", "SPRITECARD_TEXTURE_UVDISTORTION_ZOOM",
                           "SPRITECARD_TEXTURE_NORMALMAP", "SPRITECARD_TEXTURE_ANIMMOTIONVEC")
TEXTURE_DIFFUSE, TEXTURE_ZOOM, TEXTURE_COLOR_LOOKUP, TEXTURE_DISTORTION, TEXTURE_DISTORTION_ZOOM = range(5)
TEXTURE_NORMAL_MAP, TEXTURE_MOTION_VECTORS = 5, 6

CHANNELS = enum_names("SPRITECARD_TEXTURE_CHANNEL_MIX_RGB", "SPRITECARD_TEXTURE_CHANNEL_MIX_RGBA",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_A", "SPRITECARD_TEXTURE_CHANNEL_MIX_RGB_A",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_RGB_ALPHAMASK", "SPRITECARD_TEXTURE_CHANNEL_MIX_RGB_RGBMASK",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_RGBA_RGBALPHA", "SPRITECARD_TEXTURE_CHANNEL_MIX_A_RGBALPHA",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_RGB_A_RGBALPHA", "SPRITECARD_TEXTURE_CHANNEL_MIX_R",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_G", "SPRITECARD_TEXTURE_CHANNEL_MIX_B",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_RALPHA", "SPRITECARD_TEXTURE_CHANNEL_MIX_GALPHA",
                      "SPRITECARD_TEXTURE_CHANNEL_MIX_BALPHA")
CHANNEL_RGBA = 1

LAYER_BLENDS = enum_names("SPRITECARD_TEXTURE_BLEND_MULTIPLY", "SPRITECARD_TEXTURE_BLEND_MOD2X",
                          "SPRITECARD_TEXTURE_BLEND_REPLACE", "SPRITECARD_TEXTURE_BLEND_ADD",
                          "SPRITECARD_TEXTURE_BLEND_SUBTRACT", "SPRITECARD_TEXTURE_BLEND_AVERAGE",
                          "SPRITECARD_TEXTURE_BLEND_LUMINANCE")

MAX_LAYERS = 5


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, np.power(np.maximum(c + 0.055, 0.0) / 1.055, 2.4))


class Sheet:
    """A texture's sprite sheet: sequences of frames, each frame its time on screen and (per layer) the rectangle
    it takes on the texture, cropped and uncropped (min u, min v, max u, max v, V down)."""

    def __init__(self, sequences: list):
        self.sequences = []
        for sequence in sequences:
            frames = sequence["frames"]
            times = np.array([f[0] for f in frames], np.float64)
            images = []
            for _, frame_images in frames:
                image = frame_images[0] if frame_images else (0, 0, 1, 1, 0, 0, 1, 1)
                images.append(image)
            self.sequences.append({
                "clamp": bool(sequence["clamp"]),
                "total": sequence["total_time"] if sequence["total_time"] > 0 else 1.0,
                "times": times,
                "ends": np.cumsum(times)[:-1] if len(times) > 1 else np.zeros(0),
                "starts": np.concatenate([[0.0], np.cumsum(times)[:-1]]) if len(times) else np.zeros(1),
                "images": np.array(images, np.float64).reshape(-1, 8),
            })

    def frames(self, sequence_numbers, passes):
        """Per particle: (image, next image, blend) for each particle's sequence and playback position (in loops
        of the sequence)."""
        n = len(passes)
        current = np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.float64), (n, 1))
        following = current.copy()
        blend = np.zeros(n)
        if not self.sequences:
            return current, following, blend
        numbers = np.asarray(sequence_numbers, np.int64)
        numbers = np.where((numbers >= 0) & (numbers < len(self.sequences)), numbers, 0)
        for number in np.unique(numbers):
            sequence = self.sequences[number]
            rows = numbers == number
            images = sequence["images"]
            last = len(images) - 1
            if last < 1:
                if last == 0:
                    current[rows] = following[rows] = images[0]
                continue
            p = passes[rows]
            position = sequence["total"] * (saturate(p) if sequence["clamp"] else fract(p))
            frame = np.searchsorted(sequence["ends"], position, side="right")
            frame = np.minimum(frame, last)
            inside = frame < last
            start = sequence["starts"][frame]
            times = sequence["times"][frame]
            next_frame = np.where(inside, frame + 1, last if sequence["clamp"] else 0)
            tail = sequence["total"] - start
            span = np.where(inside, times, 0.0 if sequence["clamp"] else tail)
            frame_blend = np.divide(position - start, span, out=np.zeros_like(position), where=span > 0)
            current[rows] = images[frame]
            following[rows] = images[next_frame]
            blend[rows] = frame_blend
        return current, following, blend


class TextureLayer:
    """One of a renderer's texture layers (m_vecTexturesInput)."""

    def __init__(self, p: Params, library, default_texture: str):
        self.kind = p.enum("m_nTextureType", TEXTURE_TYPES, TEXTURE_DIFFUSE)
        self.channels = p.enum("m_nTextureChannels", CHANNELS, CHANNEL_RGBA)
        self.blend_mode = p.enum("m_nTextureBlendMode", LAYER_BLENDS, 0)
        self.blend = p.number("m_flTextureBlend", 1.0)
        self.gradient = None
        self.texture = None
        self.sheet = None
        if p.bool("m_bReplaceTextureWithGradient"):
            stops = ((p.get("m_Gradient") or {}).get("m_Stops") or [])
            self.gradient = sorted(((float(s.get("m_flPosition", 0.0)), tuple(float(c) / 255.0 for c in (s.get("m_Color") or [255, 255, 255])[:3]))
                                    for s in stops), key=lambda s: s[0])
        else:
            self.texture = p.str("m_hTexture") or default_texture
            sequences = library.sheet(self.texture)
            self.sheet = Sheet(sequences) if sequences else None
        controls = p.get("m_TextureControls")
        c = p.nested(controls if isinstance(controls, dict) else {})
        self.scale = (c.number("m_flFinalTextureScaleU", 1.0), c.number("m_flFinalTextureScaleV", 1.0))
        self.offset = (c.number("m_flFinalTextureOffsetU", 0.0), c.number("m_flFinalTextureOffsetV", 0.0))
        self.rotation = c.number("m_flFinalTextureUVRotation", 0.0)
        self.clamp = c.bool("m_bClampUVs")
        self.distortion = c.number("m_flDistortion", 0.0)
        self.zoom = c.number("m_flZoomScale", 0.0)
        per_particle = (c.str("m_nPerParticleScale"), c.str("m_nPerParticleDistortion"))
        if any(value and value != "SPRITECARD_TEXTURE_PP_SCALE_NONE" for value in per_particle):
            p.note("texture layer per particle scale")
        self.random_offsets = c.bool("m_bRandomizeOffsets")

    def settings(self, sim) -> dict:
        """The layer's mix and coordinate transform for a draw: the card is turned (authored in degrees: a ramp
        turned a quarter way round is 90), divided by the scale and offset before it is placed in the frame."""
        return {
            "kind": self.kind, "channels": self.channels, "blend_mode": self.blend_mode,
            "blend": float(self.blend(sim)),
            "scale": (float(self.scale[0](sim)), float(self.scale[1](sim))),
            "offset": (float(self.offset[0](sim)), float(self.offset[1](sim))),
            "rotation": math.radians(float(self.rotation(sim))),
            "clamp": self.clamp,
            "distortion": float(self.distortion(sim)),
            "zoom": float(self.zoom(sim)),
            "random_offsets": self.random_offsets,
            "texture": self.texture, "gradient": self.gradient, "sheet": self.sheet is not None,
        }


def read_layers(p: Params, library, default_texture: str) -> list[TextureLayer]:
    layers = []
    for block in p.array("m_vecTexturesInput"):
        if not block.bool("m_bEnabled", True):
            continue
        kind = block.enum("m_nTextureType", TEXTURE_TYPES, TEXTURE_DIFFUSE)
        if kind == TEXTURE_NORMAL_MAP:
            continue  # normal maps aren't colour
        if kind == TEXTURE_MOTION_VECTORS:
            p.note("texture layer motion vectors")
            continue
        if len(layers) == MAX_LAYERS:
            break
        layers.append(TextureLayer(block, library, default_texture))
    if not layers:
        layers.append(TextureLayer(p.nested({}), library, default_texture))
    return layers


class Renderer:
    """What the three spritecard renderers share: texture layers, blend mode, colour and alpha scales, lighting
    amounts and the overbright factor."""

    kind = ""
    default_texture = "materials/particle/base_sprite.vtex"
    footage_alpha = 1.0

    def __init__(self, p: Params, library, index: int):
        self.index = index
        self.blend_mode = p.enum("m_nOutputBlendMode", BLEND_MODES, BLEND_ALPHA)
        self.layers = read_layers(p, library, self.default_texture)
        self.radius_scale = p.number("m_flRadiusScale", 1.0)
        self.alpha_scale = p.number("m_flAlphaScale", 1.0)
        self.color_scale = p.vec("m_vecColorScale", (1.0, 1.0, 1.0))
        self.overbright = p.number("m_flOverbrightFactor", 1.0)
        self.add_self = p.number("m_flAddSelfAmount", 0.0)
        self.diffuse = p.number("m_flDiffuseAmount", 1.0)
        self.self_illum = p.number("m_flSelfIllumAmount", 0.0)
        self.desaturation = p.number("m_flDesaturation", 0.0)
        self.alpha_remap = (p.number("m_flSourceAlphaValueToMapToZero", 0.0), p.number("m_flSourceAlphaValueToMapToOne", 1.0))
        self.gamma_correct = p.bool("m_bGammaCorrectVertexColors", True) and not p.bool("m_bOnlyRenderInEffectsWaterPass")
        self.animation_rate = p.float("m_flAnimationRate", 0.1)
        self.animation_type = p.enum("m_nAnimationType", ANIMATION, FIXED_RATE)
        self.animate_in_fps = p.bool("m_bAnimateInFPS")
        self.blend_frames = p.bool("m_bBlendFramesSeq0", True)
        self.max_luminance_blend = p.bool("m_bMaxLuminanceBlendingSequence0")
        self.orientation = p.enum("m_nOrientationType", ORIENTATIONS, SCREEN_ALIGNED)
        # 15 is never written out in the game's files while 10, 20 and 25 are: the default, stripped on compile
        self.feathering = p.enum("m_nFeatheringMode", FEATHERING, FEATHER_OFF) != FEATHER_OFF
        self.feather_bounds = (p.number("m_flFeatheringMinDist", 0.0), p.number("m_flFeatheringMaxDist", 15.0))
        self.opaque = False
        # Reverse Z buffered renderers draw only where something is in front (a glow behind walls); Blender can't test
        # depth that way, and drawing them normally would double the renderer they copy.
        if p.bool("m_bOnlyRenderInEffectsWaterPass") or p.bool("m_bOnlyRenderInEffectsBloomPass") or p.bool("m_bReverseZBuffering"):
            self.hidden = True
        else:
            self.hidden = False

    def shading(self, sim) -> dict:
        """The renderer's values for a whole draw (evaluated at the snapshot)."""
        low, high = (float(f(sim)) for f in self.alpha_remap)
        return {
            "blend_mode": self.blend_mode,
            "overbright": float(self.overbright(sim)) * (1.0 + float(self.add_self(sim))),
            "desaturation": float(self.desaturation(sim)),
            "alpha_remap": (low, high) if low < high else None,
            "blend_frames": self.blend_frames,
            "max_luminance_blend": self.max_luminance_blend,
            "opaque": self.opaque,
            "feather": tuple(float(f(sim)) for f in self.feather_bounds) if self.feathering else None,
            "layers": [layer.settings(sim) for layer in self.layers],
        }

    def _lit_colors(self, sim, color, alpha, ambient):
        """Linear vertex colours lit as the game lights particles (self illumination plus diffuse times the light
        at the particle), with alpha: (n, 4)."""
        # The colour scale goes into the vertex colour on the CPU; the spritecard vertex shader decodes COLOR0 from
        # sRGB with no scale after it, so a CP tint scaled by 1/256 (street lamp glows) is gamma decoded too.
        rgb = np.asarray(color, np.float64) * np.asarray(self.color_scale(sim), np.float64)
        rgb = srgb_to_linear(np.clip(rgb, 0.0, None)) if self.gamma_correct else rgb
        light = float(self.self_illum(sim)) + float(self.diffuse(sim)) * np.asarray(ambient, np.float64)
        out = np.empty((len(rgb), 4))
        out[:, :3] = rgb * light
        out[:, 3] = np.asarray(alpha, np.float64) * self.footage_alpha
        return out

    def _random_offsets(self, sim, ids) -> list:
        """Per particle texture offsets (u, v, 0) for the layers that randomize theirs, drawn from the particle's
        id so each keeps its own."""
        offsets = []
        for index, layer in enumerate(self.layers):
            if not layer.random_offsets:
                offsets.append(None)
                continue
            u = sim.random.for_particle(ids, 61 + index * 2)
            v = sim.random.for_particle(ids, 62 + index * 2)
            offsets.append(np.stack([u, v, np.zeros_like(u)], axis=1))
        return offsets

    def _passes(self, sim, v, sheet: Sheet):
        if self.animation_type == MANUAL_FRAMES:
            return v.get("manual_frame")
        time = v.normalized_age if self.animation_type == FIT_LIFETIME else v.get("age")
        if self.animate_in_fps:
            totals = np.array([s["total"] for s in sheet.sequences])
            numbers = np.clip(v.get("sequence").astype(int), 0, len(totals) - 1)
            return time * self.animation_rate / totals[numbers]
        return time * self.animation_rate

    def build(self, sim, ambient):
        raise NotImplementedError


class Sprites:
    """Cards to draw: one per particle at ``position`` with a half width, turned by roll and yaw about base axes
    that depend on ``orientation`` (FACE_*: the camera's, the camera's yawed about world Z, the stored ``right`` and
    ``up``, or the stored ``right`` with up towards the camera). ``card`` is the card's offset and size in half widths
    along those axes (centre offsets and cropped sheet frames). Each layer with a sheet has its frame's rectangle
    (and the next one's, blended by ``frame_blend``)."""

    def __init__(self, renderer, n):
        self.renderer = renderer
        self.count = n
        self.orientation = FACE_SCREEN
        self.position = np.zeros((n, 3))
        self.half_width = np.zeros(n)
        self.roll = np.zeros(n)
        self.yaw = np.zeros(n)
        self.right = np.tile([0.0, -1.0, 0.0], (n, 1))
        self.up = np.tile([1.0, 0.0, 0.0], (n, 1))
        self.card = np.tile([0.0, 0.0, 1.0, 1.0], (n, 1))
        self.color = np.ones((n, 4))
        self.frame_blend = np.zeros(n)
        self.size_bounds = (0.0, 1e8, 2e8, 5000.0)
        self.rects = [None] * len(renderer.layers)
        self.next_rects = [None] * len(renderer.layers)
        self.offsets = [None] * len(renderer.layers)
        self.shading = {}


class Ribbons:
    """Strips to draw (ropes, trails): vertices on a spine at ``position``, pushed sideways by ``width`` (signed)
    across ``axis`` (the strip's direction), in the plane facing the camera (``facing`` None) or the plane with the
    stored ``facing`` normal; ``quads`` index the vertices. ``card`` is each vertex's texture coordinate (u across,
    v along, game convention: V down)."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.position = np.zeros((0, 3))
        self.axis = np.zeros((0, 3))
        self.width = np.zeros(0)
        self.facing = None
        self.card = np.zeros((0, 2))
        self.color = np.zeros((0, 4))
        self.quads = np.zeros((0, 4), np.int64)
        self.frame_blend = np.zeros(0)
        self.rects = [None] * len(renderer.layers)
        self.next_rects = [None] * len(renderer.layers)
        self.offsets = [None] * len(renderer.layers)
        self.shading = {}

    @property
    def count(self):
        return len(self.quads)


def _tile_window(images):
    """The cropped part of each image, as a window of its uncropped rectangle: (n, 4)."""
    size = images[:, 6:8] - images[:, 4:6]
    safe = np.where(size != 0, size, 1.0)
    low = np.where(size != 0, (images[:, 0:2] - images[:, 4:6]) / safe, 0.0)
    high = np.where(size != 0, (images[:, 2:4] - images[:, 4:6]) / safe, 0.0)
    return np.concatenate([low, high], axis=1)


def _window_rect(images, window):
    size = images[:, 6:8] - images[:, 4:6]
    return np.concatenate([images[:, 4:6] + window[:, 0:2] * size, images[:, 4:6] + window[:, 2:4] * size], axis=1)


def _uncropped(images):
    return images[:, 4:8].copy()


class RenderSprites(Renderer):
    kind = "sprites"
    default_texture = "materials/particle/base_sprite.vtex"

    def __init__(self, p: Params, library, index: int):
        super().__init__(p, library, index)
        self.center_offset = (p.number("m_flCenterXOffset", 0.0), p.number("m_flCenterYOffset", 0.0))
        # radii per unit of camera distance; the defaults are VRF's (no clamp, no fade)
        self.size_bounds = (p.number("m_flMinSize", 0.0), p.number("m_flStartFadeSize", 1e8),
                            p.number("m_flEndFadeSize", 2e8), p.number("m_flMaxSize", 5000.0))
        if self.orientation == FULL_ROTATION:
            p.note("sprites full 3 axis rotation")

    def build(self, sim, ambient) -> Sprites | None:
        particles = sim.particles
        if particles.count == 0:
            return None
        v = F.View(particles, slice(0, particles.count))
        radius = v.get("radius") * self.radius_scale(sim, v)
        alpha = v.get("alpha") * self.alpha_scale(sim, v)
        keep = (radius > 0) & (alpha >= 1.0 / 255.0)
        if not keep.any():
            return None
        rows = np.nonzero(keep)[0]
        out = Sprites(self, len(rows))
        out.shading = self.shading(sim)
        out.size_bounds = tuple(float(f(sim)) for f in self.size_bounds)
        out.position = v.get("position")[rows]
        out.half_width = radius[rows]
        out.roll = v.get("roll")[rows]
        out.yaw = v.get("yaw")[rows]
        out.color = self._lit_colors(sim, v.get("color")[rows], alpha[rows], ambient)

        orientation = self.orientation
        if orientation == SCREEN_ALIGNED:
            out.orientation = FACE_SCREEN
        elif orientation == SCREEN_Z_ALIGNED:
            out.orientation = FACE_SCREEN_Z
        elif orientation == WORLD_Z_ALIGNED:
            out.orientation = FACE_STORED
        elif orientation == ALIGN_TO_NORMAL:
            out.orientation = FACE_STORED
            normal = v.get("normal")[rows]
            reference = np.where((np.abs(normal[:, 2]) > 0.1)[:, None], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
            up = np.cross(normal, reference)
            up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
            out.up = up
            out.right = np.cross(up, normal)
        elif orientation == SCREEN_ALIGN_TO_NORMAL:
            out.orientation = FACE_SCREEN_NORMAL
            normal = v.get("normal")[rows]
            out.right = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        else:
            from ..probes import rotation_matrix
            out.orientation = FACE_STORED
            for i, row in enumerate(rows):
                matrix = rotation_matrix(np.degrees([v.get("pitch")[row], v.get("yaw")[row], v.get("roll")[row]]))
                out.right[i], out.up[i] = matrix[:, 0], matrix[:, 1]
            out.roll[:] = 0.0
            out.yaw[:] = 0.0

        center_x, center_y = (np.broadcast_to(np.asarray(c(sim, v), np.float64), (particles.count,))[rows] for c in self.center_offset)
        window = np.tile([0.0, 0.0, 1.0, 1.0], (len(rows), 1))
        sub = F.View(particles, slice(0, particles.count))
        for layer_index, layer in enumerate(self.layers):
            if layer.sheet is None:
                continue
            numbers = (v.get("sequence") if layer_index == 0 else v.get("sequence1"))[rows]
            current, following, blend = layer.sheet.frames(numbers, self._passes(sim, sub, layer.sheet)[rows])
            if layer_index == 0:
                current_window = _tile_window(current)
                next_window = _tile_window(following)
                window = np.concatenate([np.minimum(current_window[:, :2], next_window[:, :2]),
                                         np.maximum(current_window[:, 2:], next_window[:, 2:])], axis=1)
                out.frame_blend = blend
            out.rects[layer_index] = _window_rect(current, window)
            out.next_rects[layer_index] = _window_rect(following, window)
        size = window[:, 2:] - window[:, :2]
        out.card = np.stack([center_x + (window[:, 2] + window[:, 0] - 1.0),
                             center_y + (1.0 - window[:, 1] - window[:, 3]), size[:, 0], size[:, 1]], axis=1)
        out.offsets = self._random_offsets(sim, v.get("particle_id")[rows])
        return out


def _catmull_rom(p0, p1, p2, p3, t):
    t = t[:, None] if np.ndim(p1) else t
    t2, t3 = t * t, t * t * t
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _catmull_rom_tangent(p0, p1, p2, p3, t):
    t = t[:, None]
    return 0.5 * ((-p0 + p2) + (4 * p0 - 10 * p1 + 8 * p2 - 2 * p3) * t + (-3 * p0 + 9 * p1 - 9 * p2 + 3 * p3) * t * t)


class RenderRopes(Renderer):
    """A ribbon through the particles in emission order (split where the segment attribute changes), its texture
    running along it by world distance."""

    kind = "ropes"
    default_texture = "materials/particle/base_trail.vtex"

    def __init__(self, p: Params, library, index: int):
        super().__init__(p, library, index)
        self.orientation_field = p.int("m_nVectorFieldForOrientation", F.NORMAL)
        self.min_tessellation = p.int("m_nMinTesselation", 1)
        self.max_tessellation = p.int("m_nMaxTesselation", 128)
        self.tess_scale = p.float("m_flTessScale", 1.0)
        self.v_world_size = p.number("m_flTextureVWorldSize", 10.0)
        self.v_scroll = p.number("m_flTextureVScrollRate", 0.0)
        self.v_offset = p.number("m_flTextureVOffset", 0.0)
        self.v_params_cp = p.int("m_nTextureVParamsCP", -1)
        self.clamp_v = p.bool("m_bClampV")
        self.use_scalar = p.bool("m_bUseScalarForTextureCoordinate")
        self.scalar_field = p.int("m_nScalarFieldForTextureCoordinate", F.CREATION_TIME)
        self.scalar_scale = p.float("m_flScalarAttributeTextureCoordScale", 1.0)
        self.reverse = p.bool("m_bReverseOrder")
        self.closed = p.bool("m_bClosedLoop")
        self.split_field = p.int("m_nSplitField", F.ROPE_SEGMENT_ID)
        self.sort_by_segment = p.bool("m_bSortBySegmentID")
        self.opaque = p.bool("m_bDrawAsOpaque")
        # Full 3-axis rotation on a rope lays it against the particles' normal (the fan wind's sheet, normals from
        # C_INIT_NormalAlignToCP); facing the camera instead turns half of it into the ground.
        if self.orientation == FULL_ROTATION:
            self.orientation = ALIGN_TO_NORMAL
        elif self.orientation > SCREEN_ALIGN_TO_NORMAL:
            self.orientation = SCREEN_ALIGN_TO_NORMAL

    def _subdivisions(self, nodes: int) -> int:
        """The game picks a tessellation from how much of the screen the rope covers; this is its choice up close."""
        raw = int(1024.0 / max(nodes, 1) * self.tess_scale * 0.1)
        count = min(raw, self.max_tessellation)
        count = max(count, self.min_tessellation)
        count = min(max(count, 1), 128)
        level = 0
        while level < 7 and (1 << level) < count:
            level += 1
        return 1 << level

    def build(self, sim, ambient) -> Ribbons | None:
        particles = sim.particles
        n = particles.count
        if n < 2:
            return None
        v = F.View(particles, slice(0, n))
        order = np.arange(n)[::-1] if self.reverse else np.arange(n)
        position = v.get("position")[order]
        radius = (v.get("radius") * self.radius_scale(sim, v))[order]
        alpha = (v.get("alpha") * v.get("alpha_alternate") * self.alpha_scale(sim, v))[order]
        color = v.get("color")[order]
        splits = F.name(self.split_field) is not None and not F.is_vector(self.split_field)
        segment = v.field(self.split_field)[order] if splits else np.zeros(n)
        normal = v.field(self.orientation_field)[order] if F.is_vector(self.orientation_field) else np.tile([0, 0, 1.0], (n, 1))
        scalar = (v.field(self.scalar_field)[order] * self.scalar_scale) if self.use_scalar and not F.is_vector(self.scalar_field) else np.zeros(n)
        if self.sort_by_segment and splits:
            _, first = np.unique(segment, return_index=True)
            rank = {value: r for r, value in enumerate(segment[np.sort(first)])}
            keys = np.array([rank[s] for s in segment])
            sort = np.argsort(keys, kind="stable")
            position, radius, alpha, color, segment, normal, scalar = (a[sort] for a in (position, radius, alpha, color, segment, normal, scalar))

        if self.v_params_cp >= 0:
            params = sim.control_point(self.v_params_cp).position
            scroll, v_offset, inverse_size = params[1], params[2], 1.0 / max(params[0], 1e-30)
        else:
            scroll, v_offset = float(self.v_scroll(sim)), float(self.v_offset(sim))
            inverse_size = 1.0 / max(float(self.v_world_size(sim)), 1e-30)

        def index(i):
            if self.closed:
                return i % n
            return min(max(i, 0), n - 1)

        subdivisions = self._subdivisions(n)
        segments = n if self.closed else n - 1
        arc = scroll * sim.age
        out_position, out_tangent, out_radius, out_color, out_v, out_normal = [], [], [], [], [], []
        quads = []
        samples_so_far = 0
        strip_start = None  # index of the first sample of the strip being built
        for s in range(segments):
            i1, i2 = index(s), index(s + 1)
            v_start = arc
            v_end = arc + np.linalg.norm(position[i2] - position[i1])
            arc = v_end
            if segment[i1] != segment[i2]:
                strip_start = None
                continue
            i0, i3 = index(s - 1), index(s + 2)
            if segment[i0] != segment[i1]:
                i0 = i1
            if segment[i3] != segment[i2]:
                i3 = i2
            if self.use_scalar:
                # A scalar coordinate is in world units too: lamp glows map 0-100 over a V world size of 100.
                v_start, v_end = scalar[i1] + scroll * sim.age, scalar[i2] + scroll * sim.age
            t = np.arange(0 if strip_start is None else 1, subdivisions + 1) / subdivisions
            p0, p1, p2, p3 = position[i0], position[i1], position[i2], position[i3]
            samples = _catmull_rom(p0, p1, p2, p3, t)
            tangent = _catmull_rom_tangent(p0, p1, p2, p3, t)
            length = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = np.where(length > 1e-6, tangent / np.maximum(length, 1e-12), [1.0, 0.0, 0.0])
            out_position.append(samples)
            out_tangent.append(tangent)
            out_radius.append(_catmull_rom(radius[i0], radius[i1], radius[i2], radius[i3], t))
            rgba = np.concatenate([lerp(color[i1], color[i2], t[:, None]), lerp(alpha[i1], alpha[i2], t)[:, None]], axis=1)
            out_color.append(np.clip(rgba, 0.0, 1.0))
            coordinate = lerp(v_start, v_end, t)
            out_v.append(coordinate * inverse_size + v_offset)
            out_normal.append(lerp(normal[i1], normal[i2], t[:, None]))
            first = samples_so_far
            samples_so_far += len(t)
            if strip_start is None:
                strip_start = first
                local = range(first, first + len(t) - 1)
            else:
                local = range(first - 1, first + len(t) - 1)
            for sample in local:
                quads.append(sample)
        if not out_position:
            return None
        spine = np.concatenate(out_position)
        tangents = np.concatenate(out_tangent)
        radii = np.concatenate(out_radius)
        colors = np.concatenate(out_color)
        vs = np.concatenate(out_v)
        if self.clamp_v:
            vs = saturate(vs)
        normals = np.concatenate(out_normal)
        count = len(spine)

        out = Ribbons(self)
        out.shading = self.shading(sim)
        # Two vertices per sample: the -width side (u 0) then the +width side (u 1).
        out.position = np.repeat(spine, 2, axis=0)
        out.axis = np.repeat(tangents, 2, axis=0)
        out.width = np.stack([-radii, radii], axis=1).reshape(-1)
        out.card = np.stack([np.tile([0.0, 1.0], count), np.repeat(vs, 2)], axis=1)
        out.color = np.repeat(self._lit_colors(sim, colors[:, :3], colors[:, 3], ambient), 2, axis=0)
        if self.orientation == ALIGN_TO_NORMAL:
            out.facing = np.repeat(normals, 2, axis=0)
        elif self.orientation == WORLD_Z_ALIGNED:
            out.facing = np.tile([0.0, 0.0, 1.0], (count * 2, 1))
        starts = np.array(quads, np.int64)
        out.quads = np.stack([starts * 2, starts * 2 + 2, starts * 2 + 3, starts * 2 + 1], axis=1)
        self._sheet_rects(sim, v, out, count * 2)
        out.offsets = [None if o is None else np.repeat(o, count * 2, axis=0)
                       for o in self._random_offsets(sim, v.get("particle_id")[:1])]
        return out

    def _sheet_rects(self, sim, v, out, vertices):
        """The whole rope shows its first particle's sheet frame."""
        head = F.View(sim.particles, slice(0, 1))
        for layer_index, layer in enumerate(self.layers):
            if layer.sheet is None:
                continue
            current, following, blend = layer.sheet.frames(head.get("sequence"), self._passes(sim, head, layer.sheet))
            out.rects[layer_index] = np.repeat(_uncropped(current), vertices, axis=0)
            out.next_rects[layer_index] = np.repeat(_uncropped(following), vertices, axis=0)
            if layer_index == 0:
                out.frame_blend = np.full(vertices, blend[0])


class RenderTrails(Renderer):
    """A card per particle stretched from where it is back along its motion, head and tail tapered and tinted."""

    kind = "trails"
    default_texture = "materials/particle/base_trail.vtex"

    def __init__(self, p: Params, library, index: int):
        super().__init__(p, library, index)
        self.max_length = p.float("m_flMaxLength", 2000.0)
        self.min_length = p.float("m_flMinLength")
        self.length_scale = p.float("m_flLengthScale", 1.0)
        self.length_fade_in = p.float("m_flLengthFadeInTime")
        self.ignore_dt = p.bool("m_bIgnoreDT")
        self.radius_to_length = p.float("m_flConstrainRadiusToLengthRatio", 1.0)
        self.forward_shift = p.float("m_flForwardShift")
        self.previous_source = p.int("m_nPrevPntSource", F.PREVIOUS_POSITION)
        self.head_taper = p.number("m_flRadiusHeadTaper", 1.0)
        self.tail_taper = p.number("m_flRadiusTaper", 1.0)
        self.head_alpha = p.number("m_flHeadAlphaScale", 1.0)
        self.tail_alpha = p.number("m_flTailAlphaScale", 1.0)
        self.head_color = p.vec("m_vecHeadColorScale", (1.0, 1.0, 1.0))
        self.tail_color = p.vec("m_vecTailColorScale", (1.0, 1.0, 1.0))

    def build(self, sim, ambient) -> Ribbons | None:
        particles = sim.particles
        n = particles.count
        if n == 0:
            return None
        v = F.View(particles, slice(0, n))
        position = v.get("position")
        difference = v.field(self.previous_source) - position
        moved = np.linalg.norm(difference, axis=1)
        verlet = self.previous_source == F.PREVIOUS_POSITION
        one_over_dt = 1.0 if self.ignore_dt or not verlet or sim.frame_time == 0 else 1.0 / sim.frame_time
        length = self.length_scale * v.get("trail_length") * moved * one_over_dt
        length *= np.minimum(1.0, v.get("age") / max(self.length_fade_in, 1e-9))
        length = np.maximum(self.min_length, np.minimum(self.max_length, length))
        radius = v.get("radius") * self.radius_scale(sim, v)
        alpha = v.get("alpha") * v.get("alpha_alternate") * self.alpha_scale(sim, v)
        keep = (moved > 0) & (length != 0)
        if not keep.any():
            return None
        rows = np.nonzero(keep)[0]
        k = len(rows)
        direction = difference[rows] / moved[rows, None]
        length = length[rows]
        radius = np.minimum(radius[rows], self.radius_to_length * length)
        half = length * 0.5
        center = position[rows] + direction * (length * (0.5 - self.forward_shift))[:, None]
        head_width = radius * np.broadcast_to(np.asarray(self.head_taper(sim, v), np.float64), (n,))[rows]
        tail_width = radius * np.broadcast_to(np.asarray(self.tail_taper(sim, v), np.float64), (n,))[rows]
        tint = v.get("color")[rows]
        alpha = alpha[rows]
        head_alpha = np.clip(alpha * np.broadcast_to(np.asarray(self.head_alpha(sim, v), np.float64), (n,))[rows], 0, 1)
        tail_alpha = np.clip(alpha * np.broadcast_to(np.asarray(self.tail_alpha(sim, v), np.float64), (n,))[rows], 0, 1)
        head_rgb = np.clip(tint * np.asarray(self.head_color(sim), np.float64), 0, 1)
        tail_rgb = np.clip(tint * np.asarray(self.tail_color(sim), np.float64), 0, 1)

        out = Ribbons(self)
        out.shading = self.shading(sim)
        # Corners (-1, 1), (-1, -1), (1, -1), (1, 1): x across, y from head (-1) to tail (1).
        corners = np.array([[-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
        cx = np.tile(corners[:, 0], k)
        cy = np.tile(corners[:, 1], k)
        head = cy < 0
        out.position = np.repeat(center, 4, axis=0) + np.repeat(direction, 4, axis=0) * (np.repeat(half, 4) * cy)[:, None]
        out.axis = np.repeat(direction, 4, axis=0)
        out.width = cx * np.where(head, np.repeat(head_width, 4), np.repeat(tail_width, 4))
        out.card = np.stack([cx * 0.5 + 0.5, cy * 0.5 + 0.5], axis=1)
        head_color = self._lit_colors(sim, head_rgb, head_alpha, ambient)
        tail_color = self._lit_colors(sim, tail_rgb, tail_alpha, ambient)
        out.color = np.where(head[:, None], np.repeat(head_color, 4, axis=0), np.repeat(tail_color, 4, axis=0))
        if self.orientation == ALIGN_TO_NORMAL:
            out.facing = np.repeat(v.get("normal")[rows], 4, axis=0)
        elif self.orientation == WORLD_Z_ALIGNED:
            out.facing = np.tile([0.0, 0.0, 1.0], (k * 4, 1))
        out.quads = np.arange(k * 4, dtype=np.int64).reshape(k, 4)
        out.offsets = [None if o is None else np.repeat(o, 4, axis=0)
                       for o in self._random_offsets(sim, v.get("particle_id")[rows])]
        for layer_index, layer in enumerate(self.layers):
            if layer.sheet is None:
                continue
            current, following, blend = layer.sheet.frames(v.get("sequence")[rows], self._passes(sim, v, layer.sheet)[rows])
            out.rects[layer_index] = np.repeat(_uncropped(current), 4, axis=0)
            out.next_rects[layer_index] = np.repeat(_uncropped(following), 4, axis=0)
            if layer_index == 0:
                out.frame_blend = np.repeat(blend, 4)
        return out


class Models:
    """Model instances to place: per particle a model (index into ``paths``), a model to world matrix in game units,
    a tint and an alpha. ``material`` overrides every material of the models."""

    def __init__(self, renderer, n):
        self.renderer = renderer
        self.count = n
        self.paths = []
        self.material = ""
        self.skin = ""
        self.model = np.zeros(n, np.int64)
        self.matrix = np.tile(np.eye(4), (n, 1, 1))
        self.tint = np.ones((n, 3))
        self.alpha = np.ones(n)


def _normal_basis(normal, orient_z: bool) -> np.ndarray:
    """The frame a normal turns a model into (columns: the model's x, y, z): the normal becomes its forward axis, or
    its up axis when oriented on Z (ValveResourceFormat's RenderModels)."""
    length = np.linalg.norm(normal)
    if length == 0 or not np.isfinite(length):
        return np.eye(3)
    d = normal / length
    perpendicular = np.array([(1.0 - d[2]) * d[1] * d[1] + d[2], 0.0, -d[0]])
    perpendicular /= max(np.linalg.norm(perpendicular), 1e-12)
    perpendicular -= d * (perpendicular @ d)
    perpendicular /= max(np.linalg.norm(perpendicular), 1e-12)
    third = np.cross(d, perpendicular)
    x, y, z = (third, -perpendicular, d) if orient_z else (d, perpendicular, third)
    return np.stack([x, y, z], 1)


def _angle_matrices(pitch, yaw, roll) -> np.ndarray:
    """Source angles (radians) as (n, 3, 3) rotations, columns forward, left, up."""
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    sr, cr = np.sin(roll), np.cos(roll)
    forward = np.stack([cp * cy, cp * sy, -sp], 1)
    left = np.stack([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp], 1)
    up = np.stack([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp], 1)
    return np.stack([forward, left, up], 2)


def _per(values, n) -> np.ndarray:
    """A per particle value as an (n,) array."""
    return np.broadcast_to(np.asarray(values, np.float64), (n,))


def _rows3(values, n) -> np.ndarray:
    """A per particle vector as an (n, 3) array."""
    return np.broadcast_to(np.asarray(values, np.float64), (n, 3))


class RenderModels(Renderer):
    """A model per particle (C_OP_RenderModels): placed at it, turned by its angles and normal, scaled by its radius,
    tinted by its colour; the models' own materials or an override."""

    kind = "models"

    def __init__(self, p: Params, library, index: int):
        super().__init__(p, library, index)
        self.paths, self.weights = [], []
        for entry in p.array("m_ModelList"):
            path = entry.str("m_model")
            if path:
                self.paths.append(path)
                self.weights.append(max(entry.float("m_flRelativeProbabilityOfSpawn", 1.0), 0.0))
        if not self.paths:
            raise ValueError("no models")
        self.material = p.str("m_hOverrideMaterial")
        self.ignore_normal = p.bool("m_bIgnoreNormal")
        self.orient_z = p.bool("m_bOrientZ")
        self.ignore_radius = p.bool("m_bIgnoreRadius")
        self.local_scale = p.bool("m_bLocalScale")
        self.suppress_tint = p.bool("m_bSuppressTint")
        self.scale_cp = p.int("m_nModelScaleCP", -1)
        self.local_offset = p.vec("m_vecLocalOffset", (0.0, 0.0, 0.0))
        self.local_rotation = p.vec("m_vecLocalRotation", (0.0, 0.0, 0.0))
        self.component_scale = p.vec("m_vecComponentScale", (1.0, 1.0, 1.0))
        self.manual = p.number("m_flManualModelSelection", -1.0)
        self.roll_scale = p.number("m_flRollScale", 1.0)
        self.alpha2_field = p.int("m_nAlpha2Field", F.ALPHA_ALTERNATE)

    def build(self, sim, ambient) -> Models | None:
        n = sim.particles.count
        if n == 0:
            return None
        v = F.View(sim.particles, slice(0, n))
        out = Models(self, n)
        out.paths = list(self.paths)
        out.material = self.material

        manual = _per(self.manual(sim, v), n).astype(np.int64)
        weights = np.asarray(self.weights, np.float64)
        if weights.sum() > 0:
            draw = sim.random.for_particle(v.get("particle_id"), 0) * weights.sum()
            picked = np.searchsorted(np.cumsum(weights), draw, side="right").clip(0, len(weights) - 1)
        else:
            picked = np.zeros(n, np.int64)
        out.model = np.where((manual >= 0) & (manual < len(self.paths)), manual, picked)

        cp_scale = sim.control_point(self.scale_cp).position if self.scale_cp >= 0 else np.ones(3)
        component = _rows3(self.component_scale(sim, v), n)
        radius = np.ones(n) if self.ignore_radius else v.get("radius") * _per(self.radius_scale(sim, v), n)
        scale = component * cp_scale[None, :] * radius[:, None]
        roll_scale = float(np.mean(self.roll_scale(sim)))
        local = np.radians(_rows3(self.local_rotation(sim, v), n))
        turn = _angle_matrices(local[:, 0], local[:, 1], local[:, 2]) @             _angle_matrices(v.get("pitch"), v.get("yaw"), v.get("roll") * roll_scale)
        normals = v.field(F.NORMAL)
        offset = _rows3(self.local_offset(sim, v), n)
        position = v.get("position")
        for i in range(n):
            basis = np.eye(3) if self.ignore_normal else _normal_basis(normals[i], self.orient_z)
            s = np.diag(np.maximum(scale[i], 1e-6))
            linear = basis @ (turn[i] @ s if self.local_scale else s @ turn[i])
            out.matrix[i, :3, :3] = linear
            out.matrix[i, :3, 3] = position[i] + linear @ offset[i]
        color = v.get("color") * _rows3(self.color_scale(sim, v), n)
        out.tint = np.ones((n, 3)) if self.suppress_tint else color
        alpha2 = v.field(self.alpha2_field) if F.name(self.alpha2_field) and not F.is_vector(self.alpha2_field) else 1.0
        out.alpha = v.get("alpha") * alpha2 * _per(self.alpha_scale(sim, v), n)
        return out


class Lights:
    """Lights to place: per particle a position (game units), linear colour, intensity and radius (game units)."""

    def __init__(self, renderer, n):
        self.renderer = renderer
        self.count = n
        self.position = np.zeros((n, 3))
        self.color = np.ones((n, 3))
        self.intensity = np.zeros(n)
        self.radius = np.zeros(n)


class RenderStandardLight:
    """A light per particle (C_OP_RenderStandardLight): its colour times the colour scale, the intensity times its
    alpha, reaching its radius times the radius multiplier. Hosts that don't place lights leave these out."""

    kind = "light"

    def __init__(self, p: Params, library, index: int):
        self.index = index
        self.hidden = False
        self.intensity = p.number("m_flIntensity", 1.0)
        self.radius_scale = p.number("m_flRadiusMultiplier", 1.0)
        self.color_scale = p.vec("m_vecColorScale", (1.0, 1.0, 1.0))

    def build(self, sim, ambient) -> Lights | None:
        n = sim.particles.count
        if n == 0:
            return None
        v = F.View(sim.particles, slice(0, n))
        out = Lights(self, n)
        out.position = v.get("position").copy()
        out.color = v.get("color") * _rows3(self.color_scale(sim, v), n)
        out.intensity = _per(self.intensity(sim, v), n) * v.get("alpha")
        out.radius = v.get("radius") * _per(self.radius_scale(sim, v), n)
        return out


class RenderProjected(RenderSprites):
    """A decal pressed onto whatever is under the particle (C_OP_RenderProjected), drawn as a flat card on the
    ground with its material's colour map: Kelvin's dome edge, scorch and darken marks. The material's self
    illumination scale stands in for the sprite's (a lit decal glows)."""

    kind = "sprites"

    def __init__(self, p: Params, library, index: int):
        materials = [m.str("m_hMaterial") for m in p.array("m_vecProjectedMaterials") if m.str("m_hMaterial")]
        material = library.material(materials[0]) if materials else None
        if material is None:
            raise ValueError("no projected material")
        texture = material.textures.get("g_tColor")
        if not texture:
            raise ValueError("projected material has no colour map")
        tint = material.vector("g_vColorTint", (1.0, 1.0, 1.0))[:3]
        data = dict(p.data)
        data["m_vecTexturesInput"] = [{"m_hTexture": texture}]
        data["m_nOrientationType"] = "PARTICLE_ORIENTATION_WORLD_Z_ALIGNED"
        data["m_flSelfIllumAmount"] = material.float("g_flSelfIllumScale", 0.0)
        data["m_vecColorScale"] = list(tint)
        data["m_flAlphaScale"] = material.float("g_flOpacityScale", 1.0) * material.float("g_flAlbedoOpacity", 1.0)
        super().__init__(p.nested(data), library, index)

    def build(self, sim, ambient) -> Sprites | None:
        out = super().build(sim, ambient)
        if out is not None:
            out.position = out.position + [0.0, 0.0, 1.0]  # a game unit up: the card sits on the floor it projects on
        return out


RENDERERS = {
    "C_OP_RenderStandardLight": RenderStandardLight,
    "C_OP_RenderProjected": RenderProjected,
    "C_OP_RenderSprites": RenderSprites,
    "C_OP_RenderRopes": RenderRopes,
    "C_OP_RenderTrails": RenderTrails,
    "C_OP_RenderModels": RenderModels,
}
