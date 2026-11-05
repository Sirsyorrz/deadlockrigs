"""The lights Deadlock lights in realtime (the sun, dynamic and stationary light2 lights) as Blender lights.

Static lights only exist in the baked lightmaps and probes, so they get no Blender light. Brightness follows the game's
deferred pass (deadlock_lighting): light2 lights are ``colour * 2^brightness * 100^2 / d^2`` in game units (barn lights
from the eye behind their luminaire), the sun and parallel barns ``colour * brightness`` (Blender suns), times N.L and
albedo. Measured in Blender,
a sun of strength S gives ``albedo S / pi`` and a point or spot light of P watts ``albedo P / (4 pi^2 d^2)`` (Cycles and
EEVEE alike), which sets the conversions below.

Cycles also reproduces the rest of the game's falloff through light node trees: the skirt fade at the light's range,
the cone ramp, and barn lights' frustum, soft edges and cookie. EEVEE has no light nodes, so it gets the cone as a
spot cone and the range as the light's cut-off distance.
"""
import math

import numpy as np

import bpy
from mathutils import Matrix

from ..core.probes import rotation_matrix
from .lighting import color_temperature, srgb_to_linear
from .nodes import Builder

LINEAR_BRIGHTNESS_DISTANCE = 100.0


def is_realtime(e) -> bool:
    light2 = e.classname in ("light_omni2", "light_barn", "light_rect")
    direct = e.get_int("directlight", 2 if light2 else 1)
    baked_index = e.get_int("bakedshadowindex", e.get_int("bakelightindex", -1))
    if direct in (2, 3):
        return True
    return direct == 1 and baked_index >= 0


def blender_rotation(angles) -> Matrix:
    """A light's orientation: local -Z along the game's forward, +Y up."""
    r = rotation_matrix(angles)
    forward, left, up = r[:, 0], r[:, 1], r[:, 2]
    m = np.stack([-left, up, -forward], 1)
    return Matrix([list(m[0]), list(m[1]), list(m[2])]).to_4x4()


def _color(e):
    if e.get_int("colormode") != 0:
        rgb = color_temperature(e.get_float("colortemperature", 6600.0))
    else:
        rgb = [c / 255.0 for c in e.get_vector("color", 3, [255, 255, 255])]
    return [srgb_to_linear(c) for c in rgb]


def _set_color(light, linear, energy):
    peak = max(max(linear), 1e-6)
    light.color = tuple(c / peak for c in linear)
    light.energy = energy * peak


def create_light(e, scale: float, collection, cookies=None) -> bpy.types.Object | None:
    cls = e.classname
    origin = np.array(e.get_vector("origin", 3, [0, 0, 0]))
    angles = e.get_vector("angles", 3, [0, 0, 0])
    name = e.targetname or cls

    if cls == "light_environment":
        light = bpy.data.lights.new(name, "SUN")
        color = [srgb_to_linear(c / 255.0) for c in e.get_vector("color", 3, [255, 255, 255])]
        brightness = e.get_float("brightness", 1.0) * e.get_float("brightnessscale", 1.0)
        _set_color(light, color, math.pi * brightness)
        light.angle = math.radians(max(e.get_float("angulardiameter", 1.0), 0.0))
        light.use_shadow = e.get_int("castshadows", 1) != 0
    elif cls in ("light_omni2", "light_rect", "light_barn"):
        light = _light2(e, name, scale, cookies)
        if light is None:
            return None
    elif cls in ("light_omni", "light_spot", "light_ortho"):
        light = _legacy(e, name, scale)
    else:
        return None

    light.diffuse_factor = 1.0 if e.get_int("renderdiffuse", 1) != 0 else 0.0
    light.specular_factor = 1.0 if e.get_int("renderspecular", 1) == 1 else 0.0
    obj = bpy.data.objects.new(name, light)
    matrix = blender_rotation(angles)
    if light.get("dl_rotation") is not None:
        matrix = Matrix([list(row) for row in light["dl_rotation"]]).to_4x4()
    if light.get("dl_eye") is not None:
        origin = np.array(light["dl_eye"])
    matrix.translation = tuple(origin * scale)
    obj.matrix_world = matrix
    obj["dl_class"] = cls
    obj["dl_stationary"] = e.get_int("bakedshadowindex", e.get_int("bakelightindex", -1)) >= 0
    collection.objects.link(obj)
    if not e.get_bool("enabled", True) or e.get_bool("startdisabled"):
        obj.hide_render = True
        obj.hide_viewport = True
    return obj


def _light2(e, name, scale, cookies):
    cls = e.classname
    rng = e.get_float("range", 512.0)
    if rng <= 1e-4:
        return None
    intensity = 2.0 ** e.get_float("brightness") * e.get_float("brightnessscale", 1.0)
    color = _color(e)
    skirt = e.get_float("skirt", 0.1)
    size = e.get_vector("size_params", 3, [0, 0, 0])
    k = 4.0 * math.pi ** 2 * scale ** 2

    if cls == "light_barn":
        if size[2] <= 0:
            return _parallel_barn(e, name, intensity, color)
        return _barn(e, name, scale, cookies, rng, intensity, color, skirt, size, k)

    outer = 90.0 if cls == "light_rect" else e.get_float("outer_angle", 180.0)
    inner = 0.0 if cls == "light_rect" else e.get_float("inner_angle", 0.0)
    if outer >= 180.0:
        light = bpy.data.lights.new(name, "POINT")
    else:
        light = bpy.data.lights.new(name, "SPOT")
        light.spot_size = math.radians(min(outer, 179.9) * 2.0)
        cos_outer = math.cos(math.radians(outer))
        cos_inner = math.cos(math.radians(min(max(inner, 0.0), outer - 0.001))) if inner > 0 else 1.0
        light.spot_blend = max(0.0, min(1.0, (cos_inner - cos_outer) / max(1.0 - cos_outer, 1e-6)))
        light.show_cone = False
    _set_color(light, color, k * LINEAR_BRIGHTNESS_DISTANCE ** 2 * intensity)
    luminaire = size[1] if e.get_int("shape") in (1, 2) else e.get_float("luminaire_size", 0.0)
    light.shadow_soft_size = max(luminaire, 1.0) * 0.5 * scale
    light.use_shadow = e.get_int("castshadows", 1) != 0
    light.use_custom_distance = True
    light.cutoff_distance = rng * scale

    # The skirt: saturate(offset + rate * d) through a smoothstep, as the game fades light2 lights at their range.
    if skirt <= 0:
        offset, rate = 1.0, 0.0
    elif skirt >= 1:
        offset, rate = 1.0, -1.0 / rng
    else:
        offset, rate = 1.0 / skirt, -1.0 / (rng * skirt)
    cone = None
    if outer < 180.0:
        cos_outer = min(math.cos(math.radians(outer)), 1 - 1e-7)
        cos_inner = math.cos(math.radians(min(max(inner, 0.0), outer - 0.001))) if inner > 0 else 1.0
        denominator = cos_outer - cos_inner
        if denominator >= 0:
            denominator = -1e-7
        cone = (cos_outer / denominator, 1.0 / denominator)
    _falloff_tree(light, scale, offset, rate, cone, rng)
    light["dl_range"] = rng
    return light


def _falloff_tree(light, scale, offset, rate, cone, rng):
    light.use_nodes = True
    tree = light.node_tree
    emission = next(n for n in tree.nodes if n.type == "EMISSION")
    b = Builder(tree)
    b.column(-6)
    falloff = b.node("ShaderNodeLightFalloff")
    falloff.inputs["Strength"].default_value = 1.0
    d = b.mul(b.math("SQRT", falloff.outputs["Constant"]), 1.0 / scale)
    fade = b.smoothstep(0.0, 1.0, b.madd(d, rate, offset))
    strength = b.mul(fade, b.math("LESS_THAN", d, rng))
    if cone is not None:
        coord = b.node("ShaderNodeTexCoord")
        _, _, z = b.separate(b.normalize(coord.outputs["Normal"]))
        cos = b.mul(z, -1.0)
        strength = b.mul(strength, b.saturate(b.madd(cos, cone[1], cone[0])))
    b.set(emission.inputs["Strength"], strength)


def _barn_frame(e):
    """A light_barn's frustum axes (x, y, depth) in the world."""
    r = rotation_matrix(e.get_vector("angles", 3, [0, 0, 0]))
    forward, up = r[:, 0], r[:, 2]
    side = np.cross(up, forward)  # the frustum's x axis: ValveResourceFormat's "right", the game's left
    side /= max(np.linalg.norm(side), 1e-9)
    return side, up, forward


def barn_box(e):
    """A parallel light_barn (no near plane: a box of parallel rays) as the lighting group lights it: its linear colour
    times brightness, and its box (see lighting._barn_box). Like the sun, it doesn't fall off with distance."""
    side, up, forward = _barn_frame(e)
    size = e.get_vector("size_params", 3, [0, 0, 0])
    color = [c * 2.0 ** e.get_float("brightness") * e.get_float("brightnessscale", 1.0) for c in _color(e)]
    box = {"origin": e.get_vector("origin", 3, [0, 0, 0]), "side": tuple(side), "up": tuple(up), "forward": tuple(forward),
           "size": (max(size[0], 1e-3), max(size[1], 1e-3)), "range": max(e.get_float("range", 512.0), 1e-3),
           "shear": tuple(e.get_vector("shear", 2, [0.0, 0.0])),
           "soft": (e.get_float("soft_x", 0.25), e.get_float("soft_y", 0.25)),
           "skirt": e.get_float("skirt", 0.1), "skirt_near": e.get_float("skirt_near", 0.05)}
    return color, box


def _parallel_barn(e, name, intensity, color):
    """A parallel light_barn as a sun: it covers whole maps (moonlight), and its rays don't fall off."""
    light = bpy.data.lights.new(name, "SUN")
    _set_color(light, color, math.pi * intensity)
    light.angle = math.radians(1.0)
    light.use_shadow = e.get_int("castshadows", 1) != 0
    light["dl_range"] = e.get_float("range", 512.0)
    return light


def _barn(e, name, scale, cookies, rng, intensity, color, skirt, size, k):
    """light_barn with a near plane: a frustum projecting a soft rectangle or a cookie. The Blender spot sits at the
    frustum's eye aimed through its centre; Cycles masks it to the frustum."""
    soft_x = e.get_float("soft_x", 0.25)
    soft_y = e.get_float("soft_y", 0.25)
    skirt_near = e.get_float("skirt_near", 0.05)
    shear = e.get_vector("shear", 2, [0.0, 0.0])
    side, up, forward = _barn_frame(e)
    origin = np.array(e.get_vector("origin", 3, [0, 0, 0]))
    sx, sy = max(size[0], 1e-3), max(size[1], 1e-3)

    near = 1.0 / size[2]
    cx = shear[0] / rng * near
    cy = shear[1] / rng * near
    eye = origin - near * forward - cx * side - cy * up
    falloff_sq = cx * cx + cy * cy + near * near
    scale_factor = (math.sqrt(falloff_sq) + LINEAR_BRIGHTNESS_DISTANCE) ** 2 / falloff_sq
    energy = k * falloff_sq * intensity * scale_factor
    far = near + rng

    # Aim through the frustum's centre at its far plane, the cone wide enough for its corners.
    frame = np.stack([side, up, forward], 0)  # world -> frustum (x, y, z)
    corners = [np.array([(cx + x * sx) / near, (cy + y * sy) / near, 1.0]) for x in (-1, 1) for y in (-1, 1)]
    centre = np.array([cx / near, cy / near, 1.0])
    centre /= np.linalg.norm(centre)
    half_angle = max(math.acos(min(1.0, float(np.dot(c / np.linalg.norm(c), centre)))) for c in corners)
    axis = frame.T @ centre

    light = bpy.data.lights.new(name, "SPOT")
    light.show_cone = False
    light.use_shadow = e.get_int("castshadows", 1) != 0
    light.use_custom_distance = True
    light.cutoff_distance = far * scale * 1.05
    _set_color(light, color, energy)
    light.spot_size = min(2.0 * half_angle * 1.02, math.radians(179.9))
    light.spot_blend = max(0.05, min(1.0, 0.5 * (soft_x + soft_y)))
    light.shadow_soft_size = max(e.get_float("luminaire_size", 4.0), 1.0) * 0.5 * scale
    light["dl_eye"] = list(eye)
    rotation = _aim(axis, up)
    light["dl_rotation"] = [list(row) for row in rotation]

    skirt_near_scale = (near + skirt_near * rng) / (far * skirt_near) if skirt_near > 0 else 0.0
    skirt_far_scale = (near + (1 - skirt) * rng) / (near * skirt) if skirt > 0 else 0.0
    cookie_image = cookies(e.get_str("lightcookie")) if cookies is not None and e.get_str("lightcookie") else None

    # Light space -> frustum frame: the light's local axes are rotation's columns.
    to_frustum = frame @ rotation

    light.use_nodes = True
    tree = light.node_tree
    emission = next(n for n in tree.nodes if n.type == "EMISSION")
    b = Builder(tree)
    b.column(-9)
    coord = b.node("ShaderNodeTexCoord")
    falloff = b.node("ShaderNodeLightFalloff")
    falloff.inputs["Strength"].default_value = 1.0
    d = b.mul(b.math("SQRT", falloff.outputs["Constant"]), 1.0 / scale)
    direction = b.normalize(coord.outputs["Normal"])
    b.column(-8)
    fx = b.mul(b.dot(direction, tuple(to_frustum[0])), d)
    fy = b.mul(b.dot(direction, tuple(to_frustum[1])), d)
    fz = b.mul(b.dot(direction, tuple(to_frustum[2])), d)
    b.column(-6)
    safe = b.math("MAXIMUM", fz, 1e-4)
    ndc_x = b.div(b.sub(b.mul(b.div(fx, safe), near), cx), sx)
    ndc_y = b.div(b.sub(b.mul(b.div(fy, safe), near), cy), sy)
    ndc_z = b.mul(far / (far - near), b.sub(1.0, b.div(near, safe)))
    inside = b.mul(b.mul(b.math("LESS_THAN", b.math("ABSOLUTE", ndc_x), 1.0), b.math("LESS_THAN", b.math("ABSOLUTE", ndc_y), 1.0)),
                   b.mul(b.math("GREATER_THAN", ndc_z, 0.0), b.math("LESS_THAN", ndc_z, 1.0)))
    strength = inside
    if skirt_near_scale > 0:
        strength = b.mul(strength, b.smoothstep(0.0, 1.0, b.mul(ndc_z, skirt_near_scale)))
    if skirt_far_scale > 0:
        strength = b.mul(strength, b.smoothstep(0.0, 1.0, b.mul(b.sub(1.0, ndc_z), skirt_far_scale)))
    b.column(-4)
    if cookie_image is not None:
        uv = b.combine(b.madd(ndc_x, 0.5, 0.5), b.madd(ndc_y, 0.5, 0.5), 0.0)
        tex = b.image(cookie_image, uv, extension="EXTEND", label="Cookie")
        b.set(emission.inputs["Color"], tex.outputs["Color"])
    else:
        sx_mask = b.smoothstep(1.0, 1.0 - soft_x, b.math("ABSOLUTE", ndc_x))
        sy_mask = b.smoothstep(1.0, 1.0 - soft_y, b.math("ABSOLUTE", ndc_y))
        strength = b.mul(strength, b.mul(sx_mask, sy_mask))
    b.set(emission.inputs["Strength"], strength)
    light["dl_range"] = rng
    return light


# Light at a point ---------------------------------------------------------------------------------------------------
# What the game lights particles with: each realtime light's colour arriving at a point, facing nowhere in particular
# (no N.L), stationary lights behind their baked shadows from the light probes.

def _smoothstep(edge0, edge1, x):
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0) if edge1 != edge0 else float(x >= edge0)
    return t * t * (3.0 - 2.0 * t)


def light_at(e, position, baked_shadows=None) -> np.ndarray:
    """Linear light from a realtime light entity at a point (game units). ``baked_shadows``: the probes' four baked
    shadow channels there, for stationary lights."""
    if not is_realtime(e) or not e.get_bool("enabled", True) or e.get_bool("startdisabled"):
        return np.zeros(3)
    cls = e.classname
    p = np.asarray(position, np.float64)
    index = e.get_int("bakedshadowindex", e.get_int("bakelightindex", -1))
    visible = 1.0
    if baked_shadows is not None and 0 <= index < 4:
        visible = min(max(1.0 - float(baked_shadows[index]), 0.0), 1.0)
    if visible <= 0:
        return np.zeros(3)

    if cls == "light_environment":
        brightness = e.get_float("brightness", 1.0) * e.get_float("brightnessscale", 1.0)
        return np.array([srgb_to_linear(c / 255.0) for c in e.get_vector("color", 3, [255, 255, 255])]) * brightness * visible
    if cls not in ("light_omni2", "light_rect", "light_barn"):
        return np.zeros(3)

    color = np.array(_color(e)) * 2.0 ** e.get_float("brightness") * e.get_float("brightnessscale", 1.0) * visible
    rng = e.get_float("range", 512.0)
    skirt = e.get_float("skirt", 0.1)
    origin = np.array(e.get_vector("origin", 3, [0, 0, 0]))
    size = e.get_vector("size_params", 3, [0, 0, 0])

    if cls == "light_barn":
        side, up, forward = _barn_frame(e)
        soft_x, soft_y = e.get_float("soft_x", 0.25), e.get_float("soft_y", 0.25)
        skirt_near = e.get_float("skirt_near", 0.05)
        shear = e.get_vector("shear", 2, [0.0, 0.0])
        sx, sy = max(size[0], 1e-3), max(size[1], 1e-3)
        if size[2] <= 0:
            local = p - origin
            z = float(local @ forward)
            ndc = ((float(local @ side) - z * shear[0] / rng) / sx, (float(local @ up) - z * shear[1] / rng) / sy, z / rng)
            near_scale = 1.0 / skirt_near if skirt_near > 0 else 0.0
            far_scale = 1.0 / skirt if skirt > 0 else 0.0
            strength = 1.0
        else:
            near = 1.0 / size[2]
            cx, cy = shear[0] / rng * near, shear[1] / rng * near
            eye = origin - near * forward - cx * side - cy * up
            far = near + rng
            v = p - eye
            fz = float(v @ forward)
            if fz <= 1e-4:
                return np.zeros(3)
            ndc = ((float(v @ side) / fz * near - cx) / sx, (float(v @ up) / fz * near - cy) / sy,
                   far / (far - near) * (1.0 - near / fz))
            near_scale = (near + skirt_near * rng) / (far * skirt_near) if skirt_near > 0 else 0.0
            far_scale = (near + (1 - skirt) * rng) / (near * skirt) if skirt > 0 else 0.0
            falloff_sq = cx * cx + cy * cy + near * near
            scale_factor = (math.sqrt(falloff_sq) + LINEAR_BRIGHTNESS_DISTANCE) ** 2 / falloff_sq
            strength = falloff_sq * scale_factor / max(float(v @ v), 1e-6)
        if not (abs(ndc[0]) < 1 and abs(ndc[1]) < 1 and 0 < ndc[2] < 1):
            return np.zeros(3)
        if near_scale > 0:
            strength *= _smoothstep(0.0, 1.0, ndc[2] * near_scale)
        if far_scale > 0:
            strength *= _smoothstep(0.0, 1.0, (1.0 - ndc[2]) * far_scale)
        strength *= _smoothstep(1.0, 1.0 - soft_x, abs(ndc[0])) * _smoothstep(1.0, 1.0 - soft_y, abs(ndc[1]))
        return color * strength

    d = float(np.linalg.norm(p - origin))
    if d >= rng:
        return np.zeros(3)
    if skirt <= 0:
        fade = 1.0
    else:
        fade = _smoothstep(0.0, 1.0, (1.0 - d / rng) / min(skirt, 1.0))
    strength = fade * LINEAR_BRIGHTNESS_DISTANCE ** 2 / max(d * d, 1.0)
    outer = 90.0 if cls == "light_rect" else e.get_float("outer_angle", 180.0)
    if outer < 180.0 and d > 0:
        inner = 0.0 if cls == "light_rect" else e.get_float("inner_angle", 0.0)
        forward = rotation_matrix(e.get_vector("angles", 3, [0, 0, 0]))[:, 0]
        cos = float((p - origin) @ forward) / d
        cos_outer = min(math.cos(math.radians(outer)), 1 - 1e-7)
        cos_inner = math.cos(math.radians(min(max(inner, 0.0), outer - 0.001))) if inner > 0 else 1.0
        strength *= min(max((cos - cos_outer) / max(cos_inner - cos_outer, 1e-7), 0.0), 1.0)
    return color * strength


def _aim(axis, up) -> np.ndarray:
    """A rotation whose -Z is ``axis``, +Y as close to ``up`` as it can be."""
    z = -axis / max(np.linalg.norm(axis), 1e-9)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def _legacy(e, name, scale):
    cls = e.classname
    brightness = e.get_float("brightness", 1.0) * e.get_float("brightnessscale", 1.0)
    color = [srgb_to_linear(c / 255.0) for c in e.get_vector("color", 3, [255, 255, 255])]
    if cls == "light_spot":
        light = bpy.data.lights.new(name, "SPOT")
        outer = e.get_float("outerconeangle", e.get_float("outer_angle", 45.0))
        inner = e.get_float("innerconeangle", e.get_float("inner_angle", 0.0))
        light.spot_size = math.radians(min(outer, 89.9) * 2)
        light.spot_blend = max(0.0, min(1.0, 1.0 - inner / max(outer, 1e-3)))
        light.show_cone = False
    else:
        light = bpy.data.lights.new(name, "POINT")
    _set_color(light, color, 4.0 * math.pi ** 2 * scale ** 2 * LINEAR_BRIGHTNESS_DISTANCE ** 2 * brightness)
    light.use_custom_distance = True
    light.cutoff_distance = e.get_float("range", 512.0) * scale
    return light
