"""Node groups every Deadlock material shares: normal map decoding, colour matrices, UV transforms and the
environment shaders' layer and reveal logic. Ports of the s&box mount's shader functions.

Groups carry a version in their name's custom property; a newer add-on rebuilds them.
"""
import bpy

from .nodes import new_group

GROUP_VERSION = 3


def _get(name):
    group = bpy.data.node_groups.get(name)
    if group is not None and group.get("dl_version") == GROUP_VERSION:
        return group
    return None


def _stamp(tree):
    tree["dl_version"] = GROUP_VERSION
    return tree


def normal_gradient():
    """Hemi-octahedral normal map (RG of g_tNormalRoughness) -> the surface gradient (x, y) such that the shaded
    normal is normalize(N + T * x + B * y) in Blender's tangent frame. Matches DecodeHemiOctahedronNormal: the game's
    bitangent runs along +V in D3D texture space, Blender's along -V, which cancels the decode's Y flip."""
    name = "DL Normal Gradient"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("Normal Roughness", "color", (0.5, 0.5, 1.0)), ("Strength", "float", 1.0)],
                              [("Gradient", "vector")])
    r, g, _ = b.separate_color(i["Normal Roughness"])
    tx = b.sub(b.add(r, g), 1.003922)
    ty = b.sub(r, g)
    tz = b.sub(b.sub(1.0, b.math("ABSOLUTE", tx)), b.math("ABSOLUTE", ty))
    n = b.normalize(b.combine(tx, ty, tz))
    nx, ny, nz = b.separate(n)
    den = b.math("MAXIMUM", b.math("ABSOLUTE", nz), b.mul(0.0078125, b.math("MAXIMUM", b.math("ABSOLUTE", nx), b.math("ABSOLUTE", ny))))
    gx = b.mul(b.div(nx, den), i["Strength"])
    gy = b.mul(b.div(ny, den), i["Strength"])
    b.set(o["Gradient"], b.combine(gx, gy, 0.0))
    return _stamp(tree)


def gradient_normal():
    """A summed surface gradient back to a world normal, through the UV map's tangent frame."""
    name = "DL Gradient Normal"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("Gradient", "vector", (0, 0, 0))], [("Normal", "vector")])
    gx, gy, _ = b.separate(i["Gradient"])
    v = b.normalize(b.combine(gx, gy, 1.0))
    color = b.vadd(b.vscale(v, 0.5), (0.5, 0.5, 0.5))
    node = b.node("ShaderNodeNormalMap", space="TANGENT", uv_map="UVMap")
    b.set(node.inputs["Color"], color)
    b.set(o["Normal"], node.outputs[0])
    return _stamp(tree)


def color_matrix():
    """A 3x4 colour matrix applied as dot(row, (rgb, 1)) per channel."""
    name = "DL Color Matrix"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("Color", "color", (1, 1, 1)), ("R", "vector", (1, 0, 0)), ("G", "vector", (0, 1, 0)),
                                     ("B", "vector", (0, 0, 1)), ("Offset", "vector", (0, 0, 0))], [("Color", "color")])
    c = b.vmath("ADD", i["Color"], (0, 0, 0))
    ox, oy, oz = b.separate(i["Offset"])
    r = b.add(b.dot(c, i["R"]), ox)
    g = b.add(b.dot(c, i["G"]), oy)
    bl = b.add(b.dot(c, i["B"]), oz)
    b.set(o["Color"], b.combine_color(r, g, bl))
    return _stamp(tree)


def uv_transform():
    """The environment shaders' ApplyUvTransform, (scale * sin, scale * cos, offset) of a rotation and scale after an
    offset, in the game's texture space (V down) on Blender's UVs (V up)."""
    name = "DL UV Transform"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("UV", "vector", (0, 0, 0)), ("Sin Cos", "vector", (0, 1, 0)), ("Offset", "vector", (0, 0, 0))],
                              [("UV", "vector")])
    u, v, _ = b.separate(i["UV"])
    v = b.sub(1.0, v)
    ox, oy, _ = b.separate(i["Offset"])
    s, c, _ = b.separate(i["Sin Cos"])
    x = b.add(u, ox)
    y = b.add(v, oy)
    nu = b.sub(b.mul(c, x), b.mul(s, y))
    nv = b.add(b.mul(s, x), b.mul(c, y))
    b.set(o["UV"], b.combine(nu, b.sub(1.0, nv), 0.0))
    return _stamp(tree)


def pbr_uv_transform():
    """pbr.vfx's TransformUv: rotate about the centre, scale, then offset, in the game's texture space."""
    name = "DL PBR UV Transform"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("UV", "vector", (0, 0, 0)), ("Scale", "vector", (1, 1, 0)), ("Offset", "vector", (0, 0, 0)),
                                     ("Rotation", "float", 0.0)], [("UV", "vector")])
    u, v, _ = b.separate(i["UV"])
    v = b.sub(1.0, v)
    angle = b.math("RADIANS", i["Rotation"])
    s = b.math("SINE", angle)
    c = b.math("COSINE", angle)
    cu = b.sub(u, 0.5)
    cv = b.sub(v, 0.5)
    ru = b.sub(b.mul(c, cu), b.mul(s, cv))
    rv = b.add(b.mul(s, cu), b.mul(c, cv))
    sx, sy, _ = b.separate(i["Scale"])
    ox, oy, _ = b.separate(i["Offset"])
    nu = b.add(b.add(b.mul(sx, ru), 0.5), ox)
    nv = b.add(b.add(b.mul(sy, rv), 0.5), oy)
    b.set(o["UV"], b.combine(nu, b.sub(1.0, nv), 0.0))
    return _stamp(tree)


def contrast_brightness(b, value, cb):
    """saturate(((value - 0.5) + cb.y) * cb.x + 0.5), inline."""
    x, y, _ = b.separate(cb)
    return b.saturate(b.add(b.mul(b.add(b.sub(value, 0.5), y), x), 0.5))


def env_layer():
    """One environment_blend layer (SampleLayer): colour corrected and tinted albedo, the vertex paint, the normal map as
    a surface gradient, and the packed roughness, metalness, tint mask and self illumination."""
    name = "DL Environment Layer"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [
        ("Color", "color", (1, 1, 1)), ("Alpha", "float", 1.0), ("Normal Roughness", "color", (0.5, 0.5, 1.0)),
        ("Packed", "color", (0, 0, 0)), ("Vertex Color", "color", (1, 1, 1)),
        ("Correct R", "vector", (1, 0, 0)), ("Correct G", "vector", (0, 1, 0)), ("Correct B", "vector", (0, 0, 1)),
        ("Correct Offset", "vector", (0, 0, 0)),
        ("Tint R", "vector", (1, 0, 0)), ("Tint G", "vector", (0, 1, 0)), ("Tint B", "vector", (0, 0, 1)),
        ("Tint Offset", "vector", (0, 0, 0)),
        ("Mask Color Tint", "float", 1.0), ("Mask Vertex Color Tint", "float", 1.0), ("Model Tint", "float", 1.0),
        ("Vertex Color Strength", "float", 1.0), ("Normal Strength", "float", 1.0), ("Self Illum Scale", "float", 1.0),
        ("Roughness CB", "vector", (1, 0, 0)), ("Tint Mask CB", "vector", (1, 0, 0)), ("Metalness CB", "vector", (1, 0, 0)),
    ], [("Albedo", "color"), ("Gradient", "vector"), ("Roughness", "float"), ("Metalness", "float"),
        ("Tint Mask", "float"), ("Self Illum", "float")])

    tint_mask = contrast_brightness(b, i["Alpha"], i["Tint Mask CB"])
    _, _, nr_b = b.separate_color(i["Normal Roughness"])
    roughness = contrast_brightness(b, nr_b, i["Roughness CB"])
    p_r, p_g, _ = b.separate_color(i["Packed"])
    metalness = contrast_brightness(b, p_g, i["Metalness CB"])
    self_illum = b.mul(p_r, i["Self Illum Scale"])

    b.column(4)
    gradient = b.group(normal_gradient(), Normal_Roughness=i["Normal Roughness"], Strength=i["Normal Strength"]).outputs[0]

    corrected = b.group(color_matrix(), Color=i["Color"], R=i["Correct R"], G=i["Correct G"], B=i["Correct B"],
                        Offset=i["Correct Offset"]).outputs[0]
    tinted = b.group(color_matrix(), Color=corrected, R=i["Tint R"], G=i["Tint G"], B=i["Tint B"], Offset=i["Tint Offset"]).outputs[0]
    b.column(8)
    albedo = b.clerp(corrected, tinted, b.lerp(1.0, tint_mask, i["Mask Color Tint"]))
    amount = b.mul(b.mul(i["Model Tint"], b.lerp(1.0, tint_mask, i["Mask Vertex Color Tint"])), i["Vertex Color Strength"])
    albedo = b.clerp(albedo, b.clerp(albedo, i["Vertex Color"], 1.0, "MULTIPLY"), amount)

    b.set(o["Albedo"], albedo)
    b.set(o["Gradient"], gradient)
    b.set(o["Roughness"], roughness)
    b.set(o["Metalness"], metalness)
    b.set(o["Tint Mask"], tint_mask)
    b.set(o["Self Illum"], self_illum)
    return _stamp(tree)


def reveal_weight():
    """How much a layer above the first shows: its reveal mask against the vertex paint (RevealThreshold), softened."""
    name = "DL Reveal"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("Mask", "float", 0.5), ("Blend", "float", 0.0), ("Invert", "float", 0.0),
                                     ("Offset", "float", 0.0), ("Softness", "float", 0.125)], [("Weight", "float")])
    reveal = b.lerp(i["Mask"], b.sub(1.0, i["Mask"]), i["Invert"])
    half = b.mul(i["Offset"], 0.5)
    middle = b.add(0.5, half)
    low = b.mul(b.saturate(b.mul(reveal, 2.0)), middle)
    high = b.add(b.mul(b.saturate(b.mul(b.sub(reveal, 0.5), 2.0)), b.sub(0.5, half)), middle)
    remapped = b.select(b.math("LESS_THAN", reveal, 0.5), low, high)
    clamped = b.math("MINIMUM", b.math("MAXIMUM", remapped, 1e-4), 0.9999)
    threshold = b.add(b.mul(clamped, b.sub(1.0, b.mul(2.0, i["Softness"]))), i["Softness"])
    soft = b.smoothstep(b.mul(i["Softness"], -1.0), i["Softness"], b.sub(i["Blend"], threshold))
    weight = b.select(b.math("GREATER_THAN", i["Blend"], 0.99999), 1.0, soft)
    b.set(o["Weight"], weight)
    return _stamp(tree)


def env_composite():
    """The environment shaders' G-buffer packing: self illumination mixed into the albedo, glowing up to ten times."""
    name = "DL Self Illum Pack"
    if (g := _get(name)) is not None:
        return g
    tree, b, i, o = new_group(name, [("Albedo", "color", (1, 1, 1)), ("Illum", "color", (0, 0, 0)), ("Self Illum", "float", 0.0)],
                              [("Albedo", "color"), ("Emission", "color")])
    numerator = b.vadd(b.vscale(i["Albedo"], 0.1), i["Illum"])
    albedo = b.vscale(numerator, b.div(1.0, b.add(0.1, i["Self Illum"])))
    glow = b.mul(b.saturate(b.mul(i["Self Illum"], 0.1)), 10.0)
    b.set(o["Albedo"], albedo)
    b.set(o["Emission"], b.vscale(albedo, glow))
    return _stamp(tree)
