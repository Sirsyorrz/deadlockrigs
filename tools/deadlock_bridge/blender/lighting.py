"""The map's baked lighting as a node group every material ends in, as Deadlock's deferred lighting pass does it.

``DL Lighting <map>`` takes what the game's G-buffer holds (albedo, metalness, roughness, normal, self illumination)
and returns the shaded surface:

* indirect light from the second order harmonic lightmaps (lightmap version 8.5: a BC6H DC term and a shared
  luminance direction, ``L1 = (lum * 2 - 1) * dc * sqrt(3)``, with the ZH3 sharpening of the decompiled
  ``deferred_lighting`` compute shader) or, off the lightmaps, the light probes' ambient cubes (per object);
* weighted by the game's split sum energy from its own BRDF lookup table (``textures/dev/brdf_lut``);
* emitted (it is baked light), with a Principled BSDF on top for the lights the game lights in realtime (the sun and
  the dynamic and stationary light2 lights, which become Blender lights) and for reflections;
* then the map's gradient and cubemap fog, on camera rays, as Deadlock's forward shaders apply them.
"""
import math

import bpy

from .nodes import Builder, new_group

SQRT3 = 1.73205077648162841796875


def _lightmap_irradiance(b: Builder, normal, dc_image, lum_image):
    """Irradiance and the lightmapped flag from the lightmaps at the vertex attribute ``dl_lightmap``."""
    attr = b.attribute("dl_lightmap")
    lx, ly, _ = b.separate(attr.outputs["Vector"])
    flag = b.math("GREATER_THAN", lx, 0.5)
    uv = b.combine(b.sub(lx, 1.0), b.sub(ly, 1.0), 0.0)

    b.column(2)
    dc = b.image(dc_image, uv, extension="EXTEND", label="SH2 DC").outputs["Color"]
    if lum_image is None:
        return b.clerp(dc, dc, 0.0), flag
    lum = b.image(lum_image, uv, extension="EXTEND", label="SH2 Lum").outputs["Color"]

    # L1 direction (x, y, z) from the texture's (y, z, x) storage: (-b, -r, g) after the 2x - 1 expansion.
    b.column(3)
    lr, lg, lb = b.separate_color(lum)
    lx_ = b.mul(b.sub(b.mul(lb, 2.0), 1.0), -SQRT3)
    ly_ = b.mul(b.sub(b.mul(lr, 2.0), 1.0), -SQRT3)
    lz_ = b.mul(b.sub(b.mul(lg, 2.0), 1.0), SQRT3)
    l1 = b.combine(lx_, ly_, lz_)

    b.column(4)
    d = b.dot(l1, normal)
    ratio = b.vmath("LENGTH", l1)
    cos_theta = b.div(d, b.math("MAXIMUM", ratio, 1e-5))
    linear = b.mul(b.madd(d, 1.0233277, 0.886227548), 0.318309873)
    zonal = b.mul(b.mul(ratio, b.madd(ratio, 0.6, 0.08)), b.mul(0.0788478926, b.sub(b.mul(3.0, b.mul(cos_theta, cos_theta)), 1.0)))
    k = b.math("MAXIMUM", b.add(linear, zonal), 0.0)
    return b.vscale(dc, k), flag


def _probe_irradiance(b: Builder, normal):
    """The light probes' ambient cube at the object, from its dl_amb_* custom properties: each face weighed by the
    normal's squared component, picked by its sign."""
    nx, ny, nz = b.separate(normal)
    total = None
    for axis, n in (("x", nx), ("y", ny), ("z", nz)):
        pos = b.attribute(f"dl_amb_p{axis}", "OBJECT").outputs["Color"]
        neg = b.attribute(f"dl_amb_n{axis}", "OBJECT").outputs["Color"]
        face = b.clerp(neg, pos, b.math("GREATER_THAN", n, 0.0))
        term = b.vscale(face, b.mul(n, n))
        total = term if total is None else b.vadd(total, term)
    return total


def build_lighting_group(map_name: str, images: dict, unit_scale: float, fog_group, suns: list | None = None) -> bpy.types.NodeTree:
    """The per-map lighting group. ``images``: 'dc', 'lum' (lightmaps, may be None), 'shadows' (direct light shadows)
    and 'brdf' (lookup table). ``suns``: the stationary directional lights (the light_environment, parallel light_barn
    lights), lit here on lightmapped surfaces, each as 'direction' (towards it), 'color' (linear, times brightness),
    'mask' (its channel of the baked shadows) and for a barn 'box' (see _barn_box)."""
    tree, b, i, o = new_group(f"DL Lighting {map_name}", [
        ("Albedo", "color", (0.8, 0.8, 0.8)), ("Metalness", "float", 0.0), ("Roughness", "float", 0.5),
        ("Normal", "vector", (0, 0, 0)), ("Emission", "color", (0, 0, 0)), ("Alpha", "float", 1.0),
        ("Ambient Occlusion", "float", 1.0), ("Transmission", "float", 0.0), ("Fog", "float", 1.0),
    ], [("Shader", "shader")])

    b.column(0)
    geometry = b.node("ShaderNodeNewGeometry")
    # A material that leaves Normal unconnected gets the shading normal.
    n_in = i["Normal"]
    has_normal = b.math("GREATER_THAN", b.vmath("LENGTH", n_in), 0.5)
    normal = b.normalize(b.vlerp(geometry.outputs["Normal"], n_in, has_normal))
    albedo = i["Albedo"]
    metal = b.saturate(i["Metalness"])
    roughness = b.math("MAXIMUM", b.saturate(i["Roughness"]), 0.05)

    b.column(1)
    b.frame("Lightmaps")
    if images.get("dc") is not None:
        lightmap, flag = _lightmap_irradiance(b, normal, images["dc"], images.get("lum"))
    else:
        lightmap, flag = b.rgb((0, 0, 0)), 0.0
    b.frame(None)

    b.column(5)
    b.frame("Light probes")
    # World geometry off the lightmaps has the probes baked per corner along its normal (dl_vamb, where the float
    # dl_vprobe is 1), as the game samples them per pixel there; elsewhere the object's ambient cube. The flag is a
    # float of its own: a missing colour attribute can read alpha 1.
    per_vertex = b.attribute("dl_vprobe").outputs["Fac"]
    probes = b.clerp(_probe_irradiance(b, normal), b.attribute("dl_vamb").outputs["Color"], per_vertex)
    b.frame(None)

    b.column(6)
    indirect = b.clerp(probes, lightmap, flag)
    strength = b.value(1.0, "Baked Light")
    strength.node.name = "dl_baked_light"
    indirect = b.vscale(indirect, b.mul(i["Ambient Occlusion"], strength))

    # Split sum energy from the game's BRDF lookup (array slice 1).
    n_dot_v = b.saturate(b.dot(normal, geometry.outputs["Incoming"]))
    ar, ag, ab = b.separate_color(albedo)
    albedo_max = b.math("MAXIMUM", ar, b.math("MAXIMUM", ag, ab))
    spec_fade = b.saturate(b.mul(albedo_max, 25.0))
    spec_color = b.vscale(b.vlerp((0.04, 0.04, 0.04), albedo, metal), spec_fade)

    b.column(7)
    if images.get("brdf") is not None:
        lu = b.madd(roughness, 0.984375, 0.0078125)
        lv = b.madd(b.math("SQRT", b.sub(1.0, n_dot_v)), 0.984375, 0.0078125)
        lookup = b.image(images["brdf"], b.combine(lu, b.sub(1.0, lv), 0.0), extension="EXTEND", label="BRDF LUT").outputs["Color"]
        lx, ly, _ = b.separate_color(lookup)
    else:
        lx, ly = b.value(0.9), b.value(1.0)

    b.column(8)
    env_brdf = b.vadd(b.combine(lx, lx, lx), b.vscale(spec_color, b.sub(ly, lx)))
    missing = b.sub(1.0, ly)
    average = b.vadd(spec_color, b.vscale(b.vsub((1, 1, 1), spec_color), 0.0476190485))
    denominator = b.vsub((1, 1, 1), b.vscale(average, missing))
    multi = b.vscale(b.vmath("DIVIDE", b.vmul(env_brdf, average), denominator), missing)
    diffuse_energy = b.vsub((1, 1, 1), b.vadd(env_brdf, multi))

    b.column(9)
    diffuse_albedo = b.vscale(albedo, b.mul(b.sub(1.0, metal), b.sub(1.0, i["Transmission"])))
    baked = b.vmul(b.vmul(indirect, diffuse_energy), diffuse_albedo)
    baked = b.vadd(baked, b.vmul(multi, indirect))
    emission = b.vadd(baked, i["Emission"])

    # Stationary directional lights on lightmapped surfaces, shadowed by their channel of the baked shadows (as the
    # game does for static geometry); props get Blender lights instead.
    if suns and images.get("dc") is not None:
        b.column(12)
        b.frame("Stationary lights")
        shadows = None
        if images.get("shadows") is not None:
            attr = b.attribute("dl_lightmap")
            sx, sy, _ = b.separate(attr.outputs["Vector"])
            node = b.image(images["shadows"], b.combine(b.sub(sx, 1.0), b.sub(sy, 1.0), 0.0), extension="EXTEND",
                           label="Direct Light Shadows")
            shadows = (*b.separate_color(node.outputs["Color"]), node.outputs["Alpha"])
        # Off the lightmaps, the probes' baked shadows baked per vertex (dl_vshd, where dl_vprobe is 1).
        probe_node = b.attribute("dl_vshd")
        probe_shadows = (*b.separate_color(probe_node.outputs["Color"]), probe_node.outputs["Alpha"])
        off_lightmap = b.mul(b.sub(1.0, flag), per_vertex)

        def masked_visibility(channels, mask):
            masked = None
            for channel, weight in zip(channels, mask):
                if weight:
                    term = b.mul(channel, weight)
                    masked = term if masked is None else b.add(masked, term)
            return b.saturate(b.sub(1.0, masked))

        for index, sun in enumerate(suns):
            b.column(13)
            m = sun["mask"]
            visibility = flag
            if shadows is not None and any(m):
                visibility = b.mul(flag, masked_visibility(shadows, m))
            visibility = b.add(visibility, b.mul(off_lightmap, masked_visibility(probe_shadows, m)))
            if sun.get("box") is not None:
                visibility = b.mul(visibility, _barn_box(b, geometry.outputs["Position"], sun["box"], unit_scale))
            sun_strength = b.value(1.0, sun.get("label", "Sun"))
            sun_strength.node.name = "dl_sun" if index == 0 else f"dl_sun{index}"
            color = b.vscale(tuple(sun["color"]), b.mul(visibility, sun_strength))
            b.column(14)
            diffuse, specular = _direct_light(b, normal, geometry.outputs["Incoming"], tuple(sun["direction"]), roughness,
                                              spec_color, n_dot_v, color)
            emission = b.vadd(emission, b.vadd(b.vmul(diffuse, diffuse_albedo), specular))
        b.frame(None)

    # Render passes for checking the shading against the game: View Layer > Shader AOV "dl_albedo", "dl_baked".
    for aov_name, value in (("dl_albedo", albedo), ("dl_baked", indirect)):
        aov = b.node("ShaderNodeOutputAOV")
        aov.aov_name = aov_name
        b.set(aov.inputs["Color"], value)

    b.column(10)
    bsdf = b.node("ShaderNodeBsdfPrincipled")
    b.set(bsdf.inputs["Base Color"], albedo)
    b.set(bsdf.inputs["Metallic"], metal)
    b.set(bsdf.inputs["Roughness"], roughness)
    b.set(bsdf.inputs["Normal"], normal)
    b.set(bsdf.inputs["Alpha"], i["Alpha"])
    b.set(bsdf.inputs["Specular IOR Level"], b.mul(0.5, spec_fade))
    b.set(bsdf.inputs["Transmission Weight"], i["Transmission"])
    b.set(bsdf.inputs["IOR"], 1.05)
    b.set(bsdf.inputs["Emission Color"], emission)
    b.set(bsdf.inputs["Emission Strength"], 1.0)

    b.column(11)
    fog = b.group(fog_group, Amount=i["Fog"])
    b.set(fog.inputs["Shader"], bsdf.outputs[0])

    # Levels of detail (dl_lod_range: the camera distances a fragment is drawn between, 0 0 for always), switched
    # per pixel by its distance from the camera, as the game switches each tree to its canopy blob.
    b.column(12)
    lod_near, lod_far, _ = b.separate(b.attribute("dl_lod_range").outputs["Vector"])
    # Detail Distance scales the distance the levels are picked by: 1 as the game, 0 always the most detailed.
    detail = b.value(1.0, "Detail Distance")
    detail.node.name = "dl_detail_distance"
    view_distance = b.mul(b.node("ShaderNodeCameraData").outputs["View Distance"], detail)
    in_range = b.mul(b.sub(1.0, b.math("LESS_THAN", view_distance, lod_near)), b.math("LESS_THAN", view_distance, lod_far))
    shown = b.math("MAXIMUM", b.math("LESS_THAN", lod_far, 1e-6), in_range)
    hidden = b.node("ShaderNodeBsdfTransparent")
    switch = b.node("ShaderNodeMixShader")
    b.set(switch.inputs[0], shown)
    b.set(switch.inputs[1], hidden.outputs[0])
    b.set(switch.inputs[2], fog.outputs[0])
    b.set(o["Shader"], switch.outputs[0])
    tree["dl_map"] = map_name
    return tree


def _barn_box(b: Builder, position, box: dict, unit_scale: float):
    """A parallel light_barn's reach at a world position (metres): inside its box of parallel rays (sheared along its
    depth), with its soft edges and near and far skirts. ``box``: 'origin' (game units), 'side', 'up', 'forward',
    'size' (half extents), 'range', 'shear', 'soft' (x, y), 'skirt', 'skirt_near'."""
    local = b.vsub(b.vscale(position, 1.0 / unit_scale), tuple(box["origin"]))
    rng = box["range"]
    z = b.dot(local, tuple(box["forward"]))
    ndc_x = b.div(b.sub(b.dot(local, tuple(box["side"])), b.mul(z, box["shear"][0] / rng)), box["size"][0])
    ndc_y = b.div(b.sub(b.dot(local, tuple(box["up"])), b.mul(z, box["shear"][1] / rng)), box["size"][1])
    ndc_z = b.div(z, rng)
    ax, ay = b.math("ABSOLUTE", ndc_x), b.math("ABSOLUTE", ndc_y)
    reach = b.mul(b.smoothstep(1.0, 1.0 - box["soft"][0], ax), b.smoothstep(1.0, 1.0 - box["soft"][1], ay))
    reach = b.mul(reach, b.mul(b.math("GREATER_THAN", ndc_z, 0.0), b.math("LESS_THAN", ndc_z, 1.0)))
    if box["skirt_near"] > 0:
        reach = b.mul(reach, b.smoothstep(0.0, 1.0, b.mul(ndc_z, 1.0 / box["skirt_near"])))
    if box["skirt"] > 0:
        reach = b.mul(reach, b.smoothstep(0.0, 1.0, b.mul(b.sub(1.0, ndc_z), 1.0 / box["skirt"])))
    return reach


def _direct_light(b: Builder, normal, to_camera, to_light, roughness, spec_color, n_dot_v, color):
    """DeadlockDirectLight: GGX with height correlated Smith visibility and Schlick fresnel, and the direct light's
    multiple scattering. Returns (diffuse irradiance, specular radiance)."""
    n_dot_l = b.saturate(b.dot(normal, to_light))
    half = b.normalize(b.vadd(to_light, to_camera))
    alpha = b.mul(roughness, roughness)
    alpha2 = b.mul(alpha, alpha)
    n_dot_h = b.dot(half, normal)
    d = b.add(b.mul(b.sub(b.mul(n_dot_h, alpha2), n_dot_h), n_dot_h), 1.0)
    k = b.sub(1.0, alpha)
    denominator = b.add(b.mul(n_dot_l, b.madd(n_dot_v, k, alpha)), b.mul(n_dot_v, b.madd(n_dot_l, k, alpha)))
    visibility = b.div(0.5, b.math("MAXIMUM", denominator, 1e-5))
    l_dot_h = b.saturate(b.dot(to_light, half))
    schlick = b.math("POWER", b.math("MAXIMUM", b.sub(1.0, l_dot_h), 0.0), 5.0)
    fresnel = b.vadd(spec_color, b.vscale(b.vsub((1, 1, 1), spec_color), schlick))
    roughness4 = b.mul(alpha2, 1.0)
    multi_scatter = b.vadd((1, 1, 1), b.vscale(spec_color, b.mul(b.mul(2.0, roughness4), n_dot_v)))
    # A point sun on a near mirror peaks at 1 / alpha^2 (~10^5 at the 0.05 floor), which supersampling catches and the
    # bloom would smear over the frame; the game shades each pixel once. Held to what a 1 degree sun could give.
    lobe = b.mul(b.mul(b.div(alpha2, b.math("MAXIMUM", b.mul(d, d), 1e-8)), visibility), n_dot_l)
    lobe = b.math("MINIMUM", lobe, 250.0)
    specular = b.vmul(b.vmul(b.vscale(fresnel, lobe), multi_scatter), color)
    return b.vscale(color, n_dot_l), specular


def build_unlit_group(map_name: str, fog_group) -> bpy.types.NodeTree:
    """Unlit surfaces (F_UNLIT, black.vfx): their colour as is, fogged."""
    tree, b, i, o = new_group(f"DL Unlit {map_name}", [("Color", "color", (1, 1, 1)), ("Alpha", "float", 1.0),
                                                        ("Fog", "float", 1.0)], [("Shader", "shader")])
    emission = b.node("ShaderNodeEmission")
    b.set(emission.inputs["Color"], i["Color"])
    transparent = b.node("ShaderNodeBsdfTransparent")
    mix = b.node("ShaderNodeMixShader")
    b.set(mix.inputs[0], i["Alpha"])
    b.set(mix.inputs[1], transparent.outputs[0])
    b.set(mix.inputs[2], emission.outputs[0])
    fog = b.group(fog_group, Amount=i["Fog"])
    b.set(fog.inputs["Shader"], mix.outputs[0])
    b.set(o["Shader"], fog.outputs[0])
    return tree


def build_additive_group(map_name: str, fog_group) -> bpy.types.NodeTree:
    """Additive surfaces: their colour times alpha added over what's behind, faded (not tinted) by the fog."""
    tree, b, i, o = new_group(f"DL Additive {map_name}", [("Color", "color", (1, 1, 1)), ("Alpha", "float", 1.0),
                                                          ("Fog", "float", 1.0)], [("Shader", "shader")])
    fog = b.group(fog_group, Amount=i["Fog"])
    attenuation = fog.outputs["Attenuation"]
    emission = b.node("ShaderNodeEmission")
    b.set(emission.inputs["Color"], i["Color"])
    b.set(emission.inputs["Strength"], b.mul(i["Alpha"], attenuation))
    transparent = b.node("ShaderNodeBsdfTransparent")
    add = b.node("ShaderNodeAddShader")
    b.set(add.inputs[0], transparent.outputs[0])
    b.set(add.inputs[1], emission.outputs[0])
    b.set(o["Shader"], add.outputs[0])
    return tree


class FogSettings:
    """The map's env_gradient_fog and env_cubemap_fog, in game units."""

    def __init__(self):
        self.gradient = None   # dict
        self.cubemap = None    # dict
        self.cubemap_image = None


def build_fog_group(map_name: str, settings: FogSettings, unit_scale: float) -> bpy.types.NodeTree:
    """``DL Fog <map>``: lerps a shader towards the fog colours by Valve's gradient and cubemap fog formulas, for
    camera rays (reflections see unfogged surfaces, as the game's cubemaps do). Attenuation is what's left of an
    additive surface."""
    tree, b, i, o = new_group(f"DL Fog {map_name}", [("Shader", "shader"), ("Amount", "float", 1.0)],
                              [("Shader", "shader"), ("Attenuation", "float")])
    tree["dl_map"] = map_name
    shader = i["Shader"]
    attenuation = b.value(1.0)

    b.column(0)
    camera = b.node("ShaderNodeCameraData")
    geometry = b.node("ShaderNodeNewGeometry")
    light_path = b.node("ShaderNodeLightPath")
    to_game = 1.0 / unit_scale
    distance = b.mul(camera.outputs["View Distance"], to_game)
    _, _, iz = b.separate(geometry.outputs["Incoming"])
    horizontal = b.mul(distance, b.math("SQRT", b.math("MAXIMUM", b.sub(1.0, b.mul(iz, iz)), 0.0)))
    _, _, pz = b.separate(geometry.outputs["Position"])
    height = b.mul(pz, to_game)
    gate = b.mul(light_path.outputs["Is Camera Ray"], i["Amount"])

    b.column(2)
    g = settings.gradient
    if g is not None:
        b.frame("Gradient fog")
        dist_scale = 1.0 / max(g["end"] - g["start"], 1e-3)
        dist_bias = -g["start"] * dist_scale
        height_delta = g["start_height"] - g["end_height"]
        height_scale = 1.0 / height_delta if abs(height_delta) > 1e-3 else 0.0
        height_bias = -g["end_height"] * height_scale if height_scale else 1.0
        d = b.math("POWER", b.saturate(b.madd(horizontal, b.value(dist_scale, "Distance Scale"), b.value(dist_bias, "Distance Bias"))),
                   b.value(g["exponent"], "Distance Exponent"))
        h = b.math("POWER", b.saturate(b.madd(height, b.value(height_scale, "Height Scale"), b.value(height_bias, "Height Bias"))),
                   b.value(g["vertical_exponent"], "Height Exponent"))
        amount = b.mul(b.mul(d, h), b.value(g["max_opacity"], "Max Opacity"))
        amount = b.mul(amount, gate)
        color = b.rgb(g["color"])
        color.node.label = "Gradient Fog Color"
        emission = b.node("ShaderNodeEmission")
        b.set(emission.inputs["Color"], color)
        mix = b.node("ShaderNodeMixShader")
        b.set(mix.inputs[0], amount)
        b.set(mix.inputs[1], shader)
        b.set(mix.inputs[2], emission.outputs[0])
        shader = mix.outputs[0]
        attenuation = b.mul(attenuation, b.sub(1.0, amount))
        b.frame(None)

    b.column(5)
    c = settings.cubemap
    if c is not None:
        b.frame("Cubemap fog")
        scale = 1.0 / max(c["end"] - c["start"], 1e-3)
        offset = -c["start"] * scale
        x = b.math("POWER", b.math("MAXIMUM", b.madd(distance, b.value(scale, "Distance Scale"), b.value(offset, "Distance Offset")), 0.0001),
                   b.value(c["exponent"], "Distance Exponent"))
        amount = b.saturate(x)
        if c["height_width"] > 0:
            hs = 1.0 / -c["height_width"]
            ho = 1.0 - c["height_start"] * hs
            y = b.math("POWER", b.math("MAXIMUM", b.madd(height, hs, ho), 0.0001), c["height_exponent"])
            amount = b.mul(amount, b.saturate(y))
        amount = b.mul(b.mul(amount, b.value(c["opacity"], "Max Opacity")), gate)
        if settings.cubemap_image is not None:
            env = b.node("ShaderNodeTexEnvironment")
            env.image = settings.cubemap_image
            b.set(env.inputs["Vector"], b.vscale(geometry.outputs["Incoming"], -1.0))
            color = b.vscale(env.outputs["Color"], b.value(c["exposure"], "Sky Exposure"))
        else:
            color = b.rgb((0.5, 0.5, 0.5))
        emission = b.node("ShaderNodeEmission")
        b.set(emission.inputs["Color"], color)
        mix = b.node("ShaderNodeMixShader")
        b.set(mix.inputs[0], amount)
        b.set(mix.inputs[1], shader)
        b.set(mix.inputs[2], emission.outputs[0])
        shader = mix.outputs[0]
        attenuation = b.mul(attenuation, b.sub(1.0, amount))
        b.frame(None)

    b.set(o["Shader"], shader)
    b.set(o["Attenuation"], attenuation)
    return tree


def gradient_fog_settings(entity, linear) -> dict:
    start = entity.get_float("fogstart")
    end = entity.get_float("fogend")
    color = entity.get_vector("fogcolor", 3, [255, 255, 255])
    strength = entity.get_float("fogstrength", 1.0)
    height_fog = entity.has("fogverticalexponent") or entity.has("fogstartheight")
    settings = {
        "start": start, "end": end,
        "exponent": entity.get_float("fogfalloffexponent", 1.0),
        "start_height": entity.get_float("fogstartheight"),
        "end_height": entity.get_float("fogendheight"),
        "vertical_exponent": entity.get_float("fogverticalexponent", 1.0),
        "max_opacity": entity.get_float("fogmaxopacity", 1.0),
        "color": tuple(linear(c / 255.0) * strength for c in color[:3]),
    }
    if not height_fog:
        settings.update(start_height=start, end_height=end, vertical_exponent=1.0)
    return settings


def cubemap_fog_settings(entity, exposure: float) -> dict:
    return {
        "start": entity.get_float("cubemapfogstartdistance"),
        "end": entity.get_float("cubemapfogenddistance", 4096.0),
        "exponent": entity.get_float("cubemapfogfalloffexponent", 1.0),
        "height_start": entity.get_float("cubemapfogheightstart"),
        "height_width": entity.get_float("cubemapfogheightwidth"),
        "height_exponent": entity.get_float("cubemapfogheightexponent", 1.0),
        "opacity": entity.get_float("cubemapfogmaxopacity", 1.0),
        "lod_bias": entity.get_float("cubemapfoglodbiase"),
        "exposure": exposure,
    }


def sky_world(name: str, image, exposure: float, tint=(1.0, 1.0, 1.0)) -> bpy.types.World:
    """A world showing the sky to the camera and in reflections, lighting nothing: the map's baked lighting already
    holds the sky's light."""
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    b = Builder(tree)
    b.column(0)
    light_path = b.node("ShaderNodeLightPath")
    env = b.node("ShaderNodeTexEnvironment")
    env.image = image
    b.column(1)
    visible = b.math("MAXIMUM", light_path.outputs["Is Camera Ray"], light_path.outputs["Is Glossy Ray"])
    visible = b.math("MAXIMUM", visible, light_path.outputs["Is Transmission Ray"])
    background = b.node("ShaderNodeBackground")
    b.set(background.inputs["Color"], b.vmul(env.outputs["Color"], tuple(t * exposure for t in tint)))
    b.set(background.inputs["Strength"], visible)
    b.column(2)
    output = b.node("ShaderNodeOutputWorld")
    output.hide = False
    b.set(output.inputs["Surface"], background.outputs[0])
    world["dl_sky_exposure"] = exposure
    return world


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def color_temperature(kelvin: float):
    """Blackbody colour (0-1 gamma space), ValveResourceFormat's approximation."""
    t = kelvin / 100.0
    red = 329.69873 * (t - 60.0) ** -0.13320476 if t > 66 else 255.0
    green = 288.12216 * (t - 60.0) ** -0.075514846 if t > 66 else 99.4708 * math.log(t) - 161.11957
    blue = (138.51773 * math.log(t - 10.0) - 305.0448 if t > 10 else 0.0) if t < 66 else 255.0

    def channel(v):
        return 0.0 if v != v else min(max(v, 0.0), 255.0) / 255.0

    return channel(red), channel(green), channel(blue)


def ambient_cube_group() -> bpy.types.NodeTree:
    """``DL Ambient Cube``: the object's light probe ambient cube (its dl_amb_* custom properties) along a direction,
    each face weighed by the direction's squared component, picked by its sign."""
    name = "DL Ambient Cube"
    group = bpy.data.node_groups.get(name)
    if group is not None and group.get("dl_version") == 1:
        return group
    tree, b, i, o = new_group(name, [("Direction", "vector", (0, 0, 1))], [("Color", "color")])
    b.set(o["Color"], _probe_irradiance(b, b.normalize(i["Direction"])))
    tree["dl_version"] = 1
    return tree


# NPR settings: engine-wide in the game; the values of materials/dev/npr_dummy.vmat (as the mount has them).
NPR = {
    "wrap": 0.8, "step_sharpness": 0.9, "pbr_blend": 0.25, "bounce": 1.0, "camera_weight": 0.35,
    "exposure_control": 1.0, "targets": (1.0, 0.5, 0.1), "rim_strength": 2.0, "rim_wrap": 0.7,
    "rim_falloff": 0.7, "rim_up_ramp": (0.2, 0.2),
}


def _gain(b, x, k):
    """NprGain: a symmetric S-curve about 0.5, k its hardness."""
    upper = b.math("GREATER_THAN", x, 0.5)
    folded = b.lerp(x, b.sub(1.0, x), upper)
    curved = b.mul(0.5 ** (1.0 - k), b.math("POWER", b.math("MAXIMUM", folded, 0.0), k))
    return b.lerp(curved, b.sub(1.0, curved), upper)


def _xyY(b, color):
    x = b.dot(color, (0.4124, 0.3576, 0.1805))
    y = b.dot(color, (0.2126, 0.7152, 0.0722))
    z = b.dot(color, (0.0193, 0.1192, 0.9505))
    total = b.math("MAXIMUM", b.add(b.add(x, y), z), 1e-8)
    return b.div(x, total), b.div(y, total), y


def _exposure_control(b, color, average, target):
    """NprExposureControl: the colour's chroma (towards the average's as it darkens) at a fixed luminance."""
    cx, cy, cl = _xyY(b, color)
    ax, ay, al = average
    t = b.saturate(b.div(cl, b.math("MAXIMUM", al, 1e-8)))
    chroma_x = b.lerp(ax, cx, t)
    chroma_y = b.math("MAXIMUM", b.lerp(ay, cy, t), 1e-6)
    s = b.div(target, chroma_y)
    X = b.mul(s, chroma_x)
    Z = b.mul(s, b.sub(b.sub(1.0, chroma_x), chroma_y))
    xyz = b.combine(X, target, Z)
    r = b.dot(xyz, (3.2406, -1.5372, -0.4986))
    g = b.dot(xyz, (-0.9689, 1.8758, 0.0415))
    bl = b.dot(xyz, (0.0557, -0.2040, 1.0570))
    return b.combine(r, g, bl)


def build_npr_group(map_name: str, fog_group, exposure: float) -> bpy.types.NodeTree:
    """``DL NPR Lighting <map>``: pbr.vfx's NPR path (heroes, and props like the hideout's shrines and buttons) on
    the map's lighting. The bounce light is the object's ambient cube looked up along the game's banded normal, held
    to fixed brightnesses after exposure (exposure control), the unlit side picking up the transmissive colour from
    above, and the masked rim light. Blender's lights give the direct light and reflections."""
    tree, b, i, o = new_group(f"DL NPR Lighting {map_name}", [
        ("Albedo", "color", (0.8, 0.8, 0.8)), ("Metalness", "float", 0.0), ("Roughness", "float", 0.5),
        ("Normal", "vector", (0, 0, 0)), ("Emission", "color", (0, 0, 0)), ("Alpha", "float", 1.0),
        ("Ambient Occlusion", "float", 1.0), ("Rim Mask", "float", 1.0), ("Transmissive", "color", (0, 0, 0)),
        ("Fog", "float", 1.0),
    ], [("Shader", "shader")])
    cube = ambient_cube_group()
    k = 1.0 / (1.0 - NPR["step_sharpness"])

    b.column(0)
    geometry = b.node("ShaderNodeNewGeometry")
    has_normal = b.math("GREATER_THAN", b.vmath("LENGTH", i["Normal"]), 0.5)
    normal = b.normalize(b.vlerp(geometry.outputs["Normal"], i["Normal"], has_normal))
    to_camera = geometry.outputs["Incoming"]
    ao = i["Ambient Occlusion"]

    # The direction the bounce light is banded against: up, leaning towards the camera.
    b.column(1)
    tx, ty, _ = b.separate(to_camera)
    towards = b.normalize(b.combine(tx, ty, 1e-8))
    primary = b.normalize(b.vlerp((0, 0, 1), towards, NPR["camera_weight"]))
    half_lambert = b.madd(b.dot(primary, normal), 0.5, 0.5)
    stepped = b.saturate(b.sub(b.mul(b.mul(half_lambert, ao), 2.0), 0.5))
    band = b.math("FRACT", stepped)
    banded = b.add(b.sub(stepped, band), _gain(b, band, k))
    lit = b.mul(b.add(banded, 0.5), 0.5)

    b.column(4)
    horizontal = b.normalize(b.vadd(b.combine(tx, ty, 0.0), (0.0001, 0, 0)))
    low = b.vlerp((0, 0, -1), horizontal, b.add(banded, 0.5))
    high = b.vlerp(horizontal, (0, 0, 1), b.sub(banded, 0.5))
    bounce_normal = b.normalize(b.vlerp(high, low, b.math("LESS_THAN", lit, 0.5)))

    b.column(5)
    ambient = b.group(cube, Direction=bounce_normal).outputs[0]
    above = b.group(cube, Direction=(0, 0, 1)).outputs[0]
    if NPR["exposure_control"] > 0 and exposure > 0:
        b.column(6)
        b.frame("Exposure control")
        faces = [b.attribute(f"dl_amb_{f}", "OBJECT").outputs["Color"] for f in ("px", "py", "pz", "nx", "ny", "nz")]
        average = faces[0]
        for f in faces[1:]:
            average = b.vadd(average, f)
        average = b.vscale(average, 1.0 / 6.0)
        avg = _xyY(b, average)
        _, _, bz = b.separate(bounce_normal)
        bn2 = b.vmul(bounce_normal, bounce_normal)
        t = NPR["targets"]
        targets = b.combine(t[1], t[1], b.select(b.math("GREATER_THAN", bz, 0.0), t[0], t[2]))
        ambient = _exposure_control(b, ambient, avg, b.div(b.dot(targets, bn2), exposure))
        above = _exposure_control(b, above, avg, t[0] / exposure)
        b.frame(None)

    b.column(10)
    bounce = b.vadd(ambient, b.vscale(b.vmul(above, i["Transmissive"]), b.sub(1.0, lit)))
    albedo = i["Albedo"]
    # Jimenez's multi bounce fit on the occlusion, blended in at the NPR PBR blend.
    a = b.vsub(b.vscale(albedo, 2.0404), (0.3324, 0.3324, 0.3324))
    bb = b.vsub(b.vscale(albedo, 4.7951), (0.6417, 0.6417, 0.6417))
    c = b.vadd(b.vscale(albedo, 2.7552), (0.6903, 0.6903, 0.6903))
    multi = b.vadd(b.vscale(b.vsub(b.vscale(a, ao), bb), b.mul(ao, ao)), b.vscale(c, ao))
    bounce_weight = b.vlerp((1, 1, 1), multi, NPR["pbr_blend"])
    metal = b.saturate(i["Metalness"])
    diffuse = b.vmul(b.vmul(bounce, bounce_weight), b.vscale(albedo, b.sub(1.0, metal)))

    # The rim light, only on parts facing up past the ramp.
    b.column(13)
    wrap = NPR["rim_wrap"]
    from_camera_dot = b.mul(b.dot(normal, to_camera), -1.0)
    rim = b.math("POWER", b.saturate(b.div(b.add(from_camera_dot, wrap), (1 + wrap) ** 2)), NPR["rim_falloff"])
    ramp = NPR["rim_up_ramp"]
    _, _, nz = b.separate(normal)
    span = ramp[1] - ramp[0]
    up = b.saturate(b.div(b.sub(nz, ramp[0]), span)) if span > 0 else b.math("GREATER_THAN", nz, ramp[0])
    rim = b.mul(b.mul(b.mul(rim, up), NPR["rim_strength"]), b.mul(ao, i["Rim Mask"]))
    emission = b.vadd(b.vadd(diffuse, b.vscale(bounce, rim)), i["Emission"])

    b.column(16)
    bsdf = b.node("ShaderNodeBsdfPrincipled")
    b.set(bsdf.inputs["Base Color"], albedo)
    b.set(bsdf.inputs["Metallic"], metal)
    b.set(bsdf.inputs["Roughness"], b.math("MAXIMUM", i["Roughness"], 0.05))
    b.set(bsdf.inputs["Normal"], normal)
    b.set(bsdf.inputs["Alpha"], i["Alpha"])
    b.set(bsdf.inputs["Emission Color"], emission)
    b.set(bsdf.inputs["Emission Strength"], 1.0)
    b.column(18)
    fog = b.group(fog_group, Amount=i["Fog"])
    b.set(fog.inputs["Shader"], bsdf.outputs[0])
    b.set(o["Shader"], fog.outputs[0])
    tree["dl_map"] = map_name
    return tree
