"""Render settings and the game's post processing (deadlock post_process: exposure, bloom and its filmic tonemapping
curve from the map's .vpost) as compositor nodes."""
import math

import bpy

from ..core.resource import ResourceFile

# The game blurs its bloom at a quarter of the screen, halved four times with a four tap Gaussian at each level:
# level k's blur, in pixels of a 1080 line frame (a texel of 2^(k+2) pixels, the filter's radius about 4.5 texels).
BLOOM_LEVEL_PIXELS = (36.0, 72.0, 144.0, 288.0)
BLOOM_CLAMP = 16.0


def setup_render(scene, unit_scale: float):
    """Settings that keep Blender from relighting what the game baked: no diffuse bounces (the lightmaps and probes are
    the indirect light), glossy bounces for reflections, Standard view (the tonemap is in the compositor)."""
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    cycles = getattr(scene, "cycles", None)
    if cycles is not None:
        cycles.max_bounces = 6
        cycles.diffuse_bounces = 0
        cycles.glossy_bounces = 2
        cycles.transmission_bounces = 6
        cycles.transparent_max_bounces = 32
        cycles.volume_bounces = 0
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False
        cycles.sample_clamp_indirect = 10.0

    eevee = scene.eevee
    for attr, value in (("use_raytracing", True), ("use_fast_gi", False), ("ray_tracing_method", "SCREEN"),
                        ("use_shadows", True), ("taa_render_samples", 64), ("clamp_surface_indirect", 10.0),
                        ("shadow_pool_size", "2048"), ("use_volumetric_shadows", True)):
        try:
            setattr(eevee, attr, value)
        except (AttributeError, TypeError, ValueError):
            pass
    scene.render.film_transparent = False

    # Clip far enough for the whole map.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.clip_start = 2 * unit_scale
                        space.clip_end = 60000 * unit_scale


def _compositor_tree(scene):
    """The scene's compositor node tree (Blender 5 makes it a node group the scene points at)."""
    if hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new(f"{scene.name} Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree, True
    scene.use_nodes = True
    return scene.node_tree, False


def read_vpost(content, path: str) -> dict:
    data = content.try_read((path or "").replace("\\", "/") + "_c") if path else None
    if data is None:
        return {}
    return ResourceFile(data).keyvalues("DATA") or {}


def exposure_scale(content, entity) -> float:
    """The camera exposure times the .vpost's exposure bias, as the game multiplies the scene by before tonemapping."""
    exposure = 0.0
    bias = 0.8992
    if entity is not None:
        if entity.get_bool("enableexposure", True):
            lo = entity.get_float("minlogexposure")
            hi = max(lo, entity.get_float("maxlogexposure"))
            exposure = min(max(0.0, lo), hi)
        exposure += entity.get_float("exposurecompensation")
        params = read_vpost(content, entity.get_str("postprocessing"))
        if params.get("m_bHasTonemapParams"):
            bias = float((params.get("m_toneMapParams") or {}).get("m_flExposureBias", bias))
    return 2.0 ** (exposure + bias)


def setup_post_processing(scene, content, entity):
    """Exposure, bloom and the filmic curve of the map's master post_processing_volume."""
    exposure = 0.0
    params = {}
    if entity is not None:
        if entity.get_bool("enableexposure", True):
            lo = entity.get_float("minlogexposure")
            hi = max(lo, entity.get_float("maxlogexposure"))
            exposure = min(max(0.0, lo), hi)
        exposure += entity.get_float("exposurecompensation")
        params = read_vpost(content, entity.get_str("postprocessing"))

    tonemap = params.get("m_toneMapParams") if params.get("m_bHasTonemapParams") else None
    curve = {
        "bias": 0.8992, "A": 0.15, "B": 0.5, "C": 0.1, "D": 0.2, "E": 0.02, "F": 0.3, "W": 16.0,
    }
    if tonemap:
        curve.update(bias=float(tonemap.get("m_flExposureBias", curve["bias"])),
                     A=float(tonemap.get("m_flShoulderStrength", curve["A"])),
                     B=float(tonemap.get("m_flLinearStrength", curve["B"])),
                     C=float(tonemap.get("m_flLinearAngle", curve["C"])),
                     D=float(tonemap.get("m_flToeStrength", curve["D"])),
                     E=float(tonemap.get("m_flToeNum", curve["E"])),
                     F=float(tonemap.get("m_flToeDenom", curve["F"])),
                     W=float(tonemap.get("m_flWhitePoint", curve["W"])))
    bloom = params.get("m_bloomParams") if params.get("m_bHasBloomParams") else None

    tree, _is_group = _compositor_tree(scene)
    tree.nodes.clear()
    nodes, links = tree.nodes, tree.links

    def node(kind, x, y, **props):
        n = nodes.new(kind)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math_node(op, a, b=None, x=0, y=0, clamp=False):
        n = node("ShaderNodeMath", x, y, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    if _is_group:
        # Blender 5: the render comes from a Render Layers node, the result leaves through the group's output.
        tree.interface.clear()
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    source = node("CompositorNodeRLayers", -1400, 0)
    image = source.outputs["Image"]

    scale = 2.0 ** (exposure + curve["bias"])
    color = image

    if bloom and float(bloom.get("m_flBloomStrength", 0.0)) > 0:
        color = _bloom(scene, tree, node, math_node, color, bloom)

    separate = node("CompositorNodeSeparateColor", -900, 0)
    links.new(color, separate.inputs[0])
    combine = node("CompositorNodeCombineColor", 600, 0)

    A, B, C, D, E, F, W = (curve[k] for k in "ABCDEFW")

    def curve_value(x):
        return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F

    normalise = 1.0 / curve_value(W)
    for channel in range(3):
        y = -300 + channel * -0  # stacked per channel
        base_y = 200 - channel * 220
        c = separate.outputs[channel]
        x = math_node("MULTIPLY", c, scale * 2.8, -700, base_y)
        x = math_node("MINIMUM", x, W, -550, base_y)
        x = math_node("MAXIMUM", x, 0.0, -450, base_y)
        a = math_node("MULTIPLY", x, A, -350, base_y)
        num = math_node("ADD", math_node("MULTIPLY", x, math_node("ADD", a, C * B, -250, base_y), -150, base_y), D * E, -50, base_y)
        den = math_node("ADD", math_node("MULTIPLY", x, math_node("ADD", a, B, -250, base_y - 60), -150, base_y - 60), D * F, -50, base_y - 60)
        mapped = math_node("SUBTRACT", math_node("DIVIDE", num, den, 50, base_y), E / F, 150, base_y)
        mapped = math_node("MULTIPLY", mapped, normalise, 300, base_y, clamp=True)
        links.new(mapped, combine.inputs[channel])

    if _is_group:
        out = node("NodeGroupOutput", 900, 0)
        links.new(combine.outputs[0], out.inputs["Image"])
    else:
        composite = node("CompositorNodeComposite", 900, 0)
        links.new(combine.outputs[0], composite.inputs[0])
        viewer = node("CompositorNodeViewer", 900, -200)
        links.new(combine.outputs[0], viewer.inputs[0])

    scene["dl_exposure"] = exposure
    scene["dl_tonemap"] = [A, B, C, D, E, F, W, curve["bias"]]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        try:
                            space.shading.use_compositor = "ALWAYS"
                        except (AttributeError, TypeError):
                            pass


def _set_input(node, names, value, attribute=None):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            try:
                socket.default_value = value
                return
            except (TypeError, AttributeError):
                pass
    if attribute and hasattr(node, attribute):
        try:
            setattr(node, attribute, value if attribute != "size" else 6)
        except (TypeError, AttributeError):
            pass


def _bloom(scene, tree, node, math_node, image, bloom):
    """bloom_downsample and gaussian_blur_bloom: the scene weighed by how far its luminance passes the threshold, blurred
    at four widening levels summed by the .vpost's blur weights, added at the bloom strength (before exposure)."""
    links = tree.links
    threshold = float(bloom.get("m_flBloomThreshold", 1.0))
    width = max(float(bloom.get("m_flBloomThresholdWidth", 0.999)), 0.001)
    strength = float(bloom.get("m_flBloomStrength", 0.0))
    weights = [float(w) for w in bloom.get("m_flBlurWeight") or [0.0, 0.07, 0.18, 0.3, 0.45]][1:5]

    separate = node("CompositorNodeSeparateColor", -1300, -400)
    links.new(image, separate.inputs[0])
    lum = math_node("ADD", math_node("ADD", math_node("MULTIPLY", separate.outputs[0], 0.3, -1200, -400),
                                     math_node("MULTIPLY", separate.outputs[1], 0.59, -1200, -450), -1100, -420),
                    math_node("MULTIPLY", separate.outputs[2], 0.11, -1200, -500), -1000, -420)
    passing = math_node("MULTIPLY_ADD", lum, 1.0 / width, -1000, -500)
    passing.node.inputs[2].default_value = -threshold / width
    passing.node.use_clamp = True
    scaled = node("ShaderNodeVectorMath", -900, -400, operation="SCALE")
    links.new(image, scaled.inputs[0])
    links.new(passing, scaled.inputs["Scale"])
    # Single bright pixels (sun glints) would bloom over the whole frame; the game's quarter resolution source averages
    # them away, which a clamp stands in for.
    bright = node("ShaderNodeVectorMath", -850, -400, operation="MINIMUM")
    links.new(scaled.outputs[0], bright.inputs[0])
    bright.inputs[1].default_value = (BLOOM_CLAMP,) * 3

    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    total = None
    for k, (pixels, weight) in enumerate(zip(BLOOM_LEVEL_PIXELS, weights)):
        if weight <= 0:
            continue
        blur = node("CompositorNodeBlur", -750, -300 - k * 150)
        blur.inputs["Size"].default_value = (pixels * height / 1080.0,) * 2
        blur.inputs["Type"].default_value = "Gaussian"
        links.new(bright.outputs[0], blur.inputs["Image"])
        level = node("ShaderNodeVectorMath", -550, -300 - k * 150, operation="SCALE")
        links.new(blur.outputs[0], level.inputs[0])
        level.inputs["Scale"].default_value = weight * strength
        if total is None:
            total = level.outputs[0]
        else:
            add = node("ShaderNodeVectorMath", -400, -300 - k * 150, operation="ADD")
            links.new(total, add.inputs[0])
            links.new(level.outputs[0], add.inputs[1])
            total = add.outputs[0]
    if total is None:
        return image
    out = node("ShaderNodeVectorMath", -250, -200, operation="ADD")
    links.new(image, out.inputs[0])
    links.new(total, out.inputs[1])
    out.label = "Deadlock bloom"
    scene["dl_bloom_height"] = height
    return out.outputs[0]
