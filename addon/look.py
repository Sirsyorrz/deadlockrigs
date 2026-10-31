# SPDX-License-Identifier: GPL-3.0-or-later
"""Game Look: the game's post-processing as the scene's compositor, so renders get Deadlock's brightness, contrast,
bloom and vignette.

In the order of Deadlock's post_process.vfx (citadel, decompiled): the scene exposed by the map's exposure times
2^ExposureBias, plus bloom, then the vignette, then the 2.8 and the Uncharted 2 style filmic curve normalized to a
white point. Blender's Standard view transform then does the sRGB encoding the game does; the colour correction LUT
that follows is the identity in basepostprocess_deadlock.vpost, which every map uses, so it is left out.

Bloom (downsample_bloomthreshold.vfx and gaussian_bloom_blur.vfx, core): the unexposed scene where its exposed
luminance passes the threshold, times the bloom strength, blurred down a chain of mips from quarter resolution and
summed with the blur weights, added after exposure without being exposed itself. Each mip is a Gaussian here with
the chain's accumulated width. The chain is in pixels, so the game's bloom is relatively narrower at higher
resolutions; the widths are for BLOOM_REFERENCE_WIDTH.

Vignette: from the shader, the radius is where it starts and radius + feather where it is full, over a superellipse in
[-1, 1] screen space; strength below 0 darkens toward 1 + strength. The roundness to superellipse exponent mapping is
not in the shader (it is set on the CPU) and is a guess: 2 / roundness, an ellipse at 1, a rectangle toward 0.
"""
import bpy

GROUP = 'Deadlock Game Look'
SAVED = 'dlr_look_saved'
# basepostprocess_deadlock.vpost m_toneMapParams, game version 6821
DEFAULT_TONEMAP = {'exposure_bias': 0.8992, 'shoulder_strength': 0.3538, 'linear_strength': 0.3258,
                   'linear_angle': 0.2528, 'toe_strength': 0.6966, 'toe_num': 0.0, 'toe_denom': 0.7819,
                   'white_point': 3.9996}
SHADER_EXPOSURE = 2.8
# basepostprocess_deadlock.vpost m_bloomParams and m_vignetteParams, game version 6835
BLOOM = {'strength': 0.5, 'threshold': 1.0, 'threshold_width': 0.999,
         'blur_weights': (0.0, 0.070259, 0.181076, 0.300013, 0.448652)}
VIGNETTE = {'strength': -0.5982, 'center': (0.0, 0.0), 'radius': 0.4386, 'roundness': 0.2134, 'feather': 0.6098}
BLOOM_LUMA = (0.3, 0.59, 0.11)
# the game's screenshots are 4K
BLOOM_REFERENCE_WIDTH = 3840.0
# the blur pass's 4 bilinear taps, a Gaussian of this variance in texels of its mip, and the downsamples between mips
BLUR_VARIANCE = 1.667


def bloom_sigmas():
    """Each bloom mip's accumulated Gaussian width, in fractions of the image width."""
    out, variance = [], 0.0
    for i in range(len(BLOOM['blur_weights'])):
        texel = 4.0 * 2 ** i
        variance += BLUR_VARIANCE * texel ** 2 + 0.5 * (texel / 2) ** 2
        out.append(variance ** 0.5 / BLOOM_REFERENCE_WIDTH)
    return out


def tonemap_params(scene):
    for c in scene.collection.children_recursive:
        params = c.get('dlr_tonemap')
        if params:
            p = {k: float(params.get(k, v)) for k, v in DEFAULT_TONEMAP.items()}
            p['exposure'] = float(params.get('exposure', 0.0))
            return p
    return dict(DEFAULT_TONEMAP, exposure=0.0)


def curve(p, x):
    a, b, c = p['shoulder_strength'], p['linear_strength'], p['linear_angle']
    d, e, f = p['toe_strength'], p['toe_num'], p['toe_denom']
    return (x * (a * x + b * c) + d * e) / (x * (a * x + b) + d * f) - e / f


def build_group(p):
    tree = bpy.data.node_groups.get(GROUP) or bpy.data.node_groups.new(GROUP, 'CompositorNodeTree')
    tree.nodes.clear()
    if not tree.interface.items_tree:
        tree.interface.new_socket('Image', in_out='INPUT', socket_type='NodeSocketColor')
        tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    nodes, links = tree.nodes, tree.links
    inp, out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
    inp.location, out.location = (-1400, 0), (900, 0)

    def put(socket, value):
        if hasattr(value, 'node'):
            links.new(value, socket)
        elif socket.type == 'VECTOR' and not isinstance(value, (tuple, list)):
            socket.default_value = (value,) * len(socket.default_value)
        else:
            socket.default_value = value

    def vec(op, a, b=None, x=0, y=0):
        n = nodes.new('ShaderNodeVectorMath')
        n.operation = op
        n.location = (x, y)
        put(n.inputs[0], a)
        if b is not None:
            put(n.inputs[3] if op == 'SCALE' else n.inputs[1], b)
        return n.outputs['Value' if op in ('DOT_PRODUCT', 'LENGTH') else 'Vector']

    def math(op, a, b=None, x=0, y=0, clamp=False):
        n = nodes.new('ShaderNodeMath')
        n.operation = op
        n.use_clamp = clamp
        n.location = (x, y)
        put(n.inputs[0], a)
        if b is not None:
            put(n.inputs[1], b)
        return n.outputs[0]

    def mad(a, b, c, x=0, y=0, clamp=False):
        out = math('MULTIPLY_ADD', a, b, x, y, clamp)
        put(out.node.inputs[2], c)
        return out

    def value(name, v, x, y):
        n = nodes.new('ShaderNodeValue')
        n.name = n.label = name
        n.outputs[0].default_value = v
        n.location = (x, y)
        return n.outputs[0]

    def split(v, x, y):
        n = nodes.new('ShaderNodeSeparateXYZ')
        n.location = (x, y)
        links.new(v, n.inputs[0])
        return n.outputs

    white = 2.0 ** p['white_point']
    a, b, c = p['shoulder_strength'], p['linear_strength'], p['linear_angle']
    d, e, f = p['toe_strength'], p['toe_num'], p['toe_denom']
    scene = inp.outputs[0]
    exposure = value('Exposure', 2.0 ** p['exposure_bias'], -2600, -200)
    exposed = vec('SCALE', scene, exposure, -2400)

    # threshold on the exposure without the bias, as the game's tonemap scalar
    luma = vec('DOT_PRODUCT', scene, BLOOM_LUMA, -2600, -600)
    width = BLOOM['threshold_width']
    weight = mad(luma, 2.0 ** p['exposure'] / width, -BLOOM['threshold'] / width, -2400, -600, clamp=True)
    source = vec('SCALE', scene, math('MULTIPLY', weight, BLOOM['strength'], -2200, -600), -2000, -600)
    bloom_on = value('Bloom', 1.0, -2000, -900)
    bloom = None
    for i, (w, sigma) in enumerate(zip(BLOOM['blur_weights'], bloom_sigmas())):
        if w <= 0:
            continue
        size = nodes.new('CompositorNodeRelativeToPixel')
        size.data_type, size.reference_dimension = 'VECTOR', 'X'
        # its float and vector sockets share a name
        next(i for i in size.inputs if i.type == 'VECTOR').default_value = (3 * sigma, 3 * sigma)
        links.new(scene, size.inputs['Image'])
        size.location = (-1800, -500 - 200 * i)
        blur = nodes.new('CompositorNodeBlur')
        blur.inputs['Type'].default_value = 'Gaussian'
        blur.location = (-1600, -500 - 200 * i)
        links.new(source, blur.inputs['Image'])
        links.new(next(o for o in size.outputs if o.type == 'VECTOR'), blur.inputs['Size'])
        part = vec('SCALE', blur.outputs[0], w, -1400, -500 - 200 * i)
        bloom = part if bloom is None else vec('ADD', bloom, part, -1200, -500 - 200 * i)
    lit = vec('ADD', exposed, vec('SCALE', bloom, bloom_on, -1000, -600), -800)

    v = VIGNETTE
    power = 2 / max(v['roundness'], 0.01)
    coords = nodes.new('CompositorNodeImageCoordinates')
    coords.location = (-2600, -1400)
    links.new(scene, coords.inputs[0])
    centred = vec('SUBTRACT', vec('SCALE', coords.outputs['Normalized'], 2.0, -2400, -1400),
                  (1.0 + v['center'][0], 1.0 + v['center'][1], 0.0), -2200, -1400)
    dx, dy, _ = split(vec('ABSOLUTE', centred, x=-2000, y=-1400), -1800, -1400)
    radius = math('POWER', math('ADD', math('POWER', dx, power, -1600, -1300), math('POWER', dy, power, -1600, -1500),
                                -1400, -1400), 1 / power, -1200, -1400)
    edge = math('DIVIDE', math('SUBTRACT', radius, v['radius'], -1000, -1400), v['feather'], -800, -1400, clamp=True)
    smooth = math('MULTIPLY', math('MULTIPLY', edge, edge, -600, -1300), mad(edge, -2.0, 3.0, -600, -1500), -400, -1400)
    s = v['strength']
    vignette_on = value('Vignette', 1.0, -400, -1600)
    amount = math('MULTIPLY', smooth, vignette_on, -200, -1400)
    darken = mad(amount, -abs(s), 1.0, 0, -1400)
    vignetted = vec('SCALE', lit, darken, -600)
    if s > 0:
        vignetted = vec('ADD', vignetted, vec('SCALE', (1.0, 1.0, 1.0), math('MULTIPLY', amount, s, 0, -1600),
                                              -400, -200), -400)

    x = vec('MINIMUM', vec('SCALE', vignetted, SHADER_EXPOSURE, -400), white, -1000)
    ax = vec('SCALE', x, a, -800)
    num = vec('ADD', vec('MULTIPLY', x, vec('ADD', ax, b * c, -600), -400), d * e, -200)
    den = vec('ADD', vec('MULTIPLY', x, vec('ADD', ax, b, -600), -400), d * f, -200)
    toned = vec('SUBTRACT', vec('DIVIDE', num, den, 0), e / f, 200)
    result = vec('SCALE', toned, 1.0 / curve(p, white), 400)

    # the vector maths drops alpha; put it back for transparent film
    alpha = nodes.new('CompositorNodeSeparateColor')
    alpha.location = (400, -300)
    links.new(inp.outputs[0], alpha.inputs[0])
    set_alpha = nodes.new('CompositorNodeSetAlpha')
    set_alpha.inputs['Type'].default_value = 'Replace Alpha'
    set_alpha.location = (650, 0)
    links.new(result, set_alpha.inputs['Image'])
    links.new(alpha.outputs['Alpha'], set_alpha.inputs['Alpha'])
    links.new(set_alpha.outputs[0], out.inputs[0])
    return tree


def scene_tree(scene):
    """The scene's compositor, made if it has none: Render Layers straight to the output."""
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new(f'{scene.name} Compositor', 'CompositorNodeTree')
        tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        layers = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('NodeGroupOutput')
        layers.location, out.location = (-400, 0), (400, 0)
        tree.links.new(layers.outputs['Image'], out.inputs[0])
        scene.compositing_node_group = tree
    return tree


def look_node(tree):
    return next((n for n in tree.nodes if n.type == 'GROUP' and n.node_tree and n.node_tree.name == GROUP), None)


def enable(scene):
    group = build_group(tonemap_params(scene))
    tree = scene_tree(scene)
    if look_node(tree) is None:
        out = next((n for n in tree.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output), None) or \
            next(n for n in tree.nodes if n.type == 'GROUP_OUTPUT')
        node = tree.nodes.new('CompositorNodeGroup')
        node.node_tree = group
        node.location = out.location.x, out.location.y - 150
        out.location.x += 250
        target = out.inputs[0]
        if target.is_linked:
            tree.links.new(target.links[0].from_socket, node.inputs[0])
        tree.links.new(node.outputs[0], target)
    vs = scene.view_settings
    if SAVED not in scene:
        scene[SAVED] = {'view_transform': vs.view_transform, 'look': vs.look, 'exposure': vs.exposure,
                        'gamma': vs.gamma}
    vs.view_transform, vs.look, vs.exposure, vs.gamma = 'Standard', 'None', 0.0, 1.0


def disable(scene):
    tree = scene.compositing_node_group
    node = look_node(tree) if tree else None
    if node is not None:
        src = node.inputs[0].links[0].from_socket if node.inputs[0].is_linked else None
        targets = [link.to_socket for link in node.outputs[0].links]
        tree.nodes.remove(node)
        for t in targets:
            if src is not None:
                tree.links.new(src, t)
    saved = scene.get(SAVED)
    if saved:
        vs = scene.view_settings
        vs.view_transform, vs.look = saved['view_transform'], saved['look']
        vs.exposure, vs.gamma = saved['exposure'], saved['gamma']
        del scene[SAVED]


def is_enabled(scene):
    tree = scene.compositing_node_group
    return bool(tree and look_node(tree))


def show_in_viewport(context):
    """The viewport only runs the compositor in rendered and material preview shading when asked to."""
    for area in context.screen.areas if context.screen else ():
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.use_compositor = 'ALWAYS'


class DLR_OT_game_look(bpy.types.Operator):
    bl_idname = 'dlr.game_look'
    bl_label = 'Game Look'
    bl_description = ("Toggle the game's tonemapping on the scene (as its compositor, with the Standard view "
                      'transform), so renders get the brightness and contrast of Deadlock')
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        if is_enabled(sc):
            disable(sc)
        else:
            enable(sc)
            show_in_viewport(context)
        return {'FINISHED'}


def draw(layout, context):
    on = is_enabled(context.scene)
    layout.operator('dlr.game_look', icon='CHECKBOX_HLT' if on else 'CHECKBOX_DEHLT', depress=on)
    group = bpy.data.node_groups.get(GROUP)
    if on and group:
        col = layout.column(align=True)
        for name in ('Bloom', 'Vignette'):
            if name in group.nodes:
                col.prop(group.nodes[name].outputs[0], 'default_value', text=name)


def register():
    bpy.utils.register_class(DLR_OT_game_look)


def unregister():
    bpy.utils.unregister_class(DLR_OT_game_look)
