# SPDX-License-Identifier: GPL-3.0-or-later
"""Game Look: the game's tonemapping as the scene's compositor, so renders get Deadlock's brightness and contrast.

Deadlock exposes the scene by 2^ExposureBias and the 2.8 its post-processing shader applies, then runs it through an
Uncharted 2 style filmic curve normalized to a white point (ValveResourceFormat, Renderer/Shaders/
post_processing.frag.slang, MIT licence). Blender's Standard view transform then does the sRGB encoding the game does.
The curve's values come from the map's post-processing (stored on the map collection when it was built), or are the
ones from basepostprocess_deadlock.vpost.
"""
import bpy

GROUP = 'Deadlock Game Look'
SAVED = 'dlr_look_saved'
# basepostprocess_deadlock.vpost m_toneMapParams, game version 6821
DEFAULT_TONEMAP = {'exposure_bias': 0.8992, 'shoulder_strength': 0.3538, 'linear_strength': 0.3258,
                   'linear_angle': 0.2528, 'toe_strength': 0.6966, 'toe_num': 0.0, 'toe_denom': 0.7819,
                   'white_point': 3.9996}
SHADER_EXPOSURE = 2.8


def tonemap_params(scene):
    for c in scene.collection.children_recursive:
        params = c.get('dlr_tonemap')
        if params:
            return {k: float(params.get(k, v)) for k, v in DEFAULT_TONEMAP.items()}
    return dict(DEFAULT_TONEMAP)


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

    def vec(op, a, b=None, x=0):
        n = nodes.new('ShaderNodeVectorMath')
        n.operation = op
        n.location = (x, 0)
        for socket, value in ((n.inputs[0], a), (n.inputs[3] if op == 'SCALE' else n.inputs[1], b)):
            if value is None:
                continue
            if hasattr(value, 'node'):
                links.new(value, socket)
            elif op == 'SCALE' and socket == n.inputs[3]:
                socket.default_value = value
            else:
                socket.default_value = (value,) * 3
        return n.outputs['Vector']

    white = 2.0 ** p['white_point']
    a, b, c = p['shoulder_strength'], p['linear_strength'], p['linear_angle']
    d, e, f = p['toe_strength'], p['toe_num'], p['toe_denom']
    exposure = nodes.new('ShaderNodeValue')
    exposure.name = exposure.label = 'Exposure'
    exposure.outputs[0].default_value = 2.0 ** p['exposure_bias'] * SHADER_EXPOSURE
    exposure.location = (-1400, -200)
    x = vec('MINIMUM', vec('SCALE', inp.outputs[0], exposure.outputs[0], -1200), white, -1000)
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


def register():
    bpy.utils.register_class(DLR_OT_game_look)


def unregister():
    bpy.utils.unregister_class(DLR_OT_game_look)
