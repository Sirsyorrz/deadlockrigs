# SPDX-License-Identifier: GPL-3.0-or-later
"""Lights rigs with the map's light probes, as the game lights characters. The map brings the group 'Deadlock probe
light' (tools/map_probes.py); each rig material gets albedo * (1 - metalness) * occlusion * the probe irradiance as
emission on top of its Principled BSDF, through the local group 'Deadlock rig light', whose Strength value is shared
by every rig."""
import bpy

PROBE_GROUP = 'Deadlock probe light'
RIG_GROUP = 'Deadlock rig light'
NODE = 'dlr_rig_light'


def source_of(socket):
    return socket.links[0].from_socket if socket.is_linked else None


def probe_group():
    return next((g for g in bpy.data.node_groups if g.name == PROBE_GROUP), None)


def rig_group(probes):
    tree = bpy.data.node_groups.get(RIG_GROUP)
    if tree is None or tree.library is not None:
        tree = bpy.data.node_groups.new(RIG_GROUP, 'ShaderNodeTree')
        for name, kind, default in (('Base Color', 'NodeSocketColor', None), ('Alpha', 'NodeSocketFloat', 1.0),
                                    ('Metallic', 'NodeSocketFloat', 0.0), ('Occlusion', 'NodeSocketFloat', 1.0),
                                    ('Normal', 'NodeSocketVector', None)):
            sock = tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
            if default is not None:
                sock.default_value = default
        tree.interface.new_socket(name='Emission', in_out='OUTPUT', socket_type='NodeSocketColor')
        nodes, links = tree.nodes, tree.links
        inp, out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
        probe = nodes.new('ShaderNodeGroup')
        probe.name = 'Probes'
        probe.node_tree = probes
        links.new(inp.outputs['Normal'], probe.inputs['Normal'])
        strength = nodes.new('ShaderNodeValue')
        strength.name, strength.label = 'Strength', 'Rig light strength'
        strength.outputs[0].default_value = 1.0

        def math(op, a, b):
            n = nodes.new('ShaderNodeMath')
            n.operation = op
            for sock, v in zip(n.inputs, (a, b)):
                if hasattr(v, 'node'):
                    links.new(v, sock)
                else:
                    sock.default_value = v
            return n.outputs[0]
        diffuse = math('SUBTRACT', 1.0, inp.outputs['Metallic'])
        scale = math('MULTIPLY', math('MULTIPLY', diffuse, inp.outputs['Occlusion']),
                     math('MULTIPLY', inp.outputs['Alpha'], strength.outputs[0]))
        mul = nodes.new('ShaderNodeMix')
        mul.data_type, mul.blend_type = 'RGBA', 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        links.new(inp.outputs['Base Color'], mul.inputs[6])
        links.new(probe.outputs['Irradiance'], mul.inputs[7])
        vs = nodes.new('ShaderNodeVectorMath')
        vs.operation = 'SCALE'
        links.new(mul.outputs[2], vs.inputs[0])
        links.new(scale, vs.inputs[3])
        links.new(vs.outputs[0], out.inputs['Emission'])
    tree.nodes['Probes'].node_tree = probes
    return tree


def principled(mat):
    """The output, its surface socket and the Principled BSDF under it (first input of any Add Shaders), or None."""
    if not mat or not mat.use_nodes or mat.library is not None:
        return None
    output = next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    surface = source_of(output.inputs['Surface']) if output else None
    node = surface.node if surface else None
    while node is not None and node.type == 'ADD_SHADER':
        node = source_of(node.inputs[0]).node if node.inputs[0].is_linked else None
    if node is None or node.type != 'BSDF_PRINCIPLED':
        return None
    return output, surface, node


def occlusion_source(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'GROUP' and n.node_tree and n.node_tree.name.startswith('glTF') and 'Occlusion' in n.inputs:
            return source_of(n.inputs['Occlusion'])
    return None


def light_material(mat, group):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    if NODE in nodes:
        nodes[NODE].node_tree = group
        return True
    found = principled(mat)
    if found is None:
        return False
    output, surface, bsdf = found
    node = nodes.new('ShaderNodeGroup')
    node.name = node.label = NODE
    node.node_tree = group
    node.location = bsdf.location.x, bsdf.location.y - 700
    for name in ('Base Color', 'Alpha', 'Metallic'):
        src = source_of(bsdf.inputs[name])
        if src:
            links.new(src, node.inputs[name])
        else:
            node.inputs[name].default_value = bsdf.inputs[name].default_value
    occlusion = occlusion_source(mat)
    if occlusion:
        links.new(occlusion, node.inputs['Occlusion'])
    normal = source_of(bsdf.inputs['Normal']) or nodes.new('ShaderNodeNewGeometry').outputs['Normal']
    links.new(normal, node.inputs['Normal'])
    emission = nodes.new('ShaderNodeEmission')
    emission.location = node.location.x + 250, node.location.y
    links.new(node.outputs['Emission'], emission.inputs['Color'])
    add = nodes.new('ShaderNodeAddShader')
    add.location = output.location.x, output.location.y - 200
    links.new(surface, add.inputs[0])
    links.new(emission.outputs[0], add.inputs[1])
    links.new(add.outputs[0], output.inputs['Surface'])
    return True


def rig_meshes(scene):
    rigs = {ob for ob in scene.objects if ob.type == 'ARMATURE' and 'dlrig' in ob}
    for ob in scene.objects:
        if ob.type != 'MESH':
            continue
        parent = ob.parent
        while parent is not None and parent not in rigs:
            parent = parent.parent
        if parent is not None or any(m.type == 'ARMATURE' and m.object in rigs for m in ob.modifiers):
            yield ob


def light_rigs(scene):
    """Adds the probe light to the scene's rig materials. Returns the number of materials, or None if no map in the
    file has light probes for rigs."""
    probes = probe_group()
    if probes is None:
        return None
    group = rig_group(probes)
    mats = {slot.material for ob in rig_meshes(scene) for slot in ob.material_slots if slot.material}
    return sum(light_material(m, group) for m in mats)


class DLR_OT_light_rigs(bpy.types.Operator):
    bl_idname = 'dlr.light_rigs'
    bl_label = 'Light Rigs'
    bl_description = ("Light the scene's rigs with the map's light probes, as the game lights heroes. Linking or "
                      'appending a map, or appending a rig with a map in the scene, does this already')
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        done = light_rigs(context.scene)
        if done is None:
            self.report({'ERROR'}, 'No map with light probes for rigs in this file; rebuild the map and link or '
                                   'append it again')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Rig light on {done} materials')
        return {'FINISHED'}


def draw(layout, context):
    tree = bpy.data.node_groups.get(RIG_GROUP)
    row = layout.row(align=True)
    row.operator('dlr.light_rigs', icon='LIGHTPROBE_VOLUME')
    if tree is not None and 'Strength' in tree.nodes:
        row.prop(tree.nodes['Strength'].outputs[0], 'default_value', text='Strength')


def register():
    bpy.utils.register_class(DLR_OT_light_rigs)


def unregister():
    bpy.utils.unregister_class(DLR_OT_light_rigs)
