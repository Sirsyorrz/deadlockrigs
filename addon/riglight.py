# SPDX-License-Identifier: GPL-3.0-or-later
"""Lights rigs with the map's light probes the way the game's NPR hero shading does (pbr.vfx with F_USE_NPR_LIGHTING,
decompiled from game version 6821). The map brings the group 'Deadlock probe light' (tools/map_probes.py); each rig
material gets the local group 'Deadlock rig light' as emission on top of its Principled BSDF, which keeps the direct
lights and specular.

What the game does for the ambient part, and so does the group:
- It does not sample the probes along the normal. A toon value from the normal against a "primary light" (up, leaned
  35% toward the camera) picks a direction between 45 degrees down and 45 degrees up, facing the camera, and the
  ambient cube is sampled along that.
- Exposure control replaces that sample's luminance with fixed targets (1 up, 0.5 sideways, 0.1 down, over the
  tonemap scalar) and keeps its chromaticity, so heroes read the same in dark and bright rooms.
- A per-material transmissive colour lit from above fills the shadow side.
- Occlusion only reaches the ambient through the toon value and a 25% multi-bounce term.
- Rim light: the light on the point times a wrap term toward the silhouette, on upward facing normals only, not
  multiplied by albedo. The game also rims the direct light and occludes rims by screen depth; this only rims the
  ambient.
"""
import bpy

PROBE_GROUP = 'Deadlock probe light'
RIG_GROUP = 'Deadlock rig light'
NODE = 'dlr_rig_light'
VERSION = 2

# materials/dev/npr_dummy.vmat, which the engine feeds to every NPR material (game version 6821)
NPR = {'diffuse_step_sharpness': 0.9, 'diffuse_pbr_blend': 0.25, 'primary_light_camera_weight': 0.35,
       'exposure_targets': (1.0, 0.5, 0.1), 'rim_wrap': 0.7, 'rim_falloff': 0.7, 'rim_strength': 2.0,
       'rim_up_ramp': (0.2, 0.2)}
TO_XYZ = ((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505))
TO_RGB = ((3.2406, -1.5372, -0.4986), (-0.9689, 1.8758, 0.0415), (0.0557, -0.2040, 1.0570))
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1))


def source_of(socket):
    return socket.links[0].from_socket if socket.is_linked else None


def probe_group():
    return next((g for g in bpy.data.node_groups if g.name == PROBE_GROUP), None)


class Graph:
    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links

    def put(self, socket, value):
        if hasattr(value, 'node'):
            self.links.new(value, socket)
        elif value is not None:
            socket.default_value = value if not isinstance(value, (int, float)) or socket.type == 'VALUE' \
                else (value,) * 3

    def math(self, op, a, b=None, c=None):
        n = self.nodes.new('ShaderNodeMath')
        n.operation = op
        for socket, v in zip(n.inputs, (a, b, c)):
            self.put(socket, v)
        return n.outputs[0]

    def vec(self, op, a, b=None, scale=None):
        n = self.nodes.new('ShaderNodeVectorMath')
        n.operation = op
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        if scale is not None:
            self.put(n.inputs[3], scale)
        return n.outputs['Value' if op in ('DOT_PRODUCT', 'LENGTH') else 'Vector']

    def combine(self, x, y, z):
        n = self.nodes.new('ShaderNodeCombineXYZ')
        for socket, v in zip(n.inputs, (x, y, z)):
            self.put(socket, v)
        return n.outputs[0]

    def split(self, v):
        n = self.nodes.new('ShaderNodeSeparateXYZ')
        self.links.new(v, n.inputs[0])
        return n.outputs

    def value(self, name, default):
        n = self.nodes.new('ShaderNodeValue')
        n.name = n.label = name
        n.outputs[0].default_value = default
        return n.outputs[0]

    def sat(self, x):
        return self.math('MAXIMUM', self.math('MINIMUM', x, 1.0), 0.0)

    def lerp(self, a, b, f):
        return self.math('ADD', a, self.math('MULTIPLY', self.math('SUBTRACT', b, a), f))

    def vlerp(self, a, b, f):
        return self.vec('ADD', a, self.vec('SCALE', self.vec('SUBTRACT', b, a), scale=f))

    def select(self, f, a, b):
        """a where f is 1, b where it is 0."""
        return self.lerp(b, a, f)

    def vselect(self, f, a, b):
        return self.vlerp(b, a, f)

    def step_curve(self, x, sharpness):
        """The game's toon step: an S-curve through 0.5 with slope 1 / (1 - sharpness) there."""
        s = 1.0 / (1.0 - sharpness)
        k = 0.5 ** (1.0 - s)
        low = self.math('MULTIPLY', self.math('POWER', x, s), k)
        high = self.math('SUBTRACT', 1.0, self.math('MULTIPLY', self.math('POWER', self.math('SUBTRACT', 1.0, x), s),
                                                    k))
        return self.select(self.math('LESS_THAN', x, 0.5), low, high)

    def xyY(self, rgb):
        X, Y, Z = (self.vec('DOT_PRODUCT', rgb, row) for row in TO_XYZ)
        total = self.math('MAXIMUM', self.math('ADD', self.math('ADD', X, Y), Z), 1e-8)
        return self.math('DIVIDE', X, total), self.math('DIVIDE', Y, total), Y

    def exposure_controlled(self, rgb, reference, target):
        """rgb at luminance `target`, its chromaticity pulled toward the reference's where it is dimmer than it."""
        x, y, lum = self.xyY(rgb)
        rx, ry, rlum = reference
        f = self.sat(self.math('DIVIDE', lum, self.math('MAXIMUM', rlum, 1e-8)))
        cx, cy = self.lerp(rx, x, f), self.lerp(ry, y, f)
        k = self.math('DIVIDE', target, self.math('MAXIMUM', cy, 1e-8))
        xyz = self.combine(self.math('MULTIPLY', k, cx), target,
                           self.math('MULTIPLY', k, self.math('SUBTRACT', self.math('SUBTRACT', 1.0, cx), cy)))
        return self.combine(*(self.vec('DOT_PRODUCT', xyz, row) for row in TO_RGB))

    def ambient_cube(self, faces, d):
        """The six faces (+X +Y +Z -X -Y -Z) weighted by the squared direction, as the game samples them."""
        total = None
        for axis, comp in enumerate(self.split(d)):
            face = self.vselect(self.math('GREATER_THAN', comp, 0.0), faces[axis], faces[axis + 3])
            part = self.vec('SCALE', face, scale=self.math('MULTIPLY', comp, comp))
            total = part if total is None else self.vec('ADD', total, part)
        return total


def build_rig_group(tree, probes):
    for name, kind, default in (('Base Color', 'NodeSocketColor', None), ('Alpha', 'NodeSocketFloat', 1.0),
                                ('Metallic', 'NodeSocketFloat', 0.0), ('Occlusion', 'NodeSocketFloat', 1.0),
                                ('Normal', 'NodeSocketVector', None), ('Rim Mask', 'NodeSocketFloat', 1.0),
                                ('Transmissive', 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0))):
        sock = tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
        if default is not None:
            sock.default_value = default
    tree.interface.new_socket(name='Emission', in_out='OUTPUT', socket_type='NodeSocketColor')
    g = Graph(tree)
    inp, out = g.nodes.new('NodeGroupInput'), g.nodes.new('NodeGroupOutput')
    strength = g.value('Strength', 1.0)
    tonemap_scale = g.value('Tonemap scale', 1.0)
    control = g.value('Exposure control', 1.0)
    faces = []
    for axis in AXES:
        n = g.nodes.new('ShaderNodeGroup')
        n.name = n.label = f'Probes {axis}'
        n.node_tree = probes
        n.inputs['Normal'].default_value = axis
        faces.append(n.outputs['Irradiance'])

    normal = g.vec('NORMALIZE', inp.outputs['Normal'])
    nx, ny, nz = g.split(normal)
    ao = inp.outputs['Occlusion']
    to_camera = g.nodes.new('ShaderNodeNewGeometry').outputs['Incoming']
    ix, iy, _ = g.split(to_camera)
    across = g.vec('NORMALIZE', g.combine(ix, iy, 0.0))
    w = NPR['primary_light_camera_weight']
    primary = g.vec('NORMALIZE', g.vlerp((0.0, 0.0, 1.0), across, w))

    lit = g.math('MULTIPLY_ADD', g.vec('DOT_PRODUCT', primary, normal), 0.5, 0.5)
    lit = g.sat(g.math('MULTIPLY_ADD', g.math('MULTIPLY', lit, ao), 2.0, -0.5))
    toon = g.step_curve(lit, NPR['diffuse_step_sharpness'])
    below = g.vlerp((0.0, 0.0, -1.0), across, g.math('ADD', toon, 0.5))
    above = g.vlerp(across, (0.0, 0.0, 1.0), g.math('SUBTRACT', toon, 0.5))
    d = g.vec('NORMALIZE', g.vselect(g.math('LESS_THAN', toon, 0.5), below, above))

    average = g.vec('SCALE', faces[0], scale=1.0)
    for f in faces[1:]:
        average = g.vec('ADD', average, f)
    reference = g.xyY(g.vec('SCALE', average, scale=1 / 6))
    up_t, side_t, down_t = NPR['exposure_targets']
    dx, dy, dz = g.split(d)
    target = g.math('ADD', g.math('MULTIPLY', side_t, g.math('ADD', g.math('MULTIPLY', dx, dx),
                                                                  g.math('MULTIPLY', dy, dy))),
                    g.math('MULTIPLY', g.select(g.math('GREATER_THAN', dz, 0.0), up_t, down_t),
                           g.math('MULTIPLY', dz, dz)))
    sample = g.ambient_cube(faces, d)
    bounce = g.vselect(control, g.exposure_controlled(sample, reference, g.math('DIVIDE', target, tonemap_scale)),
                       sample)
    up = g.vselect(control, g.exposure_controlled(faces[2], reference, g.math('DIVIDE', up_t, tonemap_scale)),
                   faces[2])
    shade = g.math('SUBTRACT', 1.0, g.math('MULTIPLY', g.math('ADD', toon, 0.5), 0.5))
    ambient = g.vec('ADD', bounce, g.vec('SCALE', g.vec('MULTIPLY', up, inp.outputs['Transmissive']), scale=shade))

    albedo = inp.outputs['Base Color']
    ao2 = g.math('MULTIPLY', ao, ao)
    def affine(scale, offset):
        return g.vec('ADD', g.vec('SCALE', albedo, scale=scale), (offset,) * 3)
    # the game's multi-bounce occlusion fit (Jimenez et al. 2016)
    multibounce = g.vec('ADD', g.vec('SUBTRACT', g.vec('SCALE', affine(2.0404, -0.3324), scale=g.math('MULTIPLY', ao2, ao)),
                                     g.vec('SCALE', affine(4.7951, -0.6417), scale=ao2)),
                        g.vec('SCALE', affine(2.7552, 0.6903), scale=ao))
    occlusion = g.vlerp((1.0, 1.0, 1.0), multibounce, NPR['diffuse_pbr_blend'])
    diffuse = g.vec('SCALE', albedo, scale=g.math('SUBTRACT', 1.0, inp.outputs['Metallic']))
    lit_ambient = g.vec('MULTIPLY', g.vec('MULTIPLY', ambient, occlusion), diffuse)

    wrap = NPR['rim_wrap']
    facing = g.vec('DOT_PRODUCT', normal, g.vec('SCALE', to_camera, scale=-1.0))
    rim = g.math('POWER', g.sat(g.math('DIVIDE', g.math('ADD', facing, wrap), (1 + wrap) ** 2)), NPR['rim_falloff'])
    lo, hi = NPR['rim_up_ramp']
    upward = g.math('GREATER_THAN', nz, lo) if hi == lo else g.sat(g.math('DIVIDE', g.math('SUBTRACT', nz, lo),
                                                                             hi - lo))
    rim = g.math('MULTIPLY', g.math('MULTIPLY', rim, upward),
                 g.math('MULTIPLY', g.math('MULTIPLY', ao, inp.outputs['Rim Mask']), NPR['rim_strength']))
    total = g.vec('ADD', lit_ambient, g.vec('SCALE', ambient, scale=rim))
    total = g.vec('SCALE', total, scale=g.math('MULTIPLY', inp.outputs['Alpha'], strength))
    g.links.new(total, out.inputs['Emission'])
    tree['dlr_version'] = VERSION
    return tree


def rig_group(probes):
    tree = bpy.data.node_groups.get(RIG_GROUP)
    if tree is not None and tree.library is None and tree.get('dlr_version') == VERSION:
        for n in tree.nodes:
            if n.name.startswith('Probes'):
                n.node_tree = probes
        return tree
    if tree is not None and tree.library is None:
        tree.name = RIG_GROUP + ' (old)'
    return build_rig_group(bpy.data.node_groups.new(RIG_GROUP, 'ShaderNodeTree'), probes)


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


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def npr_inputs(mat):
    """Rim mask and transmissive colour. Only the ones the compiler made from a constant are known; the rig has no
    copy of authored mask and transmissive textures, so those get full rim and no transmission."""
    vmat = mat.get('vmat') or {}
    textures, vectors = vmat.get('TextureParams') or {}, vmat.get('VectorParams') or {}
    rim, transmissive = 1.0, (0.0, 0.0, 0.0)
    if '_vmat_g_ttintmaskrimlightmask' in str(textures.get('g_tTintMaskRimLightMask', '')):
        rim = float((vectors.get('TextureRimLightMask1') or (1.0,))[0])
    if '_vmat_g_tnprtransmissivecolor' in str(textures.get('g_tNprTransmissiveColor', '')):
        # the generated texture is sRGB (its reflectivity is the linearized constant)
        transmissive = tuple(srgb_to_linear(float(c)) for c in
                             list(vectors.get('TextureNprTramsissiveColor1') or (0.0, 0.0, 0.0))[:3])
    return rim, transmissive


def light_material(mat, group):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    if NODE in nodes:
        node = nodes[NODE]
        node.node_tree = group
        for name in ('Base Color', 'Alpha', 'Metallic', 'Occlusion', 'Normal'):
            if node.inputs[name].is_linked:
                continue
            found = principled(mat)
            src = source_of(found[2].inputs[name]) if found and name in found[2].inputs else None
            if name == 'Occlusion':
                src = occlusion_source(mat)
            if src:
                links.new(src, node.inputs[name])
    else:
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
    rim, transmissive = npr_inputs(mat)
    node.inputs['Rim Mask'].default_value = rim
    node.inputs['Transmissive'].default_value = (*transmissive, 1.0)
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


def map_exposure(scene):
    """The map's fixed exposure (log2), the game's tonemap scalar, from the map collection in the scene."""
    for c in scene.collection.children_recursive:
        params = c.get('dlr_tonemap')
        if params:
            return float(params.get('exposure', 0.0))
    return 0.0


def light_rigs(scene):
    """Adds the probe light to the scene's rig materials. Returns the number of materials, or None if no map in the
    file has light probes for rigs."""
    probes = probe_group()
    if probes is None:
        return None
    group = rig_group(probes)
    group.nodes['Tonemap scale'].outputs[0].default_value = 2.0 ** map_exposure(scene)
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
        if 'Exposure control' in tree.nodes:
            layout.prop(tree.nodes['Exposure control'].outputs[0], 'default_value', text='Exposure Control')


def register():
    bpy.utils.register_class(DLR_OT_light_rigs)


def unregister():
    bpy.utils.unregister_class(DLR_OT_light_rigs)
