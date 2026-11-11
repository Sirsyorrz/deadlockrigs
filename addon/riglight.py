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
- Reflections, when the map has 'Deadlock reflections' (tools/map_reflections.py): the cubemaps through the split
  sum, scaled by the ambient's luminance over what the cubemaps saw, capped at 1 (the per-view cap from
  lb_cubemap_normalization_max, which Deadlock's gameinfo.gi sets to 1), 32 under exposure control. No multiple scattering, unlike map surfaces, and none at full roughness where the
  material asks (F_NO_SPECULAR_AT_FULL_ROUGHNESS). Every term checks out against the shader, yet that reads about
  ten times too shiny next to the game, which also has screen space AO on it and its direct and rim light under
  it; REFLECTION_STRENGTH is matched to in-game screenshots by eye.

Everything else in the scene that isn't part of the map (the user's own models, the default cube) gets the group
'Deadlock object light' instead: the plain probe irradiance at the normal and the cubemap specular, as the game
lights its props (map props bake the same probes per vertex, map_lighting.bake_probes). No toon step, no exposure
control: those are for heroes only.
"""
import bpy
from bpy.app.handlers import persistent

PROBE_GROUP = 'Deadlock probe light'
REFLECTION_GROUP = 'Deadlock reflections'
RIG_GROUP = 'Deadlock rig light'
NODE = 'dlr_rig_light'
VERSION = 5
OBJECT_GROUP = 'Deadlock object light'
OBJECT_NODE = 'dlr_object_light'
OBJECT_VERSION = 1
MAP_BAKED_GROUP = 'Deadlock baked light'
REFLECTION_STRENGTH = 0.1

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


def reflection_group():
    return next((g for g in bpy.data.node_groups if g.name == REFLECTION_GROUP), None)


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


def add_reflections(g, reflections, inp, ambient, normal, ao, control):
    """The game's hero env map specular; see the module docstring."""
    albedo, metallic = inp.outputs['Base Color'], inp.outputs['Metallic']
    rough = g.sat(inp.outputs['Roughness'])
    f0 = g.vlerp((0.04, 0.04, 0.04), albedo, metallic)
    ax, ay, az = g.split(albedo)
    f0 = g.vec('SCALE', f0, scale=g.sat(g.math('MULTIPLY', g.math('MAXIMUM', ax, g.math('MAXIMUM', ay, az)), 25.0)))
    node = g.nodes.new('ShaderNodeGroup')
    node.name = node.label = 'Reflections'
    node.node_tree = reflections
    g.put(node.inputs['Normal'], normal)
    g.put(node.inputs['Roughness'], rough)
    g.put(node.inputs['Specular Color'], f0)
    luma = g.vec('DOT_PRODUCT', ambient, (0.2125, 0.7154, 0.0721))
    baked = node.outputs['Baked Luma']
    norm = g.math('MINIMUM', g.math('DIVIDE', luma, g.math('MAXIMUM', baked, 1e-6)), g.lerp(1.0, 32.0, control))
    norm = g.select(g.math('GREATER_THAN', baked, 0.0), norm, 1.0)
    kill = g.sat(g.math('MULTIPLY_ADD', g.math('SUBTRACT', rough, 0.99608), -255.0, 1.0))
    kill = g.lerp(1.0, kill, inp.outputs['No Rough Specular'])
    to_camera = g.nodes.new('ShaderNodeNewGeometry').outputs['Incoming']
    n_dot_v = g.sat(g.vec('DOT_PRODUCT', normal, to_camera))
    occlusion = g.sat(g.math('ADD', g.math('SUBTRACT', g.math('POWER', g.math('ADD', n_dot_v, ao),
                                                                g.math('MULTIPLY', rough, rough)), 1.0), ao))
    scale = g.math('MULTIPLY', g.math('MULTIPLY', norm, kill), occlusion)
    return g.vec('SCALE', g.vec('MULTIPLY', node.outputs['Radiance'], node.outputs['Single Scatter']), scale=scale)


def build_rig_group(tree, probes, reflections=None):
    for name, kind, default in (('Base Color', 'NodeSocketColor', None), ('Alpha', 'NodeSocketFloat', 1.0),
                                ('Metallic', 'NodeSocketFloat', 0.0), ('Roughness', 'NodeSocketFloat', 0.5),
                                ('Occlusion', 'NodeSocketFloat', 1.0),
                                ('Normal', 'NodeSocketVector', None), ('Rim Mask', 'NodeSocketFloat', 1.0),
                                ('Transmissive', 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0)),
                                ('No Rough Specular', 'NodeSocketFloat', 0.0)):
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
    if reflections is not None:
        specular = add_reflections(g, reflections, inp, bounce, normal, ao, control)
        total = g.vec('ADD', total, g.vec('SCALE', specular, scale=g.value('Reflection strength', REFLECTION_STRENGTH)))
    total = g.vec('SCALE', total, scale=g.math('MULTIPLY', inp.outputs['Alpha'], strength))
    g.links.new(total, out.inputs['Emission'])
    tree['dlr_version'] = VERSION
    tree['dlr_reflections'] = reflections is not None
    return tree


def build_object_group(tree, probes, reflections=None):
    """The map's light for props, per pixel: albedo times the probe irradiance, plus the cubemaps as
    map_lighting.add_env_specular does them (without the lightmap part)."""
    for name, kind, default in (('Base Color', 'NodeSocketColor', None), ('Alpha', 'NodeSocketFloat', 1.0),
                                ('Metallic', 'NodeSocketFloat', 0.0), ('Roughness', 'NodeSocketFloat', 0.5),
                                ('Normal', 'NodeSocketVector', None)):
        sock = tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
        if default is not None:
            sock.default_value = default
    tree.interface.new_socket(name='Emission', in_out='OUTPUT', socket_type='NodeSocketColor')
    g = Graph(tree)
    inp, out = g.nodes.new('NodeGroupInput'), g.nodes.new('NodeGroupOutput')
    strength = g.value('Strength', 1.0)
    normal = g.vec('NORMALIZE', inp.outputs['Normal'])
    n = g.nodes.new('ShaderNodeGroup')
    n.name = n.label = 'Probes'
    n.node_tree = probes
    g.put(n.inputs['Normal'], normal)
    irradiance = n.outputs['Irradiance']
    albedo, metallic = inp.outputs['Base Color'], inp.outputs['Metallic']
    diffuse = g.vec('MULTIPLY', g.vec('SCALE', albedo, scale=g.math('SUBTRACT', 1.0, metallic)), irradiance)
    total = diffuse
    if reflections is not None:
        rough = g.math('MAXIMUM', g.sat(inp.outputs['Roughness']), 0.05)
        f0 = g.vlerp((0.04, 0.04, 0.04), albedo, metallic)
        ax, ay, az = g.split(albedo)
        f0 = g.vec('SCALE', f0, scale=g.sat(g.math('MULTIPLY', g.math('MAXIMUM', ax, g.math('MAXIMUM', ay, az)),
                                                   25.0)))
        node = g.nodes.new('ShaderNodeGroup')
        node.name = node.label = 'Reflections'
        node.node_tree = reflections
        g.put(node.inputs['Normal'], normal)
        g.put(node.inputs['Roughness'], rough)
        g.put(node.inputs['Specular Color'], f0)
        single, multi = node.outputs['Single Scatter'], node.outputs['Multi Scatter']
        luma = g.vec('DOT_PRODUCT', irradiance, (0.2125, 0.7154, 0.0721))
        baked = node.outputs['Baked Luma']
        norm = g.math('MINIMUM', g.math('DIVIDE', luma, g.math('MAXIMUM', baked, 1e-6)), 1.0)
        norm = g.select(g.math('GREATER_THAN', baked, 0.0), norm, 1.0)
        specular = g.vec('ADD', g.vec('MULTIPLY', g.vec('SCALE', node.outputs['Radiance'], scale=norm), single),
                         g.vec('MULTIPLY', multi, irradiance))
        keep = g.vec('MAXIMUM', g.vec('SUBTRACT', (1.0, 1.0, 1.0), g.vec('ADD', single, multi)), (0.0, 0.0, 0.0))
        total = g.vec('ADD', g.vec('MULTIPLY', diffuse, keep), specular)
    total = g.vec('SCALE', total, scale=g.math('MULTIPLY', inp.outputs['Alpha'], strength))
    g.links.new(total, out.inputs['Emission'])
    tree['dlr_version'] = OBJECT_VERSION
    tree['dlr_reflections'] = reflections is not None
    return tree


def rig_group(probes, reflections=None):
    return current_group(RIG_GROUP, VERSION, build_rig_group, probes, reflections)


def object_group(probes, reflections=None):
    return current_group(OBJECT_GROUP, OBJECT_VERSION, build_object_group, probes, reflections)


def current_group(name, version, build, probes, reflections):
    tree = bpy.data.node_groups.get(name)
    if tree is not None and tree.library is None and tree.get('dlr_version') == version and \
            tree.get('dlr_reflections') == (reflections is not None):
        for n in tree.nodes:
            if n.name.startswith('Probes'):
                n.node_tree = probes
            elif n.name == 'Reflections':
                n.node_tree = reflections
        return tree
    if tree is not None and tree.library is None:
        tree.name = name + ' (old)'
    return build(bpy.data.node_groups.new(name, 'ShaderNodeTree'), probes, reflections)


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
    no_rough_spec = float(bool((vmat.get('IntParams') or {}).get('F_NO_SPECULAR_AT_FULL_ROUGHNESS')))
    if '_vmat_g_ttintmaskrimlightmask' in str(textures.get('g_tTintMaskRimLightMask', '')):
        rim = float((vectors.get('TextureRimLightMask1') or (1.0,))[0])
    if '_vmat_g_tnprtransmissivecolor' in str(textures.get('g_tNprTransmissiveColor', '')):
        # the generated texture is sRGB (its reflectivity is the linearized constant)
        transmissive = tuple(srgb_to_linear(float(c)) for c in
                             list(vectors.get('TextureNprTramsissiveColor1') or (0.0, 0.0, 0.0))[:3])
    return rim, transmissive, no_rough_spec


def light_material(mat, group, node_name=NODE):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    rig = node_name == NODE
    if node_name in nodes:
        node = nodes[node_name]
        node.node_tree = group
        for name in ('Base Color', 'Alpha', 'Metallic', 'Roughness', 'Occlusion', 'Normal'):
            if name not in node.inputs or node.inputs[name].is_linked:
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
        node.name = node.label = node_name
        node.node_tree = group
        node.location = bsdf.location.x, bsdf.location.y - 700
        for name in ('Base Color', 'Alpha', 'Metallic', 'Roughness'):
            src = source_of(bsdf.inputs[name])
            if src:
                links.new(src, node.inputs[name])
            else:
                node.inputs[name].default_value = bsdf.inputs[name].default_value
        occlusion = occlusion_source(mat) if rig else None
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
    if not rig:
        return True
    rim, transmissive, no_rough_spec = npr_inputs(mat)
    node.inputs['Rim Mask'].default_value = rim
    node.inputs['Transmissive'].default_value = (*transmissive, 1.0)
    node.inputs['No Rough Specular'].default_value = no_rough_spec
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


MAP_COLLECTION_KEYS = ('dlr_map', 'dlr_crop_of', 'dlr_npcs')
MAP_OBJECT_KEYS = ('dlr_barn_mask', 'dlr_crop_box')


def other_objects(scene, rigs):
    """Objects with materials that are neither rig meshes nor part of a map (its collection, a cropped copy of it,
    its NPCs)."""
    skip = set(rigs)
    for c in scene.collection.children_recursive:
        if any(k in c for k in MAP_COLLECTION_KEYS):
            skip.update(c.all_objects)
    for ob in scene.objects:
        if ob in skip or ob.library is not None or not hasattr(ob, 'material_slots') or not ob.material_slots:
            continue
        if any(k in ob for k in MAP_OBJECT_KEYS):
            continue
        yield ob


def map_lit(mat):
    return any(n.type == 'GROUP' and n.node_tree is not None and n.node_tree.name.startswith(MAP_BAKED_GROUP)
               for n in mat.node_tree.nodes)


def map_exposure(scene):
    """The map's fixed exposure (log2), the game's tonemap scalar, from the map collection in the scene."""
    for c in scene.collection.children_recursive:
        params = c.get('dlr_tonemap')
        if params:
            return float(params.get('exposure', 0.0))
    return 0.0


BARN_MASK = 'dlr_barn_mask'


def exclude_barn_masks(scene):
    """Shadow links the scene's own lights past the map's barn light masks, which only their barn light should see.
    The map's lights are linked at build time. Returns the number of lights changed."""
    masks = [ob for ob in scene.objects if BARN_MASK in ob]
    if not masks:
        return 0
    changed = 0
    for light in scene.objects:
        if light.type != 'LIGHT' or light.library is not None or light.get('dlr_map_light'):
            continue
        coll = light.light_linking.blocker_collection
        if coll is not None and coll.library is not None:
            continue
        if coll is None:
            coll = bpy.data.collections.new(f'{light.name} barn masks')
            light.light_linking.blocker_collection = coll
        missing = [m for m in masks if m.name_full not in {o.name_full for o in coll.objects}]
        for m in missing:
            coll.objects.link(m)
        excluded = {m.name_full for m in masks}
        # the light linking entries follow the order of coll.objects
        for ob, item in zip(coll.objects, coll.collection_objects):
            if ob.name_full in excluded:
                item.light_linking.link_state = 'EXCLUDE'
        changed += bool(missing)
    return changed


def light_rigs(scene):
    """Adds the probe light to the scene's rig materials and to every other object's that isn't the map's, and keeps
    the scene's own lights from seeing the barn light masks. Returns the number of rig and other materials, or None
    if no map in the file has light probes."""
    exclude_barn_masks(scene)
    probes = probe_group()
    if probes is None:
        return None
    reflections = reflection_group()
    group = rig_group(probes, reflections)
    group.nodes['Tonemap scale'].outputs[0].default_value = 2.0 ** map_exposure(scene)
    rigs = set(rig_meshes(scene))
    rig_mats = {slot.material for ob in rigs for slot in ob.material_slots if slot.material}
    done = sum(light_material(m, group) for m in rig_mats)
    others = {slot.material for ob in other_objects(scene, rigs) for slot in ob.material_slots if slot.material}
    others = [m for m in others - rig_mats if m.use_nodes and m.library is None and NODE not in m.node_tree.nodes
              and not map_lit(m)]
    if not others:
        return done, 0
    group = object_group(probes, reflections)
    return done, sum(light_material(m, group, OBJECT_NODE) for m in others)


_counts = None
_edited = set()


def sync_defaults(mat):
    """Copies the BSDF's unlinked values into the light node, which got copies of them when it was added."""
    found = principled(mat)
    if found is None:
        return
    bsdf = found[2]
    for node in (mat.node_tree.nodes.get(NODE), mat.node_tree.nodes.get(OBJECT_NODE)):
        if node is None:
            continue
        for name in ('Base Color', 'Alpha', 'Metallic', 'Roughness'):
            src, dst = bsdf.inputs[name], node.inputs[name]
            if not src.is_linked and not dst.is_linked:
                value = src.default_value if isinstance(src.default_value, float) else tuple(src.default_value)
                current = dst.default_value if isinstance(dst.default_value, float) else tuple(dst.default_value)
                if value != current:
                    dst.default_value = value


def sync_edited():
    while _edited:
        mat = bpy.data.materials.get(_edited.pop())
        if mat is not None and mat.library is None and mat.node_tree is not None:
            sync_defaults(mat)


def relight():
    if probe_group() is not None:
        light_rigs(bpy.context.scene)
        try:
            bpy.ops.ed.undo_push(message='Probe light')
        except RuntimeError:
            pass


@persistent
def on_depsgraph(_scene, depsgraph):
    """Lights objects and materials added after the map (a new model, a new material on the cube) and keeps lit
    materials' colours in step with their BSDF. From timers: depsgraph handlers run often and shouldn't edit data
    themselves."""
    global _counts
    for update in depsgraph.updates:
        mat = update.id.original if isinstance(update.id, bpy.types.Material) else None
        if mat is not None and mat.node_tree is not None and (NODE in mat.node_tree.nodes or
                                                              OBJECT_NODE in mat.node_tree.nodes):
            _edited.add(mat.name)
    if _edited and not bpy.app.timers.is_registered(sync_edited):
        bpy.app.timers.register(sync_edited, first_interval=0.0)
    counts = len(bpy.data.objects), len(bpy.data.materials)
    if counts == _counts:
        return
    _counts = counts
    if probe_group() is not None and not bpy.app.timers.is_registered(relight):
        bpy.app.timers.register(relight, first_interval=0.0)


class DLR_OT_light_rigs(bpy.types.Operator):
    bl_idname = 'dlr.light_rigs'
    bl_label = 'Light Scene'
    bl_description = ("Light the scene's rigs with the map's light probes, as the game lights heroes, and your "
                      'own objects as it lights props, and keep lights you added from casting the map\'s barn light '
                      'masks. Linking or appending a map, appending a rig, or adding objects or materials with a '
                      'map in the scene does this already')
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        done = light_rigs(context.scene)
        if done is None:
            self.report({'ERROR'}, 'No map with light probes for rigs in this file; rebuild the map and link or '
                                   'append it again')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Probe light on {done[0]} rig and {done[1]} other materials')
        return {'FINISHED'}


def draw(layout, context):
    tree = bpy.data.node_groups.get(RIG_GROUP)
    row = layout.row(align=True)
    row.operator('dlr.light_rigs', icon='LIGHTPROBE_VOLUME')
    if tree is not None and 'Strength' in tree.nodes:
        row.prop(tree.nodes['Strength'].outputs[0], 'default_value', text='Strength')
        if 'Exposure control' in tree.nodes:
            layout.prop(tree.nodes['Exposure control'].outputs[0], 'default_value', text='Exposure Control')
        if 'Reflection strength' in tree.nodes:
            layout.prop(tree.nodes['Reflection strength'].outputs[0], 'default_value', text='Reflections')


def register():
    bpy.utils.register_class(DLR_OT_light_rigs)
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph)


def unregister():
    if on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph)
    for timer in (relight, sync_edited):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    bpy.utils.unregister_class(DLR_OT_light_rigs)
