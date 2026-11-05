"""What moves in a map (used by map_import.py): textures that scroll and vertices that jitter, as pbr.vfx animates
them, and props that loop an idle animation. The maths follows ValveResourceFormat's renderer (Renderer/Shaders/
pbr.frag.slang and pbr.vert.slang, MIT licence).

Everything runs off the scene's time in seconds, so it plays at the game's speed at any frame rate and shows the
same thing whether the timeline plays, scrubs or renders. Materials read it from the TIME_GROUP node group, whose
value a driver sets from the frame and the active scene's frame rate (a context property, so the map doesn't hold
on to its own scene when it is linked or appended). Jittering materials get a geometry nodes modifier
(MOTION_GROUP) on their meshes; only those, as Blender redraws a mesh with one every frame.
"""
import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion

from map_lighting import Graph, load_image, param, vmat_dict, vmat_params

MOTION_GROUP = 'Deadlock motion'
# bumped when the motion group changes, so a file's older one is replaced
MOTION_VERSION = 2
TIME_GROUP = 'Deadlock time'
UV2_ATTR = '_UV2'
UNIT = 0.0254
# glTF (Y up) to Blender (Z up), as the importer turns it
GLTF_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
# the glTF importer's texture labels for the game textures each scroll speed moves
SCROLLED = (('g_vAlbedoScrollSpeed1', 'g_flAlbedoScrollQuantize1', ('BASE COLOR',), None),
            ('g_vNormalAndRoughnessScrollSpeed1', 'g_flNormalAndRoughnessScrollQuantize1',
             ('NORMAL MAP', 'METALLIC ROUGHNESS'), None),
            ('g_vSelfIllumScrollSpeed1', 'g_flSelfIllumScrollQuantize1', ('EMISSIVE',),
             'g_bUseSecondaryUvForSelfIllum1'))
JITTER_VECTORS = ('g_vJitterAmplitudesXA1', 'g_vJitterAmplitudesYA1', 'g_vJitterAmplitudesZA1',
                  'g_vJitterAmplitudesXB1', 'g_vJitterAmplitudesYB1', 'g_vJitterAmplitudesZB1')


def vectors(mat):
    return dict(vmat_dict(mat).get('VectorParams') or {})


def vec(values, name, default=(0.0, 0.0, 0.0), size=3):
    v = values.get(name)
    return tuple(float(c) for c in list(v)[:size]) if v is not None else default


def scroll(values, name):
    return vec(values, name, (0.0, 0.0), 2)


def jitter_mode(mat):
    return int(vmat_params(mat)[0].get('F_JITTER_VERTICES', 0))


def moves(mat):
    ints, _ = vmat_params(mat)
    values = vectors(mat)
    scrolls = [n for n, *_ in SCROLLED] + ['g_vTranslucentScrollSpeed0_1', 'g_vTranslucentScrollSpeed1_1']
    return bool(jitter_mode(mat) or ints.get('F_ADVANCED_TRANSLUCENCY') or any(any(scroll(values, n)) for n in scrolls))


def labelled(mat, label):
    return [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.label == label]


def scrolled_uv(g, speed, quantize, secondary):
    """UV + fract(speed * time), time snapped to steps of quantize seconds. The game's v runs down the texture,
    Blender's up."""
    uv = g.attribute(UV2_ATTR) if secondary else g.node('ShaderNodeUVMap', uv_map='UVMap').outputs[0]
    if not any(speed):
        return uv
    t = g.node('ShaderNodeGroup', node_tree=time_group()).outputs[0]
    if quantize:
        t = g.math('MULTIPLY', g.math('ROUND', g.math('DIVIDE', t, quantize)), quantize)
    offset = g.vec('FRACTION', g.vec('SCALE', (speed[0], speed[1], 0.0), t))
    return g.vec('ADD', uv, g.vec('MULTIPLY', offset, (1.0, -1.0, 0.0)))


def time_group():
    """Shader node group with the scene's time in seconds."""
    tree = bpy.data.node_groups.get(TIME_GROUP)
    if tree:
        return tree
    tree = bpy.data.node_groups.new(TIME_GROUP, 'ShaderNodeTree')
    tree.interface.new_socket(name='Seconds', in_out='OUTPUT', socket_type='NodeSocketFloat')
    value = tree.nodes.new('ShaderNodeValue')
    tree.links.new(value.outputs[0], tree.nodes.new('NodeGroupOutput').inputs[0])
    driver = value.outputs[0].driver_add('default_value').driver
    driver.type = 'SCRIPTED'
    for name, path in (('fps', 'render.fps'), ('base', 'render.fps_base')):
        var = driver.variables.new()
        var.name, var.type = name, 'CONTEXT_PROP'
        var.targets[0].context_property = 'ACTIVE_SCENE'
        var.targets[0].data_path = path
    driver.expression = 'frame * base / fps'
    return tree


def animate_material(mat, objects, textures):
    """Scrolls the material's textures and, with F_ADVANCED_TRANSLUCENCY, multiplies (or adds, or subtracts) its
    second translucency texture into the opacity. objects use the material; textures maps game texture paths to
    files. Run last, once everything else reads the textures' outputs."""
    ints, floats = vmat_params(mat)
    values = vectors(mat)
    uv2 = any(UV2_ATTR in ob.data.attributes for ob in objects)
    g = Graph(mat.node_tree)
    for speed_name, quantize_name, labels, secondary in SCROLLED:
        speed = scroll(values, speed_name)
        use_uv2 = bool(secondary and ints.get(secondary) and uv2)
        if not any(speed) and not use_uv2:
            continue
        for node in (n for label in labels for n in labelled(mat, label)):
            g.put(node.inputs['Vector'], scrolled_uv(g, speed, param(floats, quantize_name, 0.0), use_uv2))
    if ints.get('F_ADVANCED_TRANSLUCENCY'):
        add_alt_translucency(mat, g, ints, floats, values, uv2, textures)


def add_alt_translucency(mat, g, ints, floats, values, uv2, textures):
    """The opacity from the colour texture's alpha at its own scroll, combined with g_tAltTranslucency's red."""
    color = next(iter(labelled(mat, 'BASE COLOR')), None)
    if color is None or not color.outputs['Alpha'].is_linked:
        return
    alpha = g.node('ShaderNodeTexImage', image=color.image, interpolation=color.interpolation,
                   extension=color.extension)
    g.put(alpha.inputs['Vector'], scrolled_uv(g, scroll(values, 'g_vTranslucentScrollSpeed0_1'),
                                              floats.get('g_flTranslucentScrollQuantize0_1', 0.0),
                                              bool(ints.get('g_bUseSecondaryUvForTranslucency') and uv2)))
    opacity = alpha.outputs['Alpha']
    alt_path = textures.get((vmat_dict(mat).get('TextureParams') or {}).get('g_tAltTranslucency'))
    if alt_path:
        alt = g.node('ShaderNodeTexImage', image=load_image(alt_path), interpolation='Linear', extension='REPEAT')
        g.put(alt.inputs['Vector'], scrolled_uv(g, scroll(values, 'g_vTranslucentScrollSpeed1_1'),
                                                floats.get('g_flTranslucentScrollQuantize1_1', 0.0),
                                                bool(ints.get('g_bUseSecondaryUvForAltTranslucency') and uv2)))
        red = g.split(alt.outputs['Color'])[0]
        mode = int(ints.get('g_nTranslucencyBlendMode1', 0))
        opacity = (g.math('MULTIPLY', opacity, red) if mode == 0 else
                   g.math('ADD' if mode == 1 else 'SUBTRACT', opacity, red, clamp=True))
    for link in list(color.outputs['Alpha'].links):
        g.links.new(opacity, link.to_socket)


def motion_group():
    """Geometry, Material and the jitter settings -> Geometry: moves the material's vertices by pbr's jitter: two
    smoothed triangle waves of position * frequency + time * speed, each axis a dot product of its amplitudes with
    them, snapped to steps, times the jitter mask's red. The normals stay as they were, as the game's vertex shader
    leaves them (Blender's custom normals are stored relative to the faces and would turn with them)."""
    old = next((t for t in bpy.data.node_groups if t.name == MOTION_GROUP and t.library is None), None)
    if old is not None and old.get('dlr_version') == MOTION_VERSION:
        return old
    tree = bpy.data.node_groups.new(MOTION_GROUP, 'GeometryNodeTree')
    tree['dlr_version'] = MOTION_VERSION
    tree.is_modifier = True
    sockets = [('Geometry', 'NodeSocketGeometry'), ('Material', 'NodeSocketMaterial'),
               ('Object Space', 'NodeSocketBool'), ('Frequency A', 'NodeSocketVector'),
               ('Velocity A', 'NodeSocketVector'), ('Frequency B', 'NodeSocketVector'),
               ('Velocity B', 'NodeSocketVector'), ('Amplitude XA', 'NodeSocketVector'),
               ('Amplitude YA', 'NodeSocketVector'), ('Amplitude ZA', 'NodeSocketVector'),
               ('Amplitude XB', 'NodeSocketVector'), ('Amplitude YB', 'NodeSocketVector'),
               ('Amplitude ZB', 'NodeSocketVector'), ('Offset', 'NodeSocketVector'),
               ('Time Step', 'NodeSocketFloat'), ('Displacement Step', 'NodeSocketFloat'),
               ('Mask', 'NodeSocketImage'), ('Mask UV', 'NodeSocketString')]
    for name, kind in sockets:
        tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
    tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    g = Graph(tree)
    inp = g.node('NodeGroupInput').outputs
    out = g.node('NodeGroupOutput')

    seconds = g.node('GeometryNodeInputSceneTime').outputs['Seconds']
    step = inp['Time Step']
    snapped = g.math('MULTIPLY', g.math('ROUND', g.math('DIVIDE', seconds, step)), step)
    t = switch(g, 'FLOAT', g.math('GREATER_THAN', step, 0.0), seconds, snapped)

    position = g.node('GeometryNodeInputPosition').outputs[0]
    p = g.split(position)
    world = g.vec('SCALE', g.combine(g.math('MULTIPLY', p[1], -1.0), p[0], p[2]), 1 / UNIT)
    noise = switch(g, 'VECTOR', inp['Object Space'], world, named(g, '_JITTER_POS'))

    def wave(frequency, velocity):
        phase = g.vec('ADD', g.vec('FRACTION', g.vec('MULTIPLY', noise, frequency)), g.vec('SCALE', velocity, t))
        tri = g.vec('ABSOLUTE', g.vec('SUBTRACT', g.vec('SCALE', g.vec('FRACTION', phase), 2.0), (1, 1, 1)))
        smooth = g.vec('MULTIPLY', g.vec('MULTIPLY', tri, tri),
                       g.vec('SUBTRACT', (3, 3, 3), g.vec('SCALE', tri, 2.0)))
        return g.vec('SUBTRACT', smooth, (0.5, 0.5, 0.5))

    a, b = wave(inp['Frequency A'], inp['Velocity A']), wave(inp['Frequency B'], inp['Velocity B'])
    axes = [g.math('ADD', g.vec('DOT_PRODUCT', inp[f'Amplitude {axis}A'], a),
                   g.vec('DOT_PRODUCT', inp[f'Amplitude {axis}B'], b)) for axis in 'XYZ']
    disp = g.vec('ADD', g.combine(*axes), inp['Offset'])
    dstep = inp['Displacement Step']
    stepped = g.vec('SCALE', g.vec('FLOOR', g.vec('ADD', g.vec('SCALE', disp, g.math('DIVIDE', 1.0, dstep)),
                                                  (0.5, 0.5, 0.5))), dstep)
    disp = switch(g, 'VECTOR', g.math('GREATER_THAN', dstep, 0.0), disp, stepped)

    texture = g.node('GeometryNodeImageTexture', interpolation='Linear', extension='REPEAT')
    g.links.new(inp['Mask'], texture.inputs['Image'])
    uv = g.node('GeometryNodeInputNamedAttribute', data_type='FLOAT_VECTOR')
    g.links.new(inp['Mask UV'], uv.inputs['Name'])
    g.links.new(uv.outputs['Attribute'], texture.inputs['Vector'])
    disp = g.vec('SCALE', disp, g.split(texture.outputs['Color'])[0])

    d = g.split(disp)
    world_offset = g.vec('SCALE', g.combine(d[1], g.math('MULTIPLY', d[0], -1.0), d[2]), UNIT)
    object_offset = g.vec('ADD', g.vec('ADD', g.vec('SCALE', named(g, '_JITTER_X'), d[0]),
                                       g.vec('SCALE', named(g, '_JITTER_Y'), d[1])),
                          g.vec('SCALE', named(g, '_JITTER_Z'), d[2]))
    offset = switch(g, 'VECTOR', inp['Object Space'], world_offset, object_offset)

    normals = g.node('GeometryNodeCaptureAttribute', domain='CORNER')
    normals.capture_items.new('VECTOR', 'Normal')
    g.links.new(inp['Geometry'], normals.inputs['Geometry'])
    g.links.new(g.node('GeometryNodeInputNormal').outputs[0], normals.inputs[1])
    selection = g.node('GeometryNodeMaterialSelection')
    g.links.new(inp['Material'], selection.inputs['Material'])
    move = g.node('GeometryNodeSetPosition')
    g.links.new(normals.outputs['Geometry'], move.inputs['Geometry'])
    g.links.new(selection.outputs[0], move.inputs['Selection'])
    g.links.new(offset, move.inputs['Offset'])
    keep = g.node('GeometryNodeSetMeshNormal', mode='FREE', domain='CORNER')
    g.links.new(move.outputs['Geometry'], keep.inputs['Mesh'])
    g.links.new(normals.outputs[1], keep.inputs['Custom Normal'])
    g.links.new(keep.outputs['Mesh'], out.inputs[0])
    if old is not None:
        old.user_remap(tree)
        bpy.data.node_groups.remove(old)
        tree.name = MOTION_GROUP
    return tree


def switch(g, kind, condition, off, on):
    n = g.node('GeometryNodeSwitch', input_type=kind)
    g.put(n.inputs['Switch'], condition)
    g.put(n.inputs['False'], off)
    g.put(n.inputs['True'], on)
    return n.outputs[0]


def named(g, name):
    n = g.node('GeometryNodeInputNamedAttribute', data_type='FLOAT_VECTOR')
    n.inputs['Name'].default_value = name
    return n.outputs['Attribute']


def set_input(mod, name, value):
    ident = next(i.identifier for i in mod.node_group.interface.items_tree
                 if i.item_type == 'SOCKET' and i.in_out == 'INPUT' and i.name == name)
    getattr(mod.properties.inputs, ident).value = value


def jitter_mask(mat, textures):
    """The jitter mask's file if the material jitters and the mask was exported, else None."""
    if not jitter_mode(mat):
        return None
    return textures.get((vmat_dict(mat).get('TextureParams') or {}).get('g_tJitterMask'))


def add_jitter(ob, mat, mask_path):
    """A motion modifier on ob jittering mat's vertices. mask_path: the mask's file, or its image."""
    ints, floats = vmat_params(mat)
    values = vectors(mat)
    mode = jitter_mode(mat)
    mod = ob.modifiers.new('Deadlock motion: ' + mat.name, 'NODES')
    mod.node_group = motion_group()
    set_input(mod, 'Material', mat)
    set_input(mod, 'Object Space', mode == 2 and '_JITTER_POS' in ob.data.attributes)
    for wave in 'AB':
        frequency = vec(values, f'g_vJitterFrequencies{wave}1')
        speed = param(floats, f'g_flJitterSpeed{wave}', 0.0)
        # a negative frequency runs its axis backwards; its size sets the wavelength
        set_input(mod, f'Frequency {wave}', [abs(f) for f in frequency])
        set_input(mod, f'Velocity {wave}', [math.copysign(speed, f) if f else 0.0 for f in frequency])
    for name in JITTER_VECTORS:
        set_input(mod, 'Amplitude ' + name[len('g_vJitterAmplitudes'):-1], list(vec(values, name)))
    set_input(mod, 'Offset', list(vec(values, 'g_vJitterExpression')))
    set_input(mod, 'Time Step', param(floats, 'g_flJitterQuantizeTime', 0.0))
    set_input(mod, 'Displacement Step', param(floats, 'g_flJitterQuantizeDisplacement', 0.0))
    set_input(mod, 'Mask', mask_path if isinstance(mask_path, bpy.types.Image) else load_image(mask_path))
    secondary = ints.get('g_bUseSecondaryUvForJitterMask') and UV2_ATTR in ob.data.attributes
    set_input(mod, 'Mask UV', UV2_ATTR if secondary else 'UVMap')
    return mod


def animate(meshes, textures):
    """Scrolling, translucency and jitter for every moving material on the meshes. Returns the materials done and
    the meshes that jitter."""
    users = {}
    for ob in meshes:
        for slot in ob.material_slots:
            if slot.material and slot.material.node_tree and moves(slot.material):
                users.setdefault(slot.material, []).append(ob)
    jittering = set()
    for mat, objects in users.items():
        animate_material(mat, objects, textures)
        mask = jitter_mask(mat, textures)
        if mask:
            for ob in objects:
                add_jitter(ob, mat, mask)
                jittering.add(ob)
    return len(users), len(jittering)


def prop_matrix(transform):
    """The root node's transform from the world glTF, in Blender's axes."""
    if 'matrix' in transform:
        m = transform['matrix']
        mat = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]]).transposed()
    else:
        x, y, z, w = transform.get('rotation', (0, 0, 0, 1))
        mat = Matrix.LocRotScale(transform.get('translation', (0, 0, 0)), Quaternion((w, x, y, z)),
                                 transform.get('scale', (1, 1, 1)))
    return GLTF_TO_BLENDER @ mat @ GLTF_TO_BLENDER.inverted()


def fcurves(action):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def import_props(props_file, coll):
    """Imports the animated props mapbuild.py exported, places them where the map has them and loops their idle
    clips. Returns the new objects."""
    data = json.load(open(props_file, encoding='utf-8'))
    folder = os.path.dirname(props_file)
    props_coll = bpy.data.collections.new(coll.name + ' animated props')
    coll.children.link(props_coll)
    new = []
    for prop in data['props']:
        before = set(bpy.data.objects)
        actions = set(bpy.data.actions)
        bpy.ops.import_scene.gltf(filepath=os.path.join(folder, prop['gltf']), import_pack_images=False,
                                  disable_bone_shape=True, import_scene_as_collection=False)
        objects = [ob for ob in bpy.data.objects if ob not in before]
        place = prop_matrix(prop['transform'])
        for ob in objects:
            for c in ob.users_collection:
                c.objects.unlink(ob)
            props_coll.objects.link(ob)
            if ob.parent is None:
                ob.matrix_world = place @ ob.matrix_world
        for action in set(bpy.data.actions) - actions:
            action.use_fake_user = False
            for fc in fcurves(action):
                fc.modifiers.new('CYCLES')
        new += objects
    return new
