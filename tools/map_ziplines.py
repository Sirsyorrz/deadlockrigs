"""The lanes' ziplines (citadel_zipline_path), which the game draws in code rather than as map geometry.

Follows the zipline2 shader (shaders/vfx/zipline2_vulkan_50, decompiled with Source2Viewer) with the settings of
materials/dev/zipline_inner_rope, _inner_secondary_rope and _outer_glow: each is a tube along the path that wobbles
by value noise of three octaves (random, changing over time) plus a sideways and an up travelling wave, all pinned at
the nodes, with a radius that swells along travelling waves. The pixel shader adds colour * brightness * 2 * the
distance through the tube in inches, so it glows most down the middle and fades at the edges.

The path is sampled in Python (positions, distance along, pinning, sideways / up axes as point attributes on a line
mesh); a geometry nodes group animates it from the scene time.
"""
import bpy
from mathutils import Vector

from map_lighting import Graph

UNIT = 0.0254
GROUP = 'DLR Zipline'
MATERIAL = 'DLR Zipline'
VERSION = 1
# sample spacing along the path, inches (the secondary rope's finest noise has a 9 inch wavelength, at 0.1 weight)
STEP = 6.0
# the vertex shader eases the motion in over 30 inches past a node
PIN = 30.0

LAYERS = (
    # name, amplitude, wavelength, speed, radius min / max, brightness, left amplitude / freq / speed,
    # up amplitude / freq / speed, travelling wave speed multiplier
    ('inner', 10.0, 6.0, 2.0, 0.5, 6.0, 0.5, 8.0, 1.0, 0.2, 15.0, 0.5, 0.1, 1.0),
    ('secondary', 20.0, 9.0, 3.0, 1.0, 2.0, 0.1, 8.0, 1.5, 0.2, 15.0, 0.5, 0.1, 2.0),
    ('glow', 0.0, 25.0, 2.0, 5.0, 20.0, 0.013, 8.0, 1.0, 0.2, 15.0, 0.5, 0.1, 1.0),
)
RADIUS_AMPLITUDE = (1.0, 0.5, 0.15)
RADIUS_FREQ = (0.3, 1.0, 3.0)
RADIUS_SPEED = (0.1, 0.3, 0.6)
GLOW_RADIUS_FREQ = (0.1, 3.0, 10.0)
OCTAVE_FREQ = (1.0, 0.05, 0.01)
OCTAVE_WEIGHT = (0.1, 0.5, 1.0)


def bezier(p0, p1, p2, p3, t):
    s = 1 - t
    return p0 * s ** 3 + p1 * (3 * s * s * t) + p2 * (3 * s * t * t) + p3 * t ** 3


def sample(segments):
    """Points every STEP inches along the path, with the distance along it (inches) and the pinning (0 at a node, 1
    from PIN * 2 on)."""
    points, along, pinned = [], [], []
    total = 0.0
    for seg in segments:
        p = [Vector(c) for c in seg]
        fine = [bezier(*p, i / 64) for i in range(65)]
        lengths = [0.0]
        for a, b in zip(fine, fine[1:]):
            lengths.append(lengths[-1] + (b - a).length / UNIT)
        n = max(1, round(lengths[-1] / STEP))
        k = 0
        for i in range(0 if not points else 1, n + 1):
            d = lengths[-1] * i / n
            while k < 63 and lengths[k + 1] < d:
                k += 1
            span = lengths[k + 1] - lengths[k]
            f = (d - lengths[k]) / span if span else 0.0
            points.append(fine[k].lerp(fine[k + 1], f))
            along.append(total + d)
            pinned.append(min(1.0, max(0.0, (min(d, lengths[-1] - d) - PIN) / PIN)))
        total += lengths[-1]
    return points, along, pinned


def frames(points):
    """Sideways and up axes at each point."""
    left, up = [], []
    for i in range(len(points)):
        a, b = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
        tangent = (b - a).normalized()
        side = tangent.cross(Vector((0, 0, 1)))
        side = side.normalized() if side.length > 1e-6 else Vector((1, 0, 0))
        left.append(side)
        up.append(side.cross(tangent).normalized())
    return left, up


def line_mesh(name, path):
    points, along, pinned = sample(path['segments'])
    left, up = frames(points)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(points))
    me.vertices.foreach_set('co', [c for p in points for c in p])
    me.edges.add(len(points) - 1)
    me.edges.foreach_set('vertices', [v for i in range(len(points) - 1) for v in (i, i + 1)])
    for attr, kind, values in (('zip_along', 'FLOAT', along), ('zip_pin', 'FLOAT', pinned),
                               ('zip_left', 'FLOAT_VECTOR', left), ('zip_up', 'FLOAT_VECTOR', up)):
        a = me.attributes.new(attr, kind, 'POINT')
        if kind == 'FLOAT':
            a.data.foreach_set('value', values)
        else:
            a.data.foreach_set('vector', [c for v in values for c in v])
    me.update()
    return me


def named(g, name, kind='FLOAT'):
    n = g.node('GeometryNodeInputNamedAttribute', data_type=kind)
    n.inputs['Name'].default_value = name
    return n.outputs['Attribute']


def smooth_wave(g, x):
    """The shader's smoothed triangle wave of x, 0..1."""
    tri = g.math('ABSOLUTE', g.math('SUBTRACT', g.math('MULTIPLY', g.math('FRACT', x), 2.0), 1.0))
    return g.math('MULTIPLY', g.math('MULTIPLY', tri, tri), g.math('SUBTRACT', 3.0, g.math('MULTIPLY', tri, 2.0)))


def layer(g, geometry, t, settings, index, radius_scale, material):
    (name, amp, wavelength, speed, rmin, rmax, brightness, left_amp, left_freq, left_speed,
     up_amp, up_freq, up_speed, travel) = settings
    s = named(g, 'zip_along')
    pin = named(g, 'zip_pin')
    left, up = named(g, 'zip_left', 'FLOAT_VECTOR'), named(g, 'zip_up', 'FLOAT_VECTOR')

    # value noise of three octaves; Blender's 4D noise stands in for the shader's hashed cells
    wobble_x, wobble_y = 0.0, 0.0
    for k, (freq, weight) in enumerate(zip(OCTAVE_FREQ, OCTAVE_WEIGHT)):
        noise = g.node('ShaderNodeTexNoise', noise_dimensions='4D')
        noise.inputs['Detail'].default_value = 0.0
        noise.inputs['Scale'].default_value = 1.0
        g.put(noise.inputs['Vector'], g.combine(g.math('MULTIPLY', s, freq / wavelength), 7.3 * k + 13.1 * index, 0.0))
        g.put(noise.inputs['W'], g.math('MULTIPLY', t, speed / (k * 10 + 1)))
        c = g.split(noise.outputs['Color'])
        wobble_x = g.math('ADD', wobble_x, g.math('MULTIPLY', g.math('SUBTRACT', c[0], 0.5), 4 * weight))
        wobble_y = g.math('ADD', wobble_y, g.math('MULTIPLY', g.math('SUBTRACT', c[1], 0.5), 4 * weight))
    side = smooth_wave(g, g.math('SUBTRACT', g.math('MULTIPLY', s, 0.003 * left_freq),
                                 g.math('MULTIPLY', t, left_speed * travel)))
    rise = smooth_wave(g, g.math('ADD', g.math('SUBTRACT', g.math('MULTIPLY', s, 0.0015 * up_freq),
                                               g.math('MULTIPLY', t, up_speed * travel)), 0.25))
    sway = g.math('MULTIPLY', g.math('SUBTRACT', side, 0.5), left_amp)
    lift = g.math('MULTIPLY', g.math('SUBTRACT', rise, 0.5), up_amp)
    along_left = g.math('ADD', g.math('MULTIPLY', wobble_x, amp), sway)
    along_up = g.math('ADD', g.math('MULTIPLY', wobble_y, amp), lift)
    offset = g.vec('ADD', g.vec('SCALE', left, along_left), g.vec('SCALE', up, along_up))
    offset = g.vec('SCALE', offset, g.math('MULTIPLY', pin, UNIT))
    moved = g.node('GeometryNodeSetPosition')
    g.links.new(geometry, moved.inputs['Geometry'])
    g.links.new(offset, moved.inputs['Offset'])

    freqs = GLOW_RADIUS_FREQ if name == 'glow' else RADIUS_FREQ
    swell = 0.0
    for a, f, v in zip(RADIUS_AMPLITUDE, freqs, RADIUS_SPEED):
        w = smooth_wave(g, g.math('SUBTRACT', g.math('MULTIPLY', s, 0.003 * f), g.math('MULTIPLY', t, v)))
        swell = g.math('ADD', swell, g.math('MULTIPLY', w, a))
    swell = g.math('POWER', g.math('DIVIDE', swell, sum(RADIUS_AMPLITUDE)), 2.0)
    radius = g.math('MULTIPLY', g.math('ADD', rmin, g.math('MULTIPLY', swell, rmax - rmin)), radius_scale)

    curve = g.node('GeometryNodeMeshToCurve')
    g.links.new(moved.outputs[0], curve.inputs['Mesh'])
    geo = curve.outputs[0]
    for attr, value in (('zip_radius', radius), ('zip_bright', brightness)):
        st = g.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT')
        st.inputs['Name'].default_value = attr
        g.links.new(geo, st.inputs['Geometry'])
        g.put(st.inputs['Value'], value)
        geo = st.outputs[0]
    profile = g.node('GeometryNodeCurvePrimitiveCircle')
    profile.inputs['Resolution'].default_value = 12 if name == 'glow' else 8
    profile.inputs['Radius'].default_value = 1.0
    tube = g.node('GeometryNodeCurveToMesh')
    g.links.new(geo, tube.inputs['Curve'])
    g.links.new(profile.outputs['Curve'], tube.inputs['Profile Curve'])
    g.links.new(g.math('MULTIPLY', named(g, 'zip_radius'), UNIT), tube.inputs['Scale'])
    smooth = g.node('GeometryNodeSetShadeSmooth')
    g.links.new(tube.outputs[0], smooth.inputs['Geometry'])
    mat = g.node('GeometryNodeSetMaterial')
    mat.inputs['Material'].default_value = material
    g.links.new(smooth.outputs[0], mat.inputs['Geometry'])
    return mat.outputs[0]


def zipline_material():
    """Additive: lane colour * brightness * 2 * the distance through the tube (inches), front faces only."""
    mat = bpy.data.materials.new(MATERIAL)
    mat.surface_render_method = 'BLENDED'
    mat.use_backface_culling = True
    tree = mat.node_tree
    tree.nodes.clear()
    g = Graph(tree)

    def attr(name, kind):
        n = g.node('ShaderNodeAttribute', attribute_name=name)
        return n.outputs[kind]
    geo = g.node('ShaderNodeNewGeometry').outputs
    facing = g.node('ShaderNodeLayerWeight')
    facing.inputs['Blend'].default_value = 0.5
    # 1 - facing is |N.V|, so 2r|N.V| is the chord through a round tube
    through = g.math('MULTIPLY', g.math('MULTIPLY', attr('zip_radius', 'Fac'), 2.0),
                     g.math('SUBTRACT', 1.0, facing.outputs['Facing']))
    strength = g.math('MULTIPLY', g.math('MULTIPLY', through, attr('zip_bright', 'Fac')), 2.0)
    strength = g.math('MULTIPLY', strength, g.math('SUBTRACT', 1.0, geo['Backfacing']))
    emit = g.node('ShaderNodeEmission')
    g.put(emit.inputs['Color'], attr('zip_color', 'Vector'))
    g.put(emit.inputs['Strength'], strength)
    add = g.node('ShaderNodeAddShader')
    g.links.new(g.node('ShaderNodeBsdfTransparent').outputs[0], add.inputs[0])
    g.links.new(emit.outputs[0], add.inputs[1])
    g.links.new(add.outputs[0], g.node('ShaderNodeOutputMaterial').inputs['Surface'])
    return mat


def zipline_group(material):
    tree = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    tree['dlr_version'] = VERSION
    tree.is_modifier = True
    tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    scale = tree.interface.new_socket(name='Radius', in_out='INPUT', socket_type='NodeSocketFloat')
    scale.default_value = 1.0
    tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    g = Graph(tree)
    inp = g.node('NodeGroupInput').outputs
    t = g.node('GeometryNodeInputSceneTime').outputs['Seconds']
    join = g.node('GeometryNodeJoinGeometry')
    for i, settings in enumerate(LAYERS):
        g.links.new(layer(g, inp['Geometry'], t, settings, i, inp['Radius'], material), join.inputs[0])
    g.links.new(join.outputs[0], g.node('NodeGroupOutput').inputs[0])
    return tree


def add_ziplines(coll, ziplines):
    """One object per lane, in a '<map> ziplines' collection; returns them."""
    if not ziplines:
        return []
    material = zipline_material()
    group = zipline_group(material)
    zcoll = bpy.data.collections.new(coll.name + ' ziplines')
    coll.children.link(zcoll)
    out = []
    for path in ziplines:
        name = f'zipline lane {path["lane"]}' if path.get('lane') else 'zipline ' + path['name']
        me = line_mesh(name, path)
        color = me.attributes.new('zip_color', 'FLOAT_COLOR', 'POINT')
        color.data.foreach_set('color', [c for _ in range(len(me.vertices)) for c in (*path['color'], 1.0)])
        ob = bpy.data.objects.new(name, me)
        zcoll.objects.link(ob)
        mod = ob.modifiers.new('Zipline', 'NODES')
        mod.node_group = group
        ob.visible_shadow = False
        # crop.py keeps effects whole (a cut line would lose its far parts' motion)
        ob['dl_effect'] = 'zipline'
        out.append(ob)
    return out
