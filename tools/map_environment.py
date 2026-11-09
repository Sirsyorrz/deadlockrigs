"""Reads what a map's look depends on from Source2Viewer's text dumps of its entities and resources (used by
mapbuild.py): the sun, sky, fog, light probe volumes and the post-processing's tonemap curve.

Source units are inches; Blender gets metres (the glTF export's scale) and Source (x, y, z) at Blender (y, -x, z).
"""
import math
import os
import re

UNIT = 0.0254
# environment_blend textures beyond layer 1's, which the glTF export leaves out (map_blend.py uses them)
BLEND_TEXTURES = ('g_tColor2', 'g_tColor3', 'g_tRevealMask2', 'g_tRevealMask3', 'g_tColorWorldOverlay',
                  'g_tColorWorldOverlay2', 'g_tColorOverlay', 'g_tRevealOverlay', 'g_tNormalRoughness2',
                  'g_tNormalRoughness3', 'g_tPacked2', 'g_tPacked3')
# environment_parallax_layered's interior layers and masks (map_parallax.py)
PARALLAX_TEXTURES = ('g_tColor2', 'g_tColor3', 'g_tMask', 'g_tMask2')

_ENTITY = re.compile(r'keyValues3Data\s*=\s*\{\s*version\s*=\s*\d+\s*values\s*=\s*\{(.*?)\n\t*\}', re.S)
_PAIR = re.compile(r'^\s*([\w.]+) = (.+?)\s*$', re.M)
# multi-line values (world panels' opacity curves) have lines with a lone } that would end the entity
_MULTILINE = re.compile(r'""".*?"""', re.S)


def value(text):
    text = text.strip()
    if text.startswith('resource_name:') or text.startswith('resource:'):
        return text.split(':', 1)[1].strip('"')
    if text.startswith('['):
        return [value(v) for v in text.strip('[] ').split(',') if v.strip()]
    if text in ('true', 'false'):
        return text == 'true'
    text = text.strip('"')
    try:
        return float(text) if re.fullmatch(r'-?[\d.]+(e-?\d+)?', text) else text
    except ValueError:
        return text


def entities(dump):
    """Keys lowercased: Source entity keys are case-insensitive, and maps mix cases (light_probe_atlas_x, _Y, _Z)."""
    dump = _MULTILINE.sub('""', dump)
    return [{k.lower(): value(v) for k, v in _PAIR.findall(body)} for body in _ENTITY.findall(dump)]


def resources(dump, ext):
    return sorted(set(re.findall(r'resource:"([^"]+\.' + ext + ')"', dump)))


def num(ent, key, default=0.0):
    v = ent.get(key, default)
    return float(v) if isinstance(v, (int, float)) else float(default)


def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def color(ent, key, default=(255, 255, 255)):
    rgb = ent.get(key) or default
    return [srgb_to_linear(float(c)) for c in list(rgb)[:3]]


def enabled(ent):
    return not ent.get('startdisabled', False)


def first(ents, cls, pred=lambda e: True):
    return next((e for e in ents if e.get('classname') == cls and enabled(e) and pred(e)), None)


def sun(ents):
    """light_environment as a Blender sun. Source 2 diffuse is N.L times the light colour, without Blender's 1/pi,
    so the strength in W/m2 is pi times the brightness. Baked (directlight 1) means the lightmap already has it."""
    e = first(ents, 'light_environment')
    if not e:
        return None
    strength = math.pi * num(e, 'brightness') * num(e, 'brightnessscale', 1.0)
    return {'strength': max(strength, 0.0), 'color': color(e, 'color'), 'baked': int(num(e, 'directlight', 1)) == 1,
            'shadows': bool(num(e, 'castshadows', 1))}


# a light2 (omni2, barn, rect) has 2 ** brightness at this many units in front of it, falling off as 1 / d^2
REFERENCE = 100.0
# Blender watts for a point light as bright as a light2 of linear brightness 1: the game's diffuse is
# albedo * N.L * colour * brightness * (REFERENCE / d)^2 with d in inches, Blender's albedo / pi * N.L * W / (4 pi d^2)
WATTS = 4 * math.pi ** 2 * (UNIT * REFERENCE) ** 2


def to_blender(v, scale=UNIT):
    return [v[1] * scale, -v[0] * scale, v[2] * scale]


def lights(ents):
    """The omni2, barn and rect lights the game draws at runtime (directlight 2 dynamic, 3 stationary): the lightmap
    only has their bounce. As Blender point, spot and area lights, in Blender metres and axes: position, forward and
    up (the entity's +X and +Z), watts, linear colour or a colour temperature, cone, cutoff distance. Barn lights are
    spots from the eye point of their frustum, which reproduces their falloff, with `near` from the eye to the
    entity. Their cone takes in the whole opening and `mask` describes the opening (half size in metres, superellipse
    exponent, 0 for a rectangle) for map_lighting.add_barn_masks to cut it to shape; orthographic barns and cookies
    aren't brought over."""
    out = []
    for e in ents:
        kind = str(e.get('classname', ''))
        if kind not in ('light_omni2', 'light_barn', 'light_rect') or not enabled(e) or e.get('enabled') is False:
            continue
        if int(num(e, 'directlight', 2)) not in (2, 3):
            continue
        linear = 2 ** num(e, 'brightness') * num(e, 'brightnessscale', 1.0)
        rot = angles_matrix(e.get('angles', (0, 0, 0)))
        # side is VRF's cross(up, forward), the entity's +Y, which the frustum shear is along
        forward, side, up = ([row[i] for row in rot] for i in range(3))
        origin = [float(v) for v in e['origin']]
        size = [float(v) for v in (e.get('size_params') or (0, 0, 0))]
        rng = num(e, 'range', 512.0)
        light = {'name': e.get('targetname') or kind, 'forward': to_blender(forward, 1.0), 'up': to_blender(up, 1.0),
                 'distance': rng * UNIT, 'shadows': int(num(e, 'castshadows', 1)) != 0,
                 # how strongly it lights the volumetric fog (fog lighting modes 1-3)
                 'fog': num(e, 'fogstrength') if int(num(e, 'fog')) in (1, 2, 3) else 0.0}
        if int(num(e, 'colormode')):
            light['temperature'] = num(e, 'colortemperature', 6600.0)
        else:
            light['color'] = color(e, 'color')
        if kind == 'light_barn':
            if size[2] <= 0:
                continue
            near = 1 / size[2]
            shear = [float(v) for v in (e.get('shear') or (0, 0))]
            cx, cy = shear[0] / rng * near, shear[1] / rng * near
            eye = [o - near * f - cx * s - cy * u for o, f, s, u in zip(origin, forward, side, up)]
            # an elliptical cone through the opening's corners: sqrt 2 times the half sizes
            reach = [size[0] + abs(cx), size[1] + abs(cy)]
            half = [math.atan2(reach[0] * math.sqrt(2), near), math.atan2(reach[1] * math.sqrt(2), near)]
            # the game's shape exponent (VRF SceneLight.GetShapeExponent); a cookie's shape is the cookie's
            shape = 0.0 if e.get('lightcookie') else min(max(num(e, 'shape'), 0.0), 1.0)
            light['mask'] = {'origin': to_blender(origin), 'half': [size[0] * UNIT, size[1] * UNIT],
                             'reach': [r * UNIT for r in reach],
                             'exponent': 2 / (0.875 * shape + 0.125) if shape > 0 else 0.0}
            light.update(type='SPOT', position=to_blender(eye), watts=WATTS * linear * ((near + REFERENCE) / REFERENCE) ** 2,
                         cone=2 * max(half), aspect=math.tan(half[0]) / max(math.tan(half[1]), 1e-6),
                         blend=min(max(num(e, 'softx'), num(e, 'softy')), 1.0), distance=(near + rng) * UNIT,
                         near=near * UNIT,
                         radius=num(e, 'luminaire_size', 4.0) * UNIT / 2)
        elif kind == 'light_rect':
            light.update(type='AREA', position=to_blender(origin), watts=WATTS * linear / 4,
                         size=[2 * size[0] * UNIT, 2 * size[1] * UNIT], disc=num(e, 'shape') == 1)
        else:
            # a half angle; Blender spots open to 180 degrees at most, wider cones light all round
            outer = num(e, 'outer_angle', 180.0)
            light.update(position=to_blender(origin), watts=WATTS * linear,
                         radius=min(size[0], size[1]) * UNIT / 2 if size[0] and size[1] else 0.0)
            if outer < 90:
                light.update(type='SPOT', cone=math.radians(2 * outer), aspect=1.0,
                             blend=1 - min(num(e, 'inner_angle'), outer) / outer)
            else:
                light['type'] = 'POINT'
        out.append(light)
    return out


def prop_skins(ents):
    """Model entities shown with a material group other than the first: model, origin (Source units), skin. The
    world export draws every entity model with its default group (Midtown's sapphire storm cloud came out amber)."""
    out = []
    for e in ents:
        skin = str(e.get('skin', 'default'))
        if e.get('model') and skin not in ('default', '0', '0.0', ''):
            out.append({'model': e['model'], 'origin': [float(v) for v in e['origin']], 'skin': skin})
    return out


def material_groups(vmdl_dump):
    """A model's m_materialGroups as [(name, [vmat paths])]."""
    block = re.search(r'm_materialGroups\s*=\s*\[(.*?)\n\t\]', vmdl_dump, re.S)
    if not block:
        return []
    return [(name, re.findall(r'resource:"([^"]+)"', mats))
            for name, mats in re.findall(r'm_name = "([^"]*)"\s*m_materials\s*=\s*\[(.*?)\]', block.group(1), re.S)]


def skin_materials(groups, skin):
    """Default material name -> the skin's, by position in the groups (names without folder or extension)."""
    if not groups:
        return {}
    chosen = next((mats for name, mats in groups if name == skin), None)
    if chosen is None and skin.replace('.0', '').isdigit() and int(float(skin)) < len(groups):
        chosen = groups[int(float(skin))][1]
    base = lambda p: os.path.splitext(os.path.basename(p))[0]
    return {base(a): base(b) for a, b in zip(groups[0][1], chosen or ()) if base(a) != base(b)}


def lane_colors(generic_data_dump):
    """Lane number -> colour (sRGB 0-255) from scripts/generic_data.vdata's m_LaneInfo, which the game's ziplines
    use (the zipline entities' color_tint are from before the lanes were renamed)."""
    m = re.search(r'm_LaneInfo\s*=\s*\[(.*?)\n\t\]', generic_data_dump, re.S)
    if not m:
        return {}
    out = {}
    for i, body in enumerate(re.findall(r'\{(.*?)\}', m.group(1), re.S)):
        rgb = re.search(r'm_Color = \[([^\]]+)\]', body)
        if rgb and 'm_bIsEnemyLane' not in body:
            out[i] = [float(c) for c in rgb.group(1).split(',')[:3]]
    return out


def ziplines(ents, colors=None):
    """The lanes' zipline paths: Bezier segments (start, start handle, end handle, end) in Blender metres and the
    lane's linear colour. Node tangents are relative to the node."""
    colors = colors or {}
    nodes = {}
    for e in ents:
        if e.get('classname') == 'citadel_zipline_path_node' and e.get('path_uniqueid'):
            nodes.setdefault(e['path_uniqueid'], []).append(e)
    out = []
    for path in ents:
        if path.get('classname') != 'citadel_zipline_path' or path.get('hammeruniqueid') not in nodes:
            continue
        chain = sorted(nodes[path['hammeruniqueid']], key=lambda n: num(n, 'path_index'))
        if len(chain) < 2:
            continue

        def at(n, key=None):
            o = [float(c) for c in n['origin']]
            if key:
                o = [a + float(b) for a, b in zip(o, str(n.get(key) or '0 0 0').split())]
            return to_blender(o)
        segments = [[at(a), at(a, 'out_tangent_local'), at(b, 'in_tangent_local'), at(b)]
                    for a, b in zip(chain, chain[1:])]
        lane = int(num(path, 'lane_number'))
        rgb = colors.get(lane) or list(path.get('color_tint') or (255, 255, 255))[:3]
        out.append({'name': str(path.get('targetname') or path['hammeruniqueid']).replace('[PR#]', ''),
                    'lane': lane, 'color': [srgb_to_linear(float(c)) for c in rgb],
                    'radius': num(path, 'radius', 1.0), 'segments': segments})
    return out


def sky(ents, name=None):
    pred = (lambda e: e.get('targetname') == name) if name else (lambda e: True)
    e = first(ents, 'env_sky', pred) or first(ents, 'env_sky')
    if not e or not e.get('skyname'):
        return None
    return {'material': e['skyname'], 'tint': color(e, 'tint_color'), 'brightness': num(e, 'brightnessscale', 1.0)}


def cubemap_fog(ents):
    """env_cubemap_fog, distances and heights in Blender metres. HeightStart > HeightEnd is no height fog."""
    e = first(ents, 'env_cubemap_fog')
    if not e:
        return None
    height_start = num(e, 'cubemapfogheightstart') * UNIT
    width = num(e, 'cubemapfogheightwidth') * UNIT
    if 'cubemapfogheightend' in e:
        width = num(e, 'cubemapfogheightend') * UNIT - height_start
    return {'start': num(e, 'cubemapfogstartdistance') * UNIT, 'end': num(e, 'cubemapfogenddistance') * UNIT,
            'falloff': num(e, 'cubemapfogfalloffexponent', 1.0), 'height_start': height_start,
            'height_width': width, 'height_exponent': num(e, 'cubemapfogheightexponent', 1.0),
            'lod_bias': num(e, 'cubemapfoglodbiase'), 'opacity': num(e, 'cubemapfogmaxopacity', 1.0),
            'sky_entity': e.get('cubemapfogskyentity') if int(num(e, 'cubemapfogsource')) == 1 else None}


def gradient_fog(ents):
    e = first(ents, 'env_gradient_fog')
    if not e:
        return None
    # VRF uses the colour as it is stored (0-255 scaled), not linearised
    rgb = [float(c) / 255 * num(e, 'fogstrength', 1.0) for c in list(e.get('fogcolor') or (255, 255, 255))[:3]]
    return {'start': num(e, 'fogstart') * UNIT, 'end': num(e, 'fogend') * UNIT,
            'falloff': num(e, 'fogfalloffexponent', 1.0), 'height_start': num(e, 'fogstartheight') * UNIT,
            'height_end': num(e, 'fogendheight') * UNIT, 'vertical_exponent': num(e, 'fogverticalexponent', 1.0),
            'color': rgb, 'opacity': num(e, 'fogmaxopacity', 1.0)}


def angles_matrix(angles):
    """Source pitch, yaw, roll in degrees to a rotation matrix (rows), in Source axes."""
    p, y, r = (math.radians(float(a)) for a in (list(angles) + [0, 0, 0])[:3])
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, -sp)
    left = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    # columns are the local axes
    return [[forward[i], left[i], up[i]] for i in range(3)]


def volumetric_fog(ents):
    """The master env_volumetric_fog_controller and its env_volumetric_fog_volumes, Source units. Only the indirect
    light baked into the controller's fog irradiance volume lights the fog."""
    c = first(ents, 'env_volumetric_fog_controller', lambda e: e.get('ismaster', True))
    if not c or not c.get('indirectenabled') or not c.get('fogirradiancevolume'):
        return None
    origin = [float(v) for v in c['origin']]
    bounds = [[o + float(v) for o, v in zip(origin, str(c[k]).split())] for k in ('box_mins', 'box_maxs')]
    volumes = []
    for e in ents:
        if e.get('classname') != 'env_volumetric_fog_volume' or not enabled(e) or num(e, 'fogstrength', 1.0) <= 0:
            continue
        volumes.append({'origin': [float(v) for v in e['origin']], 'rotation': angles_matrix(e.get('angles', (0, 0, 0))),
                        'mins': [float(v) for v in e['box_mins']], 'maxs': [float(v) for v in e['box_maxs']],
                        'strength': num(e, 'fogstrength', 1.0), 'exponent': num(e, 'falloffexponent', 1.0),
                        'sphere': int(num(e, 'shape')) == 1, 'indirect': num(e, 'fogindirectstrength', 1.0),
                        'tint': color(e, 'tintcolor')})
    return {'texture': c['fogirradiancevolume'], 'mins': bounds[0], 'maxs': bounds[1],
            'dims': [int(num(c, f'indirectvoxeldim{a}')) for a in 'xyz'],
            'strength': num(c, 'fogstrength', 1.0), 'indirect': num(c, 'indirectstrength', 1.0),
            'tint': color(c, 'tintcolor'), 'draw_distance': num(c, 'drawdistance', 600.0),
            'fade_in': [num(c, 'fadeinstart', 20.0), num(c, 'fadeinend', 100.0)], 'volumes': volumes}


def probe_volumes(ents):
    """Light probe volumes in the baked atlas, in the order the game's shaders try them: highest indoor/outdoor
    level first, then the smallest. Source units."""
    vols = []
    for e in ents:
        if e.get('classname') not in ('env_light_probe_volume', 'env_combined_light_probe_volume') or not enabled(e):
            continue
        if 'light_probe_atlas_x' not in e or not e.get('lightprobetexture'):
            continue
        mins, maxs = [float(v) for v in e['box_mins']], [float(v) for v in e['box_maxs']]
        vols.append({'origin': [float(v) for v in e['origin']], 'rotation': angles_matrix(e.get('angles', (0, 0, 0))),
                     'mins': mins, 'maxs': maxs, 'level': int(num(e, 'indoor_outdoor_level')),
                     'size': [int(num(e, f'light_probe_size_{a}')) for a in 'xyz'],
                     'offset': [int(num(e, f'light_probe_atlas_{a}')) for a in 'xyz'],
                     'texture': e['lightprobetexture']})
    vols.sort(key=lambda v: (-v['level'], math.prod(b - a for a, b in zip(v['mins'], v['maxs']))))
    return vols


def env_maps(ents):
    """Box projected reflection cubemaps (env_combined_light_probe_volume, env_cubemap_box) in the order the game
    blends them: highest indoor/outdoor level first, then the smaller box, then by array index (as Source 2 Viewer's
    renderer has it since 16fc2fa2b, "Match baked lighting of lightmap versions 8.1 to 8.4"). Source units. Sphere
    cubemaps are left out (neither map has any)."""
    maps = []
    for e in ents:
        if e.get('classname') not in ('env_combined_light_probe_volume', 'env_cubemap_box') or not enabled(e):
            continue
        if not e.get('cubemaptexture') or e.get('customcubemaptexture') or 'array_index' not in e:
            continue
        maps.append({'origin': [float(v) for v in e['origin']], 'rotation': angles_matrix(e.get('angles', (0, 0, 0))),
                     'mins': [float(v) for v in e['box_mins']], 'maxs': [float(v) for v in e['box_maxs']],
                     'fade': [float(v) for v in e.get('edge_fade_dists') or (0, 0, 0)],
                     'level': int(num(e, 'indoor_outdoor_level')), 'index': int(num(e, 'array_index')),
                     'texture': e['cubemaptexture']})
    maps.sort(key=lambda m: (-m['level'], math.prod(b - a for a, b in zip(m['mins'], m['maxs'])), m['index']))
    return maps


def tonemap(vpost_dump):
    block = re.search(r'm_toneMapParams\s*=\s*\{(.*?)\}', vpost_dump, re.S)
    if not block:
        return None
    raw = {k: float(v) for k, v in re.findall(r'm_fl(\w+) = (-?[\d.e-]+)', block.group(1))}
    names = {'ExposureBias': 'exposure_bias', 'ShoulderStrength': 'shoulder_strength',
             'LinearStrength': 'linear_strength', 'LinearAngle': 'linear_angle', 'ToeStrength': 'toe_strength',
             'ToeNum': 'toe_num', 'ToeDenom': 'toe_denom', 'WhitePoint': 'white_point'}
    params = {ours: raw[theirs] for theirs, ours in names.items() if theirs in raw}
    return params if len(params) == len(names) else None


def post_processing(ents):
    """The master post_processing_volume's .vpost and its fixed exposure (log2), when auto exposure is pinned."""
    e = first(ents, 'post_processing_volume', lambda e: e.get('master')) or first(ents, 'post_processing_volume')
    if not e:
        return None, 0.0
    lo, hi = num(e, 'minlogexposure'), num(e, 'maxlogexposure')
    exposure = (lo + hi) / 2 if e.get('enableexposure', True) else 0.0
    return e.get('postprocessing') or None, exposure + num(e, 'exposurecompensation')


def sky_texture(vmat_dump):
    m = re.search(r'm_name = "g_tSkyTexture"\s*m_pValue = resource:"([^"]+)"', vmat_dump)
    bias = re.search(r'm_name = "g_flBrightnessExposureBias"\s*m_flValue = (-?[\d.e-]+)', vmat_dump)
    return (m.group(1) if m else None), (float(bias.group(1)) if bias else 0.0)
