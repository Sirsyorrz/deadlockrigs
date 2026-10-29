"""Reads what a map's look depends on from Source2Viewer's text dumps of its entities and resources (used by
mapbuild.py): the sun, sky, fog, light probe volumes and the post-processing's tonemap curve.

Source units are inches; Blender gets metres (the glTF export's scale) and Source (x, y, z) at Blender (y, -x, z).
"""
import math
import re

UNIT = 0.0254

_ENTITY = re.compile(r'keyValues3Data\s*=\s*\{\s*version\s*=\s*\d+\s*values\s*=\s*\{(.*?)\n\t*\}', re.S)
_PAIR = re.compile(r'^\s*([\w.]+) = (.+?)\s*$', re.M)


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
