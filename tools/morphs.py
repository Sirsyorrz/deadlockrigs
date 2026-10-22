"""blender -b -P morphs.py -- <hero> : ARKit keys and face control positions from a model's own game morphs
(Billy, Rem, Victor) -> data/keys.npz, data/face.json

The game morphs are FACS shapes (eyeClosedL, lipCornerPullerR, ...). Each ARKit key is a sum of them. A morph with
an in-between (`<morph>_50`, the shape at 0.5) also gives `<key>_mid`, which rig.py drives with a tent peaking at
0.5, so the key passes through the in-between instead of straight to the full shape. Eye looks turn the eyeball
vertices about the eye bones. Controls sit in front of the skin each morph moves most.
"""
import json
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402

# ARKit key -> alternatives, the first whose morphs all exist is used; {s} = L/R, {S} = Left/Right
MAP = {
    'eyeBlink{S}': [['eyeClosed{s}']],
    'eyeWide{S}': [['upperLidRaiser{s}']],
    'eyeSquint{S}': [['squint{s}']],
    'browDown{S}': [['browLowerer{s}'], ['innerBrowLowerer{s}']],
    'browInnerUp': [['innerBrowRaiserL', 'innerBrowRaiserR']],
    'browOuterUp{S}': [['outerBrowRaiser{s}']],
    'cheekSquint{S}': [['cheekRaiser{s}']],
    'noseSneer{S}': [['noseWrinkler{s}'], ['nostrilDilator{s}']],
    'jawOpen': [['jawOpen']],
    'jawForward': [['jawThrust']],
    'mouthSmile{S}': [['lipCornerPuller{s}']],
    'mouthFrown{S}': [['lipCornerDepressor{s}']],
    'mouthStretch{S}': [['lipStretcher{s}']],
    'mouthUpperUp{S}': [['upperLipRaiser{s}']],
    'mouthLowerDown{S}': [['lowerLipDepressor{s}']],
    'mouthPress{S}': [['upperLipTightener{s}', 'lowerLipTightener{s}']],
    'mouthPucker': [['lipPuckererL', 'lipPuckererR']],
    'mouthFunnel': [['upperLipForwardL', 'upperLipForwardR', 'lowerLipForwardL', 'lowerLipForwardR']],
    'mouthShrugLower': [['chinRaiserL', 'chinRaiserR']],
}
# control landmark -> morphs whose most-moved skin places it (first present wins)
LANDMARKS = {
    'lid': ['eyeClosed'],
    'lidlow': ['squint'],
    'brow': ['browLowerer', 'innerBrowLowerer', 'outerBrowRaiser'],
    'cheek': ['cheekRaiser'],
    'nose': ['noseWrinkler', 'nostrilDilator'],
    'corner': ['lipStretcher', 'lipCornerPuller', 'lipCornerDepressor'],
    'lipup': ['upperLipRaiser', 'upperLipForward'],
    'liplow': ['lowerLipDepressor', 'lowerLipForward', 'chinRaiser'],
}
# Violet's hand-placed controls (heroes.json), the reference the control sizes and offsets were tuned on
REF_EYE_X, REF_CORNER_X = 0.044, 0.033
AHEAD = {'lid': 0.010, 'lidlow': 0.011, 'brow': 0.015, 'cheek': 0.010, 'nose': 0.013, 'corner': 0.012,
         'lipup': 0.015, 'liplow': 0.022}


def load(hero):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=hero.raw_glb)
    part = hero.face['part']
    ob = next(o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name.endswith('.' + part))
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    n = len(ob.data.vertices)
    mw = np.array(ob.matrix_world)
    rot = mw[:3, :3]

    def co(kb):
        a = np.empty(n * 3, np.float32)
        kb.data.foreach_get('co', a)
        return a.reshape(-1, 3).astype(np.float64)
    kbs = ob.data.shape_keys.key_blocks
    base = co(kbs[0])
    world = base @ rot.T + mw[:3, 3]
    morphs = {kb.name: (co(kb) - base) @ rot.T for kb in kbs[1:]}
    eyes = {}
    for s in 'LR':
        g = ob.vertex_groups.get(f'eye_0_{s}')
        if g is None or f'eye_0_{s}' not in arm.pose.bones:
            continue
        ball = np.array([any(x.group == g.index and x.weight > 0.5 for x in v.groups) for v in ob.data.vertices])
        if ball.any():
            eyes[s] = (np.array(arm.matrix_world @ arm.pose.bones[f'eye_0_{s}'].head), ball)
    return world, morphs, eyes


def arkit_keys(morphs):
    keys = {}
    for pattern, options in MAP.items():
        for s, side in (('L', 'Left'), ('R', 'Right')) if '{S}' in pattern else ((None, None),):
            name = pattern.replace('{S}', side) if s else pattern
            sided = [[m.replace('{s}', s or '') for m in o] for o in options]
            opt = next((o for o in sided if all(m in morphs for m in o)), None)
            if opt is None:
                continue
            keys[name] = sum(morphs[m] for m in opt)
            mids = [morphs[m + '_50'] - 0.5 * morphs[m] for m in opt if m + '_50' in morphs]
            if mids:
                keys[name + '_mid'] = sum(mids)
    return keys


def rotate(p, piv, axis, deg):
    a = np.radians(deg)
    k = np.array(axis, float)
    q = p - piv
    q = q * np.cos(a) + np.cross(k, q) * np.sin(a) + np.outer(q @ k, k) * (1 - np.cos(a))
    return q + piv


def eye_keys(world, eyes):
    keys = {}
    for s, side, sx in (('L', 'Left', 1), ('R', 'Right', -1)):
        if s not in eyes:
            continue
        piv, ball = eyes[s]
        # turning about +x by a negative angle takes the front (-y) up
        for key, axis, deg in (('eyeLookUp', (1, 0, 0), -15), ('eyeLookDown', (1, 0, 0), 15),
                               ('eyeLookOut', (0, 0, 1), sx * 22), ('eyeLookIn', (0, 0, 1), -sx * 22)):
            d = np.zeros_like(world)
            d[ball] = rotate(world[ball], piv, axis, deg) - world[ball]
            keys[key + side] = d
    return keys


def centre(world, d, side=1):
    """Rest position of the skin a morph moves most, on the given side."""
    m = np.linalg.norm(d, axis=1)
    on = (world[:, 0] * side > 0.002) if side else np.ones(len(m), bool)
    m = np.where(on, m, 0)
    top = m > 0.6 * m.max()
    return (world[top] * m[top, None]).sum(0) / m[top].sum()


def controls(world, morphs, eyes):
    face = np.zeros(len(world), bool)
    for d in morphs.values():
        face |= np.linalg.norm(d, axis=1) > 1e-4

    def front_y(x, z, r):
        near = face & (np.abs(world[:, 0] - x) < r) & (np.abs(world[:, 2] - z) < r)
        if not near.any():
            near = face & (np.hypot(world[:, 0] - x, world[:, 2] - z) < 3 * r)
        return world[near, 1].min()

    def place(k):
        # the brow control drives inner and outer brow keys, so it sits over the middle of all of them
        names = LANDMARKS[k] if k == 'brow' else [next((n for n in LANDMARKS[k] if n + 'L' in morphs), None)]
        pts = [(centre(world, morphs[n + 'L'], 1) + centre(world, morphs[n + 'R'], -1) * [-1, 1, 1]) / 2
               for n in names if n and n + 'L' in morphs and n + 'R' in morphs]
        return sum(pts) / len(pts) if pts else None
    raw = {k: p for k in LANDMARKS if (p := place(k)) is not None}
    corner = raw['corner']
    s = abs(corner[0]) / REF_CORNER_X if 'L' not in eyes else eyes['L'][0][0] / REF_EYE_X
    lz = corner[2]
    eye = eyes['L'][0] if 'L' in eyes else raw.get('lid', corner + [0, 0, 0.09 * s])
    ez = eye[2]
    em = ez - lz
    fallback = {
        'lidlow': (eye[0] + 0.002 * s, ez - 0.019 * s),
        'cheek': (eye[0] + 0.014 * s, lz + 0.475 * em),
        'lipup': (0.012 * s, lz + 0.015 * s),
        'liplow': (0.012 * s, lz - 0.017 * s),
    }
    if 'L' in eyes:
        # an upper lid's morph moves the whole lid down over the eye, so its centre sits on the eye; keep the lid
        # controls above and below the eye like Violet's
        for k, dz, pick in (('lid', 0.016, max), ('lidlow', -0.019, min)):
            if k in raw:
                raw[k][2] = pick(raw[k][2], ez + dz * s)
    if 'brow' in raw and 'lid' in raw:
        raw['brow'][2] = max(raw['brow'][2], raw['lid'][2] + 0.02 * s)
    lm = {}
    for k in AHEAD:
        x, z = (raw[k][0], raw[k][2]) if k in raw else fallback.get(k, (None, None))
        if x is None:
            continue
        x = max(abs(x), 0.008 * s)
        lm[k] = [round(float(x), 4), round(float(front_y(x, z, 0.006 * s) - AHEAD[k] * s), 4), round(float(z), 4)]
    brow_z = raw['brow'][2] if 'brow' in raw else ez + 0.03 * s
    jaw_d = morphs['jawOpen']
    chin = centre(world, jaw_d, 0)
    out = {
        'landmarks': lm,
        'brow_mid': [0, round(float(front_y(0, brow_z - 0.01 * s, 0.01 * s) - 0.011 * s), 4), round(float(brow_z - 0.01 * s), 4)],
        'mouth': [0, round(float(front_y(0, lz, 0.008 * s) - 0.028 * s), 4), round(float(lz), 4)],
        'jaw': [0, round(float(front_y(0, chin[2], 0.01 * s) - 0.023 * s), 4), round(float(min(chin[2], lz - 0.04 * s)), 4)],
        'face_box': [0, round(float(front_y(0, brow_z, 0.01 * s) + 0.019 * s), 4), round(float(brow_z + 0.12 * s), 4)],
        'scale': round(float(s), 3),
    }
    if 'L' in eyes:
        out['eye_target'] = [0, round(float(eye[1] - 0.51 * s), 4), round(float(ez), 4)]
        out['eye_spacing'] = round(float(eye[0]), 4)
    return out


if __name__ == '__main__':
    hero, _ = from_argv()
    world, morphs, eyes = load(hero)
    keys = {**arkit_keys(morphs), **eye_keys(world, eyes)}
    fc = controls(world, morphs, eyes)
    np.savez_compressed(os.path.join(hero.data, 'keys.npz'), **{k: v.astype(np.float32) for k, v in keys.items()})
    json.dump(fc, open(os.path.join(hero.data, 'face.json'), 'w'), indent=1)
    for k in sorted(keys):
        print('%-22s max %.1f mm' % (k, np.linalg.norm(keys[k], axis=1).max() * 1000))
    used = {m.replace('{s}', s) for opts in MAP.values() for o in opts for m in o for s in 'LR'}
    used |= {m + '_50' for m in used}
    print('morph keys', len(keys), 'from', len(morphs), 'morphs, scale %.2f' % fc['scale'],
          'eyes', ''.join(eyes), 'unused', sorted(set(morphs) - used))
