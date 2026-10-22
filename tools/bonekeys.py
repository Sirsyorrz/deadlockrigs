"""blender -b -P bonekeys.py -- <hero> : ARKit keys from a model's own face bones (Abrams' and McGinnis' FACS
skeletons: lid pivots, lip, brow, cheek, nose, jaw and tongue bones) -> data/keys.npz (the face mesh),
data/keysmesh_<part>.npz (other meshes the bones move: eyes, teeth, tongue, cigarette) and data/face.json.

Each key is a pose of those bones, baked into shape keys on every mesh it moves, so the face works like every other
hero's (sliders, controls, expressions) and the bones stay free to pose on top. Sizes are mm on Violet's face,
scaled by the eye spacing. Directions: out = towards the side's ear, up, fwd = out of the face.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402

REF_SPACING = 0.088
# lid pivot -> the lid bone it carries
UPPER = ('lid_upper_inner', 'lid_upper_mid', 'lid_upper_outer')
LOWER = ('lid_lower_inner', 'lid_lower_mid', 'lid_lower_outer')
LIPS_UP = ('lip_upper_inner', 'lip_upper_outer', 'lip_upper_corner')
LIPS_LOW = ('lip_lower_inner', 'lip_lower_outer', 'lip_lower_corner')


def mv(*bones, out=0, up=0, fwd=0):
    return [('loc', b, (out, up, fwd)) for b in bones]


# sided recipes are written for the left side ({s}); the right side is mirrored. Centre bones (no {s}) moved by a
# sided key take half the move, so both sides together make the full move.
SIDED = {
    'eyeWide': [('lids', -0.25, -0.1)] + mv('aboveeye_inner_{s}', 'aboveeye_mid_{s}', 'aboveeye_outer_{s}', up=1.5)
    + mv('brow_inner_{s}', 'brow_mid_{s}', 'brow_outer_{s}', up=1.5),
    'eyeSquint': [('lids', 0.0, 0.4)] + mv('squint_inner_{s}', 'squint_mid_{s}', 'squint_outer_{s}', up=2)
    + mv('cheek_upper_{s}', up=1.5),
    'eyeLookUp': [('eye', 'x', -15), ('lids', -0.15, 0.08)] + mv('aboveeye_mid_{s}', 'aboveeye_inner_{s}',
                                                                 'aboveeye_outer_{s}', up=1.5)
    + mv('brow_inner_{s}', 'brow_mid_{s}', 'brow_outer_{s}', up=1.2),
    'eyeLookDown': [('eye', 'x', 15), ('lids', 0.3, -0.12)] + mv('aboveeye_mid_{s}', 'aboveeye_inner_{s}',
                                                                  'aboveeye_outer_{s}', up=-1)
    + mv('brow_inner_{s}', 'brow_mid_{s}', 'brow_outer_{s}', up=-0.6),
    'eyeLookOut': [('eye', 'z', 22)],
    'eyeLookIn': [('eye', 'z', -22)],
    'browDown': mv('brow_inner_{s}', up=-4, out=-2) + mv('brow_mid_{s}', up=-3.5) + mv('brow_outer_{s}', up=-2.5)
    + mv('brow_outer_lower_{s}', up=-1.5) + mv('aboveeye_inner_{s}', 'aboveeye_mid_{s}', up=-1.5)
    + mv('brow_mid', up=-1.5) + mv('nosebridge_upper', up=-1),
    'browOuterUp': mv('brow_outer_{s}', up=6) + mv('brow_outer_lower_{s}', up=4) + mv('brow_mid_{s}', up=3)
    + mv('forehead_outer_{s}', up=2.5) + mv('aboveeye_outer_{s}', up=2),
    'cheekSquint': mv('cheek_upper_{s}', up=3, fwd=0.5) + mv('cheekbone_{s}', up=2) + mv('cheekbone_outer_{s}', up=1)
    + mv('squint_inner_{s}', 'squint_mid_{s}', 'squint_outer_{s}', up=1.5) + [('lids', 0.0, 0.15)],
    'noseSneer': mv('nose_corner_{s}', up=3, fwd=0.5) + mv('nostril_{s}', up=2.5) + mv('nasolabial_upper_{s}', up=3)
    + mv('philtrum_corner_{s}', up=1) + mv('lip_upper_outer_{s}', up=1.5) + mv('nosebridge_lower', up=1)
    + mv('cheek_upper_{s}', up=1),
    'mouthSmile': mv('lip_corner_{s}', up=9, out=6, fwd=-3) + mv('mouth_corner_{s}', up=6, out=4, fwd=-3)
    + mv('lip_upper_corner_{s}', up=6.5, out=4.5) + mv('lip_lower_corner_{s}', up=6.5, out=4.5)
    + mv('mouth_corner_upper_{s}', up=4.5, out=3) + mv('mouth_corner_lower_{s}', up=4, out=3)
    + mv('nasolabial_mid_{s}', up=3.5, out=2) + mv('cheek_mid_{s}', up=3, out=1.5) + mv('cheek_upper_{s}', up=3)
    + mv('lip_upper_outer_{s}', up=2.5, out=1.5) + mv('lip_lower_outer_{s}', up=2.5, out=1.5),
    'mouthFrown': mv('lip_corner_{s}', up=-5) + mv('mouth_corner_{s}', up=-3) + mv('lip_lower_corner_{s}', up=-3)
    + mv('lip_upper_corner_{s}', up=-3) + mv('mouth_corner_lower_{s}', up=-2.5) + mv('chin_corner_{s}', up=-1),
    'mouthStretch': mv('lip_corner_{s}', out=5, up=-2) + mv('mouth_corner_{s}', out=3, up=-1)
    + mv('lip_upper_corner_{s}', out=3) + mv('lip_lower_corner_{s}', out=3, up=-1.5)
    + mv('mouth_corner_lower_{s}', out=2, up=-1.5) + mv('lip_lower_outer_{s}', out=1.5, up=-1),
    'mouthDimple': mv('lip_corner_{s}', fwd=-3, out=2) + mv('mouth_corner_{s}', fwd=-2, out=1)
    + mv('lip_upper_corner_{s}', 'lip_lower_corner_{s}', fwd=-1.5),
    'mouthPress': mv('lip_upper_inner_{s}', 'lip_upper_outer_{s}', up=-1.2) + mv('lip_lower_inner_{s}', 'lip_lower_outer_{s}', up=1.2)
    + mv('lip_upper_mid', up=-1.2) + mv('lip_lower_mid', up=1.2) + mv('lip_corner_{s}', out=0.8),
    'mouthUpperUp': mv('lip_upper_inner_{s}', 'lip_upper_outer_{s}', up=4) + mv('lip_upper_corner_{s}', up=2.5)
    + mv('lip_upper_mid', up=3.5) + mv('nasolabial_upper_{s}', up=1) + mv('philtrum_corner_{s}', up=1.5),
    'mouthLowerDown': mv('lip_lower_inner_{s}', 'lip_lower_outer_{s}', up=-4.5) + mv('lip_lower_corner_{s}', up=-2.5)
    + mv('lip_lower_mid', up=-4.5),
}

CENTRE = {
    'browInnerUp': mv('brow_inner_{L}', 'brow_inner_{R}', up=6) + mv('brow_mid', up=4.5)
    + mv('forehead_inner_{L}', 'forehead_inner_{R}', up=3) + mv('brow_mid_{L}', 'brow_mid_{R}', up=2.5)
    + mv('aboveeye_inner_{L}', 'aboveeye_inner_{R}', up=1.5),
    'cheekPuff': mv('cheek_mid_{L}', 'cheek_lower_{L}', 'cheek_mid_{R}', 'cheek_lower_{R}', out=5)
    + mv('mouth_corner_{L}', 'mouth_corner_{R}', out=2.5, fwd=0.5) + mv('jawline_{L}', 'jawline_{R}', out=2)
    + mv('nasolabial_mid_{L}', 'nasolabial_mid_{R}', out=2, fwd=1),
    'jawOpen': [('jaw', 'x', 14)] + mv('lip_corner_{L}', 'lip_corner_{R}', up=-4, out=-1)
    + mv('mouth_corner_{L}', 'mouth_corner_{R}', up=-3) + mv('lip_upper_corner_{L}', 'lip_upper_corner_{R}', up=-2),
    'jawForward': mv('jaw', fwd=5),
    'jawLeft': mv('jaw', out=5),
    'jawRight': mv('jaw', out=-5),
    'mouthClose': [('close',)],
    'mouthLeft': [('lips', 6, 0, 0)],
    'mouthRight': [('lips', -6, 0, 0)],
    'mouthPucker': [('pucker', 6, 0.3)],
    'mouthFunnel': [('funnel', 4.5, 0.15, 2.5)],
    'mouthRollUpper': mv('lip_upper_mid', 'lip_upper_inner_{L}', 'lip_upper_inner_{R}', 'lip_upper_outer_{L}',
                         'lip_upper_outer_{R}', fwd=-2.5, up=-2.5),
    'mouthRollLower': mv('lip_lower_mid', 'lip_lower_inner_{L}', 'lip_lower_inner_{R}', 'lip_lower_outer_{L}',
                         'lip_lower_outer_{R}', fwd=-2.5, up=3),
    'mouthShrugUpper': mv('lip_upper_mid', 'lip_upper_inner_{L}', 'lip_upper_inner_{R}', 'lip_upper_outer_{L}',
                          'lip_upper_outer_{R}', up=2.5, fwd=1),
    'mouthShrugLower': mv('lip_lower_mid', 'lip_lower_inner_{L}', 'lip_lower_inner_{R}', 'lip_lower_outer_{L}',
                          'lip_lower_outer_{R}', up=3, fwd=1.5) + mv('chin_upper', up=2.5, fwd=1.5)
    + mv('chin_mid', up=1.5, fwd=1),
    'tongueOut': [('tongue',)],
}


class Face:
    def __init__(self, hero):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=hero.raw_glb)
        self.arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
        self.pb = self.arm.pose.bones
        for b in self.pb:
            b.rotation_mode = 'QUATERNION'
        self.meshes = {o.name.rsplit('.', 1)[-1]: o for o in bpy.context.scene.objects
                       if o.type == 'MESH' and o.parent == self.arm}
        self.part = hero.face['part']
        self.mw = self.arm.matrix_world
        self.rest = {b.name: self.mw @ b.matrix_local for b in self.arm.data.bones}
        eye = self.head('eye_L') if 'eye_L' in self.pb else self.head('eye_0_L')
        self.eye = {'L': eye, 'R': self.head('eye_R') if 'eye_R' in self.pb else self.head('eye_0_R')}
        self.s = (self.eye['L'].x - self.eye['R'].x) / REF_SPACING
        self.mm = 0.001 * 1.1 * self.s
        self.base = self.evaluate()

    def head(self, b):
        return self.rest[b].translation.copy()

    def has(self, b):
        return b in self.pb

    def evaluate(self):
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        out = {}
        for part, o in self.meshes.items():
            e = o.evaluated_get(dg)
            me = e.to_mesh()
            a = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get('co', a)
            m = np.array(o.matrix_world)
            out[part] = a.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
            e.to_mesh_clear()
        return out

    def reset(self):
        for b in self.pb:
            b.location = (0, 0, 0)
            b.rotation_quaternion = (1, 0, 0, 0)

    def move(self, bone, d):
        """Adds an armature-space move (metres) to a bone whose parents are at rest."""
        if bone not in self.pb:
            return
        r = self.rest[bone].to_quaternion()
        self.pb[bone].location += r.inverted() @ Vector(d)

    def turn(self, bone, axis, deg):
        """Turns a bone about its head, axis in armature space."""
        if bone not in self.pb:
            return
        r = self.rest[bone].to_quaternion()
        q = r.inverted() @ Quaternion(Vector(axis), math.radians(deg)) @ r
        self.pb[bone].rotation_quaternion = q @ self.pb[bone].rotation_quaternion

    def lid_angle(self, s, name):
        """Angle (degrees, down positive) of a lid bone about its pivot, above the pivot's horizontal."""
        p = self.head(f'{name}_pivot_{s}')
        q = self.head(f'{name}_{s}')
        return math.degrees(math.atan2(q.z - p.z, p.y - q.y))

    def lids(self, s, upper, lower):
        """upper/lower: fraction of the open gap each lid travels towards closing (negative opens)."""
        for u, lo in zip(UPPER, LOWER):
            if not (self.has(f'{u}_pivot_{s}') and self.has(f'{lo}_pivot_{s}')):
                continue
            gap = self.lid_angle(s, u) - self.lid_angle(s, lo)
            self.turn(f'{u}_pivot_{s}', (1, 0, 0), upper * gap)
            self.turn(f'{lo}_pivot_{s}', (1, 0, 0), -lower * gap)

    def blink(self, s, amount=1.0):
        """Both lids meet a quarter of the way up the gap."""
        self.lids(s, 0.75 * amount, 0.25 * amount)


def apply(face, op, s, sx, half):
    kind = op[0]
    mm = face.mm
    if kind == 'loc':
        _, bone, (out, up, fwd) = op
        bone = bone.format(s=s, L='L', R='R')
        centre = not bone.endswith(('_L', '_R'))
        f = 0.5 if half and centre else 1.0
        bx = sx if half else -1 if bone.endswith('_R') else 1
        face.move(bone, Vector((bx * out, -fwd, up)) * mm * f)
    elif kind == 'lids':
        face.lids(s, op[1], op[2])
    elif kind == 'eye':
        _, ax, deg = op
        bone = f'eye_{s}' if face.has(f'eye_{s}') else f'eye_0_{s}'
        face.turn(bone, (1, 0, 0) if ax == 'x' else (0, 0, sx), deg)
    elif kind == 'jaw':
        face.turn('jaw', (1, 0, 0), op[2])
    elif kind == 'lips':
        for b in [f'{n}_{x}' for n in LIPS_UP + LIPS_LOW + ('lip_corner', 'mouth_corner') for x in 'LR'] + \
                 ['lip_upper_mid', 'lip_lower_mid']:
            k = 0.5 if b.startswith('mouth_corner') else 1.0
            face.move(b, Vector((op[1], 0, 0)) * mm * k)
    elif kind in ('pucker', 'funnel'):
        # lips forward (most in the middle) and gathered towards it
        fwd, gather = op[1], op[2]
        for b in [f'{n}_{x}' for n in LIPS_UP + LIPS_LOW + ('lip_corner',) for x in 'LR'] + \
                 ['lip_upper_mid', 'lip_lower_mid']:
            if not face.has(b):
                continue
            p = face.head(b)
            mid = 1 - min(abs(p.x) / max(face.head('lip_corner_L').x, 1e-4), 1)
            d = Vector((-gather * p.x, -fwd * (0.6 + 0.4 * mid) * mm, 0))
            if kind == 'funnel':
                d.z += (1 if 'upper' in b else -1 if 'lower' in b else 0) * op[3] * mid * mm
            face.move(b, d)
        for x, sxx in (('L', 1), ('R', -1)):
            face.move(f'mouth_corner_{x}', Vector((-sxx * 2, -1, 0)) * mm)
    elif kind == 'close':
        # with jawOpen, the lips meet: the lower lip rises back most of the way, the upper comes down a little
        drop = jaw_drop(face)
        for b in [f'{n}_{x}' for n in LIPS_LOW for x in 'LR'] + ['lip_lower_mid']:
            face.move(b, Vector((0, 0, 0.8 * drop)))
        for b in [f'{n}_{x}' for n in LIPS_UP for x in 'LR'] + ['lip_upper_mid']:
            face.move(b, Vector((0, 0, -0.2 * drop)))
    elif kind == 'tongue':
        tongue_out(face)


def jaw_drop(face):
    """How far the lower lip drops with jawOpen."""
    p = face.head('lip_lower_mid') - face.head('jaw')
    a = math.radians(14)
    return abs(p.y * math.sin(a) + p.z * (1 - math.cos(a)))


def tongue_out(face):
    """Out over the lower lip: the root stays, each bone further along slides further forward and the tip rises to
    lip height."""
    chain = [b for b in ('tongue_0', 'tongue_1', 'tongue_2', 'tongue_3', 'tongue_end') if face.has(b)]
    if not chain:
        return
    lip = face.head('lip_lower_mid')
    tip = face.head(chain[-1])
    reach = (tip.y - lip.y) + 0.03 * face.s
    rise = max(lip.z - tip.z, 0) + 0.002 * face.s
    for i, b in enumerate(chain[1:], 1):
        t = i / (len(chain) - 1)
        prev = (i - 1) / (len(chain) - 1)
        face.move(b, Vector((0, -(t - prev) * reach, (t - prev) * rise)))


def build_keys(face):
    keys = {}

    def bake(name, ops, sided):
        face.reset()
        for s, sx in (sided,) if sided else (('L', 1),):
            for op in ops:
                apply(face, op, s, sx, half=bool(sided))
        out = face.evaluate()
        keys[name] = {p: out[p] - face.base[p] for p in out}

    for name, ops in SIDED.items():
        for s, sx, side in (('L', 1, 'Left'), ('R', -1, 'Right')):
            bake(name + side, ops, (s, sx))
    for side, s in (('Left', 'L'), ('Right', 'R')):
        face.reset()
        face.blink(s)
        out = face.evaluate()
        keys['eyeBlink' + side] = {p: out[p] - face.base[p] for p in out}
    for name, ops in CENTRE.items():
        bake(name, ops, None)
    face.reset()
    return keys


def attach(face, keys, bones):
    """Props held on the face (Abrams' cigarette) ride the skin nearest their bone instead of the head."""
    sk = face.base[face.part]
    for bone in bones:
        if not face.has(bone):
            continue
        near = int(np.argmin(np.linalg.norm(sk - np.array(face.head(bone)), axis=1)))
        for part, o in face.meshes.items():
            g = o.vertex_groups.get(bone)
            if g is None:
                continue
            idx = [v.index for v in o.data.vertices if any(x.group == g.index and x.weight > 0.5 for x in v.groups)]
            for parts in keys.values():
                if face.part not in parts:
                    continue
                d = parts[face.part][near]
                parts.setdefault(part, np.zeros_like(face.base[part]))
                parts[part][idx] = d


def mouth_bag(face):
    """Per face vertex: 0 outside the mouth, 1 just inside the lips, 2 deeper in (build.py darkens those faces).
    Inside is skin well behind the front surface between the lip corners."""
    sk = face.base[face.part]
    s = face.s
    cx = face.head('lip_corner_L').x
    lz = (face.head('lip_upper_mid').z + face.head('lip_lower_mid').z) / 2
    # up to just above the upper lip (the nostrils are deep folds too), well down into the jaw
    top = face.head('lip_upper_mid').z + 0.008 * s
    box = (np.abs(sk[:, 0]) < cx * 1.05) & (sk[:, 2] < top) & (sk[:, 2] > lz - 0.045 * s)
    cell = 0.003 * s
    key = np.round(sk[:, [0, 2]] / cell).astype(int)
    front = {}
    for k, y in zip(map(tuple, key[box]), sk[box, 1]):
        front[k] = min(front.get(k, y), y)
    depth = np.array([sk[i, 1] - front.get(tuple(key[i]), sk[i, 1]) if box[i] else 0 for i in range(len(sk))])
    # behind the lips, so deep folds in front of them (nostrils) stay skin
    behind = sk[:, 1] - min(face.head('lip_upper_mid').y, face.head('lip_lower_mid').y)
    return np.where((depth > 0.007 * s) & (behind > 0.014 * s), 2,
                    np.where((depth > 0.004 * s) & (behind > 0.006 * s), 1, 0)).astype(np.int8)


def iris_keys(face, part, inner=30, outer=55):
    """Iris size: the eyeball's front cap slides over the sphere, away from or towards the gaze axis."""
    o = face.meshes[part]
    co = face.base[part]
    out = {}
    for s, side in (('L', 'Left'), ('R', 'Right')):
        bone = f'eye_{s}' if face.has(f'eye_{s}') else f'eye_0_{s}'
        g = o.vertex_groups.get(bone)
        if g is None:
            continue
        ball = np.array([any(x.group == g.index and x.weight > 0.5 for x in v.groups) for v in o.data.vertices])
        if not ball.any():
            continue
        idx = np.where(ball)[0]
        q = co[idx]
        c = q.mean(0)
        d = q - c
        rho = np.linalg.norm(d, axis=1)
        u = d / rho[:, None]
        front = u[u[:, 1] < -np.cos(np.radians(12))]
        pole = front.mean(0)
        pole /= np.linalg.norm(pole)
        alpha = np.arccos(np.clip(u @ pole, -1, 1))
        perp = u - np.outer(u @ pole, pole)
        perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
        fade = 1 - np.clip((alpha - np.radians(inner)) / np.radians(outer - inner), 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        for key, k in (('irisLarge', 1.3), ('irisSmall', 0.75)):
            a2 = alpha * (1 + (k - 1) * fade)
            moved = c + rho[:, None] * (np.outer(np.cos(a2), pole) + np.sin(a2)[:, None] * perp)
            off = np.zeros_like(co)
            off[idx] = moved - q
            out[key + side] = off
    return out


def controls(face):
    """Face control positions like keys.controls(), from the bones."""
    sk = face.base[face.part]
    s = face.s
    c = face.eye['L']
    ez = c.z
    corner = face.head('lip_corner_L')
    lz = (face.head('lip_upper_mid').z + face.head('lip_lower_mid').z) / 2
    em = ez - lz

    def front_y(x, z, r=0.006):
        near = (np.abs(sk[:, 0] - x) < r * s) & (np.abs(sk[:, 2] - z) < r * s)
        if not near.any():
            near = np.hypot(sk[:, 0] - x, sk[:, 2] - z) < 4 * r * s
        return float(sk[near, 1].min())

    def at(x, z, ahead):
        return [round(float(x), 4), round(front_y(x, z) - ahead * s, 4), round(float(z), 4)]
    lm = {
        'lid': at(c.x, ez + 0.016 * s, 0.010),
        'lidlow': at(c.x + 0.002 * s, ez - 0.019 * s, 0.011),
        'brow': at(c.x + 0.006 * s, ez + 0.045 * s, 0.015),
        'cheek': at(c.x + 0.014 * s, lz + 0.475 * em, 0.010),
        'corner': at(corner.x, corner.z, 0.012),
        'lipup': at(0.012 * s, lz + 0.015 * s, 0.015),
        'liplow': at(0.012 * s, lz - 0.017 * s, 0.022),
        'nose': at(0.019 * s, lz + 0.36 * em, 0.013),
    }
    chin = face.head('chin_mid') if face.has('chin_mid') else Vector((0, front_y(0, lz - 0.05 * s), lz - 0.05 * s))
    top = float(sk[:, 2].max())
    return {
        'landmarks': lm,
        'brow_mid': at(0, ez + 0.033 * s, 0.011),
        'mouth': at(0, lz, 0.028),
        'jaw': [0, round(float(chin.y - 0.023 * s), 4), round(float(lz - 0.055 * s), 4)],
        'eye_target': [0, round(float(c.y - 0.51), 4), round(float(ez), 4)],
        'eye_spacing': round(float(c.x), 4),
        'face_box': [0, round(front_y(0, ez) + 0.019, 4), round(top + 0.03 * s, 4)],
        'scale': round(float(s), 3),
    }


if __name__ == '__main__':
    hero, _ = from_argv()
    face = Face(hero)
    keys = build_keys(face)
    eye_part = next(p for p, o in face.meshes.items()
                    if o.vertex_groups.get('eye_L') or o.vertex_groups.get('eye_0_L'))
    for k, v in iris_keys(face, eye_part).items():
        keys[k] = {eye_part: v}
    attach(face, keys, hero.face.get('attach', []))
    np.save(os.path.join(hero.data, 'keys_bag.npy'), mouth_bag(face))
    per_mesh = {}
    for k, parts in keys.items():
        for p, d in parts.items():
            if np.abs(d).max() > 2e-5:
                per_mesh.setdefault(p, {})[k] = d.astype(np.float32)
    for f in os.listdir(hero.data):
        if f.startswith('keysmesh_'):
            os.remove(os.path.join(hero.data, f))
    for p, ks in per_mesh.items():
        name = 'keys.npz' if p == face.part else f'keysmesh_{p}.npz'
        np.savez_compressed(os.path.join(hero.data, name), **ks)
    json.dump(controls(face), open(os.path.join(hero.data, 'face.json'), 'w'), indent=1)
    for k in sorted(keys):
        sizes = {p: np.linalg.norm(d, axis=1).max() * 1000 for p, d in keys[k].items()}
        print('%-22s %s' % (k, ' '.join(f'{p} {v:.1f}' for p, v in sizes.items() if v > 0.01)))
    print('bone keys', len(keys), 'scale %.2f' % face.s, 'meshes', {p: len(k) for p, k in per_mesh.items()})
