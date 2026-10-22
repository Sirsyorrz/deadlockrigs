"""blender -b out/<hero>.blend -P rig.py -- <hero>   (reads data/gamerig.json, writes out/<hero>_rig.blend)

Adds an animation rig on top of the exported model:
- face controls (CTRL_* bones in front of the face) and raw ARKit sliders (custom properties on the
  FACE bone) that drive the head's shape keys, for heroes with a face config,
- IK legs and arms with pole targets and FK/IK switches (custom properties on the armature object),
- bone collections, colours and widget shapes,
- expression, viseme and hand poses as Pose Library assets and in a sidebar panel (rig_ui.py),
- the game's procedural twist bones (forearm, upper arm, thigh) as constraints,
- the game's cloth chains (hair, apron, ribbons, tassel, shirt flaps) as bakeable secondary motion,
- a loader for the game's animation clips (<hero>_anims.blend, built by anims.py).
The deform bones keep their imported rest pose, so the skinning is untouched. Parts a hero's skeleton lacks
(arms, fingers, twist bones) are skipped.

"""
import json
import math
import os
import re
import sys
import uuid

import bpy
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.dirname(__file__))
from expressions import ARKIT, EXPRESSIONS, GROUPS, VISEMES  # noqa: E402
import hands  # noqa: E402
from hero import from_argv  # noqa: E402

hero, _ = from_argv()
out = hero.rig_blend
GAMERIG = json.load(open(os.path.join(hero.data, 'gamerig.json')))
bpy.context.scene.render.fps = 30
arm = bpy.data.objects[hero.name]
HAS_FACE = bool(hero.face)
head_ob = bpy.data.objects[hero.face['mesh']] if HAS_FACE else None
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
B = arm.data.bones
rest = {b.name: b.matrix_local.copy() for b in B}
hpos = {b.name: b.head_local.copy() for b in B}

# bone categories

CATS = [
    ('Helpers', r'_TWIST|^attach|IKTARGET|AimIntent|_target_|^eye_0|^jaw_0|^weaponHand|_aim|_up(_|$)|_offset$'),
    ('Root', r'^root_motion$'),
    ('Torso', r'^(pelvis|spine_\d|neck_0|head)$'),
    ('Arms FK', r'^(clavicle|arm_upper|arm_lower|hand)_[LR]$'),
    ('Fingers', r'^finger_'),
    ('Legs FK', r'^(leg_upper|leg_lower|ankle|ball)_[LR]$'),
    ('Weapon', r'^weapon|^brush_'),
    ('Secondary', r'.'),
]
COLL_ORDER = ['Root', 'Torso', 'Arms FK', 'Legs FK', 'IK', 'Fingers', 'Props', 'Face', 'Secondary', 'Weapon', 'Helpers']
HIDDEN = {'Helpers'}
HIDDEN_BY_DEFAULT = {'Helpers', 'Secondary'}


def category(name):
    return next(c for c, rx in CATS if re.search(rx, name))


def side(name):
    m = re.search(r'_([LR])(_|$)', name) or re.search(r'([LR])$', name)
    return m.group(1) if m else None


COLOURS = {'L': 'THEME04', 'R': 'THEME01', None: 'THEME09'}

# face controls: name -> (position, widget radius, travel x, travel y, allow forward, allow scale)

CTRLS = {}
FACE = 'FACE'
if HAS_FACE:
    # hand-placed in heroes.json (Violet), else the positions keys.py derived from the head
    FC = hero.face.get('controls') or json.load(open(os.path.join(hero.data, 'face.json')))
    for s, sx in (('L', 1), ('R', -1)):
        for k, (x, y, z) in FC['landmarks'].items():
            CTRLS[f'CTRL_{k}_{s}'] = (Vector((sx * x, y, z)), 0.0035, 0.01, 0.01, False, False)
    CTRLS['CTRL_brow_mid'] = (Vector(FC['brow_mid']), 0.004, 0.0, 0.01, False, False)
    CTRLS['CTRL_mouth'] = (Vector(FC['mouth']), 0.006, 0.01, 0.01, False, True)
    CTRLS['CTRL_jaw'] = (Vector(FC['jaw']), 0.007, 0.01, 0.025, True, False)
    EYE_TARGET = Vector(FC['eye_target'])
    EYE_SPACING = FC['eye_spacing']
    FACE_BONE = Vector(FC['face_box'])


def terms():
    """shape key -> list of (bone, channel, sign, travel). sign +1 counts positive motion."""
    t = {k: [] for k in ARKIT}
    for s, side_name, out_sign in (('L', 'Left', 1), ('R', 'Right', -1)):
        t['eyeBlink' + side_name].append((f'CTRL_lid_{s}', 'LOC_Y', -1, 0.008))
        t['eyeWide' + side_name].append((f'CTRL_lid_{s}', 'LOC_Y', 1, 0.005))
        t['eyeSquint' + side_name].append((f'CTRL_lidlow_{s}', 'LOC_Y', 1, 0.005))
        t['eyeLookUp' + side_name].append(('CTRL_eyes', 'LOC_Y', 1, 0.15))
        t['eyeLookDown' + side_name].append(('CTRL_eyes', 'LOC_Y', -1, 0.15))
        t['eyeLookOut' + side_name].append(('CTRL_eyes', 'LOC_X', out_sign, 0.22))
        t['eyeLookIn' + side_name].append(('CTRL_eyes', 'LOC_X', -out_sign, 0.22))
        t['browDown' + side_name].append((f'CTRL_brow_{s}', 'LOC_Y', -1, 0.006))
        t['browDown' + side_name].append(('CTRL_brow_mid', 'LOC_Y', -1, 0.012))
        t['browOuterUp' + side_name].append((f'CTRL_brow_{s}', 'LOC_Y', 1, 0.008))
        t['cheekSquint' + side_name].append((f'CTRL_cheek_{s}', 'LOC_Y', 1, 0.006))
        t['cheekPuff'].append((f'CTRL_cheek_{s}', 'LOC_X', out_sign, 0.006))
        t['noseSneer' + side_name].append((f'CTRL_nose_{s}', 'LOC_Y', 1, 0.005))
        t['mouthSmile' + side_name].append((f'CTRL_corner_{s}', 'LOC_Y', 1, 0.008))
        t['mouthFrown' + side_name].append((f'CTRL_corner_{s}', 'LOC_Y', -1, 0.008))
        t['mouthStretch' + side_name].append((f'CTRL_corner_{s}', 'LOC_X', out_sign, 0.008))
        t['mouthDimple' + side_name].append((f'CTRL_corner_{s}', 'LOC_X', -out_sign, 0.006))
        t['mouthUpperUp' + side_name].append((f'CTRL_lipup_{s}', 'LOC_Y', 1, 0.005))
        t['mouthPress' + side_name].append((f'CTRL_lipup_{s}', 'LOC_Y', -1, 0.004))
        t['mouthPress' + side_name].append((f'CTRL_liplow_{s}', 'LOC_Y', 1, 0.004))
        t['mouthLowerDown' + side_name].append((f'CTRL_liplow_{s}', 'LOC_Y', -1, 0.005))
    t['browInnerUp'].append(('CTRL_brow_mid', 'LOC_Y', 1, 0.008))
    t['jawOpen'].append(('CTRL_jaw', 'LOC_Y', -1, 0.025))
    t['jawLeft'].append(('CTRL_jaw', 'LOC_X', 1, 0.008))
    t['jawRight'].append(('CTRL_jaw', 'LOC_X', -1, 0.008))
    t['jawForward'].append(('CTRL_jaw', 'LOC_Z', 1, 0.008))
    t['mouthLeft'].append(('CTRL_mouth', 'LOC_X', 1, 0.008))
    t['mouthRight'].append(('CTRL_mouth', 'LOC_X', -1, 0.008))
    for k in ('mouthShrugUpper', 'mouthShrugLower'):
        t[k].append(('CTRL_mouth', 'LOC_Y', 1, 0.005))
    for k in ('mouthRollUpper', 'mouthRollLower'):
        t[k].append(('CTRL_mouth', 'LOC_Y', -1, 0.005))
    t['mouthPucker'].append(('CTRL_mouth', 'SCALE_AVG', -1, 0.5))
    t['mouthFunnel'].append(('CTRL_mouth', 'SCALE_AVG', 1, 0.5))
    return t


# held props (heroes.json "props"): the prop's bone becomes a visible control, its default hold comes from the first
# frame of an idle clip (at rest it sits at the origin), and a chain tying it to the hand stays attached through IK

PROPS = hero.cfg.get('props', [])
HOLD_CLIPS = ('game_out_of_combat_stand_idle', 'game_primary_stand_idle', 'game_weapon_stand_idle', 'game_stand_idle')


def descendants(name):
    return [name] + [c.name for c in B[name].children_recursive]


def sample_hold(bones):
    """Basis (of the keyed bones) and armature-space matrices of bones on the first frame of the hold clip."""
    with bpy.data.libraries.load(hero.anims_blend) as (src, dst):
        dst.actions = [next(n for n in HOLD_CLIPS if n in src.actions)]
    a = dst.actions[0]
    arm.animation_data_create()
    arm.animation_data.action = a
    arm.animation_data.action_slot = a.slots[0]
    bpy.context.scene.frame_set(int(a.frame_range[0]))
    bpy.context.view_layer.update()
    pbs = arm.pose.bones
    keyed = {fc.data_path.split('"')[1] for fc in a.layers[0].strips[0].channelbags[0].fcurves
             if fc.data_path.startswith('pose.bones')}
    basis = {b: [*pbs[b].location, *pbs[b].rotation_quaternion, *pbs[b].scale] for b in bones if b in keyed}
    mats = {b: pbs[b].matrix.copy() for b in bones}
    arm.animation_data.action = None
    bpy.data.actions.remove(a)
    for b in pbs:
        b.location, b.rotation_quaternion, b.scale = (0, 0, 0), (1, 0, 0, 0), (1, 1, 1)
    bpy.context.view_layer.update()
    return basis, mats


PROP_POSE, PROP_CHAINS = {}, []
for prop in PROPS:
    chain = prop.get('chain', [])
    held = [b for b in descendants(prop['bone']) if b not in chain and b != prop.get('chain_end')]
    extra = chain + [prop['chain_end'], prop['anchor'], B[chain[0]].parent.name] if chain else []
    basis, idle = sample_hold(held + extra)
    # the chain end's own keyed offset is part of how it meets the prop
    PROP_POSE.update({b: v for b, v in basis.items() if b in held or b == prop.get('chain_end')})
    if chain:
        PROP_CHAINS.append((prop, idle))
PROP_CHAIN_BONES = {b for prop, _ in PROP_CHAINS for b in prop['chain'] + [prop['chain_end']]}

# edit mode: new bones

bpy.ops.object.mode_set(mode='EDIT')
eb = arm.data.edit_bones


def new_bone(name, head, tail, parent, roll_to=None, copy=None):
    b = eb.new(name)
    if copy:
        src = eb[copy]
        b.head, b.tail, b.roll = src.head.copy(), src.tail.copy(), src.roll
    else:
        b.head, b.tail = head, tail
        if roll_to is not None:
            b.align_roll(roll_to)
    b.parent = eb[parent] if parent else None
    b.use_deform = False
    return b


FWD = Vector((0, -1, 0))
UP = Vector((0, 0, 1))
for name, (pos, *_rest) in CTRLS.items():
    new_bone(name, pos, pos + UP * 0.006, 'head', roll_to=FWD)
if HAS_FACE:
    new_bone('CTRL_eyes', EYE_TARGET, EYE_TARGET + UP * 0.03, 'head', roll_to=FWD)
    new_bone(FACE, FACE_BONE, FACE_BONE + UP * 0.03, 'head', roll_to=FWD)

ROOT = 'root_motion' if 'root_motion' in eb else None
# widget sizes and pole distances were tuned on Violet, whose head bone sits 2.094 m up
SCALE = eb['head'].head.z / 2.094 if 'head' in eb else 1.0
LIMB_DEFS = (('leg', 'knee', 'leg_upper', 'leg_lower', 'ankle', 'IK_foot', 0.55 * SCALE),
             ('arm', 'elbow', 'arm_upper', 'arm_lower', 'hand', 'IK_hand', 0.45 * SCALE))
LIMBS = [(s, d) for s in 'LR' for d in LIMB_DEFS if all(f'{n}_{s}' in eb for n in d[2:5])]
print('IK limbs', [f'{d[0]}_{s}' for s, d in LIMBS])
POLES = {}
for s, (_lname, limb, a, b_, c, ik_name, dist) in LIMBS:
    new_bone(f'{ik_name}_{s}', None, None, ROOT, copy=f'{c}_{s}')
    pa, pb_, pc = (eb[f'{n}_{s}'].head.copy() for n in (a, b_, c))
    t = (pb_ - pa).dot(pc - pa) / (pc - pa).length_squared
    off = pb_ - (pa + (pc - pa) * t)
    if off.length < 1e-4:
        off = FWD if limb == 'knee' else -FWD
    pole = pb_ + off.normalized() * dist
    POLES[f'pole_{limb}_{s}'] = pole
    parent = f'IK_foot_{s}' if limb == 'knee' else ROOT
    new_bone(f'pole_{limb}_{s}', pole, pole + UP * 0.05, parent, roll_to=FWD)
    # The imported bones' tails point anywhere, and Blender's IK solves along head-to-tail, so it
    # runs on a clean chain (IKC-*); MCH-* copies of the deform bones ride that chain and the deform
    # bones copy their transforms (same rest pose, so the skinning is unchanged).
    up_ = new_bone(f'IKC-{a}_{s}', pa, pb_, eb[f'{a}_{s}'].parent.name)
    up_.align_roll(off)
    lo_ = new_bone(f'IKC-{b_}_{s}', pb_, pc, up_.name)
    lo_.align_roll(off)
    # Game clips key bone translations too (longer limbs, shifted hands), so the chain is not connected:
    # each IKC bone takes its deform bone's location offset (drivers below). IKC-<end> is a tiny locked bone
    # carrying the end's offset; its tail is the IK effector, aimed at IKE-<end> which sits on the IK control.
    eps = (pc - pb_).normalized() * 0.0001
    en_ = new_bone(f'IKC-{c}_{s}', pc, pc + eps, lo_.name)
    en_.align_roll(off)
    new_bone(f'IKE-{c}_{s}', pc + eps, pc + eps + UP * 0.01, f'{ik_name}_{s}', roll_to=FWD)
    # Tweaks copy the IK chain's local transforms and add their own rotation on top, so the upper/lower
    # bones can still be rotated by hand in IK mode (the lower one follows the upper's tweak).
    tu = new_bone(f'IK_tweak_{a}_{s}', None, None, up_.parent.name, copy=up_.name)
    tl = new_bone(f'IK_tweak_{b_}_{s}', None, None, tu.name, copy=lo_.name)
    te = new_bone(f'IKT-{c}_{s}', None, None, tl.name, copy=en_.name)
    new_bone(f'MCH-{a}_{s}', None, None, tu.name, copy=f'{a}_{s}')
    new_bone(f'MCH-{b_}_{s}', None, None, tl.name, copy=f'{b_}_{s}')
    new_bone(f'MCH-{c}_{s}', None, None, te.name, copy=f'{c}_{s}')

# Twist bones, as the game's AnimConstraintTiltTwist: (twist bone, weight, bone whose twist is read,
# frame bone, limb start, limb end). TW-ref rides the frame bone along the limb, TW-src the source bone
# with the same rest, so TW-src seen from TW-ref is the source's rotation in a limb-aligned frame.
TWISTS = []
for s in 'LR':
    TWISTS += [
        (f'arm_upper_{s}_TWIST', -1.0, f'arm_upper_{s}', f'clavicle_{s}', f'arm_upper_{s}', f'arm_lower_{s}'),
        (f'arm_upper_{s}_TWIST1', -0.5, f'arm_upper_{s}', f'clavicle_{s}', f'arm_upper_{s}', f'arm_lower_{s}'),
        (f'arm_lower_{s}_TWIST', 0.5, f'hand_{s}', f'arm_lower_{s}', f'arm_lower_{s}', f'hand_{s}'),
        (f'arm_lower_{s}_TWIST1', 1.0, f'hand_{s}', f'arm_lower_{s}', f'arm_lower_{s}', f'hand_{s}'),
        # used when the vmdl has no TiltTwist for the bone: Violet's thigh weights fitted to her baked clips
        (f'leg_upper_{s}_TWIST', -0.75, f'leg_upper_{s}', 'pelvis', f'leg_upper_{s}', f'leg_lower_{s}'),
        (f'leg_upper_{s}_TWIST1', 0.5 if s == 'L' else -0.5, f'leg_upper_{s}', 'pelvis', f'leg_upper_{s}',
         f'leg_lower_{s}'),
    ]
TWISTS = [t for t in TWISTS if all(n in eb for n in (t[0],) + t[2:])]
GAME_TWIST = {t['bone']: t['weight'] for t in GAMERIG.get('twists', [])}
TWISTS = [(tw, GAME_TWIST.get(tw, w), *rest_) for tw, w, *rest_ in TWISTS]
for tw, w, src, frame, a, b_ in TWISTS:
    key_ = f'{src}'
    if f'TW-ref-{key_}' in eb:
        continue
    pa, pb_ = eb[a].head.copy(), eb[b_].head.copy()
    r = new_bone(f'TW-ref-{key_}', pa, pb_, frame, roll_to=FWD)
    new_bone(f'TW-fol-{key_}', None, None, src, copy=r.name)
    new_bone(f'TW-src-{key_}', None, None, r.name, copy=r.name)

# The prop chain: CHC-* is a clean IK chain (head to next head) whose pose follows the deform chain's own keys through
# drivers, so a clip's chain shape is IK's starting point; IK then only bends it to keep the end on ANC-* (on the prop).
# The chain is not connected because clips also key the chain bones' locations.
for prop, idle in PROP_CHAINS:
    chain, end = prop['chain'], prop['chain_end']
    prev = eb[chain[0]].parent.name
    for k, n in enumerate(chain):
        nxt = chain[k + 1] if k + 1 < len(chain) else end
        cb_ = new_bone('CHC-' + n, eb[n].head.copy(), eb[nxt].head.copy(), prev, roll_to=UP)
        new_bone('MCH-' + n, None, None, cb_.name, copy=n)
        prev = cb_.name
    # the end's own keyed offset varies between clips, but where it lands on the prop does not: CHC-<end> is a tiny
    # locked bone carrying that offset, and its tail is the IK effector
    eps = (eb[end].head - eb[chain[-1]].head).normalized() * 0.0001
    new_bone('CHC-' + end, eb[end].head.copy(), eb[end].head + eps, prev, roll_to=UP)
    at = (rest[prop['anchor']] @ idle[prop['anchor']].inverted() @ idle[end]).translation
    new_bone('ANC-' + end, at, at + UP * 0.02, prop['anchor'], roll_to=FWD)

PHYS_CHAINS = []
for c in GAMERIG['chains']:
    js = c['joints']
    if any(j['bone'] in PROP_CHAIN_BONES for j in js):
        print('cloth chain', c['name'], 'is held by its prop instead of simulated')
        continue
    if not all(j['bone'] in eb for j in js) or \
            any(eb[j1['bone']].parent.name != j0['bone'] for j0, j1 in zip(js, js[1:])):
        print('skipping cloth chain', c['name'], [j['bone'] for j in js])
        continue
    PHYS_CHAINS.append(c)
    for j in js[:-1]:
        new_bone('PHYS-' + j['bone'], None, None, None, copy=j['bone'])
bpy.ops.object.mode_set(mode='POSE')
rest = {b.name: b.matrix_local.copy() for b in B}

# collections, colours

for c in list(arm.data.collections_all):
    arm.data.collections.remove(c)
colls = {n: arm.data.collections.new(n) for n in COLL_ORDER}
for b in B:
    n = b.name
    if n.startswith(('CTRL_', FACE)):
        cat = 'Face'
    elif n.startswith(('IK_', 'pole_')):
        cat = 'IK'
    elif n.startswith(('IKC-', 'IKE-', 'IKT-', 'MCH-', 'TW-', 'PHYS-', 'CHC-', 'ANC-')):
        cat = 'Helpers'
    elif n in {p['bone'] for p in PROPS}:
        cat = 'Props'
    else:
        cat = category(n)
    colls[cat].assign(b)
    pal = {'Face': 'THEME03', 'Secondary': 'THEME06', 'Props': 'THEME07'}.get(cat, COLOURS[side(n)])
    b.color.palette = pal
    arm.pose.bones[n].color.palette = pal
for n in HIDDEN_BY_DEFAULT:
    colls[n].is_visible = False
arm.data.display_type = 'STICK'
arm.show_in_front = True

# widgets: built in armature space, stored in bone space with bone size off

wcoll = bpy.data.collections.new('Widgets')
bpy.context.scene.collection.children.link(wcoll)
wcoll.hide_viewport = True
wcoll.hide_render = True
bpy.context.view_layer.layer_collection.children[wcoll.name].exclude = True
for c in bpy.data.collections:
    if c.name.startswith('glTF_not_exported'):
        bpy.data.collections.remove(c)


def circle(c, normal, r, n=24):
    normal = normal.normalized()
    a = normal.orthogonal().normalized()
    b = normal.cross(a)
    pts = [c + (a * math.cos(2 * math.pi * i / n) + b * math.sin(2 * math.pi * i / n)) * r for i in range(n)]
    return pts, [(i, (i + 1) % n) for i in range(n)]


def box(c, half):
    pts = [c + Vector((sx * half.x, sy * half.y, sz * half.z)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return pts, e


def diamond(c, r):
    pts = [c + v * r for v in (Vector((1, 0, 0)), Vector((-1, 0, 0)), Vector((0, 1, 0)), Vector((0, -1, 0)),
                               Vector((0, 0, 1)), Vector((0, 0, -1)))]
    e = [(a, b) for a in (0, 1) for b in (2, 3, 4, 5)] + [(2, 4), (2, 5), (3, 4), (3, 5)]
    return pts, e


def widget(bone, shape, width=1.5):
    pts, edges = shape
    inv = arm.data.bones[bone].matrix_local.inverted()
    me = bpy.data.meshes.new('WGT-' + bone)
    me.from_pydata([inv @ p for p in pts], edges, [])
    ob = bpy.data.objects.new('WGT-' + bone, me)
    wcoll.objects.link(ob)
    pb = arm.pose.bones[bone]
    pb.custom_shape = ob
    pb.use_custom_shape_bone_size = False
    pb.custom_shape_wire_width = width
    # the glTF importer leaves a small display scale on its bones
    pb.custom_shape_scale_xyz = (1, 1, 1)
    pb.custom_shape_translation = (0, 0, 0)
    pb.custom_shape_rotation_euler = (0, 0, 0)


NEXT = {'pelvis': 'spine_0', 'spine_0': 'spine_1', 'spine_1': 'spine_2', 'spine_2': 'spine_3', 'spine_3': 'neck_0',
        'neck_0': 'head'}
for s in 'LR':
    NEXT.update({f'clavicle_{s}': f'arm_upper_{s}', f'arm_upper_{s}': f'arm_lower_{s}', f'arm_lower_{s}': f'hand_{s}',
                 f'hand_{s}': f'finger_middle_0_{s}', f'leg_upper_{s}': f'leg_lower_{s}',
                 f'leg_lower_{s}': f'ankle_{s}', f'ankle_{s}': f'ball_{s}'})
RADIUS = {'pelvis': 0.17, 'spine': 0.15, 'neck_0': 0.06, 'head': 0.12, 'clavicle': 0.045, 'arm_upper': 0.055,
          'arm_lower': 0.045, 'hand': 0.045, 'leg_upper': 0.085, 'leg_lower': 0.065, 'ankle': 0.05, 'ball': 0.045,
          'finger': 0.009}


RADIUS = {k: v * SCALE for k, v in RADIUS.items()}


def seg_end(b):
    if NEXT.get(b.name) in hpos:
        return hpos[NEXT[b.name]]
    kids = [c for c in b.children if c.name in hpos and (hpos[c.name] - hpos[b.name]).length > 1e-3]
    if len(kids) == 1:
        return hpos[kids[0].name]
    if b.name == 'head':
        return hpos['head'] + UP * 0.2
    d = hpos[b.name] - hpos[b.parent.name] if b.parent else UP
    if d.length < 1e-4:
        d = UP
    return hpos[b.name] + d.normalized() * (0.08 if b.name.startswith('ball') else 0.03)


for b in B:
    n = b.name
    if n.startswith(('CTRL_', 'FACE', 'IK_', 'pole_', 'IKC-', 'IKE-', 'IKT-', 'MCH-', 'TW-', 'PHYS-', 'CHC-', 'ANC-')):
        continue
    cat = category(n)
    if cat in HIDDEN:
        continue
    if n == 'root_motion':
        widget(n, circle(Vector((0, 0, 0.002)), UP, 0.55), 2.5)
        continue
    end = seg_end(b)
    d = end - hpos[n]
    r = next((v for k, v in RADIUS.items() if n.startswith(k)), 0.018 * SCALE)
    if n == 'pelvis':
        widget(n, circle(hpos[n], UP, r), 2.5)
    elif n == 'head':
        widget(n, circle(hpos[n] + UP * 0.1, UP, r), 2)
    else:
        widget(n, circle(hpos[n] + d * 0.5, d, r), 2 if cat != 'Secondary' else 1.2)

for s, (lname, limb, a_, b_, c_, ik_name, _dist) in LIMBS:
    if lname == 'leg':
        a = hpos[f'ankle_{s}']
        ball = hpos.get(f'ball_{s}', a + FWD * 0.12 * SCALE)
        toe_y, heel_y = ball.y - 0.09 * SCALE, a.y + 0.07 * SCALE
        widget(f'IK_foot_{s}', box(Vector((ball.x, (toe_y + heel_y) / 2, 0.012 * SCALE)),
                                   Vector((0.055 * SCALE, (heel_y - toe_y) / 2, 0.012 * SCALE))), 2.5)
    else:
        h = hpos[f'hand_{s}']
        knuckle = hpos.get(f'finger_middle_0_{s}', h + (h - hpos[f'arm_lower_{s}']).normalized() * 0.09 * SCALE)
        widget(f'IK_hand_{s}', box(h + (knuckle - h) * 0.4, Vector((0.05, 0.05, 0.05)) * SCALE), 2.5)
    widget(f'pole_{limb}_{s}', diamond(POLES[f'pole_{limb}_{s}'], 0.03 * SCALE), 2)
    for a, b2 in ((a_, b_), (b_, c_)):
        p0, p1 = hpos[f'{a}_{s}'], hpos[f'{b2}_{s}']
        widget(f'IK_tweak_{a}_{s}', circle(p0.lerp(p1, 0.5), p1 - p0, RADIUS[a] * 1.15), 2)
def build_face():
    """Face control widgets, limits, the FACE slider bone and the shape key drivers."""
    for name, (pos, r, *_rest) in CTRLS.items():
        widget(name, circle(pos, FWD, r), 2.5)
    e1, e2 = EYE_TARGET + Vector((EYE_SPACING, 0, 0)), EYE_TARGET - Vector((EYE_SPACING, 0, 0))
    p1, ed1 = circle(e1, FWD, 0.03)
    p2, ed2 = circle(e2, FWD, 0.03)
    widget('CTRL_eyes', (p1 + p2, ed1 + [(a + len(p1), b + len(p1)) for a, b in ed2]), 2.5)
    widget(FACE, box(FACE_BONE, Vector((0.04, 0.002, 0.025))), 2)

    # face controls: limits, locks, raw sliders, drivers

    for name, (pos, r, tx, ty, fwd, scl) in CTRLS.items():
        pb = arm.pose.bones[name]
        pb.lock_rotation = (True, True, True)
        pb.lock_rotation_w = True
        pb.lock_location = (tx == 0, ty == 0, not fwd)
        pb.lock_scale = (not scl,) * 3
        c = pb.constraints.new('LIMIT_LOCATION')
        c.owner_space = 'LOCAL'
        c.use_transform_limit = True
        for ax, t in (('x', tx), ('y', ty), ('z', 0.01 if fwd else 0)):
            setattr(c, f'use_min_{ax}', True)
            setattr(c, f'use_max_{ax}', True)
            setattr(c, f'min_{ax}', -t)
            setattr(c, f'max_{ax}', t)
        if scl:
            c = pb.constraints.new('LIMIT_SCALE')
            c.owner_space = 'LOCAL'
            c.use_transform_limit = True
            for ax in 'xyz':
                setattr(c, f'use_min_{ax}', True)
                setattr(c, f'use_max_{ax}', True)
                setattr(c, f'min_{ax}', 0.5)
                setattr(c, f'max_{ax}', 1.5)
    jaw = arm.pose.bones['CTRL_jaw']
    jaw['seal'] = 0.0
    jaw.id_properties_ui('seal').update(min=0, max=1, soft_min=0, soft_max=1,
                                        description='Keep the lips together while the jaw opens (mouthClose)')
    eyes = arm.pose.bones['CTRL_eyes']
    eyes.lock_rotation = (True, True, True)
    eyes.lock_rotation_w = True
    eyes.lock_scale = (True, True, True)
    fb = arm.pose.bones[FACE]
    fb.lock_location = fb.lock_rotation = fb.lock_scale = (True, True, True)
    fb.lock_rotation_w = True
    for k in ARKIT:
        fb[k] = 0.0
        fb.id_properties_ui(k).update(min=0, max=1, soft_min=0, soft_max=1)

    # control bones point up with z forward: local x = her left, y = up, z = forward
    key = head_ob.data.shape_keys
    for k, tl in terms().items():
        fc = key.driver_add(f'key_blocks["{k}"].value')
        d = fc.driver
        d.type = 'SCRIPTED'
        v = d.variables.new()
        v.name = 'raw'
        v.type = 'SINGLE_PROP'
        v.targets[0].id_type = 'OBJECT'
        v.targets[0].id = arm
        v.targets[0].data_path = f'pose.bones["{FACE}"]["{k}"]'
        parts = []
        for i, (bone, ch, sign, travel) in enumerate(tl):
            v = d.variables.new()
            v.name = f'c{i}'
            v.type = 'TRANSFORMS'
            v.targets[0].id = arm
            v.targets[0].bone_target = bone
            v.targets[0].transform_type = ch
            v.targets[0].transform_space = 'LOCAL_SPACE'
            x = f'(c{i} - 1)' if ch == 'SCALE_AVG' else f'c{i}'
            parts.append(f'max(0, {"-" if sign < 0 else ""}{x} / {travel})')
        if k == 'mouthClose':
            v = d.variables.new()
            v.name = 'seal'
            v.type = 'SINGLE_PROP'
            v.targets[0].id_type = 'OBJECT'
            v.targets[0].id = arm
            v.targets[0].data_path = 'pose.bones["CTRL_jaw"]["seal"]'
            v = d.variables.new()
            v.name = 'jy'
            v.type = 'TRANSFORMS'
            v.targets[0].id = arm
            v.targets[0].bone_target = 'CTRL_jaw'
            v.targets[0].transform_type = 'LOC_Y'
            v.targets[0].transform_space = 'LOCAL_SPACE'
            v = d.variables.new()
            v.name = 'rj'
            v.type = 'SINGLE_PROP'
            v.targets[0].id_type = 'OBJECT'
            v.targets[0].id = arm
            v.targets[0].data_path = f'pose.bones["{FACE}"]["jawOpen"]'
            parts.append('seal * min(1, rj + max(0, -jy / 0.025))')
        d.expression = 'clamp(' + ' + '.join(['raw'] + parts) + ', 0, 1)'
        assert d.is_simple_expression, d.expression
        kb = key.key_blocks[k]
        kb.slider_max = 1


if HAS_FACE:
    build_face()

# IK

for p in [f'ik_{d[0]}_{s}' for s, d in LIMBS]:
    arm[p] = 1.0
    arm.id_properties_ui(p).update(min=0, max=1, soft_min=0, soft_max=1, description='0 = FK, 1 = IK')


def influence_driver(con, prop):
    d = con.driver_add('influence').driver
    d.type = 'AVERAGE'
    v = d.variables.new()
    v.type = 'SINGLE_PROP'
    v.targets[0].id_type = 'OBJECT'
    v.targets[0].id = arm
    v.targets[0].data_path = f'["{prop}"]'


def chain_error(names):
    bpy.context.view_layer.update()
    return sum(arm.pose.bones[n].matrix.to_quaternion().rotation_difference(rest[n].to_quaternion()).angle
               for n in names)


for s, (lname, limb, a_, b_, c_, ik_name, _dist) in LIMBS:
    end, upper, lower, target, pole, prop = (f'{c_}_{s}', f'{a_}_{s}', f'{b_}_{s}', f'{ik_name}_{s}', f'pole_{limb}_{s}',
                                             f'ik_{lname}_{s}')
    eff = arm.pose.bones['IKC-' + end]
    eff.lock_ik_x = eff.lock_ik_y = eff.lock_ik_z = True
    ik = eff.constraints.new('IK')
    ik.target = arm
    ik.subtarget = 'IKE-' + end
    ik.pole_target = arm
    ik.pole_subtarget = pole
    ik.chain_count = 3
    chain = ['IKC-' + upper, 'IKC-' + lower]
    best = min(range(-180, 180, 2), key=lambda a: (setattr(ik, 'pole_angle', math.radians(a)), chain_error(chain))[1])
    fine = min((best + x * 0.25 for x in range(-8, 9)), key=lambda a: (setattr(ik, 'pole_angle', math.radians(a)), chain_error(chain))[1])
    ik.pole_angle = math.radians(fine)
    err = chain_error(chain)
    print(f'IK {end}: pole angle {fine:.2f} deg, rest error {math.degrees(err):.3f} deg')
    if err > math.radians(1.0):
        print(f'WARNING: IK {end} does not reproduce the rest pose')
    for n in (upper, lower):
        tw = arm.pose.bones['IK_tweak_' + n]
        c = tw.constraints.new('COPY_TRANSFORMS')
        c.name = 'IK'
        c.target = arm
        c.subtarget = 'IKC-' + n
        c.target_space = c.owner_space = 'LOCAL'
        c.mix_mode = 'BEFORE_FULL'
        tw.lock_location = tw.lock_scale = (True, True, True)
        c = arm.pose.bones[n].constraints.new('COPY_TRANSFORMS')
        c.name = 'IK'
        c.target = arm
        c.subtarget = 'MCH-' + n
        influence_driver(c, prop)
    cr = arm.pose.bones[end].constraints.new('COPY_ROTATION')
    cr.name = 'IK rotation'
    cr.target = arm
    cr.subtarget = target
    influence_driver(cr, prop)
    # the end bone's own location offset is already in the chain (IKC-<end>), so it sits on the chain end
    cl = arm.pose.bones[end].constraints.new('COPY_LOCATION')
    cl.name = 'IK location'
    cl.target = arm
    cl.subtarget = 'MCH-' + end
    influence_driver(cl, prop)
    for n in (upper, lower, end):
        to_ikc = rest['IKC-' + n].to_3x3().inverted() @ rest[n].to_3x3()
        for i in range(3):
            d = arm.pose.bones['IKC-' + n].driver_add('location', i).driver
            d.type = 'SCRIPTED'
            for k in range(3):
                v = d.variables.new()
                v.name = 'xyz'[k]
                v.type = 'SINGLE_PROP'
                v.targets[0].id_type = 'OBJECT'
                v.targets[0].id = arm
                v.targets[0].data_path = f'pose.bones["{n}"].location[{k}]'
            d.expression = ' + '.join(f'{to_ikc[i][k]:.9f} * {"xyz"[k]}' for k in range(3))
    c = arm.pose.bones['IKT-' + end].constraints.new('COPY_TRANSFORMS')
    c.target = arm
    c.subtarget = 'IKC-' + end
    c.target_space = c.owner_space = 'LOCAL'

# only the active mode's limb controls are shown; pose bone visibility is on Bone before 5.0
POSE_HIDE = 'hide' in bpy.types.PoseBone.bl_rna.properties
for s, (lname, limb, a_, b_, c_, ik_name, _dist) in LIMBS:
    fk = (a_, b_, c_)
    ik_bones = (ik_name, f'pole_{limb}', f'IK_tweak_{a_}', f'IK_tweak_{b_}')
    for names, expr in ((fk, 'ik > 0.5'), (ik_bones, 'ik < 0.5')):
        for n in names:
            d = (arm.driver_add(f'pose.bones["{n}_{s}"].hide') if POSE_HIDE else
                 arm.data.driver_add(f'bones["{n}_{s}"].hide')).driver
            d.type = 'SCRIPTED'
            v = d.variables.new()
            v.name = 'ik'
            v.type = 'SINGLE_PROP'
            v.targets[0].id_type = 'OBJECT'
            v.targets[0].id = arm
            v.targets[0].data_path = f'["ik_{lname}_{s}"]'
            d.expression = expr

# twist constraints

# TW-src (child of TW-ref) copies TW-fol, so its local rotation is the source bone's rotation relative to
# the frame bone, in a limb-aligned frame. The twist is applied about the limb axis of the twist bone's own
# parent (upper arm/thigh: the source itself; forearm: the frame bone), through a custom space.
for src in {t[2] for t in TWISTS}:
    c = arm.pose.bones[f'TW-src-{src}'].constraints.new('COPY_TRANSFORMS')
    c.target = arm
    c.subtarget = f'TW-fol-{src}'
for tw, w, src, frame, a, b_ in TWISTS:
    tw_parent = arm.data.bones[tw].parent.name
    if tw_parent not in (src, frame):
        print('skipping twist bone with an unexpected parent', tw, tw_parent)
        continue
    c = arm.pose.bones[tw].constraints.new('TRANSFORM')
    c.name = 'Twist'
    c.target = arm
    c.subtarget = f'TW-src-{src}'
    c.space_object = arm
    c.space_subtarget = f'TW-fol-{src}' if tw_parent == src else f'TW-ref-{src}'
    c.target_space = 'LOCAL'
    c.owner_space = 'CUSTOM'
    c.map_from = 'ROTATION'
    c.from_rotation_mode = 'SWING_TWIST_Y'
    c.from_min_y_rot, c.from_max_y_rot = -math.pi, math.pi
    c.map_to = 'ROTATION'
    c.map_to_x_from, c.map_to_y_from, c.map_to_z_from = 'X', 'Y', 'Z'
    c.to_min_y_rot, c.to_max_y_rot = -math.pi * w, math.pi * w
    c.mix_mode_rot = 'BEFORE'
    c.to_euler_order = 'YXZ'

def fit_twists():
    """Twist bones with a TiltTwist constraint in the vmdl use its weight: the game evaluates it at runtime, over
    whatever the clips have baked (which for some heroes is not twisted at all). The others get the weight that
    best reproduces the twist baked into a few of the hero's clips."""
    tws = [t[0] for t in TWISTS if 'Twist' in arm.pose.bones[t[0]].constraints and t[0] not in GAME_TWIST]
    for t in TWISTS:
        if t[0] in GAME_TWIST:
            print(f'twist {t[0]}: weight {t[1]:+.3f} from the game')
    if not tws or not os.path.exists(hero.anims_blend):
        return
    with bpy.data.libraries.load(hero.anims_blend) as (src, dst):
        pref = ['game_out_of_combat_run_n', 'game_ui_hero_select', 'game_out_of_combat_stand_idle', 'game_jump_ground',
                'game_out_of_combat_sprint_n', 'game_primary_run_n', 'game_weapon_run_n', 'game_item_run_n']
        dst.actions = [n for n in pref if n in src.actions] or sorted(src.actions)[:4]
    clips = list(dst.actions)
    pbs = arm.pose.bones
    mods = [m for o in bpy.data.objects if o.type == 'MESH' for m in o.modifiers if m.type == 'ARMATURE' and m.show_viewport]
    for m in mods:
        m.show_viewport = False
    props = {p: arm[p] for p in arm.keys() if p.startswith('ik_')}
    for p in props:
        arm[p] = 0.0
    arm.animation_data_create()
    sc = bpy.context.scene
    samples = []
    for a in clips:
        f0, f1 = a.frame_range
        cb = a.layers[0].strips[0].channelbags[0]
        keyed = {fc.data_path.split('"')[1] for fc in cb.fcurves if '_TWIST' in fc.data_path}
        for k in range(6):
            samples.append((a, f0 + (f1 - f0) * k / 5, keyed))
    game = []
    for a, f, keyed in samples:
        arm.animation_data.action = a
        arm.animation_data.action_slot = a.slots[0]
        for b in tws:
            pbs[b].constraints['Twist'].mute = True
        sc.frame_set(int(f), subframe=f % 1)
        game.append({b: pbs[b].matrix.to_quaternion() for b in tws if b in keyed})
    for a in clips:
        for fc in a.layers[0].strips[0].channelbags[0].fcurves:
            if '_TWIST' in fc.data_path:
                fc.mute = True
    for b in tws:
        pbs[b].constraints['Twist'].mute = False
        pbs[b].rotation_quaternion = (1, 0, 0, 0)
        pbs[b].location = (0, 0, 0)
    weights = sorted({x / 8 for x in range(-12, 13)} | {t[1] for t in TWISTS})
    err = {b: [0.0] * len(weights) for b in tws}
    for wi, w in enumerate(weights):
        for b in tws:
            c = pbs[b].constraints['Twist']
            c.to_min_y_rot, c.to_max_y_rot = -math.pi * w, math.pi * w
        for (a, f, _), g in zip(samples, game):
            arm.animation_data.action = a
            arm.animation_data.action_slot = a.slots[0]
            sc.frame_set(int(f), subframe=f % 1)
            for b, q in g.items():
                d = q.rotation_difference(pbs[b].matrix.to_quaternion()).angle
                err[b][wi] += min(d, 2 * math.pi - d)
    base = {t[0]: t[1] for t in TWISTS}
    fitted = []
    for b in tws:
        n = sum(b in g for g in game)
        c = pbs[b].constraints['Twist']
        w = base[b]
        msg = 'no baked twist in the clips'
        if n:
            w = weights[min(range(len(weights)), key=lambda i: err[b][i])]
            msg = f'mean error {math.degrees(err[b][weights.index(w)] / n):.2f} deg vs baked clips'
            fitted.append(b)
        c.to_min_y_rot, c.to_max_y_rot = -math.pi * w, math.pi * w
        print(f'twist {b}: weight {w:+.3f} fitted, {msg}')
    arm.animation_data.action = None
    for a in clips:
        bpy.data.actions.remove(a)
    for p, v in props.items():
        arm[p] = v
    for b in pbs:
        b.location = (0, 0, 0)
        b.rotation_quaternion = (1, 0, 0, 0)
        b.rotation_euler = (0, 0, 0)
        b.scale = (1, 1, 1)
    for m in mods:
        m.show_viewport = True
    sc.frame_set(0)
    arm['twist_fitted'] = fitted


fit_twists()


def prop_widget(bone):
    """Box around the rest-pose vertices skinned to the prop's bones."""
    names = set(descendants(bone)) - PROP_CHAIN_BONES
    pts = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_viewport or o.name.startswith('WGT-'):
            continue
        groups = {g.index for g in o.vertex_groups if g.name in names}
        if groups:
            pts += [o.matrix_world @ v.co for v in o.data.vertices if any(g.group in groups and g.weight > 0.3 for g in v.groups)]
    if not pts:
        return
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    widget(bone, box((lo + hi) / 2, (hi - lo) / 2 + Vector((0.01, 0.01, 0.01))), 2.5)


for prop in PROPS:
    prop_widget(prop['bone'])
if PROP_CHAINS:
    arm['prop_chain'] = 1.0
    arm.id_properties_ui('prop_chain').update(min=0, max=1, soft_min=0, soft_max=1,
                                              description='Keep the chain attached to the held prop')
for prop, _ in PROP_CHAINS:
    chain = prop['chain']
    for n in chain:
        pbn = arm.pose.bones[n]
        c = pbn.constraints.new('COPY_TRANSFORMS')
        c.name = 'Prop chain'
        c.target = arm
        c.subtarget = 'MCH-' + n
        influence_driver(c, 'prop_chain')
        pbn.constraints.move(len(pbn.constraints) - 1, 0)
    for n in chain + [prop['chain_end']]:
        # CHC basis = O^-1 @ deform basis @ O, with O the (rotation-only) rest offset between the two bones
        o = (rest[n].to_3x3().inverted() @ rest['CHC-' + n].to_3x3()).to_quaternion()
        oi = o.inverted()
        # columns: the conjugated quaternion for each unit input component
        basis = [oi @ Quaternion(e) @ o for e in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))]
        to_chc = rest['CHC-' + n].to_3x3().inverted() @ rest[n].to_3x3()
        for prop_name, size, coef in (('rotation_quaternion', 4, lambda i, k: basis[k][i]),
                                      ('location', 3, lambda i, k: to_chc[i][k])):
            for i in range(size):
                d = arm.pose.bones['CHC-' + n].driver_add(prop_name, i).driver
                d.type = 'SCRIPTED'
                names = 'wxyz' if size == 4 else 'xyz'
                for k in range(size):
                    v = d.variables.new()
                    v.name = names[k]
                    v.type = 'SINGLE_PROP'
                    v.targets[0].id_type = 'OBJECT'
                    v.targets[0].id = arm
                    v.targets[0].data_path = f'pose.bones["{n}"].{prop_name}[{k}]'
                d.expression = ' + '.join(f'{coef(i, k):.9f} * {names[k]}' for k in range(size))
    eff = arm.pose.bones['CHC-' + prop['chain_end']]
    eff.lock_ik_x = eff.lock_ik_y = eff.lock_ik_z = True
    ik = eff.constraints.new('IK')
    ik.target = arm
    ik.subtarget = 'ANC-' + prop['chain_end']
    ik.chain_count = len(chain) + 1
arm['prop_pose'] = PROP_POSE
for b, v in PROP_POSE.items():
    pbn = arm.pose.bones[b]
    pbn.location, pbn.rotation_quaternion, pbn.scale = v[0:3], v[3:7], v[7:10]

# physics: each simulated bone gets its baked offset from PHYS-<bone> on top of the animation

arm['physics'] = 1.0
arm.id_properties_ui('physics').update(min=0, max=1, soft_min=0, soft_max=1,
                                       description='Mix of the baked secondary motion (hair, cloth)')
phys_cfg = {'chains': [], 'capsules': []}
for c in PHYS_CHAINS:
    js = c['joints']
    for j in js[:-1]:
        pb = arm.pose.bones[j['bone']]
        con = pb.constraints.new('COPY_ROTATION')
        con.name = 'Physics'
        con.target = arm
        con.subtarget = 'PHYS-' + j['bone']
        con.target_space = con.owner_space = 'LOCAL'
        con.mix_mode = 'AFTER'
        influence_driver(con, 'physics')
        arm.pose.bones['PHYS-' + j['bone']].rotation_mode = 'QUATERNION'
    phys_cfg['chains'].append({'name': c['name'], 'joints': js})
for cap in GAMERIG['capsules']:
    if cap['bone'] not in rest:
        continue
    inv = rest[cap['bone']].inverted()
    phys_cfg['capsules'].append({'bone': cap['bone'], 'p0': list(inv @ Vector(cap['p0'])),
                                 'p1': list(inv @ Vector(cap['p1'])), 'r0': cap['r0'], 'r1': cap['r1']})
txt = bpy.data.texts.new('dlrig_physics.json')
txt.write(json.dumps(phys_cfg))

HAND_POSES, reports = hands.hand_poses(arm, [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_viewport])
for r in reports:
    print('hand', *r)

# pose assets

cat_file = os.path.join(os.path.dirname(os.path.abspath(out)), 'blender_assets.cats.txt')
cats = {n: str(uuid.uuid5(uuid.NAMESPACE_URL, f'{hero.slug}/{n}')) for n in ('Face', 'Mouth Shapes', 'Hands')}
with open(cat_file, 'w') as f:
    f.write('VERSION 1\n\n')
    for n, u in cats.items():
        f.write(f'{u}:{hero.name}/{n}:{hero.name}-{n}\n')
arm.animation_data_create()


def ensure_fcurve(a, path, index, group):
    # fcurve_ensure_for_datablock() has no group_name before Blender 4.5
    fc = a.fcurve_ensure_for_datablock(arm, path, index=index)
    cb = a.layers[0].strips[0].channelbags[0]
    fc.group = cb.groups.get(group) or cb.groups.new(group)
    return fc


def pose_asset(name, catalog, props=None, rots=None):
    a = bpy.data.actions.new(name)
    arm.animation_data.action = a
    for k in ARKIT if props is not None else ():
        fc = ensure_fcurve(a, f'pose.bones["{FACE}"]["{k}"]', 0, FACE)
        fc.keyframe_points.insert(1, props.get(k, 0.0))
    for bone, q in (rots or {}).items():
        for i, v in enumerate(q):
            fc = ensure_fcurve(a, f'pose.bones["{bone}"].rotation_quaternion', i, bone)
            fc.keyframe_points.insert(1, v)
    arm.animation_data.action = None
    a.asset_mark()
    a.asset_data.catalog_id = cats[catalog]


if HAS_FACE:
    for n, p in EXPRESSIONS.items():
        pose_asset('Face ' + n, 'Face', props=p)
    pose_asset('Face Neutral', 'Face', props={})
    for n, p in VISEMES.items():
        pose_asset('Mouth ' + n, 'Mouth Shapes', props=p)
for n, sides in HAND_POSES.items():
    for s, rots in sides.items():
        pose_asset(f'Hand {n} {s}', 'Hands', rots=rots)

# panel

ui = bpy.data.texts.new('dlrig_ui.py')
header = (f'HERO = {hero.name!r}\nHAS_FACE = {HAS_FACE!r}\nCLOTH_LINKS = {hero.cloth_links!r}\n'
          f'ARKIT = {ARKIT!r}\nGROUPS = {GROUPS!r}\nEXPRESSIONS = {EXPRESSIONS!r}\nVISEMES = {VISEMES!r}\n'
          f'HAND_POSES = {HAND_POSES!r}\n')
with open(os.path.join(os.path.dirname(__file__), 'rig_ui.py')) as f:
    ui.write(header + f.read())
ui.use_module = True
arm['anim_library'] = '//' + os.path.basename(hero.anims_blend)
arm['dlrig'] = hero.slug
arm['rig_scale'] = SCALE
# the timeline otherwise only shows keys of the selected bones in pose mode
bpy.context.scene.show_keys_from_selected_only = False

# show every key in the timeline/dope sheet, not only the selected bones'
for scr in bpy.data.screens:
    for area in scr.areas:
        for sp in area.spaces:
            if sp.type == 'DOPESHEET_EDITOR':
                sp.dopesheet.show_only_selected = False
            if sp.type == 'VIEW_3D':
                sp.show_region_ui = True

bpy.ops.object.mode_set(mode='POSE')
bpy.ops.wm.save_as_mainfile(filepath=out)
print('wrote', out)
