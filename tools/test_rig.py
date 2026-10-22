"""blender -b out/<hero>_rig.blend -P test_rig.py : numeric checks of the rig (no renders).
Prints PASS/FAIL lines and a final 'FAILS [...]'; checks for parts the hero lacks are skipped."""
import json
import math
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

t = bpy.data.texts['dlrig_ui.py']
ns = {'__name__': 'dlrig_ui'}
exec(compile(t.as_string(), t.name, 'exec'), ns)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'dlrig' in o)
sc = bpy.context.scene
bpy.context.view_layer.objects.active = arm
pb = arm.pose.bones
fails = []
LIMB = {'arm': ('arm_upper', 'arm_lower', 'hand', 'IK_hand'), 'leg': ('leg_upper', 'leg_lower', 'ankle', 'IK_foot')}
limbs = [(limb, s) for limb in LIMB for s in 'LR' if f'ik_{limb}_{s}' in arm]
IK_PROPS = [f'ik_{limb}_{s}' for limb, s in limbs]
HELPERS = ('IK_', 'pole_', 'IKC-', 'IKE-', 'IKT-', 'MCH-')


def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail)
    if not ok:
        fails.append(name)


def update():
    arm.update_tag()
    bpy.context.view_layer.update()


def verts(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', a)
    ev.to_mesh_clear()
    return a.reshape(-1, 3)


def reset():
    if arm.animation_data:
        arm.animation_data.action = None
    for b in pb:
        b.location = (0, 0, 0)
        b.rotation_quaternion = (1, 0, 0, 0)
        b.rotation_euler = (0, 0, 0)
        b.scale = (1, 1, 1)
    if 'FACE' in pb:
        for k in pb['FACE'].keys():
            pb['FACE'][k] = 0.0
    for p in IK_PROPS:
        arm[p] = 1.0
    update()


def mats():
    return {b.name: b.matrix.copy() for b in pb}


def names(limb, s):
    return [f'{n}_{s}' for n in LIMB[limb]]


# tails: visible FK rings that bend the skin, and the tip control pulling the tail with IK on
for t in arm.get('tails', []):
    reset()
    chain = [b.name for b in pb if b.name.startswith(t + '_') and b.name[len(t) + 1:].isdigit()]
    tip = next(n for n in (t + '_end', chain[-1]) if n in pb)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.vertex_groups.get(chain[1])]
    # Mirage's tail is on the djinn hand, hidden until his ability
    hidden = [o for o in meshes if o.hide_viewport or o.hide_get()]
    for o in hidden:
        o.hide_viewport = False
        o.hide_set(False)
    update()
    before = [verts(o) for o in meshes]
    pb[chain[1]].rotation_quaternion = Quaternion((1, 0, 0), math.radians(30))
    update()
    skin = max((np.linalg.norm(verts(o) - v, axis=1).max() for o, v in zip(meshes, before)), default=0)
    shown = any(c.is_visible and c.name == 'Tail' for c in arm.data.bones[chain[0]].collections)
    check(f'{t} FK bends the skin', skin > 0.01 and shown and pb[chain[1]].custom_shape is not None,
          f'{len(chain)} bones, {skin * 1000:.0f} mm, collection shown {shown}')
    for o in hidden:
        o.hide_viewport = True
        o.hide_set(True)
    # the tip of an FK curl is in reach: the control put there pulls the straight tail onto it
    reset()
    arm['ik_' + t] = 0.0
    along = arm.data.bones[tip].head_local - arm.data.bones[chain[0]].head_local
    across = along.cross(Vector((0, 0, 1))) if abs(along.normalized().z) < 0.9 else along.cross(Vector((1, 0, 0)))
    for n in chain:
        axis = arm.data.bones[n].matrix_local.to_3x3().inverted() @ across
        pb[n].rotation_quaternion = Quaternion(axis.normalized(), math.radians(60 / len(chain)))
    update()
    goal = pb[tip].head.copy()
    for n in chain:
        pb[n].rotation_quaternion = (1, 0, 0, 0)
    update()
    ctrl = pb['IK_' + t]
    ctrl.matrix = Matrix.Translation(goal) @ ctrl.matrix.to_3x3().to_4x4()
    update()
    fk = (pb[tip].head - goal).length
    arm['ik_' + t] = 1.0
    update()
    gap = (pb[tip].head - goal).length
    check(f'{t} IK reaches its control', gap < 0.002 and fk > 0.002, f'{gap * 1000:.1f} mm off, FK tip {fk * 1000:.0f} mm away')
    # the control rides the FK tip, so with IK on and the control left alone a curl (a clip) plays as FK
    reset()
    arm['ik_' + t] = 1.0
    for n in chain:
        axis = arm.data.bones[n].matrix_local.to_3x3().inverted() @ across
        pb[n].rotation_quaternion = Quaternion(axis.normalized(), math.radians(60 / len(chain)))
    update()
    on = pb[tip].head.copy()
    arm['ik_' + t] = 0.0
    update()
    off = (pb[tip].head - on).length
    check(f'{t} IK left alone keeps the FK pose', off < 0.002, f'{off * 1000:.1f} mm')
    arm['ik_' + t] = 0.0
    ctrl.location = (0, 0, 0)
    update()

clips = sorted(c[0] for c in ns['clip_items'](None, bpy.context))


def find_clip(*cands, contains=None):
    for c in cands:
        if c in clips:
            return c
    return next((c for c in clips if contains and contains in c), None)


RUN = find_clip('game_out_of_combat_run_n', 'game_primary_run_n', 'game_weapon_run_n', contains='run_n')
IDLE = ns['idle_clip'](bpy.context)
POSE = find_clip('game_ui_hero_select', contains='ui_') or IDLE
JUMP = find_clip('game_jump_ground', contains='jump')
print('clips', len(clips), 'run', RUN, 'idle', IDLE, 'pose', POSE, 'jump', JUMP, 'limbs', limbs)

# rest pose untouched by the rig (a held prop's chain would reach for the prop, which sits at the origin at rest)
reset()
if 'prop_chain' in arm:
    arm['prop_chain'] = 0.0
    update()
for ob in [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_viewport and not o.name.startswith('WGT-')]:
    base = np.array([v.co for v in ob.data.vertices])
    if not len(base):
        continue
    drift = np.abs(verts(ob) - base).max() * 1000
    check(f'rest drift {ob.name}', drift < 0.05, f'{drift:.4f} mm')

if 'prop_chain' in arm:
    arm['prop_chain'] = 1.0

# X-mirror and Paste Flipped copy a left bone's pose to the right one as (w, x, -y, -z); some skeletons are not
# symmetric in position, so only the resulting rotations are compared
FLIP_X = Matrix.Diagonal((-1, 1, 1))
for limb, s in limbs:
    if s != 'L' or (limb, 'R') not in limbs:
        continue
    reset()
    arm[f'ik_{limb}_L'] = arm[f'ik_{limb}_R'] = 0.0
    q = Quaternion(Vector((0.3, -0.5, 0.7)).normalized(), 0.8)
    for n in LIMB[limb][:3]:
        pb[n + '_L'].rotation_quaternion = q
        pb[n + '_R'].rotation_quaternion = (q.w, q.x, -q.y, -q.z)
    update()
    err = max((FLIP_X @ pb[n + '_L'].matrix.to_3x3() @ FLIP_X).to_quaternion().rotation_difference(
        pb[n + '_R'].matrix.to_quaternion()).angle for n in LIMB[limb][:3])
    check(f'x-mirror {limb}', err < math.radians(0.1), f'{math.degrees(err):.3f} deg')
reset()

# a held prop's chain stays on the prop when the prop is moved, and matches the game clips' chain
for name, rot in (('object_hand_L', 0.0), ('object_hand_L', 0.8)):
    if name not in pb or 'prop_chain' not in arm:
        break
    end = next(b.name for b in pb if b.name.startswith('ANC-'))
    reset()
    bpy.ops.dlrig.reset_pose()
    m = pb[name].matrix.copy()
    m.translation += Vector((0.05, -0.15, 0.1))
    pb[name].matrix = m
    pb[name].rotation_quaternion = Quaternion((0, 0, 1), rot) @ pb[name].rotation_quaternion
    update()
    gap = (pb[end[4:]].head - pb[end].head).length
    check(f'prop chain follows {name} (turn {rot})', gap < 0.001, f'{gap * 1000:.2f} mm')
if 'prop_chain' in arm and POSE:
    reset()
    bpy.ops.dlrig.game_anim(clip=POSE, mode='NEW')
    chain_bones = [b.name[4:] for b in pb if b.name.startswith('CHC-')]
    sc.frame_set(sc.frame_start)
    update()
    with_ik = {n: pb[n].head.copy() for n in chain_bones}
    arm['prop_chain'] = 0.0
    update()
    diff = max((pb[n].head - p).length for n, p in with_ik.items())
    arm['prop_chain'] = 1.0
    check(f'prop chain matches the clip ({POSE})', diff < 0.001, f'{diff * 1000:.2f} mm')

# IK hands and feet follow their controls with IK on (default)
for limb, s in limbs:
    up, lo, end, ctrl = names(limb, s)
    reset()
    m = pb[ctrl].matrix.copy()
    # a third of the way towards the shoulder/hip and a little forward: always within reach
    goal = m.translation.lerp(pb[up].head, 0.33) + Vector((0, -0.05, 0)) * arm.get('rig_scale', 1.0)
    m.translation = goal
    pb[ctrl].matrix = m
    update()
    err = (pb[end].head - goal).length * 1000
    check(f'IK reach {ctrl}', err < 1, f'{err:.2f} mm')
    rot = pb[end].matrix.to_quaternion().rotation_difference(pb[ctrl].matrix.to_quaternion()).angle
    check(f'IK rotation {ctrl}', math.degrees(rot) < 0.5, f'{math.degrees(rot):.3f} deg')

# FK <-> IK snapping keeps the pose
for limb, s in limbs:
    up, lo, end, ctrl = names(limb, s)
    reset()
    arm[f'ik_{limb}_{s}'] = 0.0
    pb[up].rotation_quaternion = Quaternion((1, 0.3, 0.2, 0.1)).normalized()
    pb[lo].rotation_quaternion = Quaternion((1, 0.0, 0.0, 0.5)).normalized()
    update()
    before = mats()
    bpy.ops.dlrig.snap(limb=limb, side=s, to_ik=True)
    update()
    after = mats()
    worst, wn = max(((before[n].translation - after[n].translation).length, n) for n in before if not n.startswith(HELPERS))
    if worst > 0.001:
        print('  worst bone', wn)
    check(f'snap FK->IK {limb}_{s}', worst < 0.001 and arm[f'ik_{limb}_{s}'] == 1.0, f'{worst * 1000:.2f} mm')
    m = pb[ctrl].matrix.copy()
    m.translation += Vector((0.05, -0.1, 0.1))
    pb[ctrl].matrix = m
    update()
    before = mats()
    bpy.ops.dlrig.snap(limb=limb, side=s, to_ik=False)
    update()
    after = mats()
    worst = max((before[n].translation - after[n].translation).length for n in before if not n.startswith(HELPERS))
    check(f'snap IK->FK {limb}_{s}', worst < 0.001 and arm[f'ik_{limb}_{s}'] == 0.0, f'{worst * 1000:.2f} mm')

# snapping to the mode a limb is already in leaves the pose alone
for limb, s in limbs:
    up, lo, end, ctrl = names(limb, s)
    reset()
    m = pb[ctrl].matrix.copy()
    m.translation = m.translation.lerp(pb[up].head, 0.33)
    pb[ctrl].matrix = m
    update()
    before = pb[end].head.copy()
    r = bpy.ops.dlrig.snap(limb=limb, side=s, to_ik=True)
    update()
    moved = (pb[end].head - before).length
    check(f'snap to IK while in IK {limb}_{s}', r == {'CANCELLED'} and moved < 1e-5, f'{moved * 1000:.2f} mm')

# IK tweaks rotate the limb bones on top of IK, the lower bone follows the upper's tweak
for limb, s in limbs:
    up, lo, end, _ = names(limb, s)
    reset()
    before = mats()
    q = Quaternion((0, 0, 1), math.radians(25))
    pb[f'IK_tweak_{lo}'].rotation_quaternion = q
    update()
    turn = (before[lo].to_quaternion().inverted() @ pb[lo].matrix.to_quaternion()).angle
    turn = math.degrees(min(turn, 2 * math.pi - turn))
    still = (pb[up].matrix.translation - before[up].translation).length + \
        pb[up].matrix.to_quaternion().rotation_difference(before[up].to_quaternion()).angle
    check(f'tweak lower {lo}', abs(turn - 25) < 0.1 and still < 1e-5, f'{turn:.2f} deg')
    reset()
    pb[f'IK_tweak_{up}'].rotation_quaternion = q
    update()
    rel_b = before[up].inverted() @ before[lo]
    rel_a = pb[up].matrix.inverted() @ pb[lo].matrix
    moved = (pb[end].head - before[end].translation).length
    keep = max(abs(x - y) for ra, rb in zip(rel_a, rel_b) for x, y in zip(ra, rb))
    check(f'tweak upper {up} carries {lo}', keep < 1e-4 and moved > 0.02, f'{moved * 1000:.0f} mm')
    tweaked = mats()
    bpy.ops.dlrig.snap(limb=limb, side=s, to_ik=False)
    update()
    worst = max((tweaked[n].translation - pb[n].matrix.translation).length for n in (up, lo, end))
    ok = worst < 0.001 and pb[f'IK_tweak_{up}'].rotation_quaternion.angle < 1e-6
    check(f'snap IK->FK keeps tweak {limb}_{s}', ok, f'{worst * 1000:.2f} mm')

# twist weights fitted to baked clips reproduce them (the game's own TiltTwist weights are taken as they are)
tw = list(arm.get('twist_fitted', []))
lib = bpy.path.abspath(arm.get('anim_library', ''))
for name in [c for c in (RUN, POSE) if c] if tw else []:
    reset()
    with bpy.data.libraries.load(lib) as (src, dst):
        dst.actions = [name]
    clip = dst.actions[0]
    for p in IK_PROPS:
        arm[p] = 0.0
    arm.animation_data_create()
    arm.animation_data.action = clip
    arm.animation_data.action_slot = clip.slots[0]
    cb = clip.layers[0].strips[0].channelbags[0]
    keyed = {fc.data_path.split('"')[1] for fc in cb.fcurves if '_TWIST' in fc.data_path}
    worst, wb = 0, None
    for f in (0, 4, 9, 20):
        sc.frame_set(f)
        for b in tw:
            pb[b].constraints['Twist'].mute = True
        update()
        game = {b: pb[b].matrix.copy() for b in tw}
        saved = []
        for fc in list(cb.fcurves):
            if '_TWIST' in fc.data_path:
                fc.mute = True
                saved.append(fc)
        for b in tw:
            pb[b].constraints['Twist'].mute = False
            pb[b].rotation_quaternion = (1, 0, 0, 0)
            pb[b].location = (0, 0, 0)
        update()
        for b in [b for b in tw if b in keyed]:
            d = game[b].to_quaternion().rotation_difference(pb[b].matrix.to_quaternion()).angle
            d = math.degrees(min(d, 2 * math.pi - d))
            if d > worst:
                worst, wb = d, (b, f)
        for fc in saved:
            fc.mute = False
    check(f'twist matches game ({clip.name})', worst < 10, f'worst {worst:.2f} deg at {wb}')

# game animation loader
if RUN:
    reset()
    bpy.ops.dlrig.game_anim(clip=RUN, mode='NEW')
    act = arm.animation_data.action
    check('game anim loads', act and act.name.startswith(RUN[5:]),
          f'{act.name if act else None}, range {sc.frame_start}-{sc.frame_end}')

# vertex cloth follows the body when posed by hand (its bones are loose in the game's skeleton)
phys = json.loads(bpy.data.texts['dlrig_physics.json'].as_string())
cloth = [nd['name'] for nd in phys.get('cloth', {}).get('nodes', []) if nd['free'] and nd['name'] in pb]
if cloth:
    reset()
    before = {b: pb[b].head.copy() for b in cloth}
    # FK, since with the feet held by IK the legs (Grey Talon's cloth follows them) only partly follow the hips
    for p in IK_PROPS:
        arm[p] = 0.0
    update()
    before = {b: pb[b].head.copy() for b in cloth}
    pb['pelvis'].location.z += 0.1
    update()
    moved = min((pb[b].head - before[b]).length for b in cloth)
    pb['pelvis'].location.z -= 0.1
    for p in IK_PROPS:
        arm[p] = 1.0
    update()
    check('vertex cloth follows the body', moved > 0.02, f'{len(cloth)} nodes, least {moved * 1000:.0f} mm')
    if RUN:
        bpy.ops.dlrig.game_anim(clip=RUN, mode='NEW')

# physics bake on the run
if RUN and (phys['chains'] or cloth):
    t0 = time.time()
    sc.dlrig_phys_loop = True
    r = bpy.ops.dlrig.bake_physics()
    dt = time.time() - t0
    cb = arm.animation_data.action.layers[0].strips[0].channelbags[0]
    n_phys = len([f for f in cb.fcurves if 'PHYS-' in f.data_path])
    check('physics bake', r == {'FINISHED'} and n_phys > 0, f'{n_phys} curves, {dt:.1f}s for {sc.frame_end - sc.frame_start + 1} frames')
    bad = [f.data_path for f in cb.fcurves if 'PHYS-' in f.data_path and any(math.isnan(k.co[1]) for k in f.keyframe_points)]
    check('physics no NaN', not bad, str(bad[:3]))
    sc.frame_set(sc.frame_start + 8)
    update()
    tips = [c['joints'][-1]['bone'] for c in phys['chains']] + cloth
    p1 = {b: pb[b].head.copy() for b in tips}
    for b in pb:
        for c in b.constraints:
            if c.name == 'Physics':
                c.mute = True
    update()
    moved = max((pb[b].head - p1[b]).length for b in tips)
    moved_cloth = max([(pb[b].head - p1[b]).length for b in cloth] or [0])
    for b in pb:
        for c in b.constraints:
            if c.name == 'Physics':
                c.mute = False
    update()
    check('physics moves cloth', moved > 0.001, f'{moved * 1000:.1f} mm')
    if cloth:
        check('physics moves the vertex cloth', 0.005 < moved_cloth < 1.0, f'{moved_cloth * 1000:.1f} mm')

# copy pose from a game clip with IK on
ends = [n for n in ('hand_L', 'hand_R', 'ankle_L', 'ankle_R', 'head', 'pelvis') if n in pb]
if POSE:
    reset()
    bpy.ops.dlrig.game_anim(clip=POSE, mode='POSE')
    update()
    with bpy.data.libraries.load(lib) as (src, dst):
        dst.actions = [POSE]
    ref = dst.actions[0]
    ik_pose = {n: pb[n].head.copy() for n in ends}
    reset()
    for p in IK_PROPS:
        arm[p] = 0.0
    arm.animation_data.action = ref
    arm.animation_data.action_slot = ref.slots[0]
    sc.frame_set(int(ref.frame_range[0]))
    update()
    worst = max((pb[n].head - p).length for n, p in ik_pose.items())
    check('copy pose (IK on) matches clip', worst < 0.002, f'{worst * 1000:.2f} mm')

# convert a game clip to IK: feet and hands stay where the FK clip had them
if RUN and limbs:
    reset()
    bpy.ops.dlrig.game_anim(clip=RUN, mode='NEW')
    watch = [names(limb, s)[2] for limb, s in limbs] + [names(limb, s)[1] for limb, s in limbs]
    uppers = [names(limb, s)[0] for limb, s in limbs]
    fk, fk_rot = {}, {}
    for f in (0, 5, 10):
        sc.frame_set(sc.frame_start + f)
        update()
        fk[f] = {n: pb[n].head.copy() for n in watch}
        fk_rot[f] = {n: pb[n].matrix.to_quaternion() for n in uppers}
    bpy.ops.dlrig.bake_ik(limbs='ALL')
    worst = turn = 0
    for f in (0, 5, 10):
        sc.frame_set(sc.frame_start + f)
        update()
        worst = max(worst, max((pb[n].head - p).length for n, p in fk[f].items()))
        turn = max(turn, max(q.rotation_difference(pb[n].matrix.to_quaternion()).angle for n, q in fk_rot[f].items()))
    check('convert to IK keeps the clip', worst < 0.003 and math.degrees(turn) < 0.1 and arm[IK_PROPS[0]] == 1.0,
          f'{worst * 1000:.2f} mm, upper bones {math.degrees(turn):.3f} deg')

# insert a clip into an existing action
if IDLE and JUMP:
    reset()
    bpy.ops.dlrig.game_anim(clip=IDLE, mode='NEW')
    sc.frame_set(40)
    bpy.ops.dlrig.game_anim(clip=JUMP, mode='INSERT')
    check('insert clip', sc.frame_end >= 40, f'range now {sc.frame_start}-{sc.frame_end}')

# bake time on a long clip
if IDLE and phys['chains']:
    reset()
    bpy.ops.dlrig.game_anim(clip=IDLE, mode='NEW')
    t0 = time.time()
    bpy.ops.dlrig.bake_physics()
    check('physics bake speed', time.time() - t0 < 20, f'{time.time() - t0:.1f}s for {sc.frame_end - sc.frame_start + 1} frames')

# idle layer on a static pose
if IDLE:
    reset()
    hand_poses = ns['HAND_POSES']
    if hand_poses['Fist']:
        bpy.ops.dlrig.hand(name='Fist', side='B')
    r = bpy.ops.dlrig.idle_layer()
    top = 'head' if 'head' in pb else 'pelvis'
    heads = []
    for f in (1, 30, 60, 90):
        sc.frame_set(f)
        heads.append(pb[top].head.copy())
    sway = max((h - heads[0]).length for h in heads)
    check('idle layer applies', r == {'FINISHED'} and sway < 0.3, f'{top} sways {sway * 1000:.1f} mm')
    if hand_poses['Fist']:
        side = next(iter(hand_poses['Fist']))
        fist = pb[f'finger_index_1_{side}'].rotation_quaternion.copy()
        check('idle layer keeps the pose', abs(fist.w - 1) > 0.01, str(tuple(round(x, 3) for x in fist)))
# face: every ARKit key moves the head by a sane amount, and the controls drive the keys
if 'FACE' in pb:
    reset()
    head = next(o for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys and 'jawOpen' in o.data.shape_keys.key_blocks)
    kbs = head.data.shape_keys.key_blocks
    base = np.array([v.co for v in kbs[0].data])
    fs = arm.get('face_scale', 1.0)
    morphs = arm.get('face_morphs', False)
    # game-morph faces keep the game's own morphs next to the ARKit keys built from them
    sliders = set(pb['FACE'].keys())
    # a key's size is its largest move on any mesh (face-bone keys also move the eyes, teeth and tongue)
    sizes = {}
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.data.shape_keys:
            continue
        ob_kbs = o.data.shape_keys.key_blocks
        ob_base = np.array([v.co for v in ob_kbs[0].data])
        for kb in ob_kbs[1:]:
            if o != head and kb.name not in kbs:
                continue
            if morphs and kb.name not in sliders and kb.name[:-4] not in sliders:
                continue
            m = np.linalg.norm(np.array([v.co for v in kb.data]) - ob_base, axis=1).max() * 1000
            sizes[kb.name] = max(sizes.get(kb.name, 0), m)
    # in-betweens only add what the straight blend misses, so they can be small; the game's jaws open wide
    top = 250 * fs if morphs else 50 * fs
    odd = {k: round(v, 1) for k, v in sizes.items() if not (0.05 if k.endswith('_mid') else 0.5) < v < top}
    check(f'face keys move 0.5-{top:.0f} mm', not odd, f'{len(sizes)} keys {odd}')
    lr = {k: (sizes[k], sizes[k[:-4] + 'Right']) for k in sizes if k.endswith('Left')}
    lop = {k: tuple(round(x, 1) for x in v) for k, v in lr.items() if abs(v[0] - v[1]) > 0.35 * max(v)}
    check('face keys left/right alike', not lop, str(lop))
    rest = verts(head)
    pb['CTRL_jaw'].location.y = -0.025 * fs
    update()
    jaw = np.linalg.norm(verts(head) - rest, axis=1).max() * 1000
    check('jaw control opens the mouth', kbs['jawOpen'].value > 0.99 and jaw > 10, f'jawOpen {kbs["jawOpen"].value:.2f}, {jaw:.1f} mm')
    reset()
    pb['FACE']['eyeBlinkLeft'] = 1.0
    update()
    blink = np.linalg.norm(verts(head) - rest, axis=1).max() * 1000
    check('face slider drives its key', kbs['eyeBlinkLeft'].value > 0.99 and blink > 3, f'{blink:.1f} mm')
    reset()
    mids = [kb.name for kb in kbs if kb.name.endswith('_mid') and kb.name[:-4] in sliders]
    if mids:
        # an in-between corrective is full at its key's 0.5 and gone at 0 and 1
        bad = {}
        for m in mids:
            got = []
            for x in (0.0, 0.5, 1.0):
                pb['FACE'][m[:-4]] = x
                update()
                got.append(round(kbs[m].value, 3))
            pb['FACE'][m[:-4]] = 0.0
            if got != [0.0, 1.0, 0.0]:
                bad[m] = got
        update()
        check('in-betweens peak at 0.5', not bad, f'{len(mids)} {bad}')
    if 'CTRL_eye_L' in pb:
        pb['CTRL_eye_L'].location.y = 0.15
        pb['CTRL_eye_L'].scale = (1.5, 1.5, 1.5)
        update()
        iris = kbs['irisLargeLeft'].value if 'irisLargeLeft' in kbs else 1.0
        ok = kbs['eyeLookUpLeft'].value > 0.99 and kbs['eyeLookUpRight'].value < 0.01 and iris > 0.99
        check('eye control looks and sizes one iris', ok,
              f'look {kbs["eyeLookUpLeft"].value:.2f}/{kbs["eyeLookUpRight"].value:.2f} iris {iris:.2f}')
        reset()

print('FAILS', fails)
