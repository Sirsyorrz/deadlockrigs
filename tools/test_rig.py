"""blender -b out/<hero>_rig.blend -P test_rig.py : numeric checks of the rig (no renders).
Prints PASS/FAIL lines and a final 'FAILS [...]'; checks for parts the hero lacks are skipped."""
import json
import math
import time

import bpy
import numpy as np
from mathutils import Quaternion, Vector

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

# rest pose untouched by the rig
reset()
for ob in [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_viewport and not o.name.startswith('WGT-')]:
    base = np.array([v.co for v in ob.data.vertices])
    if not len(base):
        continue
    drift = np.abs(verts(ob) - base).max() * 1000
    check(f'rest drift {ob.name}', drift < 0.05, f'{drift:.4f} mm')

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
    worst = max((before[n].translation - after[n].translation).length for n in before if not n.startswith(HELPERS))
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
lib = bpy.path.abspath(arm['anim_library'])
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

# physics bake on the run
phys = json.loads(bpy.data.texts['dlrig_physics.json'].as_string())
if RUN and phys['chains']:
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
    tips = [c['joints'][-1]['bone'] for c in phys['chains']]
    p1 = {b: pb[b].head.copy() for b in tips}
    for b in pb:
        for c in b.constraints:
            if c.name == 'Physics':
                c.mute = True
    update()
    moved = max((pb[b].head - p1[b]).length for b in tips)
    for b in pb:
        for c in b.constraints:
            if c.name == 'Physics':
                c.mute = False
    update()
    check('physics moves cloth', moved > 0.001, f'{moved * 1000:.1f} mm')

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
    fk = {}
    for f in (0, 5, 10):
        sc.frame_set(sc.frame_start + f)
        fk[f] = {n: pb[n].head.copy() for n in watch}
    bpy.ops.dlrig.bake_ik(limbs='ALL')
    worst = 0
    for f in (0, 5, 10):
        sc.frame_set(sc.frame_start + f)
        worst = max(worst, max((pb[n].head - p).length for n, p in fk[f].items()))
    check('convert to IK keeps the clip', worst < 0.003 and arm[IK_PROPS[0]] == 1.0, f'{worst * 1000:.2f} mm')

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
print('FAILS', fails)
