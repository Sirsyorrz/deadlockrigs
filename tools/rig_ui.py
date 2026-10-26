# Hero rig panel: 3D View sidebar (N) > <hero>. rig.py prepends HERO, HAS_FACE, CLOTH_LINKS and the preset data
# (ARKIT, GROUPS, EXPRESSIONS, VISEMES, HAND_POSES) and stores this as a registered text block.
import json
import math
import os
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

FACE = 'FACE'
FACE_CTRLS = ('CTRL_')
LIMBS = {
    'arm': ('arm_upper', 'arm_lower', 'hand', 'IK_hand', 'pole_elbow', 0.45),
    'leg': ('leg_upper', 'leg_lower', 'ankle', 'IK_foot', 'pole_knee', 0.55),
}
PHYS_TEXT = 'dlrig_physics.json'


def rigs():
    return [o for o in bpy.data.objects if o.type == 'ARMATURE' and 'dlrig' in o]


def rig(context):
    ob = context.active_object
    if ob and ob.type == 'ARMATURE' and 'dlrig' in ob:
        return ob
    return next(iter(rigs()), None)


def limb_list(ob):
    """The rig's limbs: arm and leg, plus prefixed extras (the Mid-boss' up_arm, the Walker's front_leg)."""
    return list(ob.get('limbs', LIMBS))


def ik_props(ob):
    return [f'ik_{limb}_{s}' for limb in limb_list(ob) for s in 'LR' if f'ik_{limb}_{s}' in ob]


def autokey(context):
    return context.scene.tool_settings.use_keyframe_insert_auto


def key_bone(pb, rot=True, loc=True):
    if loc:
        pb.keyframe_insert('location', group=pb.name)
    if rot:
        pb.keyframe_insert('rotation_quaternion' if pb.rotation_mode == 'QUATERNION' else 'rotation_euler',
                           group=pb.name)


def update(context):
    # custom properties set from Python (the IK/FK switches) only reach their drivers once the object is tagged
    for ob in rigs():
        ob.update_tag()
    context.view_layer.update()


class FastEval:
    """Turns off mesh deformation while stepping through frames, so baking only evaluates the armature."""

    def __init__(self, ob):
        self.mods = [m for o in bpy.data.objects if o.type == 'MESH' for m in o.modifiers
                     if m.type == 'ARMATURE' and m.object == ob and m.show_viewport]

    def __enter__(self):
        for m in self.mods:
            m.show_viewport = False

    def __exit__(self, *exc):
        for m in self.mods:
            m.show_viewport = True


def frame_range(context):
    sc = context.scene
    if sc.use_preview_range:
        return sc.frame_preview_start, sc.frame_preview_end
    return sc.frame_start, sc.frame_end


def channelbag(ob, create=True):
    ad = ob.animation_data
    if not ad or not ad.action:
        if not create:
            return None
        ob.animation_data_create()
        ob.animation_data.action = bpy.data.actions.new(ob.name + 'Action')
        ad = ob.animation_data
    a = ad.action
    if not ad.action_slot:
        ad.action_slot = a.slots[0] if len(a.slots) else a.slots.new('OBJECT', ob.name)
    if not a.layers:
        if not create:
            return None
        a.layers.new('Layer')
    layer = a.layers[0]
    if not layer.strips:
        layer.strips.new(type='KEYFRAME')
    strip = layer.strips[0]
    cb = strip.channelbag(ad.action_slot, ensure=create)
    return cb


def new_fcurve(cb, path, index, group):
    # fcurves.new() has no group_name before Blender 4.5
    fc = cb.fcurves.new(path, index=index)
    if group:
        fc.group = cb.groups.get(group) or cb.groups.new(group)
    return fc


# ---------------------------------------------------------------- faces and hands


class DLRIG_OT_expression(bpy.types.Operator):
    """Set the face to this preset (scaled by Strength; Shift-click adds on top)"""
    bl_idname = 'dlrig.expression'
    bl_label = 'Expression'
    bl_options = {'REGISTER', 'UNDO'}
    name: bpy.props.StringProperty()
    viseme: bpy.props.BoolProperty(options={'SKIP_SAVE'})
    add: bpy.props.BoolProperty(options={'SKIP_SAVE'})

    def invoke(self, context, event):
        self.add = event.shift
        return self.execute(context)

    def execute(self, context):
        ob = rig(context)
        pb = ob.pose.bones[FACE]
        preset = (VISEMES if self.viseme else EXPRESSIONS)[self.name]
        s = context.scene.dlrig_strength
        for k in ARKIT:
            v = preset.get(k, 0.0) * s
            pb[k] = min(1.0, pb[k] + v) if self.add else v
            if autokey(context):
                pb.keyframe_insert(f'["{k}"]', group='Face')
        ob.update_tag()
        return {'FINISHED'}


class DLRIG_OT_reset_face(bpy.types.Operator):
    """Zero every face slider and put the face controls back"""
    bl_idname = 'dlrig.reset_face'
    bl_label = 'Reset Face'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = rig(context)
        pb = ob.pose.bones[FACE]
        for k in ARKIT:
            pb[k] = 0.0
            if autokey(context):
                pb.keyframe_insert(f'["{k}"]', group='Face')
        for b in ob.pose.bones:
            if b.name.startswith(FACE_CTRLS):
                b.location = (0, 0, 0)
                b.scale = (1, 1, 1)
                if 'seal' in b:
                    b['seal'] = 0.0
                if autokey(context):
                    b.keyframe_insert('location', group=b.name)
                    b.keyframe_insert('scale', group=b.name)
        ob.update_tag()
        return {'FINISHED'}


class DLRIG_OT_key_face(bpy.types.Operator):
    """Insert keyframes for every face slider and face control on the current frame"""
    bl_idname = 'dlrig.key_face'
    bl_label = 'Key Whole Face'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = rig(context)
        pb = ob.pose.bones[FACE]
        for k in ARKIT:
            pb.keyframe_insert(f'["{k}"]', group='Face')
        for b in ob.pose.bones:
            if b.name.startswith(FACE_CTRLS):
                b.keyframe_insert('location', group=b.name)
                b.keyframe_insert('scale', group=b.name)
                if 'seal' in b:
                    b.keyframe_insert('["seal"]', group=b.name)
        return {'FINISHED'}


class DLRIG_OT_hand(bpy.types.Operator):
    """Pose the fingers (scaled by Strength)"""
    bl_idname = 'dlrig.hand'
    bl_label = 'Hand Pose'
    bl_options = {'REGISTER', 'UNDO'}
    name: bpy.props.StringProperty()
    side: bpy.props.StringProperty()

    def execute(self, context):
        ob = rig(context)
        s = context.scene.dlrig_strength
        for side in ('L', 'R') if self.side == 'B' else (self.side,):
            for bone, q in HAND_POSES[self.name].get(side, {}).items():
                pb = ob.pose.bones[bone]
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = Quaternion().slerp(Quaternion(q), s)
                if autokey(context):
                    pb.keyframe_insert('rotation_quaternion', group=bone)
        return {'FINISHED'}


class DLRIG_OT_reset_pose(bpy.types.Operator):
    """Clear every bone transform (body, IK controls and face controls) and the face sliders"""
    bl_idname = 'dlrig.reset_pose'
    bl_label = 'Reset Whole Pose'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = rig(context)
        reset_bones(ob, [b for b in ob.pose.bones])
        if FACE in ob.pose.bones:
            bpy.ops.dlrig.reset_face()
        return {'FINISHED'}


def reset_bones(ob, bones):
    """Back to the rest pose, except held props, which go back to their default hold (at rest they sit at the origin)."""
    hold = ob.get('prop_pose', {})
    for b in bones:
        b.location = (0, 0, 0)
        b.rotation_quaternion = (1, 0, 0, 0)
        b.rotation_euler = (0, 0, 0)
        b.scale = (1, 1, 1)
        if b.name in hold:
            v = list(hold[b.name])
            b.location, b.rotation_quaternion, b.scale = v[0:3], v[3:7], v[7:10]


# ---------------------------------------------------------------- FK / IK


def limb_names(limb, s):
    kind = 'leg' if limb.endswith('leg') else 'arm'
    p = limb[:-len(kind)]
    up, lo, end, ik, pole, dist = LIMBS[kind]
    ik, pole = ik.replace('_', '_' + p, 1), pole.replace('_', '_' + p, 1)
    return f'{p}{up}_{s}', f'{p}{lo}_{s}', f'{p}{end}_{s}', f'{ik}_{s}', f'{pole}_{s}', dist


def foot_names(ob, limb, s):
    """IK heel, toe and ankle controls and the FK toe, for legs with a foot rig."""
    if limb == 'leg' and f'IK_heel_{s}' in ob.pose.bones:
        return f'IK_heel_{s}', f'IK_toe_{s}', f'IK_ankle_{s}', f'ball_{s}'
    return ()


def isolate_prop(ob, name):
    return next((p for p, b in ob.get('isolate', {}).items() if b == name), None)


def set_matrix(context, ob, name, m):
    """pb.matrix = m, also for a bone whose follow switch is off: its rotation then comes from MCH-<bone>_iso, which
    takes the bone's keys under the torso, so the keys are solved for that."""
    pb = ob.pose.bones[name]
    pb.matrix = m
    prop = isolate_prop(ob, name)
    if prop is None or ob[prop] >= 0.5:
        return
    update(context)
    iso = ob.pose.bones[f'MCH-{name}_iso']
    frame = iso.matrix.to_3x3().normalized() @ pb.rotation_quaternion.to_matrix().inverted()
    pb.rotation_quaternion = (frame.inverted() @ m.to_3x3().normalized()).to_quaternion()


def keyed_anyway(ob, props, bones):
    """Without auto-key a change would still be thrown away on the next frame change wherever the action already
    animates these properties or bones, so those get keyed anyway."""
    cb = channelbag(ob, create=False)
    return cb is not None and any(f.data_path in [f'["{p}"]' for p in props] or
                                  (f.data_path.startswith('pose.bones["') and f.data_path.split('"')[1] in bones)
                                  for f in cb.fcurves)


def reach_of(length, soft, stretch, d):
    """Where the soft/stretchy IK puts the limb's end for a control at distance d from its root (rig.py's drivers)."""
    ds = max(length * soft, 0.001)
    start = length - ds
    dp = d if d <= start else start + ds * (1 - math.exp(min((start - d) / ds, 0)))
    return dp + stretch * (d - dp)


def control_distance(length, soft, stretch, want):
    """The control distance whose reach is want (bisection; the reach grows with the distance)."""
    lo, hi = 0.0, max(want, length) * 4 + 1
    if reach_of(length, soft, stretch, hi) < want:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if reach_of(length, soft, stretch, mid) < want else (lo, mid)
    return (lo + hi) / 2


def tweak_names(limb, s):
    up, lo = limb_names(limb, s)[:2]
    return f'IK_tweak_{up}', f'IK_tweak_{lo}'


def clear_tweaks(ob, limb, s, key=False):
    for n in tweak_names(limb, s):
        pb = ob.pose.bones[n]
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        if key:
            key_bone(pb, loc=False)


def match_twist(context, ob, limb, s, fk, key=False):
    """IK sets the upper and lower bones' roll from the pole, so any twist the FK pose has about the limb axis
    (which moves clothing and props parented to the arm/thigh) goes into the IK tweak bones instead. IK must be on;
    fk holds the FK matrices of the upper and lower bones."""
    pb = ob.pose.bones
    bones = ob.data.bones
    pairs = list(zip(limb_names(limb, s)[:2], tweak_names(limb, s)))

    def world(tw):
        # the stretch scales the tweaks along their length only
        return pb[tw].matrix.to_3x3().normalized().to_quaternion()
    update(context)
    # The tweaks take the FK bones' angles, which also absorbs what IK misses of a clip's lengthened limbs. With soft IK
    # or stretch the IK bend is meant to differ from the FK one, so they take only the twist about each bone's length,
    # decided before either turns (IK may roll the upper bone half a turn from FK, which the lower one takes back).
    exact = not (ob.get(f'soft_{limb}_{s}', 0.0) or ob.get(f'stretch_{limb}_{s}', 0.0))
    want = {}
    for n, tw in pairs:
        rel = (bones[tw].matrix_local.inverted() @ bones[n].matrix_local).to_quaternion()
        fk_tw = fk[n].to_quaternion() @ rel.inverted()
        _swing, twist = (world(tw).inverted() @ fk_tw).to_swing_twist('Y')
        want[tw] = fk_tw if exact else world(tw) @ Quaternion((0, 1, 0), twist)
    for _n, tw in pairs:
        update(context)
        pb[tw].rotation_mode = 'QUATERNION'
        pb[tw].rotation_quaternion = pb[tw].rotation_quaternion @ (world(tw).inverted() @ want[tw])
        if key:
            key_bone(pb[tw], loc=False)
    update(context)


def ik_from_fk(context, ob, limb, s, key=False):
    """Move the IK target, pole and foot controls to where the FK chain currently is (FK must be active)."""
    up, lo, end, ik, pole, dist = limb_names(limb, s)
    pb = ob.pose.bones
    feet = foot_names(ob, limb, s)
    ball_m = pb[feet[3]].matrix.copy() if feet else None
    a, b, c = pb[up].head.copy(), pb[lo].head.copy(), pb[end].head.copy()
    end_m = pb[end].matrix.copy()
    for n in feet[:3]:
        pb[n].rotation_euler = (0, 0, 0)
        pb[n].rotation_quaternion = (1, 0, 0, 0)
    t = (b - a).dot(c - a) / max((c - a).length_squared, 1e-9)
    off = b - (a + (c - a) * t)
    if off.length < 1e-4:
        # straight limb: keep the pole on the side it already is
        off = pb[pole].head - b
        off -= (c - a).normalized() * off.dot((c - a).normalized())
    # the control's rest differs from the end bone's (the foot control lies flat)
    bones = ob.data.bones
    pb[ik].matrix = end_m @ bones[end].matrix_local.inverted() @ bones[ik].matrix_local
    update(context)
    soft, stretch = ob.get(f'soft_{limb}_{s}', 0.0), ob.get(f'stretch_{limb}_{s}', 0.0)
    if soft > 0 or stretch > 0:
        # with soft IK the end lags behind the control, so the control goes as far out as puts the end on the FK end
        root, goal = pb[f'IKR-{end}'].head, pb[f'IKE-{end}'].head
        want = (goal - root).length
        d = control_distance(ob['limb_length'][end], soft, stretch, want)
        m = pb[ik].matrix.copy()
        m.translation += (goal - root).normalized() * (d - want)
        pb[ik].matrix = m
        update(context)
    m = pb[pole].matrix.copy()
    m.translation = b + off.normalized() * dist * ob.get('rig_scale', 1.0)
    pb[pole].matrix = m
    update(context)
    if feet:
        toe = pb[feet[1]]
        m = ball_m.to_3x3().normalized().to_4x4()
        m.translation = toe.matrix.translation
        toe.matrix = m
        update(context)
    if key:
        key_bone(pb[ik])
        key_bone(pb[pole], rot=False)
        for n in feet[:3]:
            key_bone(pb[n], loc=False)


class DLRIG_OT_snap(bpy.types.Operator):
    """Switch a limb between FK and IK without it jumping"""
    bl_idname = 'dlrig.snap'
    bl_label = 'Snap FK/IK'
    bl_options = {'REGISTER', 'UNDO'}
    limb: bpy.props.StringProperty()
    side: bpy.props.StringProperty()
    to_ik: bpy.props.BoolProperty()

    def execute(self, context):
        ob = rig(context)
        prop = f'ik_{self.limb}_{self.side}'
        up, lo, end, ik, pole, dist = limb_names(self.limb, self.side)
        pb = ob.pose.bones
        feet = foot_names(ob, self.limb, self.side)
        fk_bones = (up, lo, end) + feet[3:]
        names = (up, lo, end, ik, pole) + tweak_names(self.limb, self.side) + feet
        key = autokey(context) or keyed_anyway(ob, [prop], names)
        # snapping "to IK" while already in IK would rebuild the IK pose from the FK bones, which have not
        # moved (usually the rest pose), so the limb would jump back
        if (ob[prop] >= 0.5) == self.to_ik:
            self.report({'INFO'}, f'Already in {"IK" if self.to_ik else "FK"}')
            return {'CANCELLED'}
        if self.to_ik:
            ob[prop] = 0.0
            clear_tweaks(ob, self.limb, self.side, key)
            update(context)
            fk = {n: pb[n].matrix.copy() for n in (up, lo)}
            ik_from_fk(context, ob, self.limb, self.side, key)
            ob[prop] = 1.0
            match_twist(context, ob, self.limb, self.side, fk, key)
        else:
            mats = {n: pb[n].matrix.copy() for n in fk_bones}
            ob[prop] = 0.0
            clear_tweaks(ob, self.limb, self.side, key)
            update(context)
            for n in fk_bones:
                set_matrix(context, ob, n, mats[n])
                update(context)
                if key:
                    key_bone(pb[n])
        if key:
            ob.keyframe_insert(f'["{prop}"]', group='FK/IK')
        update(context)
        return {'FINISHED'}


class DLRIG_OT_bake_ik(bpy.types.Operator):
    """Key the IK hands/feet and poles on every frame so they follow the current (FK) animation, then switch to IK.
Use it on game animations to edit them with IK"""
    bl_idname = 'dlrig.bake_ik'
    bl_label = 'Convert Animation to IK'
    bl_options = {'REGISTER', 'UNDO'}
    limbs: bpy.props.EnumProperty(items=[('ALL', 'Arms and legs', ''), ('LEGS', 'Legs', ''), ('ARMS', 'Arms', '')])

    def execute(self, context):
        ob = rig(context)
        sc = context.scene
        todo = [(limb, s) for limb in limb_list(ob) for s in 'LR' if f'ik_{limb}_{s}' in ob and
                (self.limbs == 'ALL' or (self.limbs == 'LEGS') == limb.endswith('leg'))]
        for limb, s in todo:
            ob[f'ik_{limb}_{s}'] = 0.0
        cb = channelbag(ob)
        for limb, s in todo:
            for n in limb_names(limb, s)[3:5] + tweak_names(limb, s) + foot_names(ob, limb, s)[:3]:
                for f in [f for f in cb.fcurves if f.data_path.startswith(f'pose.bones["{n}"]')]:
                    cb.fcurves.remove(f)
            clear_tweaks(ob, limb, s)
        f0, f1 = frame_range(context)
        keep = sc.frame_current
        with FastEval(ob):
            for f in range(f0, f1 + 1):
                sc.frame_set(f)
                update(context)
                fk = {}
                for limb, s in todo:
                    fk[limb, s] = {n: ob.pose.bones[n].matrix.copy() for n in limb_names(limb, s)[:2]}
                    ik_from_fk(context, ob, limb, s, key=True)
                for limb, s in todo:
                    ob[f'ik_{limb}_{s}'] = 1.0
                for limb, s in todo:
                    match_twist(context, ob, limb, s, fk[limb, s], key=True)
                for limb, s in todo:
                    ob[f'ik_{limb}_{s}'] = 0.0
        for limb, s in todo:
            ob[f'ik_{limb}_{s}'] = 1.0
        sc.frame_set(keep)
        self.report({'INFO'}, f'IK keyed on frames {f0}-{f1}')
        return {'FINISHED'}


class DLRIG_OT_bake_fk(bpy.types.Operator):
    """Key the FK arm and leg bones on every frame so they follow the current (IK) animation, then switch to FK"""
    bl_idname = 'dlrig.bake_fk'
    bl_label = 'Convert Animation to FK'
    bl_options = {'REGISTER', 'UNDO'}
    limbs: bpy.props.EnumProperty(items=[('ALL', 'Arms and legs', ''), ('LEGS', 'Legs', ''), ('ARMS', 'Arms', '')])

    def execute(self, context):
        ob = rig(context)
        sc = context.scene
        todo = [(limb, s) for limb in limb_list(ob) for s in 'LR' if f'ik_{limb}_{s}' in ob and
                (self.limbs == 'ALL' or (self.limbs == 'LEGS') == limb.endswith('leg'))]
        bones = {t: limb_names(*t)[:3] + foot_names(ob, *t)[3:] for t in todo}
        f0, f1 = frame_range(context)
        keep = sc.frame_current
        channelbag(ob)
        with FastEval(ob):
            for f in range(f0, f1 + 1):
                sc.frame_set(f)
                for limb, s in todo:
                    ob[f'ik_{limb}_{s}'] = 1.0
                update(context)
                mats = {t: {n: ob.pose.bones[n].matrix.copy() for n in bones[t]} for t in todo}
                for limb, s in todo:
                    ob[f'ik_{limb}_{s}'] = 0.0
                update(context)
                for t in todo:
                    for n in bones[t]:
                        set_matrix(context, ob, n, mats[t][n])
                        update(context)
                        key_bone(ob.pose.bones[n])
        cb = channelbag(ob)
        for limb, s in todo:
            f = cb.fcurves.find(f'["ik_{limb}_{s}"]')
            if f:
                cb.fcurves.remove(f)
            ob[f'ik_{limb}_{s}'] = 0.0
        sc.frame_set(keep)
        self.report({'INFO'}, f'FK keyed on frames {f0}-{f1}')
        return {'FINISHED'}


class DLRIG_OT_switch_parent(bpy.types.Operator):
    """Make this control follow another bone, without it jumping"""
    bl_idname = 'dlrig.switch_parent'
    bl_label = 'Switch Parent'
    bl_options = {'REGISTER', 'UNDO'}
    bone: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        ob = rig(context)
        prop = self.bone + '_parent'
        pb = ob.pose.bones[self.bone]
        key = autokey(context) or keyed_anyway(ob, [prop], [self.bone])
        m = pb.matrix.copy()
        ob[prop] = self.index
        update(context)
        pb.matrix = m
        update(context)
        if key:
            ob.keyframe_insert(f'["{prop}"]', group='Parents')
            key_bone(pb)
        return {'FINISHED'}


class DLRIG_OT_switch_follow(bpy.types.Operator):
    """Switch whether this bone follows the body or keeps its own angle, without it jumping"""
    bl_idname = 'dlrig.switch_follow'
    bl_label = 'Switch Follow'
    bl_options = {'REGISTER', 'UNDO'}
    prop: bpy.props.StringProperty()

    def execute(self, context):
        ob = rig(context)
        bone = ob['isolate'][self.prop]
        pb = ob.pose.bones[bone]
        key = autokey(context) or keyed_anyway(ob, [self.prop], [bone])
        m = pb.matrix.copy()
        ob[self.prop] = 0.0 if ob[self.prop] >= 0.5 else 1.0
        update(context)
        set_matrix(context, ob, bone, m)
        update(context)
        if key:
            ob.keyframe_insert(f'["{self.prop}"]', group='Follow')
            key_bone(pb, loc=False)
        return {'FINISHED'}


# ---------------------------------------------------------------- game animations

_clips = []


def anim_library(ob):
    # the add-on's Export Animations writes <hero>_anims.blend next to rigs it built earlier, which have no path set
    return bpy.path.abspath(ob.get('anim_library') or f'//{ob["dlrig"]}_anims.blend')


def clip_items(self, context):
    if not _clips:
        ob = rig(context)
        path = anim_library(ob) if ob else ''
        try:
            with bpy.data.libraries.load(path) as (src, _dst):
                names = sorted(src.actions)
        except OSError:
            names = []
        _clips.extend((n, n[5:] if n.startswith('game_') else n, '') for n in names)
    return _clips


def append_clip(ob, name):
    with bpy.data.libraries.load(anim_library(ob)) as (_src, dst):
        dst.actions = [name]
    a = dst.actions[0]
    a.use_fake_user = False
    cb = a.layers[0].strips[0].channelbags[0]
    # the twist bones with a Twist constraint are rebuilt by it from the same inputs the game uses
    # the vertex cloth bones follow the body through their Cloth Follow constraint, so the clip's keys for them
    # would count twice
    driven = {b.name for b in ob.pose.bones if 'Twist' in b.constraints or 'Cloth Follow' in b.constraints}
    for f in [f for f in cb.fcurves if f.data_path.startswith('pose.bones["') and f.data_path.split('"')[1] in driven]:
        cb.fcurves.remove(f)
    return a


def body_bones(ob):
    helpers = ('CTRL_', FACE, 'PHYS-', 'TW-', 'IKC-', 'IKE-', 'IKT-', 'IKR-', 'IKS-', 'MCH-')
    return [b for b in ob.pose.bones if not b.name.startswith(helpers)]


class DLRIG_OT_game_anim(bpy.types.Operator):
    """Use one of the hero's in-game animations"""
    bl_idname = 'dlrig.game_anim'
    bl_label = 'Game Animation'
    bl_options = {'REGISTER', 'UNDO'}
    bl_property = 'clip'
    clip: bpy.props.EnumProperty(items=clip_items, name='Animation')
    mode: bpy.props.EnumProperty(items=[
        ('NEW', 'New Action', 'Load the clip as a new action you can edit; arms and legs switch to FK'),
        ('POSE', 'Pose Only', "Copy the clip's first frame onto the current pose (keyed when auto-key is on)"),
        ('INSERT', 'Insert at Frame', 'Copy the clip into the current action, starting at the current frame'),
    ])

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ob = rig(context)
        sc = context.scene
        clip = append_clip(ob, self.clip)
        cb_src = clip.layers[0].strips[0].channelbags[0]
        if 'weapon_hold' in ob:
            ob['weapon_hold'] = 0.0
        start = clip.frame_range[0]
        if self.mode == 'NEW':
            clip.name = self.clip[5:] if self.clip.startswith('game_') else self.clip
            reset_bones(ob, body_bones(ob))
            ob.animation_data_create()
            ob.animation_data.action = clip
            ob.animation_data.action_slot = clip.slots[0]
            for p in ik_props(ob):
                ob[p] = 0.0
            sc.frame_start = int(start)
            sc.frame_end = max(int(math.ceil(clip.frame_range[1])), sc.frame_start + 1)
            sc.frame_set(sc.frame_start)
            if sc.dlrig_phys_auto:
                bpy.ops.dlrig.bake_physics()
            return {'FINISHED'}
        ik_on = {p: ob[p] for p in ik_props(ob)}
        if self.mode == 'POSE':
            reset_bones(ob, body_bones(ob))
            for f in cb_src.fcurves:
                ob.path_resolve(f.data_path)[f.array_index] = f.evaluate(start)
            for p in ik_on:
                ob[p] = 0.0
            update(context)
            for p, v in ik_on.items():
                if v > 0.5:
                    limb, s = p[3:].rsplit('_', 1)
                    ik_from_fk(context, ob, limb, s)
                    ob[p] = 1.0
            if autokey(context):
                for pb in body_bones(ob):
                    key_bone(pb)
        else:
            cb = channelbag(ob)
            off = sc.frame_current - start
            for f in cb_src.fcurves:
                dst = cb.fcurves.find(f.data_path, index=f.array_index) or \
                    new_fcurve(cb, f.data_path, f.array_index, f.group.name if f.group else '')
                for k in f.keyframe_points:
                    dst.keyframe_points.insert(k.co[0] + off, k.co[1], options={'FAST'})
                dst.update()
            sc.frame_end = max(sc.frame_end, int(sc.frame_current + clip.frame_range[1] - start))
            act = ob.animation_data.action
            if act.use_frame_range:
                act.frame_end = max(act.frame_end, sc.frame_end)
            for p in ik_on:
                ob[p] = 0.0
        bpy.data.actions.remove(clip)
        update(context)
        return {'FINISHED'}


IDLE_TRACK = 'Idle motion'
IDLE_CLIPS = ('game_out_of_combat_stand_idle', 'game_weapon_stand_idle', 'game_primary_stand_idle',
              'game_stand_idle', 'game_item_stand_idle', 'game_primary_idle_no_target', 'game_idle')


def idle_clip(context):
    names = {c[0] for c in clip_items(None, context)}
    return next((c for c in IDLE_CLIPS if c in names),
                next((n for n in sorted(names) if 'stand_idle' in n), next((n for n in sorted(names) if 'idle' in n), None)))
IDLE_BONES = ('pelvis', 'spine_0', 'spine_1', 'spine_2', 'spine_3', 'up_body_0', 'neck_0', 'head', 'clavicle_L',
              'clavicle_R')


class DLRIG_OT_idle_layer(bpy.types.Operator):
    """Layer the breathing and sway of one of the game's idles on top of your animation (an NLA track in
Combine mode, so your own keys still work; change its strength in the panel)"""
    bl_idname = 'dlrig.idle_layer'
    bl_label = 'Add Idle Motion'
    bl_options = {'REGISTER', 'UNDO'}
    clip: bpy.props.StringProperty(default='')
    strength: bpy.props.FloatProperty(default=1.0, min=0.0, max=2.0)

    def execute(self, context):
        ob = rig(context)
        sc = context.scene
        ad = ob.animation_data or ob.animation_data_create()
        remove_idle(ob)
        clip = self.clip or idle_clip(context)
        if not clip:
            self.report({'ERROR'}, 'No idle clip among the game animations')
            return {'CANCELLED'}
        src = append_clip(ob, clip)
        cb_src = src.layers[0].strips[0].channelbags[0]
        f0, f1 = int(src.frame_range[0]), int(src.frame_range[1])
        act = bpy.data.actions.new(IDLE_TRACK)
        slot = act.slots.new('OBJECT', ob.name)
        act.layers.new('Layer').strips.new(type='KEYFRAME')
        cb = act.layers[0].strips[0].channelbag(slot, ensure=True)
        for bone in [b for b in IDLE_BONES if b in ob.pose.bones]:
            path_q = f'pose.bones["{bone}"].rotation_quaternion'
            path_l = f'pose.bones["{bone}"].location'
            fq = [cb_src.fcurves.find(path_q, index=i) for i in range(4)]
            fl = [cb_src.fcurves.find(path_l, index=i) for i in range(3)]
            if not any(fq) and not any(fl):
                continue

            def q_at(f):
                return Quaternion([c.evaluate(f) if c else (1.0 if i == 0 else 0.0) for i, c in enumerate(fq)]).normalized()

            def l_at(f):
                return Vector([c.evaluate(f) if c else 0.0 for c in fl])

            q0, l0 = q_at(f0), l_at(f0)
            out_q = [new_fcurve(cb, path_q, i, bone) for i in range(4)]
            out_l = [new_fcurve(cb, path_l, i, bone) for i in range(3)]
            for f in range(f0, f1 + 1):
                dq = q0.inverted() @ q_at(f)
                dl = l_at(f) - l0
                for i in range(4):
                    out_q[i].keyframe_points.insert(f, dq[i], options={'FAST'})
                for i in range(3):
                    out_l[i].keyframe_points.insert(f, dl[i], options={'FAST'})
        bpy.data.actions.remove(src)
        track = ad.nla_tracks.new()
        track.name = IDLE_TRACK
        start = sc.frame_start
        strip = track.strips.new(IDLE_TRACK, start, act)
        strip.action_slot = slot
        strip.blend_type = 'COMBINE'
        strip.extrapolation = 'HOLD'
        length = max(1, f1 - f0)
        strip.repeat = max(1.0, math.ceil((sc.frame_end - start + 1) / length) + 1)
        strip.use_animated_influence = True
        context.scene.dlrig_idle_strength = self.strength
        set_idle_strength(ob, self.strength)
        ad.action_blend_type = 'COMBINE'
        update(context)
        return {'FINISHED'}


def set_idle_strength(ob, value):
    strip = idle_strip(ob)
    if not strip:
        return
    fc = strip.fcurves.find('influence')
    if fc is None:
        strip.keyframe_insert('influence')
        fc = strip.fcurves.find('influence')
    fc.keyframe_points.clear()
    fc.keyframe_points.insert(strip.frame_start, value)
    fc.update()
    ob.update_tag(refresh={'TIME'})


def idle_strip(ob):
    ad = ob.animation_data
    track = ad and ad.nla_tracks.get(IDLE_TRACK)
    return track.strips[0] if track and track.strips else None


def remove_idle(ob):
    ad = ob.animation_data
    if not ad:
        return
    track = ad.nla_tracks.get(IDLE_TRACK)
    if track:
        acts = [st.action for st in track.strips]
        ad.nla_tracks.remove(track)
        for a in acts:
            if a and a.users == 0:
                bpy.data.actions.remove(a)


class DLRIG_OT_remove_idle(bpy.types.Operator):
    """Remove the idle motion layer"""
    bl_idname = 'dlrig.remove_idle'
    bl_label = 'Remove'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_idle(rig(context))
        update(context)
        return {'FINISHED'}


# ---------------------------------------------------------------- physics (secondary motion)


def seg_closest(p, a, b):
    ab = b - a
    t = 0.0 if ab.length_squared < 1e-12 else min(max((p - a).dot(ab) / ab.length_squared, 0.0), 1.0)
    return a + ab * t, t


# neighbouring apron chains skin the same panel; without links a leg between them parts them like curtains
CLOTH_LINKS = [('apron_outer_0_L', 'apron_inner_0_L'), ('apron_inner_0_L', 'apron_inner_0_R'),
               ('apron_inner_0_R', 'apron_outer_0_R')]


def collide_point(p, rad, cp, w, allow):
    for k, (a0, a1, b0, b1, r0, r1) in enumerate(cp):
        a, b = a0.lerp(a1, w), b0.lerp(b1, w)
        q, t = seg_closest(p, a, b)
        # the game's capsules overlap some of the cloth at rest, so the radius is capped at the rest-pose
        # distance (else the apron would pop off the hips)
        r = min(r0 + (r1 - r0) * t + rad, allow[k])
        d = p - q
        if d.length_squared < r * r:
            p = q + (d.normalized() if d.length > 1e-9 else Vector((0, -1, 0))) * r
    return p


def simulate(cfg, goals, caps, fps, settings, allow, warm, links=(), link_allow=()):
    """goals[key][chain][joint] -> world positions from the animation; caps[key] -> [(a0, a1, b0, b1, r0, r1)].
    allow[chain][joint][capsule] caps the collision radius at the rest-pose distance. warm is the list of
    keys to run before recording frames. links are (chain, chain) pairs held at their animated spacing, whose
    midpoints also collide (link_allow[link][joint][capsule]). Position-based: inertia, gravity, a pull towards
    the animated pose (the game's goal strength), bend stiffness, fixed segment lengths, capsule collisions."""
    frames = sorted(k for k in goals if isinstance(k, int))
    sub = settings['substeps']
    dt = 1.0 / fps / sub
    g = Vector((0, 0, -9.81 * settings['gravity'])) * dt * dt
    damp = 1.0 - (1.0 - settings['damping']) ** (1.0 / sub)
    collide = settings['collide']
    chains = cfg['chains']
    lengths = [[(goals[frames[0]][ci][j + 1] - goals[frames[0]][ci][j]).length for j in range(len(c['joints']) - 1)]
               for ci, c in enumerate(chains)]
    pull = []
    for c in chains:
        row = []
        for j in c['joints']:
            k = min(1.0, 2.0 * (j['goal_strength'] ** 2) * settings['stiffness'])
            row.append(1.0 - (1.0 - k) ** (1.0 / sub))
        pull.append(row)
    bend = [[min(1.0, j['bend'] * 0.35 * settings['stiffness']) for j in c['joints']] for c in chains]

    def chain_pass(ci, G, cp, w):
        js = chains[ci]['joints']
        P = pos[ci]
        for j in range(1, len(js)):
            if not js[j]['simulate']:
                continue
            if j >= 2 and bend[ci][j] > 0:
                ga, gb = G[j - 1] - G[j - 2], P[j - 1] - P[j - 2]
                if ga.length_squared > 1e-12 and gb.length_squared > 1e-12:
                    want = P[j - 1] + ga.rotation_difference(gb) @ (G[j] - G[j - 1])
                    P[j] = P[j].lerp(want, bend[ci][j])
            if collide:
                P[j] = collide_point(P[j], js[j]['radius'], cp, w, allow[ci][j])
            seg = P[j] - P[j - 1]
            L = lengths[ci][j - 1]
            P[j] = P[j - 1] + (seg.normalized() if seg.length > 1e-9 else (G[j] - G[j - 1]).normalized()) * L

    def link_pass(Gs, cp, w):
        for li, (ca, cb) in enumerate(links):
            ja, jb = chains[ca]['joints'], chains[cb]['joints']
            for j in range(1, min(len(ja), len(jb))):
                if not (ja[j]['simulate'] and jb[j]['simulate']):
                    continue
                Pa, Pb = pos[ca], pos[cb]
                rest = (Gs[cb][j] - Gs[ca][j]).length
                d = Pb[j] - Pa[j]
                if d.length > 1e-9:
                    fix = d * (0.5 * (d.length - rest) / d.length)
                    Pa[j] = Pa[j] + fix
                    Pb[j] = Pb[j] - fix
                if collide:
                    m = (Pa[j] + Pb[j]) * 0.5
                    push = collide_point(m, 0.5 * (ja[j]['radius'] + jb[j]['radius']), cp, w, link_allow[li][j]) - m
                    Pa[j] = Pa[j] + push
                    Pb[j] = Pb[j] + push

    first = goals[warm[0]]
    pos = [[p.copy() for p in ch] for ch in first]
    prev = [[p.copy() for p in ch] for ch in first]
    out = {}
    passes = [('warm', warm), ('rec', frames)]
    last_goal = first
    for kind, seq in passes:
        for f in seq:
            gl, cp = goals[f], caps[f]
            for k in range(1, sub + 1):
                w = k / sub
                Gs = []
                for ci, c in enumerate(chains):
                    js = c['joints']
                    G = [last_goal[ci][j].lerp(gl[ci][j], w) for j in range(len(js))]
                    Gs.append(G)
                    P, Q = pos[ci], prev[ci]
                    P[0] = G[0]
                    Q[0] = G[0]
                    for j in range(1, len(js)):
                        if not js[j]['simulate']:
                            P[j] = G[j]
                            Q[j] = G[j]
                            continue
                        v = (P[j] - Q[j]) * (1.0 - damp)
                        Q[j] = P[j].copy()
                        P[j] = P[j] + v + g * js[j]['gravity']
                        P[j] = P[j].lerp(G[j], pull[ci][j])
                for ci in range(len(chains)):
                    chain_pass(ci, Gs[ci], cp, w)
                if links:
                    link_pass(Gs, cp, w)
                    for ci in {c for pair in links for c in pair}:
                        chain_pass(ci, Gs[ci], cp, w)
            last_goal = gl
            if kind == 'rec':
                out[f] = [[p.copy() for p in ch] for ch in pos]
    return out


def simulate_cloth(cloth, goals, caps, fps, settings, allow, seq):
    """Vertex cloth (Abrams' coat): goals[key] -> (n, 3) world positions of every node without simulation; fixed
    nodes stay on theirs. Position-based like simulate(): inertia, gravity, the game's two animation attractions
    (force: the share of the goal's own motion a node takes along, vertex: a pull towards the goal), rods kept
    between their min and max length (the game's), capsule collisions. Returns {key: (n, 3)} for the keys after the
    warm-up."""
    import numpy as np
    nodes, rods = cloth['nodes'], np.array(cloth['rods'], float).reshape(-1, 4)
    free = np.array([nd['free'] for nd in nodes])
    a, b = rods[:, 0].astype(int), rods[:, 1].astype(int)
    lo, hi = rods[:, 2], rods[:, 3]
    inv = free.astype(float)
    wa, wb = inv[a], inv[b]
    wsum = np.maximum(wa + wb, 1e-9)
    count = np.maximum(np.bincount(a, wa > 0, len(nodes)) + np.bincount(b, wb > 0, len(nodes)), 1)[:, None]
    rad = np.array([nd['radius'] for nd in nodes])
    sub = settings['substeps']
    dt = 1.0 / fps / sub
    grav = np.array([0, 0, -9.81 * settings['gravity']]) * dt * dt
    gscale = np.array([nd['gravity'] for nd in nodes])[:, None]
    damp = 1.0 - (1.0 - settings['damping']) ** (1.0 / sub)
    # both attractions applied per 1/240 s (the game's cloth runs several iterations a tick; at 60 Hz Grey Talon's
    # apron flew up to his waist). Nodes near a belt take all of the body's motion (force 1) and are barely pulled;
    # without the force term they hung in world space and sagged off it, and a stiffer pull to make up for that
    # locked Pocket's coat hem to his hips so his knees went through it
    tick = dt * 240
    k = np.clip(np.array([nd['goal'] for nd in nodes]) * settings['stiffness'], 0, 1)
    pull = (1.0 - (1.0 - k) ** tick)[:, None]
    fa = np.clip(np.array([nd.get('force', 0.0) for nd in nodes]), 0, 1)
    keep = ((1.0 - fa) ** tick)[:, None] * (1.0 - damp)
    allow = np.asarray(allow)
    first = goals[seq[0]]
    P, Q = first.copy(), first.copy()
    last = first
    G = first
    out = {}
    for key in seq:
        G1 = goals[key]
        for i in range(1, sub + 1):
            w = i / sub
            Gp, G = G, last + (G1 - last) * w
            vg = G - Gp
            v = vg + (P - Q - vg) * keep
            Q = P.copy()
            P = P + v + grav * gscale
            P += (G - P) * pull
            P[~free] = G[~free]
            for _ in range(8):
                d = P[b] - P[a]
                ln = np.linalg.norm(d, axis=1)
                tgt = np.clip(ln, lo, hi)
                corr = d * ((ln - tgt) / np.maximum(ln, 1e-9) / wsum)[:, None]
                delta = np.zeros_like(P)
                np.add.at(delta, a, corr * wa[:, None])
                np.add.at(delta, b, -corr * wb[:, None])
                P += delta / count
            if settings['collide']:
                for c, (a0, a1, b0, b1, r0, r1) in enumerate(caps[key]):
                    ca = np.array(a0.lerp(a1, w))
                    cb = np.array(b0.lerp(b1, w))
                    ab = cb - ca
                    t = np.clip(((P - ca) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                    q = ca + t[:, None] * ab
                    r = np.minimum(r0 + (r1 - r0) * t + rad, allow[:, c])
                    dv = P - q
                    dl = np.linalg.norm(dv, axis=1)
                    hit = free & (dl < r) & (dl > 1e-9)
                    P[hit] = q[hit] + dv[hit] * (r[hit] / dl[hit])[:, None]
        last = G1
        if isinstance(key, int):
            out[key] = P.copy()
    return out


def cloth_goals(cloth, m, rest_inv):
    """Every node where the body bones' blend (its follow weights) takes it, from bone matrices m."""
    import numpy as np
    out = np.zeros((len(cloth['nodes']), 3))
    for i, nd in enumerate(cloth['nodes']):
        p = Vector(nd['rest'])
        acc = Vector()
        tot = 0.0
        for bn, w in nd['follow'].items():
            if bn in m:
                acc += (m[bn] @ rest_inv[bn] @ p) * w
                tot += w
        out[i] = acc / tot if tot else p
    return out


def capsule_frames(cp_prev, cp):
    return [(a0, a1, b0, b1, r0, r1) for (a0, b0, r0, r1), (a1, b1, _, _) in zip(cp_prev, cp)]


def bake_cloth(ob, cloth, mats, rest_inv, rest_caps, lcaps, root, fps, settings, seq, frames):
    """Simulates the vertex cloth and keys each $cloth bone's PHYS- offset (armature axes). Returns the curves."""
    import numpy as np
    if not cloth['nodes']:
        return []
    world = {k: cloth_goals(cloth, mats[k], rest_inv) for k in seq}

    def to_local(k, P):
        m = np.array(root[k].inverted())
        return P @ m[:3, :3].T + m[:3, 3]
    goals = {k: to_local(k, world[k]) for k in seq}
    rest_p = [Vector(nd['rest']) for nd in cloth['nodes']]
    allow = [[(p - seg_closest(p, a, b)[0]).length for a, b in rest_caps] for p in rest_p] if rest_caps else \
        np.zeros((len(rest_p), 0))
    sim = simulate_cloth(cloth, goals, lcaps, fps, settings, allow, seq)
    sim = {k: v @ np.array(root[k])[:3, :3].T + np.array(root[k])[:3, 3] for k, v in sim.items()}
    goals = world
    to_arm = ob.matrix_world.inverted().to_3x3()
    cb = channelbag(ob)
    out = []
    for i, nd in enumerate(cloth['nodes']):
        name = 'PHYS-' + nd['name']
        if not nd['free'] or name not in ob.pose.bones:
            continue
        fcs = [new_fcurve(cb, f'pose.bones["{name}"].location', j, 'Physics') for j in range(3)]
        for fc in fcs:
            fc.keyframe_points.add(len(frames))
        for fi, f in enumerate(frames):
            d = to_arm @ Vector(sim[f][i] - goals[f][i])
            for j, fc in enumerate(fcs):
                kp = fc.keyframe_points[fi]
                kp.co = (f, d[j])
                kp.interpolation = 'LINEAR'
        out.append(fcs)
    return out


class DLRIG_OT_bake_physics(bpy.types.Operator):
    """Simulate the hair and cloth chains over the frame range (preview range if set) and key the result.
Uses the game's own cloth settings and body collision capsules"""
    bl_idname = 'dlrig.bake_physics'
    bl_label = 'Bake Physics'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = rig(context)
        sc = context.scene
        cfg = json.loads(bpy.data.texts[PHYS_TEXT].as_string())
        t0 = time.time()
        clear_physics(ob)
        pb = ob.pose.bones
        f0, f1 = frame_range(context)
        keep = sc.frame_current
        chain_bones = [[j['bone'] for j in c['joints']] for c in cfg['chains']]
        need = {b for bones in chain_bones for b in bones} | {c['bone'] for c in cfg['capsules']}
        need |= {pb[bones[0]].parent.name for bones in chain_bones}
        cloth = cfg.get('cloth') or {'nodes': [], 'rods': []}
        need |= {bn for nd in cloth['nodes'] for bn in nd['follow'] if bn in pb}
        if 'root_motion' in pb:
            need.add('root_motion')
        cap_local = [(Vector(c['p0']), Vector(c['p1']), c['r0'], c['r1'], c['bone']) for c in cfg['capsules']]
        mats, goals, caps = {}, {}, {}
        with FastEval(ob):
            for f in range(f0, f1 + 1):
                sc.frame_set(f)
                mw = ob.matrix_world
                m = {b: mw @ pb[b].matrix for b in need}
                mats[f] = m
                goals[f] = [[m[b].translation.copy() for b in bones] for bones in chain_bones]
                caps[f] = [(m[bn] @ p0, m[bn] @ p1, r0, r1) for p0, p1, r0, r1, bn in cap_local]
        frames = list(range(f0, f1 + 1))
        fps = sc.render.fps / sc.render.fps_base
        mw = ob.matrix_world
        rest = {b: mw @ ob.data.bones[b].matrix_local for b in need}
        rest_goals = [[rest[b].translation.copy() for b in bones] for bones in chain_bones]
        rest_caps = [(rest[bn] @ p0, rest[bn] @ p1) for p0, p1, _, _, bn in cap_local]
        rest_inv = {b: rest[b].inverted() for b in need}
        allow = [[[(p - seg_closest(p, a, b)[0]).length for a, b in rest_caps] for p in chain]
                 for chain in rest_goals]
        if sc.dlrig_phys_loop:
            warm = list(frames)
        else:
            # start from the rest pose and move into the first frame, so a leg that is already raised on
            # frame 1 sweeps the apron forward instead of starting inside it
            n = max(2, int(fps))
            warm = []
            for i in range(n):
                s = min(1.0, i / (0.6 * n))
                s = s * s * (3 - 2 * s)
                m = {b: rest[b].lerp(mats[f0][b], s) for b in need}
                key = ('warm', i)
                goals[key] = [[m[b].translation.copy() for b in bones] for bones in chain_bones]
                caps[key] = [(m[bn] @ p0, m[bn] @ p1, r0, r1) for p0, p1, r0, r1, bn in cap_local]
                warm.append(key)
                mats[key] = m
        seq = warm + frames
        caps2 = {}
        for i, f in enumerate(seq):
            caps2[f] = capsule_frames(caps[seq[i - 1] if i else f], caps[f])
        settings = {'substeps': 4, 'gravity': sc.dlrig_phys_gravity, 'damping': sc.dlrig_phys_damping,
                    'stiffness': sc.dlrig_phys_stiffness, 'collide': sc.dlrig_phys_collide,
                    'loop': sc.dlrig_phys_loop}
        idx = {bones[0]: ci for ci, bones in enumerate(chain_bones)}
        links = [(idx[a], idx[b]) for a, b in CLOTH_LINKS if a in idx and b in idx]
        link_allow = [[[((pa + pb) * 0.5 - seg_closest((pa + pb) * 0.5, a, b)[0]).length for a, b in rest_caps]
                       for pa, pb in zip(rest_goals[ca], rest_goals[cb])] for ca, cb in links]
        # simulated in the root motion's frame: a clip's travel would otherwise drag hair and coats out behind, and a
        # looping clip jumps back to its start every cycle
        root = {k: mats[k]['root_motion'] if 'root_motion' in need else mw for k in seq}
        inv = {k: m.inverted() for k, m in root.items()}
        lgoals = {k: [[inv[k] @ p for p in ch] for ch in goals[k]] for k in seq}
        lcaps = {k: [(inv[k] @ a0, inv[k] @ a1, inv[k] @ b0, inv[k] @ b1, r0, r1)
                     for a0, a1, b0, b1, r0, r1 in caps2[k]] for k in seq}
        sim = simulate(cfg, lgoals, lcaps, fps, settings, allow, warm, links, link_allow)
        sim = {f: [[root[f] @ p for p in ch] for ch in v] for f, v in sim.items()}
        cloth_curves = bake_cloth(ob, cloth, mats, rest_inv, rest_caps, lcaps, root, fps, settings, seq, frames)
        rest_rot = {b.name: b.matrix_local.to_quaternion() for b in ob.data.bones}
        cb = channelbag(ob)
        curves = {}
        for bones in chain_bones:
            for b in bones[:-1]:
                path = f'pose.bones["PHYS-{b}"].rotation_quaternion'
                curves[b] = [new_fcurve(cb, path, i, 'Physics') for i in range(4)]
                for fc in curves[b]:
                    fc.keyframe_points.add(len(frames))
        last = {}
        for fi, f in enumerate(frames):
            m = mats[f]
            for ci, bones in enumerate(chain_bones):
                parent = pb[bones[0]].parent.name
                final_parent = m[parent].to_quaternion()
                anim_parent = final_parent
                for j, b in enumerate(bones[:-1]):
                    a = goals[f][ci][j + 1] - goals[f][ci][j]
                    s_ = sim[f][ci][j + 1] - sim[f][ci][j]
                    r = a.rotation_difference(s_) if a.length > 1e-9 and s_.length > 1e-9 else Quaternion()
                    anim = m[b].to_quaternion()
                    final = r @ anim
                    p_name = bones[j - 1] if j else parent
                    rel = rest_rot[p_name].inverted() @ rest_rot[b]
                    la = (anim_parent @ rel).inverted() @ anim
                    lf = (final_parent @ rel).inverted() @ final
                    d = la.inverted() @ lf
                    d.normalize()
                    if b in last:
                        d.make_compatible(last[b])
                    last[b] = d
                    for i, fc in enumerate(curves[b]):
                        kp = fc.keyframe_points[fi]
                        kp.co = (f, d[i])
                        kp.interpolation = 'LINEAR'
                    anim_parent, final_parent = anim, final
        for fcs in list(curves.values()) + cloth_curves:
            for fc in fcs:
                fc.update()
        ob['physics'] = 1.0
        sc.frame_set(keep)
        self.report({'INFO'}, f'Physics baked for frames {f0}-{f1} in {time.time() - t0:.1f}s')
        return {'FINISHED'}


def clear_physics(ob):
    cb = channelbag(ob, create=False)
    if cb:
        for f in [f for f in cb.fcurves if f.data_path.startswith('pose.bones["PHYS-')]:
            cb.fcurves.remove(f)
    for b in ob.pose.bones:
        if b.name.startswith('PHYS-'):
            b.rotation_quaternion = (1, 0, 0, 0)
            b.location = (0, 0, 0)


class DLRIG_OT_clear_physics(bpy.types.Operator):
    """Remove the baked secondary motion from the current action"""
    bl_idname = 'dlrig.clear_physics'
    bl_label = 'Clear'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clear_physics(rig(context))
        update(context)
        return {'FINISHED'}


@bpy.app.handlers.persistent
def dlrig_auto_physics(scene, *_args):
    if not scene.dlrig_phys_auto:
        return
    ob = bpy.context.active_object if bpy.context.active_object in rigs() else next(iter(rigs()), None)
    if not ob or not ob.animation_data or not ob.animation_data.action:
        return
    try:
        with bpy.context.temp_override(active_object=ob, object=ob):
            bpy.ops.dlrig.bake_physics()
    except RuntimeError as e:
        print('dlrig physics:', e)


# ---------------------------------------------------------------- panels


class DLRIG_PT_main(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = HERO + ' Rig'

    @classmethod
    def poll(cls, context):
        return rig(context) is not None

    def draw(self, context):
        ob = rig(context)
        col = self.layout.column(align=True)
        col.prop(context.scene.tool_settings, 'use_keyframe_insert_auto', text='Auto-Key (buttons set keys)',
                 toggle=True, icon='RECORD_OFF')
        col.separator()
        col.label(text='Show bones')
        grid = col.grid_flow(columns=2, align=True)
        for c in ob.data.collections_all:
            grid.prop(c, 'is_visible', text=c.name, toggle=True)
        col.separator()
        col.prop(context.scene, 'dlrig_strength', slider=True)
        if HAS_FACE:
            row = col.row(align=True)
            row.operator('dlrig.reset_face', icon='LOOP_BACK')
            row.operator('dlrig.key_face', icon='KEY_HLT')
        col.operator('dlrig.reset_pose', icon='ARMATURE_DATA')


class DLRIG_PT_body(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Arms & Legs (FK / IK)'
    bl_parent_id = 'DLRIG_PT_main'

    def draw(self, context):
        ob = rig(context)
        col = self.layout.column(align=True)
        for limb in limb_list(ob):
            for s, side in (('L', 'Left'), ('R', 'Right')):
                p = f'ik_{limb}_{s}'
                if p not in ob:
                    continue
                is_ik = ob[p] >= 0.5
                row = col.row(align=True)
                row.label(text=f'{side} {limb.replace("_", " ")}: {"IK" if is_ik else "FK"}',
                          icon='CON_KINEMATIC' if is_ik else 'BONE_DATA')
                op = row.operator('dlrig.snap', text='Switch to FK' if is_ik else 'Switch to IK')
                op.limb, op.side, op.to_ik = limb, s, not is_ik
        col.label(text='Switching keeps the pose', icon='INFO')
        col.label(text='IK: rings rotate bones too', icon='INFO')
        row = col.row(align=True)
        row.operator('dlrig.bake_ik', text='Animation to IK', icon='CON_KINEMATIC').limbs = 'ALL'
        row.operator('dlrig.bake_fk', text='Animation to FK', icon='BONE_DATA').limbs = 'ALL'
        for t in ob.get('tails', []):
            row = col.row(align=True)
            row.prop(ob, f'["ik_{t}"]', text=f'{t.replace("_", " ").capitalize()} IK', slider=True)
            if f'stretch_{t}' in ob:
                row.prop(ob, f'["stretch_{t}"]', text='Stretch', slider=True)
        if 'prop_chain' in ob:
            col.separator()
            col.label(text='Held prop: move its box', icon='INFO')
            col.prop(ob, '["prop_chain"]', text='Chain follows prop', slider=True)
        if 'weapon_hold' in ob:
            col.separator()
            floats = not ob.get('weapon_anchor', 'hand_').startswith('hand_')
            col.prop(ob, '["weapon_hold"]', text='Weapon follows body' if floats else 'Weapon in hand', slider=True)


def parent_row(layout, ob, bone, label):
    opts = ob.get('parent_options', {}).get(bone)
    if not opts:
        return
    row = layout.row(align=True)
    row.label(text=label)
    for i, lbl in enumerate(opts):
        op = row.operator('dlrig.switch_parent', text=lbl, depress=ob[bone + '_parent'] == i)
        op.bone, op.index = bone, i


class DLRIG_PT_ik(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'IK Options'
    bl_parent_id = 'DLRIG_PT_body'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return bool(ik_props(rig(context)))

    def draw(self, context):
        ob = rig(context)
        layout = self.layout
        for limb in limb_list(ob):
            for s, side in (('L', 'Left'), ('R', 'Right')):
                if f'ik_{limb}_{s}' not in ob:
                    continue
                box = layout.box()
                col = box.column(align=True)
                col.label(text=f'{side} {limb.replace("_", " ")}')
                row = col.row(align=True)
                row.prop(ob, f'["soft_{limb}_{s}"]', text='Soft', slider=True)
                row.prop(ob, f'["stretch_{limb}_{s}"]', text='Stretch', slider=True)
                _up, _lo, _end, ik, pole, _dist = limb_names(limb, s)
                parent_row(col, ob, ik, 'Follows')
                parent_row(col, ob, pole, 'Pole')
        if any(foot_names(ob, 'leg', s) for s in 'LR'):
            col = layout.column(align=True)
            col.prop(ob, '["foot_roll_break"]', text='Foot roll break')
            col.label(text='Heel arc: X rolls, Y banks, Z swivels', icon='INFO')
            col.label(text='Ankle ring turns the foot alone', icon='INFO')


class DLRIG_PT_follow(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Head & Arms Follow'
    bl_parent_id = 'DLRIG_PT_body'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return bool(rig(context).get('isolate'))

    def draw(self, context):
        ob = rig(context)
        col = self.layout.column(align=True)
        for prop, bone in ob['isolate'].items():
            follows = ob[prop] >= 0.5
            name = {'head': 'Head'}.get(bone, f'{"Left" if bone.endswith("_L") else "Right"} FK arm')
            row = col.row(align=True)
            row.label(text=f'{name}: {"follows body" if follows else "keeps its angle"}')
            row.operator('dlrig.switch_follow', text='Keep Angle' if follows else 'Follow').prop = prop


class DLRIG_PT_game(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Game Animations'
    bl_parent_id = 'DLRIG_PT_main'

    @classmethod
    def poll(cls, context):
        ob = rig(context)
        return ob is not None and os.path.exists(anim_library(ob))

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator('dlrig.game_anim', text='Load as New Action...', icon='ACTION').mode = 'NEW'
        col.operator('dlrig.game_anim', text='Insert at Current Frame...', icon='NLA_PUSHDOWN').mode = 'INSERT'
        col.operator('dlrig.game_anim', text='Copy Pose...', icon='POSE_HLT').mode = 'POSE'
        idle = idle_clip(context)
        col.label(text=f'Idle motion uses {idle[5:]}' if idle else 'No idle clip found', icon='INFO')
        col.separator()
        strip = idle_strip(rig(context))
        if strip:
            row = col.row(align=True)
            row.prop(context.scene, 'dlrig_idle_strength', text='Idle motion', slider=True)
            row.operator('dlrig.remove_idle', text='', icon='X')
        else:
            col.operator('dlrig.idle_layer', text='Add Idle Motion (breathing, sway)', icon='MOD_WAVE')


class DLRIG_PT_physics(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Physics (hair, cloth)'
    bl_parent_id = 'DLRIG_PT_main'

    @classmethod
    def poll(cls, context):
        t = bpy.data.texts.get(PHYS_TEXT)
        cfg = json.loads(t.as_string()) if t else {}
        return bool(cfg.get('chains') or cfg.get('cloth', {}).get('nodes'))

    def draw(self, context):
        ob = rig(context)
        sc = context.scene
        col = self.layout.column(align=True)
        row = col.row(align=True)
        row.operator('dlrig.bake_physics', icon='PHYSICS')
        row.operator('dlrig.clear_physics', icon='X')
        col.prop(sc, 'dlrig_phys_auto')
        col.prop(ob, '["physics"]', text='Mix', slider=True)
        col.separator()
        col.prop(sc, 'dlrig_phys_stiffness', slider=True)
        col.prop(sc, 'dlrig_phys_damping', slider=True)
        col.prop(sc, 'dlrig_phys_gravity', slider=True)
        col.prop(sc, 'dlrig_phys_collide')
        col.prop(sc, 'dlrig_phys_loop')


class DLRIG_PT_expressions(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Expressions'
    bl_parent_id = 'DLRIG_PT_main'

    def draw(self, context):
        grid = self.layout.grid_flow(columns=3, align=True)
        for name in EXPRESSIONS:
            grid.operator('dlrig.expression', text=name).name = name
        self.layout.label(text='Mouth shapes (lip sync)')
        grid = self.layout.grid_flow(columns=5, align=True)
        for name in VISEMES:
            op = grid.operator('dlrig.expression', text=name)
            op.name = name
            op.viseme = True
        self.layout.label(text='Shift-click adds to the current face', icon='INFO')


class DLRIG_PT_hands(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = HERO
    bl_label = 'Hands'
    bl_parent_id = 'DLRIG_PT_main'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        for name in HAND_POSES:
            row = self.layout.row(align=True)
            row.label(text=name)
            sides = HAND_POSES[name]
            for side, label in (('L', 'Left'), ('R', 'Right'), ('B', 'Both')):
                if side != 'B' and side not in sides or side == 'B' and len(sides) < 2:
                    continue
                op = row.operator('dlrig.hand', text=label)
                op.name = name
                op.side = side


def make_group_panel(title):
    class P(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = HERO
        bl_label = 'Sliders: ' + title
        bl_parent_id = 'DLRIG_PT_main'
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            pb = rig(context).pose.bones[FACE]
            col = self.layout.column(align=True)
            for k in GROUPS[title]:
                col.prop(pb, f'["{k}"]', text=k, slider=True)
    P.__name__ = 'DLRIG_PT_sliders_' + ''.join(ch for ch in title if ch.isalnum())
    return P


classes = [DLRIG_OT_expression, DLRIG_OT_reset_face, DLRIG_OT_key_face, DLRIG_OT_hand, DLRIG_OT_reset_pose,
           DLRIG_OT_snap, DLRIG_OT_bake_ik, DLRIG_OT_bake_fk, DLRIG_OT_switch_parent, DLRIG_OT_switch_follow,
           DLRIG_OT_game_anim, DLRIG_OT_idle_layer, DLRIG_OT_remove_idle,
           DLRIG_OT_bake_physics, DLRIG_OT_clear_physics, DLRIG_PT_main, DLRIG_PT_body, DLRIG_PT_ik, DLRIG_PT_follow,
           DLRIG_PT_game, DLRIG_PT_physics]
if HAS_FACE:
    classes += [DLRIG_PT_expressions]
if any(HAND_POSES.values()):
    classes += [DLRIG_PT_hands]
if HAS_FACE:
    classes += [make_group_panel(t) for t in GROUPS]


def show_tab():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for r in area.regions:
                    if r.type == 'UI':
                        try:
                            r.active_panel_category = HERO
                        except (TypeError, AttributeError):
                            pass
    return None


def register():
    S = bpy.types.Scene
    P = bpy.props
    S.dlrig_strength = P.FloatProperty(name='Strength', default=1.0, min=0.0, max=1.0)
    S.dlrig_phys_stiffness = P.FloatProperty(name='Stiffness', default=1.0, min=0.0, max=3.0,
                                              description='How strongly cloth and hair hold the animated shape')
    S.dlrig_phys_damping = P.FloatProperty(name='Damping', default=0.2, min=0.0, max=1.0,
                                            description='How quickly swinging settles')
    S.dlrig_phys_gravity = P.FloatProperty(name='Gravity', default=1.0, min=0.0, max=3.0)
    S.dlrig_idle_strength = P.FloatProperty(name='Idle Motion', default=1.0, min=0.0, max=2.0,
                                             update=lambda self, ctx: set_idle_strength(rig(ctx), self.dlrig_idle_strength))
    S.dlrig_phys_collide = P.BoolProperty(name='Collide with body', default=True)
    S.dlrig_phys_loop = P.BoolProperty(name='Looping animation', default=False,
                                        description='Simulate the range twice so the end flows into the start')
    S.dlrig_phys_auto = P.BoolProperty(name='Re-bake when playback starts', default=True,
                                        description='Bake physics automatically every time you press play')
    for c in classes:
        try:
            bpy.utils.unregister_class(getattr(bpy.types, c.__name__))
        except (AttributeError, RuntimeError):
            pass
        bpy.utils.register_class(c)
    bpy.app.timers.register(show_tab, first_interval=0.5)
    hs = bpy.app.handlers.animation_playback_pre
    for h in [h for h in hs if getattr(h, '__name__', '') == 'dlrig_auto_physics']:
        hs.remove(h)
    hs.append(dlrig_auto_physics)


register()
