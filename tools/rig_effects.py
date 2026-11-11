"""The particle effects a clip plays in the game (clip_effects.py) built onto a rig: simulated by the bridge with the
control points where the game puts them, on the rig's bones as the clip moves them, and played with the scene's
time from the frame the clip starts at. Ability effects loop with the clip; a clip's own events play once from
their moment. Each effect's objects only show while the clip plays.
"""
import math

import bpy
import numpy as np
from clip_effects import ClipEffects, quat_matrix
from deadlock_bridge import session
from deadlock_bridge.blender import particles as bridge
from deadlock_bridge.core import particles as plib
from mathutils import Matrix

UNIT = 0.0254
# game model space -> the rig's: the glTF the rig is built from is a quarter turn about Z from it (as maps are)
FRAME = Matrix.Rotation(-math.pi / 2, 4, 'Z')
FRAME3 = np.array(FRAME.to_3x3(), np.float64)
RATE = 30.0
# how long a clip's own effect keeps playing past the clip's end (sparks settling, smoke clearing)
TAIL = 1.5
# a clip event's effect that never ends by itself is cut here
LONGEST = 8.0
ATTACHED = ('PATTACH_POINT', 'PATTACH_POINT_FOLLOW')
TAG = 'dlr_clip_effect'


def model_of(arm):
    from clip_effects import model_of as hero_model
    return arm.get('dlr_model') or hero_model(arm['dlrig'])


class RigPoints:
    """The rig's transforms an effect's control points sit on, in game units and axes of the rig's space, sampled
    at the scene's frames."""

    def __init__(self, context, arm, fx):
        self.context, self.arm = context, arm
        bind = fx.bind_pose()
        self.attach = {}
        for name, (bone, offset, rotation) in fx.attachments().items():
            rest = arm.data.bones.get(bone)
            if rest is None or bone not in bind:
                continue
            r, t = bind[bone]
            at = np.eye(4)
            at[:3, :3] = FRAME3 @ r @ quat_matrix(rotation)
            at[:3, 3] = FRAME3 @ (t + r @ np.asarray(offset, np.float64)) * UNIT
            # kept relative to the rig's own bone: its rest frame is turned from the game's (mirror.py, rolls)
            self.attach[name] = (bone, np.linalg.inv(np.array(rest.matrix_local)) @ at)
        self.root = 'root_motion' if 'root_motion' in arm.pose.bones else None
        self._wanted = {bone for bone, _ in self.attach.values()} | ({self.root} if self.root else set())
        self._cache = {}

    def _pose(self, frame):
        key = round(frame, 3)
        if key not in self._cache:
            scene = self.context.scene
            scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))
            self._cache[key] = {b.name: np.array(b.matrix) for b in self.arm.pose.bones
                                if b.name in self._wanted}
        return self._cache[key]

    def sample(self, effect, frames):
        """{frame: {cp: (position, rotation)}} for the effect's drivers at the given scene frames. Drivers that
        don't follow take the first frame's."""
        drivers = effect['drivers'] or [{'cp': 0, 'type': 'PATTACH_ABSORIGIN_FOLLOW', 'attachment': '',
                                         'offset': (0, 0, 0), 'angles': (0, 0, 0), 'entity': 'self'}]
        follow = [d for d in drivers if d['type'].endswith('_FOLLOW')]
        first = {d['cp']: self._point(d, self._pose(frames[0])) for d in drivers}
        first.setdefault(0, (np.zeros(3), np.eye(3)))
        if first[0][1] is None:
            # the effect's origin needs a frame even when its driver is a literal
            first[0] = (first[0][0], np.eye(3))
        out = {frames[0]: first}
        for frame in frames[1:]:
            out[frame] = {**first, **{d['cp']: self._point(d, self._pose(frame)) for d in follow}} if follow \
                else first
        return out

    def _point(self, d, pose):
        offset = np.asarray(d['offset'], np.float64)
        if d['type'] == 'PATTACH_WORLDORIGIN':
            # a literal: values the effect reads (a radius), or a spot in the effect's own space
            return offset, None
        m = np.eye(4)
        if d['type'] in ATTACHED and d['attachment'] in self.attach:
            bone, rel = self.attach[d['attachment']]
            m = pose[bone] @ rel
        elif self.root:
            m = pose[self.root].copy()
            # the entity's origin: on the ground, upright, where root motion takes it
            m[:3, :3] = np.eye(3)
            m[2, 3] = 0.0
        rotation = FRAME3.T @ m[:3, :3]
        rotation = rotation / np.linalg.norm(rotation, axis=0)
        position = FRAME3.T @ m[:3, 3] / UNIT + rotation @ offset
        return position, rotation


def add(context, arm, clip, start_frame, duration=None, seed=0, progress=None):
    """Builds every effect clip ``clip`` starts onto the rig, the clip playing from ``start_frame`` for ``duration``
    seconds (the game's clip length if None). Returns the effects' empties."""
    content = session.content()
    fx = ClipEffects(content, model_of(arm))
    effects = fx.effects(clip)
    duration = duration or fx.duration(clip)
    if not effects or duration <= 0:
        return []
    scene = context.scene
    fps = scene.render.fps / scene.render.fps_base
    library = plib.library(content)
    rig_points = RigPoints(context, arm, fx)
    target = bridge._Target.for_scene(context, UNIT)
    target.collection = arm.users_collection[0]
    builder = bridge._ObjectBuilder(target)
    frame_was = scene.frame_current, scene.frame_subframe
    holders = []
    try:
        for n, effect in enumerate(effects):
            if progress:
                progress(n, len(effects))
            definition = library.get(effect['path'])
            if definition is None:
                continue
            begin = start_frame + effect['start'] * duration * fps
            if effect['loop']:
                warm, length = 2.0, duration
                steps = round(length * RATE)

                def clip_frame(age, warm=warm, length=length):
                    return start_frame + ((age - warm) % length) * fps
            else:
                warm, length = 0.0, min(duration * (1.0 - effect['start']) + TAIL, LONGEST)
                steps = round(length * RATE)

                def clip_frame(age, begin=begin, end=start_frame + duration * fps):
                    return min(begin + age * fps, end)
            # the clip's frames the bake samples, on its own steps
            frames = sorted({round(clip_frame(warm + k / RATE), 3) for k in range(steps + 1)})
            sampled = rig_points.sample(effect, frames)

            def moving(age, sampled=sampled, clip_frame=clip_frame):
                return sampled[min(sampled, key=lambda f: abs(f - clip_frame(age)))]

            points = sampled[frames[0]]
            ambient = bridge._scene_light(context, np.array(arm.matrix_world.translation) / target.scale,
                                          target.scale, target)
            playback = {'loop': effect['loop'], 'length': length, 'start': begin / fps, 'rate': RATE}
            holder = bridge._build_played(builder, f'{clip} {bridge._stem(effect["path"])}', definition, points,
                                          warm, seed + n, ambient, playback, local=True, unset_value=1.0,
                                          moving=moving)
            if holder is None:
                continue
            holder.parent = arm
            holder.matrix_parent_inverse = Matrix()
            holder.matrix_basis = FRAME
            holder[TAG] = clip
            holder['dlr_clip_start'] = float(start_frame)
            show_between(holder, begin if not effect['loop'] else start_frame,
                         start_frame + duration * fps if effect['loop'] else begin + length * fps)
            holders.append(holder)
    finally:
        scene.frame_set(frame_was[0], subframe=frame_was[1])
    return holders


def show_between(holder, first, last):
    """The effect's objects hidden outside the clip's frames (a looping bake would otherwise play on forever)."""
    for ob in holder.children_recursive:
        for frame, hidden in ((first - 1, True), (first, False), (math.ceil(last), True)):
            ob.hide_viewport = ob.hide_render = hidden
            ob.keyframe_insert('hide_viewport', frame=frame)
            ob.keyframe_insert('hide_render', frame=frame)
        for fc in fcurves(ob.animation_data.action):
            for k in fc.keyframe_points:
                k.interpolation = 'CONSTANT'
        ob.hide_viewport = ob.hide_render = False


def fcurves(action):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def remove(arm, clip=None):
    """The rig's clip effects (of one clip, or all) and their objects."""
    remove_holders([o for o in arm.children if o.get(TAG) and (clip is None or o[TAG] == clip)])


def remove_holders(holders):
    for holder in holders:
        bridge._clear_children(holder)
        bpy.data.objects.remove(holder)
