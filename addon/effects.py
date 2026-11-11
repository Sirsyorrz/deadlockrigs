# SPDX-License-Identifier: GPL-3.0-or-later
"""Clip Effects: the particle effects a hero's clip plays in the game (ability effects such as Dynamo's singularity,
the clip's own sparks and glows) added to its rig for the clip it plays (tools/rig_effects.py), and any of its
abilities' effects over the clips of theirs the action plays (Ability Effects)."""
import re

import bpy

from . import particles

TAG = 'dlr_clip_effect'
# set on effects added by Add Ability Effects: Add Clip Effects leaves them be
PICKED = 'dlr_picked_effect'
_found = {}
_catalogs = {}
_abilities = {}


def _is_rig(ob):
    return ob is not None and ob.type == 'ARMATURE' and 'dlrig' in ob


def rig_of(ob):
    """The rig the Game Animations panel shows for ``ob`` (the active object): its own, a deformed mesh's, or the
    file's first when the selection isn't part of one."""
    start = ob
    while ob is not None:
        if _is_rig(ob):
            return ob
        ob = ob.parent
    arm = start.find_armature() if start is not None and start.type == 'MESH' else None
    if _is_rig(arm):
        return arm
    return next((o for o in bpy.data.objects if _is_rig(o)), None)


def clips_of(arm):
    """[(clip, first frame, last frame)] of the game clips the rig's action plays: every clip loaded or inserted
    (rig_ui.py records them), or the clip an older action is named after. Empty without an action."""
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        return []
    if 'dlrig_clips' in act:
        return [(c['clip'], c['first'], c['last']) for c in act['dlrig_clips']]
    name = re.sub(r'\.\d{3}$', '', act.name).removeprefix('game_')
    first, last = act.frame_range
    return [(name, first, last)]


def effects_of(arm, clip):
    """The effects the game plays with a clip of this rig's hero (cached per hero and clip)."""
    from . import tools_module
    key = (arm['dlrig'], clip)
    if key not in _found:
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        clip_effects = tools_module('clip_effects')
        from deadlock_bridge import session
        try:
            fx = clip_effects.for_model(session.content(), rig_effects.model_of(arm))
            _found[key] = fx.effects(clip) if clip else []
        except Exception as e:
            # not kept: the game folder may just not be set yet
            print(f'Clip effects of {clip}: {e}')
            return []
    return _found[key]


def catalog_of(arm):
    """[(ability, title, [effect])] of the rig's hero (ClipEffects.catalog), cached per hero."""
    from . import tools_module
    if arm['dlrig'] not in _catalogs:
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        clip_effects = tools_module('clip_effects')
        from deadlock_bridge import session
        try:
            _catalogs[arm['dlrig']] = clip_effects.for_model(session.content(), rig_effects.model_of(arm)).catalog()
        except Exception as e:
            print(f'Ability effects of {arm["dlrig"]}: {e}')
            return []
    return _catalogs[arm['dlrig']]


def abilities_of(arm, clip):
    """The hero's abilities (catalog roots) a clip of the rig plays for, cached per hero and clip."""
    from . import tools_module
    key = (arm['dlrig'], clip)
    if key not in _abilities:
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        clip_effects = tools_module('clip_effects')
        from deadlock_bridge import session
        try:
            _abilities[key] = clip_effects.for_model(session.content(), rig_effects.model_of(arm)).clip_abilities(clip)
        except Exception as e:
            print(f'Abilities of {clip}: {e}')
            return []
    return _abilities[key]


def ability_spans(arm):
    """[(ability, title, first, last, [clip])]: the frames each ability's clips in the action cover, clips back to
    back (or overlapping) as one span."""
    titles = {a: title for a, title, _ in catalog_of(arm)}
    spans = []
    for clip, first, last in sorted(clips_of(arm), key=lambda c: c[1]):
        for ability in abilities_of(arm, clip):
            if ability not in titles:
                continue
            span = next((s for s in reversed(spans) if s[0] == ability), None)
            if span is not None and first - span[3] <= 1.5:
                span[3] = max(span[3], last)
                span[4].append(clip)
            else:
                spans.append([ability, titles[ability], first, last, [clip]])
    return spans


def merged(clips, arm):
    """Back to back copies of a clip whose effects all loop (a channel inserted twice) as one span, so the effect
    carries on instead of starting over."""
    out = []
    for clip, first, last in clips:
        if (out and out[-1][0] == clip and first - out[-1][2] <= 1.5
                and all(e['loop'] for e in effects_of(arm, clip))):
            out[-1] = (clip, out[-1][1], last)
        else:
            out.append((clip, first, last))
    return out


class DLR_OT_clip_effects(bpy.types.Operator):
    bl_idname = 'dlr.clip_effects'
    bl_label = 'Add Clip Effects'
    bl_description = ("Add the particle effects the game plays with each clip in the rig's action (loaded or "
                      "inserted Game Animations), over that clip's frames. Replaces the ones added before")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = rig_of(context.active_object)
        return arm is not None and bool(clips_of(arm))

    def execute(self, context):
        from . import tools_module
        arm = rig_of(context.active_object)
        clips = merged([c for c in clips_of(arm) if effects_of(arm, c[0])], arm)
        if not clips:
            self.report({'WARNING'}, 'The game plays no effects with these clips')
            return {'CANCELLED'}
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        old = [o for o in arm.children if o.get(TAG) and not o.get(PICKED)]
        sc = context.scene
        fps = sc.render.fps / sc.render.fps_base
        wm = context.window_manager
        wm.progress_begin(0, 100)
        holders = []
        try:
            for i, (clip, first, last) in enumerate(clips):
                def progress(n, total, i=i):
                    wm.progress_update(100 * (i + n / max(total, 1)) // len(clips))
                holders += rig_effects.add(context, arm, clip, first, (last - first) / fps, progress=progress)
        except Exception as e:
            rig_effects.remove_holders(holders)
            self.report({'ERROR'}, f'Clip effects failed: {e}')
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        rig_effects.remove_holders([o for o in old if not o.get(PICKED)])
        for holder in holders:
            # built while the old ones still held their names
            holder.name = re.sub(r'\.\d{3}$', '', holder.name)
            particles.jitter(holder)
        names = ', '.join(dict.fromkeys(c[0] for c in clips))
        self.report({'INFO'}, f'{len(holders)} effects added for {names}' if holders
                    else f'The effects of {names} draw nothing here')
        return {'FINISHED'}


class DLR_OT_ability_effects(bpy.types.Operator):
    bl_idname = 'dlr.ability_effects'
    bl_label = 'Add Ability Effects'
    bl_description = ("Add the effects of the abilities the action's clips belong to, over those clips' frames: "
                      "buffs, domes, walls and the rest no clip times itself. Replaces the ones added before")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = rig_of(context.active_object)
        return arm is not None and bool(clips_of(arm))

    def execute(self, context):
        from . import tools_module
        arm = rig_of(context.active_object)
        effects = {a: fx for a, _, fx in catalog_of(arm)}
        jobs = []
        spans = ability_spans(arm)
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        from deadlock_bridge import session
        fx = tools_module('clip_effects').for_model(session.content(), rig_effects.model_of(arm))
        sc = context.scene
        fps = sc.render.fps / sc.render.fps_base
        for ability, title, first, last, clips in spans:
            # a dome or buff outlasts its cast clip
            last = max(last, first + fx.ability_duration(ability) * fps)
            timed = {e['path'] for clip in clips for e in effects_of(arm, clip)}
            # what shows on whoever is hit goes on someone else; the clip's own are Add Clip Effects'
            found = [e for e in effects[ability] if e['kind'] != 'target' and e['path'] not in timed]
            if found:
                jobs.append((ability, title, first, last, found))
        if not jobs:
            self.report({'WARNING'}, f'All effects of {", ".join(dict.fromkeys(s[1] for s in spans))} are timed by '
                        'their clips: Add Clip Effects' if spans else 'These clips are of no ability')
            return {'CANCELLED'}
        old = [o for o in arm.children if o.get(PICKED)]
        wm = context.window_manager
        wm.progress_begin(0, 100)
        holders = []
        try:
            for i, (ability, title, first, last, found) in enumerate(jobs):
                def progress(n, total, i=i):
                    wm.progress_update(100 * (i + n / max(total, 1)) // len(jobs))
                added = rig_effects.add(context, arm, title, first, max(last - first, 1.0) / fps, effects=found,
                                        progress=progress)
                for holder in added:
                    holder[PICKED] = ability
                holders += added
        except Exception as e:
            rig_effects.remove_holders(holders)
            self.report({'ERROR'}, f'Ability effects failed: {e}')
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        rig_effects.remove_holders(old)
        for holder in holders:
            holder.name = re.sub(r'\.\d{3}$', '', holder.name)
            particles.jitter(holder)
        names = ', '.join(dict.fromkeys(j[1] for j in jobs))
        self.report({'INFO'}, f'{len(holders)} effects added for {names}' if holders
                    else f'The effects of {names} draw nothing here')
        return {'FINISHED'}


class DLR_OT_remove_clip_effects(bpy.types.Operator):
    bl_idname = 'dlr.remove_clip_effects'
    bl_label = 'Remove'
    bl_description = "Remove the rig's clip effects"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = rig_of(context.active_object)
        return arm is not None and any(o.get(TAG) for o in arm.children)

    def execute(self, context):
        from . import tools_module
        particles.bridge_particles()
        tools_module('rig_effects').remove(rig_of(context.active_object))
        return {'FINISHED'}


class DLR_PT_clip_effects(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hero Rig'
    bl_label = 'Clip Effects'
    bl_parent_id = 'DLRIG_PT_game'

    @classmethod
    def poll(cls, context):
        return rig_of(context.active_object) is not None

    def draw(self, context):
        layout = self.layout
        arm = rig_of(context.active_object)
        clips = clips_of(arm)
        if not clips:
            layout.label(text='Load a Game Animation first', icon='INFO')
        else:
            col = layout.column(align=True)
            for clip, first, last in clips:
                key = (arm['dlrig'], clip)
                none = key in _found and not _found[key]
                col.label(text=f'{int(first)}-{int(round(last))}  {clip}' + ('  (no effects)' if none else ''),
                          icon='ACTION')
            layout.operator('dlr.clip_effects', icon='PARTICLES')
            layout.operator('dlr.ability_effects', icon='SHADERFX')
        added = sorted({o[TAG] for o in arm.children if o.get(TAG)})
        if added:
            row = layout.row()
            row.label(text=f'Added: {", ".join(added)}')
            row.operator('dlr.remove_clip_effects', text='', icon='X')


CLASSES = (DLR_OT_clip_effects, DLR_OT_ability_effects, DLR_OT_remove_clip_effects)


_parent = [None]


def attach_panel():
    """(Re)registers the panel under the rig panel's Game Animations. That panel is registered by rig_ui.py,
    whenever a rig is opened or appended (and by a rig's own text block), and a sub-panel registered before its
    parent, or left over from a parent that was replaced, doesn't show."""
    if DLR_PT_clip_effects.is_registered:
        bpy.utils.unregister_class(DLR_PT_clip_effects)
    parent = getattr(bpy.types, DLR_PT_clip_effects.bl_parent_id, None)
    _parent[0] = parent
    if parent is not None:
        bpy.utils.register_class(DLR_PT_clip_effects)


def _watch_parent():
    """Re-attaches the panel when Game Animations was registered anew behind the add-on's back (a rig's text
    block run by hand, or on load with auto run scripts on)."""
    if getattr(bpy.types, DLR_PT_clip_effects.bl_parent_id, None) is not _parent[0]:
        attach_panel()
    return 1.0


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    attach_panel()
    bpy.app.timers.register(_watch_parent, first_interval=1.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_watch_parent):
        bpy.app.timers.unregister(_watch_parent)
    if DLR_PT_clip_effects.is_registered:
        bpy.utils.unregister_class(DLR_PT_clip_effects)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
