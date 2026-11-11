# SPDX-License-Identifier: GPL-3.0-or-later
"""Clip Effects: the particle effects a hero's clip plays in the game (ability effects such as Dynamo's singularity,
the clip's own sparks and glows) added to its rig for the clip it plays (tools/rig_effects.py), and any of its
abilities' effects added by hand from the current frame (Ability Effects)."""
import re

import bpy

from . import particles

TAG = 'dlr_clip_effect'
# set on effects added from Ability Effects: Add Clip Effects leaves them be
PICKED = 'dlr_picked_effect'
KINDS = {'hero': 'on hero', 'cast': 'cast (guess)', 'target': 'hit target', 'area': 'area'}
_found = {}
_catalogs = {}
_ability_items = []


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


def ability_items(self, context):
    arm = rig_of(context.active_object)
    # Blender keeps only the strings of the list it was last given
    _ability_items[:] = [(a, title, '') for a, title, _ in catalog_of(arm)] if arm else []
    return _ability_items or [('NONE', 'No abilities', '')]


def listed_for(context):
    arm = rig_of(context.active_object)
    return f'{arm.name}|{context.window_manager.dlr_fx_ability}' if arm else ''


def fill_list(self, context):
    wm = context.window_manager
    wm.dlr_fx_list.clear()
    arm = rig_of(context.active_object)
    found = next((fx for a, _, fx in catalog_of(arm) if a == wm.dlr_fx_ability), []) if arm else []
    for e in found:
        item = wm.dlr_fx_list.add()
        item.path, item.kind = e['path'], e['kind']
        folder, stem = e['path'].removesuffix('.vpcf').rsplit('/', 2)[-2:]
        item.label = stem.removeprefix(folder + '_').replace('_', ' ')
        item.use = e['kind'] != 'target'
    wm.dlr_fx_for = listed_for(context)


class DLR_PG_ability_effect(bpy.types.PropertyGroup):
    path: bpy.props.StringProperty()
    label: bpy.props.StringProperty()
    kind: bpy.props.StringProperty()
    use: bpy.props.BoolProperty(name='Add', description='Add this effect')


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


class DLR_OT_list_ability_effects(bpy.types.Operator):
    bl_idname = 'dlr.list_ability_effects'
    bl_label = 'List Effects'
    bl_description = "List the chosen ability's effects"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        fill_list(None, context)
        return {'FINISHED'}


class DLR_OT_ability_effects(bpy.types.Operator):
    bl_idname = 'dlr.ability_effects'
    bl_label = 'Add Ability Effects'
    bl_description = ("Add the ticked effects of the ability to the rig, playing from the current frame to the end of "
                      "the action (or the scene). For effects no clip says the timing of: buffs, domes, walls, "
                      "what a hit hero shows")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return (rig_of(context.active_object) is not None and wm.dlr_fx_for == listed_for(context)
                and any(i.use for i in wm.dlr_fx_list))

    def execute(self, context):
        from . import tools_module
        wm = context.window_manager
        arm = rig_of(context.active_object)
        ability = next(((a, title, fx) for a, title, fx in catalog_of(arm) if a == wm.dlr_fx_ability), None)
        if ability is None:
            return {'CANCELLED'}
        ticked = {i.path for i in wm.dlr_fx_list if i.use}
        picked = [e for e in ability[2] if e['path'] in ticked]
        sc = context.scene
        fps = sc.render.fps / sc.render.fps_base
        start = float(sc.frame_current)
        ends = [last for _, _, last in clips_of(arm) if last > start + 1]
        end = max(ends) if ends else float(sc.frame_end)
        if end <= start + 1:
            end = start + 2 * fps
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        try:
            holders = rig_effects.add(context, arm, ability[1], start, (end - start) / fps, effects=picked)
        except Exception as e:
            self.report({'ERROR'}, f'Ability effects failed: {e}')
            return {'CANCELLED'}
        for holder in holders:
            holder[PICKED] = ability[0]
            holder.name = re.sub(r'\.\d{3}$', '', holder.name)
            particles.jitter(holder)
        self.report({'INFO'}, f'{len(holders)} effects of {ability[1]} added' if holders
                    else f'The effects of {ability[1]} draw nothing here')
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
        wm = context.window_manager
        box = layout.box()
        box.label(text='Ability Effects', icon='SHADERFX')
        box.prop(wm, 'dlr_fx_ability', text='')
        if wm.dlr_fx_for != listed_for(context):
            box.operator('dlr.list_ability_effects', icon='VIEWZOOM')
        else:
            col = box.column(align=True)
            for item in wm.dlr_fx_list:
                row = col.row(align=True).split(factor=0.7)
                row.prop(item, 'use', text=item.label)
                row.label(text=KINDS.get(item.kind, item.kind))
            box.operator('dlr.ability_effects', icon='PARTICLES')
        added = sorted({o[TAG] for o in arm.children if o.get(TAG)})
        if added:
            row = layout.row()
            row.label(text=f'Added: {", ".join(added)}')
            row.operator('dlr.remove_clip_effects', text='', icon='X')


CLASSES = (DLR_PG_ability_effect, DLR_OT_clip_effects, DLR_OT_list_ability_effects, DLR_OT_ability_effects,
           DLR_OT_remove_clip_effects)
WM_PROPS = {
    'dlr_fx_ability': lambda: bpy.props.EnumProperty(name='Ability', items=ability_items, update=fill_list,
                                                     description="The hero ability whose effects to list"),
    'dlr_fx_list': lambda: bpy.props.CollectionProperty(type=DLR_PG_ability_effect),
    'dlr_fx_for': lambda: bpy.props.StringProperty(),
}


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
    for name, prop in WM_PROPS.items():
        setattr(bpy.types.WindowManager, name, prop())
    attach_panel()
    bpy.app.timers.register(_watch_parent, first_interval=1.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_watch_parent):
        bpy.app.timers.unregister(_watch_parent)
    if DLR_PT_clip_effects.is_registered:
        bpy.utils.unregister_class(DLR_PT_clip_effects)
    for name in WM_PROPS:
        delattr(bpy.types.WindowManager, name)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
