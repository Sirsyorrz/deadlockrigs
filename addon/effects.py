# SPDX-License-Identifier: GPL-3.0-or-later
"""Clip Effects: the particle effects a hero's clip plays in the game (ability effects such as Dynamo's singularity,
the clip's own sparks and glows) added to its rig for the clip it plays (tools/rig_effects.py)."""
import re

import bpy

from . import particles

TAG = 'dlr_clip_effect'
_found = {}


def rig_of(ob):
    while ob is not None:
        if ob.type == 'ARMATURE' and 'dlrig' in ob:
            return ob
        ob = ob.parent
    return None


def clip_of(arm):
    """The clip the rig's action is (a Game Animation loaded as a new action is named after it), or ''."""
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        return ''
    name = re.sub(r'\.\d{3}$', '', act.name)
    return name.removeprefix('game_')


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
            fx = clip_effects.ClipEffects(session.content(), rig_effects.model_of(arm))
            _found[key] = fx.effects(clip) if clip else []
        except Exception as e:
            # not kept: the game folder may just not be set yet
            print(f'Clip effects of {clip}: {e}')
            return []
    return _found[key]


class DLR_OT_clip_effects(bpy.types.Operator):
    bl_idname = 'dlr.clip_effects'
    bl_label = 'Add Clip Effects'
    bl_description = ("Add the particle effects the game plays with the rig's current clip, from the frame its "
                      "action starts. Replaces the ones added before for that clip")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = rig_of(context.active_object)
        return arm is not None and bool(clip_of(arm))

    def execute(self, context):
        from . import tools_module
        arm = rig_of(context.active_object)
        clip = clip_of(arm)
        if not effects_of(arm, clip):
            self.report({'WARNING'}, f'The game plays no effects with {clip}')
            return {'CANCELLED'}
        particles.bridge_particles()
        rig_effects = tools_module('rig_effects')
        old = [o for o in arm.children if o.get(TAG) == clip]
        first, last = arm.animation_data.action.frame_range
        sc = context.scene
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            holders = rig_effects.add(context, arm, clip, first, (last - first) / (sc.render.fps / sc.render.fps_base),
                                      progress=lambda n, total: wm.progress_update(100 * n // max(total, 1)))
        except Exception as e:
            self.report({'ERROR'}, f'Clip effects failed: {e}')
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        rig_effects.remove_holders(old)
        for holder in holders:
            # built while the old ones still held their names
            holder.name = re.sub(r'\.\d{3}$', '', holder.name)
            particles.jitter(holder)
        self.report({'INFO'}, f'{len(holders)} effects added for {clip}' if holders
                    else f'The effects of {clip} draw nothing here')
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
        clip = clip_of(arm)
        if not clip:
            layout.label(text='Load a Game Animation first', icon='INFO')
        else:
            layout.label(text=clip, icon='ACTION')
            key = (arm['dlrig'], clip)
            if key in _found and not _found[key]:
                layout.label(text='No effects in the game for this clip', icon='INFO')
            else:
                layout.operator('dlr.clip_effects', icon='PARTICLES')
        added = sorted({o[TAG] for o in arm.children if o.get(TAG)})
        if added:
            row = layout.row()
            row.label(text=f'Added: {", ".join(added)}')
            row.operator('dlr.remove_clip_effects', text='', icon='X')


CLASSES = (DLR_OT_clip_effects, DLR_OT_remove_clip_effects)


def attach_panel():
    """(Re)registers the panel under the rig panel's Game Animations. That panel is registered by rig_ui.py,
    whenever a rig is opened or appended (and by a rig's own text block), and a sub-panel registered before its
    parent, or left over from a parent that was replaced, doesn't show."""
    if DLR_PT_clip_effects.is_registered:
        bpy.utils.unregister_class(DLR_PT_clip_effects)
    if hasattr(bpy.types, DLR_PT_clip_effects.bl_parent_id):
        bpy.utils.register_class(DLR_PT_clip_effects)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    attach_panel()


def unregister():
    if DLR_PT_clip_effects.is_registered:
        bpy.utils.unregister_class(DLR_PT_clip_effects)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
