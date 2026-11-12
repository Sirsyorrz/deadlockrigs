# SPDX-License-Identifier: GPL-3.0-or-later
"""Particle effects of an appended map simulated again at another moment or with another seed, playing as they
did. The simulation and shading are Divined's Deadlock Bridge (tools/deadlock_bridge), reading the game's files
directly.
"""
import os

import bpy


def bridge_particles():
    """The bridge's particle module, its game folder and cache set to the add-on's."""
    from . import output_dir, prefs, tools_module
    session = tools_module('deadlock_bridge.session')
    particles = tools_module('deadlock_bridge.blender.particles')
    map_particles = tools_module('map_particles')
    particles.model_source = map_particles.scene_models
    particles.light_source = map_particles.scene_light
    mapbuild = tools_module('mapbuild')
    given = prefs().deadlock_dir
    install = mapbuild.install_dir(mapbuild.find_pak(bpy.path.abspath(given) if given else None))
    if session.install_override != install:
        session.install_override = install
        session.reset()
    # rebuilt effects' textures are referenced from here by the .blend, so not a temporary folder
    session.cache_override = os.path.join(output_dir(), 'particles')
    return particles


def jitter(holder):
    from . import tools_module
    tools_module('map_particles').scene_jitter(holder.children)


def effect_holder(ob):
    """The empty holding an effect's objects, from it or one of its objects. Not a rig's clip effect: those follow
    its bones and are rebuilt with Add Clip Effects."""
    while ob is not None:
        if ob.type == 'EMPTY' and ob.get('dl_effect') and 'dl_time' in ob:
            return None if ob.get('dlr_clip_effect') else ob
        ob = ob.parent
    return None


class DLR_OT_rebuild_particle(bpy.types.Operator):
    bl_idname = 'dlr.rebuild_particle'
    bl_label = 'Rebuild'
    bl_description = ('Simulate the selected particle effects again where their empties are now, at their time and '
                      'seed, playing as they did. Effects of a linked map can\'t be rebuilt')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any((h := effect_holder(o)) and h.library is None for o in context.selected_objects)

    def execute(self, context):
        holders = {h.name: h for o in context.selected_objects
                   if (h := effect_holder(o)) is not None and h.library is None}
        particles = bridge_particles()
        empty = 0
        for holder in holders.values():
            try:
                if not particles.rebuild_particle_system(context, holder):
                    empty += 1
                jitter(holder)
            except Exception as e:
                self.report({'ERROR'}, f"'{holder.name}' failed: {e}")
                return {'CANCELLED'}
        if empty:
            self.report({'WARNING'}, f'{empty} of {len(holders)} effects draw nothing at their time')
        return {'FINISHED'}


def draw(layout, context):
    holder = effect_holder(context.active_object)
    if holder is None:
        return
    box = layout.box()
    path = holder['dl_effect']
    box.label(text=path.rsplit('/', 1)[-1], icon='PARTICLE_DATA')
    if 'dl_start_active' in holder and not holder['dl_start_active']:
        box.label(text="Started by the map's logic", icon='INFO')
    if holder.get('dl_animated'):
        if holder.get('dl_loop', True):
            box.label(text=f"Loops every {holder['dl_steps'] / holder['dl_rate']:.1f} s", icon='FILE_REFRESH')
        else:
            start = holder['dl_start'] * context.scene.render.fps / context.scene.render.fps_base
            box.label(text=f"Plays once from frame {start:.0f}", icon='PLAY')
    else:
        box.label(text='Still', icon='PAUSE')
    if holder.library is not None:
        box.label(text='In a linked map: append it to rebuild', icon='LINKED')
        return
    row = box.row(align=True)
    row.prop(holder, '["dl_time"]', text='Time')
    row.prop(holder, '["dl_seed"]', text='Seed')
    box.operator('dlr.rebuild_particle', icon='FILE_REFRESH')


CLASSES = (DLR_OT_rebuild_particle,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
