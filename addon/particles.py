# SPDX-License-Identifier: GPL-3.0-or-later
"""Particle effects: any of the game's ~18,000 particle systems placed at the 3D cursor, and effects (placed, or
built with an appended map) simulated again at another moment or with another seed. The simulation and shading are
Divined's Deadlock Bridge (tools/deadlock_bridge), reading the game's files directly.

An effect plays with the scene's time: looping (as a map's effects do), or once from the frame it was placed at; or
it is a still of one moment.

Placed effects go into a 'Deadlock Particles' collection, lit as the nearest map particle model is lit (else fully,
as the game's particle editor shows them) and fogged with the map's fog when the scene has one.
"""
import os
import sys

import bpy

SEARCH_LIMIT = 400
_paths = []


def bridge_particles():
    """The bridge's particle module, its game folder and cache set to the add-on's."""
    from . import output_dir, prefs, tools_dir, tools_module
    if tools_dir() not in sys.path:
        sys.path.insert(0, tools_dir())
    from deadlock_bridge import session
    from deadlock_bridge.blender import particles
    map_particles = tools_module('map_particles')
    particles.model_source = map_particles.scene_models
    particles.light_source = map_particles.scene_light
    mapbuild = tools_module('mapbuild')
    given = prefs().deadlock_dir
    install = mapbuild.install_dir(mapbuild.find_pak(bpy.path.abspath(given) if given else None))
    if session.install_override != install:
        session.install_override = install
        session.reset()
    # placed effects' textures are referenced from here by the .blend, so not a temporary folder
    session.cache_override = os.path.join(output_dir(), 'particles')
    return particles


def effect_paths():
    if not _paths:
        particles = bridge_particles()
        from deadlock_bridge import session
        _paths.extend(particles.effect_paths(session.content()))
    return _paths


def search(self, context, text):
    """The effects whose path holds every word typed, in any order."""
    try:
        paths = effect_paths()
    except Exception:
        return []
    words = text.lower().replace('\\', '/').split()
    found = []
    for path in paths:
        if all(w in path for w in words):
            found.append(path)
            if len(found) == SEARCH_LIMIT:
                break
    return found


EFFECT_DESC = 'A particle system from the game\'s files; type words of its path to search ("candle flame")'
TIME_DESC = 'How long the effect has been playing when it starts looping, or when it is shown as a still, in seconds'
BUSIEST_DESC = ('Start at the moment the effect draws the most (many are bursts over in a fraction of a second); '
                'off, at the time set')
SEED_DESC = "Which of the effect's random variations to show"
PLAYBACK_ITEMS = (
    ('LOOP', 'Loop', "Play with the timeline in a seamless loop, from the time set (or its busiest moment) on, as a "
                     "map's effects do"),
    ('ONCE', 'Once', 'Play from its start at the current frame, for the length set (bursts, explosions)'),
    ('STILL', 'Still', 'One moment of it, frozen'),
)
LENGTH_DESC = 'How long to play it for, in seconds'


class DLR_ParticleSettings(bpy.types.PropertyGroup):
    effect: bpy.props.StringProperty(name='Effect', description=EFFECT_DESC, search=search)
    playback: bpy.props.EnumProperty(name='Playback', items=PLAYBACK_ITEMS, default='LOOP')
    busiest: bpy.props.BoolProperty(name='Busiest Moment', default=True, description=BUSIEST_DESC)
    time: bpy.props.FloatProperty(name='Time', default=2.0, min=0.0, soft_max=20.0, step=10, precision=2,
                                  description=TIME_DESC)
    length: bpy.props.FloatProperty(name='Length', default=3.0, min=0.1, soft_max=20.0, step=10, precision=1,
                                    description=LENGTH_DESC)
    seed: bpy.props.IntProperty(name='Seed', default=0, min=0, description=SEED_DESC)


def scene_seconds(scene):
    return scene.frame_current * scene.render.fps_base / scene.render.fps


def playback(settings, scene):
    """The bridge's playback for the settings, or None for a still."""
    if settings.playback == 'LOOP':
        return {'loop': True}
    if settings.playback == 'ONCE':
        return {'loop': False, 'length': settings.length, 'start': scene_seconds(scene)}
    return None


def jitter(holder):
    from . import tools_module
    tools_module('map_particles').scene_jitter(holder.children)


def effect_holder(ob):
    """The empty holding an effect's objects, from it or one of its objects."""
    while ob is not None:
        if ob.type == 'EMPTY' and ob.get('dl_effect') and 'dl_time' in ob:
            return ob
        ob = ob.parent
    return None


class DLR_OT_add_particle(bpy.types.Operator):
    bl_idname = 'dlr.add_particle'
    bl_label = 'Add at 3D Cursor'
    bl_description = "Place one of Deadlock's particle effects at the 3D cursor, playing with the timeline or still"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.dlr_particles
        path = settings.effect.strip().replace('\\', '/').lower()
        if path.endswith('_c'):
            path = path[:-2]
        if not path:
            self.report({'ERROR'}, 'Choose an effect first')
            return {'CANCELLED'}
        once = settings.playback == 'ONCE'
        try:
            particles = bridge_particles()
            holder = particles.spawn_particle_system(context, path, context.scene.cursor.matrix,
                                                     None if settings.busiest and not once else settings.time,
                                                     settings.seed, playback(settings, context.scene))
            if holder is not None:
                jitter(holder)
        except LookupError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"'{path}' failed: {e}")
            return {'CANCELLED'}
        if holder is None:
            skipped = particles.unsupported_renderers(path)
            if once:
                self.report({'WARNING'}, f"'{path}' draws nothing in its first {settings.length:g} s")
            elif settings.busiest:
                why = f" (it only has {', '.join(skipped)}, which aren't drawn)" if skipped else ''
                self.report({'WARNING'}, f"'{path}' draws nothing at any moment{why}")
            else:
                self.report({'WARNING'}, f"'{path}' draws nothing {settings.time:g} s in (try Busiest Moment)")
            return {'CANCELLED'}
        holder['dlr_particles'] = True
        if settings.busiest and not once:
            self.report({'INFO'}, f"Shown {holder['dl_time']:g} s in, its busiest moment")
        for ob in context.selected_objects:
            ob.select_set(False)
        holder.select_set(True)
        context.view_layer.objects.active = holder
        return {'FINISHED'}


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
    settings = context.scene.dlr_particles
    box = layout.box()
    box.label(text='Particle Effects', icon='PARTICLES')
    col = box.column(align=True)
    col.prop(settings, 'effect', text='', icon='VIEWZOOM')
    col.row(align=True).prop(settings, 'playback', expand=True)
    row = col.row(align=True)
    if settings.playback == 'ONCE':
        row.prop(settings, 'length')
    else:
        row.prop(settings, 'busiest')
        row = col.row(align=True)
        sub = row.row(align=True)
        sub.active = not settings.busiest
        sub.prop(settings, 'time')
    row.prop(settings, 'seed')
    col.operator('dlr.add_particle', icon='PARTICLES')

    holder = effect_holder(context.active_object)
    if holder is None:
        return
    sub = box.box()
    path = holder['dl_effect']
    sub.label(text=path.rsplit('/', 1)[-1], icon='PARTICLE_DATA')
    if 'dl_start_active' in holder and not holder['dl_start_active']:
        sub.label(text="Started by the map's logic", icon='INFO')
    if holder.get('dl_animated'):
        if holder.get('dl_loop', True):
            sub.label(text=f"Loops every {holder['dl_steps'] / holder['dl_rate']:.1f} s", icon='FILE_REFRESH')
        else:
            start = holder['dl_start'] * context.scene.render.fps / context.scene.render.fps_base
            sub.label(text=f"Plays once from frame {start:.0f}", icon='PLAY')
    else:
        sub.label(text='Still', icon='PAUSE')
    if holder.library is not None:
        sub.label(text='In a linked map: append it to rebuild', icon='LINKED')
        return
    row = sub.row(align=True)
    row.prop(holder, '["dl_time"]', text='Time')
    row.prop(holder, '["dl_seed"]', text='Seed')
    sub.operator('dlr.rebuild_particle', icon='FILE_REFRESH')


CLASSES = (DLR_ParticleSettings, DLR_OT_add_particle, DLR_OT_rebuild_particle)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.dlr_particles = bpy.props.PointerProperty(type=DLR_ParticleSettings)


def unregister():
    del bpy.types.Scene.dlr_particles
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
