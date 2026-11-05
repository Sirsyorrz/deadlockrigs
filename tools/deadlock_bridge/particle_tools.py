"""Particle effects on demand: any of the game's particle systems placed at the 3D cursor (from the Deadlock sidebar
or Add > Deadlock Particle Effect), and effects, placed or imported with a map, built again at another moment of
their life or with another seed.

The add-on's register() calls ``particle_tools.register()`` after its own classes (the panel sits under the Deadlock
sidebar panel).
"""
import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import session

_paths: list[str] = []
_SEARCH_LIMIT = 400


def _effect_paths() -> list[str]:
    global _paths
    if not _paths:
        from .blender.particles import effect_paths
        _paths = effect_paths(session.content())
    return _paths


def _search(_self, _context, text: str):
    """The effects whose path holds every word typed (folder or name parts, in any order)."""
    try:
        paths = _effect_paths()
    except Exception:
        return []
    words = text.lower().replace("\\", "/").split()
    found = []
    for path in paths:
        if all(word in path for word in words):
            found.append(path)
            if len(found) == _SEARCH_LIMIT:
                break
    return found


_EFFECT_DESCRIPTION = "A particle system from the game's files; type words of its path to search (\"candle flame\")"
_TIME_DESCRIPTION = "How long the effect has been playing when it is frozen, in seconds"
_BUSIEST_DESCRIPTION = ("Show the effect at the moment it draws the most (many are bursts over in a fraction of a "
                        "second); off, at the time set")
_SEED_DESCRIPTION = "Which of the effect's random variations to show"


class DeadlockParticleSettings(bpy.types.PropertyGroup):
    effect: StringProperty(name="Effect", description=_EFFECT_DESCRIPTION, search=_search)
    busiest: BoolProperty(name="Busiest moment", default=True, description=_BUSIEST_DESCRIPTION)
    time: FloatProperty(name="Time", default=2.0, min=0.0, soft_max=20.0, step=10, precision=2,
                        description=_TIME_DESCRIPTION)
    seed: IntProperty(name="Seed", default=0, min=0, description=_SEED_DESCRIPTION)


def _effect_holder(obj):
    """The empty holding an effect's objects, from it or one of its objects."""
    while obj is not None:
        if obj.type == "EMPTY" and obj.get("dl_effect") and "dl_time" in obj:
            return obj
        obj = obj.parent
    return None


class DEADLOCK_OT_add_particle_effect(bpy.types.Operator):
    bl_idname = "deadlock.add_particle_effect"
    bl_label = "Add Particle Effect"
    bl_description = "Place one of Deadlock's particle effects at the 3D cursor, as it looks a moment into its life"
    bl_options = {"REGISTER", "UNDO"}

    # Not remembered between calls (SKIP_SAVE): each call takes the sidebar's choice unless given its own.
    effect: StringProperty(name="Effect", description=_EFFECT_DESCRIPTION, search=_search, options={"SKIP_SAVE"})
    busiest: BoolProperty(name="Busiest moment", default=True, description=_BUSIEST_DESCRIPTION, options={"SKIP_SAVE"})
    time: FloatProperty(name="Time", default=2.0, min=0.0, soft_max=20.0, step=10, precision=2,
                        description=_TIME_DESCRIPTION, options={"SKIP_SAVE"})
    seed: IntProperty(name="Seed", default=0, min=0, description=_SEED_DESCRIPTION, options={"SKIP_SAVE"})
    choose: BoolProperty(name="Choose", default=False, options={"HIDDEN", "SKIP_SAVE"},
                         description="Ask which effect first")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and session.install_path() is not None

    def invoke(self, context, event):
        settings = context.scene.deadlock_particles
        for name in ("effect", "busiest", "time", "seed"):
            if not self.properties.is_property_set(name):
                setattr(self, name, getattr(settings, name))
        if self.choose or not self.effect:
            return context.window_manager.invoke_props_dialog(self, width=460)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "effect", text="", icon="VIEWZOOM")
        layout.prop(self, "busiest")
        row = layout.row(align=True)
        row.prop(self, "time")
        row.active = not self.busiest
        row.prop(self, "seed")

    def execute(self, context):
        from .blender.particles import spawn_particle_system
        if not self.properties.is_property_set("effect") and not self.effect:
            self.effect = context.scene.deadlock_particles.effect
        path = self.effect.strip().replace("\\", "/").lower()
        if not path:
            self.report({"ERROR"}, "Choose an effect first")
            return {"CANCELLED"}
        if path.endswith("_c"):
            path = path[:-2]
        try:
            holder = spawn_particle_system(context, path, context.scene.cursor.matrix,
                                           None if self.busiest else self.time, self.seed)
        except LookupError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"'{path}' failed: {error}")
            return {"CANCELLED"}
        if holder is None:
            from .blender.particles import unsupported_renderers
            skipped = unsupported_renderers(path)
            if self.busiest:
                why = f" (it only has {', '.join(skipped)}, which the add-on doesn't draw)" if skipped else ""
                self.report({"WARNING"}, f"'{path}' draws nothing at any moment{why}")
            else:
                self.report({"WARNING"}, f"'{path}' draws nothing {self.time:g} s in (try Busiest moment)")
            return {"CANCELLED"}
        settings = context.scene.deadlock_particles
        settings.effect, settings.busiest, settings.seed = path, self.busiest, self.seed
        if not self.busiest:
            settings.time = self.time
        else:
            self.report({"INFO"}, f"Shown {holder['dl_time']:g} s in, its busiest moment")
        for obj in context.selected_objects:
            obj.select_set(False)
        holder.select_set(True)
        context.view_layer.objects.active = holder
        return {"FINISHED"}


class DEADLOCK_OT_rebuild_particle_effect(bpy.types.Operator):
    bl_idname = "deadlock.rebuild_particle_effect"
    bl_label = "Rebuild Particle Effect"
    bl_description = ("Simulate the selected particle effects again where their empties are now, at their time and "
                      "seed (control points other than the first move with the empty)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return session.install_path() is not None and any(_effect_holder(o) for o in context.selected_objects)

    def execute(self, context):
        from .blender.particles import rebuild_particle_system
        holders = {h.name: h for h in (_effect_holder(o) for o in context.selected_objects) if h is not None}
        empty = 0
        for holder in holders.values():
            try:
                if not rebuild_particle_system(context, holder):
                    empty += 1
            except Exception as error:
                self.report({"ERROR"}, f"'{holder.name}' failed: {error}")
                return {"CANCELLED"}
        if empty:
            self.report({"WARNING"}, f"{empty} of {len(holders)} effects draw nothing at their time")
        return {"FINISHED"}


class DEADLOCK_PT_particles(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Deadlock"
    bl_label = "Particle Effects"
    bl_parent_id = "DEADLOCK_PT_sidebar"

    @classmethod
    def poll(cls, context):
        return session.install_path() is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.deadlock_particles
        col = layout.column(align=True)
        col.prop(settings, "effect", text="", icon="VIEWZOOM")
        col.prop(settings, "busiest")
        row = col.row(align=True)
        sub = row.row(align=True)
        sub.active = not settings.busiest
        sub.prop(settings, "time")
        row.prop(settings, "seed")
        col.operator(DEADLOCK_OT_add_particle_effect.bl_idname, text="Add at 3D Cursor", icon="PARTICLES")

        holder = _effect_holder(context.active_object)
        if holder is None:
            return
        box = layout.box()
        path = holder["dl_effect"]
        box.label(text=path.rsplit("/", 1)[-1], icon="PARTICLE_DATA")
        box.label(text=path.rsplit("/", 1)[0] + "/")
        parts = box.column(align=True)
        for child in holder.children:
            count = len(child.data.vertices) if child.type == "MESH" else 0
            kind = "cards" if any(m.type == "NODES" and m.node_group and "Sprites" in m.node_group.name
                                  for m in child.modifiers) else "vertices"
            parts.label(text=f"{child.name}  ({count} {kind})")
        if "dl_start_active" in holder and not holder["dl_start_active"]:
            box.label(text="Started by the map's logic", icon="INFO")
        row = box.row(align=True)
        row.prop(holder, '["dl_time"]', text="Time")
        row.prop(holder, '["dl_seed"]', text="Seed")
        box.operator(DEADLOCK_OT_rebuild_particle_effect.bl_idname, text="Rebuild", icon="FILE_REFRESH")


def _menu_add(self, _context):
    layout = self.layout
    layout.operator_context = "INVOKE_DEFAULT"
    op = layout.operator(DEADLOCK_OT_add_particle_effect.bl_idname, text="Deadlock Particle Effect", icon="PARTICLES")
    op.choose = True


classes = (DeadlockParticleSettings, DEADLOCK_OT_add_particle_effect, DEADLOCK_OT_rebuild_particle_effect,
           DEADLOCK_PT_particles)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.deadlock_particles = PointerProperty(type=DeadlockParticleSettings)
    bpy.types.VIEW3D_MT_add.append(_menu_add)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu_add)
    del bpy.types.Scene.deadlock_particles
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
