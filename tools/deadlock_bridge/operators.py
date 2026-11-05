import bpy
from bpy.props import BoolProperty, EnumProperty

from . import session
from .preferences import cache_dir, get_prefs

# Friendly names for the maps the game ships; others show their file name.
MAP_NAMES = {
    "dl_midtown": "Midtown (the main map)",
    "dl_hideout": "The Hideout",
    "new_player_basics": "New Player Basics",
    "hero_testing": "Hero Testing",
    "start": "Start",
}

_map_items_cache = []


def _map_items(_self, _context):
    global _map_items_cache
    try:
        maps = sorted(session.content().maps)
    except Exception as error:
        _map_items_cache = [("NONE", "Deadlock not found", str(error))]
        return _map_items_cache
    official = [m for m in maps if m in MAP_NAMES]
    others = [m for m in maps if m not in MAP_NAMES]
    items = [(m, MAP_NAMES[m], m) for m in official] + [(m, m, "Community map") for m in others]
    _map_items_cache = items or [("NONE", "No maps", "")]
    return _map_items_cache


class DEADLOCK_OT_import_map(bpy.types.Operator):
    bl_idname = "deadlock.import_map"
    bl_label = "Import Deadlock Map"
    bl_description = "Load a Deadlock map from the game's files, with its materials, baked lighting, lights, sky and fog"
    # No undo step: Blender would copy the whole imported map into undo memory (gigabytes on dl_midtown).
    bl_options = {"REGISTER"}

    map_name: EnumProperty(name="Map", items=_map_items)
    props: BoolProperty(name="Props", default=True, description="Entity props (prop_dynamic and friends)")
    lights: BoolProperty(name="Realtime lights", default=True,
                         description="The sun and the lights the game lights in realtime (the rest is in the lightmaps)")
    markers: BoolProperty(name="Particle markers", default=True, description="Empties where the map plays particle systems")
    volumetric_fog: BoolProperty(name="Volumetric fog", default=True,
                                 description="The map's volumetric fog volumes, lit by its baked fog irradiance")
    particles: BoolProperty(name="Particles", default=True,
                            description="The map's particle effects, simulated a few seconds in and frozen")
    detail_levels: BoolProperty(
        name="Game detail levels", default=False,
        description=("Import every level of detail and switch between them by distance from the camera as the game "
                     "does (trees become a round canopy without a trunk a few metres out). Off: everything at full "
                     "detail"))
    reflection_probes: BoolProperty(name="EEVEE reflection probes", default=True,
                                    description="Sphere probes with parallax boxes where the game has its cubemaps")
    setup_render: BoolProperty(name="Set up render", default=True,
                               description="Standard view, no diffuse bounces (the baked lighting is the bounce light), EEVEE ray traced reflections")
    post_processing: BoolProperty(name="Post processing", default=True,
                                  description="The map's exposure, bloom and filmic tonemapping in the compositor")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "map_name")
        col = layout.column(heading="Include")
        col.prop(self, "props")
        col.prop(self, "lights")
        col.prop(self, "particles")
        col.prop(self, "markers")
        col.prop(self, "volumetric_fog")
        col.prop(self, "reflection_probes")
        col.prop(self, "detail_levels")
        col = layout.column(heading="Scene")
        col.prop(self, "setup_render")
        col.prop(self, "post_processing")

    def execute(self, context):
        if self.map_name == "NONE":
            self.report({"ERROR"}, "No Deadlock maps found")
            return {"CANCELLED"}
        from .blender.map_import import MapImporter, Options
        prefs = get_prefs()
        options = Options(
            unit_scale=prefs.unit_scale if prefs else 0.0254,
            max_texture_size=int(prefs.texture_limit) if prefs else 0,
            compact_textures=prefs.compact_textures if prefs else True,
            props=self.props, lights=self.lights, markers=self.markers,
            reflection_probes=self.reflection_probes, setup_render=self.setup_render,
            volumetric_fog=self.volumetric_fog, particles=self.particles,
            post_processing=self.post_processing, detail_levels=self.detail_levels)
        wm = context.window_manager
        wm.progress_begin(0, 100)

        def progress(fraction, text):
            wm.progress_update(int(fraction * 100))
            print(f"[Deadlock] {text}")

        try:
            importer = MapImporter(context, session.content(), self.map_name, cache_dir(), options, progress)
            stats = importer.run()
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"Import failed: {error}")
            return {"CANCELLED"}
        finally:
            wm.progress_end()

        _write_report(self.map_name, stats, importer.log)
        self.report({"INFO"}, f"{self.map_name}: {stats.get('world_vertices', 0):,} world vertices, "
                              f"{stats.get('scene_objects', 0) + stats.get('entity_props', 0)} props, "
                              f"{stats.get('lights', 0)} lights in {stats.get('seconds')} s"
                              + (f" ({len(importer.log)} warnings, see the Deadlock Report text)" if importer.log else ""))
        return {"FINISHED"}


def _write_report(map_name, stats, log):
    text = bpy.data.texts.get("Deadlock Report") or bpy.data.texts.new("Deadlock Report")
    text.clear()
    text.write(f"{map_name}\n\n")
    for key, value in stats.items():
        text.write(f"{key}: {value}\n")
    if log:
        text.write("\nWarnings\n")
        for line in log:
            text.write(f"- {line}\n")


def menu_import(self, _context):
    self.layout.operator(DEADLOCK_OT_import_map.bl_idname, text="Deadlock Map")


classes = (DEADLOCK_OT_import_map,)
