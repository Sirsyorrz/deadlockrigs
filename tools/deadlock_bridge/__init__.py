"""Deadlock Bridge: load Deadlock maps into Blender straight from the game's VPKs, shaded as the game shades them.

Package layout:
    core/      file formats: VPK, compiled resources, KV3, meshes, models, textures, materials, worlds, probes (no bpy)
    native/    LZ4 and meshoptimizer decoders in C (Python fallbacks in core/)
    blender/   everything that creates Blender data: textures, node groups, materials, meshes, lights, the map importer
"""
try:
    import bpy
except ImportError:  # the parsers are also used by tests outside Blender
    bpy = None

if bpy is not None:
    from . import operators, preferences, ui

    _classes = (preferences.DEADLOCK_AP_preferences, *operators.classes, *ui.classes)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    from . import particle_tools
    particle_tools.register()
    bpy.types.TOPBAR_MT_file_import.append(operators.menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(operators.menu_import)
    from . import particle_tools
    particle_tools.unregister()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    from . import session
    session.reset()
