import os
import tempfile

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty


def get_prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def default_cache_dir() -> str:
    try:
        return bpy.utils.extension_path_user(__package__, path="cache", create=True)
    except (ValueError, AttributeError):
        path = os.path.join(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "deadlock_bridge", "cache")
        os.makedirs(path, exist_ok=True)
        return path


def cache_dir() -> str:
    prefs = get_prefs()
    path = bpy.path.abspath(prefs.cache_dir) if prefs and prefs.cache_dir else default_cache_dir()
    os.makedirs(path, exist_ok=True)
    return path


class DEADLOCK_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    game_path: StringProperty(
        name="Deadlock folder", subtype="DIR_PATH",
        description="The Deadlock install folder, holding game/citadel (leave empty to find it through Steam)")
    cache_dir: StringProperty(
        name="Cache folder", subtype="DIR_PATH",
        description="Where converted textures are kept (empty = the add-on's user folder)")
    unit_scale: FloatProperty(
        name="Metres per unit", default=0.0254, min=0.001, max=1.0, precision=5,
        description="Size of one game unit in metres (0.0254: one inch)")
    texture_limit: EnumProperty(
        name="Texture size limit", default="0",
        items=[("0", "Full size", "Every texture at the size the game ships it"),
               ("2048", "2048", "Textures larger than 2048 load a mip down"),
               ("1024", "1024", "Textures larger than 1024 load mips down (what the s&box mount does on dl_midtown)"),
               ("512", "512", "Smallest; quick previews")],
        description="Large material textures load smaller mips (lightmaps always load in full)")

    compact_textures: BoolProperty(
        name="Compact GPU textures", default=True,
        description=("Store BC7/BC4/BC5 textures as DXT1/DXT5 with mips, which Blender keeps compressed on the GPU. "
                     "Off, they're uploaded uncompressed: a map like dl_hideout then needs about 8 GB of video memory"))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "game_path")
        layout.prop(self, "cache_dir")
        row = layout.row()
        row.prop(self, "unit_scale")
        row.prop(self, "texture_limit")
        layout.prop(self, "compact_textures")
        from . import session
        install = session.install_path()
        layout.label(text=f"Deadlock: {install}" if install else "Deadlock was not found; set its folder above",
                     icon="CHECKMARK" if install else "ERROR")
