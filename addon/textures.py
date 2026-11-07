# SPDX-License-Identifier: GPL-3.0-or-later
"""Map Textures: points a map's images at the smaller copies the build writes (tools/map_textures.py), so rendered
view and F12 load a quarter or a sixteenth of the texture memory. Lightmaps keep their full size.

A linked map's images can be repointed but the change isn't saved with the file, so it is redone on load."""
import os
import re

import bpy
from bpy.app.handlers import persistent

FOLDER = re.compile(r'([/\\])textures(?:_\d+)?([/\\])([^/\\]+)$')


def map_images():
    """Images in a map's textures folder: from a linked map's library, or appended from one."""
    for img in bpy.data.images:
        if img.source != 'FILE' or not FOLDER.search(img.filepath):
            continue
        lib = img.library.filepath if img.library else img.filepath
        if re.search(r'[/\\]maps[/\\]dl_\w+[/\\]', bpy.path.abspath(lib, library=img.library)):
            yield img


def apply(scene):
    size = scene.dlr_map_textures
    folder = 'textures' if size == 'FULL' else f'textures_{size}'
    changed = 0
    for img in map_images():
        path = FOLDER.sub(lambda m: f'{m[1]}{folder}{m[2]}{m[3]}', img.filepath)
        if size != 'FULL' and not os.path.exists(bpy.path.abspath(path, library=img.library)):
            # not larger than the size, or a map built before the smaller copies
            path = FOLDER.sub(lambda m: f'{m[1]}textures{m[2]}{m[3]}', img.filepath)
        if path != img.filepath:
            img.filepath = path
            if img.has_data:
                img.reload()
            changed += 1
    return changed


def update(self, context):
    apply(self)


@persistent
def on_load(_dummy):
    scene = bpy.context.scene
    if scene is not None and scene.dlr_map_textures != 'FULL':
        apply(scene)


def draw(layout, context):
    layout.prop(context.scene, 'dlr_map_textures')


def register():
    bpy.types.Scene.dlr_map_textures = bpy.props.EnumProperty(
        name='Map Textures', default='FULL', update=update,
        items=[('FULL', 'Full Size', 'The textures as the game has them'),
               ('1024', '1024', 'At most 1024 pixels: about a quarter of the memory. Fine for most shots'),
               ('512', '512', 'At most 512 pixels: for wide shots or when renders run out of memory')],
        description=('Texture size of the map in this scene. Smaller textures make rendered view and F12 use much '
                     'less RAM and VRAM; the lighting stays full size'))
    bpy.app.handlers.load_post.append(on_load)


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    del bpy.types.Scene.dlr_map_textures
