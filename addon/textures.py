# SPDX-License-Identifier: GPL-3.0-or-later
"""Map Textures: points a map's images at the smaller copies the build writes (tools/map_textures.py). The viewport
uses one size (512 by default) and F12/animation renders another (full by default): the render handlers swap the
images for the length of the render. Lightmaps keep their full size.

A linked map's images can be repointed but the change isn't saved with the file, so it is redone on load."""
import os
import re

import bpy
from bpy.app.handlers import persistent

FOLDER = re.compile(r'([/\\])textures(?:_\d+)?([/\\])([^/\\]+)$')

SIZES = [('FULL', 'Full Size', 'The textures as the game has them'),
         ('1024', '1024', 'At most 1024 pixels: about a quarter of the memory. Fine for most shots'),
         ('512', '512', 'At most 512 pixels: about a sixteenth of the memory')]

_rendering = None  # the scene a render swapped to its render size, until the render ends


def map_images():
    """Images in a map's textures folder: from a linked map's library, or appended from one."""
    for img in bpy.data.images:
        if img.source != 'FILE' or not FOLDER.search(img.filepath):
            continue
        lib = img.library.filepath if img.library else img.filepath
        if re.search(r'[/\\]maps[/\\]dl_\w+[/\\]', bpy.path.abspath(lib, library=img.library)):
            yield img


def repoint(size):
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


def apply(scene):
    """The viewport size, and Lock Interface when renders swap sizes: changing images from the render thread while
    the viewport draws them can crash Blender (bpy.app.handlers docs)."""
    if scene.dlr_render_textures != scene.dlr_map_textures and any(map_images()):
        scene.render.use_lock_interface = True
    return repoint(scene.dlr_render_textures if _rendering == scene.name else scene.dlr_map_textures)


def update(self, context):
    apply(self)


@persistent
def on_load(_dummy):
    global _rendering
    _rendering = None
    scene = bpy.context.scene
    if scene is not None:
        apply(scene)


@persistent
def on_undo(scene, *_args):
    # undo brings back the paths of the step it returns to, which may predate a repoint
    if _rendering is None and scene is not None:
        apply(scene)


@persistent
def on_render_init(scene, *_args):
    global _rendering
    if scene.dlr_render_textures == scene.dlr_map_textures:
        return
    if not (bpy.app.background or scene.render.use_lock_interface):
        return
    _rendering = scene.name
    repoint(scene.dlr_render_textures)


@persistent
def on_render_end(*_args):
    if _rendering is not None and bpy.app.background:
        _restore()


def _restore():
    global _rendering
    name, _rendering = _rendering, None
    scene = bpy.data.scenes.get(name or '')
    if scene is not None:
        repoint(scene.dlr_map_textures)


def _watch():
    """Back to the viewport size once the render job is gone. A timer registered from render_complete (the render
    thread) fired many seconds late, so this one lives on the main thread for the whole session."""
    if _rendering is not None and not bpy.app.is_job_running('RENDER'):
        _restore()
    return 0.5


def draw(layout, context):
    col = layout.column(align=True)
    col.prop(context.scene, 'dlr_map_textures')
    col.prop(context.scene, 'dlr_render_textures')


HANDLERS = [('load_post', on_load), ('undo_post', on_undo), ('redo_post', on_undo),
            ('render_init', on_render_init), ('render_complete', on_render_end), ('render_cancel', on_render_end)]


def register():
    bpy.types.Scene.dlr_map_textures = bpy.props.EnumProperty(
        name='Viewport Textures', default='512', update=update, items=SIZES,
        description=('Texture size of the map in the viewport. Smaller textures make Material Preview and Rendered '
                     'view use much less RAM and VRAM; the lighting stays full size'))
    bpy.types.Scene.dlr_render_textures = bpy.props.EnumProperty(
        name='Render Textures', default='FULL', update=update, items=SIZES,
        description=('Texture size of the map in F12 and animation renders. When it differs from the viewport, '
                     'the textures are swapped for the render and Lock Interface is turned on'))
    for name, handler in HANDLERS:
        getattr(bpy.app.handlers, name).append(handler)
    if not bpy.app.background and not bpy.app.timers.is_registered(_watch):
        bpy.app.timers.register(_watch, first_interval=1, persistent=True)


def unregister():
    for name, handler in HANDLERS:
        handlers = getattr(bpy.app.handlers, name)
        if handler in handlers:
            handlers.remove(handler)
    if bpy.app.timers.is_registered(_watch):
        bpy.app.timers.unregister(_watch)
    del bpy.types.Scene.dlr_map_textures, bpy.types.Scene.dlr_render_textures
