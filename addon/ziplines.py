# SPDX-License-Identifier: GPL-3.0-or-later
"""The lanes' ziplines on a map (tools/map_ziplines.py), from <map>/ziplines.json, which the map build writes; a map
built before that gets it in a few seconds, with npcs.json."""
import json
import os

import bpy

ROOT = 'dlr_ziplines'


def ziplines_path(slug, context=None):
    from . import output_dir
    return os.path.join(output_dir(context), 'maps', slug, 'ziplines.json')


def zipline_root(scene):
    return next((c for c in scene.collection.children_recursive if c.get(ROOT)), None)


def remove(scene):
    root = zipline_root(scene)
    if root is None:
        return
    meshes = {o.data for o in root.all_objects}
    bpy.data.batch_remove(list(root.all_objects))
    bpy.data.collections.remove(root)
    bpy.data.batch_remove([m for m in meshes if m.users == 0])


def place(scene, slug, context=None):
    from . import tools_module
    remove(scene)
    paths = json.load(open(ziplines_path(slug, context), encoding='utf-8'))['ziplines']
    title = tools_module('mapbuild').MAPS.get(slug, slug)
    coll = tools_module('map_ziplines').add_ziplines(scene.collection, f'{title} ziplines', paths)
    coll[ROOT] = slug


def toggle(self, context):
    from .npcs import scene_map
    scene = context.scene
    if not scene.dlr_ziplines:
        remove(scene)
        return
    slug = scene_map(scene)
    if slug:
        place_when_ready(slug)


def place_when_ready(slug):
    """Places them, finding them first for a map built without ziplines.json (waiting for a running job)."""
    def ready(ok):
        sc = bpy.context.scene
        sc.dlr_ziplines_status = '' if ok else 'failed'
        if ok and sc.dlr_ziplines and os.path.exists(ziplines_path(slug)):
            place(sc, slug)

    def attempt():
        from . import _job, queue_position, start_queue
        if not bpy.context.scene.dlr_ziplines:
            return None
        if os.path.exists(ziplines_path(slug)):
            ready(True)
            return None
        if queue_position() is not None or (_job is not None and not _job.done):
            return 1.0
        start_queue(bpy.context, [(slug, 'NPCS')], ready)
        return None

    bpy.context.scene.dlr_ziplines_status = ''
    if attempt() is not None:
        bpy.app.timers.register(attempt, first_interval=1.0)


def draw(box, context):
    sc = context.scene
    row = box.row()
    row.prop(sc, 'dlr_ziplines', text='Ziplines', icon='IPO_EASE_IN_OUT')
    if sc.dlr_ziplines_status == 'failed':
        box.label(text='Finding the ziplines failed', icon='ERROR')


def register():
    S, P = bpy.types.Scene, bpy.props
    S.dlr_ziplines = P.BoolProperty(
        name='Ziplines', default=False, update=toggle,
        description="The lanes' ziplines in their lane colours, moving with the scene's time as in the game")
    S.dlr_ziplines_status = P.StringProperty(options={'SKIP_SAVE'})


def unregister():
    for p in ('dlr_ziplines', 'dlr_ziplines_status'):
        delattr(bpy.types.Scene, p)
