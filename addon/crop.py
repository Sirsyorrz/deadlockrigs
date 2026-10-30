# SPDX-License-Identifier: GPL-3.0-or-later
"""Cutting a map down to the part a shot needs: inside crop boxes, or what the camera sees over the scene's frame
range.

An appended map is cropped in place. Nothing is deleted until Delete Cropped: hidden objects keep their mesh, and a cut
object keeps its whole mesh (with a fake user) under the 'dlr_crop_mesh' property, so Restore Map can put everything
back.

A linked map can't be changed, so cropping it excludes its collection from the view layers and shows a local '(cropped)'
collection instead: the linked objects that stay whole, local copies of the cut ones with their trimmed meshes, and
local copies of the map's light-linked lights whose link collections also cover the cut copies (the rig light would
otherwise light them). Restore Map removes it and brings back the linked collection.
"""
import re

import bmesh
import bpy
import numpy as np

BOX = 'dlr_crop_box'
ORIGINAL = 'dlr_crop_mesh'
HIDDEN = 'dlr_crop_hidden'
CROP_OF = 'dlr_crop_of'

# map objects are named <cell>__<material>; these match the start of the material
KEEP_KINDS = (
    ('SKYLINE', 'Skyline Buildings', 'The backdrop buildings and their signs', r'backdrop_'),
    ('CLOUDS', 'Sky Clouds', 'The cloud funnels and rings in the sky', r'(cloud_funnel|funnelcloud|cloud_ring)'),
    ('SMOKE', 'Smoke', 'Chimney smoke and factory pollution clouds', r'(chimney_smoke|campus_smoke|factory_pollution)'),
    ('FOG', 'Fog Planes', 'Flat fog cards', r'fog_plane'),
)


def map_collections(scene, linked=False):
    """The scene's appended maps, or with linked=True its linked ones."""
    return [c for c in scene.collection.children_recursive if c.get('dlr_map') and (c.library is not None) == linked]


def any_map(scene):
    return bool(map_collections(scene) or map_collections(scene, linked=True))


def map_meshes(scene):
    seen = set()
    for c in map_collections(scene):
        for ob in c.all_objects:
            if ob.type == 'MESH' and ob.name not in seen:
                seen.add(ob.name)
                yield ob


def keep_pattern(kinds):
    parts = [pattern for ident, _, _, pattern in KEEP_KINDS if ident in kinds]
    return re.compile('|'.join(parts)) if parts else None


def is_kept(ob, pattern):
    return pattern is not None and pattern.match(ob.name.lower().split('__', 1)[-1]) is not None


def original_mesh(ob):
    return bpy.data.meshes.get(ob.get(ORIGINAL, '')) or ob.data


def world_vertices(ob, mesh):
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get('co', co)
    m = np.array(ob.matrix_world, np.float32)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def face_vertices(mesh):
    starts = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get('loop_start', starts)
    totals = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get('loop_total', totals)
    loops = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get('vertex_index', loops)
    return starts, totals, loops


def outcodes_box(points, box, margin):
    """Six bits per point, one per side of the box it is beyond. The box is an empty's cube: its display size in
    its own (scaled) axes, grown by margin metres."""
    m = np.array(box.matrix_world.inverted(), np.float32)
    local = (points @ m[:3, :3].T + m[:3, 3]) / box.empty_display_size
    limit = 1 + margin / (np.array(box.matrix_world.to_scale(), np.float32) * box.empty_display_size)
    code = np.zeros(len(points), np.uint8)
    for axis in range(3):
        code |= (local[:, axis] < -limit[axis]).astype(np.uint8) << (2 * axis)
        code |= (local[:, axis] > limit[axis]).astype(np.uint8) << (2 * axis + 1)
    return code


def outcodes_camera(points, clip_from_world, fov_margin):
    """Clip-space outcodes for one camera view, the frustum widened by fov_margin (a fraction of the frame)."""
    h = np.c_[points, np.ones(len(points), np.float32)] @ clip_from_world.T
    w = h[:, 3]
    code = np.zeros(len(points), np.uint8)
    for axis in range(2):
        code |= (h[:, axis] < -w * (1 + fov_margin)).astype(np.uint8) << (2 * axis)
        code |= (h[:, axis] > w * (1 + fov_margin)).astype(np.uint8) << (2 * axis + 1)
    code |= (h[:, 2] < -w).astype(np.uint8) << 4
    code |= (h[:, 2] > w).astype(np.uint8) << 5
    return code


def faces_inside(codes, starts, loops):
    """A face is outside only when all its corners are beyond the same side; anything else may show."""
    corner = codes[loops]
    face_and = np.bitwise_and.reduceat(corner, starts) if len(starts) else np.zeros(0, np.uint8)
    return face_and == 0


def camera_views(scene, step):
    cam = scene.camera
    if cam is None:
        return []
    frame = scene.frame_current
    views = []
    res = scene.render
    x, y = res.resolution_x * res.pixel_aspect_x, res.resolution_y * res.pixel_aspect_y
    try:
        for f in range(scene.frame_start, scene.frame_end + 1, max(1, step)):
            scene.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            ev = cam.evaluated_get(dg)
            proj = np.array(ev.calc_matrix_camera(dg, x=int(x), y=int(y)), np.float32)
            view = np.array(ev.matrix_world.inverted(), np.float32)
            views.append(proj @ view)
    finally:
        scene.frame_set(frame)
    return views


def trimmed(whole, keep):
    """A local copy of the mesh with only the kept faces."""
    bm = bmesh.new()
    bm.from_mesh(whole)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, keep) if not k], context='FACES')
    piece = whole.copy()
    piece.name = whole.name + '.crop'
    bm.to_mesh(piece)
    bm.free()
    return piece


def cut(ob, keep):
    """Gives the object a copy of its whole mesh with only the kept faces; the whole mesh stays for Restore."""
    whole = original_mesh(ob)
    if ORIGINAL not in ob:
        ob[ORIGINAL] = whole.name
        whole.use_fake_user = True
    old = ob.data
    ob.data = trimmed(whole, keep)
    if old is not whole and old.users == 0:
        bpy.data.meshes.remove(old)


def set_hidden(ob, hidden):
    ob.hide_viewport = ob.hide_render = hidden
    if hidden:
        ob[HIDDEN] = True
    elif HIDDEN in ob:
        del ob[HIDDEN]


def restore(ob):
    whole = bpy.data.meshes.get(ob.get(ORIGINAL, ''))
    if whole is not None:
        piece = ob.data
        ob.data = whole
        whole.use_fake_user = False
        if piece is not whole and piece.users == 0:
            bpy.data.meshes.remove(piece)
    if ORIGINAL in ob:
        del ob[ORIGINAL]
    if HIDDEN in ob:
        set_hidden(ob, False)


def faces_kept(ob, mesh, region_codes, whole_objects, keep_re):
    """Per face, whether the crop keeps it, and the faces' corner counts."""
    starts, totals, loops = face_vertices(mesh)
    if is_kept(ob, keep_re):
        return np.ones(len(starts), bool), totals
    points = world_vertices(ob, mesh)
    keep = np.zeros(len(starts), bool)
    for codes in region_codes(points):
        keep |= faces_inside(codes, starts, loops)
        if keep.all():
            break
    if whole_objects and keep.any():
        keep[:] = True
    return keep, totals


def layer_collections(scene, coll):
    def find(lc):
        if lc.collection == coll:
            return lc
        return next((hit for child in lc.children if (hit := find(child))), None)
    return [lc for vl in scene.view_layers if (lc := find(vl.layer_collection))]


def crop_view(coll):
    return next((c for c in bpy.data.collections if c.library is None and c.get(CROP_OF) == coll.name_full), None)


def restore_linked(scene, coll):
    view = crop_view(coll)
    if view is not None:
        local = [ob for ob in view.objects if ob.library is None]
        data = {ob.data for ob in local if ob.data and ob.data.library is None}
        links = {c for ob in local for c in (ob.light_linking.receiver_collection, ob.light_linking.blocker_collection)
                 if c and c.library is None}
        bpy.data.batch_remove(local + [view])
        bpy.data.batch_remove([d for d in data | links if d.users == 0])
    for lc in layer_collections(scene, coll):
        lc.exclude = False


def mirror_links(src, cuts):
    """A local copy of a light linking collection that also links each cut copy the way it links the original."""
    dst = bpy.data.collections.new(src.name + ' (cropped)')
    states = []
    for ob, item in zip(src.objects, src.collection_objects):
        for target in (ob, cuts.get(ob)):
            if target is not None:
                dst.objects.link(target)
                states.append(item.light_linking.link_state)
    for item, state in zip(dst.collection_objects, states):
        item.light_linking.link_state = state
    for child, item in zip(src.children, src.collection_children):
        dst.children.link(child)
        dst.collection_children[-1].light_linking.link_state = item.light_linking.link_state
    return dst


def crop_linked(scene, coll, region_codes, whole_objects, keep_re):
    restore_linked(scene, coll)
    view = bpy.data.collections.new(coll.name + ' (cropped)')
    view[CROP_OF] = coll.name_full
    scene.collection.children.link(view)
    kept = total = 0
    cuts = {}
    lights = []
    for ob in coll.all_objects:
        if ob.type == 'LIGHT' and (ob.light_linking.receiver_collection or ob.light_linking.blocker_collection):
            lights.append(ob)
            continue
        if ob.type != 'MESH':
            view.objects.link(ob)
            continue
        keep, totals = faces_kept(ob, ob.data, region_codes, whole_objects, keep_re)
        total += int((totals - 2).sum())
        if not keep.any():
            continue
        kept += int((totals[keep] - 2).sum())
        if keep.all():
            view.objects.link(ob)
            continue
        piece = ob.copy()
        piece.data = trimmed(ob.data, keep)
        view.objects.link(piece)
        cuts[ob] = piece
    for ob in lights:
        light = ob.copy()
        ll = ob.light_linking
        if ll.receiver_collection:
            light.light_linking.receiver_collection = mirror_links(ll.receiver_collection, cuts)
        if ll.blocker_collection:
            light.light_linking.blocker_collection = mirror_links(ll.blocker_collection, cuts)
        view.objects.link(light)
    for lc in layer_collections(scene, coll):
        lc.exclude = True
    return kept, total


def crop(scene, region_codes, whole_objects, keep):
    """region_codes(points) -> list of outcode arrays, one per region; a face stays if it is inside any.
    Returns (triangles kept, triangles before)."""
    kept = total = 0
    keep_re = keep_pattern(keep)
    for ob in list(map_meshes(scene)):
        restore(ob)
        keep, totals = faces_kept(ob, ob.data, region_codes, whole_objects, keep_re)
        total += int((totals - 2).sum())
        if not keep.any():
            set_hidden(ob, True)
            continue
        kept += int((totals[keep] - 2).sum())
        if not keep.all():
            cut(ob, keep)
    for coll in map_collections(scene, linked=True):
        k, t = crop_linked(scene, coll, region_codes, whole_objects, keep_re)
        kept += k
        total += t
    return kept, total


class DLR_PT_crop_keep(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = 'Never Crop'

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text='Never Crop')
        for ident, *_ in KEEP_KINDS:
            col.prop(context.scene, keep_prop(ident))


def keep_prop(ident):
    return f'dlr_crop_keep_{ident.lower()}'


def keep_kinds(scene):
    return {ident for ident, *_ in KEEP_KINDS if getattr(scene, keep_prop(ident))}


def keep_summary(scene):
    names = [name for ident, name, _, _ in KEEP_KINDS if getattr(scene, keep_prop(ident))]
    return ', '.join(names) if names else 'Nothing'


class DLR_OT_crop_add_box(bpy.types.Operator):
    bl_idname = 'dlr.crop_add_box'
    bl_label = 'Add Crop Box'
    bl_description = 'Add a box at the 3D cursor. Move and scale it over the part of the map to keep'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        box = bpy.data.objects.new('Crop Box', None)
        box.empty_display_type = 'CUBE'
        box.empty_display_size = 15.0
        box.show_in_front = True
        box.location = context.scene.cursor.location
        box[BOX] = True
        context.scene.collection.objects.link(box)
        for ob in context.selected_objects:
            ob.select_set(False)
        box.select_set(True)
        context.view_layer.objects.active = box
        return {'FINISHED'}


class CropOperator(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any_map(context.scene)

    def report_result(self, kept, total):
        self.report({'INFO'}, f'Kept {kept:,} of {total:,} map triangles')


class DLR_OT_crop_boxes(CropOperator):
    bl_idname = 'dlr.crop_boxes'
    bl_label = 'Crop to Boxes'
    bl_description = 'Hide the map outside the crop boxes; objects crossing a box edge keep the polygons that reach in'

    def execute(self, context):
        sc = context.scene
        boxes = [ob for ob in sc.objects if ob.get(BOX)]
        if not boxes:
            self.report({'ERROR'}, 'Add a crop box first')
            return {'CANCELLED'}
        def regions(points):
            return [outcodes_box(points, box, sc.dlr_crop_margin) for box in boxes]

        self.report_result(*crop(sc, regions, sc.dlr_crop_mode == 'OBJECTS', keep_kinds(sc)))
        return {'FINISHED'}


class DLR_OT_crop_camera(CropOperator):
    bl_idname = 'dlr.crop_camera'
    bl_label = 'Crop to Camera'
    bl_description = ("Hide the map the scene camera never sees over the frame range (sampled every few frames, with "
                      'a margin around the frame)')

    def execute(self, context):
        sc = context.scene
        views = camera_views(sc, sc.dlr_crop_step)
        if not views:
            self.report({'ERROR'}, 'The scene has no camera')
            return {'CANCELLED'}
        fov = sc.dlr_crop_fov_margin

        def regions(points):
            return [outcodes_camera(points, v, fov) for v in views]

        self.report_result(*crop(sc, regions, sc.dlr_crop_mode == 'OBJECTS', keep_kinds(sc)))
        return {'FINISHED'}


class DLR_OT_crop_restore(CropOperator):
    bl_idname = 'dlr.crop_restore'
    bl_label = 'Restore Map'
    bl_description = 'Undo the crop: show and restore every map object'

    def execute(self, context):
        for ob in list(map_meshes(context.scene)):
            restore(ob)
        for coll in map_collections(context.scene, linked=True):
            restore_linked(context.scene, coll)
        return {'FINISHED'}


class DLR_OT_crop_delete(CropOperator):
    bl_idname = 'dlr.crop_delete'
    bl_label = 'Delete Cropped'
    bl_description = ('Delete the hidden map objects and the cut-away polygons for good, to make the file smaller '
                      'and faster. Restore Map can no longer bring them back. Only for appended maps: a linked '
                      "map's cropped parts aren't in this file")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(map_collections(context.scene))

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        removed = 0
        unused = []
        for ob in list(map_meshes(context.scene)):
            whole = bpy.data.meshes.get(ob.get(ORIGINAL, ''))
            if whole is not None:
                whole.use_fake_user = False
                unused.append(whole)
                del ob[ORIGINAL]
            if ob.get(HIDDEN):
                unused.append(ob.data)
                bpy.data.objects.remove(ob)
                removed += 1
        for mesh in unused:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        self.report({'INFO'}, f'Deleted {removed} hidden objects and the cut polygons')
        return {'FINISHED'}


def draw(layout, context):
    sc = context.scene
    box = layout.box()
    box.label(text='Crop', icon='MOD_BOOLEAN')
    col = box.column(align=True)
    col.operator('dlr.crop_add_box', icon='CUBE')
    col.operator('dlr.crop_boxes', icon='SELECT_SET')
    col.operator('dlr.crop_camera', icon='CAMERA_DATA')
    col = box.column(align=True)
    col.prop(sc, 'dlr_crop_mode', text='')
    col.prop(sc, 'dlr_crop_margin')
    col.prop(sc, 'dlr_crop_fov_margin')
    col.prop(sc, 'dlr_crop_step')
    row = col.row(align=True)
    row.label(text='Keep:')
    row.popover('DLR_PT_crop_keep', text=keep_summary(sc))
    row = box.row(align=True)
    row.operator('dlr.crop_restore', icon='LOOP_BACK')
    row.operator('dlr.crop_delete', icon='TRASH')


CLASSES = (DLR_PT_crop_keep, DLR_OT_crop_add_box, DLR_OT_crop_boxes, DLR_OT_crop_camera, DLR_OT_crop_restore, DLR_OT_crop_delete)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    S = bpy.types.Scene
    S.dlr_crop_mode = bpy.props.EnumProperty(
        name='Edges', default='POLYS',
        items=[('POLYS', 'Whole Polygons', 'Objects crossing the edge keep the polygons that reach in'),
               ('OBJECTS', 'Whole Objects', 'Objects crossing the edge stay whole (faster, keeps more)')])
    S.dlr_crop_margin = bpy.props.FloatProperty(
        name='Box Margin', default=2.0, min=0.0, soft_max=50.0, subtype='DISTANCE', unit='LENGTH',
        description='Keep this much more around each crop box')
    S.dlr_crop_fov_margin = bpy.props.FloatProperty(
        name='Frame Margin', default=0.15, min=0.0, soft_max=1.0, subtype='FACTOR',
        description='Keep this much more around the camera frame, as a fraction of it (for reflections, motion blur)')
    S.dlr_crop_step = bpy.props.IntProperty(
        name='Frame Step', default=5, min=1, soft_max=50,
        description='Sample the camera every this many frames')
    for ident, name, desc, _ in KEEP_KINDS:
        setattr(S, keep_prop(ident), bpy.props.BoolProperty(name=name, description=desc))


def unregister():
    keep = [keep_prop(ident) for ident, *_ in KEEP_KINDS]
    for name in ('dlr_crop_mode', 'dlr_crop_margin', 'dlr_crop_fov_margin', 'dlr_crop_step', *keep):
        delattr(bpy.types.Scene, name)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
