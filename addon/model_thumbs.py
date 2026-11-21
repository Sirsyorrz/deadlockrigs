"""Thumbnail renderer for the game model browser, run as a background Blender:
blender -b --factory-startup -P model_thumbs.py

Reads JSON jobs from stdin, one per line: {"items": [[glb, png], ...], "size": 160, "textured": true}. Renders
each glb to its png (or writes png + '.none' when the model has nothing to draw) and prints DONE once the job is
finished."""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

VIEW = Vector((1.0, -1.4, 0.75)).normalized()
HALF_FOV = math.radians(15)


def setup(size):
    sc = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    shading = sc.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'TEXTURE'
    shading.show_cavity = True
    shading.cavity_type = 'WORLD'
    sc.display.render_aa = '8'
    cam = bpy.data.objects.new('thumb_cam', bpy.data.cameras.new('thumb_cam'))
    cam.data.angle = HALF_FOV * 2
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def frame(cam, meshes):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for obj in meshes:
        ev = obj.evaluated_get(depsgraph)
        mw = ev.matrix_world
        # evaluated corners: skinned meshes sit where their armature puts them, not at the raw mesh bounds
        for corner in ev.bound_box:
            p = mw @ Vector(corner)
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)
    cam.location = center + VIEW * radius / math.sin(HALF_FOV)
    cam.rotation_euler = (-VIEW).to_track_quat('-Z', 'Y').to_euler()
    cam.data.clip_start = radius * 0.01
    cam.data.clip_end = radius * 10


def show_base_color(mat):
    """Workbench's texture colour is the material's active image node: make that the base colour image."""
    if mat is None or not mat.node_tree or mat.get('thumb_ready'):
        return
    mat['thumb_ready'] = True
    tree = mat.node_tree
    for node in tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            socket = node.inputs['Base Color']
            todo = [link.from_node for link in socket.links]
            while todo:
                src = todo.pop()
                if src.type == 'TEX_IMAGE':
                    tree.nodes.active = src
                    return
                todo += [link.from_node for inp in src.inputs for link in inp.links]
    images = [node for node in tree.nodes if node.type == 'TEX_IMAGE' and node.image
              and ('color' in node.image.name.lower() or 'albedo' in node.image.name.lower())]
    if images:
        tree.nodes.active = images[0]


def clear(new):
    for obj in new:
        bpy.data.objects.remove(obj)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures, bpy.data.actions,
                 bpy.data.cameras, bpy.data.node_groups):
        for block in list(coll):
            if not block.users:
                coll.remove(block)


def render(cam, glb, png, textured):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=glb, import_shading='NORMALS')
    except Exception as error:  # noqa: BLE001 (a broken export only costs its own thumbnail)
        print('thumb failed', glb, error, file=sys.stderr)
        new = []
    else:
        new = [obj for obj in bpy.data.objects if obj not in before]
    # Outline shells are inverted hulls drawn black around the model; they hide it in a plain render. Meshes
    # without a material in a textured export are helpers (S2V adds a 2 m icosphere to some), not something the
    # game draws; geometry-only exports have no materials at all.
    meshes = [obj for obj in new if obj.type == 'MESH' and not obj.name.endswith('_outline')
              and (not textured or any(slot.material for slot in obj.material_slots))]
    for obj in meshes:
        for slot in obj.material_slots:
            show_base_color(slot.material)
    for obj in new:
        if obj.type == 'MESH' and obj not in meshes:
            obj.hide_render = True
    os.makedirs(os.path.dirname(png), exist_ok=True)
    if meshes:
        frame(cam, meshes)
        bpy.context.scene.render.filepath = png
        bpy.ops.render.render(write_still=True)
    else:
        open(png + '.none', 'w').close()
    clear(new)


def main():
    cam = None
    size = None
    for line in sys.stdin:
        job = json.loads(line)
        if job.get('size') != size:
            size = job.get('size', 160)
            cam = setup(size)
        for glb, png in job['items']:
            render(cam, glb, png, job.get('textured', True))
        print('DONE', flush=True)


main()
