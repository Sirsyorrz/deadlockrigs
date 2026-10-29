"""Imports a rebatched map into Blender and saves it with its textures next to it (run by mapbuild.py).

blender -b --factory-startup -P tools/map_import.py -- <map.gltf> <out.blend> <map slug> <title> <game version>
"""
import os
import shutil
import sys

import bpy

gltf, out_blend, slug, title, version = sys.argv[sys.argv.index('--') + 1:][:5]
out_dir = os.path.dirname(os.path.abspath(out_blend))
tex_dir = os.path.join(out_dir, 'textures')
# the game's sun has no brightness in the export; this is a usable daylight starting point
SUN_STRENGTH = 4.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=gltf, import_pack_images=False)

coll = bpy.data.collections.new(title)
bpy.context.scene.collection.children.link(coll)
for ob in list(bpy.context.scene.collection.objects):
    coll.objects.link(ob)
    bpy.context.scene.collection.objects.unlink(ob)
coll['dlr_map'] = slug
coll['dlr_map_version'] = version

for ob in coll.objects:
    if ob.type == 'LIGHT' and ob.data.type == 'SUN':
        ob.name = ob.data.name = f'{title} sun'
        ob.data.energy = SUN_STRENGTH

os.makedirs(tex_dir, exist_ok=True)
for img in bpy.data.images:
    src = bpy.path.abspath(img.filepath) if img.filepath else ''
    if img.packed_file or not os.path.isfile(src):
        continue
    dest = os.path.join(tex_dir, os.path.basename(src))
    if not os.path.exists(dest):
        shutil.copyfile(src, dest)
    img.filepath = dest

bpy.ops.wm.save_as_mainfile(filepath=out_blend, relative_remap=True)
meshes = sum(1 for ob in coll.all_objects if ob.type == 'MESH')
print(f'wrote {out_blend}: {meshes} meshes, {len(bpy.data.images)} textures')
