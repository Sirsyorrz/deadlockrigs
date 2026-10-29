"""Imports a rebatched map into Blender and saves it with its textures next to it (run by mapbuild.py).

blender -b --factory-startup -P tools/map_import.py -- <map.gltf> <out.blend> <map slug> <title> <game version>
    [<lighting folder>]

The lighting folder (from mapbuild.py's lighting step) has the lightmaps, the light probe atlas, the sky and
settings.json with what map_environment.py read from the map's entities.
"""
import glob
import json
import os
import shutil
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_lighting  # noqa: E402

args = sys.argv[sys.argv.index('--') + 1:]
gltf, out_blend, slug, title, version = args[:5]
lighting_dir = args[5] if len(args) > 5 else ''
out_dir = os.path.dirname(os.path.abspath(out_blend))
tex_dir = os.path.join(out_dir, 'textures')
light_dir = os.path.join(out_dir, 'lightmaps')
# lights the rigs where the map's own sun is off (Midtown's); the map doesn't see it
RIG_SUN_STRENGTH = 3.0


def load_settings():
    try:
        return json.load(open(os.path.join(lighting_dir, 'settings.json'), encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def find(pattern):
    return glob.glob(os.path.join(lighting_dir, '**', pattern), recursive=True)


def keep(path):
    """Copies a lighting file next to the map, where the .blend finds it."""
    os.makedirs(light_dir, exist_ok=True)
    dest = os.path.join(light_dir, os.path.basename(path))
    shutil.copyfile(path, dest)
    return dest


def lightmap_files():
    """The DC texture and the band textures (one for 8.5, R/G/B for 8.3 and 8.4)."""
    found = {os.path.splitext(os.path.basename(f))[0].rsplit('_', 1)[-1]: f
             for f in find('directional_irradiance_sh2_*')}
    bands = [found['lum']] if 'lum' in found else [found[c] for c in 'rgb' if c in found]
    if 'dc' not in found or len(bands) not in (1, 3):
        return None
    return keep(found['dc']), [keep(b) for b in bands]


settings = load_settings() if lighting_dir else {}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=gltf, import_pack_images=False)

coll = bpy.data.collections.new(title)
bpy.context.scene.collection.children.link(coll)
for ob in list(bpy.context.scene.collection.objects):
    coll.objects.link(ob)
    bpy.context.scene.collection.objects.unlink(ob)
coll['dlr_map'] = slug
coll['dlr_map_version'] = version
if settings.get('tonemap'):
    coll['dlr_tonemap'] = dict(settings['tonemap'], exposure_bias=settings['tonemap']['exposure_bias'] +
                               settings.get('exposure', 0.0))

meshes = [ob for ob in coll.all_objects if ob.type == 'MESH']
lit = [ob for ob in meshes if '_LIGHTMAP_UV' in ob.data.attributes]
unlit = [ob for ob in meshes if '_LIGHTMAP_UV' not in ob.data.attributes]
mats = {slot.material for ob in meshes for slot in ob.material_slots if slot.material}

kinds = [map_lighting.apply_vmat(m) for m in mats]
print(f'materials: {kinds.count("unlit")} unlit, {kinds.count("self_illum")} self-illuminated')

sun = next((ob for ob in coll.objects if ob.type == 'LIGHT' and ob.data.type == 'SUN'), None)
game_sun = settings.get('sun')
if sun:
    sun.name = sun.data.name = f'{title} sun'
    if game_sun and game_sun['strength'] > 0:
        sun.data.energy = game_sun['strength']
        sun.data.color = game_sun['color']
        sun.data.use_shadow = game_sun['shadows']
        if game_sun['baked'] and lit:
            map_lighting.exclude_from_light(sun, lit, f'{title} lightmapped')
    else:
        sun.name = sun.data.name = f'{title} rig light'
        sun.data.energy = RIG_SUN_STRENGTH
        map_lighting.exclude_from_light(sun, meshes, f'{title} map meshes')

probes = 0
volumes = settings.get('probe_volumes')
if volumes and unlit:
    atlas = map_lighting.load_probe_atlas(lighting_dir)
    if atlas is not None:
        probes = map_lighting.bake_probes(unlit, atlas, volumes)
        del atlas
lightmaps = lightmap_files() if lit else None
if lightmaps or probes:
    group = map_lighting.baked_light_group(*(lightmaps or (None, ())))
    done = sum(map_lighting.add_baked_light(m, group) for m in mats)
    print(f'baked light: {len(lit)} lightmapped meshes, {probes:,} vertices lit from light probes, '
          f'{done}/{len(mats)} materials')
else:
    print('WARNING: no supported lightmaps or light probes (needs lightmap version 8.3 to 8.5); '
          'the map has no baked light')

sky = settings.get('sky') or {}
sky_files = find('sky.exr')
sky_mips = None
if sky_files:
    sky_img = bpy.data.images.load(keep(sky_files[0]))
    sky_img.colorspace_settings.name = 'Linear Rec.709'
    strength = sky.get('strength', 1.0)
    world = map_lighting.make_world(f'{title} sky', sky_img, strength)
    world.use_fake_user = True
    coll['dlr_world'] = world.name
    bpy.context.scene.world = world
    sky_mips = [(4, map_lighting.downsample(sky_img, 16)), (7, map_lighting.downsample(sky_img, 128))]
cube, gradient = settings.get('cubemap_fog'), settings.get('gradient_fog')
if (cube and sky_mips) or gradient:
    face = sky_mips[0][1].size[0] * 16 // 4 if sky_mips else 1024
    fog = map_lighting.fog_group(cube, gradient, sky_mips, sky.get('strength', 1.0), face)
    done = sum(map_lighting.add_fog(m, fog) for m in mats)
    print(f'fog: {"cubemap " if cube and sky_mips else ""}{"gradient " if gradient else ""}on {done} materials')

os.makedirs(tex_dir, exist_ok=True)
for img in bpy.data.images:
    src = bpy.path.abspath(img.filepath) if img.filepath else ''
    # the lighting files are already next to the map
    if img.packed_file or not os.path.isfile(src) or os.path.abspath(src).startswith(out_dir + os.sep):
        continue
    dest = os.path.join(tex_dir, os.path.basename(src))
    if not os.path.exists(dest):
        shutil.copyfile(src, dest)
    img.filepath = dest

bpy.ops.wm.save_as_mainfile(filepath=out_blend, relative_remap=True)
print(f'wrote {out_blend}: {len(meshes)} meshes, {len(bpy.data.images)} textures')
