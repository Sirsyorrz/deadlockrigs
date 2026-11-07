"""Imports a rebatched map into Blender and saves it with its textures next to it (run by mapbuild.py).

blender -b --factory-startup -P tools/map_import.py -- <map.gltf> <out.blend> <map slug> <title> <game version>
    [<lighting folder> [<props.json> [<Deadlock install folder>]]]

The lighting folder (from mapbuild.py's lighting step) has the lightmaps, the light probe atlas, the sky and
settings.json with what map_environment.py read from the map's entities. props.json (from the props step) lists
the animated props, each exported on its own. With the install folder, the map's particle effects are built from the
game's files (map_particles.py).
"""
import glob
import json
import os
import shutil
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_anim  # noqa: E402
import map_blend  # noqa: E402
import map_fog  # noqa: E402
import map_lighting  # noqa: E402
import map_probes  # noqa: E402
import map_reflections  # noqa: E402
import map_textures  # noqa: E402

# so the build log shows each line as it happens
sys.stdout.reconfigure(line_buffering=True)
args = sys.argv[sys.argv.index('--') + 1:]
gltf, out_blend, slug, title, version = args[:5]
lighting_dir = args[5] if len(args) > 5 else ''
props_file = args[6] if len(args) > 6 else ''
install = args[7] if len(args) > 7 else ''
out_dir = os.path.dirname(os.path.abspath(out_blend))
tex_dir = os.path.join(out_dir, 'textures')
light_dir = os.path.join(out_dir, 'lightmaps')
# lights the rigs where the map's own sun is off (Midtown's); the map doesn't see it
RIG_SUN_STRENGTH = 3.0
# where each part of the import ends, as a share of a Midtown import's time; mapbuild.py counts the glTF import's
# meshes up to GLTF_END
GLTF_END = 0.22
PROGRESS = {'props': 0.23, 'materials': 0.47, 'paint': 0.48, 'lights': 0.485, 'bake': 0.69, 'probes': 0.905,
            'baked': 0.96, 'fog': 0.97, 'motion': 0.97, 'volumes': 0.98, 'particles': 0.985, 'textures': 0.99}


def progress(frac):
    print(f'PROGRESS {frac:.3f}', flush=True)


_mark = [time.time()]


def reached(part):
    """Progress at the end of a part of the import, and how long it took (in the build log)."""
    now = time.time()
    print(f'  ({part}: {now - _mark[0]:.1f} s)')
    _mark[0] = now
    progress(PROGRESS[part])


def tracked(items, start, end):
    """The items, reporting progress from part `start` to part `end` as they're gone through."""
    items = list(items)
    for i, item in enumerate(items):
        if i % 10 == 0:
            between(start, end, i / len(items))
        yield item


def between(start, end, frac):
    progress(PROGRESS[start] + (PROGRESS[end] - PROGRESS[start]) * frac)


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
print(f'glTF import: {time.time() - _mark[0]:.1f} s')
_mark[0] = time.time()

coll = bpy.data.collections.new(title)
bpy.context.scene.collection.children.link(coll)
for ob in list(bpy.context.scene.collection.objects):
    coll.objects.link(ob)
    bpy.context.scene.collection.objects.unlink(ob)
coll['dlr_map'] = slug
coll['dlr_map_version'] = version
if props_file and os.path.exists(props_file):
    props = map_anim.import_props(props_file, coll)
    print(f'animated props: {sum(ob.type == "ARMATURE" for ob in props)}')
reached('props')
if settings.get('tonemap'):
    coll['dlr_tonemap'] = dict(settings['tonemap'], exposure_bias=settings['tonemap']['exposure_bias'] +
                               settings.get('exposure', 0.0), exposure=settings.get('exposure', 0.0))

meshes = [ob for ob in coll.all_objects if ob.type == 'MESH']
lit = [ob for ob in meshes if '_LIGHTMAP_UV' in ob.data.attributes]
unlit = [ob for ob in meshes if '_LIGHTMAP_UV' not in ob.data.attributes]
mats = {slot.material for ob in meshes for slot in ob.material_slots if slot.material}

glass_masks = {path: found[0] for path, name in (settings.get('glass_masks') or {}).items() if (found := find(name))}
layer_textures = {path: os.path.join(lighting_dir, f) for path, f in (settings.get('layer_textures') or {}).items()}
kinds = [map_lighting.apply_vmat(m, glass_masks, layer_textures) for m in tracked(mats, 'props', 'materials')]
print(f'materials: {kinds.count("unlit")} unlit, {kinds.count("glass")} glass, '
      f'{kinds.count("self_illum")} self-illuminated, {kinds.count("overlay")} decals')
print(f'blend layers: {map_blend.applied} materials, {len(layer_textures)} layer textures')
print(f'vertex paint: decoded on {map_lighting.decode_paint(meshes)} meshes')
reached('paint')

runtime_lights = map_lighting.add_lights(coll, settings.get('lights') or [])
meshes, split = map_lighting.unlight_before_barns(runtime_lights, meshes)
print(f'barn lights: {split} meshes split at their near planes')
lit = [ob for ob in meshes if '_LIGHTMAP_UV' in ob.data.attributes]
unlit = [ob for ob in meshes if '_LIGHTMAP_UV' not in ob.data.attributes]

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

print(f'runtime lights: {len(runtime_lights)}, '
      f'{map_lighting.unblock_own_light(runtime_lights, lambda f: between("paint", "lights", f))} '
      'shadow linked past their own fixtures')
# after unblock_own_light, which would take the masks for fixtures
barn_masks = map_lighting.add_barn_masks(coll, runtime_lights, settings.get('lights') or [], title)
print(f'barn masks: {len(barn_masks)}')
reached('lights')

probes = 0
volumes = settings.get('probe_volumes')
if volumes:
    atlas = map_lighting.load_probe_atlas(lighting_dir)
    if atlas is not None and unlit:
        probes = map_lighting.bake_probes(unlit, atlas, volumes)
    reached('bake')
    map_probes.add_probe_light(coll, title, atlas, volumes, light_dir)
    del atlas
lightmaps = lightmap_files() if lit else None
reflections = None
if settings.get('env_maps'):
    reflections = map_reflections.add_reflections(coll, title, settings['env_maps'],
                                                  os.path.join(lighting_dir, 'cubemaps'),
                                                  os.path.join(lighting_dir, 'brdf'), light_dir)
reached('probes')
if lightmaps or probes:
    group = map_lighting.baked_light_group(*(lightmaps or (None, ())), reflections=reflections)
    done = sum(map_lighting.add_baked_light(m, group) for m in tracked(mats, 'probes', 'baked'))
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
fog = None
if (cube and sky_mips) or gradient:
    face = sky_mips[0][1].size[0] * 16 // 4 if sky_mips else 1024
    fog = map_lighting.fog_group(cube, gradient, sky_mips, sky.get('strength', 1.0), face)
    done = sum(map_lighting.add_fog(m, fog) for m in tracked(mats, 'baked', 'fog'))
    print(f'fog: {"cubemap " if cube and sky_mips else ""}{"gradient " if gradient else ""}on {done} materials')

reached('fog')
print(f'additive: {sum(map_lighting.make_additive(m) for m in mats)} lit materials')
anim_textures = {path: os.path.join(lighting_dir, f) for path, f in (settings.get('anim_textures') or {}).items()}
moving, moved = map_anim.animate(meshes, anim_textures)
print(f'motion: {moving} scrolling or jittering materials, {moved} jittering meshes')
reached('motion')

fog_volume = settings.get('volumetric_fog')
if fog_volume:
    os.makedirs(light_dir, exist_ok=True)
    fog_ob = map_fog.add_volumetric_fog(coll, fog_volume, os.path.join(lighting_dir, 'volumetric_fog'),
                                        os.path.join(light_dir, 'volumetric_fog.vdb'), title)
    if fog_ob:
        fog_lights = map_fog.light_fog(coll, [ob for ob in coll.objects if ob.type == 'LIGHT'])
        print(f'volumetric fog: {len(fog_volume["volumes"])} fog volumes, lit by {fog_lights} lights')
    else:
        print('volumetric fog: none')

reached('volumes')
if install:
    import map_particles
    try:
        built, warnings = map_particles.add_particles(coll, slug, install, os.path.join(os.path.dirname(lighting_dir),
                                                                                       'particles'), fog)
        print(f'particles: {built} effects')
        for w in warnings:
            print(f'  {w}')
    except Exception as e:  # the map is still worth having without them
        import traceback
        traceback.print_exc()
        print(f'WARNING: particle effects failed: {e}')
reached('particles')
os.makedirs(tex_dir, exist_ok=True)
written = set()
for img in tracked(bpy.data.images, 'volumes', 'textures'):
    src = bpy.path.abspath(img.filepath) if img.filepath else ''
    # the lighting files are already next to the map
    if img.packed_file or not os.path.isfile(src):
        continue
    if os.path.abspath(src).startswith(out_dir + os.sep):
        written.add(os.path.abspath(src))
        continue
    dest = os.path.join(tex_dir, os.path.basename(src))
    # always overwritten: a texture left by an earlier build can be a mod's (Include Mods) or an older game's
    if dest not in written:
        shutil.copyfile(src, dest)
        written.add(dest)
    img.filepath = dest
for name in os.listdir(tex_dir):
    path = os.path.join(tex_dir, name)
    if path not in written and os.path.isfile(path):
        os.remove(path)
for size in map_textures.SIZES:
    shutil.rmtree(map_textures.tier_dir(out_dir, size), ignore_errors=True)
print(f'smaller textures: {map_textures.make_tiers(out_dir)}')

reached('textures')
bpy.ops.wm.save_as_mainfile(filepath=out_blend, relative_remap=True)
print(f'wrote {out_blend}: {len(meshes)} meshes, {len(bpy.data.images)} textures')
