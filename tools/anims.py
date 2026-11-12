"""blender -b -P anims.py -- <hero>

Imports the game's animation clips (exported with --gltf_export_animations) and saves them as a light
action library: names shortened, channels that stay at the rest pose dropped, constant channels reduced
to one key. Each action gets a fake user and one slot named after the rig's armature. Writes
out/<hero>_anims.blend."""
import json
import os
import re
import struct
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402
import mirror  # noqa: E402

hero, _ = from_argv()
src, out = hero.anim_glb, hero.anims_blend


def progress(frac, part=None):
    """For pipeline.py: PART names what runs now, PROGRESS is the share of this script done."""
    if part:
        print(f'PART {part}', flush=True)
    print(f'PROGRESS {frac:.3f}', flush=True)


# where each part ends, as a share of the script's time (haze, 404 clips: 29 s)
IMPORTED, TURNED, EXTRA, USED, REDUCED = 0.65, 0.78, 0.8, 0.87, 0.955
# within an import: shortening the clip names, then the importer reading the file (no hook to report from)
SHORTENED, LOADED = 0.08, 0.48


def short_clip_glb(path):
    """Copy of the glb with clips named by their last path part. Blender cuts action names at 63 bytes, so long
    model paths (models/npc/boss_tier_01_brazier_guardian/clips/...) lost the clip name's end. Bare duplicates of
    clips (e.g. 'weapon_stand_idle' vs '.../clips/weapon_stand_idle') are dropped, the first one wins."""
    data = open(path, 'rb').read()
    jlen = struct.unpack_from('<I', data, 12)[0]
    gltf = json.loads(data[20:20 + jlen])
    seen, anims = set(), []
    for a in gltf.get('animations', []):
        a['name'] = a.get('name', '').rsplit('/', 1)[-1]
        if a['name'] not in seen:
            seen.add(a['name'])
            anims.append(a)
    gltf['animations'] = anims
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    rest = data[20 + jlen:]
    fd, tmp = tempfile.mkstemp(suffix='.glb')
    with os.fdopen(fd, 'wb') as f:
        f.write(data[:8] + struct.pack('<I', 20 + len(js) + len(rest)) + struct.pack('<I', len(js)) + b'JSON')
        f.write(js + rest)
    return tmp


def import_clips(path, lo, hi):
    """Imports the glb's clips with progress from lo to hi. Creating the actions is most of the import's time, so
    the importer's per-clip call is wrapped to report each one."""
    clip_glb = short_clip_glb(path)
    progress(lo + (hi - lo) * SHORTENED)
    try:
        from io_scene_gltf2.blender.imp.animation import BlenderAnimation
        anim = BlenderAnimation.anim
    except (ImportError, AttributeError):
        BlenderAnimation = None
    if BlenderAnimation:
        done = [0]

        def anim_with_progress(gltf, anim_idx):
            anim(gltf, anim_idx)
            done[0] += 1
            if done[0] % 10 == 0:
                share = LOADED + (1 - LOADED) * done[0] / len(gltf.data.animations)
                progress(lo + (hi - lo) * share)
        BlenderAnimation.anim = staticmethod(anim_with_progress)
    try:
        bpy.ops.import_scene.gltf(filepath=clip_glb)
    finally:
        if BlenderAnimation:
            BlenderAnimation.anim = staticmethod(anim)
        os.remove(clip_glb)
    progress(hi)


progress(0.0, 'anims_import')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
import_clips(src, 0.0, IMPORTED)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

REST = {'location': 0.0, 'rotation_quaternion': None, 'scale': 1.0}
EPS = 1e-4

keep_names = {'game_' + a.name: a for a in bpy.data.actions}

# the rig turns these right-side rest frames (mirror.py), so the keys are stored for the turned bones
TURNS = mirror.turns({b.name: b.matrix_local for b in arm.data.bones})
uneven = set()
for i, a in enumerate(keep_names.values()):
    if i % 20 == 0:
        progress(IMPORTED + (TURNED - IMPORTED) * i / len(keep_names))
    uneven.update(mirror.turn_keys(a.layers[0].strips[0].channelbags[0], TURNS))
progress(TURNED)

# an extra model's clips (the patrons' weakened core) key its bones as build.py named them in the hero's armature,
# <key>__<bone>
for i, (key, glb) in enumerate(hero.extra_clips):
    n = len(hero.extra_clips)
    before = set(bpy.data.actions)
    import_clips(glb, TURNED + (EXTRA - TURNED) * i / n, TURNED + (EXTRA - TURNED) * (i + 1) / n)
    own = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and o != arm)
    turns = mirror.turns({b.name: b.matrix_local for b in own.data.bones})
    for a in [a for a in bpy.data.actions if a not in before]:
        cb = a.layers[0].strips[0].channelbags[0]
        uneven.update(f'{key}__{b}' for b in mirror.turn_keys(cb, turns))
        for f in cb.fcurves:
            if f.data_path.startswith('pose.bones["'):
                f.data_path = f'pose.bones["{key}__' + f.data_path[len('pose.bones["'):]
        for g in cb.groups:
            g.name = f'{key}__{g.name}'
        # Blender numbers a clip named like one of the hero's (the core's idle)
        keep_names[f'game_{key}_' + re.sub(r'\.\d{3}$', '', a.name)] = a
    bpy.data.objects.remove(own, do_unlink=True)
if uneven:
    print('clips: non-uniform scale on turned bones, not exact:', sorted(uneven))


def channels(a):
    cb = a.layers[0].strips[0].channelbags[0]
    by_path = {}
    for f in cb.fcurves:
        by_path.setdefault(f.data_path, []).append(f)
    return cb, by_path


def at_rest(path, fs):
    prop = path.rsplit('.', 1)[-1]
    for f in fs:
        rest = (1.0 if f.array_index == 0 else 0.0) if prop == 'rotation_quaternion' else REST.get(prop, 0.0)
        if any(abs(k.co[1] - rest) > EPS for k in f.keyframe_points):
            return False
    return True


# a channel is dropped only if it stays at rest in every clip; otherwise clips that leave it at rest still
# key it once, so switching clips never leaves a stale value behind
progress(EXTRA, 'anims_reduce')
used = set()
for a in keep_names.values():
    for path, fs in channels(a)[1].items():
        if not at_rest(path, fs):
            used.add(path)

progress(USED)
total_before = total_after = 0
for i, (n, a) in enumerate(keep_names.items()):
    if i % 10 == 0:
        progress(USED + (REDUCED - USED) * i / len(keep_names))
    a.name = n
    a.slots[0].name_display = hero.name
    cb, by_path = channels(a)
    for path, fs in by_path.items():
        total_before += sum(len(f.keyframe_points) for f in fs)
        if path not in used:
            for f in fs:
                cb.fcurves.remove(f)
            continue
        for f in fs:
            vs = [k.co[1] for k in f.keyframe_points]
            if len(vs) > 1 and max(vs) - min(vs) < EPS:
                f.keyframe_points.clear()
                f.keyframe_points.insert(0, vs[0])
            total_after += len(f.keyframe_points)
    a.use_fake_user = True
    r = a.frame_range
    a.use_frame_range = True
    a.frame_start, a.frame_end = r[0], max(r[1], r[0] + 1)
    a['game_clip'] = True
    a['mirror_keys'] = True

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.textures):
    for d in list(coll):
        coll.remove(d)
print(f'clips {len(keep_names)}, keys {total_before} -> {total_after}')
progress(REDUCED, 'anims_save')
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out), compress=True)
