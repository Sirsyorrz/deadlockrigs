"""blender -b -P anims.py -- <hero>

Imports the game's animation clips (exported with --gltf_export_animations) and saves them as a light
action library: names shortened, channels that stay at the rest pose dropped, constant channels reduced
to one key. Each action gets a fake user and one slot named after the rig's armature. Writes
out/<hero>_anims.blend."""
import json
import os
import struct
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402
import mirror  # noqa: E402

hero, _ = from_argv()
src, out = hero.anim_glb, hero.anims_blend


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


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
clip_glb = short_clip_glb(src)
bpy.ops.import_scene.gltf(filepath=clip_glb)
os.remove(clip_glb)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

REST = {'location': 0.0, 'rotation_quaternion': None, 'scale': 1.0}
EPS = 1e-4

keep_names = {'game_' + a.name: a for a in bpy.data.actions}

# the rig turns these right-side rest frames (mirror.py), so the keys are stored for the turned bones
TURNS = mirror.turns({b.name: b.matrix_local for b in arm.data.bones})
uneven = set()
for a in keep_names.values():
    uneven.update(mirror.turn_keys(a.layers[0].strips[0].channelbags[0], TURNS))
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
used = set()
for a in keep_names.values():
    for path, fs in channels(a)[1].items():
        if not at_rest(path, fs):
            used.add(path)

total_before = total_after = 0
for n, a in keep_names.items():
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
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out), compress=True)
