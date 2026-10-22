"""blender -b -P dump.py -- <hero> : face mesh arrays (world space, Blender axes), eye bone heads and eyeball
vertices (skinned to the eye bones) -> data/head.npz"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bodygroups import mesh_parts, split_meshes  # noqa: E402
from hero import from_argv  # noqa: E402

hero, _ = from_argv()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=hero.raw_glb)
split_meshes(hero)
o = next(o for o in bpy.context.scene.objects if o.name == hero.face['part'] or o.name.endswith('.' + hero.face['part']))
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
me = o.data
n = len(me.vertices)
co = np.empty(n * 3, np.float32)
me.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
mw = np.array(o.matrix_world)
co = co @ mw[:3, :3].T + mw[:3, 3]
nrm = np.empty(n * 3, np.float32)
me.vertices.foreach_get('normal', nrm)
loops = np.empty(len(me.loops), np.int32)
me.loops.foreach_get('vertex_index', loops)
starts = np.empty(len(me.polygons), np.int32)
me.polygons.foreach_get('loop_start', starts)
counts = np.empty(len(me.polygons), np.int32)
me.polygons.foreach_get('loop_total', counts)
mats = np.empty(len(me.polygons), np.int32)
me.polygons.foreach_get('material_index', mats)
assert (counts == 3).all(), 'expected triangles'
tris = loops.reshape(-1, 3)
bones = {b: np.array(arm.matrix_world @ arm.pose.bones[b].head) for b in ('head', 'jaw_0', 'eye_0_L', 'eye_0_R', 'neck_0')}
balls = {}
for s in 'LR':
    gi = o.vertex_groups[f'eye_0_{s}'].index
    balls['ball_' + s] = np.array([any(g.group == gi and g.weight > 0.5 for g in v.groups) for v in me.vertices])
# other shown meshes around the head (hair, hats, collars): the face keys must not push skin through them
parts = mesh_parts(open(hero.vmdl).read())
others = []
for m in bpy.context.scene.objects:
    if m.type != 'MESH' or m is o or m.parent is None:
        continue
    part = next((p for p in sorted(parts, key=len, reverse=True) if m.name.endswith('.' + p)), m.name.rsplit('.', 1)[-1])
    if part in hero.drop or not parts.get(part, True):
        continue
    q = np.empty(len(m.data.vertices) * 3, np.float32)
    m.data.vertices.foreach_get('co', q)
    w = np.array(m.matrix_world)
    q = q.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3]
    others.append(q[np.linalg.norm(q - bones['head'], axis=1) < 0.3])
other = np.concatenate(others) if others else np.zeros((0, 3), np.float32)
np.savez(os.path.join(hero.data, 'head.npz'), co=co, nrm=nrm.reshape(-1, 3), tris=tris, mats=mats, matnames=np.array([m.name for m in me.materials]),
         other=other, **{'bone_' + k: v for k, v in bones.items()}, **balls)
print('dumped', n, len(tris), 'other', len(other))
