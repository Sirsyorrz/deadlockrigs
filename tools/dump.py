"""blender -b -P dump.py -- <hero> : face mesh arrays (world space, Blender axes) and eye bone heads -> data/head.npz"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hero import from_argv  # noqa: E402

hero, _ = from_argv()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=hero.raw_glb)
o = next(o for o in bpy.context.scene.objects if o.name.endswith('.' + hero.face['part']))
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
np.savez(os.path.join(hero.data, 'head.npz'), co=co, nrm=nrm.reshape(-1, 3), tris=tris, mats=mats, matnames=np.array([m.name for m in me.materials]),
         **{'bone_' + k: v for k, v in bones.items()})
print('dumped', n, len(tris))
