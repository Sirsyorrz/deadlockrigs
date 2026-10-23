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
# some skeletons name the eye bones eye_L/R (Wraith); keys.py reads them as eye_0_L/R
ALIAS = {'eye_0_L': 'eye_L', 'eye_0_R': 'eye_R'}
bones = {}
for b in ('head', 'jaw_0', 'eye_0_L', 'eye_0_R', 'neck_0'):
    name = b if b in arm.pose.bones else ALIAS.get(b)
    if name and name in arm.pose.bones:
        bones[b] = np.array(arm.matrix_world @ arm.pose.bones[name].head)
vmat = np.zeros(len(co), int)
vmat[tris.ravel()] = np.repeat(mats, 3)
eye_mat = hero.face.get('eyes')
if isinstance(eye_mat, list):
    # roughly where the left eye is (Shiv: no eye bones, eyeballs in the head material); the pieces there are found
    # by shape below
    bones['eye_0_L'] = np.array(eye_mat, float)
    bones['eye_0_R'] = bones['eye_0_L'] * [-1, 1, 1]
    eye_mat = None
if eye_mat:
    # eyeballs by material (heroes.json face "eyes"), for heads with no eye bones (Kelvin): each side's eye bone is
    # the centre of the sphere through its vertices
    idx = [i for i, m in enumerate(me.materials) if m and m.name == eye_mat]
    on = np.isin(vmat, idx)
    for s, sx in (('L', 1), ('R', -1)):
        q = co[on & (co[:, 0] * sx > 0)]
        a = np.c_[2 * q, np.ones(len(q))]
        c = np.linalg.lstsq(a, (q ** 2).sum(1), rcond=None)[0][:3]
        bones[f'eye_0_{s}'] = c
        print('eye', s, 'from', eye_mat, np.round(c, 4))
if 'jaw_0' not in bones:
    # no jaw bone (Silver): where the others' sits from their eyes, behind and below them by eye spacing
    eyes = (bones['eye_0_L'] + bones['eye_0_R']) / 2
    s = abs(bones['eye_0_L'][0] - bones['eye_0_R'][0]) / 0.088
    bones['jaw_0'] = np.array([0, eyes[1] + 0.07 * s, eyes[2] - 0.047 * s])
    print('no jaw_0: jaw pivot placed at', bones['jaw_0'])
balls = {}
for s, sx in (('L', 1), ('R', -1)):
    if eye_mat:
        balls['ball_' + s] = np.isin(vmat, idx) & (co[:, 0] * sx > 0)
        continue
    gi = o.vertex_groups[f'eye_0_{s}'].index if f'eye_0_{s}' in o.vertex_groups else -1
    balls['ball_' + s] = np.array([any(g.group == gi and g.weight > 0.5 for g in v.groups) for v in me.vertices])
if not (balls['ball_L'].any() and balls['ball_R'].any()):
    # eyeballs skinned to the head, not the eye bones (Lady Geist's): the separate piece around each eye bone
    # pieces by position-welded edges (Blender's Python has no scipy): labels spread to the smallest until stable
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    e = weld[np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]]])]
    lab = np.arange(weld.max() + 1)
    while True:
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        new = lab.copy()
        np.minimum.at(new, e[:, 0], m)
        np.minimum.at(new, e[:, 1], m)
        new = new[new]
        if (new == lab).all():
            break
        lab = new
    comp = lab[weld]
    for s in 'LR':
        best, best_d = None, 0.02
        for c in np.unique(comp[np.linalg.norm(co - bones[f'eye_0_{s}'], axis=1) < 0.03]):
            m = comp == c
            size = np.ptp(co[m], axis=0).max()
            d = np.linalg.norm(co[m].mean(0) - bones[f'eye_0_{s}'])
            if m.sum() > 50 and size < 0.06 and d < best_d:
                best, best_d = m, d
        if best is not None:
            balls['ball_' + s] = best
    print('eyeballs found by shape:', int(balls['ball_L'].sum()), int(balls['ball_R'].sum()))
    if isinstance(hero.face.get('eyes'), list):
        for s in 'LR':
            q = co[balls['ball_' + s]]
            a = np.c_[2 * q, np.ones(len(q))]
            fit = np.linalg.lstsq(a, (q ** 2).sum(1), rcond=None)[0]
            c, r = fit[:3], np.sqrt(fit[3] + fit[:3] @ fit[:3])
            # the cornea shell over the ball (Shiv's) and glints are other small pieces: everything small that sits on
            # the sphere. Not the tear duct in the inner corner (Pocket's), which reaches well off it and turned
            # with the eye onto its white
            for k in np.unique(comp[np.linalg.norm(co - c, axis=1) < r * 1.3]):
                m = comp == k
                if np.ptp(co[m], axis=0).max() < 2.5 * r and \
                        (np.abs(np.linalg.norm(co[m] - c, axis=1) / r - 1) < 0.15).all():
                    balls['ball_' + s] |= m
            q = co[balls['ball_' + s]]
            a = np.c_[2 * q, np.ones(len(q))]
            bones[f'eye_0_{s}'] = np.linalg.lstsq(a, (q ** 2).sum(1), rcond=None)[0][:3]
            print('eye', s, 'centre', np.round(bones[f'eye_0_{s}'], 4))
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
