"""blender -b -P topo.py -- in.glb : components and boundary loops of the head mesh"""
import sys
from collections import defaultdict

import bmesh
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=argv[0])
o = next(o for o in bpy.context.scene.objects if o.name.endswith('.head'))
bm = bmesh.new()
bm.from_mesh(o.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.verts.ensure_lookup_table()
mw = o.matrix_world

seen = set()
comps = []
for v in bm.verts:
    if v.index in seen:
        continue
    stack = [v]
    comp = []
    seen.add(v.index)
    while stack:
        a = stack.pop()
        comp.append(a)
        for e in a.link_edges:
            b = e.other_vert(a)
            if b.index not in seen:
                seen.add(b.index)
                stack.append(b)
    comps.append(comp)


def box(vs):
    ws = [mw @ v.co for v in vs]
    lo = Vector(map(min, *ws)) if len(ws) > 1 else ws[0]
    hi = Vector(map(max, *ws)) if len(ws) > 1 else ws[0]
    return tuple(round(x, 3) for x in lo), tuple(round(x, 3) for x in hi)


for c in sorted(comps, key=len, reverse=True):
    mats = {f.material_index for v in c for f in v.link_faces}
    print('COMP', len(c), mats, box(c))

# welded positions: verts sharing a position (UV seams split them)
pos = defaultdict(list)
for v in bm.verts:
    pos[tuple(round(x, 5) for x in v.co)].append(v.index)
print('unique positions', len(pos), 'verts', len(bm.verts))

bedges = [e for e in bm.edges if e.is_boundary]
adj = defaultdict(list)
for e in bedges:
    adj[e.verts[0].index].append(e.verts[1].index)
    adj[e.verts[1].index].append(e.verts[0].index)
done = set()
loops = []
for s in adj:
    if s in done:
        continue
    stack = [s]
    lp = []
    done.add(s)
    while stack:
        a = stack.pop()
        lp.append(a)
        for b in adj[a]:
            if b not in done:
                done.add(b)
                stack.append(b)
    loops.append(lp)
for lp in sorted(loops, key=len, reverse=True)[:40]:
    print('LOOP', len(lp), box([bm.verts[i] for i in lp]))
