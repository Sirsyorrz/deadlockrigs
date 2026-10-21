"""blender -b -P slice.py -- in.glb axis value tol zmin zmax : print verts of the welded head near a plane"""
import sys

import bmesh
import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
axis, val, tol, zmin, zmax = argv[1], float(argv[2]), float(argv[3]), float(argv[4]), float(argv[5])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=argv[0])
o = next(o for o in bpy.context.scene.objects if o.name.endswith('.head'))
bm = bmesh.new()
bm.from_mesh(o.data)
bm.transform(o.matrix_world)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.verts.ensure_lookup_table()
ai = 'xyz'.index(axis)
sel = [v for v in bm.verts if abs(v.co[ai] - val) < tol and zmin <= v.co.z <= zmax]
ids = {v.index for v in sel}
for v in sorted(sel, key=lambda v: (round(v.co.z, 4), v.co.y)):
    nb = [e.other_vert(v).index for e in v.link_edges if e.other_vert(v).index in ids]
    print('V', v.index, 'x=%.4f y=%.4f z=%.4f' % tuple(v.co), 'mat', v.link_faces[0].material_index if v.link_faces else -1, 'nb', nb)
