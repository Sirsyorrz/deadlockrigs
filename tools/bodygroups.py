"""Default visibility of a hero's render meshes, read from the decompiled (KV3 text) vmdl."""
import re


def mesh_parts(vmdl_text):
    """mesh name -> visible by default. A bodygroup shows its first choice, so meshes only reachable through a
    later choice (alternate heads, reload props) start hidden; meshes outside every bodygroup are always on."""
    meshes = [m for m in re.findall(r'_class = "RenderMeshFile"\s*name = "([^"]+)"', vmdl_text)
              if not re.search(r'_lod\d+$', m)]
    default_on, listed = set(), set()
    groups = vmdl_text[vmdl_text.find('"BodyGroupList"'):]
    for g in re.finditer(r'_class = "BodyGroup"\s*name = "([^"]+)"(.*?)(?=_class = "BodyGroup"\s*name|\Z)', groups, re.S):
        choices = [re.findall(r'"([^"]+)"', c) for c in re.findall(r'meshes =\s*\[(.*?)\]', g.group(2), re.S)]
        for i, choice in enumerate(choices):
            listed.update(choice)
            if i == 0:
                default_on.update(choice)
    return {m: m in default_on or m not in listed for m in meshes}


def keep_faces(o, keep):
    """Deletes the faces keep(face) rejects, and the vertices left with none."""
    import bmesh
    import bpy
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not keep(f)], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.to_mesh(o.data)
    bm.free()
    with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o], selected_editable_objects=[o]):
        bpy.ops.object.material_slot_remove_unused()


def split_meshes(hero):
    """Pieces baked into another mesh (Violet's portfolio, heads inside a body mesh) become their own objects, named
    after their heroes.json "split" key; build.py and dump.py split the same way, so vertex order matches.
    A split is a list of materials, or {"mats": [...], "bones": [...]} for a head that shares the body's material
    (Infernus'): the faces whose every vertex is mostly weighted to those bones or below them; "mesh" picks the
    mesh by name when several share the material."""
    import bpy
    for name, spec in hero.cfg.get('split', {}).items():
        mats = spec if isinstance(spec, list) else spec['mats']
        cands = [o for o in bpy.data.objects if o.type == 'MESH' and any(m and m.name in mats for m in o.data.materials)
                 and (not isinstance(spec, dict) or 'mesh' not in spec or o.name.split('.')[-1] == spec['mesh'])]
        src, inside = cands[0], None
        if isinstance(spec, dict) and 'bones' in spec:
            def weighted(o):
                under = {b.name for b in o.find_armature().data.bones
                         if any(b.name == r or any(p.name == r for p in b.parent_recursive) for r in spec['bones'])}
                groups = {g.index for g in o.vertex_groups if g.name in under}
                return [sum(g.weight for g in v.groups if g.group in groups) >= 0.5 for v in o.data.vertices]
            # the material can be shared by several meshes (Warden's coat backfaces): the one holding the head
            src, inside = max(((o, weighted(o)) for o in cands), key=lambda c: sum(c[1]))
        idx = {i for i, m in enumerate(src.data.materials) if m and m.name in mats}
        part = src.copy()
        part.data = src.data.copy()
        part.name = part.data.name = name
        for c in src.users_collection:
            c.objects.link(part)

        def mine(f):
            return f.material_index in idx and (inside is None or all(inside[v.index] for v in f.verts))
        keep_faces(part, mine)
        keep_faces(src, lambda f: not mine(f))
        print('split', name, 'from', src.name, len(part.data.vertices), 'vertices')
