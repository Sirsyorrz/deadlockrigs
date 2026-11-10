"""Rigid parts of a hard-surface head or body (heroes.json "mech_parts"): Bebop's jaw, monocle and head switches.

The game skins them to one bone with the rest (Bebop's whole head is on 'head'). Each part here is the pieces of
the mesh (connected, mostly on that bone) whose centre lies in one of its boxes; they move to a new bone with the
same weight, so the rest pose is unchanged. Boxes and pivots are in mm from the head bone, Blender axes
(+x the hero's left, -y forward, +z up):

    "mech_parts": {"jaw": {"bone": "head", "boxes": [[x0, y0, z0, x1, y1, z1]], "head": [x, y, z],
                           "tail": [x, y, z]}}

"mesh" limits a part to one mesh (Paradox's hourglass, not the glass around it); "limits" see pose().

A soft part (Krill's snout, part of his head skin) has "sphere": [x, y, z, r, soft] instead of boxes: it takes the
share 1 - smoothstep(r, r + soft, distance) of each vertex's bone weight, so the skin bends into it.
"""
import math

import bmesh
import numpy as np
from mathutils import Vector


def load(cfg, arm):
    parts = cfg.get('mech_parts', {})
    if not parts:
        return {}
    origin = arm.data.bones['head'].head_local if 'head' in arm.data.bones else Vector()
    out = {}
    for name, p in parts.items():
        out[name] = dict(p, head=origin + Vector(p['head']) / 1000, tail=origin + Vector(p['tail']) / 1000,
                         boxes=[np.array(b, float).reshape(2, 3) / 1000 + np.array(origin) for b in p.get('boxes', [])])
        if 'sphere' in p:
            x, y, z, r, soft = p['sphere']
            out[name]['sphere'] = (np.array((x, y, z)) / 1000 + np.array(origin), r / 1000, soft / 1000)
    return out


def edit(parts, new_bone):
    for name, p in parts.items():
        b = new_bone(name, p['head'], p['tail'], p['bone'], roll_to=Vector((0, 0, 1)))
        b.use_deform = True


def pieces(o, bone, to_arm):
    """Connected pieces of the mesh mostly on bone: (vertex indices, centre in armature space)."""
    gi = o.vertex_groups[bone].index
    w = np.zeros(len(o.data.vertices))
    for v in o.data.vertices:
        for g in v.groups:
            if g.group == gi:
                w[v.index] = g.weight
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.ensure_lookup_table()
    seen = np.zeros(len(w), bool)
    co = np.array([tuple(to_arm @ v.co) for v in o.data.vertices])
    out = []
    for v in bm.verts:
        if seen[v.index]:
            continue
        stack, comp = [v], []
        seen[v.index] = True
        while stack:
            x = stack.pop()
            comp.append(x.index)
            for e in x.link_edges:
                y = e.other_vert(x)
                if not seen[y.index]:
                    seen[y.index] = True
                    stack.append(y)
        if w[comp].mean() >= 0.5:
            out.append((comp, co[comp].mean(0)))
    bm.free()
    return out, w


def soft_share(o, part, w, to_arm):
    centre, r, soft = part['sphere']
    co = np.array([tuple(to_arm @ v.co) for v in o.data.vertices])
    t = np.clip((np.linalg.norm(co - centre, axis=1) - r) / soft, 0, 1)
    return w * (1 - t * t * (3 - 2 * t))


def weights(parts, arm):
    """Moves each part's pieces onto its bone; returns {part: vertex count}."""
    counts = {n: 0 for n in parts}
    meshes = [o for o in arm.children_recursive if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
    for o in meshes:
        to_arm = arm.matrix_world.inverted() @ o.matrix_world
        need = {p['bone'] for p in parts.values() if p['bone'] in o.vertex_groups}
        for bone in need:
            found, w = pieces(o, bone, to_arm)
            src = o.vertex_groups[bone]
            for name, p in parts.items():
                if p['bone'] != bone or p.get('mesh', o.name) != o.name:
                    continue
                if 'sphere' in p:
                    share = soft_share(o, p, w, to_arm)
                    g = o.vertex_groups.get(name) or o.vertex_groups.new(name=name)
                    for i in np.where(share > 1e-4)[0]:
                        g.add([int(i)], float(share[i]), 'REPLACE')
                        left = float(w[i] - share[i])
                        if left > 1e-4:
                            src.add([int(i)], left, 'REPLACE')
                        else:
                            src.remove([int(i)])
                    counts[name] += int((share > 1e-4).sum())
                    continue
                take = [comp for comp, c in found if any(np.all(c >= b[0]) and np.all(c <= b[1]) for b in p['boxes'])]
                if not take:
                    continue
                g = o.vertex_groups.get(name) or o.vertex_groups.new(name=name)
                for comp in take:
                    for i in comp:
                        if w[i] > 0:
                            g.add([int(i)], float(w[i]), 'REPLACE')
                    src.remove([int(i) for i in comp])
                    counts[name] += len(comp)
    return counts


def pose(parts, widget, circle, arm=None):
    """Widgets, and the part's "limits" ({"rot_x": [min, max] degrees, "loc_y": [min, max] mm, ...}) as Limit
    constraints, so a jaw can't close through the head."""
    for name, p in parts.items():
        lim = p.get('limits', {})
        pbone = arm.pose.bones[name] if arm else None
        rot = {k[-1]: v for k, v in lim.items() if k.startswith('rot_')}
        loc = {k[-1]: v for k, v in lim.items() if k.startswith('loc_')}
        if pbone and rot:
            c = pbone.constraints.new('LIMIT_ROTATION')
            c.owner_space = 'LOCAL'
            c.use_transform_limit = True
            for ax, (lo, hi) in rot.items():
                setattr(c, f'use_limit_{ax}', True)
                setattr(c, f'min_{ax}', math.radians(lo))
                setattr(c, f'max_{ax}', math.radians(hi))
        if pbone and loc:
            c = pbone.constraints.new('LIMIT_LOCATION')
            c.owner_space = 'LOCAL'
            c.use_transform_limit = True
            for ax, (lo, hi) in loc.items():
                setattr(c, f'use_min_{ax}', True)
                setattr(c, f'use_max_{ax}', True)
                setattr(c, f'min_{ax}', lo / 1000)
                setattr(c, f'max_{ax}', hi / 1000)
        d = p['tail'] - p['head']
        # a ring around the tip: grabbing the far end is what turns a hinge
        widget(name, circle(p['tail'], d, max(d.length * 0.25, 0.008)), 2)
