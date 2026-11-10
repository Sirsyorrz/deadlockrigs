"""Builds the bone face (facebones.py's data/facerig.json + facerig.npz) into the rig. Runs inside rig.py.

Chain bones (snout, whiskers) are controls and deform the skin themselves. A tweak is three bones: <name>, the control,
under MCH-follow_<name>, which drivers move with the shape keys so the control stays on the skin whatever the face
does; and DEF-<name>, which deforms, under the deform copy of the tweak's parent, copying the control's own (local)
transform. The skin already carries the shape keys, so the deform side must not get the follow offset too.
"""
import json
import math
import os

import numpy as np
from mathutils import Vector

UP = Vector((0, 0, 1))
FWD = Vector((0, -1, 0))
# drivers past 256 characters are not evaluated as simple expressions; stacked follow bones split longer sums
EXPR_MAX = 250
COLL_MAIN, COLL_FINE = 'Face Bones', 'Face Detail'
PAL = {'chain': 'THEME07', 'main': 'THEME05', 'fine': 'THEME09'}


def load(data_dir):
    path = os.path.join(data_dir, 'facerig.json')
    if not os.path.exists(path):
        return None
    rig = json.load(open(path))
    rig['weights'] = dict(np.load(os.path.join(data_dir, 'facerig.npz')))
    rig['by_name'] = {b['name']: b for b in rig['bones']}
    return rig


def is_tweak(rig, name):
    b = rig['by_name'].get(name)
    return b is not None and b['kind'] == 'tweak'


def ctrl_of(rig, name):
    """The control a bone hangs under: tweaks, chains and game bones are their own controls."""
    return name


def def_of(rig, name):
    return 'DEF-' + name if is_tweak(rig, name) else name


def anchor_offsets(b, head_ob, arm):
    """Shape key -> how far it moves the tweak's anchor vertex (armature space)."""
    a = b.get('anchor', -1)
    key = head_ob.data.shape_keys
    if a < 0 or key is None:
        return {}
    to_arm = (arm.matrix_world.inverted() @ head_ob.matrix_world).to_3x3()
    base = key.key_blocks[0].data[a].co
    return {kb.name: to_arm @ (kb.data[a].co - base) for kb in key.key_blocks[1:]}


def follow_terms(rig, b, head_ob, arm, rest3):
    """(key name, local offset) per shape key that moves the tweak's anchor vertex, in the follow bone's rest frame.
    Under another tweak only what the keys move it past that tweak's anchor: the parent already follows its own."""
    own = anchor_offsets(b, head_ob, arm)
    parent = rig['by_name'].get(b['parent'])
    if parent is not None and parent['kind'] == 'tweak':
        theirs = anchor_offsets(parent, head_ob, arm)
        own = {k: d - theirs.get(k, Vector()) for k, d in own.items()}
    inv = rest3.inverted()
    return [(k, inv @ d) for k, d in own.items() if d.length > 0.0003]


def chunks(terms):
    """Split the terms so each axis' expression fits a simple driver."""
    out, cur = [], []
    for t in terms:
        if len(cur) >= 14:
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def edit(rig, eb, new_bone, head_ob, arm):
    """Edit mode: the bones."""
    rig['follow'] = {}
    for b in rig['bones']:
        n, parent = b['name'], b['parent']
        h, t = Vector(b['head']), Vector(b['tail'])
        if b['kind'] == 'chain':
            d = (t - h).normalized()
            bone = new_bone(n, h, t, ctrl_of(rig, parent), roll_to=UP if abs(d.dot(UP)) < 0.9 else FWD)
            bone.use_deform = True
            continue
        size = b['size']
        tail = h + UP * size
        prev = ctrl_of(rig, parent)
        # the follow bones share the control's rest frame, so the offsets are in that frame
        mch = new_bone(f'MCH-follow_{n}', h, tail, prev, roll_to=FWD)
        terms = follow_terms(rig, b, head_ob, arm, mch.matrix.to_3x3())
        groups = chunks(terms)
        names = [mch.name]
        for k in range(1, len(groups)):
            names.append(new_bone(f'MCH-follow{k + 1}_{n}', h, tail, names[-1], roll_to=FWD).name)
        rig['follow'][n] = list(zip(names, groups)) if terms else []
        new_bone(n, h, tail, names[-1], roll_to=FWD)
        dfm = new_bone(f'DEF-{n}', h, tail, def_of(rig, parent), roll_to=FWD)
        dfm.use_deform = True
    for child, parent in rig['reparent'].items():
        if child in eb and parent in eb:
            eb[child].parent = eb[parent]


def category(rig, name):
    """Bone collection for the face rig's bones (None: not one of them)."""
    b = rig['by_name'].get(name)
    if b is not None:
        return COLL_MAIN if b['main'] else COLL_FINE
    if name in rig['show']:
        return COLL_MAIN
    return None


def palette(rig, name):
    b = rig['by_name'].get(name)
    if b is None:
        return PAL['main']
    return PAL['chain'] if b['kind'] == 'chain' else PAL['main'] if b['main'] else PAL['fine']


def sphere(c, r, n=16):
    pts, edges = [], []
    for axes in ((Vector((1, 0, 0)), Vector((0, 1, 0))), (Vector((0, 1, 0)), Vector((0, 0, 1))),
                 (Vector((1, 0, 0)), Vector((0, 0, 1)))):
        k = len(pts)
        pts += [c + (axes[0] * math.cos(2 * math.pi * i / n) + axes[1] * math.sin(2 * math.pi * i / n)) * r
                for i in range(n)]
        edges += [(k + i, k + (i + 1) % n) for i in range(n)]
    return pts, edges


def ring(c, normal, r, n=24):
    normal = normal.normalized()
    a = normal.orthogonal().normalized()
    b = normal.cross(a)
    pts = [c + (a * math.cos(2 * math.pi * i / n) + b * math.sin(2 * math.pi * i / n)) * r for i in range(n)]
    return pts, [(i, (i + 1) % n) for i in range(n)]


def pose(rig, arm, head_ob, widget, scale):
    """Pose mode: constraints, follow drivers, widgets, colours; then the skin weights."""
    pb = arm.pose.bones
    key = head_ob.data.shape_keys
    s = rig.get('scale', 1.0)
    for b in rig['bones']:
        n = b['name']
        h, t = Vector(b['head']), Vector(b['tail'])
        if b['kind'] == 'chain':
            widget(n, ring(h.lerp(t, 0.5), t - h, b['size'] * s), 2)
            continue
        r = b['size'] * s * (0.5 if b['main'] else 0.4)
        widget(n, sphere(h, r), 2.5 if b['main'] else 1.6)
        if b.get('reach'):
            # a tweak pushed further sinks the skin into what lies under it (Rat King's brows into his eyes)
            lim = pb[n].constraints.new('LIMIT_DISTANCE')
            lim.name = 'Reach'
            lim.target, lim.subtarget = arm, pb[n].parent.name
            lim.distance, lim.limit_mode = b['reach'], 'LIMITDIST_INSIDE'
        if b.get('down'):
            # the control's Y is up: only its downward travel is capped (Baba's brows into her glasses)
            lim = pb[n].constraints.new('LIMIT_LOCATION')
            lim.name = 'Down'
            lim.owner_space = 'LOCAL'
            lim.use_min_y, lim.min_y = True, -b['down']
            lim.use_transform_limit = True
        c = pb[f'DEF-{n}'].constraints.new('COPY_TRANSFORMS')
        c.name = 'Tweak'
        c.target, c.subtarget = arm, n
        c.owner_space = c.target_space = 'LOCAL'
        for bone, terms in rig['follow'][n]:
            for i in range(3):
                d = pb[bone].driver_add('location', i).driver
                d.type = 'SCRIPTED'
                parts = []
                for j, (k, off) in enumerate(terms):
                    if abs(off[i]) < 1e-5:
                        continue
                    v = d.variables.new()
                    v.name = f'k{j}'
                    v.type = 'SINGLE_PROP'
                    v.targets[0].id_type = 'KEY'
                    v.targets[0].id = key
                    v.targets[0].data_path = f'key_blocks["{k}"].value'
                    parts.append(f'k{j}*{off[i]:.5f}')
                d.expression = ('+'.join(parts) or '0').replace('+-', '-')
                assert len(d.expression) <= EXPR_MAX and d.is_simple_expression, d.expression
    weights(rig, head_ob)
    # for test_rig.py: what each face bone is and the vertex its control rides
    arm['face_bones'] = {b['name']: {'kind': b['kind'], 'main': b['main'], 'anchor': b.get('anchor', -1)}
                         for b in rig['bones']}


def weights(rig, head_ob):
    W = rig['weights']
    n = len(head_ob.data.vertices)
    for name, w in W.items():
        # generated parts after the head's vertices (handkeys.py's tongue) keep the skinning build.py copied to them
        assert len(w) <= n, (name, len(w), n)
        g = head_ob.vertex_groups.get(def_of(rig, name)) or head_ob.vertex_groups.new(name=def_of(rig, name))
        on = np.where(w > 1e-5)[0]
        off = np.where(w <= 1e-5)[0]
        if len(off):
            g.remove(off.tolist())
        for i in on:
            g.add([int(i)], float(w[i]), 'REPLACE')
