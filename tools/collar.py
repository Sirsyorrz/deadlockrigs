"""Separate head meshes (Violet) reach down into the collar skinned almost fully to neck_0, while the collar
itself mostly follows spine_3. Turning the neck swung the hidden stump out through the collar, and the skin
above it pulled hard. Head vertices covered by a torso mesh take that mesh's weights instead, blending back to
their own above the collar."""
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

TORSO = ('spine_', 'neck_', 'clavicle_', 'head')


def _weights(ob):
    names = [g.name for g in ob.vertex_groups]
    return [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in ob.data.vertices]


def _torso(w):
    """The weights if they are a torso piece (some spine, nothing but torso bones), else None."""
    if sum(x for b, x in w.items() if b.startswith('spine_')) < 0.2 or not all(b.startswith(TORSO) for b in w):
        return None
    return w


def fix_collar(head, others, scale, n_head):
    """head: the head mesh; others: visible skinned meshes; n_head: head vertices before the tongue was added.
    Returns the number of covered vertices."""
    covers = []
    for o in others:
        me = o.data
        mw = o.matrix_world
        co = [mw @ v.co for v in me.vertices]
        covers.append((BVHTree.FromPolygons(co, [p.vertices for p in me.polygons]), me, co, _weights(o)))
    hw = _weights(head)
    mw = head.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    reach = 0.04 * scale
    pos = [mw @ v.co for v in head.data.vertices[:n_head]]
    src = {}
    for i, v in enumerate(head.data.vertices[:n_head]):
        if not any(b.startswith('neck_') for b in hw[i]):
            continue
        n = (nm @ v.normal).normalized()
        p = mw @ v.co
        best = None
        for tree, me, co, ws in covers:
            hit, _normal, face, dist = tree.ray_cast(p + n * 0.0005, n, reach)
            if hit is None or (best and dist >= best[0]):
                continue
            corner = min(me.polygons[face].vertices, key=lambda k: (co[k] - hit).length)
            w = _torso(ws[corner])
            if w:
                best = (dist, w)
        if best:
            src[i] = best[1]
    if not src:
        return 0
    tree = KDTree(len(src))
    for i in src:
        tree.insert(pos[i], i)
    tree.balance()
    blend = 0.03 * scale
    groups = {g.name: g for g in head.vertex_groups}
    for i in range(n_head):
        _co, k, d = tree.find(pos[i])
        if d >= blend:
            continue
        f = 1 - d / blend
        f = f * f * (3 - 2 * f)
        s = src[k]
        mix = {b: (1 - f) * hw[i].get(b, 0) + f * s.get(b, 0) for b in set(hw[i]) | set(s)}
        total = sum(mix.values())
        for b, x in mix.items():
            g = groups.get(b) or head.vertex_groups.new(name=b)
            groups[b] = g
            if x / total > 1e-4:
                g.add([i], x / total, 'REPLACE')
            else:
                g.remove([i])
    return len(src)
