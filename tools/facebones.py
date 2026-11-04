"""The bone face: deform bones on the skin of a hand-made face (faces/<hero>.json "rig"), on top of its shape keys.
handkeys.py calls build(); rig.py reads the result (data/facerig.json, data/facerig.npz).

Chains bend whole parts (a snout, whiskers); tweaks are small controls on the skin, mains carrying a group of
secondaries. Each bone takes its share of the skin from the game bone it hangs under (its root: head, jaw_0) and the
face bones already under that root, so at rest the skin is where it was and every vertex keeps its total weight.
"""
import numpy as np

from keys import smoothstep

SIDE_OF = {'L': 1, 'R': -1}


def sub(text, side, **kw):
    out = text.replace('{s}', side or '').replace('{S}', {'L': 'Left', 'R': 'Right'}.get(side, ''))
    for k, v in kw.items():
        out = out.replace('{' + k + '}', str(v))
    return out


def sided(name):
    return ('L', 'R') if '{s}' in name else (None,)


class FaceBones:
    def __init__(self, face, spec, arrays):
        self.f, self.spec = face, spec
        self.h = face.h
        self.bones = {}     # name -> dict
        self.order = []
        self.game = {k[2:]: arrays[k].astype(np.float64) for k in arrays if k.startswith('w_')}
        self.n = len(self.h.co)

    # geometry helpers

    def snap(self, q, on):
        """Nearest welded vertex to q among on, and its position."""
        idx = np.where(on)[0]
        j = idx[np.argmin(np.linalg.norm(self.f.p[idx] - q, axis=1))]
        return j, self.f.p[j]

    def orig(self, j):
        return int(np.where(self.h.weld == j)[0][0])

    def field(self, r, side):
        return self.f.region(r, side, side == 'R')[self.h.weld]

    def add(self, name, **kw):
        assert name not in self.bones, name
        self.bones[name] = dict(name=name, **kw)
        self.order.append(name)

    def root(self, name):
        while name in self.bones:
            name = self.bones[name]['parent']
        return name

    # chains

    def chain(self, tmpl, c):
        for side in sided(tmpl):
            mirror = side == 'R'
            pts = [self.f.point(p, side, mirror) for p in c['points']]
            parent = sub(c['parent'], side)
            mask = self.field(c['mask'], side) if 'mask' in c else np.ones(self.n)
            # u: arc length along the polyline at each vertex's nearest point on it
            P = self.h.co
            seg = np.array([np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])])
            start = np.r_[0, np.cumsum(seg)]
            best, u = np.full(self.n, np.inf), np.zeros(self.n)
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                d = b - a
                t = ((P - a) @ d) / (d @ d)
                tc = np.clip(t, 0 if i else -np.inf, 1 if i < len(seg) - 1 else np.inf)
                dist = np.linalg.norm(P - (a + np.outer(tc, d)), axis=1)
                better = dist < best
                best[better] = dist[better]
                u[better] = start[i] + tc[better] * seg[i]
            names = [sub(tmpl, side, i=i) for i in range(len(seg))]
            self.add_chain(names, pts, u * 1000, start * 1000, c.get('blend', 40), mask, parent, c)

    def add_chain(self, names, pts, u, joints, blend, mask, parent, c, fade_in=True):
        """Bones from joint to joint, each weighted where u (mm along the chain) is between its joints, blending over
        blend mm at each joint; the first fades in from the parent at joint 0 (with fade_in)."""
        fields = []
        for i in range(len(names)):
            w = smoothstep(joints[i] - blend / 2, joints[i] + blend / 2, u) if i or fade_in else np.ones(len(u))
            if i + 1 < len(names):
                w = w * (1 - smoothstep(joints[i + 1] - blend / 2, joints[i + 1] + blend / 2, u))
            fields.append(w * mask)
        prev = parent
        for i, n in enumerate(names):
            self.add(n, kind='chain', head=pts[i], tail=pts[i + 1], parent=prev, field=fields[i],
                     batch=names[0], size=c.get('size', 10), main=True)
            prev = n

    def pieces(self, tmpl, c):
        """A chain along each separate piece in a region (whiskers): from the end nearest "root" to the far end."""
        for side in sided(tmpl):
            mirror = side == 'R'
            on = self.f.region(c['pieces'], side, mirror) > 0.5
            comps = np.unique(self.f.comp[on])
            near = self.f.point(c['root'], side, mirror)
            found = []
            for comp in comps:
                m = self.f.comp == comp
                idx = np.where(m)[0]
                if side and np.sign(self.f.p[idx, 0].mean()) != SIDE_OF[side]:
                    continue
                r = idx[np.argmin(np.linalg.norm(self.f.p[idx] - near, axis=1))]
                # straight-line distance from the root: along a thin strip the shortest path zigzags across it
                d = np.full(len(self.f.p), np.inf)
                d[idx] = np.linalg.norm(self.f.p[idx] - self.f.p[r], axis=1)
                found.append((self.f.p[r], idx, d))
            # top to bottom, so both sides number theirs alike
            found.sort(key=lambda t: -t[0][2])
            nb = c.get('bones', 3)
            for k, (rp, idx, d) in enumerate(found):
                length = d[idx].max()
                joints = np.linspace(0, length, nb + 1)
                pts = []
                for t in joints:
                    j = idx[np.argmin(np.abs(d[idx] - t))]
                    pts.append(self.f.p[j])
                u = np.where(np.isfinite(d), d, -1)[self.h.weld] * 1000
                mask = np.isfinite(d)[self.h.weld].astype(float)
                cands = [sub(p, side) for p in c['parent']]
                parent = min(cands, key=lambda b: self.dist_to_bone(b, rp))
                names = [sub(tmpl, side, n=k, i=i) for i in range(nb)]
                self.add_chain(names, pts, u, joints * 1000, c.get('blend', 10), mask, parent, c, fade_in=False)

    def dist_to_bone(self, name, q):
        if name in self.bones:
            a, b = self.bones[name]['head'], self.bones[name]['tail']
        else:
            a = b = self.h.bones[name]
        d = b - a
        t = np.clip(((q - a) @ d) / max(d @ d, 1e-12), 0, 1)
        return np.linalg.norm(q - (a + d * t))

    # tweaks

    def tweak(self, tmpl, c):
        for side in sided(tmpl):
            mirror = side == 'R'
            mask = self.f.region(c['mask'], side, mirror) if 'mask' in c else self.f.skin.astype(float)
            q = self.f.point(c['at'], side, mirror)
            j, p = self.snap(q, mask > 0.5) if c.get('snap', True) else (None, q)
            if 'field' in c:
                w = self.f.region(c['field'], side, mirror)
            else:
                r = c.get('r', 20)
                geo = {'geo': c['at'], 'r': r, 'soft': c.get('soft', r * 2)}
                if 'mask' in c:
                    geo['on'] = c['mask']
                w = self.f.region(geo, side, mirror)
            w = w * mask
            name = sub(tmpl, side)
            parent = sub(c['parent'], side)
            self.add(name, kind='tweak', head=p, tail=p + np.array([0, 0, c.get('size', 8) / 1000]),
                     parent=parent, field=w[self.h.weld], batch=parent, size=c.get('size', 8),
                     main=c.get('main', False), anchor=self.orig(j) if j is not None else -1, side=side,
                     reach=c.get('reach'))

    # weights

    def weights(self):
        """Every face bone's weights, and the game bones' left over: split off the root's family batch by batch,
        parents before children."""
        W = {k: v.copy() for k, v in self.game.items()}
        done = set()
        batches = []
        for n in self.order:
            b = self.bones[n]['batch']
            if b not in batches:
                batches.append(b)
        for batch in batches:
            members = [n for n in self.order if self.bones[n]['batch'] == batch]
            root = self.root(members[0])
            family = [root] + [n for n in done if self.root(n) == root]
            f = np.array([self.bones[n]['field'] for n in members])
            total = f.sum(0)
            f = f / np.maximum(total, 1)[None]
            take = f.sum(0)
            have = sum(W.get(s, 0) for s in family) * np.ones(self.n)
            for n, fn in zip(members, f):
                W[n] = have * fn
            for s in family:
                if s in W:
                    W[s] = W[s] * (1 - take)
            done.update(members)
        return W

    def build(self):
        rig = self.spec['rig']
        chains = rig.get('chains', {})
        for tmpl, c in chains.items():
            if 'pieces' not in c:
                self.chain(tmpl, c)
        for tmpl, c in rig.get('tweaks', {}).items():
            self.tweak(tmpl, c)
        # piece chains (whiskers) last: they can hang off tweaks
        for tmpl, c in chains.items():
            if 'pieces' in c:
                self.pieces(tmpl, c)
        W = self.weights()
        if 'rigid' in rig:
            # separate pieces (teeth) take their mean weights and move whole instead of bending with the gum
            on = (self.f.region(rig['rigid']) > 0.5)[self.h.weld]
            comp = self.f.comp[self.h.weld]
            for k in np.unique(comp[on]):
                m = comp == k
                for w in W.values():
                    if np.ndim(w):
                        w[m] = w[m].mean()
        origin = self.f.origin
        out = {'bones': [], 'reparent': {}, 'show': [], 'scale': rig.get('scale', 1.0)}
        for n in self.order:
            b = self.bones[n]
            out['bones'].append({
                'name': n, 'kind': b['kind'], 'parent': b['parent'], 'main': b['main'],
                'head': [round(float(x), 5) for x in b['head']], 'tail': [round(float(x), 5) for x in b['tail']],
                'size': b['size'] / 1000, 'anchor': b.get('anchor', -1)})
            if b.get('reach'):
                out['bones'][-1]['reach'] = b['reach'] / 1000
        for k, v in rig.get('reparent', {}).items():
            for side in sided(k):
                out['reparent'][sub(k, side)] = sub(v, side)
        for k in rig.get('show', []):
            out['show'] += [sub(k, side) for side in sided(k)]
        arrays = {}
        for k, w in W.items():
            if k in self.game and np.allclose(w, self.game[k]):
                continue
            arrays[k] = w.astype(np.float32)
        tot = sum(W.values())
        print('face bones', len(self.order), 'weight sum drift', float(np.abs(tot - sum(self.game.values())).max()),
              'origin', np.round(origin, 3))
        return out, arrays
