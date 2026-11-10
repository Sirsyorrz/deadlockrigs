"""python handkeys.py <hero> [--regions] : ARKit keys from a hand-made face spec, faces/<hero>.json
(data/head.npz from dump.py) -> data/keys.npz, data/face.json; --regions also writes data/regions.npz (every region's
weights, for face_preview.py).

For heads the generator (keys.py) can't read: snouts, beaks, masks, robots. The spec places everything by hand; see
faces/README.md for the format.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(__file__))
import facegeo as fg  # noqa: E402
from facebones import FaceBones  # noqa: E402
from hero import ROOT, Hero  # noqa: E402
from keys import fib_sphere, harmonic, ray_cast, rot_axis, smoothstep, sphere_fit  # noqa: E402

SIDES = {'L': 'Left', 'R': 'Right'}


def sub(text, side):
    return text.replace('{s}', side).replace('{S}', SIDES[side]) if side else text


def mirror_point(v):
    return np.array([-v[0], v[1], v[2]])


def mirror_axis(v):
    # a turn's axis is a pseudo-vector: its mirror image turns about (x, -y, -z)
    return np.array([v[0], -v[1], -v[2]])


class Face:
    def __init__(self, h: fg.Head, spec: dict, arrays: dict):
        self.h, self.spec = h, spec
        self.p = h.wco
        self.nw = len(self.p)
        self.origin = h.bones['head']
        _, comp = connected_components(h.wgraph, directed=False)
        self.comp = comp
        balls = np.zeros(self.nw, bool)
        self.ball = {}
        for s in 'LR':
            if h.ball is not None and h.ball[s].any():
                m = np.zeros(self.nw, bool)
                m[h.weld[h.ball[s]]] = True
                self.ball[s] = m
                balls |= m
        self.balls = balls
        sizes = np.bincount(comp).astype(float)
        sizes[np.unique(comp[balls])] = 0
        self.skin = comp == sizes.argmax()
        self.weights = {}
        for k, v in arrays.items():
            if k.startswith('w_'):
                w = np.zeros(self.nw)
                np.maximum.at(w, h.weld, v)
                self.weights[k[2:]] = w
        self.mats = {}
        for i, name in enumerate(h.matnames):
            m = np.zeros(self.nw, bool)
            m[h.weld[h.vmat == i]] = True
            self.mats[str(name)] = m
        self.eye_c = {}
        for s in self.ball:
            q = h.wco[self.ball[s]]
            self.eye_c[s] = sphere_fit(q)
        self.cache = {}
        self.lids = {}
        a = h.wgraph.copy()
        a.data[:] = 1
        self.adj = a
        self.deg = np.asarray(a.sum(1)).ravel()

    # lids: each eye's opening, measured by casting rays out of the eyeball past the skin

    def lid_frame(self, s):
        """(gaze g, lid axis m, up u): +deg about m closes the upper lid (u turns towards g)."""
        if s in self.lids:
            return self.lids[s]
        cfg = self.spec.get('eyes', {})
        g = np.array(cfg.get('gaze', [0, -1, 0]), float)
        m = np.array(cfg.get('lid_axis', [1, 0, 0]), float)
        if s == 'R':
            g, m = mirror_point(g), mirror_point(m)
        m /= np.linalg.norm(m)
        g = g - (g @ m) * m
        g /= np.linalg.norm(g)
        u = np.cross(g, m)
        if u[2] < 0:
            m, u = -m, -u
        c, r = self.eye_c[s]
        dirs = fib_sphere(12000)
        dirs = dirs[dirs @ g > np.cos(np.radians(85))]
        tris = self.h.wco[self.h.weld[self.h.tris]]
        tris = tris[self.skin[self.h.weld[self.h.tris[:, 0]]]]
        t, _ = ray_cast(c + dirs * r * 1.02, dirs, tris, 0.5)
        od = dirs[~np.isfinite(t)]
        e = np.degrees(np.arcsin(np.clip(od @ m, -1, 1)))
        phi = np.degrees(np.arctan2(od @ u, od @ g))
        bins = np.arange(np.floor(e.min()), np.ceil(e.max()) + 3, 3.0)
        mids, hi, lo = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            k = (e >= a) & (e < b)
            if k.sum() >= 2:
                mids.append((a + b) / 2)
                hi.append(phi[k].max())
                lo.append(phi[k].min())
        prof = np.array(mids), np.array(hi), np.array(lo)
        print(f'eye {s}: opening {prof[0].min():.0f}..{prof[0].max():.0f} deg along the lid axis, '
              f'widest {np.max(prof[1] - prof[2]):.0f} deg')
        self.lids[s] = (g, m, u, prof)
        return self.lids[s]

    def lid_point(self, s, t, part, out=1.0):
        """A point on a lid's margin: t 0 at the inner corner, 1 at the outer; out scales the eye's radius."""
        g, m, u, (mids, hi, lo) = self.lid_frame(s)
        c, r = self.eye_c[s]
        # t runs from the corner nearer the middle of the face
        e0, e1 = (mids.min(), mids.max()) if abs((c + m * r)[0]) > abs((c - m * r)[0]) else (mids.max(), mids.min())
        e = e0 + t * (e1 - e0)
        phi = np.interp(e, mids, hi if part == 'upper' else lo)
        er, pr = np.radians(e), np.radians(phi)
        d = m * np.sin(er) + (g * np.cos(pr) + u * np.sin(pr)) * np.cos(er)
        return c + d * r * out

    def lid_angles(self, s, pts):
        """Per point: (angle along the lid axis, elevation over the opening's middle line, gap there), degrees."""
        g, m, u, (mids, hi, lo) = self.lid_frame(s)
        c = self.eye_c[s][0]
        d = pts - c
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        e = np.degrees(np.arcsin(np.clip(d @ m, -1, 1)))
        phi = np.degrees(np.arctan2(d @ u, d @ g))
        mid_line = np.interp(e, mids, (hi + lo) / 2)
        # past the corners the gap closes over a few degrees
        fade = np.interp(e, np.r_[mids[0] - 12, mids, mids[-1] + 12], np.r_[0, np.ones(len(mids)), 0])
        # the 3 degree bins undersample the opening's narrow ends, so each takes the wider of its neighbours there
        # (a blink otherwise leaves a sliver of eye at the corners)
        width = hi - lo
        wide = np.maximum(width, np.maximum(np.r_[width[1:], 0], np.r_[0, width[:-1]]))
        ends = np.minimum(np.arange(len(width)), np.arange(len(width))[::-1]) < 2
        width = np.where(ends, wide, width)
        gap = np.interp(e, mids, width) * fade
        return e, phi - mid_line, gap

    # positions: mm from the head bone in Blender axes, or a name

    def point(self, v, side, mirror):
        if isinstance(v, dict) and 'lid' in v:
            return self.lid_point(sub(v['lid'], side), v.get('t', 0.5), v.get('part', 'upper'), v.get('out', 1.0))
        if isinstance(v, str):
            name = sub(v, side)
            if name.startswith('eye_') and name[4:] in self.eye_c:
                return self.eye_c[name[4:]][0]
            if name in self.h.bones:
                return self.h.bones[name]
            raise KeyError(f'unknown point {name!r}')
        q = np.array(v, float)
        return self.origin + (mirror_point(q) if mirror else q) / 1000

    def vec(self, v, mirror, axis=False):
        q = np.array(v, float)
        if not mirror:
            return q
        return mirror_axis(q) if axis else mirror_point(q)

    # regions: weights 0..1 per welded vertex

    def region(self, r, side=None, mirror=False):
        if isinstance(r, str):
            name = sub(r, side)
            key = (name, mirror)
            if key not in self.cache:
                regions = self.spec.get('regions', {})
                if name in regions:
                    self.cache[key] = self.region(regions[name], side, mirror)
                else:
                    tmpl = next((t for t in regions if '{s}' in t and sub(t, side) == name), None)
                    if tmpl is None:
                        raise KeyError(f'unknown region {name!r}')
                    self.cache[key] = self.region(regions[tmpl], side, mirror)
            return self.cache[key]
        p = self.p
        if 'bone' in r:
            names = r['bone'] if isinstance(r['bone'], list) else [r['bone']]
            return np.clip(sum(self.weights.get(sub(b, side), 0) for b in names) * np.ones(self.nw), 0, 1)
        if 'sphere' in r:
            c = self.point(r['sphere'], side, mirror)
            d = np.linalg.norm(p - c, axis=1) * 1000
            return 1 - smoothstep(r['r'], r['r'] + r.get('soft', r['r']), d)
        if 'box' in r:
            # an ellipsoid: sphere scaled per axis
            c = self.point(r['box'], side, mirror)
            d = np.linalg.norm((p - c) * 1000 / np.array(r['size'], float), axis=1)
            return 1 - smoothstep(1, 1 + r.get('soft', 1), d)
        if 'geo' in r:
            c = self.point(r['geo'], side, mirror)
            on = self.region(r['on'], side, mirror) > 0.5 if 'on' in r else self.skin
            seeds = np.where(on)[0]
            seed = seeds[np.argmin(np.linalg.norm(p[seeds] - c, axis=1))]
            reach = (r['r'] + r.get('soft', r['r'])) / 1000
            d = fg.geodesic(self.h, [seed], limit=reach * 1.01) * 1000
            return 1 - smoothstep(r['r'], r['r'] + r.get('soft', r['r']), np.where(np.isfinite(d), d, 1e9))
        if 'half' in r:
            c = self.point(r['half'], side, mirror)
            n = self.vec(r['n'], mirror)
            n = n / np.linalg.norm(n)
            soft = r.get('soft', 2) / 1000
            return smoothstep(-soft / 2, soft / 2, (p - c) @ n)
        if 'lid' in r:
            # above (part upper) or below the middle line of an eye's opening
            s_ = sub(r['lid'], side)
            _, elev, _ = self.lid_angles(s_, p)
            soft = r.get('soft', 4)
            w = smoothstep(-soft / 2, soft / 2, elev)
            return w if r.get('part', 'upper') == 'upper' else 1 - w
        if 'mat' in r:
            names = r['mat'] if isinstance(r['mat'], list) else [r['mat']]
            return np.any([self.mats[m] for m in names], axis=0).astype(float)
        if 'piece' in r:
            return {'skin': self.skin, 'balls': self.balls, 'other': ~self.skin & ~self.balls}[r['piece']].astype(float)
        if 'pieces_near' in r:
            # whole separate pieces (teeth, cards) with any vertex inside the region
            inside = self.region(r['pieces_near'], side, mirror) > r.get('min', 0.5)
            keep = np.isin(self.comp, np.unique(self.comp[inside & ~self.skin & ~self.balls]))
            return keep.astype(float)
        if 'all' in r:
            w = np.ones(self.nw)
            for q in r['all']:
                w = w * self.region(q, side, mirror)
            return w
        if 'any' in r:
            return np.max([self.region(q, side, mirror) for q in r['any']], axis=0)
        if 'not' in r:
            return 1 - self.region(r['not'], side, mirror)
        if 'times' in r:
            return self.region(r['of'], side, mirror) * r['times']
        if 'whole' in r:
            w = self.region(r['whole'], side, mirror)
            mean = np.bincount(self.comp, w) / np.maximum(np.bincount(self.comp), 1)
            # the skin stays per vertex; only the separate pieces (teeth) take one weight each
            return np.where(self.skin, w, mean[self.comp])
        if 'ramp' in r:
            lo, hi = r['ramp']
            return smoothstep(lo, hi, self.region(r['of'], side, mirror))
        if 'smooth' in r:
            w = self.region(r['of'], side, mirror).copy()
            for _ in range(r['smooth']):
                w = 0.5 * w + 0.5 * (self.adj @ w) / np.maximum(self.deg, 1)
            return w
        if 'harmonic' in r:
            one = self.region(r['harmonic']['one'], side, mirror) > 0.5
            zero = (self.region(r['harmonic']['zero'], side, mirror) > 0.5) & ~one
            on = self.skin if r.get('skin_only', True) else np.ones(self.nw, bool)
            w = np.zeros(self.nw)
            w[on] = harmonic(self.h, one, zero | ~on)[on]
            return w
        raise ValueError(f'bad region {r!r}')

    # ops: offsets (metres) per welded vertex, at amount t of the key

    def op(self, o, side, mirror, t=1.0):
        p = self.p
        out = np.zeros((self.nw, 3))
        if 'key' in o:
            return self.key(sub(o['key'], side), t) * o.get('times', 1.0)
        kind = next(k for k in ('move', 'turn', 'scale', 'push', 'lid') if k in o)
        w = self.region(o[kind], side, mirror)
        if not o.get('balls'):
            w = w * ~self.balls
        if 'mask' in o:
            w = w * self.region(o['mask'], side, mirror)
        if kind == 'move':
            out = np.outer(w, self.vec(o['by'], mirror)) / 1000 * t
        elif kind == 'turn':
            c = self.point(o['about'], side, mirror)
            if isinstance(o['axis'], str):
                axis = self.lid_frame(sub(o['axis'], side)[4:])[1]
            else:
                axis = self.vec(o['axis'], mirror, axis=True)
            out = rot_axis(p, c, axis, np.radians(o['deg']) * t * w) - p
        elif kind == 'scale':
            c = self.point(o['about'], side, mirror)
            f = np.ones(3) * np.array(o['by'], float)
            out = (p - c) * ((f - 1) * t)[None] * w[:, None]
        elif kind == 'lid':
            # turn about the eye by a fraction of the gap where each vertex sits: close 1 shuts the opening
            s_ = sub(o['eye'], side)
            _, m, _, _ = self.lid_frame(s_)
            _, _, gap = self.lid_angles(s_, p)
            sign = 1 if o.get('part', 'upper') == 'upper' else -1
            c = self.eye_c[s_][0]
            out = rot_axis(p, c, m, np.radians(gap * o['close'] * sign) * t * w) - p
        elif kind == 'push':
            out = self.h.wnrm * (w * o['by'] / 1000 * t)[:, None]
        return out

    def key(self, name, t=1.0):
        """A key's offsets: the spec's ops for it (sided keys come from their {S} template)."""
        keys = self.spec['keys']
        if name in keys:
            ops, side = keys[name], None
        else:
            tmpl = next((k for k in keys if '{S}' in k and any(sub(k, s) == name for s in 'LR')), None)
            if tmpl is None:
                raise KeyError(f'unknown key {name!r}')
            side = next(s for s in 'LR' if sub(tmpl, s) == name)
            ops = keys[tmpl]
        out = np.zeros((self.nw, 3))
        for o in ops:
            if side:
                out += self.op(o, side, side == 'R', t)
            elif o.get('sym'):
                out += self.op(o, 'L', False, t) + self.op(o, 'R', True, t)
            else:
                out += self.op(o, None, False, t)
        return out

    def names(self):
        out = []
        for k in self.spec['keys']:
            out += [sub(k, s) for s in 'LR'] if '{S}' in k else [k]
        return out

    # eyes

    def eye_keys(self, cfg):
        out = {}
        for s, sx in (('L', 1), ('R', -1)):
            if s not in self.ball:
                continue
            ball = self.ball[s]
            c, r = self.eye_c[s]
            q = self.p[ball]
            g = np.array(cfg.get('gaze', [0, -1, 0]), float)
            g = (mirror_point(g) if s == 'R' else g) / np.linalg.norm(g)
            k = np.cross(g, [0, 0, 1.0])
            k /= np.linalg.norm(k)

            def turn(pts, pitch, yaw):
                n = np.ones(len(pts))
                pts = rot_axis(pts, c, k, n * np.radians(pitch))
                return rot_axis(pts, c, np.array([0, 0, 1.0]), n * np.radians(yaw))

            looks = {'eyeLookUp': (cfg.get('up', 15), 0), 'eyeLookDown': (-cfg.get('down', 15), 0),
                     'eyeLookOut': (0, sx * cfg.get('out', 22)), 'eyeLookIn': (0, -sx * cfg.get('in', 22))}
            name = SIDES[s]
            for key, (pitch, yaw) in looks.items():
                d = np.zeros((self.nw, 3))
                d[ball] = turn(q, pitch, yaw) - q
                out[key + name] = d
                mid = np.zeros((self.nw, 3))
                mid[ball] = turn(q, pitch / 2, yaw / 2) - q - d[ball] / 2
                out[key + name + '_mid'] = mid
            for up in ('eyeLookUp', 'eyeLookDown'):
                for across in ('eyeLookOut', 'eyeLookIn'):
                    d = np.zeros((self.nw, 3))
                    d[ball] = (turn(q, looks[up][0], looks[across][1]) - q
                               - out[up + name][ball] - out[across + name][ball])
                    out[f'{up}{name}_{across}{name}'] = d
            if cfg.get('iris', True):
                out.update(self.iris(s, g, c))
        return out

    def iris(self, s, pole, c, inner=20, outer=40):
        ball = self.ball[s]
        d = self.p[ball] - c
        rho = np.linalg.norm(d, axis=1)
        u = d / rho[:, None]
        alpha = np.arccos(np.clip(u @ pole, -1, 1))
        perp = u - np.outer(u @ pole, pole)
        perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
        fade = 1 - smoothstep(np.radians(inner), np.radians(outer), alpha)
        out = {}
        for key, k in (('irisLarge', 1.3), ('irisSmall', 0.75)):
            a2 = alpha * (1 + (k - 1) * fade)
            moved = c + rho[:, None] * (np.outer(np.cos(a2), pole) + np.sin(a2)[:, None] * perp)
            off = np.zeros((self.nw, 3))
            off[ball] = moved - self.p[ball]
            out[key + SIDES[s]] = off
        return out

    def build(self):
        out = {}
        for name in self.names():
            out[name] = self.key(name)
        for name in self.spec.get('mid', []):
            for n in ([sub(name, s) for s in 'LR'] if '{S}' in name else [name]):
                out[n + '_mid'] = self.key(n, 0.5) - 0.5 * out[n]
        if 'hold' in self.spec:
            keep = 1 - self.region(self.spec['hold'])
            for k in out:
                out[k] *= keep[:, None]
        for k, v in self.eye_keys(self.spec.get('eyes', {})).items():
            out[k] = out[k] + v if k in out else v
        return {k: v[self.h.weld].astype(np.float32) for k, v in out.items()}

    def bag(self, cfg):
        """The mouth bag for build.py's dark interior: skin in cfg's region hidden from the front by skin, 1 when the
        skin in front is over "inner" mm away, 2 over "deep" mm (build.py darkens faces reaching 2)."""
        where = np.where((self.region(cfg['region']) > 0.5) & self.skin)[0]
        tris = self.h.wco[self.h.weld[self.h.tris]]
        tris = tris[self.skin[self.h.weld[self.h.tris[:, 0]]]]
        fwd = np.tile([0.0, -1.0, 0.0], (len(where), 1))
        t, _ = ray_cast(self.p[where], fwd, tris, 0.06)
        out = np.zeros(self.nw, np.int8)
        out[where] = np.where(t > cfg.get('deep', 7) / 1000, 2, np.where(t > cfg.get('inner', 3) / 1000, 1, 0))
        out[where[~np.isfinite(t)]] = 0
        return out

    def controls(self):
        c = self.spec['controls']
        lift = c.get('lift', 6) / 1000

        def at(v):
            q = self.point(v, None, False)
            # on the skin under the point, lifted off it along its normal
            i = np.where(self.skin)[0]
            j = i[np.argmin(np.linalg.norm(self.p[i] - q, axis=1))] if c.get('snap', True) else None
            if j is None:
                return q
            return self.p[j] + self.h.wnrm[j] * lift

        fj = {'landmarks': {k: [round(float(x), 4) for x in at(v)] for k, v in c['landmarks'].items()}}
        for k in ('brow_mid', 'mouth', 'jaw'):
            fj[k] = [round(float(x), 4) for x in at(c[k])]
        if 'eye_target' in c:
            fj['eye_target'] = [round(float(x), 4) for x in self.point(c['eye_target'], None, False)]
            fj['eye_spacing'] = c['eye_spacing'] / 1000
        fj['face_box'] = [round(float(x), 4) for x in self.point(c['face_box'], None, False)]
        fj['scale'] = c.get('scale', 1.0)
        return fj


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    hero = Hero(args[0])
    spec = json.load(open(os.path.join(ROOT, 'faces', hero.slug + '.json')))
    arrays = dict(np.load(os.path.join(hero.data, 'head.npz')))
    h = fg.from_arrays(arrays)
    face = Face(h, spec, arrays)
    keys = face.build()
    np.savez(os.path.join(hero.data, 'keys.npz'), **keys)
    # keys.py's lip split, cut, generated mouth and tongue add vertices these keys don't have
    for f in ('keys_split.npz', 'keys_cut.npy', 'keys_mouth.npz', 'keys_tongue.npz'):
        if os.path.exists(os.path.join(hero.data, f)):
            os.remove(os.path.join(hero.data, f))
    bag_path = os.path.join(hero.data, 'keys_bag.npy')
    if 'bag' in spec:
        np.save(bag_path, face.bag(spec['bag'])[h.weld])
    elif os.path.exists(bag_path):
        os.remove(bag_path)
    json.dump(face.controls(), open(os.path.join(hero.data, 'face.json'), 'w'), indent=1)
    print('hand-made keys', len(keys))
    rig_files = [os.path.join(hero.data, f) for f in ('facerig.json', 'facerig.npz')]
    if 'rig' in spec:
        rig, weights = FaceBones(face, spec, arrays).build()
        json.dump(rig, open(rig_files[0], 'w'), indent=1)
        np.savez_compressed(rig_files[1], **weights)
    else:
        for f in rig_files:
            if os.path.exists(f):
                os.remove(f)
    if '--regions' in sys.argv:
        regs = {}
        for name in spec.get('regions', {}):
            for s in ('L', 'R') if '{s}' in name else (None,):
                regs[sub(name, s)] = face.region(name, s, s == 'R')[h.weld].astype(np.float32)
        np.savez(os.path.join(hero.data, 'regions.npz'), **regs)
        print('regions', list(regs))


if __name__ == '__main__':
    main()
