"""Builds ARKit facial shape keys for a hero's head mesh as per-vertex offsets.

python keys.py head.npz keys.npz      (also writes keys_bag.npy and face.json next to keys.npz)

World axes (Blender, glTF import): +x the hero's left, -y forward, +z up, metres.
The heads have no face bones (only head, an unused jaw_0, eyes), so keys come from:
- lids: rotations about the eyeball sphere centre, upper/lower lid split by geodesic distance to the
  lid margins,
- jaw: rotation about jaw_0, harmonic weights between the lower lip/chin and the upper face,
- everything else: translation-only virtual bones (geodesic Gaussian weights, normalised like skinning
  with a static root),
- lash and liner cards turn rigidly with the lid margin; other cards ride the skin under them.

The face is found from the geometry, so it works across heads with different UV layouts:
- eyeballs: the vertices skinned to eye_0_L/R; centre and radius from a sphere fit,
- eye sockets and the mouth bag: skin from which almost no rays escape (it faces the eyeball or the other side
  of the mouth),
- lid margins: walking out from straight ahead in each direction around an eye, the first skin met on or just
  off the eyeball,
- lip line: the edge of the mouth bag; upper and lower lip by which way the edge faces, corners at its ends.
Sizes scale with the eye spacing; Violet's (88 mm) is the reference the numbers were tuned on.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import facegeo as fg

REF_SPACING = 0.088
# jawOpen's turn and eyeBlink's lid fractions; the face bones' correctives are solved at the same
JAW_DEG = 14
BLINK_UPPER, BLINK_LOWER = 0.8, 0.2


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def rot_x(p, c, ang):
    """Rotate points p about the x axis through c by ang (radians per point, positive lifts the front)."""
    d = p - c
    # forward is -y: a positive angle turns -y towards +z, i.e. a negative turn about +x
    ca, sa = np.cos(-ang), np.sin(-ang)
    y = d[:, 1] * ca - d[:, 2] * sa
    z = d[:, 1] * sa + d[:, 2] * ca
    return np.c_[d[:, 0], y, z] + c


def rot_axis(p, c, axis, ang):
    k = axis / np.linalg.norm(axis)
    d = p - c
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return c + d * ca + np.cross(k, d) * sa + np.outer(d @ k, k) * (1 - ca)


def harmonic(h, one, zero):
    """Solve the edge-weighted graph Laplace equation with w=1 on one, 0 on zero."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    g = h.wgraph.copy()
    g.data = 1 / np.maximum(g.data, 1e-5)
    lap = diags(np.asarray(g.sum(1)).ravel()) - g
    fixed = one | zero
    free = ~fixed
    w = one.astype(float)
    a = lap[free][:, free].tocsc()
    b = -lap[free][:, fixed] @ w[fixed]
    w[free] = spsolve(a, b)
    return np.clip(w, 0, 1)


def sphere_fit(p):
    a = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    c = x[:3]
    return c, np.sqrt(x[3] + c @ c)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    t = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(t), r * np.sin(t), z]


def ray_cast(orig, dirs, tris_co, max_t, min_t=1e-4):
    """Nearest hit (distance, triangle index) of each ray between min_t and max_t (Moller-Trumbore); inf/-1 for
    misses."""
    v0, e1, e2 = tris_co[:, 0], tris_co[:, 1] - tris_co[:, 0], tris_co[:, 2] - tris_co[:, 0]
    best_t = np.full(len(orig), np.inf)
    best_i = np.full(len(orig), -1)
    for i in range(0, len(orig), 64):
        o, d = orig[i:i + 64, None], dirs[i:i + 64, None]
        pv = np.cross(d, e2[None])
        det = (e1[None] * pv).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = o - v0[None]
        u = (tv * pv).sum(-1) * inv
        qv = np.cross(tv, e1[None])
        v = (d * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        t = np.where(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > min_t) & (t < max_t), t, np.inf)
        best_i[i:i + 64] = np.where(np.isfinite(t.min(1)), t.argmin(1), -1)
        best_t[i:i + 64] = t.min(1)
    return best_t, best_i


def ray_hits(orig, dirs, tris_co, max_t):
    return np.isfinite(ray_cast(orig, dirs, tris_co, max_t)[0])


class Rig:
    def __init__(self, h: fg.Head):
        self.h = h
        self.n = len(h.co)
        self.nw = len(h.wco)
        _, comp = connected_components(h.wgraph, directed=False)
        self.comp = comp
        ball_w = {s: np.zeros(self.nw, bool) for s in 'LR'}
        for s in 'LR':
            ball_w[s][h.weld[h.ball[s]]] = True
        self.ball_w = ball_w
        # the deformable skin: the biggest welded piece that is not an eyeball
        sizes = np.bincount(comp, minlength=comp.max() + 1).astype(float)
        sizes[np.unique(comp[ball_w['L'] | ball_w['R']])] = 0
        self.soft_w = comp == sizes.argmax()

        self.eye_c, self.eye_r = {}, {}
        for s in 'LR':
            q = h.co[h.ball[s]]
            c, r = sphere_fit(q)
            # glints and cornea shells sit off the sphere; refit on the main shell
            res = np.abs(np.linalg.norm(q - c, axis=1) - r)
            c, r = sphere_fit(q[res < max(np.percentile(res, 70), 3e-4)])
            self.eye_c[s], self.eye_r[s] = c, r
        self.s = abs(self.eye_c['L'][0] - self.eye_c['R'][0]) / REF_SPACING
        self.mm = 0.001 * 1.1 * self.s  # Crud's sizes are for a pirate head; Violet's is about 10% larger
        self.jaw_pivot = h.bones['jaw_0'].copy()
        self.ez = (self.eye_c['L'][2] + self.eye_c['R'][2]) / 2

        # teeth: their own pieces, by material name (they block rays into the mouth, and the lower row rides the jaw)
        mats = np.array([any(k in str(m).lower() for k in ('teeth', 'tooth')) for m in h.matnames])
        teeth = np.zeros(self.nw, bool)
        teeth[h.weld[mats[h.vmat]]] = True
        # or pieces of the head's own material sitting centred behind the lips (Lady Geist's)
        p = h.wco
        for c in np.unique(self.comp[~self.soft_w & ~self.ball_w['L'] & ~self.ball_w['R']]):
            m = self.comp == c
            q = p[m]
            mid = q.mean(0)
            if (m.sum() > 50 and abs(mid[0]) < 0.01 * self.s and 0.03 * self.s < np.ptp(q[:, 0]) < 0.09 * self.s
                    and self.ez - 0.14 * self.s < mid[2] < self.ez - 0.05 * self.s
                    and mid[1] > p[self.soft_w & (np.hypot(p[:, 0], p[:, 2] - mid[2]) < 0.006 * self.s), 1].min(
                        initial=np.inf) + 0.004 * self.s):
                teeth |= m
        self.teeth_w = teeth & ~self.soft_w
        self._mouth()
        if self.hole is not None or self.fused is not None or self.lip_cut is not None:
            # the caller opens the lips (fused) or builds a mouth bag onto the edge (add_bag) and starts over
            return
        self.face_w = self.soft_w & ~self.bag
        self.handles = {}
        self.lid_specs = {}
        self.sock = {}
        self.margins = {s: self._margin(s) for s in 'LR'}
        self.face_w &= ~(self.sock['L'] | self.sock['R'])
        self._jaw_weights()
        self._lid_weights()
        self.handle_vid = {}
        self.d_lips = fg.geodesic(h, np.r_[self.rim_u, self.rim_l], limit=0.06 * self.s)
        self.corner_x = {side: abs(h.wco[self.corner[side], 0]) for side in 'LR'}

    def landmark(self, x, z, r=0.004):
        """Frontmost face vertex near (x, z)."""
        p = self.h.wco
        d = np.hypot(p[:, 0] - x, p[:, 2] - z)
        # where the face skin is sparse, the nearest vertex within r could be on the back of the head
        front = self.face_w & (p[:, 1] < self.eye_c['L'][1])
        cand = np.where(front & (d < r * self.s))[0]
        if len(cand) == 0:
            cand = np.where(front)[0]
            return cand[d[cand].argmin()]
        return cand[p[cand, 1].argmin()]

    def front_y(self, x, z, r=0.004):
        return self.h.wco[self.landmark(x, z, r), 1]

    def openness(self, verts, centre, reach):
        """Fraction of hemisphere rays (about each vertex normal) that leave the skin and eyeballs within reach of
        centre without hitting them."""
        h = self.h
        p = h.wco
        tri_w = h.weld[h.tris]
        solid = self.soft_w | self.ball_w['L'] | self.ball_w['R'] | self.teeth_w
        near = solid[tri_w].all(1) & (np.linalg.norm(p[tri_w].mean(1) - centre, axis=1) < reach)
        tris = p[tri_w[near]]
        dirs = fib_sphere(48)
        dirs = np.r_[dirs, dirs * [-1, 1, 1]]  # mirrored so both sides of the face get the same answer
        frac = np.zeros(len(verts))
        for i, v in enumerate(verts):
            d = dirs[dirs @ h.wnrm[v] > 0.15]
            hit = ray_hits(np.repeat(p[v][None] + h.wnrm[v] * 1e-4, len(d), 0), d, tris, reach)
            frac[i] = 1 - hit.mean()
        return frac

    def enclosed(self, window, centre, reach, limit=0.08):
        """Skin vertices in window that are mostly closed in (mouth bag, eye socket lining)."""
        ids = np.where(self.soft_w & window)[0]
        out = np.zeros(self.nw, bool)
        out[ids[self.openness(ids, centre, reach) < limit]] = True
        return out

    def fill(self, region):
        """region plus the skin it encloses (anything cut off from the rest of the skin by it)."""
        rest = self.soft_w & ~region
        ids = np.where(rest)[0]
        _, lab = connected_components(self.h.wgraph[rest][:, rest], directed=False)
        out = region.copy()
        out[ids[lab != np.bincount(lab).argmax()]] = True
        return out

    def rim(self, region):
        """Vertices of region with a neighbour on the skin outside it."""
        g = self.h.wgraph.tocoo()
        edge = region[g.row] & ~region[g.col] & self.soft_w[g.col]
        return np.unique(g.row[edge])

    def region_near(self, mask, pick):
        """The connected piece of mask chosen by pick(ids) (a vertex id in mask)."""
        ids = np.where(mask)[0]
        _, lab = connected_components(self.h.wgraph[mask][:, mask], directed=False)
        seed = pick(ids)
        out = np.zeros(self.nw, bool)
        out[ids[lab == lab[np.searchsorted(ids, seed)]]] = True
        return out

    # mouth

    def _mouth(self):
        h, s = self.h, self.s
        p = h.wco
        ez = self.ez
        centre = np.array([0, self.jaw_pivot[1] - 0.06 * s, ez - 0.08 * s])
        win = (np.abs(p[:, 0]) < 0.05 * s) & (np.abs(p[:, 2] - centre[2]) < 0.04 * s) \
            & (p[:, 1] < self.jaw_pivot[1] + 0.05 * s)
        self.fused = None
        self.lip_cut = None
        self.hole = self.mouth_hole(win)
        if self.hole is not None:
            return
        gen = np.where(np.char.equal(h.matnames.astype(str), 'mouth_bag'))[0]
        if len(gen):
            # a bag add_bag built: its own faces, not whatever the closed-in test makes of them
            self.bag = np.zeros(self.nw, bool)
            self.bag[h.weld[np.isin(h.vmat, gen)]] = True
        else:
            closed = self.enclosed(win, centre, 0.12 * s)
            # the bag is the biggest closed-in piece around the middle (nostrils and folds are small)
            ids = np.where(closed)[0]
            _, lab = connected_components(h.wgraph[closed][:, closed], directed=False)
            mid = np.abs(p[ids, 0]) < 0.01 * s
            if not mid.any():
                self.fused = self.lip_strip()
                if self.fused is not None:
                    return
                self.lip_cut = self.lip_line()
                if self.lip_cut is not None:
                    return
                sys.exit('no mouth bag behind the lips and no lip line to open: this head needs another jaw setup')
            size = np.bincount(lab[mid], minlength=lab.max() + 1)
            self.bag = np.zeros(self.nw, bool)
            self.bag[ids[lab == size.argmax()]] = True
            self.bag = self.fill(self.bag)
        rim = self.rim(self.bag)
        # a shallow cup sculpted behind parted lips (Celeste's) has no room for a tongue or an open jaw: drop it and
        # build a full bag on the lips instead, like for a closed mouth
        if p[self.bag, 1].max() - p[rim, 1].min() < 0.03 * s:
            tw = h.weld[h.tris]
            inside = self.bag.copy()
            inside[rim] = False
            self.fused = np.where(self.bag[tw].all(1) & inside[tw].any(1))[0]
            return
        q = p[rim]
        self.corner = {'L': rim[q[:, 0].argmax()], 'R': rim[q[:, 0].argmin()]}
        cl, cr = p[self.corner['L']], p[self.corner['R']]
        # the lips meet face to face, so the upper lip's edge faces down and the lower's up (the lip line
        # arches, so a height split would put the middle of the lower lip with the upper)
        down = h.wnrm[rim, 2] < 0
        if len(gen) or not down.any() or down.all():
            # lips opened from one surface have edges facing every way: above or below the middle of the edge at the
            # same x instead
            near = np.abs(q[:, 0][:, None] - q[:, 0][None]) < 0.002 * s
            mid = np.array([(q[n, 2].max() + q[n, 2].min()) / 2 for n in near])
            down = q[:, 2] >= mid
        self.rim_u, self.rim_l = rim[down], rim[~down]
        if not down.any() or down.all():
            sys.exit('no mouth bag behind the lips (open hole or fused lips?): this head needs another jaw setup')
        self.lip_z = (cl[2] + cr[2]) / 2
        self.lip_y = q[:, 1].min()
        # split the bag level with the lip line (a geodesic split leans towards the shorter lip)
        o = np.argsort(q[:, 0])
        zr = np.interp(p[:, 0], q[o, 0], q[o, 2])
        upper = p[:, 2] >= zr - 0.001 * s
        # except the lips' own inner faces, which belong with their lip
        du = fg.geodesic(h, self.rim_u, limit=0.01 * s)
        dl = fg.geodesic(h, self.rim_l, limit=0.01 * s)
        lip = np.minimum(du, dl) < 0.004 * s
        upper[lip] = du[lip] < dl[lip]
        upper[self.rim_u] = True
        upper[self.rim_l] = False
        self.bag_u = np.where(self.bag & upper)[0]
        self.bag_l = np.where(self.bag & ~upper)[0]

    def lip_strip(self):
        """Unwelded triangle ids joining the lips where they are one surface (The Doorman's sculpted closed mouth):
        a band one triangle thick between the upper lip's underside (facing down) and the lower lip's top (facing
        up), under a few mm tall. None if there is no such band across the middle."""
        h, s, p = self.h, self.s, self.h.wco
        tw = h.weld[h.tris]
        nz = h.wnrm[tw, 2]
        c = p[tw].mean(1)
        strip = ((nz.min(1) < -0.3) & (nz.max(1) > 0.2) & (np.ptp(p[tw, 2], axis=1) < 0.0025 * s)
                 & self.soft_w[tw].all(1) & (np.abs(c[:, 0]) < 0.045 * s)
                 & (c[:, 2] < self.ez - 0.06 * s) & (c[:, 2] > self.ez - 0.13 * s))
        ids = np.where(strip)[0]
        if not len(ids):
            return None
        nt = len(tw)
        e = np.concatenate([np.c_[ids, nt + tw[ids, k]] for k in range(3)])
        _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                                                 shape=(nt + self.nw, nt + self.nw)), directed=False)
        best = None
        for k in np.unique(lab[ids]):
            ts = ids[lab[ids] == k]
            # the lip line crosses the middle and reaches out to both corners
            if (c[ts, 0].min() < -0.015 * s and c[ts, 0].max() > 0.015 * s
                    and (best is None or np.ptp(c[ts, 0]) > np.ptp(c[best, 0]))):
                best = ts
        if best is None:
            return None
        # where the lips pinch together (a vertex both edges share once the band is gone) is the corner: the band
        # past it opened a torn sliver out in the cheek (The Doorman's)
        keep = np.ones(len(tw), bool)
        keep[best] = False
        k = tw[keep]
        e = np.sort(np.concatenate([k[:, [0, 1]], k[:, [1, 2]], k[:, [2, 0]]]), axis=1)
        u, cnt = np.unique(e, axis=0, return_counts=True)
        deg = np.bincount(u[cnt == 1].ravel(), minlength=self.nw)
        band = np.unique(tw[best])
        pinch = band[deg[band] >= 4]
        for sx in (1, -1):
            side = pinch[p[pinch, 0] * sx > 0.01 * s]
            if len(side):
                limit = np.abs(p[side, 0]).min()
                best = best[~((c[best, 0] * sx > 0) & (np.abs(c[best, 0]) > limit))]
        return best

    def lip_line(self, recess=False):
        """For closed lips sculpted as one surface with no band to drop (Grey Talon's, Lady Geist's): the crease
        where they meet, as the cheapest path through the mouth's middle when walking along concave skin is cheap,
        out to where the crease fades at the corners. The chin crease can be deeper (Solomon's), so of the creases
        through the middle the widest wins. Returns (path of welded vertices, a lower-lip vertex) or None.
        A broad groove (Infernus' parted lips) is flat at the bottom; failing a crease, recess also counts how far
        each vertex sits behind the skin around it."""
        h, s, p, nrm = self.h, self.s, self.h.wco, self.h.wnrm
        g = h.wgraph.tocoo()
        d = p[g.col] - p[g.row]
        # curvature along each edge, most concave per vertex (1/m, times s so heads compare)
        k = ((nrm[g.col] - nrm[g.row]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12)
        val = np.full(self.nw, np.inf)
        np.minimum.at(val, g.row, k * s)
        band = (self.soft_w & (np.abs(p[:, 0]) < 0.07 * s) & (p[:, 2] < self.ez - 0.065 * s)
                & (p[:, 2] > self.ez - 0.17 * s) & (nrm[:, 1] < -0.2))
        if recess:
            ids = np.where(band)[0]
            soft_p = p[self.soft_w]
            for v, nb in zip(ids, cKDTree(soft_p).query_ball_point(p[ids], 0.01 * s)):
                val[v] = min(val[v], (p[v] - soft_p[nb].mean(0)) @ nrm[v] / s * 1e5)
        cost = np.exp(np.clip((val[g.row] + val[g.col]) / 2, -800, 200) / 250)
        keep = band[g.row] & band[g.col]
        wg = coo_matrix((np.sqrt((d * d).sum(1))[keep] * cost[keep], (g.row[keep], g.col[keep])),
                        shape=(self.nw, self.nw)).tocsr()
        mids = np.where(band & (np.abs(p[:, 0]) < 0.003 * s) & (val < -150))[0]
        mids = mids[np.argsort(p[mids, 2])]
        # one candidate per crease: the deepest of each run of middle vertices close in height
        runs = np.split(mids, np.where(np.diff(p[mids, 2]) > 0.004 * s)[0] + 1) if len(mids) else []
        best = None
        for run in runs:
            m = run[val[run].argmin()]
            dist, pred = dijkstra(wg, directed=False, indices=m, return_predecessors=True, limit=0.12 * s)
            ends = []
            for sx in (1, -1):
                x = p[:, 0] * sx
                # the crease's end: the farthest point reached while walking mostly in the valley
                ok = np.isfinite(dist) & (x > 0.008 * s) & (dist < 0.6 * np.maximum(x, 1e-9)) & (val < -100)
                if not ok.any():
                    break
                ends.append(np.where(ok)[0][x[ok].argmax()])
            if len(ends) < 2:
                continue
            width = p[ends[0], 0] - p[ends[1], 0]
            if best is None or width > best[0]:
                best = (width, m, ends, pred)
        if best is None or best[0] < 0.04 * s:
            return None if recess else self.lip_line(recess=True)
        _, m, ends, pred = best
        halves = []
        for e in ends:
            seg = [e]
            while seg[-1] != m:
                seg.append(pred[seg[-1]])
            halves.append(seg)
        path = halves[0][:-1] + halves[1][::-1]
        # no detours: where two path vertices further apart share an edge, the faces between them would be cut
        # off on the wrong side (Holliday's)
        nbr = h.wgraph.tocsr()
        i = 0
        while i < len(path) - 2:
            linked = set(nbr.indices[nbr.indptr[path[i]]:nbr.indptr[path[i] + 1]].tolist())
            j = max((j for j in range(i + 2, min(i + 6, len(path))) if path[j] in linked), default=None)
            if j is not None:
                path = path[:i + 1] + path[j:]
            i += 1
        path = np.array(path)
        # the lower lip: the neighbour of the middle with the lowest point
        nb = h.wgraph.tocsr()
        cand = nb.indices[nb.indptr[m]:nb.indptr[m + 1]]
        cand = cand[~np.isin(cand, path)]
        below = cand[p[cand, 2].argmin()]
        print('lip line', len(path), 'verts, x', round(p[ends[1], 0], 4), '..', round(p[ends[0], 0], 4),
              'z', round(p[m, 2], 4))
        return path, below

    def mouth_hole(self, win):
        """The skin's open edge between the lips, ordered around, when the mouth is a hole into the head with no
        bag behind it (Venator's), else None."""
        h, s, p = self.h, self.s, self.h.wco
        if h.lip_loop is not None:
            return h.weld[h.lip_loop]
        tw = h.weld[h.tris]
        e = np.sort(np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]]), axis=1)
        u, cnt = np.unique(e, axis=0, return_counts=True)
        open_e = u[cnt == 1]
        open_e = open_e[(self.soft_w & win)[open_e].all(1)]
        if not len(open_e):
            return None
        _, lab = connected_components(coo_matrix((np.ones(len(open_e)), (open_e[:, 0], open_e[:, 1])),
                                                 shape=(self.nw, self.nw)), directed=False)
        for k in np.unique(lab[open_e[:, 0]]):
            loop_e = open_e[lab[open_e[:, 0]] == k]
            ids = np.unique(loop_e)
            # a mouth edge crosses the middle, spans both sides, and closes on itself
            if len(ids) < 12 or not (np.abs(p[ids, 0]) < 0.004 * s).any() or np.ptp(p[ids, 0]) < 0.03 * s:
                continue
            # corner to corner twice (once along each lip), so stray open edges (a hole cut open with a spare
            # face or two) don't break the loop
            lens = np.linalg.norm(p[loop_e[:, 0]] - p[loop_e[:, 1]], axis=1)
            g = coo_matrix((lens, (loop_e[:, 0], loop_e[:, 1])), shape=(self.nw, self.nw)).tocsr()
            g = g.maximum(g.T)
            a, b = ids[p[ids, 0].argmax()], ids[p[ids, 0].argmin()]
            chains = []
            for _ in range(2):
                d, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
                if not np.isfinite(d[b]):
                    break
                path = [b]
                while path[-1] != a:
                    path.append(pred[path[-1]])
                chains.append(path[::-1])
                # drop the edges used, not the vertices: the lips can pinch together at a vertex both share
                g = g.tolil()
                for i, j in zip(path[:-1], path[1:]):
                    g[i, j] = g[j, i] = 0
                g = g.tocsr()
            if len(chains) < 2:
                continue
            return np.array(chains[0] + chains[1][::-1][1:-1])
        return None

    # lids

    def _margin(self, side):
        h, s = self.h, self.s
        p = h.wco
        c, r = self.eye_c[side], self.eye_r[side]
        sx = 1 if side == 'L' else -1
        win = np.linalg.norm(p - c, axis=1) < 1.8 * r
        closed = self.enclosed(win, c, 0.08 * s) & (np.linalg.norm(p - c, axis=1) < r + 0.004 * s)
        # the socket lining: the closed-in piece wrapped around the eyeball's front
        sock = self.region_near(closed, lambda ids: ids[np.linalg.norm(p[ids] - (c + np.array([0, -r, 0])), axis=1).argmin()])
        self.sock[side] = sock = self.fill(sock)
        # the margin: walking out from straight ahead in each direction around the eye, the first skin met
        # on or just off the eyeball (many heads close the socket with a cup inside the ball; that is skipped)
        tri_w = h.weld[h.tris]
        near = self.soft_w[tri_w].all(1) & (np.linalg.norm(p[tri_w].mean(1) - c, axis=1) < 2 * r)
        tris, tri_ids = p[tri_w[near]], tri_w[near]
        betas = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        alphas = np.radians(np.arange(0, 85, 0.75))
        bb, aa = np.meshgrid(betas, alphas, indexing='ij')
        dirs = np.c_[(np.sin(aa) * np.cos(bb)).ravel() * sx, -np.cos(aa).ravel(), (np.sin(aa) * np.sin(bb)).ravel()]
        t, ti = ray_cast(np.repeat(c[None], len(dirs), 0), dirs, tris, r + 0.005 * s, r - 0.002 * s)
        t, ti = t.reshape(bb.shape), ti.reshape(bb.shape)
        ids, hits = [], []
        for k in range(len(betas)):
            j = np.argmax(np.isfinite(t[k]))
            if not np.isfinite(t[k, j]) or j == 0:
                continue
            hit = c + dirs[k * len(alphas) + j] * t[k, j]
            tv = tri_ids[ti[k, j]]
            ids.append(tv[np.linalg.norm(p[tv] - hit, axis=1).argmin()])
            hits.append(hit)
        ids, hits = np.array(ids), np.array(hits)

        def split(q):
            xs = q[:, 0] * sx
            qi, qo = q[xs.argmin()], q[xs.argmax()]
            line_z = qi[2] + (q[:, 0] - qi[0]) / (qo[0] - qi[0]) * (qo[2] - qi[2])
            return q[:, 2] >= line_z, qi, qo
        up_v, _, _ = split(p[ids])
        upper, lower = np.unique(ids[up_v]), np.unique(ids[~up_v])
        # the lid turn angles come from the hit points, not the vertices: on a thick lid the nearest vertices
        # alternate between its front and back edge, which made the closed lid zigzag
        up_h, pi, po = split(hits)
        phi = lambda q: np.arctan2(q[:, 2] - c[2], -(q[:, 1] - c[1]))

        def chain(q):
            o = np.argsort(q[:, 0])
            return q[o, 0], phi(q[o])
        return {'upper': upper, 'lower': lower, 'up_chain': chain(hits[up_h]), 'lo_chain': chain(hits[~up_h]),
                'x_in': pi[0], 'x_out': po[0], 'phi': phi}

    def margin_phis(self, side, x):
        m = self.margins[side]
        xu, pu = m['up_chain']
        xl, pl = m['lo_chain']
        lo, hi = sorted((m['x_in'], m['x_out']))
        xc = np.clip(x, lo, hi)
        fu = np.interp(xc, xu, pu)
        fl = np.interp(xc, xl, pl)
        # the gap closes at the corners; beyond them it stays closed
        return fu, fl

    def _lid_weights(self):
        h, s = self.h, self.s
        self.lid = {}
        for side in 'LR':
            m = self.margins[side]
            du = fg.geodesic(h, m['upper'], limit=0.04 * s)
            dl = fg.geodesic(h, m['lower'], limit=0.03 * s)
            sock = self.sock[side]
            up = (du < dl) & self.soft_w
            lo = (dl <= du) & self.soft_w
            # wide open eyes pull the upper lid further; spread its falloff so the skin above stretches over
            # more rows instead of shearing a coarse lid into a sawtooth
            gap = np.subtract(*self.margin_phis(side, np.linspace(m['x_in'], m['x_out'], 50))).max()
            reach = 0.0095 * s * max(1.0, gap * self.eye_r[side] / (0.016 * s)) ** 1.5
            wu = np.where(up, 1 - smoothstep(0.0012 * s, reach, du), 0)
            wl = np.where(lo, 1 - smoothstep(0.001 * s, 0.007 * s, dl), 0)
            wu[sock & up] = 1
            wl[sock & lo] = 1
            self.lid[side] = {'wu': wu, 'wl': wl, 'sock': sock, 'du': du, 'dl': dl}

    def lid_angles(self, side, x, upper_frac=0.0, lower_frac=0.0, upper_deg=0.0, lower_deg=0.0):
        """Margin turns (radians) at x for the upper and lower lid. *_frac: fraction of the open gap
        (positive closes), *_deg: extra turn in degrees (positive raises), fading out at the corners."""
        fu, fl = self.margin_phis(side, x)
        gap = np.maximum(fu - fl, 0)
        m = self.margins[side]
        gmax = np.subtract(*self.margin_phis(side, np.linspace(m['x_in'], m['x_out'], 50))).max()
        taper = np.clip(gap / (0.5 * gmax), 0, 1)
        au = -upper_frac * gap + np.radians(upper_deg) * taper
        al = lower_frac * gap + np.radians(lower_deg) * taper
        return au, al

    def lid_move(self, side, key=None, **amounts):
        """Offsets (welded) for lid rotations; see lid_angles. With key, the lid cards follow too."""
        if key:
            self.lid_specs.setdefault(key, []).append((side, amounts))
        p = self.h.wco
        lw = self.lid[side]
        c, r = self.eye_c[side], self.eye_r[side]
        au, al = self.lid_angles(side, p[:, 0], **amounts)
        ang = au * lw['wu'] + al * lw['wl']
        moved = rot_x(p, c, ang)
        # keep moving lid skin outside the eyeball
        skin = (ang != 0) & ~lw['sock']
        d = moved[skin] - c
        rr = np.linalg.norm(d, axis=1)
        r0 = np.linalg.norm(p[skin] - c, axis=1)
        need = np.minimum(r + 0.0007 * self.s, r0 + 0.0015 * self.s)
        grow = np.maximum(need - rr, 0) * np.minimum(np.abs(ang[skin]) / 0.05, 1)
        moved[skin] += d / rr[:, None] * grow[:, None]
        off = moved - p
        # relax the stretch band above the margin: on coarse lids neighbouring rows shear into folds
        band = skin & (lw['du'] > 0.003 * self.s) & (lw['wu'] > 0)
        if band.any():
            adj = self.h.wgraph.copy()
            adj.data[:] = 1
            deg = np.asarray(adj.sum(1)).ravel()
            for _ in range(10):
                off[band] = (adj @ off)[band] / deg[band, None]
        return off

    # jaw

    def _jaw_weights(self):
        h, s = self.h, self.s
        p = h.wco
        du = fg.geodesic(h, self.bag_u, limit=0.2)
        dl = fg.geodesic(h, self.bag_l, limit=0.2)
        reach = np.isfinite(du) & np.isfinite(dl)
        du, dl = np.where(reach, du, 1.0), np.where(reach, dl, 1.0)
        sig = 0.0015 * s + 0.12 * np.minimum(du, dl)
        lab = 1 / (1 + np.exp(-(du - dl) / sig))
        lab[self.bag_u] = 0
        lab[self.bag_l] = 1
        lz, ly, jy = self.lip_z, self.lip_y, self.jaw_pivot[1]
        # Harmonic weights between the parts that surely follow the jaw (lower lip, chin, lower mouth
        # bag) and the parts that surely do not (upper lip and face, neck, behind the jaw), so the skin
        # under the jaw stretches smoothly into the neck instead of folding.
        one = np.zeros(self.nw, bool)
        zero = ~self.soft_w
        one[self.bag_l] = True
        zero[self.bag_u] = True
        corners = np.array([self.corner['L'], self.corner['R']])
        near_corner = np.min(np.linalg.norm(p[:, None] - p[corners][None], axis=2), axis=1) < 0.007 * s
        one |= (lab > 0.6) & (dl < 0.012 * s) & self.soft_w & ~near_corner
        zero |= (lab < 0.4) & (du < 0.012 * s) & ~near_corner
        one |= (p[:, 1] < ly + 0.018 * s) & (p[:, 2] < lz - 0.010 * s) & (lab > 0.5) & self.soft_w
        zero |= (p[:, 2] > lz + 0.025 * s) | (p[:, 2] < lz - 0.105 * s) | (p[:, 1] > jy - 0.006 * s)
        # the mouth bag floor lies a few mm over the skin under the chin: it takes that skin's weight
        # (solved without the bag), else it pokes out under the jaw
        rim_l = self.rim_l
        inner = np.setdiff1d(self.bag_l, rim_l)
        inner = inner[np.min(np.linalg.norm(p[inner][:, None] - p[rim_l][None], axis=2), axis=1) > 0.006 * s]
        one[inner] = False
        zero[inner] = True
        zero &= ~one
        w = harmonic(self.h, one, zero)
        outside = np.where(self.face_w & ~near_corner)[0]
        outside = outside[(p[outside, 2] < lz - 0.015 * s) & (h.wnrm[outside, 2] < -0.3)]
        if len(inner) and len(outside):
            w[inner] = w[outside[cKDTree(p[outside]).query(p[inner])[1]]]
        # each row of teeth moves as one piece: the lower row (below the lip line) with the jaw
        for c in np.unique(self.comp[self.teeth_w]):
            m = self.comp == c
            w[m] = float(p[m, 2].mean() < self.lip_z)
        self.jaw_w = w

    def jaw_rot(self, deg):
        p = self.h.wco
        return rot_x(p, self.jaw_pivot, -np.radians(deg) * self.jaw_w) - p

    # virtual bones

    def handle(self, name, vid, radius):
        radius *= self.s
        g = fg.geodesic(self.h, [vid], limit=radius * 3)
        w = np.exp(-(g / radius) ** 2)
        w[~self.soft_w] = 0
        # on a coarse face (Celeste's brows) the vertex under the handle took the whole move and its neighbours a
        # fraction, a spike that also differed left to right with where the handle landed; dense faces barely change
        a = self.h.wgraph.copy()
        a.data[:] = 1
        deg = np.asarray(a.sum(1)).ravel()
        for _ in range(2):
            w = 0.5 * w + 0.5 * (a @ w) / np.maximum(deg, 1)
            w[~self.soft_w] = 0
        self.handles[name] = w
        self.handle_vid[name] = vid

    def lip_arc(self, side, reach=0.016):
        """The lip line following its corner on a curve: none at the middle, all of it at the corner, fading out
        past it, so the mouth bends instead of kinking at the corner."""
        sx = 1 if side == 'L' else -1
        cx = self.corner_x[side]
        t = self.h.wco[:, 0] * sx / cx
        f = np.where(t <= 1, np.clip(t, 0, 1) ** 1.6, np.exp(-((t - 1) * cx / (0.016 * self.s)) ** 2))
        return f * np.exp(-(self.d_lips / (reach * self.s)) ** 2) * self.soft_w

    def lid_profile(self, side, x):
        """The open gap at x as a fraction of its widest: a lid bone's turn scaled by this closes the lid like
        lid_move's fractions."""
        fu, fl = self.margin_phis(side, x)
        return np.clip(fu - fl, 0, None) / self.gap_max(side)

    def gap_max(self, side):
        m = self.margins[side]
        return np.subtract(*self.margin_phis(side, np.linspace(m['x_in'], m['x_out'], 50))).max()

    def moves(self, spec):
        """spec: {handle: crud-style mm vector (forward, up, towards the hero's right)}"""
        tot = np.zeros(self.nw)
        for w in self.handles.values():
            tot += w
        norm = np.maximum(tot, 1)
        out = np.zeros((self.nw, 3))
        for name, v in spec.items():
            f, u, r = v
            world = np.array([-r, -f, u]) * self.mm
            out += np.outer(self.handles[name] / norm, world)
        return out


def build(h: fg.Head) -> tuple:
    rig = Rig(h)
    p = h.wco
    s = rig.s
    L = rig.landmark
    rim_u, rim_l, corner = rig.rim_u, rig.rim_l, rig.corner
    ez, lz = rig.ez, rig.lip_z
    em = ez - lz  # eye to mouth height; Violet's is 80 mm

    def rim_z(rim, x):
        q = p[rim]
        o = np.argsort(q[:, 0])
        return np.interp(x, q[o, 0], q[o, 2])

    for side, sx in (('L', 1), ('R', -1)):
        cx, cz = p[corner[side], 0], p[corner[side], 2]
        ex = rig.eye_c[side][0]
        rig.handle(f'{side}_LIPCORNER', corner[side], 0.009)
        rig.handle(f'{side}_UPPERLIP_INNER', rim_u[np.argmin(np.abs(p[rim_u, 0] - sx * 0.014 * s))], 0.005)
        rig.handle(f'{side}_LOWERLIP_INNER', rim_l[np.argmin(np.abs(p[rim_l, 0] - sx * 0.014 * s))], 0.005)
        rig.handle(f'{side}_UPPERLIP_OUTER', L(sx * 0.014 * s, rim_z(rim_u, sx * 0.014 * s) + 0.006 * s), 0.006)
        rig.handle(f'{side}_LOWERLIP_OUTER', L(sx * 0.014 * s, rim_z(rim_l, sx * 0.014 * s) - 0.007 * s), 0.006)
        rig.handle(f'{side}_MOUTHFOLD_UPPER', L(cx + sx * 0.001 * s, cz + 0.018 * s), 0.009)
        rig.handle(f'{side}_MOUTHFOLD_LOWER', L(cx + sx * 0.001 * s, cz - 0.02 * s), 0.009)
        rig.handle(f'{side}_CHEEK_MID', L(ex + sx * 0.006 * s, lz + 0.5125 * em), 0.013)
        rig.handle(f'{side}_NOSE_WING', L(sx * 0.014 * s, lz + 0.325 * em, 0.003), 0.006)
        rig.handle(f'{side}_BROW_INNER', L(ex - sx * 0.024 * s, ez + 0.019 * s), 0.009)
        rig.handle(f'{side}_BROW_MID', L(ex + sx * 0.001 * s, ez + 0.031 * s), 0.01)
        rig.handle(f'{side}_BROW_OUTER', L(ex + sx * 0.024 * s, ez + 0.027 * s), 0.01)
    rig.handle('M_UPPERLIP_INNER', rim_u[np.argmin(np.abs(p[rim_u, 0]))], 0.005)
    rig.handle('M_LOWERLIP_INNER', rim_l[np.argmin(np.abs(p[rim_l, 0]))], 0.005)
    rig.handle('M_UPPERLIP_OUTER', L(0, rim_z(rim_u, 0) + 0.006 * s), 0.006)
    rig.handle('M_LOWERLIP_OUTER', L(0, rim_z(rim_l, 0) - 0.007 * s), 0.006)
    rig.handle('M_UNDER_CHIN', L(0, lz - 0.4 * em), 0.012)
    rig.handle('M_NOSE', L(0, lz + 0.4125 * em), 0.008)

    keys = {}

    def mirror(spec):
        return {('R' + k[1:] if k[0] == 'L' else 'L' + k[1:] if k[0] == 'R' else k): (v[0], v[1], -v[2]) for k, v in spec.items()}

    def lr(name, spec_left):
        keys[name + 'Left'] = rig.moves(spec_left)
        keys[name + 'Right'] = rig.moves(mirror(spec_left))

    d_lips = rig.d_lips
    cx = rig.corner_x

    def arc(side, up, out, back):
        """The corner pulled with the lip line following (Rig.lip_arc). mm like moves()."""
        sx = 1 if side == 'L' else -1
        return np.outer(rig.lip_arc(side), np.array([sx * out, back, up]) * rig.mm)

    def lr_arc(name, up, out, back, spec_left):
        keys[name + 'Left'] = arc('L', up, out, back) + rig.moves(spec_left)
        keys[name + 'Right'] = arc('R', up, out, back) + rig.moves(mirror(spec_left))

    jaw_open = rig.jaw_rot(JAW_DEG) + rig.moves({'L_LIPCORNER': (1, -5, 0), 'R_LIPCORNER': (1, -5, 0),
                                            'L_CHEEK_MID': (0, -2, 0), 'R_CHEEK_MID': (0, -2, 0)})
    keys['jawOpen'] = jaw_open
    mm5 = 5 * 0.0011 * s
    keys['jawForward'] = np.outer(rig.jaw_w, np.array([0, -1, 0]) * mm5)
    keys['jawLeft'] = np.outer(rig.jaw_w, np.array([1, 0, 0]) * mm5)
    keys['jawRight'] = np.outer(rig.jaw_w, np.array([-1, 0, 0]) * mm5)

    # mouthClose: with jawOpen, brings the lips together; the lower lip rises most of the way
    # the lower lip (down to its vermilion edge) keeps its closed place; the chin below stretches
    zr = rim_z(rim_l, p[:, 0])
    lip_lo = smoothstep(zr - 0.03 * s, zr - 0.017 * s, p[:, 2]) * rig.jaw_w * ~rig.teeth_w
    keys['mouthClose'] = -jaw_open * lip_lo[:, None]

    lr('eyeBlink', {})
    lr('eyeWide', {})
    lr('eyeSquint', {'L_CHEEK_MID': (0, 3, 0)})
    lr('browDown', {'L_BROW_INNER': (0, -7, 2.5), 'L_BROW_MID': (0, -5.5, 0), 'L_BROW_OUTER': (0, -4, 0)})
    keys['browInnerUp'] = rig.moves({'L_BROW_INNER': (0, 9, 0), 'R_BROW_INNER': (0, 9, 0),
                                     'L_BROW_MID': (0, 3.5, 0), 'R_BROW_MID': (0, 3.5, 0)})
    lr('browOuterUp', {'L_BROW_OUTER': (0, 10, 0), 'L_BROW_MID': (0, 5, 0)})
    keys['cheekPuff'] = rig.moves({'L_CHEEK_MID': (2, 0, -6), 'R_CHEEK_MID': (2, 0, 6),
                                   'L_MOUTHFOLD_LOWER': (2, 0, -5), 'R_MOUTHFOLD_LOWER': (2, 0, 5),
                                   'L_MOUTHFOLD_UPPER': (1, 0, -3), 'R_MOUTHFOLD_UPPER': (1, 0, 3)})
    lr('cheekSquint', {'L_CHEEK_MID': (1, 4, 0)})
    lr('noseSneer', {'L_MOUTHFOLD_UPPER': (0, 3, 0), 'L_NOSE_WING': (0, 3, 0), 'L_UPPERLIP_OUTER': (0, 1.5, 0),
                     'M_NOSE': (0, 0.8, 0)})
    lr_arc('mouthSmile', 10, 7, 6, {'L_CHEEK_MID': (2.5, 6, -1), 'L_MOUTHFOLD_LOWER': (-0.5, 2.5, -1.5),
                                   'L_MOUTHFOLD_UPPER': (0.5, 3, -1)})
    lr_arc('mouthFrown', -6, 1, 0, {'L_MOUTHFOLD_LOWER': (0, -3, 0)})
    lr_arc('mouthStretch', -2, 7, 3, {'L_MOUTHFOLD_LOWER': (0, -2, -3)})
    lr('mouthDimple', {'L_LIPCORNER': (-3, 0, -2.5)})
    lr('mouthPress', {'L_UPPERLIP_INNER': (0, -1, 0), 'L_LOWERLIP_INNER': (0, 1, 0),
                      'L_UPPERLIP_OUTER': (-1, -1, 0), 'L_LOWERLIP_OUTER': (-1, 1, 0)})
    lr('mouthUpperUp', {'L_UPPERLIP_OUTER': (0, 4, 0), 'L_UPPERLIP_INNER': (0, 3.5, 0), 'L_MOUTHFOLD_UPPER': (0, 1.5, 0)})
    lr('mouthLowerDown', {'L_LOWERLIP_OUTER': (0, -4, 0), 'L_LOWERLIP_INNER': (0, -4, 0)})
    # whole-mouth moves use one smooth lip field: separate handles leave dips between them that crease the lips
    t_abs = np.abs(p[:, 0]) / ((cx['L'] + cx['R']) / 2)
    along = np.where(t_abs <= 1, 1, np.exp(-((t_abs - 1) / 0.45) ** 2))
    lipw = (1 - smoothstep(0.01 * s, 0.03 * s, d_lips)) * along * rig.soft_w
    # moving out, the bag behind the corners would slide out through the cheek, so only its lip edge goes along;
    # moving in, it has to follow, or an open mouth (Drifter's grin) shows the bag's corners behind the lips
    lipw_in = lipw.copy()
    lipw[rig.bag] *= 1 - smoothstep(0.001 * s, 0.005 * s, d_lips[rig.bag])
    mid = (rim_z(rim_u, p[:, 0]) + rim_z(rim_l, p[:, 0])) / 2
    upper = smoothstep(-0.002 * s, 0.002 * s, p[:, 2] - mid)
    centre = np.clip(1 - t_abs ** 2, 0, 1)
    mm = rig.mm

    def field(fwd, up, x):
        return np.stack(np.broadcast_arrays(x, -fwd, up), axis=1) * mm

    keys['mouthLeft'] = field(0, 0, 6 * np.where(p[:, 0] < 0, lipw_in, lipw))
    keys['mouthRight'] = field(0, 0, -6 * np.where(p[:, 0] > 0, lipw_in, lipw))
    # pucker: lips push out (most in the middle) and gather towards it
    keys['mouthPucker'] = field(lipw * (5 + 3 * centre), 0, 0) + field(0, 0, -0.38 * p[:, 0] / mm * lipw_in)
    # funnel: as pucker but rounder, with the lips flaring apart in the middle
    keys['mouthFunnel'] = (field(lipw * (3 + 2 * centre), lipw * centre * 3 * (2 * upper - 1), 0)
                           + field(0, 0, -0.2 * p[:, 0] / mm * lipw_in))
    keys['mouthShrugUpper'] = rig.moves({'M_UPPERLIP_OUTER': (1, 2.5, 0), 'L_UPPERLIP_OUTER': (1, 2, 0),
                                         'R_UPPERLIP_OUTER': (1, 2, 0)})
    keys['mouthShrugLower'] = rig.moves({'M_LOWERLIP_OUTER': (1.5, 3, 0), 'L_LOWERLIP_OUTER': (1.5, 2.5, 0),
                                         'R_LOWERLIP_OUTER': (1.5, 2.5, 0), 'M_UNDER_CHIN': (1.5, 2, 0)})
    keys['mouthRollUpper'] = rig.moves({'M_UPPERLIP_OUTER': (-2.5, -2.5, 0), 'L_UPPERLIP_OUTER': (-2.5, -2, 0),
                                        'R_UPPERLIP_OUTER': (-2.5, -2, 0), 'M_UPPERLIP_INNER': (-1.5, 0, 0),
                                        'L_UPPERLIP_INNER': (-1.5, 0, 0), 'R_UPPERLIP_INNER': (-1.5, 0, 0)})
    keys['mouthRollLower'] = rig.moves({'M_LOWERLIP_OUTER': (-2.5, 3, 0), 'L_LOWERLIP_OUTER': (-2.5, 2.5, 0),
                                        'R_LOWERLIP_OUTER': (-2.5, 2.5, 0), 'M_LOWERLIP_INNER': (-1.5, 0, 0),
                                        'L_LOWERLIP_INNER': (-1.5, 0, 0), 'R_LOWERLIP_INNER': (-1.5, 0, 0)})

    # lids on top of the bone-style moves
    for side, name in (('L', 'Left'), ('R', 'Right')):
        keys['eyeBlink' + name] += rig.lid_move(side, 'eyeBlink' + name, upper_frac=BLINK_UPPER, lower_frac=BLINK_LOWER)
        keys['eyeWide' + name] += rig.lid_move(side, 'eyeWide' + name, upper_deg=9, lower_deg=-4)
        keys['eyeSquint' + name] += rig.lid_move(side, 'eyeSquint' + name, upper_frac=0.12, lower_frac=0.3)
        keys['cheekSquint' + name] += rig.lid_move(side, 'cheekSquint' + name, lower_frac=0.12)
        keys['mouthSmile' + name] += rig.lid_move(side, 'mouthSmile' + name, lower_frac=0.06)

    hold = hair_hold(h, rig)
    for k in keys:
        keys[k] *= hold[:, None]
    out = {k: v[h.weld] for k, v in keys.items()}
    out.update(eye_keys(h, rig))
    for side, name in (('L', 'Left'), ('R', 'Right')):
        out['eyeBlink' + name] += glint_sink(h, rig, side)
    attach(h, rig, out)
    return out, rig


def hair_hold(h, rig):
    """Hair and hats are separate meshes the keys don't move: forehead skin under them stays put so it can't come
    through. Per welded vertex, 1 = free."""
    if not len(h.other):
        return np.ones(rig.nw)
    s = rig.s
    d_other, _ = cKDTree(h.other).query(h.wco)
    above = smoothstep(rig.ez + 0.01 * s, rig.ez + 0.025 * s, h.wco[:, 2])
    return 1 - above * (1 - smoothstep(0.004 * s, 0.015 * s, d_other))


def glint_sink(h, rig, side):
    """Glints stand off the eyeball, in front of where the closed lid lies, and would poke through it: the upper lid
    closing pushes them into the ball, twice as fast so they are under its surface by half way."""
    ball = np.where(h.ball[side])[0]
    q = h.co[ball] - rig.eye_c[side]
    rho = np.linalg.norm(q, axis=1)
    off = rho - rig.eye_r[side]
    d = np.zeros((len(h.co), 3))
    g = off > 0.0003
    d[ball[g]] = -q[g] / rho[g, None] * (2 * off[g] + 0.0005)[:, None]
    return d


def eye_keys(h, rig):
    out = {}
    for s, name, sx in (('L', 'Left', 1), ('R', 'Right', -1)):
        ball = h.ball[s]
        piv = h.bones['eye_0_' + s]
        for key, axis, deg, lid_deg in (('eyeLookUp', (1, 0, 0), 15, 5), ('eyeLookDown', (1, 0, 0), -15, -7),
                                        ('eyeLookOut', (0, 0, 1), sx * 22, 0), ('eyeLookIn', (0, 0, 1), -sx * 22, 0)):
            d = np.zeros((len(h.co), 3))
            # rot about +x lifts the front (-y) for positive angles: -y turns to +z
            q = h.co[ball]
            d[ball] = rot_axis(q, piv, np.array(axis, float), np.full(len(q), np.radians(deg) * (-1 if axis[0] else 1))) - q
            if lid_deg:
                # the lids follow vertical gaze a little
                dl = rig.lid_move(s, key + name, upper_deg=lid_deg, lower_deg=lid_deg * 0.5)[h.weld]
                d[~ball] += dl[~ball]
            out[key + name] = d
        out.update(iris_keys(h, rig, s, name))
    return out


def iris_keys(h, rig, side, name, inner=20, outer=40):
    """Iris (and pupil) size: the front cap of the eyeball slides over the sphere, away from or towards the gaze
    axis, fading out between inner and outer degrees."""
    ball = np.where(h.ball[side])[0]
    c = rig.eye_c[side]
    d = h.co[ball] - c
    rho = np.linalg.norm(d, axis=1)
    u = d / rho[:, None]
    front = u[u[:, 1] < -np.cos(np.radians(12))]
    pole = front.mean(0)
    pole /= np.linalg.norm(pole)
    alpha = np.arccos(np.clip(u @ pole, -1, 1))
    perp = u - np.outer(u @ pole, pole)
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    fade = 1 - smoothstep(np.radians(inner), np.radians(outer), alpha)
    out = {}
    for key, k in (('irisLarge', 1.3), ('irisSmall', 0.75)):
        a2 = alpha * (1 + (k - 1) * fade)
        moved = c + rho[:, None] * (np.outer(np.cos(a2), pole) + np.sin(a2)[:, None] * perp)
        off = np.zeros((len(h.co), 3))
        off[ball] = moved - h.co[ball]
        out[key + name] = off
    return out


TONGUE_MID = 'tongueOut_mid'
# combination corrective: rig.py drives it with tongueOut * jawOpen
TONGUE_JAW = 'tongueOut_jawOpen'


def tongue(h, rig, n_ring=14, n_len=16):
    """A tongue lying on the mouth bag floor (the heads have none): ring-by-ring along its length, each ring
    sized to the bag around it. Returns co, tris, jaw weights and u (0 tip .. 1 root) per vertex."""
    s, p = rig.s, h.wco
    soft_tris = rig.soft_w[h.weld[h.tris]].all(1) & ~rig.teeth_w[h.weld[h.tris]].any(1)
    tc = h.co[h.tris[soft_tris]]
    lip_y = p[np.r_[rig.rim_u, rig.rim_l], 1].mean()
    y0 = lip_y + 0.008 * s
    if rig.teeth_w.any():
        low = rig.teeth_w & (p[:, 2] < rig.lip_z)
        if low.any():
            yl = p[low & (np.abs(p[:, 0]) < 0.01 * s), 1]
            if len(yl):
                # a lower piece reaching far back (Lady Geist's) is more than a row of teeth: start no deeper
                y0 = max(y0, min(yl.max(), yl.min() + 0.03 * s) + 0.002 * s)
    ys = y0 + np.linspace(0, 0.06 * s, n_len)
    z_top = rig.lip_z - 0.002 * s

    def cast(o, d):
        return ray_cast(np.asarray(o, float), np.tile(d, (len(o), 1)).astype(float), tc, 0.1)[0]
    down = cast([[0, y, z_top] for y in ys], [0, 0, -1])
    up = cast([[0, y, z_top] for y in ys], [0, 0, 1])
    ok = np.isfinite(down) & np.isfinite(up)
    ok = np.cumprod(ok).astype(bool)
    ys, floor, ceil = ys[ok], z_top - down[ok], z_top + up[ok]
    zmid = (floor + np.minimum(ceil, z_top)) / 2
    side = cast([[0, y, z] for y, z in zip(ys, zmid)], [1, 0, 0])
    side = np.where(np.isfinite(side), side, 0.02 * s)
    u = np.linspace(0, 1, len(ys))
    # rounded tip, full width behind it
    prof = np.sqrt(np.clip(1 - (1 - np.minimum(u / 0.16, 1)) ** 2, 0, 1)) ** 0.7
    hw = np.minimum(0.023 * s, 0.85 * side) * np.maximum(prof, 0.0)
    hh = np.minimum(0.0065 * s, 0.45 * (np.minimum(ceil, z_top) - floor)) * np.maximum(prof, 0.5)
    # deep bags (Drifter): the tongue floats up to where it would show, the dark bag below hides the gap
    cz = np.maximum(floor + hh * 0.85, z_top - 0.016 * s - hh)
    a = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    sa, ca = np.sin(a), np.cos(a)
    # flatter on top, rounder underneath
    sz = np.sign(sa) * np.abs(sa) ** np.where(sa > 0, 0.6, 1.0)
    rings = np.stack([hw[:, None] * ca, np.repeat(ys[:, None], n_ring, 1), cz[:, None] + hh[:, None] * sz], -1)
    tip = np.array([[0, ys[0] - 0.001 * s, cz[0]]])
    root = np.array([[0, ys[-1] + 0.002 * s, cz[-1]]])
    co = np.concatenate([tip, rings[1:].reshape(-1, 3), root])
    uu = np.r_[0, np.repeat(u[1:], n_ring), 1]
    nr = len(ys) - 1
    tris = []
    for j in range(n_ring):
        k = (j + 1) % n_ring
        tris.append((0, 1 + j, 1 + k))
        for i in range(nr - 1):
            a0, a1 = 1 + i * n_ring + j, 1 + i * n_ring + k
            b0, b1 = a0 + n_ring, a1 + n_ring
            tris += [(a0, b1, a1), (a0, b0, b1)]
        last = 1 + (nr - 1) * n_ring
        tris.append((last + j, len(co) - 1, last + k))
    # follows the bag floor under it, which follows the jaw
    floor_v = np.setdiff1d(rig.bag_l, rig.rim_l)
    d, i = cKDTree(p[floor_v]).query(co, k=6)
    wt = 1 / np.maximum(d, 1e-4)
    w = (rig.jaw_w[floor_v][i] * wt).sum(1) / wt.sum(1)
    return co, np.array(tris), w, uu


def mouth_mid_z(rig):
    """Halfway between the lips at the middle: the lip line arches (Paige's middle is 8 mm under its corners), and
    a sculpted open mouth (Drifter's) puts it in the gap."""
    p = rig.h.wco

    def front(rim):
        return rim[(p[rim, 1] < rig.lip_y + 0.012 * rig.s) & (np.abs(p[rim, 0]) < 0.006 * rig.s)]
    zs = [p[r, 2].mean() for r in (front(rig.rim_u), front(rig.rim_l)) if len(r)]
    return float(np.mean(zs)) if zs else rig.lip_z


def tongue_keys(rig, co, w, u, names):
    s = rig.s
    out = {k: np.zeros_like(co) for k in names}
    out['jawOpen'] = rot_x(co, rig.jaw_pivot, -np.radians(JAW_DEG) * w) - co
    mm5 = 5 * 0.0011 * s
    for k, v in (('jawForward', (0, -1, 0)), ('jawLeft', (1, 0, 0)), ('jawRight', (-1, 0, 0))):
        out[k] = np.outer(w, np.array(v) * mm5)
    # out over the lower lip: slides forward, the front rises to lip height and thins
    lead = (1 - u) ** 1.5
    # each ring rises, not just the tip: a bag floor sloping down further in (Mina's) left the part that ends up at
    # the lips under them, through the lower lip; centred a little under the middle of the mouth, it rests on the
    # lower lip
    _, ring = np.unique(u, return_inverse=True)
    ring_z = (np.bincount(ring, weights=co[:, 2]) / np.bincount(ring))[ring]
    lift = np.maximum(mouth_mid_z(rig) - 0.002 * s - ring_z, 0)
    # the front half slides out whole; the root stays and the back half stretches
    slide = 1 - smoothstep(0.35, 1.0, u)
    droop = 0.006 * s * np.clip(1 - u / 0.35, 0, 1) ** 2
    thin = -0.25 * (co[:, 2] - ring_z)
    # a tongue starting far behind the lips (behind Lady Geist's deep lower teeth) slides further to get out
    lip_y = rig.h.wco[np.r_[rig.rim_u, rig.rim_l], 1].mean()
    reach = 0.036 * s + max(0.0, co[u == 0, 1].min() - lip_y - 0.022 * s)
    out['tongueOut'] = np.stack([-0.1 * co[:, 0] * lead, -reach * slide, (lift + thin) * slide - droop], 1)
    # a straight blend lifts and slides together, so half way the tip came out through the lower lip; this
    # corrective (rig.py drives it with a peak at tongueOut 0.5) lifts it to the lips first
    out[TONGUE_MID] = np.stack([np.zeros(len(co)), np.zeros(len(co)), 0.5 * lift * slide], 1)
    return out


def tongue_lips(h, rig, tongue_out):
    """The lips parting around the tongue out over them: the lower lip pushed down and rolled out under it, the
    upper lifted over it, each as far as the tongue goes into it, fading towards the corners.
    tongue_out: (vertices, triangles) of the tongue at tongueOut. Returns welded (lower, upper) offsets."""
    s, p = rig.s, h.wco
    tco, tris = tongue_out
    # the tongue's surface, sampled densely enough to stand in for it as a height field over x, y
    bary = np.array([(a, b, 1 - a - b) for a in np.linspace(0, 1, 6) for b in np.linspace(0, 1, 6) if a + b <= 1])
    surf = np.einsum('kj,tjc->tkc', bary, tco[tris]).reshape(-1, 3)
    tree = cKDTree(surf[:, :2])
    cx = (rig.corner_x['L'] + rig.corner_x['R']) / 2
    grid = np.linspace(-cx, cx, 81)
    blur = np.exp(-((grid[:, None] - grid[None]) / (0.004 * s)) ** 2)
    blur /= blur.sum(1, keepdims=True)
    taper = 1 - smoothstep(0.7, 1.0, np.abs(grid) / cx)
    teeth = rig.teeth_w
    # the bag's edge can include holes deep in the mouth (Apollo's throat): only the lips count
    lips = (p[:, 1] < rig.lip_y + 0.012 * s) & (np.abs(p[:, 2] - rig.lip_z) < 0.015 * s)
    rim = np.r_[rig.rim_u, rig.rim_l]
    d_lips = fg.geodesic(h, rim[lips[rim]], limit=0.05 * s)
    near_lips = rig.soft_w & ~teeth & lips & (d_lips < 0.012 * s)
    out = []
    # upper and lower by the jaw's smooth weights: the lip line's own split zigzags where the lips overlap (Paige's)
    for sign, own in ((-1, rig.jaw_w), (1, 1 - rig.jaw_w)):
        need = np.zeros(len(grid))
        cand = np.where(near_lips & (own > 0.5))[0]
        # only the front few mm show: further in, overlapping lips (Paige's inner lower lip rises behind the upper)
        # would part the mouth far wider, and the tongue hides what they go into
        bins = np.clip(np.searchsorted(grid, p[cand, 0]), 0, len(grid) - 1)
        front = np.full(len(grid), np.inf)
        np.minimum.at(front, bins, p[cand, 1])
        front = np.minimum.reduce([np.roll(front, k) for k in range(-2, 3)])
        for v in cand[p[cand, 1] < front[bins] + 0.004 * s]:
            near = tree.query_ball_point(p[v, :2], 0.0015 * s)
            if not near:
                continue
            z = surf[near, 2]
            # the lower lip goes under the tongue's underside, the upper over its top
            push = p[v, 2] - z.min() if sign < 0 else z.max() - p[v, 2]
            if push <= 0:
                continue
            for x in (p[v, 0], -p[v, 0]):
                i = np.abs(grid - x).argmin()
                need[i] = max(need[i], push + 0.0005 * s)
        # spread past the tongue's sides so the lip bends around it instead of creasing at its edge
        prof = np.maximum.reduce([np.roll(need, k) for k in range(-3, 4)])
        prof = blur @ prof * taper
        amount = np.interp(p[:, 0], grid, prof, left=0, right=0)
        body = (1 - smoothstep(0.006 * s, 0.035 * s, d_lips)) * own * rig.soft_w * ~teeth
        edge_w = np.exp(-(d_lips / (0.006 * s)) ** 2) * own * rig.soft_w * ~teeth
        roll = 0.3 if sign < 0 else 0.15
        out.append(np.stack([np.zeros(rig.nw), -roll * amount * edge_w, sign * amount * body], 1))
    return tuple(out)


class Cards:
    """Cards (lashes, liner, brows: pieces not welded to the skin) and how they ride the skin under them."""

    def __init__(self, h, rig):
        s = rig.s
        ball = h.ball['L'] | h.ball['R']
        cards = ~rig.soft_w[h.weld] & ~ball & ~rig.teeth_w[h.weld]
        margin = np.concatenate([rig.margins[side][k] for side in 'LR' for k in ('upper', 'lower')])
        skin = np.where(rig.face_w)[0]
        m_tree, s_tree = cKDTree(h.wco[margin]), cKDTree(h.wco[skin])
        comp = rig.comp
        anchors = np.zeros(len(h.co), int)
        lid_card = np.zeros(len(h.co), bool)
        follow = np.zeros(len(h.co))
        skin_soft = np.where(rig.soft_w)[0]
        soft_tree = cKDTree(h.wco[skin_soft])
        for c in np.unique(comp[h.weld[cards]]):
            sel = cards & (comp[h.weld] == c)
            dm, im = m_tree.query(h.co[sel])
            ds, _ = soft_tree.query(h.co[sel])
            small = np.ptp(h.co[sel], axis=0).max() < 0.08 * s
            # lashes, brows, liner, moustaches lie on the skin; hair and glasses stand off it and stay put
            on_skin = np.median(ds) < 0.01 * s and np.ptp(h.co[sel], axis=0).max() < 0.13 * s
            # beards hang off the jaw (Mirage's): they go with it whatever their size
            _, ia = soft_tree.query(h.co[sel])
            on_skin |= np.median(ds) < 0.02 * s and rig.jaw_w[skin_soft[ia]].mean() > 0.2
            follow[sel] = float(on_skin)
            if np.median(dm) < 0.012 * s and small:
                anchors[sel] = margin[im]
                lid_card[sel] = True
            else:
                anchors[sel] = skin[s_tree.query(h.co[sel])[1]]
        self.mask, self.idx = cards, np.where(cards)[0]
        self.lid_card = lid_card
        on_upper = lid_card & np.isin(anchors, np.concatenate([rig.margins[side]['upper'] for side in 'LR']))
        # a whole lash goes with one lid: per vertex, the tips of a long upper lash (Paige's) lie nearer the lower
        # margin and folded the other way
        self.upper_card = np.zeros(len(h.co), bool)
        for c in np.unique(comp[h.weld[lid_card]]):
            sel = lid_card & (comp[h.weld] == c)
            self.upper_card[sel] = on_upper[sel].mean() > 0.5
        self.follow = follow
        self.skin_soft = skin_soft
        # cards follow a smooth blend of the skin around them
        dist, self.near = soft_tree.query(h.co[self.idx], k=8)
        wts = 1 / np.maximum(dist, 1e-4) ** 2
        self.wts = wts / wts.sum(1, keepdims=True)

    def spread(self, welded):
        """Per-welded-vertex values (n,) or (n,3) blended onto the cards (unwelded rows of the card vertices)."""
        v = welded[self.skin_soft][self.near]
        f = self.follow[self.idx]
        if v.ndim == 2:
            return (self.wts * v).sum(1) * f
        return np.einsum('ik,ikj->ij', self.wts, v) * f[:, None]


def attach(h, rig, keys, cards=None):
    """Cards ride the skin under them. Liner and lash cards turn with the lid margin as one piece, else the lid
    skin's falloff stretches them over the whole closed lid. Pieces standing off the skin (hair, glasses) stay
    put; teeth go with the jaw."""
    cards = cards or Cards(h, rig)
    lid_card, upper_card = cards.lid_card, cards.upper_card
    for name, d in keys.items():
        dw = np.zeros((rig.nw, 3))
        dw[h.weld[~cards.mask]] = d[~cards.mask]
        lid_ang = np.zeros(len(h.co))
        # lid cards take the closing turns rigidly (raising turns stay blended so the skin above does not poke
        # through)
        for side, amounts in rig.lid_specs.get(name, []):
            closing = {k: v for k, v in amounts.items() if k.endswith('_frac')}
            if not closing:
                continue
            dw -= rig.lid_move(side, **closing)
            sel = lid_card & ((h.co[:, 0] > 0) == (side == 'L'))
            au, al = rig.lid_angles(side, h.co[sel, 0], **closing)
            lid_ang[sel] += np.where(upper_card[sel], au, al)
        d[cards.idx] = cards.spread(dw)
        for side in 'LR':
            ss = np.where(lid_card & (lid_ang != 0) & ((h.co[:, 0] > 0) == (side == 'L')))[0]
            if len(ss):
                d[ss] += rot_x(h.co[ss], rig.eye_c[side], lid_ang[ss]) - h.co[ss]
                # the lid skin above the margin bulges as it bunches up and came through thick liner (Paige's):
                # the card lifts off the ball as it turns
                out = h.co[ss] - rig.eye_c[side]
                out /= np.linalg.norm(out, axis=1, keepdims=True)
                d[ss] += out * (0.0025 * rig.s * np.abs(lid_ang[ss]))[:, None]


def controls(rig):
    """Face control positions (rig.py): in front of the skin at each landmark, as far out as Violet's hand-placed
    ones."""
    p = rig.h.wco
    s = rig.s
    ez, lz = rig.ez, rig.lip_z
    em = ez - lz
    c = rig.eye_c['L']
    cx, cz = p[rig.corner['L'], 0], p[rig.corner['L'], 2]

    def at(x, z, ahead):
        return [round(float(x), 4), round(float(rig.front_y(x, z) - ahead * s), 4), round(float(z), 4)]
    lm = {
        'lid': at(c[0], ez + 0.016 * s, 0.010),
        'lidlow': at(c[0] + 0.002 * s, ez - 0.019 * s, 0.011),
        'brow': at(c[0] + 0.006 * s, ez + 0.045 * s, 0.015),
        'cheek': at(c[0] + 0.014 * s, lz + 0.475 * em, 0.010),
        'corner': at(cx, cz, 0.012),
        'lipup': at(0.012 * s, lz + 0.015 * s, 0.015),
        'liplow': at(0.012 * s, lz - 0.017 * s, 0.022),
        'nose': at(0.019 * s, lz + 0.36 * em, 0.013),
    }
    chin = rig.face_w & (np.abs(p[:, 0]) < 0.01 * s) & (p[:, 2] < lz - 0.02 * s) & (p[:, 2] > lz - 0.06 * s)
    top = p[rig.soft_w, 2].max()
    return {
        'landmarks': lm,
        'brow_mid': at(0, ez + 0.033 * s, 0.011),
        'mouth': at(0, lz, 0.028),
        'jaw': [0, round(float(p[chin, 1].min() - 0.023 * s), 4), round(float(lz - 0.055 * s), 4)],
        'eye_target': [0, round(float(c[1] - 0.51), 4), round(float(ez), 4)],
        'eye_spacing': round(float(c[0]), 4),
        'face_box': [0, round(float(rig.front_y(0, ez) + 0.019), 4), round(float(top + 0.126 * s), 4)],
    }


def cut_lips(d, h, rig):
    """Splits closed lips along rig.lip_cut: the vertices along the upper lip's edge (but not the corners) get a
    copy, a hair lower, for the faces on the lower lip's side, so the lips come apart there. Returns d with the
    copies appended and the faces rewired, and what build.py needs to do the same: (copied vertex, offset) per new
    vertex and (face, corner, new vertex) per rewired corner."""
    s = rig.s
    path, below = rig.lip_cut
    inner = np.zeros(len(h.wco), bool)
    inner[path[1:-1]] = True
    cut_e = {tuple(sorted(e)) for e in zip(path[:-1].tolist(), path[1:].tolist())}
    tw = h.weld[h.tris]
    # the faces around the path, joined across their shared edges except the path's own: the lower lip's side is
    # the piece holding the face between the middle and the lower-lip vertex (a side test per corner breaks where
    # the path zigzags)
    # two rings out, so faces caught inside a bump of the path still reach the rest of their side
    ring = inner.copy()
    for _ in range(2):
        ring[tw[ring[tw].any(1)].ravel()] = True
    # (but not around the corners, where they would join the lips back up)
    cx = h.wco[tw, 0].mean(1)
    lo, hi = sorted(h.wco[[path[0], path[-1]], 0])
    fan = np.where(inner[tw].any(1) | (ring[tw].any(1) & (cx > lo + 0.004 * s) & (cx < hi - 0.004 * s)))[0]
    by_edge = {}
    for f in fan:
        for k in range(3):
            e = tuple(sorted((int(tw[f, k]), int(tw[f, (k + 1) % 3]))))
            if e not in cut_e:
                by_edge.setdefault(e, []).append(f)
    pos = {f: i for i, f in enumerate(fan)}
    pairs = [(pos[fs[0]], pos[g]) for fs in by_edge.values() for g in fs[1:]]
    pairs = np.array(pairs).reshape(-1, 2)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                             shape=(len(fan), len(fan))), directed=False)
    m = path[len(path) // 2]
    seed = [i for i, f in enumerate(fan) if below in tw[f] and (tw[f] == path[:, None]).any()]
    if not seed:
        seed = [i for i, f in enumerate(fan) if below in tw[f]]
    size = np.bincount(lab)
    on_lower = lab == lab[seed[0]]
    # stray faces cut off on their own (slivers on Wraith's lips): below the path's height there is the lower lip
    order = np.argsort(h.wco[path, 0])
    line_z = np.interp(cx[fan], h.wco[path[order], 0], h.wco[path[order], 2])
    on_lower |= (size[lab] < 10) & (h.wco[tw[fan], 2].mean(1) < line_z)
    lower = fan[on_lower]
    n0 = len(h.co)
    new_of, src, rewire = {}, [], []
    tris = d['tris'].copy()
    for f in lower:
        for k in range(3):
            if inner[tw[f, k]]:
                u = tris[f, k]
                if u not in new_of:
                    new_of[u] = n0 + len(src)
                    src.append(u)
                tris[f, k] = new_of[u]
                rewire.append((f, k, new_of[u]))
    src = np.array(src)
    off = np.tile([0, 0, -0.0002 * s], (len(src), 1))
    out = dict(d)
    out['co'] = np.concatenate([d['co'], d['co'][src] + off]).astype(d['co'].dtype)
    out['nrm'] = np.concatenate([d['nrm'], d['nrm'][src]]).astype(d['nrm'].dtype)
    out['tris'] = tris
    # the hole's edge, so the next Rig need not find it among the open edges (Wraith's lips have edges shared by
    # three faces, which leave gaps in it)
    rep = {int(w): u for u, w in enumerate(h.weld)}
    copy = {int(h.weld[u]): v for u, v in new_of.items()}
    out['lip_loop'] = np.array([rep[int(v)] for v in path] + [copy.get(int(v), rep[int(v)]) for v in path[-2:0:-1]])
    for k in ('ball_L', 'ball_R'):
        out[k] = np.concatenate([d[k], np.zeros(len(src), bool)])
    return out, {'src': src, 'off': off, 'rewire': np.array(rewire)}


def add_bag(d, h, rig, rings=10):
    """A mouth bag for a head whose mouth is a hole into nothing (Venator's): a tube from the lips' open edge back
    past the teeth, widening to hold them, closed at the back. Returns d with the bag appended (its first ring sits
    on the edge, so it welds to the lips) and the bag's (co, tris)."""
    s, p = rig.s, h.wco
    loop = rig.hole
    v = p[loop]
    n = len(loop)
    teeth = p[rig.teeth_w] if rig.teeth_w.any() else v
    zc = (teeth[:, 2].max() + teeth[:, 2].min()) / 2 if rig.teeth_w.any() else v[:, 2].mean() - 0.004 * s
    rx = max(np.abs(teeth[:, 0]).max(), np.abs(v[:, 0]).max()) + 0.004 * s
    rz = max((teeth[:, 2].max() - teeth[:, 2].min()) / 2 + 0.004 * s, 0.012 * s)
    # deep enough to pass for a real one (and the shallow-bag test) even with no teeth to go around
    y_back = min(max(teeth[:, 1].max() + 0.008 * s, v[:, 1].min() + 0.05 * s), v[:, 1].min() + 0.09 * s)
    # around the edge: upper lip 0..pi from the left corner, lower lip 0..-pi
    hx = np.abs(v[:, 0]).max()
    i_l, i_r = v[:, 0].argmax(), v[:, 0].argmin()
    a, b = sorted((i_l, i_r))
    chain = np.zeros(n, bool)
    chain[a:b + 1] = True
    upper_chain = v[chain, 2].mean() > v[~chain, 2].mean()
    is_up = chain if upper_chain else ~chain
    is_up[[i_l, i_r]] = True
    phi = np.arccos(np.clip(v[:, 0] / hx, -1, 1)) * np.where(is_up, 1, -1)
    ell = np.c_[rx * np.cos(phi), rz * np.sin(phi) + zc]
    # lips split along a line (Grey Talon's) leave a slit with no height: open up and down straight away, or the
    # front of the bag is flat
    slit = abs(v[is_up, 2].mean() - v[~is_up, 2].mean()) < 0.003 * s
    ring_pts = []
    for k in range(rings + 1):
        t = k / rings
        # straight back first (the lips' inner faces), then out around the teeth, so the walls stay inside the cheeks
        f = smoothstep(0.35, 0.8, t)
        fz = smoothstep(0.0, 0.45, t) if slit else f
        xz = np.c_[v[:, 0] * (1 - f) + ell[:, 0] * f, v[:, 2] * (1 - fz) + ell[:, 1] * fz]
        y = v[:, 1] + (y_back - v[:, 1]) * t ** 0.7
        ring_pts.append(np.c_[xz[:, 0], y, xz[:, 1]])
    cap = np.array([[0, y_back + 0.004 * s, zc]])
    co = np.concatenate(ring_pts + [cap])
    tris = []
    for k in range(rings):
        for i in range(n):
            j = (i + 1) % n
            a0, a1, b0, b1 = k * n + i, k * n + j, (k + 1) * n + i, (k + 1) * n + j
            tris += [(a0, b0, a1), (a1, b0, b1)]
    last, c = rings * n, len(co) - 1
    tris += [(last + i, c, last + (i + 1) % n) for i in range(n)]
    tris = np.array(tris)
    # face into the mouth, like the lips' skin they continue
    mid = np.array([0, (v[:, 1].mean() + y_back) / 2, zc])
    fn = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    if ((mid - co[tris].mean(1)) * fn).sum(1).mean() < 0:
        tris = tris[:, ::-1]
        fn = -fn
    nrm = np.zeros_like(co)
    for kk in range(3):
        np.add.at(nrm, tris[:, kk], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    n0 = len(d['co'])
    out = dict(d)
    out['co'] = np.concatenate([d['co'], co]).astype(d['co'].dtype)
    out['nrm'] = np.concatenate([d['nrm'], nrm]).astype(d['nrm'].dtype)
    out['tris'] = np.concatenate([d['tris'], tris + n0])
    out['matnames'] = np.append(d['matnames'], 'mouth_bag')
    out['mats'] = np.concatenate([d['mats'], np.full(len(tris), len(d['matnames']))])
    for side in 'LR':
        out['ball_' + side] = np.concatenate([d['ball_' + side], np.zeros(len(co), bool)])
    return out, co, tris


if __name__ == '__main__':
    arrays = dict(np.load(sys.argv[1]))
    h = fg.from_arrays(arrays)
    n_head = len(h.co)
    bag_mesh = None
    probe = Rig(h)
    cut = None
    if probe.fused is not None:
        # open the sculpted closed lips by dropping the faces joining them; build.py drops the same faces
        cut = probe.fused
        arrays['tris'] = np.delete(arrays['tris'], cut, axis=0)
        arrays['mats'] = np.delete(arrays['mats'], cut)
        h = fg.from_arrays(arrays)
        probe = Rig(h)
        print('opened the lips:', len(cut), 'faces')
    split = None
    if probe.lip_cut is not None:
        arrays, split = cut_lips(arrays, h, probe)
        h = fg.from_arrays(arrays)
        n_head = len(h.co)
        probe = Rig(h)
        print('split the lips along', len(split['src']), 'vertices')
    if probe.hole is not None:
        arrays, bco, btris = add_bag(arrays, h, probe)
        arrays.pop('lip_loop', None)
        h = fg.from_arrays(arrays)
        bag_mesh = (bco, btris)
        print('mouth bag built on the lips\' edge:', len(bco), 'verts')
    keys, rig = build(h)
    base = sys.argv[2][:-len('.npz')]
    # the dark mouth interior starts a little in from the lip line, so the inner lips keep their colour
    interior = rig.bag & (fg.geodesic(h, np.r_[rig.rim_u, rig.rim_l], limit=0.01) > 0.003 * rig.s)
    np.save(base + '_bag.npy', (rig.bag.astype(np.int8) + interior)[h.weld][:n_head])
    json.dump(controls(rig), open(os.path.join(os.path.dirname(sys.argv[2]), 'face.json'), 'w'), indent=1)
    for k in sorted(keys):
        print('%-20s max %.1f mm' % (k, np.linalg.norm(keys[k], axis=1).max() * 1000))
    print(len(keys), 'keys', 'scale %.3f' % rig.s, 'bag', rig.bag.sum(), 'lip line', len(rig.rim_u), len(rig.rim_l),
          'margins', [len(rig.margins[side][k]) for side in 'LR' for k in ('upper', 'lower')],
          'socket', [int(rig.sock[side].sum()) for side in 'LR'])
    tco, ttris, tw, tu = tongue(h, rig)
    lower, upper = tongue_lips(h, rig, (tco + tongue_keys(rig, tco, tw, tu, [])['tongueOut'], ttris))
    hold = hair_hold(h, rig)[:, None]
    lips = {'tongueOut': ((lower + upper) * hold)[h.weld], TONGUE_MID: (0.5 * (lower + upper) * hold)[h.weld],
            # with the jaw open the upper lip is already clear of the tongue
            TONGUE_JAW: (-upper * hold)[h.weld]}
    attach(h, rig, lips)
    for k in sorted(lips):
        print('%-20s max %.1f mm' % (k, np.linalg.norm(lips[k], axis=1).max() * 1000))
    if np.linalg.norm(lips[TONGUE_JAW], axis=1).max() < 0.0005 * rig.s:
        del lips[TONGUE_JAW]
    keys.update(lips)
    tk = tongue_keys(rig, tco, tw, tu, keys)
    print('tongue', len(tco), 'verts, jaw weight %.2f..%.2f' % (tw.min(), tw.max()))
    np.savez_compressed(base + '_tongue.npz', co=tco, tris=ttris, **{'key_' + k: v for k, v in tk.items()})
    cut_path = base + '_cut.npy'
    if cut is not None:
        np.save(cut_path, cut)
    elif os.path.exists(cut_path):
        os.remove(cut_path)
    split_path = base + '_split.npz'
    if split is not None:
        np.savez_compressed(split_path, **split)
    elif os.path.exists(split_path):
        os.remove(split_path)
    mouth_path = base + '_mouth.npz'
    if bag_mesh is not None:
        np.savez_compressed(mouth_path, co=bag_mesh[0], tris=bag_mesh[1],
                            **{'key_' + k: v[n_head:] for k, v in keys.items()})
    elif os.path.exists(mouth_path):
        os.remove(mouth_path)
    np.savez_compressed(sys.argv[2], **{k: v[:n_head] for k, v in keys.items()})
