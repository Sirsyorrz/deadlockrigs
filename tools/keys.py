"""Builds ARKit facial shape keys for Violet's head mesh as per-vertex offsets.

python keys.py head.npz keys.npz
World axes (Blender, glTF import): +x her left, -y forward, +z up, metres.
The face has no face bones (only head, jaw_0 unused, eyes), so keys come from:
- lids: rotations about the eyeball sphere centre, upper/lower lid split by geodesic distance to the
  lid margins (the face island meets the eye socket island there),
- jaw: rotation about jaw_0, weighted by geodesic distance to the upper and lower mouth bag islands,
- everything else: translation-only virtual bones (geodesic Gaussian weights, normalised like skinning
  with a static root),
- lash and liner cards follow the rigid motion of the lid skin under them (Kabsch on a patch).
"""
import sys

import numpy as np

import facegeo as fg

MM = 0.001 * 1.1  # Crud's sizes are for a pirate head; Violet's is about 10% larger

FACE, BAG_U, BAG_L = 6, 42, 87
SOCKET = {'L': 48, 'R': 55}
BALL = {'L': (0, 1, 4), 'R': (3, 2, 5)}
EYE_C = {'L': np.array([0.044, -0.0403, 2.1469]), 'R': np.array([-0.044, -0.0403, 2.1469])}
EYE_R = 0.0245


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


class Rig:
    def __init__(self, h: fg.Head):
        self.h = h
        self.n = len(h.co)
        self.nw = len(h.wco)
        isl_w = np.full(self.nw, -1)
        isl_w[h.weld] = h.island
        self.face_w = np.zeros(self.nw, bool)
        self.face_w[h.weld[h.island == FACE]] = True
        # the deformable skin: everything welded to the face (sockets, mouth bag, small UV islands)
        from scipy.sparse.csgraph import connected_components
        _, comp = connected_components(h.wgraph, directed=False)
        self.soft_w = comp == comp[h.weld[h.island == FACE][0]]
        self.bag_u = np.unique(h.weld[h.island == BAG_U])
        self.bag_l = np.unique(h.weld[h.island == BAG_L])
        self.handles = {}
        self.lid_specs = {}
        self.margins = {s: self._margin(s) for s in 'LR'}
        self._jaw_weights()
        self._lid_weights()

    def shared(self, a, b):
        return np.intersect1d(self.h.weld[self.h.island == a], self.h.weld[self.h.island == b])

    def landmark(self, x, z, r=0.004):
        """Frontmost face vertex near (x, z)."""
        p = self.h.wco
        d = np.hypot(p[:, 0] - x, p[:, 2] - z)
        cand = np.where(self.face_w & (d < r))[0]
        if len(cand) == 0:
            cand = np.where(self.face_w)[0]
            return cand[d[cand].argmin()]
        return cand[p[cand, 1].argmin()]

    # lids

    def _margin(self, side):
        ids = self.shared(FACE, SOCKET[side])
        p = self.h.wco[ids]
        c = EYE_C[side]
        sx = 1 if side == 'L' else -1
        xs = p[:, 0] * sx
        inner = ids[xs.argmin()]
        outer = ids[xs.argmax()]
        pi, po = self.h.wco[inner], self.h.wco[outer]
        t = (p[:, 0] - pi[0]) / (po[0] - pi[0])
        line_z = pi[2] + t * (po[2] - pi[2])
        upper = ids[p[:, 2] >= line_z]
        lower = ids[p[:, 2] < line_z]
        phi = lambda q: np.arctan2(q[:, 2] - c[2], -(q[:, 1] - c[1]))

        def chain(sel):
            q = self.h.wco[sel]
            o = np.argsort(q[:, 0])
            return q[o, 0], phi(q[o])
        return {'upper': upper, 'lower': lower, 'up_chain': chain(upper), 'lo_chain': chain(lower),
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
        h = self.h
        self.lid = {}
        for s in 'LR':
            m = self.margins[s]
            du = fg.geodesic(h, m['upper'], limit=0.03)
            dl = fg.geodesic(h, m['lower'], limit=0.03)
            sock = np.zeros(self.nw, bool)
            sock[h.weld[h.island == SOCKET[s]]] = True
            up = (du < dl) & self.soft_w
            lo = (dl <= du) & self.soft_w
            wu = np.where(up, 1 - smoothstep(0.0012, 0.0095, du), 0)
            wl = np.where(lo, 1 - smoothstep(0.001, 0.007, dl), 0)
            wu[sock & up] = 1
            wl[sock & lo] = 1
            self.lid[s] = {'wu': wu, 'wl': wl, 'sock': sock, 'du': du, 'dl': dl}

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
        c = EYE_C[side]
        au, al = self.lid_angles(side, p[:, 0], **amounts)
        ang = au * lw['wu'] + al * lw['wl']
        moved = rot_x(p, c, ang)
        # keep moving lid skin outside the eyeball
        skin = (ang != 0) & ~lw['sock']
        d = moved[skin] - c
        r = np.linalg.norm(d, axis=1)
        r0 = np.linalg.norm(p[skin] - c, axis=1)
        need = np.minimum(EYE_R + 0.0007, r0 + 0.0015)
        grow = np.maximum(need - r, 0) * np.minimum(np.abs(ang[skin]) / 0.05, 1)
        moved[skin] += d / r[:, None] * grow[:, None]
        return moved - p

    # jaw

    def _jaw_weights(self):
        h = self.h
        p = h.wco
        du = fg.geodesic(h, self.bag_u, limit=0.2)
        dl = fg.geodesic(h, self.bag_l, limit=0.2)
        reach = np.isfinite(du) & np.isfinite(dl)
        du, dl = np.where(reach, du, 1.0), np.where(reach, dl, 1.0)
        s = 0.0015 + 0.12 * np.minimum(du, dl)
        lab = 1 / (1 + np.exp(-(du - dl) / s))
        lab[self.bag_u] = 0
        lab[self.bag_l] = 1
        piv = self.jaw_pivot = np.array([0.0, 0.036, 2.0992])
        # Harmonic weights between the parts that surely follow the jaw (lower lip, chin, lower mouth
        # bag) and the parts that surely do not (upper lip and face, neck, behind the jaw), so the skin
        # under the jaw stretches smoothly into the neck instead of folding.
        one = np.zeros(self.nw, bool)
        zero = ~self.soft_w
        one[self.bag_l] = True
        zero[self.bag_u] = True
        rim_u = np.intersect1d(np.unique(h.weld[h.island == FACE]), self.bag_u)
        rim_l = np.intersect1d(np.unique(h.weld[h.island == FACE]), self.bag_l)
        corners = np.intersect1d(rim_u, rim_l)
        corners = corners[[np.argmax(p[corners, 0]), np.argmin(p[corners, 0])]]
        near_corner = np.min(np.linalg.norm(p[:, None] - p[corners][None], axis=2), axis=1) < 0.007
        one |= (lab > 0.6) & (dl < 0.012) & self.soft_w & ~near_corner
        zero |= (lab < 0.4) & (du < 0.012) & ~near_corner
        one |= (p[:, 1] < -0.06) & (p[:, 2] < 2.055) & (lab > 0.5) & self.soft_w
        zero |= (p[:, 2] > 2.09) | (p[:, 2] < 1.96) | (p[:, 1] > 0.03)
        # the mouth bag floor lies a few mm over the skin under the chin: it takes that skin's weight
        # (solved without the bag), else it pokes out under the jaw
        inner = np.setdiff1d(self.bag_l, rim_l)
        inner = inner[np.min(np.linalg.norm(p[inner][:, None] - p[rim_l][None], axis=2), axis=1) > 0.006]
        one[inner] = False
        zero[inner] = True
        zero &= ~one
        w = harmonic(self.h, one, zero)
        from scipy.spatial import cKDTree
        outside = np.where(self.face_w & ~near_corner)[0]
        outside = outside[(p[outside, 2] < 2.05) & (h.wnrm[outside, 2] < -0.3)]
        w[inner] = w[outside[cKDTree(p[outside]).query(p[inner])[1]]]
        self.jaw_w = w
        self.mouth_lab = lab
        self.du_bag, self.dl_bag = du, dl

    def jaw_rot(self, deg):
        p = self.h.wco
        return rot_x(p, self.jaw_pivot, -np.radians(deg) * self.jaw_w) - p

    # virtual bones

    def handle(self, name, vid, radius):
        g = fg.geodesic(self.h, [vid], limit=radius * 3)
        w = np.exp(-(g / radius) ** 2)
        w[~self.soft_w] = 0
        self.handles[name] = w

    def moves(self, spec):
        """spec: {handle: crud-style mm vector (forward, up, towards her right)}"""
        tot = np.zeros(self.nw)
        for w in self.handles.values():
            tot += w
        norm = np.maximum(tot, 1)
        out = np.zeros((self.nw, 3))
        for name, v in spec.items():
            f, u, r = v
            world = np.array([-r, -f, u]) * MM
            out += np.outer(self.handles[name] / norm, world)
        return out


def build(h: fg.Head) -> dict:
    rig = Rig(h)
    p = h.wco
    L = rig.landmark
    rim_u = np.unique(rig.shared(FACE, BAG_U))
    rim_l = np.unique(rig.shared(FACE, BAG_L))
    corner = {}
    for s, sx in (('L', 1), ('R', -1)):
        both = np.intersect1d(rim_u, rim_l)
        cand = both[(p[both, 0] * sx) > 0]
        corner[s] = cand[np.argmax(np.abs(p[cand, 0]))]

    def rim_z(rim, x):
        q = p[rim]
        o = np.argsort(q[:, 0])
        return np.interp(x, q[o, 0], q[o, 2])

    for s, sx in (('L', 1), ('R', -1)):
        cx, cz = p[corner[s], 0], p[corner[s], 2]
        rig.handle(f'{s}_LIPCORNER', corner[s], 0.009)
        rig.handle(f'{s}_UPPERLIP_INNER', rim_u[np.argmin(np.abs(p[rim_u, 0] - sx * 0.014))], 0.005)
        rig.handle(f'{s}_LOWERLIP_INNER', rim_l[np.argmin(np.abs(p[rim_l, 0] - sx * 0.014))], 0.005)
        rig.handle(f'{s}_UPPERLIP_OUTER', L(sx * 0.014, rim_z(rim_u, sx * 0.014) + 0.006), 0.006)
        rig.handle(f'{s}_LOWERLIP_OUTER', L(sx * 0.014, rim_z(rim_l, sx * 0.014) - 0.007), 0.006)
        rig.handle(f'{s}_MOUTHFOLD_UPPER', L(sx * 0.034, cz + 0.018), 0.009)
        rig.handle(f'{s}_MOUTHFOLD_LOWER', L(sx * 0.034, cz - 0.02), 0.009)
        rig.handle(f'{s}_CHEEK_MID', L(sx * 0.05, 2.108), 0.013)
        rig.handle(f'{s}_NOSE_WING', L(sx * 0.014, 2.093, 0.003), 0.006)
        rig.handle(f'{s}_BROW_INNER', L(sx * 0.02, 2.166), 0.009)
        rig.handle(f'{s}_BROW_MID', L(sx * 0.045, 2.178), 0.01)
        rig.handle(f'{s}_BROW_OUTER', L(sx * 0.068, 2.174), 0.01)
    rig.handle('M_UPPERLIP_INNER', rim_u[np.argmin(np.abs(p[rim_u, 0]))], 0.005)
    rig.handle('M_LOWERLIP_INNER', rim_l[np.argmin(np.abs(p[rim_l, 0]))], 0.005)
    rig.handle('M_UPPERLIP_OUTER', L(0, rim_z(rim_u, 0) + 0.006), 0.006)
    rig.handle('M_LOWERLIP_OUTER', L(0, rim_z(rim_l, 0) - 0.007), 0.006)
    rig.handle('M_UNDER_CHIN', L(0, 2.035), 0.012)
    rig.handle('M_NOSE', L(0, 2.1), 0.008)

    keys = {}

    def mirror(spec):
        return {('R' + k[1:] if k[0] == 'L' else 'L' + k[1:] if k[0] == 'R' else k): (v[0], v[1], -v[2]) for k, v in spec.items()}

    def lr(name, spec_left):
        keys[name + 'Left'] = rig.moves(spec_left)
        keys[name + 'Right'] = rig.moves(mirror(spec_left))

    jaw_open = rig.jaw_rot(14) + rig.moves({'L_LIPCORNER': (1, -5, 0), 'R_LIPCORNER': (1, -5, 0),
                                            'L_CHEEK_MID': (0, -2, 0), 'R_CHEEK_MID': (0, -2, 0)})
    keys['jawOpen'] = jaw_open
    fwd = np.array([0, -1, 0]) * 5 * MM
    keys['jawForward'] = np.outer(rig.jaw_w, fwd)
    keys['jawLeft'] = np.outer(rig.jaw_w, np.array([1, 0, 0]) * 5 * MM)
    keys['jawRight'] = np.outer(rig.jaw_w, np.array([-1, 0, 0]) * 5 * MM)

    # mouthClose: with jawOpen, brings the lips together; the lower lip rises most of the way
    # the lower lip (down to its vermilion edge) keeps its closed place; the chin below stretches
    zr = rim_z(rim_l, p[:, 0])
    lip_lo = smoothstep(zr - 0.03, zr - 0.017, p[:, 2]) * rig.jaw_w
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
    lr('mouthSmile', {'L_LIPCORNER': (-5, 12, -7), 'L_CHEEK_MID': (1.5, 5, -1), 'L_MOUTHFOLD_LOWER': (-1.5, 5, -3),
                      'L_MOUTHFOLD_UPPER': (0.5, 4, -1)})
    lr('mouthFrown', {'L_LIPCORNER': (0, -9, 0), 'L_MOUTHFOLD_LOWER': (0, -4.5, 0)})
    lr('mouthStretch', {'L_LIPCORNER': (-2.5, -4, -9), 'L_MOUTHFOLD_LOWER': (0, -3, -4)})
    lr('mouthDimple', {'L_LIPCORNER': (-3, 0, -2.5)})
    lr('mouthPress', {'L_UPPERLIP_INNER': (0, -1, 0), 'L_LOWERLIP_INNER': (0, 1, 0),
                      'L_UPPERLIP_OUTER': (-1, -1, 0), 'L_LOWERLIP_OUTER': (-1, 1, 0)})
    lr('mouthUpperUp', {'L_UPPERLIP_OUTER': (0, 4, 0), 'L_UPPERLIP_INNER': (0, 3.5, 0), 'L_MOUTHFOLD_UPPER': (0, 1.5, 0)})
    lr('mouthLowerDown', {'L_LOWERLIP_OUTER': (0, -4, 0), 'L_LOWERLIP_INNER': (0, -4, 0)})
    lips = [f'{s}_{k}' for s in 'MLR' for k in ('UPPERLIP_INNER', 'UPPERLIP_OUTER', 'LOWERLIP_INNER', 'LOWERLIP_OUTER')]
    side = {n: (0, 0, -6) for n in lips}
    side.update({'L_LIPCORNER': (0, 0, -6), 'R_LIPCORNER': (0, 0, -6)})
    keys['mouthLeft'] = rig.moves(side)
    keys['mouthRight'] = rig.moves(mirror(side))
    keys['mouthPucker'] = rig.moves({'L_LIPCORNER': (5.6, 0, 9.8), 'R_LIPCORNER': (5.6, 0, -9.8),
                                     'L_UPPERLIP_OUTER': (7, 0, 4.2), 'R_UPPERLIP_OUTER': (7, 0, -4.2),
                                     'L_LOWERLIP_OUTER': (7, 0, 4.2), 'R_LOWERLIP_OUTER': (7, 0, -4.2),
                                     'M_UPPERLIP_OUTER': (8.4, 0, 0), 'M_LOWERLIP_OUTER': (8.4, 0, 0),
                                     'L_UPPERLIP_INNER': (4.9, 0, 2.8), 'R_UPPERLIP_INNER': (4.9, 0, -2.8),
                                     'L_LOWERLIP_INNER': (4.9, 0, 2.8), 'R_LOWERLIP_INNER': (4.9, 0, -2.8),
                                     'M_UPPERLIP_INNER': (5.6, 0, 0), 'M_LOWERLIP_INNER': (5.6, 0, 0),
                                     'L_MOUTHFOLD_LOWER': (1.4, 0, 2.8), 'R_MOUTHFOLD_LOWER': (1.4, 0, -2.8)})
    keys['mouthFunnel'] = rig.moves({'L_LIPCORNER': (3, 0, 5), 'R_LIPCORNER': (3, 0, -5),
                                     'M_UPPERLIP_OUTER': (5, 3, 0), 'M_UPPERLIP_INNER': (4, 3, 0),
                                     'M_LOWERLIP_OUTER': (5, -3, 0), 'M_LOWERLIP_INNER': (4, -3, 0),
                                     'L_UPPERLIP_OUTER': (4, 2.5, 1.5), 'R_UPPERLIP_OUTER': (4, 2.5, -1.5),
                                     'L_UPPERLIP_INNER': (3.5, 2.5, 1.5), 'R_UPPERLIP_INNER': (3.5, 2.5, -1.5),
                                     'L_LOWERLIP_OUTER': (4, -2.5, 1.5), 'R_LOWERLIP_OUTER': (4, -2.5, -1.5),
                                     'L_LOWERLIP_INNER': (3.5, -2.5, 1.5), 'R_LOWERLIP_INNER': (3.5, -2.5, -1.5)})
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
    for s, name in (('L', 'Left'), ('R', 'Right')):
        keys['eyeBlink' + name] += rig.lid_move(s, 'eyeBlink' + name, upper_frac=0.8, lower_frac=0.2)
        keys['eyeWide' + name] += rig.lid_move(s, 'eyeWide' + name, upper_deg=9, lower_deg=-4)
        keys['eyeSquint' + name] += rig.lid_move(s, 'eyeSquint' + name, upper_frac=0.12, lower_frac=0.3)
        keys['cheekSquint' + name] += rig.lid_move(s, 'cheekSquint' + name, lower_frac=0.12)
        keys['mouthSmile' + name] += rig.lid_move(s, 'mouthSmile' + name, lower_frac=0.06)

    out = {k: v[h.weld] for k, v in keys.items()}
    out.update(eye_keys(h, rig))
    attach(h, rig, out)
    return out


def eye_keys(h, rig):
    out = {}
    for s, name, sx in (('L', 'Left', 1), ('R', 'Right', -1)):
        ball = np.isin(h.island, BALL[s])
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
    return out


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    hm = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(hm)
    dd = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, dd]) @ u.T
    return r, cb - r @ ca


def attach(h, rig, keys):
    """Cards (lashes, liner, brows: islands not welded to the skin) ride the skin under them. Liner and
    lash cards turn with the lid margin as one piece, else the lid skin's falloff stretches them over
    the whole closed lid."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    ball = np.isin(h.island, BALL['L'] + BALL['R'])
    cards = ~rig.soft_w[h.weld] & ~ball
    margin = np.concatenate([rig.margins[s][k] for s in 'LR' for k in ('upper', 'lower')])
    skin = np.where(rig.face_w)[0]
    m_tree, s_tree = cKDTree(h.wco[margin]), cKDTree(h.wco[skin])
    _, comp = connected_components(h.wgraph, directed=False)
    anchors = np.zeros(len(h.co), int)
    lid_card = np.zeros(len(h.co), bool)
    for c in np.unique(comp[h.weld[cards]]):
        sel = cards & (comp[h.weld] == c)
        dm, im = m_tree.query(h.co[sel])
        if np.median(dm) < 0.012:
            anchors[sel] = margin[im]
            lid_card[sel] = True
        else:
            anchors[sel] = skin[s_tree.query(h.co[sel])[1]]
    upper_card = lid_card & np.isin(anchors, np.concatenate([rig.margins[s]['upper'] for s in 'LR']))
    cards_idx = np.where(cards)[0]
    # cards follow a smooth blend of the skin offsets around them; lid cards also take the closing
    # turns rigidly (raising turns stay blended so the skin above does not poke through)
    skin_soft = np.where(rig.soft_w)[0]
    dist, near = cKDTree(h.wco[skin_soft]).query(h.co[cards_idx], k=8)
    wts = 1 / np.maximum(dist, 1e-4) ** 2
    wts /= wts.sum(1, keepdims=True)
    for name, d in keys.items():
        dw = np.zeros((rig.nw, 3))
        dw[h.weld[~cards]] = d[~cards]
        lid_ang = np.zeros(len(h.co))
        for side, amounts in rig.lid_specs.get(name, []):
            closing = {k: v for k, v in amounts.items() if k.endswith('_frac')}
            if not closing:
                continue
            dw -= rig.lid_move(side, **closing)
            sel = lid_card & ((h.co[:, 0] > 0) == (side == 'L'))
            au, al = rig.lid_angles(side, h.co[sel, 0], **closing)
            lid_ang[sel] += np.where(upper_card[sel], au, al)
        d[cards_idx] = np.einsum('ik,ikj->ij', wts, dw[skin_soft][near])
        for side in 'LR':
            ss = np.where(lid_card & (lid_ang != 0) & ((h.co[:, 0] > 0) == (side == 'L')))[0]
            if len(ss):
                d[ss] += rot_x(h.co[ss], EYE_C[side], lid_ang[ss]) - h.co[ss]

if __name__ == '__main__':
    h = fg.load(sys.argv[1])
    keys = build(h)
    np.save(sys.argv[2].replace('.npz', '_bag.npy'), np.isin(h.island, [BAG_U, BAG_L]))
    for k in sorted(keys):
        print('%-20s max %.1f mm' % (k, np.linalg.norm(keys[k], axis=1).max() * 1000))
    print(len(keys), 'keys')
    np.savez_compressed(sys.argv[2], **keys)
