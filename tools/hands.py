"""Finger poses for the rig. Pure mathutils so it runs inside Blender without the rig loaded.

Joint flexion is given as absolute angles (0 = straight finger) and converted to rotations on top of the
rest pose, which is already slightly curled. The thumb is solved: its base gets a free rotation and its two
hinges are searched so the tip reaches a target (e.g. across the curled index and middle fingers) without
passing through the other fingers."""
import math

from mathutils import Matrix, Quaternion, Vector

FINGERS = ('index', 'middle', 'ring', 'pinky')
FINGER_R = 0.0085
THUMB_R = 0.0095
PALM_R = 0.011

# (MCP, PIP, DIP) absolute flexion in degrees, spread in degrees (positive = towards the thumb side)
CURL = {
    'straight': (0, 0, 0),
    'relaxed': (22, 35, 18),
    'fist': (88, 100, 60),
    'grip': (58, 72, 40),
}
POSES = {
    'Relaxed': {'fingers': {f: 'relaxed' for f in FINGERS}, 'spread': {}, 'thumb': 'relaxed'},
    'Fist': {'fingers': {f: 'fist' for f in FINGERS}, 'spread': {}, 'thumb': 'fist'},
    'Point': {'fingers': {'index': 'straight', 'middle': 'fist', 'ring': 'fist', 'pinky': 'fist'}, 'spread': {},
              'thumb': 'over_middle'},
    'Peace': {'fingers': {'index': 'straight', 'middle': 'straight', 'ring': 'fist', 'pinky': 'fist'},
              'spread': {'index': 9, 'middle': -7}, 'thumb': 'over_ring'},
    'Thumbs Up': {'fingers': {f: 'fist' for f in FINGERS}, 'spread': {}, 'thumb': 'up'},
    'Open': {'fingers': {f: 'straight' for f in FINGERS}, 'spread': {'index': 7, 'ring': -6, 'pinky': -12},
             'thumb': 'open'},
    'Grip': {'fingers': {f: 'grip' for f in FINGERS}, 'spread': {}, 'thumb': 'grip'},
}


def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1.dot(d1), d2.dot(d2), d2.dot(r)
    if a < 1e-12 and e < 1e-12:
        return r.length
    if a < 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1.dot(r)
        if e < 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1.dot(d2)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return ((p1 + d1 * s) - (p2 + d2 * t)).length


def signed_angle(u, v, axis):
    return math.atan2(u.cross(v).dot(axis), u.dot(v))


class Hand:
    def __init__(self, rest, parent, tips, side):
        """rest: bone -> armature-space rest Matrix; parent: bone -> parent name; tips: finger -> rest tip point."""
        self.rest, self.parent, self.s = rest, parent, side
        s = side
        self.bones = [f'finger_{f}_{suffix}_{s}' for f in FINGERS for suffix in ('meta', '0', '1', '2')] + \
                     [f'finger_thumb_{i}_{s}' for i in range(3)]
        self.tips = tips
        P = lambda n: rest[n].translation.copy()  # noqa: E731
        self.P = P
        self.wrist = P(f'hand_{s}')
        self.r = (P(f'finger_index_0_{s}') - P(f'finger_pinky_0_{s}')).normalized()
        f = (P(f'finger_middle_0_{s}') - self.wrist).normalized()
        n = self.r.cross(f).normalized()
        bend = Vector()
        for fg in FINGERS:
            p0, p1, p2 = (P(f'finger_{fg}_{i}_{s}') for i in range(3))
            bend += (p2 - p1).normalized() - (p1 - p0).normalized()
        if n.dot(bend) < 0:
            n = -n
        self.n = n
        self.f = n.cross(self.r).normalized()
        if self.f.dot(f) < 0:
            self.f = -self.f

    def chain(self, fg):
        s = self.s
        if fg == 'thumb':
            pts = [self.P(f'finger_thumb_{i}_{s}') for i in range(3)] + [self.tips['thumb']]
        else:
            pts = [self.P(f'finger_{fg}_{k}_{s}') for k in ('meta', '0', '1', '2')] + [self.tips[fg]]
        return pts

    def local_quat(self, bone, world_rot):
        r = self.rest[bone].to_3x3()
        return (r.inverted() @ world_rot.to_matrix() @ r).to_quaternion()

    def finger_rots(self, fg, curl, spread):
        """World-frame (rest) rotations for meta, 0, 1, 2 of a finger."""
        pts = self.chain(fg)
        axis = (pts[4] - pts[1]).normalized().cross(self.n).normalized()
        rots = {}
        for j, target in enumerate(CURL[curl]):
            u = pts[j + 1] - pts[j]
            v = pts[j + 2] - pts[j + 1]
            rest_flex = math.degrees(signed_angle(u, v, axis))
            rots[j + 1] = Quaternion(axis, math.radians(target - rest_flex))
        sp = spread.get(fg, 0)
        if sp:
            fdir = (pts[4] - pts[1]).normalized()
            k = fdir.cross(self.r).normalized()
            rots[1] = Quaternion(k, math.radians(sp)) @ rots[1]
        return rots

    def fk(self, basis):
        """Posed armature-space matrices for the hand bones, hand itself at rest."""
        mats = {f'hand_{self.s}': self.rest[f'hand_{self.s}']}
        for b in self.bones:
            p = self.parent[b]
            q = basis.get(b, Quaternion())
            mats[b] = mats[p] @ self.rest[p].inverted() @ self.rest[b] @ q.to_matrix().to_4x4()
        return mats

    def posed_chain(self, mats, fg):
        s = self.s
        if fg == 'thumb':
            names = [f'finger_thumb_{i}_{s}' for i in range(3)]
        else:
            names = [f'finger_{fg}_{k}_{s}' for k in ('meta', '0', '1', '2')]
        pts = [mats[n].translation.copy() for n in names]
        last = names[-1]
        pts.append(mats[last] @ self.rest[last].inverted() @ self.tips[fg])
        return pts

    def solve(self, spec):
        s = self.s
        basis = {}
        for fg in FINGERS:
            for j, q in self.finger_rots(fg, spec['fingers'][fg], spec['spread']).items():
                name = f'finger_{fg}_{("meta", "0", "1", "2")[j]}_{s}'
                basis[name] = self.local_quat(name, q)
        mats = self.fk(basis)
        fingers = {fg: self.posed_chain(mats, fg) for fg in FINGERS}
        basis.update(self.solve_thumb(spec['thumb'], fingers, basis))
        return basis

    def thumb_target(self, mode, fingers):
        t = self.chain('thumb')
        n, r, f = self.n, self.r, self.f
        mid = lambda fg: (fingers[fg][2] + fingers[fg][3]) * 0.5  # noqa: E731
        reach = (t[3] - t[1]).length
        if mode == 'fist':
            return (mid('index') * 0.45 + mid('middle') * 0.55) + n * (FINGER_R + THUMB_R), True
        if mode == 'over_middle':
            return (mid('middle') * 0.6 + mid('ring') * 0.4) + n * (FINGER_R + THUMB_R), True
        if mode == 'over_ring':
            return (mid('ring') * 0.6 + mid('pinky') * 0.4) + n * (FINGER_R + THUMB_R), True
        if mode == 'grip':
            return (mid('index') + fingers['index'][2]) * 0.5 + n * (FINGER_R + THUMB_R + 0.004), True
        if mode == 'up':
            rest_dir = (t[3] - t[1]).normalized()
            return t[1] + rest_dir.slerp(r, 0.6) * reach, False
        if mode == 'open':
            return t[1] + (r * 0.75 + f * 0.6 - n * 0.15).normalized() * reach, False
        return t[3] + n * 0.012 + f * 0.004, False  # relaxed

    def solve_thumb(self, mode, fingers, finger_basis):
        s = self.s
        names = [f'finger_thumb_{i}_{s}' for i in range(3)]
        t = self.chain('thumb')
        target, wrap = self.thumb_target(mode, fingers)
        palm_aim = (self.P(f'finger_middle_0_{s}') + self.P(f'finger_pinky_0_{s}')) * 0.5
        hinge = (t[2] - t[1]).normalized().cross((palm_aim - t[1]).normalized()).normalized()
        frame = [self.n, self.r, self.f]
        obstacles = []
        for fg in FINGERS:
            c = fingers[fg]
            obstacles.append((c[0], c[1], PALM_R))
            obstacles += [(c[i], c[i + 1], FINGER_R) for i in range(1, 4)]

        def build(x):
            rv = frame[0] * x[0] + frame[1] * x[1] + frame[2] * x[2]
            q0 = Quaternion(rv.normalized(), rv.length) if rv.length > 1e-9 else Quaternion()
            b = {names[0]: self.local_quat(names[0], q0),
                 names[1]: self.local_quat(names[1], Quaternion(hinge, x[3])),
                 names[2]: self.local_quat(names[2], Quaternion(hinge, x[4]))}
            return b

        def cost(x):
            b = build(x)
            mats = self.fk(b)
            c = self.posed_chain(mats, 'thumb')
            e = (c[3] - target).length_squared * 1e4
            for p, q, rad in obstacles:
                for a_, b_ in ((c[1], c[2]), (c[2], c[3])):
                    d = seg_dist(a_, b_, p, q) - (rad + THUMB_R)
                    if d < 0:
                        e += (d * 1e3) ** 2
            if not wrap:
                e += (x[3] ** 2 + x[4] ** 2) * 0.5
            e += (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) * 0.02
            for v, lo, hi in ((x[3], -0.25, 1.3), (x[4], -0.25, 1.4)):
                if v < lo:
                    e += ((lo - v) * 10) ** 2
                if v > hi:
                    e += ((v - hi) * 10) ** 2
            if math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) > 1.4:
                e += 100
            return e

        best = [0.0] * 5
        best_c = cost(best)
        # coarse grid on the base rotation, then coordinate descent on all five
        g = [-0.9, -0.45, 0.0, 0.45, 0.9]
        for a in g:
            for b_ in g:
                for c_ in g:
                    for h in (0.0, 0.5, 1.0):
                        x = [a, b_, c_, h, h]
                        cc = cost(x)
                        if cc < best_c:
                            best, best_c = x, cc
        step = 0.2
        while step > 0.002:
            improved = False
            for i in range(5):
                for d in (step, -step):
                    x = best[:]
                    x[i] += d
                    cc = cost(x)
                    if cc < best_c:
                        best, best_c, improved = x, cc, True
            if not improved:
                step *= 0.5
        mats = self.fk(build(best))
        tip = self.posed_chain(mats, 'thumb')[3]
        self.report = (mode, round((tip - target).length * 1000, 1), round(best_c, 3))
        return build(best)


def tip_points(arm, meshes, side):
    """Fingertip estimate per finger: the weighted vertex farthest from the last joint along the finger.
    Falls back to the last joint plus the previous segment's length when no mesh is skinned to it."""
    rest = {b.name: b.matrix_local for b in arm.data.bones}
    tips = {}
    for fg in FINGERS + ('thumb',):
        last = f'finger_{fg}_2_{side}'
        prev = f'finger_{fg}_1_{side}'
        a, b = rest[prev].translation, rest[last].translation
        d = (b - a).normalized()
        bd = -1
        for body in meshes:
            gi = body.vertex_groups.find(last)
            if gi < 0:
                continue
            mw = arm.matrix_world.inverted() @ body.matrix_world
            for v in body.data.vertices:
                if any(g.group == gi and g.weight > 0.5 for g in v.groups):
                    t = (mw @ v.co - b).dot(d)
                    bd = max(bd, t)
        tips[fg] = b + d * (bd * 0.92 if bd > 0 else (b - a).length)
    return tips


def has_hand(arm, side):
    names = {b.name for b in arm.data.bones}
    need = [f'hand_{side}'] + [f'finger_{f}_{k}_{side}' for f in FINGERS for k in ('meta', '0', '1', '2')] + \
        [f'finger_thumb_{i}_{side}' for i in range(3)]
    return all(n in names for n in need)


def hand_poses(arm, meshes):
    """pose -> side -> bone -> quaternion, for the sides that have the full set of finger bones."""
    rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}
    parent = {b.name: b.parent.name for b in arm.data.bones if b.parent}
    out, reports = {name: {} for name in POSES}, []
    for s in [s for s in 'LR' if has_hand(arm, s)]:
        h = Hand(rest, parent, tip_points(arm, meshes, s), s)
        for name, spec in POSES.items():
            basis = h.solve(spec)
            out[name][s] = {b: tuple(round(v, 6) for v in q) for b, q in basis.items()}
            reports.append((s, name) + h.report)
    return out, reports
