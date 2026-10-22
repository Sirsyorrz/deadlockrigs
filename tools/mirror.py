"""Right-side bones whose rest frame is not the mirror image of the left one.

Many of the game's right-side bones are turned (usually 180 degrees) against the mirrored left bones, so Blender's
X-mirror posing and Paste Flipped, which flip the local rotation, bent the other side the wrong way. rig.py turns
those rest frames into the mirror of the left ones (positions stay), and anims.py converts the clips' keys to
match, so a clip still poses the bones exactly as in the game.
"""
import re

from mathutils import Matrix, Quaternion

FLIP_X = Matrix.Diagonal((-1, 1, 1))
TOL = 1e-3


def turns(rest):
    """rest: bone -> armature-space rest Matrix. Returns right bone -> 3x3 turn C, with rest' = rest @ C."""
    out = {}
    for left, m in rest.items():
        right = re.sub(r'_L(_|$)', r'_R\1', left)
        if right == left or right not in rest:
            continue
        c = rest[right].to_3x3().normalized().transposed() @ FLIP_X @ m.to_3x3().normalized() @ FLIP_X
        if max(abs(c[i][j] - (i == j)) for i in range(3) for j in range(3)) > TOL:
            out[right] = c
    return out


def turn_rest(edit_bones, tns):
    for b in edit_bones:
        b.use_connect = False
    for name, c in tns.items():
        b = edit_bones[name]
        m = (b.matrix.to_3x3() @ c).to_4x4()
        m.translation = b.head
        b.matrix = m


def _remap(cb, path, size, mat, default):
    """Replaces the channels of path by mat @ (their values), on the union of their key frames."""
    fcs = [cb.fcurves.find(path, index=i) for i in range(size)]
    if not any(fcs):
        return
    frames = sorted({k.co[0] for fc in fcs if fc for k in fc.keyframe_points})
    interp = next(fc for fc in fcs if fc).keyframe_points[0].interpolation
    old = [[fc.evaluate(f) if fc else default[i] for i, fc in enumerate(fcs)] for f in frames]
    for i in range(size):
        fc = fcs[i] or cb.fcurves.new(path, index=i)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        co = [x for f, v in zip(frames, old) for x in (f, sum(mat[i][k] * v[k] for k in range(size)))]
        fc.keyframe_points.foreach_set('co', co)
        for k in fc.keyframe_points:
            k.interpolation = interp
        fc.update()


def turn_keys(cb, tns):
    """With rest' = rest @ C the same pose has basis' = C^-1 @ basis @ C. Returns bones whose keyed scale is not
    uniform and whose turn is not about one axis, which the turned frame can't carry exactly."""
    uneven = []
    for bone, c in tns.items():
        pre = f'pose.bones["{bone}"].'
        cq = c.to_quaternion()
        ci = cq.inverted()
        cols = [ci @ Quaternion(e) @ cq for e in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))]
        _remap(cb, pre + 'rotation_quaternion', 4, [[cols[k][i] for k in range(4)] for i in range(4)], (1, 0, 0, 0))
        _remap(cb, pre + 'location', 3, c.transposed(), (0, 0, 0))
        sc = [cb.fcurves.find(pre + 'scale', index=i) for i in range(3)]
        if any(sc) and any(abs(c[i][j]) > TOL for i in range(3) for j in range(3) if i != j):
            frames = {k.co[0] for fc in sc if fc for k in fc.keyframe_points}
            for f in frames:
                v = [fc.evaluate(f) if fc else 1.0 for fc in sc]
                if max(v) - min(v) > 1e-4:
                    uneven.append(bone)
                    break
    return uneven
