"""The model a control point carries (set by the host): hitboxes by set name and bone transforms in the simulation's
space. Used by the functions that put particles on a model and keep them on it."""
import numpy as np

SHAPE_BOX = 0


def model_of(sim, control_point: int):
    return sim.control_point(control_point).model if control_point >= 0 else None


def hitbox_list(model, set_name: str, use_bones: bool) -> list:
    """The boxes particles pick from: a hitbox set (the named one, else "default", else the first), or with
    ``use_bones`` one point box per bone, in name order. A particle's hitbox index is its place in this list."""
    if use_bones:
        return [{"bone": name, "mins": (0.0, 0.0, 0.0), "maxs": (0.0, 0.0, 0.0), "radius": 0.0, "shape": SHAPE_BOX}
                for name in sorted(model["bones"])]
    sets = model.get("hitboxes") or {}
    for key in (set_name, "default"):
        if sets.get(key):
            return sets[key]
    return next((boxes for boxes in sets.values() if boxes), [])


def matrices(bones, boxes) -> np.ndarray:
    """(len(boxes), 4, 4): each box's bone to simulation space (identity for a bone the model lacks)."""
    out = np.tile(np.eye(4), (len(boxes), 1, 1))
    for i, box in enumerate(boxes):
        matrix = bones.get(box["bone"])
        if matrix is not None:
            out[i] = matrix
    return out


def transform_points(matrix, points):
    """Points (n, 3) through per point matrices (n, 4, 4)."""
    return np.einsum("nij,nj->ni", matrix[:, :3, :3], points) + matrix[:, :3, 3]


def box_arrays(boxes):
    mins = np.array([b["mins"] for b in boxes], np.float64).reshape(-1, 3)
    maxs = np.array([b["maxs"] for b in boxes], np.float64).reshape(-1, 3)
    radius = np.array([b.get("radius", 0.0) for b in boxes], np.float64)
    shape = np.array([b.get("shape", SHAPE_BOX) for b in boxes], np.int64)
    return mins, maxs, radius, shape


def local_points(arrays, index, fraction, spread):
    """Bone space points for boxes ``index`` at ``fraction`` of their extent per axis: inside a box between its
    corners; for a sphere or capsule along the segment from mins to maxs, pushed out up to its radius by the cube
    ``spread`` (n, 3 in 0..1) squashed into a ball."""
    mins, maxs, radius, shape = arrays
    low, high = mins[index], maxs[index]
    box = low + (high - low) * fraction
    cube = spread * 2.0 - 1.0
    norm = np.linalg.norm(cube, axis=1, keepdims=True)
    ball = cube * (np.abs(cube).max(axis=1, keepdims=True) / np.maximum(norm, 1e-9))
    round_ = low + (high - low) * fraction[:, :1] + ball * radius[index][:, None]
    return np.where((shape[index] != SHAPE_BOX)[:, None], round_, box)
