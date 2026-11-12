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


def joint_ids(sim, joints) -> np.ndarray:
    """(n, 4) ids for rows of four bone names (-1 for an empty slot), from the tree's own name table."""
    table = sim.root.bone_ids
    return np.array([[table.setdefault(name, len(table)) if name else -1 for name in row] for row in joints],
                    np.float64).reshape(-1, 4)


def skin_matrices(sim, model, ids, weights):
    """Each particle's skinning matrix (n, 4, 4: rest model space to simulation space) for its bone ids (n, 4) and
    weights (n, 4), and whether any of its bones was found. As the game: the last weight is 1 minus the first three,
    a particle with one bone found takes that bone whole, others a plain weighted sum."""
    n = len(ids)
    out = np.tile(np.eye(4), (n, 1, 1))
    if model is None or not model.get("bind") or n == 0:
        return out, np.zeros(n, bool)
    names = {i: name for name, i in sim.root.bone_ids.items()}
    bones, bind = model["bones"], model["bind"]
    skins = {}
    for i in np.unique(ids[ids >= 0]).astype(np.int64):
        name = names.get(int(i))
        if name in bones and name in bind:
            skins[int(i)] = bones[name] @ np.linalg.inv(bind[name])
    weights = np.asarray(weights, np.float64).copy()
    weights[:, 3] = np.clip(1.0 - weights[:, :3].sum(axis=1), 0.0, 1.0)
    found = np.zeros((n, 4), bool)
    mats = np.zeros((n, 4, 4, 4))
    for k in range(4):
        for i, matrix in skins.items():
            rows = ids[:, k] == i
            found[rows, k] = True
            mats[rows, k] = matrix
    count = found.sum(axis=1)
    weights = np.where(found, weights, 0.0)
    weights = np.where((count == 1)[:, None], found.astype(np.float64), weights)
    blended = np.einsum("nk,nkij->nij", weights, mats)
    blended[:, 3] = (0.0, 0.0, 0.0, 1.0)
    valid = count > 0
    out[valid] = blended[valid]
    return out, valid
