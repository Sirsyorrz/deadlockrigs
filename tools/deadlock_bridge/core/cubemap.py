"""Cube textures to equirectangular images, sampled the way the game's cube lookups read them (D3D face layout,
looked up with the world space direction, Z up)."""
import numpy as np


def cube_face_uv(d: np.ndarray):
    """Directions (n, 3) -> face index, u, v (v down) by the D3D cube map rules."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.where((ax >= ay) & (ax >= az), np.where(x > 0, 0, 1),
                    np.where(ay >= az, np.where(y > 0, 2, 3), np.where(z > 0, 4, 5)))
    sc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-z, z, x, x, x], -x)
    tc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-y, -y, z, -z, -y], -y)
    ma = np.select([face < 2, face < 4], [ax, ay], az)
    ma = np.maximum(ma, 1e-12)
    return face, 0.5 * (sc / ma + 1), 0.5 * (tc / ma + 1)


def equirect_directions(width: int, height: int) -> np.ndarray:
    """World directions of a Blender equirectangular image's pixel centres, bottom row first (Blender's order)."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (0.5 - u) * 2 * np.pi
    theta = (v - 0.5) * np.pi
    phi, theta = np.meshgrid(phi, theta)
    c = np.cos(theta)
    return np.stack([c * np.cos(phi), c * np.sin(phi), np.sin(theta)], -1).reshape(-1, 3)


def sample_faces(faces: list[np.ndarray], d: np.ndarray) -> np.ndarray:
    """Bilinear lookups into six (h, w, c) faces, top row first."""
    face, u, v = cube_face_uv(d)
    channels = faces[0].shape[2]
    out = np.zeros((len(d), channels), np.float32)
    for f in range(6):
        sel = face == f
        if not sel.any():
            continue
        img = faces[f]
        h, w = img.shape[:2]
        x = np.clip(u[sel] * w - 0.5, 0, w - 1)
        y = np.clip(v[sel] * h - 0.5, 0, h - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        out[sel] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
                    + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return out


def to_equirect(faces: list[np.ndarray], width: int) -> np.ndarray:
    """(height, width, c) float32, bottom row first, ready for Blender's image.pixels."""
    height = width // 2
    d = equirect_directions(width, height)
    return sample_faces(faces, d).reshape(height, width, -1)
