"""Lattice noise for the noise driven functions: value noise (a hashed lattice, trilinearly interpolated, -1 to 1)
as the game's operators sample it, and curl noise built from three channels of it."""
import numpy as np

_CHANNELS = (np.zeros(3), np.array([100000.5, 300000.25, 9000001.0]), np.array([210000.75, 610001.0, 18100001.0]))


def _hash(ix, iy, iz):
    """A float in [0, 1) per integer lattice point."""
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def value3(points) -> np.ndarray:
    """Value noise at (n, 3) points, -1 to 1."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    base = np.floor(p)
    f = p - base
    i = base.astype(np.int64)
    x0, y0, z0 = i[:, 0], i[:, 1], i[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    def corner(dx, dy, dz):
        return _hash(x0 + dx, y0 + dy, z0 + dz)

    c00 = corner(0, 0, 0) + (corner(1, 0, 0) - corner(0, 0, 0)) * fx
    c10 = corner(0, 1, 0) + (corner(1, 1, 0) - corner(0, 1, 0)) * fx
    c01 = corner(0, 0, 1) + (corner(1, 0, 1) - corner(0, 0, 1)) * fx
    c11 = corner(0, 1, 1) + (corner(1, 1, 1) - corner(0, 1, 1)) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return (c0 + (c1 - c0) * fz - 0.5) * 2.0


def vector3(points) -> np.ndarray:
    """Three decorrelated channels of value noise, (n, 3)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    return np.stack([value3(p + offset) for offset in _CHANNELS], axis=1)


def curl3(points, step: float = 0.05) -> np.ndarray:
    """The curl of a three channel noise field (divergence free swirls), by central differences."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    dx = np.array([step, 0, 0])
    dy = np.array([0, step, 0])
    dz = np.array([0, 0, step])
    ddx = (vector3(p + dx) - vector3(p - dx)) / (2 * step)
    ddy = (vector3(p + dy) - vector3(p - dy)) / (2 * step)
    ddz = (vector3(p + dz) - vector3(p - dz)) / (2 * step)
    return np.stack([ddy[:, 2] - ddz[:, 1], ddz[:, 0] - ddx[:, 2], ddx[:, 1] - ddy[:, 0]], axis=1) * 0.25
