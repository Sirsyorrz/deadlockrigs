"""Compiled materials (vmat_c): the shader and its parameters, plus the colour matrices Deadlock's shaders apply."""
import math

import numpy as np

from . import kv3
from .resource import ResourceFile


class CompiledMaterial:
    def __init__(self, data: bytes):
        root = ResourceFile(data).keyvalues("DATA")
        if root is None:
            raise ValueError("material has no DATA block")
        self.name = root.get("m_materialName", "")
        self.shader = root.get("m_shaderName", "")
        self.ints = {e.get("m_name", ""): kv3.as_int(e.get("m_nValue")) for e in root.get("m_intParams") or []}
        self.floats = {e.get("m_name", ""): kv3.as_float(e.get("m_flValue")) for e in root.get("m_floatParams") or []}
        self.vectors = {e.get("m_name", ""): _vector(e.get("m_value")) for e in root.get("m_vectorParams") or []}
        self.textures = {e.get("m_name", ""): (e.get("m_pValue") or "").replace("\\", "/") for e in root.get("m_textureParams") or []}
        self.int_attributes = {e.get("m_name", ""): kv3.as_int(e.get("m_nValue")) for e in root.get("m_intAttributes") or []}
        self.float_attributes = {e.get("m_name", ""): kv3.as_float(e.get("m_flValue")) for e in root.get("m_floatAttributes") or []}
        self.string_attributes = {e.get("m_name", ""): e.get("m_value", "") for e in root.get("m_stringAttributes") or []}
        self.dynamic = {}
        for e in (root.get("m_dynamicParams") or []) + (root.get("m_dynamicTextureParams") or []):
            if isinstance(e.get("m_value"), (bytes, bytearray)):
                self.dynamic[e.get("m_name", "")] = bytes(e["m_value"])

    def combo(self, name: str) -> bool:
        return self.ints.get(name, 0) != 0

    def int(self, name: str, fallback: int = 0) -> int:
        return self.ints.get(name, fallback)

    def float(self, name: str, fallback: float = 0.0) -> float:
        return self.floats.get(name, fallback)

    def vector(self, name: str, fallback=(0.0, 0.0, 0.0, 0.0)):
        v = self.vectors.get(name)
        return tuple(v) if v is not None else tuple(fallback) + (0.0,) * (4 - len(fallback))

    def texture(self, name: str) -> str | None:
        path = self.textures.get(name)
        return path or None


def _vector(value) -> tuple:
    values = [kv3.as_float(x) for x in (value or [])[:4]]
    return tuple(values + [0.0] * (4 - len(values)))


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


# Colour matrices (ported from ValveResourceFormat via the s&box mount). System.Numerics matrices are row-vector
# matrices, so they are composed left to right here and transposed at the end like the original.

_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
_LUMINANCE = _LUMINANCE / np.linalg.norm(_LUMINANCE)


def _translation(v):
    m = np.eye(4)
    m[3, :3] = v
    return m


def _scale(v):
    m = np.eye(4)
    if np.isscalar(v):
        m[0, 0] = m[1, 1] = m[2, 2] = v
    else:
        m[0, 0], m[1, 1], m[2, 2] = v[0], v[1], v[2]
    return m


def _axis_angle(axis, angle):
    # System.Numerics.Matrix4x4.CreateFromAxisAngle (row-vector convention).
    x, y, z = axis
    s, c = math.sin(angle), math.cos(angle)
    xx, yy, zz, xy, xz, yz = x * x, y * y, z * z, x * y, x * z, y * z
    m = np.eye(4)
    m[0, 0] = xx + c * (1 - xx)
    m[0, 1] = xy - c * xy + s * z
    m[0, 2] = xz - c * xz - s * y
    m[1, 0] = xy - c * xy - s * z
    m[1, 1] = yy + c * (1 - yy)
    m[1, 2] = yz - c * yz + s * x
    m[2, 0] = xz - c * xz + s * y
    m[2, 1] = yz - c * yz - s * x
    m[2, 2] = zz + c * (1 - zz)
    return m


def _luminance_rotation():
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(_LUMINANCE, z)
    angle = math.atan2(np.linalg.norm(cross), np.dot(_LUMINANCE, z))
    return _axis_angle(cross / np.linalg.norm(cross), angle)


def color_correct(csb, pivot=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Contrast, saturation and brightness as rows of a 3x4 matrix applied as dot(row, (rgb, 1))."""
    rotation = _luminance_rotation()
    pivot = np.array(pivot, dtype=np.float64)
    result = _translation(-pivot) @ _scale(csb[0]) @ _translation(pivot)
    result = result @ _scale(csb[2])
    result = result @ _scale(_LUMINANCE)
    result = result @ rotation
    result = result @ _scale((csb[1], csb[1], 1.0))
    result = result @ rotation.T
    result = result @ _scale(1.0 / _LUMINANCE)
    return result.T[:3, :]


def tint_matrix(linear_tint, strength: float, mode: int) -> np.ndarray:
    """A tint in one of the shader's modes: 0 multiplies, 1 preserves luminance, 2 is mod2x."""
    t = np.array(linear_tint, dtype=np.float64)
    if mode == 1:
        mx, mn = t.max(), t.min()
        saturation = 0.0 if mx == 0 else (mx - mn) / mx
        rotation = _luminance_rotation()
        lum_scale = _scale(_LUMINANCE)
        gray = (np.append(t, 1.0) @ (lum_scale @ rotation))[:3]
        desaturate = 1.0 - saturation
        factor = 1.0 - saturation * saturation * strength
        result = lum_scale @ rotation
        result = result @ _translation((-gray[0], -gray[1], 0.0))
        result = result @ _scale((desaturate, desaturate, factor))
        result = result @ _translation((gray[0], gray[1], (1.0 - factor) * gray[2]))
        result = result @ rotation.T
        result = result @ _scale(1.0 / _LUMINANCE)
        return result.T[:3, :]
    if mode == 2:
        return _scale(t * 2.0).T[:3, :]
    return _scale(t).T[:3, :]
