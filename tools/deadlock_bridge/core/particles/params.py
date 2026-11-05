"""Reading a particle definition's KeyValues3 blocks as the game's parser does: typed values with defaults, enums by
name or by number, and the curves and remaps float inputs apply. Shared maths (bias curves, remaps) lives here too.
"""
import math

import numpy as np

from .. import kv3


class Params:
    """One block of a particle function (or a block nested in one). Inputs that draw a random value per particle
    take an ordinal from a counter shared by every block of the function, so two inputs of one function read
    different slots of the random table."""

    def __init__(self, data: dict | None, version: int = 0, ordinal: list | None = None, notes=None):
        self.data = data if isinstance(data, dict) else {}
        self.version = version
        self._ordinal = ordinal if ordinal is not None else [0]
        self._notes = notes

    def nested(self, data) -> "Params":
        return Params(data, self.version, self._ordinal, self._notes)

    def function(self, data) -> "Params":
        """A particle function's block: its inputs number their random draws from 0."""
        return Params(data, self.version, None, self._notes)

    def note(self, what: str):
        """Records something the port doesn't support (counted per kind for the import report)."""
        if self._notes is not None:
            self._notes[what] += 1

    def next_ordinal(self) -> int:
        self._ordinal[0] += 1
        return self._ordinal[0] - 1

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def float(self, key: str, default: float = 0.0) -> float:
        value = self.data.get(key)
        if value is None:
            return default
        if isinstance(value, dict):
            # Newer content authors some plain floats as float inputs; their literal is what a plain read gets.
            return kv3.as_float(value.get("m_flLiteralValue"), default)
        return kv3.as_float(value, default)

    def int(self, key: str, default: int = 0) -> int:
        value = self.data.get(key)
        if value is None:
            return default
        if isinstance(value, dict):
            return int(kv3.as_float(value.get("m_flLiteralValue"), default))
        if isinstance(value, float):
            return int(value)
        return kv3.as_int(value, default)

    def bool(self, key: str, default: bool = False) -> bool:
        value = self.data.get(key)
        if value is None:
            return default
        return kv3.as_bool(value, default)

    def str(self, key: str, default: str = "") -> str:
        value = self.data.get(key)
        return value if isinstance(value, str) else default

    def vector(self, key: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
        value = self.data.get(key)
        if isinstance(value, dict):
            value = value.get("m_vLiteralValue")
        if isinstance(value, str):
            value = value.split()
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return np.array([kv3.as_float(v) for v in value[:3]], np.float64)
        return np.array(default, np.float64)

    def color(self, key: str, default=(1.0, 1.0, 1.0)) -> np.ndarray:
        """An RGB colour authored 0 to 255 (gamma space), as 0 to 1."""
        value = self.data.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return np.array([kv3.as_float(v) for v in value[:3]], np.float64) / 255.0
        return np.array(default, np.float64)

    def enum(self, key: str, names: dict, default: int) -> int:
        """An enum authored by member name or by number; names maps member names to values."""
        value = self.data.get(key)
        if value is None:
            return default
        if isinstance(value, str):
            if value in names:
                return names[value]
            try:
                return int(value)
            except ValueError:
                return default
        return kv3.as_int(value, default)

    def array(self, key: str) -> list["Params"]:
        return [self.nested(item) for item in self.data.get(key) or [] if isinstance(item, dict)]

    def functions(self, key: str) -> list["Params"]:
        return [self.function(item) for item in self.data.get(key) or [] if isinstance(item, dict)]

    # Inputs (built by inputs.py, imported late to keep this module free of the input classes) -------------------

    def number(self, key: str, default=0.0):
        from . import inputs
        return inputs.read_number(self, key, default)

    def vec(self, key: str, default=(0.0, 0.0, 0.0)):
        from . import inputs
        return inputs.read_vector(self, key, default)

    def transform(self, key: str, default=None):
        from . import inputs
        return inputs.read_transform(self, key, default)


def enum_names(*names: str, start: int = 0) -> dict:
    return {name: start + i for i, name in enumerate(names)}


SET_METHODS = enum_names("PARTICLE_SET_REPLACE_VALUE", "PARTICLE_SET_SCALE_INITIAL_VALUE",
                         "PARTICLE_SET_ADD_TO_INITIAL_VALUE", "PARTICLE_SET_RAMP_CURRENT_VALUE",
                         "PARTICLE_SET_SCALE_CURRENT_VALUE", "PARTICLE_SET_ADD_TO_CURRENT_VALUE")
SET_REPLACE, SET_SCALE_INITIAL, SET_ADD_INITIAL, SET_RAMP_CURRENT, SET_SCALE_CURRENT, SET_ADD_CURRENT = range(6)

END_CAP = {"PARTICLE_ENDCAP_ALWAYS_ON": -1, "PARTICLE_ENDCAP_ENDCAP_OFF": 0, "PARTICLE_ENDCAP_ENDCAP_ON": 1}
BIAS_TYPES = {"PF_BIAS_TYPE_INVALID": -1, "PF_BIAS_TYPE_STANDARD": 0, "PF_BIAS_TYPE_GAIN": 1,
              "PF_BIAS_TYPE_EXPONENTIAL": 2}
BIAS_STANDARD, BIAS_GAIN, BIAS_EXPONENTIAL = 0, 1, 2


# Maths -------------------------------------------------------------------------------------------------------------

def saturate(x):
    return np.clip(x, 0.0, 1.0)


def remap(x, a, b):
    """(x - a) / (b - a), or b where the range is empty (as the game's Remap)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    span = b - a
    return np.where(span == 0, b, (np.asarray(x, np.float64) - a) / np.where(span == 0, 1.0, span))


def remap_clamped(x, a, b, out_a, out_b):
    """The game's RemapValClamped: x from [a, b] to [out_a, out_b], held at the ends. An empty input range is a
    threshold, inclusive on the high side."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    x = np.asarray(x, np.float64)
    t = np.where(a == b, (x >= b).astype(np.float64), saturate(remap(x, a, b)))
    return out_a + (np.asarray(out_b, np.float64) - out_a) * t


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(edge0, edge1, x):
    t = saturate(remap(x, edge0, edge1))
    return t * t * (3.0 - 2.0 * t)


def clamp_between(x, a, b):
    """Clamps between two bounds given in either order."""
    return np.clip(x, np.minimum(a, b), np.maximum(a, b))


def bias(x, amount):
    """Schlick's bias: amount 0.5 leaves x as it is."""
    if amount <= 0.0:
        return np.zeros_like(np.asarray(x, np.float64)) if np.ndim(x) else 0.0
    if amount >= 1.0:
        return np.ones_like(np.asarray(x, np.float64)) if np.ndim(x) else 1.0
    return x / ((1.0 - x) * (1.0 / amount - 2.0) + 1.0)


def bias_from_parameter(x, parameter: float, kind: int):
    """A float input's bias curve, from its -1 to 1 parameter: exponential, standard bias or gain."""
    x = np.asarray(x, np.float64)
    if kind == BIAS_EXPONENTIAL:
        exponent = 1.0 - min(max(parameter, 0.0), 1.0) if parameter >= 0 else 20.0 - min(max(parameter + 1.0, 0.0), 1.0) * 19.0
        if exponent <= 0.0:
            return np.ones_like(x)
        return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, np.power(np.clip(x, 0, 1), min(exponent, 20.0))))
    if kind not in (BIAS_STANDARD, BIAS_GAIN):
        return np.zeros_like(x)
    amount = min(max((parameter + 1.0) * 0.5, 0.0), 1.0)
    if kind == BIAS_GAIN:
        return np.where(x < 0.5, bias(x + x, amount) * 0.5, 1.0 - bias(2.0 - x - x, amount) * 0.5)
    return bias(x, amount)


def fract(x):
    return x - np.floor(x)


class Curve:
    """A float curve (m_Curve): Hermite segments between keys, by the keys' slopes, or straight lines for linear
    tangents. Input outside the domain is clamped (or wrapped when looped), and output is held to the domain's
    value range."""

    def __init__(self, data: dict, looped: bool):
        data = data or {}
        self.looped = looped
        mins = data.get("m_vDomainMins") or [0.0, 0.0]
        maxs = data.get("m_vDomainMaxs") or [0.0, 0.0]
        self.x_min, self.y_min = kv3.as_float(mins[0]), kv3.as_float(mins[1])
        self.x_max, self.y_max = kv3.as_float(maxs[0]), kv3.as_float(maxs[1])
        keys = data.get("m_spline") or []
        tangents = data.get("m_tangents") or []
        count = min(len(keys), len(tangents)) if tangents else len(keys)
        self.x = np.array([kv3.as_float(k.get("x")) for k in keys[:count]], np.float64)
        self.y = np.array([kv3.as_float(k.get("y")) for k in keys[:count]], np.float64)
        slope_in = np.array([kv3.as_float(k.get("m_flSlopeIncoming")) for k in keys[:count]], np.float64)
        slope_out = np.array([kv3.as_float(k.get("m_flSlopeOutgoing")) for k in keys[:count]], np.float64)
        linear_in = np.array([(t or {}).get("m_nIncomingTangent") == "CURVE_TANGENT_LINEAR" for t in tangents[:count]], bool)
        linear_out = np.array([(t or {}).get("m_nOutgoingTangent") == "CURVE_TANGENT_LINEAR" for t in tangents[:count]], bool)
        if count >= 2:
            chord = np.diff(self.y) / np.maximum(np.diff(self.x), 1e-9)
            if linear_out.size:
                slope_out[:-1] = np.where(linear_out[:-1], chord, slope_out[:-1])
            if linear_in.size:
                slope_in[1:] = np.where(linear_in[1:], chord, slope_in[1:])
        self.slope_in = slope_in
        self.slope_out = slope_out

    def __call__(self, value):
        x = np.asarray(value, np.float64)
        if self.x.size < 2:
            result = np.zeros_like(x)
            return result if result.ndim else float(result)
        if self.looped and self.x_max > self.x_min:
            x = self.x_min + fract((x - self.x_min) / (self.x_max - self.x_min)) * (self.x_max - self.x_min)
        else:
            x = np.minimum(np.maximum(x, self.x_min), self.x_max)
        segment = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, self.x.size - 2)
        x0, x1 = self.x[segment], self.x[segment + 1]
        y0, y1 = self.y[segment], self.y[segment + 1]
        h = np.maximum(x1 - x0, 1e-9)
        t = np.clip((x - x0) / h, 0.0, 1.0)
        t2, t3 = t * t, t * t * t
        y = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * self.slope_out[segment]
             + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * self.slope_in[segment + 1])
        y = np.where(x <= self.x[0], self.y[0], np.where(x >= self.x[-1], self.y[-1], y))
        y = np.minimum(np.maximum(y, self.y_min), self.y_max)
        return y if y.ndim else float(y)


def degrees(value):
    return value * (math.pi / 180.0)
