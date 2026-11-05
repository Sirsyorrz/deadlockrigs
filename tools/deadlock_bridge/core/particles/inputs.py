"""The inputs particle functions read their settings through: float inputs (CParticleFloatInput), vector inputs
(CParticleVecInput) and transform inputs (CParticleTransformInput).

An input is called with the simulation and either a View of particles, giving one value per particle (an array), or
None for a value for the whole system (a float or a 3-vector), as emitters and renderers read them. Inputs that need
a particle read 0 for the whole system, as in the game.
"""
import math

import numpy as np

from . import fields as F
from . import noise
from .params import (BIAS_TYPES, Curve, Params, bias_from_parameter, clamp_between, fract, lerp, remap, saturate)


def _count(p) -> int:
    return p.n if p is not None else 1


def _per_particle(values, p):
    """A per particle array, or its single value for a system level read."""
    return values if p is not None else float(np.asarray(values).reshape(-1)[0])


# Float inputs ------------------------------------------------------------------------------------------------------

class Number:
    """A float input; the literal ones have a constant."""
    constant = None

    def __call__(self, sim, p=None):
        raise NotImplementedError


class Literal(Number):
    def __init__(self, value: float):
        self.constant = float(value)

    def __call__(self, sim, p=None):
        return self.constant


class Mapping:
    """How a float input turns what it reads into its value (m_nMapType): passed through, multiplied, remapped
    (optionally biased), a curve, a notch, rounded, or compared."""

    TYPES = {"PF_MAP_TYPE_DIRECT": 0, "PF_MAP_TYPE_MULT": 1, "PF_MAP_TYPE_REMAP": 2, "PF_MAP_TYPE_REMAP_BIASED": 3,
             "PF_MAP_TYPE_CURVE": 4, "PF_MAP_TYPE_NOTCHED": 5, "PF_MAP_TYPE_ROUND": 6, "PF_MAP_TYPE_MIN": 7,
             "PF_MAP_TYPE_MAX": 8, "PF_MAP_TYPE_MOD": 9}

    def __init__(self, p: Params):
        self.kind = p.enum("m_nMapType", self.TYPES, 0)
        self.looped = p.str("m_nInputMode") == "PF_INPUT_MODE_LOOPED"
        self.factor = p.float("m_flMultFactor")
        in0, in1 = p.float("m_flInput0"), p.float("m_flInput1")
        out0, out1 = p.float("m_flOutput0"), p.float("m_flOutput1")
        if self.kind == 2 and in0 > in1:
            in0, in1, out0, out1 = in1, in0, out1, out0
        self.input0, self.input1, self.output0, self.output1 = in0, in1, out0, out1
        self.bias_type = p.enum("m_nBiasType", BIAS_TYPES, 0)
        self.bias = p.float("m_flBiasParameter")
        self.curve = Curve(p.get("m_Curve"), self.looped) if self.kind == 4 else None
        self.notch = (p.float("m_flNotchedRangeMin"), p.float("m_flNotchedRangeMax"),
                      p.float("m_flNotchedOutputOutside"), p.float("m_flNotchedOutputInside"))
        self.round = p.str("m_nRoundType", "PF_ROUND_TYPE_NEAREST")
        self.compare = p.float("m_flCompareValue")

    def _wrap(self, x):
        if not self.looped or self.input1 == 0:
            return x
        return self.input1 * fract(x / self.input1)

    def __call__(self, x):
        kind = self.kind
        if kind == 0:
            return x
        x = np.asarray(x, np.float64)
        if kind == 1:
            return x * self.factor
        if kind == 2:
            v = self._wrap(x)
            if self.input0 == self.input1:
                return np.where(v >= self.input1, self.output1, self.output0)
            return clamp_between(lerp(self.output0, self.output1, remap(v, self.input0, self.input1)),
                                 self.output0, self.output1)
        if kind == 3:
            t = np.minimum(remap(self._wrap(x), self.input0, self.input1), 1.0)
            return clamp_between(lerp(self.output0, self.output1, bias_from_parameter(t, self.bias, self.bias_type)),
                                 self.output0, self.output1)
        if kind == 4:
            return self.curve(x)
        if kind == 5:
            low, high, outside, inside = self.notch
            return np.where((x >= low) & (x <= high), inside, outside)
        if kind == 6:
            return np.floor(x) if self.round == "PF_ROUND_TYPE_FLOOR" else np.ceil(x) if self.round == "PF_ROUND_TYPE_CEIL" else np.round(x)
        if kind == 7:
            return np.minimum(x, self.compare)
        if kind == 8:
            return np.maximum(x, self.compare)
        if kind == 9:
            return x if self.compare == 0 else np.fmod(x, self.compare)
        return x


class Random(Number):
    """A uniform (or biased) random value, the same for a particle all its life (constant mode) or drawn afresh
    every read (varying mode). Like a literal, it takes no mapping."""

    def __init__(self, p: Params, biased: bool):
        self.low = p.float("m_flRandomMin")
        self.high = p.float("m_flRandomMax", 1.0)
        self.sign_flip = p.bool("m_bHasRandomSignFlip")
        self.varying = p.str("m_nRandomMode") == "PF_RANDOM_MODE_VARYING"
        self.biased = biased
        self.bias = p.float("m_flBiasParameter") if biased else 0.0
        self.bias_type = p.enum("m_nBiasType", BIAS_TYPES, 0) if biased else 0
        self.ordinal = 0 if self.varying else p.next_ordinal()

    def __call__(self, sim, p=None):
        n = _count(p)
        ids = p.get("particle_id") if p is not None else np.zeros(1)
        draw = sim.random.next(n) if self.varying else sim.random.for_particle(ids, self.ordinal)
        if self.biased:
            draw = bias_from_parameter(draw, self.bias, self.bias_type)
        value = self.low + (self.high - self.low) * draw
        if self.sign_flip:
            sign = sim.random.next(n) if self.varying else sim.random.for_particle(ids, self.ordinal + 37)
            value = np.where(sign < 0.5, -value, value)
        return _per_particle(value, p)


class Mapped(Number):
    """An input that reads something and passes it through its mapping."""

    def __init__(self, p: Params, read, per_particle: bool):
        self.mapping = Mapping(p)
        self.read = read
        self.per_particle = per_particle

    def __call__(self, sim, p=None):
        if p is None:
            return 0.0 if self.per_particle else float(np.asarray(self.mapping(self.read(sim, None))))
        return self.mapping(self.read(sim, p))


class Noise(Number):
    """Value noise over a particle vector attribute (its position by default), scrolled over time, remapped from
    -1..1 to the output range."""

    def __init__(self, p: Params):
        self.field = p.int("m_nNoiseInputVectorAttribute", F.POSITION)
        self.control_point = p.int("m_nControlPoint")
        self.scale = p.float("m_flNoiseScale", 0.1)
        self.rate = p.vector("m_vecNoiseOffsetRate")
        self.offset = p.float("m_flNoiseOffset")
        self.low = p.float("m_flNoiseOutputMin")
        self.high = p.float("m_flNoiseOutputMax", 1.0)
        self.mapping = Mapping(p)

    def __call__(self, sim, p=None):
        if p is not None:
            points = p.field(self.field)
        else:
            points = sim.control_point(self.control_point).position[None, :] if self.control_point >= 0 else np.zeros((1, 3))
        sample = points * self.scale + self.rate * sim.age + self.offset
        value = self.low + (self.high - self.low) * (noise.value3(sample) * 0.5 + 0.5)
        return _per_particle(self.mapping(value), p)


def _cp_component(p: Params):
    cp, component = p.int("m_nControlPoint"), min(max(p.int("m_nVectorComponent"), 0), 2)

    def value(sim, _):
        point = sim.control_point(cp)
        if point.change_time < 0 and sim.root.unset_value:
            return sim.root.unset_value
        return point.position[component]
    return value


def _cp_change_age(p: Params):
    cp = p.int("m_nControlPoint")
    return lambda sim, _: sim.control_point(cp).change_age(sim)


def _cp_is_set(p: Params):
    cp = p.int("m_nControlPoint")
    return lambda sim, _: 1.0 if cp != -1 and sim.control_point(cp).change_time != -1.0 else 0.0


def _cp_distance(p: Params):
    cp = p.int("m_nControlPoint")
    return lambda sim, v: np.linalg.norm(v.get("position") - sim.control_point(cp).position, axis=1)


def _scalar_field(p: Params):
    field = p.int("m_nScalarAttribute", F.RADIUS)
    return lambda sim, v: v.field(field) if not F.is_vector(field) else np.zeros(v.n)


def _vector_component(p: Params):
    field, component = p.int("m_nVectorAttribute", F.COLOR), min(max(p.int("m_nVectorComponent"), 0), 2)
    return lambda sim, v: v.field(field)[:, component] if F.is_vector(field) else np.zeros(v.n)


def _number_normalized(sim, v):
    count = sim.particles.count
    divisor = max(count - 1, 1) if sim.version >= 12 else max(count, 1)
    start = v.rows.start or 0
    return (start + np.arange(v.n)) / divisor


def _highest_detail(p: Params) -> float:
    """The value for the highest detail level; tiers left out take the one below them."""
    value = p.float("m_flLOD0")
    for key in ("m_flLOD1", "m_flLOD2", "m_flLOD3"):
        if p.has(key):
            value = p.float(key)
    return value


def _rope_segment(p: Params):
    reverse = p.bool("m_bReverseOrder")
    def read(sim, v):
        data = v.get("rope_segment_data")
        return data[:, 2] - data[:, 0] if reverse else data[:, 0]
    return read


def read_number(params: Params, key: str, default) -> Number:
    if isinstance(default, Number):
        fallback = default
    else:
        fallback = Literal(default)
    if key not in params.data:
        return fallback
    value = params.data[key]
    if not isinstance(value, dict):
        from .. import kv3
        return Literal(kv3.as_float(value))
    kind = value.get("m_nType")
    p = params.nested(value)
    if kind is None:
        return Literal(p.float("m_flLiteralValue")) if "m_flLiteralValue" in value else fallback
    if kind == "PF_TYPE_LITERAL":
        # A literal is its value: the mapping fields it carries are left over from editing and aren't applied.
        return Literal(p.float("m_flLiteralValue"))
    if kind == "PF_TYPE_INVALID":
        return fallback
    if kind in ("PF_TYPE_RANDOM_UNIFORM", "PF_TYPE_RANDOM_BIASED"):
        return Random(p, kind == "PF_TYPE_RANDOM_BIASED")
    if kind == "PF_TYPE_PARTICLE_DETAIL_LEVEL":
        return Literal(_highest_detail(p))
    if kind == "PF_TYPE_PARTICLE_NOISE":
        return Noise(p)
    readers = {
        "PF_TYPE_COLLECTION_AGE": (lambda sim, _: sim.age, False),
        "PF_TYPE_ENDCAP_AGE": (lambda sim, _: sim.end_cap_age, False),
        "PF_TYPE_CONTROL_POINT_COMPONENT": (_cp_component(p), False),
        "PF_TYPE_CONTROL_POINT_CHANGE_AGE": (_cp_change_age(p), False),
        "PF_TYPE_CONTROL_POINT_IS_SET": (_cp_is_set(p), False),
        "PF_TYPE_CONTROL_POINT_SPEED": (lambda sim, _: 0.0, False),
        "PF_TYPE_PARTICLE_AGE": (lambda sim, v: v.get("age"), True),
        "PF_TYPE_PARTICLE_AGE_NORMALIZED": (lambda sim, v: v.normalized_age, True),
        "PF_TYPE_PARTICLE_FLOAT": (_scalar_field(p), True),
        "PF_TYPE_PARTICLE_INITIAL_FLOAT": (_scalar_field(p), True),
        "PF_TYPE_PARTICLE_VECTOR_COMPONENT": (_vector_component(p), True),
        "PF_TYPE_PARTICLE_INITIAL_VECTOR_COMPONENT": (_vector_component(p), True),
        "PF_TYPE_PARTICLE_SPEED": (lambda sim, v: np.linalg.norm(v.get("velocity"), axis=1), True),
        "PF_TYPE_PARTICLE_NUMBER": (lambda sim, v: v.get("unique_id"), True),
        "PF_TYPE_PARTICLE_NUMBER_NORMALIZED": (_number_normalized, True),
        "PF_TYPE_CONTROL_POINT_DISTANCE": (_cp_distance(p), True),
        "PF_TYPE_PARTICLE_ROPE_SEGMENT": (_rope_segment(p), True),
        "PF_TYPE_PARTICLE_ROPE_SEGMENT_NORMALIZED": (lambda sim, v: v.get("rope_segment_data")[:, 1], True),
    }
    if kind in readers:
        read, per_particle = readers[kind]
        return Mapped(p, read, per_particle)
    params.note(f"float input {kind}")
    return Literal(p.float("m_flLiteralValue"))


# Vector inputs -----------------------------------------------------------------------------------------------------

class Vector:
    """A vector input; the literal ones have a constant."""
    constant = None

    def __call__(self, sim, p=None):
        raise NotImplementedError


class LiteralVector(Vector):
    def __init__(self, value):
        self.constant = np.asarray(value, np.float64)

    def __call__(self, sim, p=None):
        return self.constant


class ParticleVector(Vector):
    """A particle vector attribute, scaled, or a control point's value when one is named."""

    def __init__(self, p: Params):
        self.field = p.int("m_nVectorAttribute", F.COLOR)
        self.scale = p.vector("m_vVectorAttributeScale", (1, 1, 1))
        self.cp = p.int("m_nControlPoint", -1)
        self.cp_scale = p.vector("m_vCPValueScale", (1, 1, 1))
        self.cp_offset = p.vector("m_vCPRelativePosition")
        self.cp_direction = p.vector("m_vCPRelativeDir")

    def __call__(self, sim, p=None):
        if self.cp >= 0:
            point = sim.control_point(self.cp)
            value = point.position * self.cp_scale + self.cp_offset
            if np.any(self.cp_direction):
                value = value + (point.forward - self.cp_direction if point.oriented else -self.cp_direction)
            return value
        if p is None:
            return np.zeros(3)
        return p.field(self.field) * self.scale


class FunctionVector(Vector):
    def __init__(self, read):
        self.read = read

    def __call__(self, sim, p=None):
        return self.read(sim, p)


class Components(Vector):
    """Three float inputs."""

    def __init__(self, p: Params):
        self.x = p.number("m_FloatComponentX", 0.0)
        self.y = p.number("m_FloatComponentY", 0.0)
        self.z = p.number("m_FloatComponentZ", 0.0)

    def __call__(self, sim, p=None):
        n = _count(p)
        out = np.empty((n, 3))
        out[:, 0] = self.x(sim, p)
        out[:, 1] = self.y(sim, p)
        out[:, 2] = self.z(sim, p)
        return out if p is not None else out[0]


class Interpolated(Vector):
    """A float input placed between two vectors, clamped or extrapolating."""

    def __init__(self, p: Params, clamped: bool):
        self.input = p.number("m_FloatInterp", 0.0)
        self.in0, self.in1 = p.float("m_flInterpInput0"), p.float("m_flInterpInput1", 1.0)
        self.out0, self.out1 = p.vector("m_vInterpOutput0"), p.vector("m_vInterpOutput1", (1, 1, 1))
        self.clamped = clamped

    def __call__(self, sim, p=None):
        t = remap(self.input(sim, p), self.in0, self.in1)
        if self.clamped:
            t = saturate(t)
        t = np.asarray(t, np.float64)
        return self.out0 + (self.out1 - self.out0) * (t[:, None] if t.ndim else t)


class Gradient(Vector):
    """A float input looked up in a colour gradient (stops authored 0 to 255)."""

    def __init__(self, p: Params):
        self.input = p.number("m_FloatInterp", 0.0)
        self.in0, self.in1 = p.float("m_flInterpInput0"), p.float("m_flInterpInput1", 1.0)
        stops = ((p.get("m_Gradient") or {}).get("m_Stops") or [])
        stops = sorted(stops, key=lambda s: float(s.get("m_flPosition", 0.0)))
        self.positions = np.array([float(s.get("m_flPosition", 0.0)) for s in stops] or [0.0])
        self.colors = np.array([[float(c) / 255.0 for c in (s.get("m_Color") or [255, 255, 255])[:3]] for s in stops]
                               or [[1.0, 1.0, 1.0]])

    def __call__(self, sim, p=None):
        t = np.atleast_1d(np.asarray(remap(self.input(sim, p), self.in0, self.in1), np.float64))
        out = np.stack([np.interp(t, self.positions, self.colors[:, c]) for c in range(3)], axis=1)
        return out if p is not None else out[0]


class RandomVector(Vector):
    """A random vector between two corners, drawn per component every read; with an attribute, added to it."""

    def __init__(self, p: Params, offset: bool):
        self.low = p.vector("m_vRandomMin")
        self.high = p.vector("m_vRandomMax")
        self.offset = offset
        self.field = p.int("m_nVectorAttribute", F.COLOR)
        self.scale = p.vector("m_vVectorAttributeScale", (1, 1, 1))

    def __call__(self, sim, p=None):
        n = _count(p)
        value = self.low if np.array_equal(self.low, self.high) else sim.random.between3(n, self.low, self.high)
        if p is None:
            value = np.asarray(value).reshape(-1, 3)[0] if np.ndim(value) == 2 else value
            return value
        if self.offset:
            value = p.field(self.field) * self.scale + value
        return value


def read_vector(params: Params, key: str, default) -> Vector:
    if isinstance(default, Vector):
        fallback = default
    else:
        fallback = LiteralVector(default)
    if key not in params.data:
        return fallback
    value = params.data[key]
    if not isinstance(value, dict) or "m_nType" not in value:
        return LiteralVector(params.vector(key))
    kind = value["m_nType"]
    p = params.nested(value)
    if kind == "PVEC_TYPE_LITERAL":
        return LiteralVector(p.vector("m_vLiteralValue"))
    if kind == "PVEC_TYPE_LITERAL_COLOR":
        return LiteralVector(p.color("m_LiteralColor", (0, 0, 0)))
    if kind in ("PVEC_TYPE_PARTICLE_VECTOR", "PVEC_TYPE_PARTICLE_INITIAL_VECTOR"):
        return ParticleVector(p)
    if kind == "PVEC_TYPE_PARTICLE_VELOCITY":
        return FunctionVector(lambda sim, v: v.get("velocity") if v is not None else np.zeros(3))
    if kind == "PVEC_TYPE_CP_VALUE":
        cp, scale = p.int("m_nControlPoint"), p.vector("m_vCPValueScale", (1, 1, 1))
        return FunctionVector(lambda sim, v: sim.control_point(cp).position * scale)
    if kind == "PVEC_TYPE_CP_DELTA":
        cp, other = p.int("m_nControlPoint"), p.int("m_nDeltaControlPoint")
        return FunctionVector(lambda sim, v: sim.control_point(cp).position - sim.control_point(other).position)
    if kind == "PVEC_TYPE_CP_RELATIVE_POSITION":
        cp, offset = p.int("m_nControlPoint"), p.vector("m_vCPRelativePosition")
        return FunctionVector(lambda sim, v: sim.control_point(cp).position + offset)
    if kind == "PVEC_TYPE_CP_RELATIVE_DIR":
        cp, direction = p.int("m_nControlPoint"), p.vector("m_vCPRelativeDir")

        def relative_direction(sim, v):
            point = sim.control_point(cp)
            return point.forward - direction if point.oriented and np.any(direction) else np.zeros(3)
        return FunctionVector(relative_direction)
    if kind == "PVEC_TYPE_FLOAT_COMPONENTS":
        return Components(p)
    if kind in ("PVEC_TYPE_FLOAT_INTERP_CLAMPED", "PVEC_TYPE_FLOAT_INTERP_OPEN"):
        return Interpolated(p, kind == "PVEC_TYPE_FLOAT_INTERP_CLAMPED")
    if kind == "PVEC_TYPE_FLOAT_INTERP_GRADIENT":
        return Gradient(p)
    if kind in ("PVEC_TYPE_RANDOM_UNIFORM", "PVEC_TYPE_RANDOM_UNIFORM_OFFSET"):
        return RandomVector(p, kind == "PVEC_TYPE_RANDOM_UNIFORM_OFFSET")
    params.note(f"vector input {kind}")
    return LiteralVector(p.vector("m_vLiteralValue"))


# Transform inputs --------------------------------------------------------------------------------------------------

class Transform:
    """A transform input: a control point's position and (optionally) rotation, or the identity."""

    def __init__(self, control_point: int | None, use_orientation: bool = True):
        self.control_point = control_point
        self.use_orientation = use_orientation

    def __call__(self, sim) -> tuple[np.ndarray, np.ndarray]:
        """(rotation 3x3, translation 3): world = rotation @ local + translation."""
        if self.control_point is None:
            return np.eye(3), np.zeros(3)
        point = sim.control_point(self.control_point)
        rotation = point.rotation if self.use_orientation and point.oriented else np.eye(3)
        return rotation, point.position

    def orientation(self, sim):
        """The control point's forward direction, or None when it has no orientation (or it isn't used)."""
        if self.control_point is None or not self.use_orientation:
            return None
        point = sim.control_point(self.control_point)
        return point.forward if point.oriented else None


def read_transform(params: Params, key: str, default) -> Transform:
    fallback = default if default is not None else Transform(0, True)
    if key not in params.data:
        return fallback
    value = params.data[key]
    if isinstance(value, dict):
        kind = value.get("m_nType", "PT_TYPE_CONTROL_POINT")
        p = params.nested(value)
        if kind in ("PT_TYPE_CONTROL_POINT", "PT_TYPE_CONTROL_POINT_RANGE"):
            return Transform(p.int("m_nControlPoint"), p.bool("m_bUseOrientation", True))
        if kind != "PT_TYPE_INVALID":
            params.note(f"transform input {kind}")
        return Transform(None)
    if value is None:
        return Transform(None)
    # Fields that became transform inputs still hold a bare control point number in older content.
    return Transform(params.int(key), True)


def transform_points(transform, points):
    """Points (n, 3) or (3,) from a transform's local space to the world."""
    rotation, translation = transform
    return np.asarray(points) @ rotation.T + translation


def transform_directions(transform, directions):
    return np.asarray(directions) @ transform[0].T


DEG = math.pi / 180.0
