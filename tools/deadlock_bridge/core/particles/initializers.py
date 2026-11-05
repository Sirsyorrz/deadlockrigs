"""Initializers: run once on each particle as it is born, in definition order, setting its starting attributes.
Each works on the particles a step emits together (a View of them)."""
import math

import numpy as np

from . import fields as F
from . import noise
from .inputs import Transform, transform_directions, transform_points
from .params import (SET_ADD_CURRENT, SET_ADD_INITIAL, SET_METHODS, SET_RAMP_CURRENT, SET_REPLACE,
                     SET_SCALE_CURRENT, SET_SCALE_INITIAL, Params, bias, lerp, remap_clamped, saturate)
from .system import Function

INITIALIZERS = {}


def initializer(name):
    def register(cls):
        INITIALIZERS[name] = cls
        return cls
    return register


class Initializer(Function):
    def initialize(self, sim, v):
        raise NotImplementedError


def column(values, n):
    """A per particle value as an (n,) array."""
    return np.broadcast_to(np.asarray(values, np.float64), (n,))


def rows3(values, n):
    """A per particle vector as an (n, 3) array."""
    return np.broadcast_to(np.asarray(values, np.float64), (n, 3))


def set_at_spawn(value, current, method: int, dt: float):
    """A set method applied at spawn, where the initial value is the current one."""
    if method == SET_REPLACE:
        return value
    if method in (SET_SCALE_INITIAL, SET_SCALE_CURRENT):
        return value * current
    if method in (SET_ADD_INITIAL, SET_ADD_CURRENT):
        return value + current
    if method == SET_RAMP_CURRENT:
        return current + value * dt
    return value


@initializer("C_INIT_InitFloat")
class InitFloat(Initializer):
    """Sets a float attribute from an input (angles authored in degrees), by a set method, blended by a strength."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nOutputField", F.RADIUS)
        self.value = p.number("m_InputValue", 0.0)
        self.input_strength = p.number("m_InputStrength", 1.0)
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)

    def initialize(self, sim, v):
        if F.is_vector(self.field):
            return
        value = column(self.value(sim, v), v.n)
        if self.field in F.ANGLES and self.method not in (SET_SCALE_INITIAL, SET_SCALE_CURRENT):
            value = np.radians(value)
        current = v.field(self.field)
        target = set_at_spawn(value, current, self.method, sim.frame_time)
        v.set_field(self.field, lerp(current, target, self.input_strength(sim, v)))


@initializer("C_INIT_InitVec")
class InitVec(Initializer):
    """Sets a vector attribute from an input."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nOutputField", F.COLOR)
        self.value = p.vec("m_InputValue")
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)

    def initialize(self, sim, v):
        if not F.is_vector(self.field):
            return
        value = rows3(self.value(sim, v), v.n)
        v.set_field(self.field, set_at_spawn(value, v.field(self.field), self.method, sim.frame_time))


class _WithinSphere(Initializer):
    """Places particles in a shell around a transform (m_fRadiusMin to m_fRadiusMax), biased per axis, with a
    speed outwards and one in the transform's own coordinates."""

    def __init__(self, p: Params, transform: Transform):
        super().__init__(p)
        self.transform = transform
        self.radius_min = p.number("m_fRadiusMin", 0.0)
        self.radius_max = p.number("m_fRadiusMax", 0.0)
        self.speed_min = p.number("m_fSpeedMin", 0.0)
        self.speed_max = p.number("m_fSpeedMax", 0.0)
        self.speed_exponent = min(max(p.float("m_fSpeedRandExp", 1.0), -255.0), 255.0)
        self.local_speed_min = p.vec("m_LocalCoordinateSystemSpeedMin")
        self.local_speed_max = p.vec("m_LocalCoordinateSystemSpeedMax")
        self.distance_bias = p.vec("m_vecDistanceBias", (1.0, 1.0, 1.0))
        self.distance_bias_abs = p.vector("m_vecDistanceBiasAbs")
        self.local = p.bool("m_bLocalCoords")
        self.field = p.int("m_nFieldOutput", F.POSITION)

    def initialize(self, sim, v):
        n = v.n
        transform = self.transform(sim)
        rotation, origin = transform
        direction, fraction = sim.random.in_unit_ball(n)
        direction = np.where(self.distance_bias_abs != 0, np.abs(direction), direction)
        direction = direction * rows3(self.distance_bias(sim, v), n)
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, length, out=np.zeros_like(direction), where=length > 0)
        low = column(self.radius_min(sim, v), n)
        high = column(self.radius_max(sim, v), n)
        distance = (high - low) * fraction + low
        speed_draw = sim.random.next(n) ** self.speed_exponent
        speed_low = column(self.speed_min(sim, v), n)
        speed = speed_low + (column(self.speed_max(sim, v), n) - speed_low) * speed_draw
        local_low = rows3(self.local_speed_min(sim, v), n)
        local_speed = local_low + (rows3(self.local_speed_max(sim, v), n) - local_low) * sim.random.next(n * 3).reshape(n, 3)

        offset = direction * distance[:, None]
        if self.local:
            position = transform_points(transform, offset)
            direction = transform_directions(transform, direction)
        else:
            position = origin + offset
        v.set_field(self.field, position)
        v.set("velocity", direction * speed[:, None] + local_speed @ rotation.T)


@initializer("C_INIT_CreateWithinSphereTransform")
class CreateWithinSphereTransform(_WithinSphere):
    def __init__(self, p: Params):
        super().__init__(p, p.transform("m_TransformInput", Transform(0)))


@initializer("C_INIT_CreateWithinSphere")
class CreateWithinSphere(_WithinSphere):
    def __init__(self, p: Params):
        super().__init__(p, Transform(p.int("m_nControlPointNumber")))


def _range_sample(sim, v, p_seed, low, high):
    """Points between two corners: random per component, or a low discrepancy sequence when distributed evenly."""
    n = v.n
    low = rows3(low, n)
    high = rows3(high, n)
    if p_seed is None:
        return low + (high - low) * sim.random.next(n * 3).reshape(n, 3)
    seed = p_seed if p_seed >= 0 else p_seed + sim.random.seed
    index = (seed + v.get("unique_id").astype(np.int64)) & 0x7FFFFFFF
    fractions = np.stack([_radical_inverse(index, base) for base in (2, 3, 5)], axis=1)
    return low + (high - low) * fractions


def _radical_inverse(index, base):
    index = index.copy()
    result = np.zeros(index.shape)
    scale = 1.0
    while np.any(index > 0):
        scale /= base
        result += (index % base) * scale
        index //= base
    return result


def _even_seed(p: Params):
    block = p.get("m_randomnessParameters")
    if isinstance(block, dict) and block.get("m_bDistributeEvenly"):
        return int(block.get("m_nSeed", 0))
    return None


@initializer("C_INIT_CreateWithinBox")
class CreateWithinBox(Initializer):
    """Places particles in a box around a control point (in its rotated frame when m_bLocalSpace)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vec("m_vecMin")
        self.high = p.vec("m_vecMax")
        self.control_point = p.int("m_nControlPointNumber")
        self.local = p.bool("m_bLocalSpace")
        self.seed = _even_seed(p)

    def initialize(self, sim, v):
        offset = _range_sample(sim, v, self.seed, self.low(sim, v), self.high(sim, v))
        point = sim.control_point(self.control_point)
        if self.local and point.oriented:
            offset = offset @ point.rotation.T
        v.set("position", point.position + offset)


@initializer("C_INIT_PositionOffset")
class PositionOffset(Initializer):
    """Moves particles by a random offset between two corners, in a transform's frame when local, scaled by
    radius when proportional."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vec("m_OffsetMin")
        self.high = p.vec("m_OffsetMax")
        self.transform = p.transform("m_TransformInput", Transform(0))
        self.local = p.bool("m_bLocalCoords")
        self.proportional = p.bool("m_bProportional")
        self.seed = _even_seed(p)

    def initialize(self, sim, v):
        offset = _range_sample(sim, v, self.seed, self.low(sim, v), self.high(sim, v))
        if self.proportional:
            offset = offset * v.get("radius")[:, None]
        if self.local:
            offset = transform_directions(self.transform(sim), offset)
        v.set("position", v.get("position") + offset)


@initializer("C_INIT_RandomColor")
class RandomColor(Initializer):
    """A colour between two (one random blend for all three channels)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.color("m_ColorMin")
        self.high = p.color("m_ColorMax")
        self.field = p.int("m_nFieldOutput", F.COLOR)
        if p.float("m_flTintPerc") > 0:
            p.note("C_INIT_RandomColor light tint")

    def initialize(self, sim, v):
        t = sim.random.next(v.n)[:, None]
        v.set_field(self.field, self.low + (self.high - self.low) * t)


@initializer("C_INIT_RandomSequence")
class RandomSequence(Initializer):
    """A sprite sheet sequence: random between two, in order, shuffled, or from a weighted list."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.int("m_nSequenceMin")
        self.high = p.int("m_nSequenceMax")
        self.shuffle = p.bool("m_bShuffle")
        self.linear = p.bool("m_bLinear")
        self.weighted = [(e.int("m_nSequence"), e.float("m_flRelativeWeight", 1.0)) for e in p.array("m_WeightedList")]
        if self.weighted:
            self.low = min(s for s, _ in self.weighted)
            self.high = max(s for s, _ in self.weighted)
        self.reset()

    def reset(self):
        self.cursor = 0
        self.order = []

    def initialize(self, sim, v):
        n = v.n
        if self.weighted:
            total = sum(w for _, w in self.weighted if w > 0)
            out = np.zeros(n)
            for i in range(n):
                pick = (self.cursor if self.linear else total * sim.random.next1())
                if self.linear:
                    self.cursor = (self.cursor + 1) % max(1, math.ceil(total))
                sequence = 0
                for entry, weight in self.weighted:
                    if weight <= 0:
                        continue
                    pick -= weight
                    sequence = entry
                    if pick < 0:
                        break
                out[i] = sequence
        elif self.shuffle or self.linear:
            out = np.zeros(n)
            for i in range(n):
                if not self.order:
                    self.order = list(range(self.low, max(self.high, self.low) + 1))
                    if self.shuffle:
                        draws = sim.random.next(len(self.order))
                        self.order = [self.order[j] for j in np.argsort(draws)]
                out[i] = self.order.pop(0)
        elif self.high > self.low:
            out = np.minimum(self.low + np.floor(sim.random.next(n) * (self.high - self.low + 1)), self.high)
        else:
            out = np.full(n, float(self.low))
        v.set("sequence", out)


@initializer("C_INIT_VelocityRandom")
class VelocityRandom(Initializer):
    """Adds a random velocity: one in a control point's frame, plus a per component world space range."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vec("m_LocalCoordinateSystemSpeedMin")
        self.high = p.vec("m_LocalCoordinateSystemSpeedMax")
        self.speed_min = p.number("m_fSpeedMin", 0.0)
        self.speed_max = p.number("m_fSpeedMax", 0.0)
        self.control_point = p.int("m_nControlPointNumber")
        self.ignore_dt = p.bool("m_bIgnoreDT")
        self.seed = _even_seed(p)

    def initialize(self, sim, v):
        n = v.n
        low, high = self.low(sim, v), self.high(sim, v)
        velocity = np.zeros((n, 3))
        if np.any(low) or np.any(high):
            local = _range_sample(sim, v, self.seed, low, high)
            velocity += transform_directions(Transform(self.control_point)(sim), local)
        speed_low = column(self.speed_min(sim, v), n)
        speed_high = column(self.speed_max(sim, v), n)
        if np.any(speed_low) or np.any(speed_high):
            velocity += speed_low[:, None] + (speed_high - speed_low)[:, None] * sim.random.next(n * 3).reshape(n, 3)
        if self.ignore_dt and sim.frame_time > 0:
            velocity /= sim.frame_time
        v.set("velocity", v.get("velocity") + velocity)


@initializer("C_INIT_RingWave")
class RingWave(Initializer):
    """Places particles on a ring around a transform (random angles, or evenly around an orbit), thickened by a
    random offset, moving outwards."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.even = p.bool("m_bEvenDistribution")
        self.per_orbit = p.number("m_flParticlesPerOrbit", -1.0)
        self.radius = p.number("m_flInitialRadius", 0.0)
        self.thickness = p.number("m_flThickness", 0.0)
        self.speed_min = p.number("m_flInitialSpeedMin", 0.0)
        self.speed_max = p.number("m_flInitialSpeedMax", 0.0)
        self.transform = p.transform("m_TransformInput", Transform(0))
        self.xy_only = p.bool("m_bXYVelocityOnly")
        self.orbit = 0

    def reset(self):
        self.orbit = 0

    def initialize(self, sim, v):
        n = v.n
        thickness = column(self.thickness(sim, v), n)
        per_orbit = column(self.per_orbit(sim, v), n)
        offset = sim.random.in_unit_ball(n)[0] * thickness[:, None]
        radius = column(self.radius(sim, v), n)
        if self.even:
            per_orbit = np.where(per_orbit == -1.0, sim.particles.count, per_orbit)
            steps = self.orbit + 1 + np.arange(n)
            self.orbit += n
            angle = steps * (math.tau / np.maximum(1.0, per_orbit))
        else:
            angle = sim.random.next(n) * math.tau
        local = np.stack([np.cos(angle), np.sin(angle), np.zeros(n)], axis=1) * radius[:, None] + offset
        transform = self.transform(sim)
        position = transform_points(transform, local)
        v.set("position", position)
        outward = position - transform[1]
        if self.xy_only:
            outward[:, 2] = 0.0
        length = np.linalg.norm(outward, axis=1, keepdims=True)
        outward = np.divide(outward, length, out=np.zeros_like(outward), where=length > 0)
        low = column(self.speed_min(sim, v), n)
        speed = low + (column(self.speed_max(sim, v), n) - low) * sim.random.next(n)
        v.set("velocity", v.get("velocity") + outward * speed[:, None])


@initializer("C_INIT_NormalAlignToCP")
class NormalAlignToCP(Initializer):
    """Sets the particle normal to one of a transform's axes."""

    AXES = {"PARTICLE_CP_AXIS_X": (1, 0, 0), "PARTICLE_CP_AXIS_Y": (0, 1, 0), "PARTICLE_CP_AXIS_Z": (0, 0, 1),
            "PARTICLE_CP_AXIS_NEGATIVE_X": (-1, 0, 0), "PARTICLE_CP_AXIS_NEGATIVE_Y": (0, -1, 0),
            "PARTICLE_CP_AXIS_NEGATIVE_Z": (0, 0, -1)}

    def __init__(self, p: Params):
        super().__init__(p)
        self.transform = p.transform("m_transformInput", Transform(0))
        self.axis = np.array(self.AXES.get(p.str("m_nControlPointAxis"), (1, 0, 0)), np.float64)

    def initialize(self, sim, v):
        normal = transform_directions(self.transform(sim), self.axis)
        length = np.linalg.norm(normal)
        if length > 0:
            v.set_field(F.NORMAL, normal / length)


@initializer("C_INIT_RandomYawFlip")
class RandomYawFlip(Initializer):
    """Turns some particles half way round in yaw (mirroring their cards)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.percent = p.float("m_flPercent", 0.5)

    def initialize(self, sim, v):
        flip = sim.random.next(v.n) < self.percent
        v.set("yaw", v.get("yaw") + np.where(flip, math.pi, 0.0))


@initializer("C_INIT_RemapParticleCountToScalar")
class RemapParticleCountToScalar(Initializer):
    """Sets a float attribute from how many particles the initializer has seen, remapped (and biased)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.input_min = p.int("m_nInputMin")
        self.input_max = p.int("m_nInputMax", 10)
        self.output_min = p.float("m_flOutputMin")
        self.output_max = p.float("m_flOutputMax", 1.0)
        self.active_range = p.bool("m_bActiveRange")
        self.scale_initial = p.bool("m_bScaleInitialRange")
        self.invert = p.bool("m_bInvert")
        self.wrap = p.bool("m_bWrap")
        self.bias = p.float("m_flRemapBias", 0.5)
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        self.control_point = min(max(p.int("m_nScaleControlPoint", -1), -1), 64)
        self.component = min(max(p.int("m_nScaleControlPointField"), -1), 2)
        if self.field in (F.ALPHA, F.ALPHA_ALTERNATE):
            self.output_min = min(max(self.output_min, 0.0), 1.0)
            self.output_max = min(max(self.output_max, 0.0), 1.0)
        self.count = 0

    def reset(self):
        self.count = 0

    def initialize(self, sim, v):
        low, high = self.input_min, self.input_max
        if self.invert:
            low, high = sim.particles.count - self.input_max - 1, sim.particles.count - self.input_min - 1
        if self.control_point >= 0 and self.component != -1:
            scale = sim.control_point(self.control_point).position[self.component]
            low, high = int(low * scale), int(high * scale)
        current = v.field(self.field).copy()
        for i in range(v.n):
            if self.active_range and self.count > high:
                continue
            if not self.active_range or self.count >= low:
                value = float(remap_clamped(self.count, low, high, self.output_min, self.output_max))
                if self.bias != 0.5:
                    value = float(bias(value, self.bias))
                current[i] = current[i] * value if self.scale_initial else set_at_spawn(value, current[i], self.method, sim.frame_time)
            self.count += 1
            if self.wrap and self.count > high:
                self.count = 0
        v.set_field(self.field, current)


@initializer("C_INIT_InitialVelocityNoise")
class InitialVelocityNoise(Initializer):
    """Adds a velocity from value noise over the particle's position and birth time, mapped per component between
    two vectors."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vec("m_vecOutputMin")
        self.high = p.vec("m_vecOutputMax", (1, 1, 1))
        self.scale = p.number("m_flNoiseScale", 0.1)
        self.scale_location = p.number("m_flNoiseScaleLoc", 0.01)
        self.offset = p.number("m_flOffset", 0.0)
        self.offset_location = p.vec("m_vecOffsetLoc")
        self.absolute = p.vector("m_vecAbsVal") != 0
        self.inverted = p.vector("m_vecAbsValInv") != 0
        self.ignore_dt = p.bool("m_bIgnoreDt")
        self.transform = p.transform("m_TransformInput", Transform(None))

    def initialize(self, sim, v):
        n = v.n
        sample = ((v.get("position") + self.offset_location(sim, v)) * column(self.scale_location(sim, v), n)[:, None]
                  + ((v.get("creation_time") + column(self.offset(sim, v), n)) * column(self.scale(sim, v), n))[:, None])
        value = noise.vector3(sample)
        value = np.where(self.absolute, np.abs(value), value)
        if np.any(self.inverted):
            value = np.where(self.inverted, 1.0 + value, -value)
        half = np.where(self.absolute, 1.0, 0.5)
        low = rows3(self.low(sim, v), n)
        span = rows3(self.high(sim, v), n) - low
        velocity = value * span * half + low + span * (1.0 - half)
        velocity = transform_directions(self.transform(sim), velocity)
        if self.ignore_dt and sim.frame_time > 0:
            velocity = velocity / sim.frame_time
        v.set("velocity", v.get("velocity") + velocity)


def _path_parameters(p: Params):
    block = p.nested(p.get("m_PathParams"))
    return {
        "start": block.int("m_nStartControlPointNumber"), "mid": block.int("m_nMidControlPointNumber", -1),
        "end": block.int("m_nEndControlPointNumber"), "bulge_control": block.int("m_nBulgeControl"),
        "bulge": block.float("m_flBulge"), "mid_point": block.float("m_flMidPoint", 0.5),
        "start_offset": block.vector("m_vStartPointOffset"), "mid_offset": block.vector("m_vMidPointOffset"),
        "end_offset": block.vector("m_vEndOffset"),
    }


def path_points(sim, path: dict, start_cp: int, end_cp: int):
    """A path's start, middle (bulged) and end: a quadratic Bezier between two control points."""
    start = sim.control_point(start_cp).position + path["start_offset"]
    end = sim.control_point(end_cp).position + path["end_offset"]
    if path["mid"] > -1:
        through = sim.control_point(path["mid"]).position
        return start, 2.0 * through - 0.5 * (start + end) + path["mid_offset"], end
    mid = lerp(start, end, path["mid_point"])
    if path["bulge_control"] != 0:
        forward_cp = end_cp if path["bulge_control"] == 2 else start_cp
        point = sim.control_point(forward_cp)
        forward = point.forward if point.oriented else np.zeros(3)
        direction = end - start
        length = np.linalg.norm(direction)
        perpendicularity = 1.0 - abs(np.dot(direction / length, forward)) if length > 1e-6 else 0.0
        forward_length = np.linalg.norm(forward)
        if forward_length > 1e-6:
            mid = mid + forward * (length * path["bulge"] * perpendicularity / forward_length)
    else:
        seed = sim.random.seed
        mid = mid + np.array([sim.random.sample(seed + i) for i in range(3)]) * 2 * path["bulge"] - path["bulge"]
    return start, mid + path["mid_offset"], end


def path_evaluate(points, t):
    start, mid, end = points
    t = np.asarray(t, np.float64)[..., None]
    return lerp(lerp(start, mid, t), lerp(mid, end, t), t)


@initializer("C_INIT_CreateSequentialPathV2")
class CreateSequentialPathV2(Initializer):
    """Places particles one after another along a path between control points (m_flNumToAssign of them from
    start to end), looping or bouncing back, optionally jittered."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.max_distance = p.number("m_fMaxDistance", 0.0)
        self.count = p.number("m_flNumToAssign", 100.0)
        self.loop = p.bool("m_bLoop", True)
        self.pairs = p.bool("m_bCPPairs")
        self.save_offset = p.bool("m_bSaveOffset")
        self.path = _path_parameters(p)
        self.direction = 1 if self.path["end"] >= self.path["start"] else -1
        if self.pairs:
            self.span = abs(self.path["end"] - self.path["start"])
            if self.span <= 1:
                self.pairs = False
        else:
            self.span = 1
        self.reset()

    def reset(self):
        self.parameter = 0.0
        self.step = 0.0
        self.cached = None

    def initialize(self, sim, v):
        n = v.n
        positions = np.zeros((n, 3))
        offsets = np.zeros((n, 3))
        jitter = column(self.max_distance(sim, v), n)
        path = self.path
        for i in range(n):
            count = float(self.count(sim))
            if count != self.cached:
                self.cached = count
                self.step = 1.0 / (count - 1.0) if count > 1 else 0.0
                if self.pairs:
                    self.step *= self.span
            t = self.parameter
            if not self.pairs:
                first, last = path["start"], path["end"]
            else:
                first = path["start"] + int(t) * self.direction
                last = first + self.direction
                t -= int(t)
                if self.direction * (self.direction + first - path["end"]) >= 1:
                    last = path["end"]
                    first = path["end"] - self.direction
                    t = 1.0
            positions[i] = path_evaluate(path_points(sim, path, first, last), t)
            offsets[i] = (t, first, last)
            previous = self.parameter
            self.parameter += self.step
            advanced = self.parameter
            if self.step > 0:
                if abs(advanced - self.span) < 1e-5:
                    advanced = self.parameter = float(self.span)
                elif abs(advanced) < 1e-5:
                    advanced = self.parameter = 0.0
            if advanced > self.span or advanced < 0:
                wrapped = -advanced if self.step <= 0 else advanced - self.span
                if self.loop:
                    self.parameter = wrapped
                    if previous == self.span:
                        self.parameter = 0.0
                else:
                    if self.step >= 0:
                        wrapped = self.span - wrapped
                    self.parameter = wrapped
                    self.step = -self.step
        positions += (sim.random.next(n * 3).reshape(n, 3) * 2.0 - 1.0) * jitter[:, None]
        v.set("position", positions)
        if self.save_offset:
            v.set("hitbox_offset", offsets)


@initializer("C_INIT_DistanceToCPInit")
class DistanceToCPInit(Initializer):
    """Sets a float attribute from the particle's distance to a control point, remapped."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.input_min = p.number("m_flInputMin", 0.0)
        self.input_max = p.number("m_flInputMax", 128.0)
        self.output_min = p.number("m_flOutputMin", 0.0)
        self.output_max = p.number("m_flOutputMax", 1.0)
        self.control_point = min(max(p.int("m_nStartCP"), 0), 63)
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        self.active_range = p.bool("m_bActiveRange")
        self.scale = p.vector("m_vecDistanceScale", (1, 1, 1))
        self.bias = p.float("m_flRemapBias", 0.5)

    def initialize(self, sim, v):
        n = v.n
        low, high = column(self.input_min(sim, v), n), column(self.input_max(sim, v), n)
        out_low, out_high = column(self.output_min(sim, v), n), column(self.output_max(sim, v), n)
        if self.field in (F.ALPHA, F.ALPHA_ALTERNATE):
            out_low, out_high = saturate(out_low), saturate(out_high)
        distance = np.linalg.norm((sim.control_point(self.control_point).position - v.get("position")) * self.scale, axis=1)
        value = remap_clamped(distance, low, high, out_low, out_high)
        if self.bias != 0.5:
            value = bias(saturate(value), self.bias)
        current = v.field(self.field)
        value = set_at_spawn(value, current, self.method, sim.frame_time)
        if self.active_range:
            value = np.where((distance < low) | (distance > high), current, value)
        v.set_field(self.field, value)


@initializer("C_INIT_CreateFromParentParticles")
class CreateFromParentParticles(Initializer):
    """Places particles where the parent system's particles are (walking through them in order, or at random),
    between their last two positions by birth time, inheriting a share of their velocity."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.velocity_scale = p.float("m_flVelocityScale")
        self.increment = p.float("m_flIncrement", 1.0)
        self.random = p.bool("m_bRandomDistribution")
        self.random_seed = p.int("m_nRandomSeed")
        self.sub_frame = p.bool("m_bSubFrame", True)
        self.rope_segment = p.bool("m_bSetRopeSegmentID")
        self.reset()

    def reset(self):
        self.index = 0.0
        self.counter = 0

    def initialize(self, sim, v):
        parent = sim.parent
        n = v.n
        if parent is None:
            return
        count = parent.particles.count
        if count == 0:
            v.set("lifetime", 0.0)
            return
        last = count - 1
        fractional = self.increment != math.floor(self.increment)
        walk = np.zeros(n)
        for i in range(n):
            if self.random:
                draw = sim.random.sample(sim.random.seed + self.random_seed + self.counter) if self.random_seed else sim.random.next1()
                self.counter += 1
                self.index = draw * last if fractional else float(int(draw * (last + 1)))
            elif self.index > last:
                self.index = 0.0
            walk[i] = self.index
            self.index += self.increment
        first = np.clip(np.floor(walk).astype(int), 0, last)
        second = np.clip(np.ceil(walk).astype(int), 0, last)
        blend = (walk - np.floor(walk))[:, None] if fractional else 0.0
        positions = parent.particles.array("position")
        previous = parent.particles.array("previous")
        position = lerp(positions[first], positions[second], blend)
        previous_position = lerp(previous[first], previous[second], blend)
        if self.sub_frame:
            born = v.get("creation_time")
            fraction = remap_clamped(born, sim.age - sim.frame_time, sim.age, 0.0, 1.0)[:, None]
        else:
            fraction = np.ones((n, 1))
        v.set("position", lerp(previous_position, position, fraction))
        parent_dt = parent.frame_time
        velocity = (position - previous_position) / parent_dt if parent_dt > 0 else parent.particles.array("velocity")[first]
        v.set("velocity", velocity * fraction * self.velocity_scale)
        v.set("parent_index", first.astype(np.float64))
        v.set("parent_id", parent.particles.array("particle_id")[first])
        if self.rope_segment:
            v.set("rope_segment_id", parent.particles.array("particle_id")[first])


@initializer("C_INIT_PositionWarp")
class PositionWarp(Initializer):
    """Scales particle positions about a control point, per axis, by a warp between two vectors (random, or ramped
    over birth time or count)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vec("m_vecWarpMin", (1, 1, 1))
        self.high = p.vec("m_vecWarpMax", (1, 1, 1))
        self.control_point = p.int("m_nControlPointNumber")
        self.scale_control_point = p.int("m_nScaleControlPointNumber", -1)
        self.radius_component = min(max(p.int("m_nRadiusComponent", -1), -1), 2)
        self.time = p.float("m_flWarpTime")
        self.start = p.float("m_flWarpStartTime")
        self.previous_scale = p.float("m_flPrevPosScale", 1.0)
        self.invert = p.bool("m_bInvertWarp")
        self.use_count = p.bool("m_bUseCount")
        self.count = 0

    def reset(self):
        self.count = 0

    def _warp(self, sim, v):
        n = v.n
        low, high = rows3(self.low(sim, v), n), rows3(self.high(sim, v), n)
        if self.invert:
            low, high = high, low
        if self.scale_control_point >= 0:
            scale = sim.control_point(self.scale_control_point).position
            low, high = low * scale, high * scale
        if self.time == 0:
            return low + (high - low) * sim.random.next(n * 3).reshape(n, 3)
        ramp = v.get("creation_time")
        if self.use_count:
            ramp = self.count + np.arange(n, dtype=np.float64)
            self.count += n
        progress = remap_clamped(ramp, self.start, self.start + self.time, 0.0, 1.0)[:, None]
        return lerp(low, high, progress)

    def initialize(self, sim, v):
        warp = self._warp(sim, v)
        _warp_about(sim, v, self.control_point, warp, self.previous_scale)
        if self.radius_component >= 0:
            v.set("radius", v.get("radius") * warp[:, self.radius_component])


def _warp_about(sim, v, control_point: int, warp, previous_scale: float):
    point = sim.control_point(control_point)
    rotation = point.rotation if point.oriented else np.eye(3)
    local = (v.get("position") - point.position) @ rotation
    v.set("position", (local * warp) @ rotation.T + point.position)
    local = (v.get("previous") - point.position) @ rotation
    v.set("previous", (local * warp * previous_scale) @ rotation.T + point.position)


@initializer("C_INIT_PositionWarpScalar")
class PositionWarpScalar(Initializer):
    """Scales particle positions about a control point, per axis, by a warp between two vectors chosen by a float
    input."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.low = p.vector("m_vecWarpMin", (1, 1, 1))
        self.high = p.vector("m_vecWarpMax", (1, 1, 1))
        self.input = p.number("m_InputValue", 0.0)
        self.control_point = p.int("m_nControlPointNumber")
        self.previous_scale = p.float("m_flPrevPosScale", 1.0)

    def initialize(self, sim, v):
        t = column(self.input(sim, v), v.n)[:, None]
        _warp_about(sim, v, self.control_point, lerp(self.low, self.high, t), self.previous_scale)
