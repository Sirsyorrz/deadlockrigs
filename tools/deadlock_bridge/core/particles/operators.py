"""Operators (run on every particle every step), pre-emission operators (set control points before emission), force
generators (run by C_OP_BasicMovement) and constraints (move particles back into place after the operators)."""
import math

import numpy as np

from . import fields as F
from . import models, noise
from .initializers import TransformOrientation, column, models_control_point, path_evaluate, path_points, rows3, _path_parameters
from .inputs import Transform, transform_directions, transform_points
from .params import (SET_ADD_CURRENT, SET_ADD_INITIAL, SET_METHODS, SET_RAMP_CURRENT, SET_REPLACE,
                     SET_SCALE_CURRENT, SET_SCALE_INITIAL, Params, bias, clamp_between, lerp, remap, remap_clamped,
                     saturate, smoothstep)
from .system import Function

OPERATORS = {}
PRE_EMISSION = {}
FORCES = {}
CONSTRAINTS = {}


def register(table, name):
    def add(cls):
        table[name] = cls
        return cls
    return add


class Operator(Function):
    def operate(self, sim, v, dt: float, strength: float):
        raise NotImplementedError


def set_by_method(v, field: int, value, method: int, dt: float):
    """A value applied to an attribute by a set method: replacing it, scaling or adding to its initial or current
    value, or ramping it by a rate."""
    if method == SET_REPLACE:
        return value
    if method == SET_SCALE_INITIAL:
        return value * v.initial_field(field)
    if method == SET_ADD_INITIAL:
        return value + v.initial_field(field)
    if method == SET_SCALE_CURRENT:
        return value * v.field(field)
    if method == SET_ADD_CURRENT:
        return value + v.field(field)
    if method == SET_RAMP_CURRENT:
        return v.field(field) + value * dt
    return value


def _float_range(field: int):
    if field in (F.ALPHA, F.ALPHA_ALTERNATE):
        return 0.0, 1.0
    if field in (F.RADIUS, F.TRAIL_LENGTH):
        return 0.0, np.inf
    return -np.inf, np.inf


# Movement ---------------------------------------------------------------------------------------------------------

@register(OPERATORS, "C_OP_BasicMovement")
class BasicMovement(Operator):
    """Verlet integration: the step since the previous position (dragged), plus gravity and the force generators'
    accelerations, weighted by each particle's force scale."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.gravity = p.vec("m_Gravity")
        self.drag = p.number("m_fDrag", 0.0)

    def operate(self, sim, v, dt, strength):
        v.set("force", 0.0)
        for force in sim.forces:
            force_strength = force.run_strength(sim)
            if force_strength > 0.0:
                force.generate(sim, v, dt, force_strength)
        drag = min(max(float(self.drag(sim)), 0.0), 0.9999999)
        drag_factor = math.exp(math.log(1.0 - drag) / (1.0 / 30.0) * dt)
        if sim.previous_frame_time > 0:
            drag_factor *= dt / sim.previous_frame_time
        forces = np.asarray(self.gravity(sim), np.float64) * dt * dt + v.get("force") * dt * dt
        position = v.get("position")
        step = v.get("force_scale")[:, None] * forces + drag_factor * (position - v.get("previous"))
        v.set("velocity", step / dt)
        v.set("previous", position.copy())
        v.set("position", position + step)


@register(OPERATORS, "C_OP_Decay")
class Decay(Operator):
    """Kills particles that have outlived their lifetime."""

    def operate(self, sim, v, dt, strength):
        v.kill(v.get("age") > v.get("lifetime"))


@register(OPERATORS, "C_OP_FadeInSimple")
class FadeInSimple(Operator):
    """Fades alpha in from 0 over the first part of the particle's life (a fraction of it)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.time = p.float("m_flFadeInTime", 0.25)
        self.field = p.int("m_nFieldOutput", F.ALPHA)

    def operate(self, sim, v, dt, strength):
        t = v.normalized_age
        mask = t <= self.time
        if mask.any():
            value = (t / self.time) * v.initial("alpha") if self.time > 0 else v.initial("alpha")
            v.set_field(self.field, np.where(mask, value, v.field(self.field)))


@register(OPERATORS, "C_OP_FadeOutSimple")
class FadeOutSimple(Operator):
    """Fades alpha out to 0 over the last part of the particle's life (a fraction of it)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.time = p.float("m_flFadeOutTime", 0.25)
        self.field = p.int("m_nFieldOutput", F.ALPHA)

    def operate(self, sim, v, dt, strength):
        left = 1.0 - v.normalized_age
        mask = left <= self.time
        if mask.any():
            value = (left / self.time) * v.initial("alpha") if self.time > 0 else 0.0
            v.set_field(self.field, np.where(mask, value, v.field(self.field)))


@register(OPERATORS, "C_OP_FadeAndKill")
class FadeAndKill(Operator):
    """Fades alpha in and out over windows of the particle's life (smoothstepped), killing it at the end of it."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.in_start = p.float("m_flStartFadeInTime")
        self.in_end = p.float("m_flEndFadeInTime", 0.5)
        self.out_start = p.float("m_flStartFadeOutTime", 0.5)
        self.out_end = p.float("m_flEndFadeOutTime", 1.0)
        self.start_alpha = p.float("m_flStartAlpha", 1.0)
        self.end_alpha = p.float("m_flEndAlpha")

    def operate(self, sim, v, dt, strength):
        expired = v.get("lifetime") < v.get("age")
        t = v.normalized_age
        initial = v.initial("alpha")
        alpha = v.get("alpha").copy()
        fade_in = (t >= self.in_start) & (t < self.in_end)
        alpha = np.where(fade_in, initial * lerp(self.start_alpha, 1.0, smoothstep(self.in_start, self.in_end, t)), alpha)
        fade_out = (t >= self.out_start) & (t < self.out_end)
        alpha = np.where(fade_out, initial * lerp(1.0, self.end_alpha, smoothstep(self.out_start, self.out_end, t)), alpha)
        v.set("alpha", np.where(expired, v.get("alpha"), lerp(v.get("alpha"), alpha, strength)))
        v.kill(expired)


@register(OPERATORS, "C_OP_InterpolateRadius")
class InterpolateRadius(Operator):
    """Scales the initial radius from one factor to another over a window of the particle's life (biased, or eased
    in and out)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.start = p.float("m_flStartTime")
        self.end = p.float("m_flEndTime", 1.0)
        self.start_scale = p.number("m_flStartScale", 1.0)
        self.end_scale = p.number("m_flEndScale", 1.0)
        self.ease = p.bool("m_bEaseInAndOut")
        amount = p.float("m_flBias", 0.5)
        self.bias = 0.5 if amount == 0 else amount

    def operate(self, sim, v, dt, strength):
        if self.end <= self.start:
            return
        lifetime = v.get("lifetime")
        t = v.normalized_age
        frame_fraction = dt / np.maximum(lifetime, 1e-9)
        window_end = np.where(lifetime * (self.end - self.start) <= dt, self.start + frame_fraction, self.end)
        mask = (lifetime > 0) & (t >= self.start) & (t <= frame_fraction + window_end)
        if not mask.any():
            return
        low = self.start_scale(sim, v)
        high = self.end_scale(sim, v)
        x = remap(t, self.start, self.end)
        if self.ease:
            x = x * x * (3 - 2 * x)
        elif self.bias != 0.5:
            x = bias(x, self.bias)
        scale = clamp_between(lerp(low, high, x), low, high)
        v.set("radius", np.where(mask, v.initial("radius") * scale, v.get("radius")))


@register(OPERATORS, "C_OP_ColorInterpolate")
class ColorInterpolate(Operator):
    """Fades the initial colour to another over a window of the particle's life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.color = p.color("m_ColorFade")
        self.start = p.float("m_flFadeStartTime")
        self.end = p.float("m_flFadeEndTime", 1.0)
        self.field = p.int("m_nFieldOutput", F.COLOR)
        self.ease = p.bool("m_bEaseInOut", True)

    def operate(self, sim, v, dt, strength):
        t = v.normalized_age
        mask = (t >= self.start) & (t <= self.end)
        if not mask.any():
            return
        x = remap(t, self.start, self.end)
        if self.ease:
            x = smoothstep(0.0, 1.0, x)
        initial = v.initial("color")
        faded = lerp(initial, self.color, x[:, None])
        v.set_field(self.field, np.where(mask[:, None], lerp(initial, faded, strength), v.field(self.field)))


@register(OPERATORS, "C_OP_SetFloat")
class SetFloat(Operator):
    """Sets a float attribute from an input every step (angles in degrees), by a set method, eased by a lerp."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nOutputField", F.RADIUS)
        self.value = p.number("m_InputValue", 0.0)
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        self.lerp = p.number("m_Lerp", 1.0)

    def operate(self, sim, v, dt, strength):
        if F.is_vector(self.field) or F.name(self.field) is None:
            return
        value = column(self.value(sim, v), v.n)
        if self.field in F.ANGLES and self.method not in (SET_SCALE_INITIAL, SET_SCALE_CURRENT):
            value = np.radians(value)
        target = set_by_method(v, self.field, value, self.method, dt)
        current = v.field(self.field)
        low, high = _float_range(self.field)
        blended = np.clip(lerp(current, target, saturate(self.lerp(sim, v))), low, high)
        base = v.initial_field(self.field) if self.method in (SET_SCALE_INITIAL, SET_ADD_INITIAL) else current
        v.set_field(self.field, lerp(base, blended, strength))


@register(OPERATORS, "C_OP_SetVec")
class SetVec(Operator):
    """Sets a vector attribute from an input every step, by a set method, eased by a lerp."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nOutputField", F.COLOR)
        self.value = p.vec("m_InputValue")
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        self.lerp = p.number("m_Lerp", 1.0)
        self.normalized = p.bool("m_bNormalizedOutput")

    def operate(self, sim, v, dt, strength):
        if not F.is_vector(self.field):
            return
        value = rows3(self.value(sim, v), v.n)
        amount = saturate(column(self.lerp(sim, v), v.n) * strength)[:, None]
        target = set_by_method(v, self.field, value, self.method, dt)
        value = lerp(v.field(self.field), target, amount)
        if self.normalized:
            length = np.linalg.norm(value, axis=1, keepdims=True)
            value = np.divide(value, length, out=value.copy(), where=length > 0)
        v.set_field(self.field, value)


@register(OPERATORS, "C_OP_ClampScalar")
class ClampScalar(Operator):
    """Holds a float attribute between two inputs (the maximum wins an inverted range)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.low = p.number("m_flOutputMin", 0.0)
        self.high = p.number("m_flOutputMax", 1.0)

    def operate(self, sim, v, dt, strength):
        current = v.field(self.field)
        clamped = np.minimum(self.high(sim, v), np.maximum(current, self.low(sim, v)))
        v.set_field(self.field, lerp(current, clamped, strength))


@register(OPERATORS, "C_OP_LerpScalar")
class LerpScalar(Operator):
    """Moves a float attribute from its initial value to an input over a window of the particle's life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.output = p.number("m_flOutput", 1.0)
        self.start = p.float("m_flStartTime")
        self.end = p.float("m_flEndTime", 1.0)

    def operate(self, sim, v, dt, strength):
        weight = saturate(remap(v.normalized_age, self.start, self.end))
        value = lerp(v.initial_field(self.field), self.output(sim, v), weight)
        v.set_field(self.field, lerp(v.field(self.field), value, strength))


@register(OPERATORS, "C_OP_RampScalarLinearSimple")
class RampScalarLinearSimple(Operator):
    """Adds a rate per second to a float attribute over a window of the particle's life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.rate = p.float("m_Rate")
        self.start = p.float("m_flStartTime")
        self.end = p.float("m_flEndTime", 1.0)
        self.field = p.int("m_nField", F.RADIUS)

    def operate(self, sim, v, dt, strength):
        t = v.normalized_age
        mask = (t > self.start) & (t < self.end)
        v.set_field(self.field, np.where(mask, v.field(self.field) + self.rate * dt * strength, v.field(self.field)))


@register(OPERATORS, "C_OP_CycleScalar")
class CycleScalar(Operator):
    """Cycles a float attribute between two values (a triangle wave over m_flCycleTime), per particle or in sync."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nDestField", F.ALPHA)
        self.start = p.float("m_flStartValue")
        self.end = p.float("m_flEndValue", 1.0)
        self.cycle = p.float("m_flCycleTime", 1.0)
        self.once = p.bool("m_bDoNotRepeatCycle")
        self.synchronized = p.bool("m_bSynchronizeParticles")
        self.scale_cp = p.int("m_nCPScale", -1)
        self.cp_min = p.int("m_nCPFieldMin")
        self.cp_max = p.int("m_nCPFieldMax")
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)

    def operate(self, sim, v, dt, strength):
        start, end = self.start, self.end
        if self.scale_cp >= 0:
            position = sim.control_point(self.scale_cp).position
            if 0 <= self.cp_min < 3:
                start *= position[self.cp_min]
            if 0 <= self.cp_max < 3:
                end *= position[self.cp_max]
        rate = 1.0 / max(1.1920929e-7, self.cycle)
        limit = self.cycle if self.once else np.inf
        if self.once:
            rate *= 0.5
        time = np.full(v.n, sim.age) if self.synchronized else v.get("age")
        phase = np.minimum(time, limit) * rate
        phase -= np.floor(phase)
        value = start + (end - start) * 2.0 * np.minimum(phase, 1.0 - phase)
        final = set_by_method(v, self.field, value, self.method, dt)
        v.set_field(self.field, lerp(v.field(self.field), final, strength))


@register(OPERATORS, "C_OP_SpinUpdate")
class SpinUpdate(Operator):
    """Turns particles by their rotation speed."""

    def operate(self, sim, v, dt, strength):
        step = np.minimum(dt, v.get("age")) * strength
        v.set("roll", v.get("roll") + v.get("roll_speed") * step)


@register(OPERATORS, "C_OP_Spin")
class Spin(Operator):
    """Spins particles' roll at a rate in degrees (decaying to a minimum by m_fSpinRateStopTime)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.rate = p.int("m_nSpinRateDegrees")
        self.minimum = p.int("m_nSpinRateMinDegrees")
        self.stop = p.float("m_fSpinRateStopTime")

    def operate(self, sim, v, dt, strength):
        if self.stop == 0:
            rate = np.full(v.n, self.rate * strength)
        else:
            stop_age = v.get("lifetime") * self.stop
            left = np.where(stop_age != 0, np.maximum(0.0, 1.0 - v.get("age") / np.where(stop_age != 0, stop_age, 1)), 0.0)
            decayed = self.rate * strength * left
            rate = np.maximum(decayed, self.minimum) if self.rate >= 0 else np.minimum(decayed, self.minimum)
        v.set("roll", v.get("roll") + np.radians(rate) * math.tau * dt)


@register(OPERATORS, "C_OP_OscillateVector")
class OscillateVector(Operator):
    """Adds a sine wave to a vector attribute per component, each particle with its own rate and frequency, over a
    window of its life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nField", F.POSITION)
        self.rate_min, self.rate_max = p.vector("m_RateMin"), p.vector("m_RateMax")
        self.frequency_min = p.vector("m_FrequencyMin", (1, 1, 1))
        self.frequency_max = p.vector("m_FrequencyMax", (1, 1, 1))
        self.multiplier = p.number("m_flOscMult", 2.0)
        self.add = p.number("m_flOscAdd", 0.5)
        self.rate_scale = p.number("m_flRateScale", 1.0)
        self.proportional = p.bool("m_bProportional", True)
        self.proportional_op = p.bool("m_bProportionalOp", True)
        self.start_min, self.start_max = p.float("m_flStartTime_min"), p.float("m_flStartTime_max")
        self.end_min, self.end_max = p.float("m_flEndTime_min", 1.0), p.float("m_flEndTime_max", 1.0)
        self.clamp = self.field in F.NORMALIZED
        self.offsets_position = p.bool("m_bOffset") and self.field == F.POSITION

    def operate(self, sim, v, dt, strength):
        ids = v.get("particle_id")
        r = sim.random
        window = v.normalized_age if self.proportional_op else v.get("age")
        start = lerp(self.start_min, self.start_max, r.for_particle(ids, 11))
        end = lerp(self.end_min, self.end_max, r.for_particle(ids, 12))
        mask = (v.get("lifetime") > 0) & (window >= start) & (window < end)
        if not mask.any():
            return
        rate = np.stack([lerp(self.rate_min[i], self.rate_max[i], r.for_particle(ids, o)) for i, o in enumerate((3, 7, 9))], axis=1)
        frequency = np.stack([lerp(self.frequency_min[i], self.frequency_max[i], r.for_particle(ids, o))
                              for i, o in enumerate((8, 12, 15))], axis=1)
        multiplier = column(self.multiplier(sim, v), v.n)[:, None]
        add = column(self.add(sim, v), v.n)[:, None]
        scale = (dt * strength * column(self.rate_scale(sim, v), v.n))[:, None]
        if self.proportional:
            delta = np.sin(math.pi * (v.normalized_age[:, None] * frequency * multiplier + add))
        else:
            delta = np.sin(math.pi * frequency * (multiplier * sim.age + add))
        value = rate * scale * delta * mask[:, None]
        result = v.field(self.field) + value
        v.set_field(self.field, np.clip(result, 0.0, 1.0) if self.clamp else result)
        if self.offsets_position:
            v.set("previous", v.get("previous") + value)


@register(OPERATORS, "C_OP_PositionLock")
class PositionLock(Operator):
    """Moves particles with a transform as it moves (and turns, with m_bLockRot), for part of their life; a locked
    rotation also points their normal along the transform."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.transform = p.transform("m_TransformInput", Transform(0))
        self.start_min = p.float("m_flStartTime_min", 1.0)
        self.start_max = p.float("m_flStartTime_max", 1.0)
        self.end_min = p.float("m_flEndTime_min", 1.0)
        self.end_max = p.float("m_flEndTime_max", 1.0)
        self.previous_scale = p.float("m_flPrevPosScale", 1.0)
        self.lock_rotation = p.bool("m_bLockRot")
        self.field = p.int("m_nFieldOutput", F.POSITION)
        self.previous_field = p.int("m_nFieldOutputPrev", F.PREVIOUS_POSITION)
        self.reset()

    def reset(self):
        self.last = None

    def operate(self, sim, v, dt, strength):
        rotation, translation = self.transform(sim)
        if self.last is None:
            self.last = (rotation.copy(), translation.copy())
        last_rotation, last_translation = self.last
        self.last = (rotation.copy(), translation.copy())
        delta = translation - last_translation
        turn = rotation @ last_rotation.T if self.lock_rotation else None
        if not delta.any() and (turn is None or np.allclose(turn, np.eye(3))):
            if self.lock_rotation:
                normal = self.transform.orientation(sim)
                if normal is not None:
                    v.set_field(F.NORMAL, normal)
            return
        if self.start_min >= 1.0:
            fade = np.ones(v.n)
        else:
            t = v.normalized_age
            start = lerp(self.start_min, self.start_max, sim.random.next(v.n))
            end = lerp(self.end_min, self.end_max, sim.random.next(v.n))
            fade = np.where(t <= start, 1.0, 1.0 - saturate(remap(t, start, end)))
        born = np.minimum(v.get("age"), dt) / dt if dt > 0 else np.ones(v.n)
        amount = (fade * strength * born)[:, None]
        if self.lock_rotation:
            position, previous = v.field(self.field), v.field(self.previous_field)
            moved = (position - last_translation) @ turn.T + translation
            moved_previous = (previous - last_translation) @ turn.T + translation
            v.set_field(self.previous_field, previous + (moved_previous - previous) * amount * self.previous_scale)
            v.set_field(self.field, position + (moved - position) * amount)
        else:
            v.set_field(self.previous_field, v.field(self.previous_field) + delta * amount * self.previous_scale)
            v.set_field(self.field, v.field(self.field) + delta * amount)


@register(OPERATORS, "C_OP_VectorNoise")
class VectorNoise(Operator):
    """Sets (or adds per second to) a vector attribute from value noise over the particle's position."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.COLOR)
        self.low = p.vector("m_vecOutputMin")
        self.high = p.vector("m_vecOutputMax", (1, 1, 1))
        self.scale = p.float("m_fl4NoiseScale", 0.1)
        self.time_scale = p.float("m_flNoiseAnimationTimeScale")
        self.additive = p.bool("m_bAdditive")

    def operate(self, sim, v, dt, strength):
        half = (self.high - self.low) * 0.5
        sample = v.get("position") * self.scale + np.array([self.time_scale * sim.age, 0.0, 0.0])
        value = noise.vector3(sample) * half + half + self.low
        if not self.additive:
            v.set_field(self.field, value)
            return
        value = value * dt * strength
        v.set_field(self.field, v.field(self.field) + value)
        if self.field == F.POSITION:
            v.set("previous", v.get("previous") + value)


@register(OPERATORS, "C_OP_MaintainSequentialPath")
class MaintainSequentialPath(Operator):
    """Pulls particles towards evenly spaced places along a path between control points, in order."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.max_distance = p.number("m_fMaxDistance", 0.0)
        self.count = p.number("m_flNumToAssign", 100.0)
        self.cohesion = p.number("m_flCohesionStrength", 1.0)
        self.loop = p.bool("m_bLoop", True)
        self.use_count = p.bool("m_bUseParticleCount")
        self.path = _path_parameters(p)
        self.reset()

    def reset(self):
        self.slot = 0
        self.slot_step = 1

    def operate(self, sim, v, dt, strength):
        count = v.n if self.use_count else float(self.count(sim))
        step = 0.0 if count <= 1 else 1.0 / (count - 1.0)
        # The slot walk carries on from step to step: each particle takes the next one along the path.
        slots = np.zeros(v.n)
        for i in range(v.n):
            if self.slot >= count or self.slot < 0:
                if self.loop:
                    self.slot = 0
                else:
                    self.slot_step = -self.slot_step
                    self.slot = max(min(self.slot, int(count) - 1), 1)
            slots[i] = self.slot
            self.slot += self.slot_step
        points = path_points(sim, self.path, self.path["start"], self.path["end"])
        target = path_evaluate(points, slots * step)
        max_offset = float(self.max_distance(sim))
        keep = 1.0 - float(self.cohesion(sim))
        for key in ("position", "previous"):
            offset = v.get(key) - target
            distance = np.linalg.norm(offset, axis=1, keepdims=True)
            direction = np.divide(offset, distance, out=np.zeros_like(offset), where=distance > 0)
            v.set(key, target + direction * np.minimum(distance, max_offset) * keep)


@register(OPERATORS, "C_OP_DistanceToTransform")
class DistanceToTransform(Operator):
    """Sets a float attribute from the particle's distance to a transform, remapped (the engine remaps the squared
    distance between the squared ends of the range)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.input_min = p.number("m_flInputMin", 0.0)
        self.input_max = p.number("m_flInputMax", 128.0)
        self.output_min = p.number("m_flOutputMin", 0.0)
        self.output_max = p.number("m_flOutputMax", 1.0)
        self.transform = p.transform("m_TransformStart", Transform(0))
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        self.active_range = p.bool("m_bActiveRange")
        self.additive = p.bool("m_bAdditive")
        self.scale = p.vec("m_vecComponentScale", (1.0, 1.0, 1.0))
        if p.bool("m_bLOS"):
            p.note("C_OP_DistanceToTransform line of sight")

    def operate(self, sim, v, dt, strength):
        n = v.n
        origin = self.transform(sim)[1]
        low, high = column(self.input_min(sim, v), n) ** 2, column(self.input_max(sim, v), n) ** 2
        out_low, out_high = column(self.output_min(sim, v), n), column(self.output_max(sim, v), n)
        if self.field in (F.ALPHA, F.ALPHA_ALTERNATE):
            out_low, out_high = saturate(out_low), saturate(out_high)
        delta = (v.get("position") - origin) * rows3(self.scale(sim, v), n)
        distance = (delta * delta).sum(axis=1)
        inside = ~self.active_range | ((distance <= high) & (low <= distance))
        divisor = np.where(high == low, 1.0, high - low)
        value = lerp(out_low, out_high, saturate((distance - low) / divisor))
        current = v.field(self.field)
        change = (set_by_method(v, self.field, value, self.method, dt) - current) * np.where(inside, strength, 0.0)
        v.set_field(self.field, current + (current + change) if self.additive else current + change)


@register(OPERATORS, "C_OP_SetToCP")
class SetToCP(Operator):
    """Puts every particle at a control point (plus an offset, in its frame when local)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.control_point = p.int("m_nControlPointNumber")
        self.offset = p.vector("m_vecOffset")
        self.local = p.bool("m_bOffsetLocal")

    def operate(self, sim, v, dt, strength):
        point = sim.control_point(self.control_point)
        offset = point.rotation @ self.offset if self.local and point.oriented else self.offset
        v.set("position", point.position + offset)
        v.set("previous", point.position + offset)


@register(OPERATORS, "C_OP_MaxVelocity")
class MaxVelocity(Operator):
    """Holds particles' speed between a minimum and a maximum."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.maximum = p.number("m_flMaxVelocity", 0.0)
        self.minimum = p.number("m_flMinVelocity", 0.0)

    def operate(self, sim, v, dt, strength):
        velocity = v.get("velocity")
        speed = np.linalg.norm(velocity, axis=1)
        high = column(self.maximum(sim, v), v.n)
        low = column(self.minimum(sim, v), v.n)
        target = np.where(speed > high, high, np.where((speed > 0) & (speed < low), low, speed))
        scale = np.divide(target, speed, out=np.ones_like(speed), where=speed > 0)
        clamped = lerp(velocity, velocity * scale[:, None], strength)
        v.set("velocity", clamped)
        v.set("previous", v.get("position") - clamped * dt)


@register(OPERATORS, "C_OP_EndCapTimedDecay")
class EndCapTimedDecay(Operator):
    """Kills every particle a time into the end cap (never reached in a snapshot)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.time = p.number("m_flDecayTime", 0.0)

    def operate(self, sim, v, dt, strength):
        if sim.in_end_cap and sim.end_cap_age > float(self.time(sim)):
            v.kill(np.ones(v.n, bool))


@register(OPERATORS, "C_OP_LerpEndCapScalar")
class LerpEndCapScalar(Operator):
    """Moves a float attribute to a value over a time once the end cap starts (alpha, by default)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.ALPHA)
        self.output = p.float("m_flOutput")
        self.time = p.float("m_flLerpTime", 1.0)

    def operate(self, sim, v, dt, strength):
        if not sim.in_end_cap or F.name(self.field) is None or F.is_vector(self.field):
            return
        # the share of the way left this step, so the value lands on the output at the lerp time
        left = self.time - sim.end_cap_age + dt
        share = 1.0 if left <= dt else dt / left
        current = v.field(self.field)
        v.set_field(self.field, current + (self.output - current) * share * strength)


@register(OPERATORS, "C_OP_RampScalarLinear")
class RampScalarLinear(Operator):
    """Adds a per particle random rate per second to a float attribute over a per particle random window of its life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.rate_min, self.rate_max = p.float("m_RateMin"), p.float("m_RateMax")
        self.start_min, self.start_max = p.float("m_flStartTime_min"), p.float("m_flStartTime_max")
        self.end_min, self.end_max = p.float("m_flEndTime_min", 1.0), p.float("m_flEndTime_max", 1.0)
        self.proportional = p.bool("m_bProportionalOp", True)
        self.field = p.int("m_nField", F.RADIUS)
        self.windowed = (self.start_min, self.start_max, self.end_min, self.end_max) != (0.0, 0.0, 1.0, 1.0) or not self.proportional

    def operate(self, sim, v, dt, strength):
        if F.name(self.field) is None or F.is_vector(self.field):
            return
        ids = v.get("particle_id")
        r = sim.random
        mask = v.get("lifetime") > 0
        if self.windowed:
            window = v.normalized_age if self.proportional else v.get("age")
            start = lerp(self.start_min, self.start_max, r.for_particle(ids, 11))
            end = lerp(self.end_min, self.end_max, r.for_particle(ids, 12))
            mask = mask & (window >= start) & (window < end)
        if not mask.any():
            return
        rate = lerp(self.rate_min, self.rate_max, r.for_particle(ids, 0))
        current = v.field(self.field)
        low, high = _float_range(self.field)
        v.set_field(self.field, np.where(mask, np.clip(current + rate * dt * strength, low, high), current))


@register(OPERATORS, "C_OP_OscillateScalar")
class OscillateScalar(Operator):
    """Adds a sine wave to a float attribute, each particle with its own rate and frequency, over a window of its life."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nField", F.ALPHA)
        self.rate_min, self.rate_max = p.float("m_RateMin"), p.float("m_RateMax")
        self.frequency_min, self.frequency_max = p.float("m_FrequencyMin", 1.0), p.float("m_FrequencyMax", 1.0)
        self.start_min, self.start_max = p.float("m_flStartTime_min"), p.float("m_flStartTime_max")
        self.end_min, self.end_max = p.float("m_flEndTime_min", 1.0), p.float("m_flEndTime_max", 1.0)
        self.multiplier = p.float("m_flOscMult", 2.0)
        self.add = p.float("m_flOscAdd", 0.5)
        self.proportional = p.bool("m_bProportional", True)
        self.proportional_window = p.bool("m_bProportionalOp", True)
        self.clamp = self.field in (F.ALPHA, F.ALPHA_ALTERNATE)

    def operate(self, sim, v, dt, strength):
        if F.name(self.field) is None or F.is_vector(self.field):
            return
        ids = v.get("particle_id")
        r = sim.random
        age = v.normalized_age
        window = age if self.proportional_window else v.get("age")
        start = lerp(self.start_min, self.start_max, r.for_particle(ids, 11))
        end = lerp(self.end_min, self.end_max, r.for_particle(ids, 12))
        mask = (v.get("lifetime") > 0) & (window >= start) & (window < end)
        if not mask.any():
            return
        frequency = lerp(self.frequency_min, self.frequency_max, r.for_particle(ids, 0))
        rate = lerp(self.rate_min, self.rate_max, r.for_particle(ids, 1))
        if self.proportional:
            phase = age * frequency * self.multiplier + self.add
        else:
            phase = frequency * (self.multiplier * sim.age + self.add)
        current = v.field(self.field)
        result = current + np.sin(math.pi * phase) * rate * dt * strength
        if self.clamp:
            result = np.clip(result, 0.0, 1.0)
        v.set_field(self.field, np.where(mask, result, current))


def _axis_rotation(axis, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + (1.0 - c) * np.outer(axis, axis) + s * k


@register(OPERATORS, "C_OP_MovementRotateParticleAroundAxis")
class MovementRotateParticleAroundAxis(Operator):
    """Turns particles (both ends of their position pair, so it doesn't read back as velocity) about an axis through
    a transform's origin; the axis is the world's unless local, then it follows the transform's orientation."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.axis = p.vec("m_vecRotAxis", (0.0, 0.0, 1.0))
        self.rate = p.number("m_flRotRate", 180.0)
        self.transform = p.transform("m_TransformInput", Transform(0))
        self.local = p.bool("m_bLocalSpace")

    def operate(self, sim, v, dt, strength):
        axis = np.asarray(self.axis(sim), np.float64)
        length = np.linalg.norm(axis)
        axis = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
        rotation, center = self.transform(sim)
        if self.local:
            axis = rotation @ axis
        turn = _axis_rotation(axis, math.radians(float(np.mean(self.rate(sim)))) * dt)
        for name in ("position", "previous"):
            points = v.get(name)
            rotated = (points - center) @ turn.T + center
            v.set(name, lerp(points, rotated, strength))


@register(OPERATORS, "C_OP_RemapTransformOrientationToRotations")
class RemapTransformOrientationToRotations(Operator):
    """Writes a transform's orientation (plus an offset in degrees) to every particle's pitch, yaw and roll, or its
    forward direction to their normal."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.orientation = TransformOrientation(p)

    def operate(self, sim, v, dt, strength):
        self.orientation.write(sim, v)


@register(OPERATORS, "C_OP_NormalLock")
class NormalLock(Operator):
    """Turns particles' normals by however much a control point's rotation changed since the last step, so they stay
    locked to its frame; a particle younger than the step gets the share of it that it was alive for."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.transform = Transform(p.int("m_nControlPointNumber"))
        self.reset()

    def reset(self):
        self.last = None

    def operate(self, sim, v, dt, strength):
        rotation = self.transform(sim)[0].copy()
        last = rotation if self.last is None else self.last
        self.last = rotation
        delta = rotation @ last.T
        if np.allclose(delta, np.eye(3)):
            return
        ratio = (np.minimum(v.get("age"), dt) / dt)[:, None] if dt > 0 else 1.0
        normal = v.get("normal")
        v.set("normal", lerp(normal, normal @ delta.T, ratio))


@register(OPERATORS, "C_OP_RemapVelocityToVector")
class RemapVelocityToVector(Operator):
    """Writes each particle's velocity over the last step to a vector attribute: per second and scaled, or with
    m_bNormalize only its direction times the scale (particles that haven't moved keep their value)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.POSITION)
        self.scale = p.float("m_flScale", 1.0)
        self.normalize = p.bool("m_bNormalize")

    def operate(self, sim, v, dt, strength):
        if self.field in (F.POSITION, F.PREVIOUS_POSITION) or not F.is_vector(self.field) or F.name(self.field) is None:
            return
        step = v.get("position") - v.get("previous")
        if self.normalize:
            length = np.linalg.norm(step, axis=1, keepdims=True)
            moved = length[:, 0] > 0
            direction = np.divide(step, length, out=np.zeros_like(step), where=length > 0)
            current = v.field(self.field)
            v.set_field(self.field, np.where(moved[:, None], direction * self.scale, current))
        else:
            frame = sim.previous_frame_time if sim.previous_frame_time > 0 else dt
            v.set_field(self.field, step * (self.scale / frame))


@register(OPERATORS, "C_OP_CalculateVectorAttribute")
class CalculateVectorAttribute(Operator):
    """Writes start + two scaled attributes + two scaled control point positions, times a per-axis scale, to a vector
    attribute (normalized when it is the normal). Defaults and maths read from Deadlock's particles.dll: by default
    normal = position - CP0, the direction out from the centre."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.start = p.vector("m_vStartValue")
        self.inputs = [(p.int("m_nFieldInput1", F.POSITION), p.float("m_flInputScale1", 1.0)),
                       (p.int("m_nFieldInput2", F.POSITION), p.float("m_flInputScale2", 0.0))]
        self.points = [(self._point(p, "m_nControlPointInput1"), p.float("m_flControlPointScale1", -1.0)),
                       (self._point(p, "m_nControlPointInput2"), p.float("m_flControlPointScale2", 0.0))]
        self.field = p.int("m_nFieldOutput", F.NORMAL)
        self.final_scale = p.vector("m_vFinalOutputScale", (1.0, 1.0, 1.0))

    @staticmethod
    def _point(p: Params, key: str) -> int:
        # The game's operate ignores the reference's m_vOffsetFromControlPoint (no Deadlock vpcf sets one).
        return p.nested(p.get(key)).int("m_controlPointNameString", 0)

    def operate(self, sim, v, dt, strength):
        if not F.is_vector(self.field) or F.name(self.field) is None:
            return
        constant = self.start.copy()
        for point, scale in self.points:
            if scale != 0 and point >= 0:
                constant += sim.control_point(point).position * scale
        value = np.broadcast_to(constant, (v.n, 3)).copy()
        for field, scale in self.inputs:
            if scale != 0:
                value += np.reshape(v.field(field), (v.n, -1)) * scale
        value *= self.final_scale
        if self.field == F.NORMAL:
            length = np.linalg.norm(value, axis=1, keepdims=True)
            value = np.divide(value, length, out=np.zeros_like(value), where=length > 1e-12)
        v.set_field(self.field, value)


@register(OPERATORS, "C_OP_RemapSpeed")
class RemapSpeed(Operator):
    """Sets a float attribute from each particle's speed over the last step (units per second, remapped and clamped,
    ported from CS:GO's builtin_particle_ops.cpp)."""

    # CS:GO's MIN_PARTICLE_SPEED: input ends below it are raised to it
    MIN_SPEED = 0.001

    def __init__(self, p: Params):
        super().__init__(p)
        self.field = p.int("m_nFieldOutput", F.RADIUS)
        self.input_range = (max(self.MIN_SPEED, p.float("m_flInputMin", 0.0)),
                            max(self.MIN_SPEED, p.float("m_flInputMax", 1.0)))
        output_range = (p.float("m_flOutputMin", 0.0), p.float("m_flOutputMax", 1.0))
        if self.field in (F.ALPHA, F.ALPHA_ALTERNATE):
            output_range = tuple(min(max(x, 0.0), 1.0) for x in output_range)
        self.output_range = output_range
        self.method = p.enum("m_nSetMethod", SET_METHODS, SET_REPLACE)
        if p.bool("m_bIgnoreDelta"):
            p.note("C_OP_RemapSpeed m_bIgnoreDelta")

    def operate(self, sim, v, dt, strength):
        if dt <= 0 or F.name(self.field) is None or F.is_vector(self.field):
            return
        speed = np.linalg.norm(v.get("position") - v.get("previous"), axis=1) / dt
        value = remap_clamped(speed, *self.input_range, *self.output_range)
        current = v.field(self.field)
        v.set_field(self.field, lerp(current, set_by_method(v, self.field, value, self.method, dt), strength))


# Pre-emission operators -------------------------------------------------------------------------------------------

def heading_rotation(forward):
    """VectorVectors: a rotation (columns forward, left, up) whose forward is the given direction, its up as near
    world up as it goes; None for a zero direction."""
    length = np.linalg.norm(forward)
    if length < 1e-9:
        return None
    forward = forward / length
    if abs(forward[0]) < 1e-6 and abs(forward[1]) < 1e-6:
        right = np.array([0.0, -1.0, 0.0])
        up = np.array([-forward[2], 0.0, 0.0])
    else:
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
    return np.stack([forward, -right, up], 1)


def particle_heading(v, row: int, field: int = F.PREVIOUS_POSITION):
    """A particle's rotation for a control point: along its motion (position - previous position), or along a vector
    attribute."""
    if field in (F.POSITION, F.PREVIOUS_POSITION) or not F.is_vector(field):
        return heading_rotation(v.get("position")[row] - v.get("previous")[row])
    return heading_rotation(v.field(field)[row])


def children_in_group(sim, group: int):
    return [child for child in sim.children if child.definition.group == group]


@register(OPERATORS, "C_OP_SetChildControlPoints")
class SetChildControlPoints(Operator):
    """Copies the positions of particles first..first+count to control points of every child in a group (ported
    from CS:GO's builtin_particle_ops.cpp; m_bReverse and m_nOrientation aren't in it and aren't read)."""

    def __init__(self, p):
        super().__init__(p)
        self.group = p.int("m_nChildGroupID")
        self.first_point = p.int("m_nFirstControlPoint")
        self.count = p.int("m_nNumControlPoints", 1)
        self.first_source = p.number("m_nFirstSourcePoint", 0.0)
        self.orient = p.bool("m_bSetOrientation")
        if p.get("m_bReverse"):
            p.note("C_OP_SetChildControlPoints m_bReverse")

    def operate(self, sim, v, dt, strength):
        first_point = min(max(self.first_point, 0), 63)
        source = int(float(self.first_source(sim)))
        count = min(v.n - source, self.count, 64 - first_point)
        if count <= 0 or source < 0:
            return
        positions = v.get("position")
        for child in children_in_group(sim, self.group):
            for i in range(count):
                rotation = particle_heading(v, source + i) if self.orient else None
                sim.set_child_control_point(child, first_point + i, positions[source + i], rotation)


@register(OPERATORS, "C_OP_SetPerChildControlPoint")
class SetPerChildControlPoint(Operator):
    """Sets one control point on each child of a group from its own particle (child i from particle first +
    i * increment), for as many children as m_nNumControlPoints or, with m_bNumBasedOnParticleCount, particles."""

    def __init__(self, p):
        super().__init__(p)
        self.group = p.int("m_nChildGroupID")
        self.point = p.int("m_nFirstControlPoint")
        self.count = p.int("m_nNumControlPoints", 1)
        self.first_source = p.number("m_nFirstSourcePoint", 0.0)
        self.increment = p.number("m_nParticleIncrement", 1.0)
        self.by_particle_count = p.bool("m_bNumBasedOnParticleCount")
        self.orient = p.bool("m_bSetOrientation")
        # The default field isn't known: the ones that orient leave it out, and motion is what CS:GO's version used.
        self.orientation_field = p.int("m_nOrientationField", F.PREVIOUS_POSITION)

    def operate(self, sim, v, dt, strength):
        children = children_in_group(sim, self.group)
        source = int(float(self.first_source(sim)))
        increment = max(1, int(float(self.increment(sim))))
        count = min(v.n if self.by_particle_count else self.count, len(children))
        if not 0 <= self.point < 64:
            return
        positions = v.get("position")
        for i in range(count):
            row = source + i * increment
            if not 0 <= row < v.n:
                break
            rotation = particle_heading(v, row, self.orientation_field) if self.orient else None
            sim.set_child_control_point(children[i], self.point, positions[row], rotation)


class PreEmission(Function):
    def __init__(self, p: Params):
        super().__init__(p)
        self.run_once = p.bool("m_bRunOnce")
        self.has_run = False
        self.set_once = p.bool("m_bSetOnce")
        self.done = False

    def reset(self):
        self.has_run = False
        self.done = False

    def operate(self, sim, dt: float):
        raise NotImplementedError


@register(PRE_EMISSION, "C_OP_StopAfterCPDuration")
class StopAfterCPDuration(PreEmission):
    """Stops the effect once the system is older than a duration: its end cap plays (unless turned off), and with
    m_bDestroyImmediately every particle in the tree goes at once."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.duration = p.number("m_flDuration", 1.0)
        self.destroy = p.bool("m_bDestroyImmediately")
        self.play_end_cap = p.bool("m_bPlayEndCap", True)

    def operate(self, sim, dt):
        if sim.in_end_cap or sim.emission_stopped:
            return
        if sim.age > float(self.duration(sim)):
            sim.stop_emission(self.destroy, self.play_end_cap)


@register(PRE_EMISSION, "C_OP_RemapSpeedtoCP")
class RemapSpeedtoCP(PreEmission):
    """Writes how fast one control point moves (units per second, remapped and clamped) into a component of another
    (ported from CS:GO's builtin_particle_ops.cpp). A point that never moves in an effect played on its own counts
    as moving at the top of the input range."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.source = p.int("m_nInControlPointNumber", 0)
        self.target = p.int("m_nOutControlPointNumber", -1)
        # the game's guard against a point feeding itself
        if self.target == self.source:
            self.target = -1
        self.component = p.int("m_nField", 0)
        self.input_range = (p.float("m_flInputMin", 0.0), p.float("m_flInputMax", 1.0))
        self.output_range = (p.float("m_flOutputMin", 0.0), p.float("m_flOutputMax", 1.0))
        if p.get("m_bUseDeltaV"):
            p.note("C_OP_RemapSpeedtoCP m_bUseDeltaV")

    def operate(self, sim, dt):
        if not 0 <= self.target < 64:
            return
        point = sim.control_point(self.source)
        speed = point.speed(sim)
        if not point.moved and sim.root.unset_value:
            # Played on its own (a rig's clip effect): the game code that moves the caster (dashes, speed buffs on
            # an in-place clip) isn't running, so show it as the game does while moving.
            speed = max(self.input_range)
        value = float(remap_clamped(speed, *self.input_range, *self.output_range))
        sim.set_control_point_component(self.target, self.component, value)


@register(PRE_EMISSION, "C_OP_SetSingleControlPointPosition")
class SetSingleControlPointPosition(PreEmission):
    """Places a control point at a position in a transform's frame."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.control_point = p.int("m_nCP1", 1)
        self.position = p.vec("m_vecCP1Pos", (128.0, 0.0, 0.0))
        self.transform = p.transform("m_transformInput", Transform(0))

    def operate(self, sim, dt):
        if self.set_once and self.done:
            return
        sim.set_control_point(self.control_point, transform_points(self.transform(sim), self.position(sim)))
        self.done = True


@register(PRE_EMISSION, "C_OP_SetControlPointPositions")
class SetControlPointPositions(PreEmission):
    """Places four control points at positions in another control point's frame (or the world)."""

    def __init__(self, p: Params):
        super().__init__(p)
        defaults = ((128, 0, 0), (0, 128, 0), (-128, 0, 0), (0, -128, 0))
        self.points = [(p.int(f"m_nCP{i + 1}", i + 1), p.vector(f"m_vecCP{i + 1}Pos", defaults[i])) for i in range(4)]
        self.world = p.bool("m_bUseWorldLocation")
        self.head = p.int("m_nHeadLocation")

    def operate(self, sim, dt):
        if self.set_once and self.done:
            return
        transform = (np.eye(3), np.zeros(3)) if self.world else Transform(self.head)(sim)
        for index, position in self.points:
            sim.set_control_point(index, transform_points(transform, position))
        self.done = True


@register(PRE_EMISSION, "C_OP_SetControlPointFieldToScalarExpression")
class SetControlPointFieldToScalarExpression(PreEmission):
    """Sets one component of a control point to an expression of two float inputs."""

    EXPRESSIONS = {name: i for i, name in enumerate(
        ("SCALAR_EXPRESSION_ADD", "SCALAR_EXPRESSION_SUBTRACT", "SCALAR_EXPRESSION_MUL", "SCALAR_EXPRESSION_DIVIDE",
         "SCALAR_EXPRESSION_INPUT_1", "SCALAR_EXPRESSION_MIN", "SCALAR_EXPRESSION_MAX", "SCALAR_EXPRESSION_MOD",
         "SCALAR_EXPRESSION_EQUAL", "SCALAR_EXPRESSION_GT", "SCALAR_EXPRESSION_LT"))}

    def __init__(self, p: Params):
        super().__init__(p)
        from .inputs import Mapping
        self.expression = p.enum("m_nExpression", self.EXPRESSIONS, 0)
        self.input1 = p.number("m_flInput1", 1.0)
        self.input2 = p.number("m_flInput2", 0.0)
        self.control_point = p.int("m_nOutputCP", 2)
        self.component = p.int("m_nOutVectorField")
        remap_block = p.get("m_flOutputRemap")
        self.remap = Mapping(p.nested(remap_block)) if isinstance(remap_block, dict) else None

    def operate(self, sim, dt):
        a, b = float(self.input1(sim)), float(self.input2(sim))
        e = self.expression
        value = (a + b if e == 0 else a - b if e == 1 else a * b if e == 2 else (a / b if b else 0.0) if e == 3
                 else a if e == 4 else min(a, b) if e == 5 else max(a, b) if e == 6 else (math.fmod(a, b) if b else 0.0) if e == 7
                 else float(a == b) if e == 8 else float(a > b) if e == 9 else float(a < b) if e == 10 else 0.0)
        if self.remap is not None:
            value = float(np.asarray(self.remap(value)))
        if 0 <= self.component < 3:
            sim.set_control_point_component(self.control_point, self.component, value)


@register(PRE_EMISSION, "C_OP_SetControlPointToVectorExpression")
class SetControlPointToVectorExpression(PreEmission):
    """Sets a control point to an expression of two vector inputs."""

    EXPRESSIONS = {name: i for i, name in enumerate(
        ("VECTOR_EXPRESSION_ADD", "VECTOR_EXPRESSION_SUBTRACT", "VECTOR_EXPRESSION_MUL", "VECTOR_EXPRESSION_DIVIDE",
         "VECTOR_EXPRESSION_INPUT_1", "VECTOR_EXPRESSION_MIN", "VECTOR_EXPRESSION_MAX", "VECTOR_EXPRESSION_CROSSPRODUCT",
         "VECTOR_EXPRESSION_LERP"))}

    def __init__(self, p: Params):
        super().__init__(p)
        self.control_point = p.int("m_nOutputCP", 2)
        self.input1 = p.vec("m_vInput1")
        self.input2 = p.vec("m_vInput2")
        self.lerp = p.number("m_flLerp", 0.0)
        self.expression = p.enum("m_nExpression", self.EXPRESSIONS, 0)

    def operate(self, sim, dt):
        a = np.asarray(self.input1(sim), np.float64)
        b = np.asarray(self.input2(sim), np.float64)
        e = self.expression
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (a + b if e == 0 else a - b if e == 1 else a * b if e == 2 else np.nan_to_num(a / b) if e == 3
                     else a if e == 4 else np.minimum(a, b) if e == 5 else np.maximum(a, b) if e == 6
                     else np.cross(a, b) if e == 7 else lerp(a, b, float(self.lerp(sim))) if e == 8 else np.zeros(3))
        sim.set_control_point(self.control_point, value)


@register(PRE_EMISSION, "C_OP_SetControlPointOrientation")
class SetControlPointOrientation(PreEmission):
    """Turns a control point to given angles (in degrees, between two sets when randomized), in another control
    point's frame unless the world's; the point keeps its position."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.control_point = p.int("m_nCP")
        self.angles = p.vector("m_vecRotation")
        self.angles_b = p.vector("m_vecRotationB")
        self.randomize = p.bool("m_bRandomize")
        self.world = p.bool("m_bUseWorldLocation")
        self.head = p.int("m_nHeadLocation")

    def operate(self, sim, dt):
        if self.set_once and self.done:
            return
        from ..probes import rotation_matrix
        angles = lerp(self.angles, self.angles_b, sim.random.next1()) if self.randomize else self.angles
        rotation = rotation_matrix(angles)
        if not self.world:
            rotation = Transform(self.head)(sim)[0] @ rotation
        sim.set_control_point(self.control_point, sim.control_point(self.control_point).position, rotation)
        self.done = True


@register(PRE_EMISSION, "C_OP_SetControlPointRotation")
class SetControlPointRotation(PreEmission):
    """Spins a control point about an axis at a rate in degrees per second (CS:GO's builtin_particle_ops.cpp; its
    defaults are what Deadlock's vpcfs strip). The game turns the axis by the local point and then still applies it
    in the spun point's own frame (matCP x matRot); kept as is."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.axis = p.vec("m_vecRotAxis", (0.0, 0.0, 1.0))
        self.rate = p.number("m_flRotRate", 180.0)
        self.control_point = p.int("m_nCP", 0)
        self.local_point = p.int("m_nLocalCP", -1)

    def operate(self, sim, dt):
        axis = np.asarray(self.axis(sim), np.float64)
        length = np.linalg.norm(axis)
        if length < 1e-9:
            return
        axis = axis / length
        if self.local_point > -1:
            local = sim.control_point(self.local_point)
            if local.oriented:
                axis = local.rotation @ axis
        point = sim.control_point(self.control_point)
        turn = _axis_rotation(axis, math.radians(float(np.mean(self.rate(sim))) * dt))
        sim.set_control_point(self.control_point, point.position, point.rotation @ turn)


@register(PRE_EMISSION, "C_OP_SetParentControlPointsToChildCP")
class SetParentControlPointsToChildCP(PreEmission):
    """Hands this system's control points out to its children, one each: the i-th child of the group gets point
    first + i as its point m_nChildControlPoint (orientation too if asked), for up to m_nNumControlPoints children.
    Defaults and loop from Deadlock's particles.dll."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.group = p.int("m_nChildGroupID")
        self.child_point = p.int("m_nChildControlPoint")
        self.count = p.int("m_nNumControlPoints", 1)
        self.first_source = p.int("m_nFirstSourcePoint")
        self.orient = p.bool("m_bSetOrientation")

    def operate(self, sim, dt):
        source = self.first_source
        for child in children_in_group(sim, self.group)[:max(self.count, 0)]:
            point = sim.control_point(source)
            sim.set_child_control_point(child, self.child_point, point.position,
                                        point.rotation if self.orient else None)
            source += 1


# Force generators -------------------------------------------------------------------------------------------------

class Force(Function):
    def generate(self, sim, v, dt: float, strength: float):
        raise NotImplementedError


@register(FORCES, "C_OP_AttractToControlPoint")
class AttractToControlPoint(Force):
    """Pulls particles towards a control point, falling off with distance."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.scale = p.vector("m_vecComponentScale", (1, 1, 1))
        self.amount = p.number("m_fForceAmount", 100.0)
        self.amount_min = p.number("m_fForceAmountMin", 0.0)
        self.apply_min = p.bool("m_bApplyMinForce")
        self.falloff = p.float("m_fFalloffPower", 2.0)
        self.transform = p.transform("m_TransformInput", Transform(0, False))

    def generate(self, sim, v, dt, strength):
        target = self.transform(sim)[1]
        difference = target - v.get("position")
        distance = np.linalg.norm(difference, axis=1)
        amount = column(self.amount(sim, v), v.n)
        if self.apply_min:
            amount = np.maximum(amount, self.amount_min(sim, v))
        safe = np.maximum(distance, 1e-6)
        force = difference / safe[:, None] * (amount / safe ** self.falloff)[:, None] * self.scale * strength
        v.set("force", v.get("force") + np.where((distance >= 1e-6)[:, None], force, 0.0))


@register(FORCES, "C_OP_LocalAccelerationForce")
class LocalAccelerationForce(Force):
    """A constant acceleration in a control point's frame."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.control_point = p.int("m_nCP")
        self.scale_control_point = p.int("m_nScaleCP", -1)
        self.acceleration = p.vec("m_vecAccel")

    def generate(self, sim, v, dt, strength):
        force = np.asarray(self.acceleration(sim), np.float64)
        if self.control_point >= 0:
            if self.scale_control_point >= 0:
                force = force * sim.control_point(self.scale_control_point).position
            force = transform_directions(Transform(self.control_point)(sim), force)
        v.set("force", v.get("force") + force * strength)


@register(FORCES, "C_OP_CurlNoiseForce")
class CurlNoiseForce(Force):
    """Swirls particles with curl noise over their position (scrolled over time), scaled per axis."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.kind = p.str("m_nNoiseType", "PARTICLE_DIR_NOISE_PERLIN")
        self.frequency = p.vec("m_vecNoiseFreq", (0.02, 0.02, 0.02))
        self.scale = p.vec("m_vecNoiseScale", (1000.0, 1000.0, 1000.0))
        self.offset = p.vec("m_vecOffset")
        self.rate = p.vec("m_vecOffsetRate")

    def generate(self, sim, v, dt, strength):
        sample = (self.offset(sim, v) + sim.age * np.asarray(self.rate(sim, v))
                  + v.get("position") * self.frequency(sim, v))
        field = noise.curl3(sample) if self.kind == "PARTICLE_DIR_NOISE_CURL" else noise.vector3(sample)
        v.set("force", v.get("force") + field * self.scale(sim, v) * strength)


# Constraints ------------------------------------------------------------------------------------------------------

class Constraint(Function):
    def apply(self, sim, v, dt: float) -> bool:
        raise NotImplementedError


@register(CONSTRAINTS, "C_OP_ConstrainDistance")
class ConstrainDistance(Constraint):
    """Keeps particles between a minimum and maximum distance of a centre (a control point plus an offset)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.minimum = p.number("m_fMinDistance", 0.0)
        self.maximum = p.number("m_fMaxDistance", 100.0)
        self.transform = p.transform("m_nControlPointNumber", Transform(0))
        self.offset = p.vec("m_CenterOffset")
        self.global_center = p.bool("m_bGlobalCenter")

    def apply(self, sim, v, dt):
        offset = np.asarray(self.offset(sim), np.float64)
        center = offset if self.global_center else transform_points(self.transform(sim), offset)
        low = float(self.minimum(sim))
        high = float(self.maximum(sim))
        if high < 0:
            high = np.inf
        delta = v.get("position") - center
        distance = np.linalg.norm(delta, axis=1)
        outside = ((distance < low) | (distance > high)) & (distance > 0)
        if not outside.any():
            return False
        target = np.where(distance < low, low, high)
        scale = np.divide(target, distance, out=np.ones_like(distance), where=distance > 0)
        v.set("position", np.where(outside[:, None], center + delta * scale[:, None], v.get("position")))
        return True


@register(CONSTRAINTS, "C_OP_ConstrainDistanceToPath")
class ConstrainDistanceToPath(Constraint):
    """Keeps particles within a distance of a point travelling along a path between control points: the point is at
    age / travel time along it (or a field's value, m_nManualTField), the allowed distance a quadratic from the
    start through the middle to the end radius. Defaults and maths from Deadlock's particles.dll (CS:GO's version
    plus the two fields; field 19 means none). The game skips the varying radius for a group of four particles when
    any of them is past the largest radius; here every particle gets its own."""

    UNSET_FIELD = 19

    def __init__(self, p: Params):
        super().__init__(p)
        self.minimum = p.float("m_fMinDistance", 0.0)
        self.radius_start = p.float("m_flMaxDistance0", 100.0)
        self.radius_mid = p.float("m_flMaxDistanceMid", -1.0)
        self.radius_end = p.float("m_flMaxDistance1", -1.0)
        self.travel_time = p.float("m_flTravelTime", 10.0)
        self.scale_field = p.int("m_nFieldScale", self.UNSET_FIELD)
        self.manual_field = p.int("m_nManualTField", self.UNSET_FIELD)
        self.path = _path_parameters(p, "m_PathParameters")

    def _uses(self, field):
        return field >= 0 and field != self.UNSET_FIELD

    def apply(self, sim, v, dt):
        if v.n == 0:
            return False
        if self._uses(self.manual_field):
            t = np.asarray(v.field(self.manual_field), np.float64)
        else:
            travel = np.full(v.n, self.travel_time)
            if self._uses(self.scale_field):
                travel = travel * v.field(self.scale_field)
            t = np.minimum(1.0, (sim.age - v.get("creation_time")) / np.maximum(travel, 1.1920929e-07))
        points = path_points(sim, self.path, self.path["start"], self.path["end"])
        center = path_evaluate(points, t)
        mid = self.radius_mid if self.radius_mid >= 0 else self.radius_start
        end = self.radius_end if self.radius_end >= 0 else mid
        first = self.radius_start + (mid - self.radius_start) * t
        second = mid + (end - mid) * t
        radius = first + (second - first) * t
        delta = v.get("position") - center
        distance = np.linalg.norm(delta, axis=1)
        too_close = distance < self.minimum
        too_far = ~too_close & (distance > radius)
        if not (too_close | too_far).any():
            return False
        direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
        target = np.where(too_close, self.minimum, radius)
        moved = too_close | too_far
        v.set("position", np.where(moved[:, None], center + direction * target[:, None], v.get("position")))
        return True


ROTATION_LOCKS = {"PARTICLE_ROTATION_LOCK_NONE": 0, "PARTICLE_ROTATION_LOCK_ROTATIONS": 1,
                  "PARTICLE_ROTATION_LOCK_NORMAL": 2}


@register(OPERATORS, "C_OP_LockToBone")
class LockToBone(Operator):
    """Keeps particles on the bone of the hitbox they were created in: each step they move by the bone's motion,
    fading out over their life (a rigid lock puts them on the bone's point outright)."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.model_cp = models_control_point(p)
        self.transform = p.transform("m_transformInput")
        self.fade_start = p.float("m_flLifeTimeFadeStart")
        self.fade_end = p.float("m_flLifeTimeFadeEnd")
        self.jump_threshold = p.float("m_flJumpThreshold", 100.0)
        self.previous_scale = p.float("m_flPrevPosScale", 1.0)
        self.set_name = p.str("m_HitboxSetName", "default")
        self.rigid = p.bool("m_bRigid")
        self.use_bones = p.bool("m_bUseBones")
        self.rotation_type = p.enum("m_nRotationSetType", ROTATION_LOCKS, 0)
        self.rigid_rotation = p.bool("m_bRigidRotationLock")
        self.rotation_lerp = p.number("m_flRotLerp", 1.0)
        if self.rotation_type == 1:
            p.note("C_OP_LockToBone rotation lock of the rotation angles")
        self._followed = None

    def operate(self, sim, v, dt, strength):
        point = sim.control_point(self.model_cp) if self.model_cp >= 0 else None
        if dt <= 0 or point is None or point.model is None or v.n == 0:
            return
        # a frame split into substeps would move the particles by the bones' motion once per substep
        if not self.rigid and point.model["bones"] is self._followed:
            return
        self._followed = point.model["bones"]
        boxes = models.hitbox_list(point.model, self.set_name, self.use_bones)
        index = v.get("hitbox_index").astype(np.int64)
        age = v.get("age")
        _, position = self.transform(sim)
        jumped = False
        if self.transform.control_point is not None:
            moved = position - sim.control_point(self.transform.control_point).previous_position
            jumped = float(moved @ moved) > self.jump_threshold ** 2
        fade_end = self.fade_end if self.fade_end > 0 else np.inf
        active = (index >= 0) & (index < len(boxes)) & ((age < fade_end) | jumped)
        if not active.any():
            return
        rows = np.flatnonzero(active)
        idx = index[rows]
        current = models.matrices(point.model["bones"], boxes)
        before = models.matrices(point.previous_bones or point.model["bones"], boxes)
        local = v.get("hitbox_offset")[rows]
        target = models.transform_points(current[idx], local)
        step = np.minimum(age[rows], dt) / dt
        inverse = 1.0 / (self.fade_end - self.fade_start) if self.fade_end > self.fade_start else 0.0
        weight = (1.0 - (age[rows] - self.fade_start) * inverse) * strength
        positions, previous = v.get("position"), v.get("previous")
        if self.rigid:
            positions[rows] = target
            previous[rows] = target
        else:
            delta = (target - models.transform_points(before[idx], local)) * step[:, None]
            if not jumped:
                delta = np.where((self.fade_start < age[rows])[:, None], delta * weight[:, None], delta)
            positions[rows] += delta
            previous[rows] += delta * (1.0 if jumped else self.previous_scale)
        if self.rotation_type == 2:
            self.lock_normal(sim, v, rows, idx, current, before, step, weight)

    def lock_normal(self, sim, v, rows, idx, current, before, step, weight):
        now = unit_rotation(current[idx])
        if self.rigid_rotation:
            target = now[:, :, 0]
            blend = column(self.rotation_lerp(sim, v), v.n)[rows]
        else:
            target = np.einsum("nij,nj->ni", now @ np.swapaxes(unit_rotation(before[idx]), 1, 2), v.get("normal")[rows])
            blend = step if self.rigid else step * weight
        normals = v.get("normal")
        normals[rows] += (target - normals[rows]) * blend[:, None]


def unit_rotation(matrices):
    basis = matrices[:, :3, :3]
    length = np.linalg.norm(basis, axis=1, keepdims=True)
    return basis / np.maximum(length, 1e-9)
