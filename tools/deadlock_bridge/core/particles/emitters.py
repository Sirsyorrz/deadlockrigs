"""Emitters: when particles are born. Each returns the ages (time already lived this step) of the particles it emits
in a step, which the simulation then spawns and initializes."""
import math

import numpy as np

from . import noise
from .params import Params
from .system import Function

EMITTERS = {}


def emitter(name):
    def register(cls):
        EMITTERS[name] = cls
        return cls
    return register


class Emitter(Function):
    def __init__(self, p: Params):
        super().__init__(p)
        self.start_age = 0.0
        self.finished = False
        # Emitting from a control point's snapshot: none exist here, which leaves such an emitter silent.
        self.snapshot = p.int("m_nSnapshotControlPoint", -1) >= 0

    def start(self, sim):
        self.start_age = sim.age
        self.finished = False

    def emit(self, sim, dt: float, strength: float):
        raise NotImplementedError


def _window(frame_start, frame_end, start, duration):
    """The part of a step the emitter is active for, or None. A duration of 0 runs until stopped."""
    if start > frame_end:
        return None
    low, high = frame_start, frame_end
    if duration != 0.0:
        low = max(start, low)
        high = min(start + duration, high)
    return (low, high) if high > low else None


@emitter("C_OP_ContinuousEmitter")
class ContinuousEmitter(Emitter):
    """Emits at a rate (m_flEmitRate per second) from a start time, for a duration or forever. A sub-step
    accumulator spawns whole particles and spreads their birth times evenly over the step."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.duration = p.number("m_flEmissionDuration", 0.0)
        self.start_time = p.number("m_flStartTime", 0.0)
        self.rate = p.number("m_flEmitRate", 100.0)
        self.per_parent = p.float("m_flScalePerParentParticle")
        self.emission_scale = max(0.0, p.float("m_flEmissionScale")) if p.version < 2 else 0.0
        self.limit = p.int("m_nLimitPerUpdate")
        self.force_first = p.bool("m_bForceEmitOnFirstUpdate")

    def start(self, sim):
        super().start(sim)
        self.pending = 0.0
        self.flushed = 0

    def emit(self, sim, dt, strength):
        if self.finished:
            return None
        elapsed = sim.age - self.start_age
        start = float(self.start_time(sim))
        duration = float(self.duration(sim))
        ages = None
        window = _window(elapsed - dt, elapsed, start, duration)
        if window is not None:
            rate = float(self.rate(sim)) * strength
            if self.per_parent > 0:
                rate *= (sim.parent.particles.count if sim.parent is not None else 1) * self.per_parent
            elif self.emission_scale > 0:
                rate *= (max(sim.root._control_points, default=0) + 1) * self.emission_scale
            if self.snapshot:
                rate = 0.0
            if rate > 0:
                minimum = 1 if self.force_first and abs(sim.age - dt) < 1e-9 else 0
                ages = self._charge(rate, window, elapsed, minimum)
        if duration != 0.0 and elapsed > start + duration:
            self.finished = True
        return ages

    def _charge(self, rate, window, frame_end, minimum):
        low, high = window
        self.pending += max(0.0, rate) * (high - low)
        count = max(int(math.floor(self.pending + 0.001)) - self.flushed, minimum)
        if self.limit > 0:
            count = min(count, self.limit)
        if count <= 0:
            return None
        self.flushed += count
        step = (high - low) / count
        born = np.minimum(low + (np.arange(count) + 1) * step, high)
        return frame_end - born


@emitter("C_OP_NoiseEmitter")
class NoiseEmitter(ContinuousEmitter):
    """Emits at a rate that wanders between m_flOutputMin and m_flOutputMax by noise over the collection's time
    (plus, with m_nWorldNoisePoint, that point's position). Defaults, maths and noise (noise.game3) from
    particles.dll."""

    def __init__(self, p: Params):
        Emitter.__init__(self, p)
        self.duration = max(0.0, p.float("m_flEmissionDuration"))
        self.start_time = p.float("m_flStartTime")
        self.emission_scale = p.float("m_flEmissionScale")
        self.scale_point = p.int("m_nScaleControlPoint", -1)
        self.scale_component = p.int("m_nScaleControlPointField")
        self.world_point = p.int("m_nWorldNoisePoint", -1)
        self.absolute = p.bool("m_bAbsVal")
        self.absolute_invert = p.bool("m_bAbsValInv")
        self.offset = p.float("m_flOffset")
        self.low = p.float("m_flOutputMin")
        self.high = p.float("m_flOutputMax", 100.0)
        self.time_scale = p.float("m_flNoiseScale", 0.1)
        self.world_scale = p.float("m_flWorldNoiseScale", 0.001)
        self.world_offset = p.vector("m_vecOffsetLoc")
        self.world_time_scale = p.float("m_flWorldTimeScale")
        self.limit = 0

    def start(self, sim):
        super().start(sim)
        # The game's accumulator starts at one: the first emitting step always gives a particle.
        self.pending = 1.0

    def rate(self, sim) -> float:
        coord = np.full(3, (sim.age + self.offset) * self.time_scale)
        if self.world_point != -1:
            coord += (sim.control_point(self.world_point).position + self.world_offset) * self.world_scale
        # Plat_MSTime in the game: wall clock milliseconds, here the system's own clock.
        coord += sim.age * 1000.0 * self.world_time_scale
        value = float(noise.game3(coord)[0])
        if self.absolute:
            value = abs(value)
            base, span = self.low, self.high - self.low
        else:
            base, span = self.low + 0.5 * (self.high - self.low), 0.5 * (self.high - self.low)
        if self.absolute_invert:
            value = 1.0 - value
        rate = max(0.0, base + span * value)
        if self.scale_point >= 0 and self.scale_component != -1:
            position = sim.control_point(self.scale_point).position
            rate *= max(0.0, float(position[self.scale_component])) if 0 <= self.scale_component <= 2 else 0.0
        if self.emission_scale != 0.0:
            rate *= max(sim.root._control_points, default=0) * self.emission_scale
        return rate

    def emit(self, sim, dt, strength):
        if self.finished:
            return None
        elapsed = sim.age - self.start_age
        ages = None
        window = _window(elapsed - dt, elapsed, self.start_time, self.duration)
        if window is not None and not self.snapshot:
            rate = self.rate(sim) * strength
            if rate > 0:
                ages = self._charge(rate, window, elapsed, 0)
        if self.duration != 0.0 and elapsed - dt > self.start_time + self.duration:
            self.finished = True
        return ages


@emitter("C_OP_InstantaneousEmitter")
class InstantaneousEmitter(Emitter):
    """Emits a burst (m_nParticlesToEmit) at a start time, at most m_nMaxEmittedPerFrame a step; optionally scaled by
    the parent's particle count, or repeated for every parent particle that died."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.count = p.number("m_nParticlesToEmit", 100.0)
        self.start_time = p.number("m_flStartTime", 0.0)
        self.parent_scale = p.number("m_flParentParticleScale", -1.0)
        self.max_per_frame = p.int("m_nMaxEmittedPerFrame", -1)
        self.from_killed = p.float("m_flInitFromKilledParentParticles")

    def start(self, sim):
        super().start(sim)
        self.remaining = 0
        self.resolved = False

    def emit(self, sim, dt, strength):
        if self.finished:
            return None
        elapsed = sim.age - self.start_age
        start = float(self.start_time(sim))
        if elapsed < start:
            return None
        if self.from_killed > 0:
            self.remaining = self._resolve(sim)
        elif not self.resolved:
            self.resolved = True
            self.remaining = self._resolve(sim)
        cap = self.max_per_frame if self.max_per_frame >= 0 else min(sim.particles.capacity, 20000)
        claimed = min(self.remaining, cap)
        self.remaining -= claimed
        if self.from_killed <= 0 and self.remaining <= 0:
            self.finished = True
        count = int(claimed * strength)
        return np.full(count, elapsed - start) if count > 0 else None

    def _resolve(self, sim) -> int:
        if self.snapshot:
            return 0
        count = float(self.count(sim))
        parent = sim.parent
        if self.from_killed > 0:
            return int((parent.killed_last if parent is not None else 0) * count * self.from_killed)
        if parent is not None:
            scale = float(self.parent_scale(sim))
            if scale != -1.0:
                count *= parent.particles.count * scale
        return int(count)


@emitter("C_OP_MaintainEmitter")
class MaintainEmitter(Emitter):
    """Keeps a number of particles alive (m_nParticlesToMaintain), replacing the ones that die, from a start time."""

    def __init__(self, p: Params):
        super().__init__(p)
        self.count = p.number("m_nParticlesToMaintain", 100.0)
        self.start_time = p.float("m_flStartTime")
        self.rate = p.float("m_flEmissionRate")

    def emit(self, sim, dt, strength):
        elapsed = sim.age - self.start_age
        if elapsed < self.start_time:
            return None
        missing = int(float(self.count(sim)) * strength) - sim.particles.count
        if self.rate > 0:
            missing = min(missing, max(1, int(self.rate * dt)))
        return np.zeros(missing) if missing > 0 else None
