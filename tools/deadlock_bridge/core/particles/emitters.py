"""Emitters: when particles are born. Each returns the ages (time already lived this step) of the particles it emits
in a step, which the simulation then spawns and initializes."""
import math

import numpy as np

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
