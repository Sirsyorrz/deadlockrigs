"""A system tree played over time for playback: what its renderers draw at every step of a stretch of its life.

Looping (the map's effects, which run for as long as the map does): the stretch starts once the effect has settled
in. Each step draws the particles born during the stretch, at every later moment they are alive, folded back onto
the stretch: a particle born near its end carries on at its start, so the loop has no seam for anything that keeps
emitting. Particles born before the stretch that outlive it (a system's one off burst of glows) are drawn as they
are during it.

Once (an effect placed to play from a frame): every step from the effect's start, as it is.

Both take ``moving``: control points that move while the effect plays (a rig's bones), a function of the seconds
since the start giving {number: (position, rotation or None[, model])}, set before each step.
"""
from contextlib import contextmanager

import numpy as np

from .system import Simulation

# The longest a looping bake runs past its stretch to see the stretch's last particles die, in stretches.
TAIL_LOOPS = 4
# A loop's length when it is left to the effect: twice its particles' mean lifetime, within these (seconds). Short
# lived flames repeat unnoticed sooner than drifting smoke.
LOOP_RANGE = (2.0, 6.0)


@contextmanager
def _only(particles, keep):
    """The particles narrowed to those ``keep`` marks while drawing."""
    if keep is None or keep.all():
        yield
        return
    saved = particles._current, particles._initial, particles.count, particles._rows
    count = particles.count
    particles._current = {k: a[:count][keep].copy() for k, a in saved[0].items()}
    particles._initial = {k: a[:count][keep].copy() for k, a in saved[1].items()}
    particles.count = particles._rows = int(keep.sum())
    try:
        yield
    finally:
        particles._current, particles._initial, particles.count, particles._rows = saved


def _drawn(simulation, ambient, masks=None):
    out = []
    for system in simulation.walk():
        if not system.started:
            continue
        keep = masks.get(id(system)) if masks is not None else None
        if keep is not None and not keep.any():
            continue
        with _only(system.particles, keep):
            for renderer in system.definition.renderers:
                if renderer.hidden:
                    continue
                batch = renderer.build(system, ambient)
                if batch is not None and batch.count:
                    out.append((system, renderer, batch))
    return out


def _new(definition, control_points, seed, unset_value):
    simulation = Simulation(definition, seed)
    simulation.unset_value = unset_value
    for index, (position, rotation, *model) in control_points.items():
        simulation.set_control_point(index, position, rotation, *model)
    return simulation


def _step(simulation, step, moving):
    if moving is not None:
        for index, (position, rotation, *model) in moving(simulation.age).items():
            simulation.set_control_point(index, position, rotation, *model)
    simulation.update(step)


def _ids(system):
    return system.particles.array("unique_id").copy()


def loop_length(simulation) -> float:
    lifetimes = [s.particles.array("lifetime") for s in simulation.walk() if s.particles.count]
    if not lifetimes:
        return LOOP_RANGE[0]
    mean = float(np.mean(np.concatenate(lifetimes)))
    return float(np.clip(2.0 * mean, *LOOP_RANGE)) if np.isfinite(mean) else LOOP_RANGE[1]


def loop(definition, control_points, seed, start, length=None, rate=30.0, ambient=(0.0, 0.0, 0.0),
         unset_value=0.0, moving=None):
    """The draws of a looping effect: a list of ``length * rate`` steps (``length`` None: loop_length()'s), each a
    list of (system, renderer, batch), for the stretch from ``start`` seconds. Systems keep their identity across
    steps."""
    step = 1.0 / rate
    warm = int(round(start * rate))

    # The particles born before the stretch still alive at its end, found on a run of its own (the same seed plays
    # the same).
    probe = _new(definition, control_points, seed, unset_value)
    for _ in range(warm):
        _step(probe, step, moving)
    if length is None:
        length = loop_length(probe)
    steps = max(1, int(round(length * rate)))
    begun = {i: s.age for i, s in enumerate(probe.walk())}
    early = {}
    for i, s in enumerate(probe.walk()):
        early[i] = set(_ids(s)[s.particles.array("creation_time") < begun[i]].tolist())
    for _ in range(steps):
        _step(probe, step, moving)
    lasting = {i: early[i] & set(_ids(s).tolist()) for i, s in enumerate(probe.walk())}

    simulation = _new(definition, control_points, seed, unset_value)
    for _ in range(warm):
        _step(simulation, step, moving)
    systems = list(simulation.walk())
    window = {id(s): (s.age, s.age + steps * step) for s in systems}
    persistent = {id(s): lasting[i] for i, s in enumerate(systems)}

    frames = [[] for _ in range(steps)]
    k = 0
    while True:
        masks, inside = {}, False
        for s in simulation.walk():
            if not s.started or not s.particles.count:
                continue
            low, high = window.get(id(s), (0.0, 0.0))
            born = s.particles.array("creation_time")
            keep = (born >= low - 1e-6) & (born < high - 1e-6)
            inside = inside or bool(keep.any())
            if k < steps and persistent.get(id(s)):
                keep |= np.isin(_ids(s), list(persistent[id(s)]))
            masks[id(s)] = keep
        if k >= steps and not inside:
            break
        frames[k % steps].extend(_drawn(simulation, ambient, masks))
        k += 1
        if k >= steps * (1 + TAIL_LOOPS):
            break
        _step(simulation, step, moving)
    return frames


def once(definition, control_points, seed, length, rate=30.0, ambient=(0.0, 0.0, 0.0), unset_value=0.0,
         moving=None, stop_at=None):
    """The draws of an effect played from its start for ``length`` seconds, a list of steps as loop() gives.
    ``stop_at``: seconds at which the effect is stopped (its end cap plays out in the rest)."""
    step = 1.0 / rate
    simulation = _new(definition, control_points, seed, unset_value)
    frames = []
    for k in range(max(1, int(round(length * rate)))):
        if stop_at is not None and k * step >= stop_at - 1e-6:
            simulation.end_cap()
        _step(simulation, step, moving)
        frames.append(_drawn(simulation, ambient))
    return frames
