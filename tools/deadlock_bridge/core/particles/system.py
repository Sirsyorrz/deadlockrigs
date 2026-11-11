"""Particle system definitions (vpcf) and their simulation: a tree of systems, each emitting, initializing and
operating on its particles step by step, its children started with it and run on their parent's clock (after their
delay), all reading the root's control points.

Follows the engine's order within a step: the system ages, pre-emission operators set control points, emitters spawn
particles (each initialized by every initializer in turn, its velocity then stored as a previous position), operators
run (C_OP_BasicMovement integrating the forces), constraints move particles back into place, and dead particles are
removed. Only the end cap (the phase a system plays when stopped) is never entered: a snapshot is of a running effect.
"""
import collections
import math
import zlib

import numpy as np

from .. import kv3
from ..resource import ResourceFile
from .fields import Particles, View
from .params import END_CAP, Params

# The engine caps a collection at this many particles whatever the definition asks for.
MAX_PARTICLES = 5000


class Random:
    """A system's draws from a shared table of random floats: a counter that runs free for the life of the system
    (``next``), or slots fixed by a particle's id (``for_particle``) for values that must not change over its life.
    The seed displaces both, giving each system instance its own sequence."""

    TABLE = np.random.default_rng(0x5EED).random(1 << 15)
    STRIDE = 17  # how far each operator's per particle draws are moved from the previous operator's

    def __init__(self, seed: int):
        self.seed = seed & 0xFFF
        self.count = 0
        self.operator_offset = 0

    def next(self, n: int = 1) -> np.ndarray:
        index = self.seed + self.count + np.arange(n)
        self.count += n
        return self.TABLE[index % self.TABLE.size]

    def next1(self) -> float:
        return float(self.next(1)[0])

    def for_particle(self, ids, offset: int = 0) -> np.ndarray:
        index = (np.asarray(ids, np.int64) + (self.seed + self.operator_offset + offset)) % self.TABLE.size
        return self.TABLE[index]

    def sample(self, index: int) -> float:
        return float(self.TABLE[index % self.TABLE.size])

    def between3(self, n: int, low, high) -> np.ndarray:
        """n points drawn per component between two corners, (n, 3)."""
        draw = self.next(n * 3).reshape(n, 3)
        return low + (np.asarray(high) - low) * draw

    def in_unit_ball(self, n: int):
        """n directions uniform through the unit ball, already scaled by their radius fraction, and the fractions."""
        draw = self.next(n * 3).reshape(n, 3) * 0.9999 + 0.0001
        cos_polar = draw[:, 0] * 2.0 - 1.0
        azimuth = draw[:, 1] * math.tau
        fraction = np.cbrt(draw[:, 2])
        ring = np.sqrt(np.maximum(0.0, 1.0 - cos_polar * cos_polar)) * fraction
        return np.stack([ring * np.cos(azimuth), ring * np.sin(azimuth), fraction * cos_polar], axis=1), fraction


class ControlPoint:
    """A control point: a position and, when something placed it with one, a rotation (columns: forward, left,
    up). Change time is when its value last changed (-1 for never)."""

    __slots__ = ("position", "rotation", "oriented", "change_time", "model", "previous_bones", "previous_position")

    def __init__(self):
        self.position = np.zeros(3)
        self.previous_position = np.zeros(3)
        # {'hitboxes': {set: [box, ...]}, 'bones': {name: 4x4 bone to simulation space}} of the model it carries
        self.model = None
        self.previous_bones = None
        self.rotation = np.eye(3)
        self.oriented = False
        self.change_time = -1.0

    @property
    def forward(self) -> np.ndarray:
        return self.rotation[:, 0]

    def change_age(self, sim) -> float:
        return sim.root.age - self.change_time


class Function:
    """What every particle function shares: when it runs (m_nOpEndCapState: always, or only outside or inside the
    end cap) and how strongly (m_flOpStrength, faded in and out over a window of the system's life)."""

    def __init__(self, p: Params):
        self.end_cap = p.enum("m_nOpEndCapState", END_CAP, -1)
        self.strength = p.number("m_flOpStrength", 1.0)
        self.fade = (p.float("m_flOpStartFadeInTime"), p.float("m_flOpEndFadeInTime"),
                     p.float("m_flOpStartFadeOutTime"), p.float("m_flOpEndFadeOutTime"))
        self.oscillate = p.float("m_flOpFadeOscillatePeriod")
        self.time_offset = (p.int("m_nOpTimeOffsetSeed"), p.float("m_flOpTimeOffsetMin"), p.float("m_flOpTimeOffsetMax"))
        self.time_scale = (p.int("m_nOpTimeScaleSeed"), p.float("m_flOpTimeScaleMin", 1.0), p.float("m_flOpTimeScaleMax", 1.0))
        self.unity = (not any(self.fade) and (self.time_offset[0] == 0 or self.time_offset[1:] == (0.0, 0.0))
                      and (self.time_scale[0] == 0 or self.time_scale[1:] == (1.0, 1.0)))
        self.always = self.unity and self.end_cap == -1 and self.strength.constant == 1.0

    def runs_in_phase(self, sim) -> bool:
        return self.end_cap == -1 or sim.in_end_cap == (self.end_cap == 1)

    def run_strength(self, sim) -> float:
        if self.always:
            return 1.0
        if not self.runs_in_phase(sim):
            return 0.0
        strength = float(self.strength(sim))
        if self.unity or strength <= 0.0:
            return strength
        return max(0.0, strength) * _fade_in_out(*self.fade, self._time(sim))

    def _time(self, sim) -> float:
        if self.oscillate > 0:
            return (sim.age / self.oscillate) % 1.0
        time = max(0.0, sim.age - sim.restart_age)
        seed, low, high = self.time_offset
        if seed:
            time = max(0.0, time + low + (high - low) * sim.random.sample(sim.random.seed + seed))
        seed, low, high = self.time_scale
        start = self.fade[0]
        if seed and time > start:
            scale = 1.0 / max(0.0001, low + (high - low) * sim.random.sample(sim.random.seed + seed))
            time = start + scale * (time - start)
        return time

    def reset(self):
        """Clears per run state when the system starts."""


def _fade_in_out(in_start, in_end, out_start, out_end, time):
    if in_start > time:
        return 0.0
    if 0.0 < out_end < time:
        return 0.0
    in_end = max(in_end, in_start)
    out_start = max(out_start, in_end)
    out_end = max(out_end, out_start)
    strength = 1.0
    if in_end > time and in_end > in_start:
        strength = min(strength, (time - in_start) / (in_end - in_start))
    if time > out_start and out_end > out_start:
        strength = min(strength, (out_end - time) / (out_end - out_start))
    return strength


# Definitions ------------------------------------------------------------------------------------------------------

class Child:
    """A child system (m_Children): started with its parent, run once the parent is ``delay`` seconds old; end cap
    children only play while the parent ends, which a snapshot never reaches."""

    __slots__ = ("path", "delay", "end_cap", "disabled")

    def __init__(self, p: Params):
        self.path = p.str("m_ChildRef")
        self.delay = p.float("m_flDelay")
        self.end_cap = p.bool("m_bEndCap")
        self.disabled = p.version >= 5 and p.bool("m_bDisableChild")


class Definition:
    """A particle system file: its settings, the constant attributes particles start with, its functions (made
    afresh for every simulation, as some keep state) and its renderers."""

    def __init__(self, path: str, data: dict, library: "Library"):
        self.path = path
        self.library = library
        p = Params(data, kv3.as_int(data.get("m_nBehaviorVersion"), 0), notes=library.unsupported)
        self.params = p
        self.version = p.version
        self.initial_particles = p.int("m_nInitialParticles")
        max_particles = p.int("m_nMaxParticles", 1000)
        self.capacity = MAX_PARTICLES if max_particles <= 0 else min(max_particles, MAX_PARTICLES)
        self.min_step = p.float("m_flMinimumTimeStep")
        self.max_step = max(self.min_step, p.float("m_flMaximumTimeStep", 0.1))
        if self.max_step <= 0:
            self.max_step = 0.1
        self.presimulation = p.float("m_flPreSimulationTime")
        self.stop_after = p.float("m_flStopSimulationAfterTime")

        constants = {}
        if p.has("m_ConstantColor"):
            color = p.get("m_ConstantColor")
            constants["color"] = tuple(kv3.as_float(c) / 255.0 for c in color[:3])
            constants["alpha"] = kv3.as_float(color[3]) / 255.0 if len(color) > 3 else 1.0
        constants["radius"] = p.float("m_flConstantRadius", 5.0)
        constants["lifetime"] = p.float("m_flConstantLifespan", 1.0)
        constants["roll"] = math.radians(p.float("m_flConstantRotation"))
        constants["roll_speed"] = math.radians(p.float("m_flConstantRotationSpeed"))
        constants["normal"] = tuple(p.vector("m_ConstantNormal", (0.0, 0.0, 1.0)))
        constants["sequence"] = float(p.int("m_nConstantSequenceNumber"))
        constants["sequence1"] = float(p.int("m_nConstantSequenceNumber1"))
        self.constants = constants

        self.children = [Child(c) for c in p.array("m_Children")]
        self._made = False

        from . import renderers
        self.renderers = []
        for index, block in enumerate(p.functions("m_Renderers")):
            if block.bool("m_bDisableOperator"):
                continue
            kind = block.str("_class")
            maker = renderers.RENDERERS.get(kind)
            if maker is None:
                library.skip("renderer", kind)
                continue
            try:
                self.renderers.append(maker(block, library, index))
            except Exception as error:  # an odd renderer must not lose the rest of the system
                library.skip("renderer", f"{kind} ({error})")

    def make_functions(self):
        """Fresh function instances for one simulation: (pre-emission, emitters, initializers, operators, forces,
        constraints). What the port skips is counted the first time only, so the counts are per system file."""
        from . import emitters, initializers, operators
        params = self.params if not self._made else Params(self.params.data, self.version)
        report = not self._made
        self._made = True
        tables = (("m_PreEmissionOperators", operators.PRE_EMISSION, "pre-emission operator"),
                  ("m_Emitters", emitters.EMITTERS, "emitter"),
                  ("m_Initializers", initializers.INITIALIZERS, "initializer"),
                  ("m_Operators", operators.OPERATORS, "operator"),
                  ("m_ForceGenerators", operators.FORCES, "force generator"),
                  ("m_Constraints", operators.CONSTRAINTS, "constraint"))
        made = []
        for key, table, label in tables:
            functions = []
            for block in params.functions(key):
                if block.bool("m_bDisableOperator"):
                    continue
                kind = block.str("_class")
                maker = table.get(kind)
                if maker is None:
                    if report:
                        self.library.skip(label, kind)
                    continue
                try:
                    functions.append(maker(block))
                except Exception as error:
                    if report:
                        self.library.skip(label, f"{kind} ({error})")
            made.append(functions)
        return made


class Library:
    """Particle system definitions read from the game's content, each once, and a count of what the port skipped
    (per kind and class name) for the import report."""

    def __init__(self, read, read_sheet=None):
        self._read = read
        self._read_sheet = read_sheet
        self._definitions: dict[str, Definition | None] = {}
        self._sheets: dict[str, list | None] = {}
        self.unsupported = collections.Counter()
        self.missing: set[str] = set()

    def get(self, path: str) -> Definition | None:
        key = (path or "").replace("\\", "/").lower()
        if key.endswith("_c"):
            key = key[:-2]
        if key not in self._definitions:
            definition = None
            data = self._read(key + "_c") if key else None
            if data is None:
                self.missing.add(key)
            else:
                document = ResourceFile(data).keyvalues("DATA")
                if document is not None:
                    definition = Definition(key, document, self)
            self._definitions[key] = definition
        return self._definitions[key]

    def sheet(self, texture: str):
        """A texture's sprite sheet sequences, or None."""
        key = (texture or "").replace("\\", "/").lower()
        if key not in self._sheets:
            self._sheets[key] = self._read_sheet(key) if self._read_sheet and key else None
        return self._sheets[key]

    def skip(self, kind: str, name: str):
        self.unsupported[f"{kind} {name}"] += 1


# Simulation -------------------------------------------------------------------------------------------------------

class Simulation:
    """One running system and its children."""

    def __init__(self, definition: Definition, seed: int, parent: "Simulation | None" = None):
        self.definition = definition
        self.parent = parent
        self.root = parent.root if parent is not None else self
        self.version = definition.version
        self.random = Random(seed)
        self.particles = Particles(definition.constants, definition.capacity)
        self.age = 0.0
        self.restart_age = 0.0
        self.in_end_cap = False
        self.end_cap_age = 0.0
        self.frame_time = definition.max_step
        self.previous_frame_time = 0.0
        self.emitted = 0
        self.killed_last = 0
        self.started = False
        self.delay = 0.0
        self.end_cap_child = False
        self._control_points: dict[int, ControlPoint] = {}
        # What a control point nothing set reads as, component by component (the root's counts): 0 as in a map, 1
        # for effects previewed on their own, where the game code that sets scales and counts isn't running.
        self.unset_value = 0.0
        (self.pre_emission, self.emitters, self.initializers, self.operators, self.forces,
         self.constraints) = definition.make_functions()

        self.children: list[Simulation] = []
        for index, child in enumerate(definition.children):
            if child.disabled or not child.path:
                continue
            child_definition = definition.library.get(child.path)
            if child_definition is None or self._is_self_or_ancestor(child_definition.path):
                continue
            simulation = Simulation(child_definition, zlib.crc32(f"{seed}/{index}".encode()), self)
            simulation.delay = child.delay
            simulation.end_cap_child = child.end_cap
            self.children.append(simulation)

    def _is_self_or_ancestor(self, path: str) -> bool:
        node = self
        while node is not None:
            if node.definition.path == path:
                return True
            node = node.parent
        return False

    # Control points (shared by the whole tree, held by the root) ----------------------------------------------

    def control_point(self, index: int) -> ControlPoint:
        points = self.root._control_points
        point = points.get(index)
        if point is None:
            point = points[index] = ControlPoint()
        return point

    def set_control_point(self, index: int, position, rotation=None, model=None):
        point = self.control_point(index)
        position = np.asarray(position, np.float64)
        point.previous_position = point.position.copy() if point.change_time >= 0 else position.copy()
        if model is not None:
            if point.model is None:
                point.previous_bones = model["bones"]
            elif point.model["bones"] is not model["bones"]:
                point.previous_bones = point.model["bones"]
            point.model = model
        if not np.array_equal(position, point.position) or point.change_time < 0:
            point.change_time = self.root.age
        point.position = position.copy()
        if rotation is not None:
            point.rotation = np.asarray(rotation, np.float64)
            point.oriented = True

    def set_control_point_component(self, index: int, component: int, value: float):
        position = self.control_point(index).position.copy()
        position[min(max(component, 0), 2)] = value
        self.set_control_point(index, position)

    # Running --------------------------------------------------------------------------------------------------

    def run(self, time: float, step: float = 1.0 / 30.0):
        """Plays the system from its start for ``time`` seconds in frames of ``step``."""
        for _ in range(max(1, int(round(time / step)))):
            self.update(step)

    def update(self, frame_time: float):
        if not self.started:
            self.start()
        definition = self.definition
        remaining = min(frame_time, definition.max_step * 10.0)
        while remaining > 1e-9:
            step = definition.max_step if remaining > definition.max_step else max(remaining, definition.min_step)
            remaining -= step
            self._simulate(step)
        if definition.stop_after > 0 and self.age >= definition.stop_after:
            return
        for child in self.children:
            if child.end_cap_child or self.age < child.delay:
                continue
            child.update(frame_time)

    def start(self):
        self.started = True
        for function in (*self.pre_emission, *self.initializers, *self.operators, *self.constraints):
            function.reset()
        if self.definition.initial_particles > 0:
            self._spawn(np.zeros(self.definition.initial_particles))
        for emitter in self.emitters:
            emitter.start(self)
        for child in self.children:
            if not child.end_cap_child:
                child.start()
        if self.definition.presimulation > 0:
            step = self.definition.max_step
            for _ in range(min(2048, int(math.ceil(self.definition.presimulation / step)))):
                self.update(step)

    def _simulate(self, dt: float):
        definition = self.definition
        if definition.stop_after > 0 and self.age >= definition.stop_after:
            return
        dt = min(dt, definition.max_step)
        self.frame_time = dt
        self.age += dt
        particles = self.particles
        if particles.count:
            particles.array("age")[:] = self.age - particles.array("creation_time")

        random = self.random
        random.operator_offset = 0
        for operator in self.pre_emission:
            if operator.run_strength(self) <= 0.0:
                continue
            if operator.run_once:
                if operator.has_run:
                    continue
                operator.has_run = True
            operator.operate(self, dt)
            random.operator_offset += Random.STRIDE

        for emitter in self.emitters:
            strength = emitter.run_strength(self)
            if strength > 0.0:
                ages = emitter.emit(self, dt, strength)
                if ages is not None and len(ages):
                    self._spawn(ages)

        random.operator_offset = 0
        if particles.count:
            view = View(particles, slice(0, particles.count))
            for operator in self.operators:
                strength = operator.run_strength(self)
                if strength > 0.0:
                    operator.operate(self, view, dt, strength)
                random.operator_offset += Random.STRIDE
            self._constrain(view, dt)
        self.killed_last = particles.remove_killed()
        self.previous_frame_time = dt

    def _constrain(self, view: View, dt: float):
        """Constraints run from a work list: each once, then again only while another one moved particles."""
        if not self.constraints:
            return
        satisfied = [False] * len(self.constraints)
        for _ in range(3):
            changed = False
            for i, constraint in enumerate(self.constraints):
                if satisfied[i]:
                    continue
                satisfied[i] = True
                if constraint.run_strength(self) <= 0.0:
                    continue
                if constraint.apply(self, view, dt):
                    changed = True
                    satisfied = [j == i for j in range(len(satisfied))]
            if not changed:
                break

    def _spawn(self, ages):
        """Emits particles of the given ages (seconds already lived this step) and initializes them."""
        particles = self.particles
        rows = particles.add(len(ages))
        n = rows.stop - rows.start
        if n <= 0:
            return
        view = View(particles, rows)
        ids = self.emitted + np.arange(n, dtype=np.float64)
        self.emitted += n
        view.set("unique_id", ids)
        view.set("particle_id", ids + self.random.seed)
        view.set("position", self.control_point(0).position)
        view.set("creation_time", self.age - np.asarray(ages[:n], np.float64))
        view.set("age", np.asarray(ages[:n], np.float64))
        view.set("velocity", 0.0)
        for initializer in self.initializers:
            if initializer.runs_in_phase(self):
                initializer.initialize(self, view)
        # The initial velocity becomes the previous position; C_OP_BasicMovement derives motion from the pair.
        view.set("previous", view.get("position") - view.get("velocity") * self.frame_time)
        particles.snapshot_initial(rows)

    def walk(self):
        """This system and every system below it."""
        yield self
        for child in self.children:
            yield from child.walk()


def simulate(definition: Definition, control_points: dict, time: float, seed: int,
             step: float = 1.0 / 30.0, unset_value: float = 0.0) -> Simulation:
    """Runs a system tree from its start for ``time`` seconds. ``control_points`` maps control point numbers to
    (position, rotation or None), in game units."""
    simulation = Simulation(definition, seed)
    simulation.unset_value = unset_value
    for index, (position, rotation, *model) in control_points.items():
        simulation.set_control_point(index, position, rotation, *model)
    simulation.run(time, step)
    return simulation


def preview_control_points(definition: Definition) -> dict:
    """The control points the particle editor previews an effect with (its first control point configuration,
    "preview" by preference), with the system at the origin: {number: (position, rotation or None)}."""
    configs = definition.params.array("m_controlPointConfigurations")
    config = next((c for c in configs if c.str("m_name").lower() == "preview"), configs[0] if configs else None)
    points = {0: (np.zeros(3), np.eye(3))}
    if config is None:
        return points
    from ..probes import rotation_matrix
    for driver in config.array("m_drivers"):
        index = driver.int("m_iControlPoint", 0)
        if index <= 0 or index >= 64:
            continue
        angles = driver.vector("m_angOffset", (0.0, 0.0, 0.0))
        points[index] = (driver.vector("m_vecOffset", (0.0, 0.0, 0.0)),
                         rotation_matrix(angles) if angles.any() else None)
    return points


PREVIEW_MOMENTS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def busiest_moment(definition: Definition, control_points: dict, seed: int, moments=PREVIEW_MOMENTS,
                   unset_value: float = 1.0, step: float = 1.0 / 30.0) -> float | None:
    """The moment (of ``moments``) the effect draws the most at, the latest of equals, playing it once; None when it
    never draws."""
    from . import draw
    simulation = Simulation(definition, seed)
    simulation.unset_value = unset_value
    for index, (position, rotation, *model) in control_points.items():
        simulation.set_control_point(index, position, rotation, *model)
    best, best_count, frames = None, 0, 0
    for moment in moments:
        # Frame for frame as simulate() plays the moment it is then built at.
        target = max(1, int(round(moment / step)))
        while frames < target:
            simulation.update(step)
            frames += 1
        count = sum(batch.count for _, _, batch in draw(simulation))
        # Ties go to the later moment: an effect that draws as much throughout is past its fade-ins then.
        if count and count >= best_count:
            best, best_count = moment, count
    return best

