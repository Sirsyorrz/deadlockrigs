"""A system's particles: one numpy array per attribute, holding every live particle in emission order. Arrays are
made when an attribute is first touched, filled with the system's constant (or the engine's default) value, and
dead particles are removed at the end of each step, keeping the order.

Attributes are numbered as the game numbers them (ParticleAttributeIndex_t), which is how functions name the field
they read or write (m_nOutputField, m_nFieldOutput, ...).
"""
import numpy as np

POSITION, LIFE_DURATION, PREVIOUS_POSITION, RADIUS, ROLL, ROLL_SPEED, COLOR, ALPHA, CREATION_TIME = range(9)
SEQUENCE, TRAIL_LENGTH, PARTICLE_ID, YAW, SEQUENCE1, HITBOX_INDEX, HITBOX_OFFSET, ALPHA_ALTERNATE = range(9, 17)
SCRATCH_VECTOR, SCRATCH_FLOAT, DISABLED, PITCH, NORMAL, GLOW_RGB, GLOW_ALPHA = range(17, 24)
SCRATCH_FLOAT1, SCRATCH_FLOAT2, SCRATCH_VECTOR2 = 26, 27, 30
PARENT_INDEX, FORCE_SCALE, MANUAL_FRAME, EXTRA1, EXTRA2 = 33, 34, 38, 39, 40
BOX_MINS, BOX_MAXS, BOX_ANGLES, BOX_FLAGS, USER_EVENTS, PARENT_ID, ROPE_SEGMENT_ID, ROPE_SEGMENT_DATA = range(41, 49)

# Attribute number -> array name, components.
ATTRIBUTES = {
    POSITION: ("position", 3), LIFE_DURATION: ("lifetime", 1), PREVIOUS_POSITION: ("previous", 3),
    RADIUS: ("radius", 1), ROLL: ("roll", 1), ROLL_SPEED: ("roll_speed", 1), COLOR: ("color", 3),
    ALPHA: ("alpha", 1), CREATION_TIME: ("creation_time", 1), SEQUENCE: ("sequence", 1),
    TRAIL_LENGTH: ("trail_length", 1), PARTICLE_ID: ("particle_id", 1), YAW: ("yaw", 1),
    SEQUENCE1: ("sequence1", 1), HITBOX_INDEX: ("hitbox_index", 1), HITBOX_OFFSET: ("hitbox_offset", 3),
    ALPHA_ALTERNATE: ("alpha_alternate", 1), SCRATCH_VECTOR: ("scratch_vector", 3),
    SCRATCH_FLOAT: ("scratch_float", 1), PITCH: ("pitch", 1), NORMAL: ("normal", 3), GLOW_RGB: ("glow_rgb", 3),
    GLOW_ALPHA: ("glow_alpha", 1), SCRATCH_FLOAT1: ("scratch_float1", 1), SCRATCH_FLOAT2: ("scratch_float2", 1),
    SCRATCH_VECTOR2: ("scratch_vector2", 3), PARENT_INDEX: ("parent_index", 1), FORCE_SCALE: ("force_scale", 1),
    MANUAL_FRAME: ("manual_frame", 1), EXTRA1: ("extra1", 1), EXTRA2: ("extra2", 1), BOX_MINS: ("box_mins", 3),
    BOX_MAXS: ("box_maxs", 3), BOX_ANGLES: ("box_angles", 3), BOX_FLAGS: ("box_flags", 1),
    USER_EVENTS: ("user_events", 1), PARENT_ID: ("parent_id", 1), ROPE_SEGMENT_ID: ("rope_segment_id", 1),
    ROPE_SEGMENT_DATA: ("rope_segment_data", 3),
}

ANGLES = {ROLL, YAW, PITCH, ROLL_SPEED}
NORMALIZED = {COLOR, ALPHA, ALPHA_ALTERNATE, GLOW_RGB, GLOW_ALPHA}
INTEGER = {SEQUENCE, SEQUENCE1, PARTICLE_ID, PARENT_INDEX, PARENT_ID, ROPE_SEGMENT_ID, USER_EVENTS}

# What a particle starts with before the system's constants and its initializers (the engine's defaults).
DEFAULTS = {
    "lifetime": 1.0, "radius": 5.0, "alpha": 1.0, "alpha_alternate": 1.0, "color": (1.0, 1.0, 1.0),
    "trail_length": 0.1, "force_scale": 1.0, "normal": (0.0, 0.0, 1.0), "parent_index": -1.0, "parent_id": -1.0,
    "hitbox_index": -1.0,
}

# Arrays the simulation keeps that aren't attributes of their own.
_INTERNAL = {"age": 1, "velocity": 3, "force": 3, "unique_id": 1, "killed": 1, "ground_lerp_from": 3}


def is_vector(field: int) -> bool:
    return ATTRIBUTES.get(field, ("", 1))[1] == 3


def name(field: int) -> str | None:
    entry = ATTRIBUTES.get(field)
    return entry[0] if entry else None


def components(array_name: str) -> int:
    if array_name in _INTERNAL:
        return _INTERNAL[array_name]
    for entry in ATTRIBUTES.values():
        if entry[0] == array_name:
            return entry[1]
    return 1


class Particles:
    """Current and initial (as they were when initialized) values of every live particle."""

    def __init__(self, defaults: dict, capacity: int):
        self.defaults = dict(DEFAULTS)
        self.defaults.update(defaults)
        self.capacity = capacity
        self.count = 0
        self._current: dict[str, np.ndarray] = {}
        self._initial: dict[str, np.ndarray] = {}
        self._rows = 0  # allocated rows

    def _new(self, array_name: str, rows: int) -> np.ndarray:
        n = components(array_name)
        dtype = bool if array_name == "killed" else np.float64
        array = np.empty((rows, n) if n > 1 else rows, dtype)
        array[...] = self.defaults.get(array_name, 0.0)
        return array

    def array(self, array_name: str) -> np.ndarray:
        """The current values of every live particle (a view: writes change the particles)."""
        array = self._current.get(array_name)
        if array is None:
            array = self._current[array_name] = self._new(array_name, self._rows)
        return array[:self.count]

    def initial(self, array_name: str) -> np.ndarray:
        """The values the particles had when they were initialized; the constant for attributes nothing set."""
        array = self._initial.get(array_name)
        if array is None:
            array = self._initial[array_name] = self._new(array_name, self._rows)
        return array[:self.count]

    def has(self, array_name: str) -> bool:
        return array_name in self._current

    def add(self, n: int) -> slice:
        """Room for up to n more particles (fewer when the system is full), set to the defaults."""
        n = max(0, min(n, self.capacity - self.count))
        end = self.count + n
        if end > self._rows:
            rows = max(end, self._rows * 2, 16)
            for store in (self._current, self._initial):
                for key, old in store.items():
                    grown = self._new(key, rows)
                    grown[:self.count] = old[:self.count]
                    store[key] = grown
            self._rows = rows
        for store in (self._current, self._initial):
            for key, array in store.items():
                array[self.count:end] = self.defaults.get(key, 0.0)
        start = self.count
        self.count = end
        return slice(start, end)

    def snapshot_initial(self, rows: slice):
        """Records the freshly initialized particles' values as their initial ones."""
        for key, array in self._current.items():
            if key not in self._initial:
                self._initial[key] = self._new(key, self._rows)
            self._initial[key][rows] = array[rows]

    def remove_killed(self) -> int:
        if "killed" not in self._current or self.count == 0:
            return 0
        keep = ~self._current["killed"][:self.count]
        alive = int(keep.sum())
        if alive == self.count:
            return 0
        for store in (self._current, self._initial):
            for key, array in store.items():
                array[:alive] = array[:self.count][keep]
        killed = self.count - alive
        self.count = alive
        return killed

    def clear(self):
        self.count = 0


class View:
    """Some of a system's particles (a slice of them: all of them for operators, the new ones for initializers),
    read and written by attribute."""

    __slots__ = ("particles", "rows", "n")

    def __init__(self, particles: Particles, rows: slice):
        self.particles = particles
        self.rows = rows
        self.n = len(range(*rows.indices(particles.count)))

    def get(self, array_name: str) -> np.ndarray:
        return self.particles.array(array_name)[self.rows]

    def initial(self, array_name: str) -> np.ndarray:
        return self.particles.initial(array_name)[self.rows]

    def set(self, array_name: str, values):
        self.particles.array(array_name)[self.rows] = values

    def field(self, field: int) -> np.ndarray:
        key = name(field)
        if key is None:
            return np.zeros((self.n, 3) if is_vector(field) else self.n)
        return self.get(key)

    def initial_field(self, field: int) -> np.ndarray:
        key = name(field)
        if key is None:
            return np.zeros((self.n, 3) if is_vector(field) else self.n)
        return self.initial(key)

    def set_field(self, field: int, values):
        key = name(field)
        if key is None:
            return
        if key == "normal":
            # A zero normal is ignored, as the engine's attribute setter does.
            values = np.broadcast_to(np.asarray(values, np.float64), (self.n, 3))
            current = self.get(key)
            values = np.where((np.abs(values).sum(axis=1) == 0)[:, None], current, values)
        if field in INTEGER:
            values = np.trunc(values)
        self.set(key, values)

    def kill(self, mask):
        killed = self.get("killed")
        self.set("killed", killed | mask)

    @property
    def normalized_age(self) -> np.ndarray:
        return self.get("age") / np.maximum(self.get("lifetime"), 0.0001)
