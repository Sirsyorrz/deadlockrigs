"""Particle snapshots (vsnap): a fixed set of points, often traced off a model's surface and skinned to its bones,
that a system publishes on a control point (m_hSnapshot, m_nSnapshotControlPoint) for its functions and its
children to read. The SNAP block holds one array per attribute the DATA block lists, behind Valve's block
compression."""
import struct

import numpy as np

from . import fields as F
from ..resource import ResourceFile

# attribute number -> the vsnap attribute holding it (ValveResourceFormat ParticleSnapshot.GetSnapshotAttributeName)
ATTRIBUTE_NAMES = {
    F.POSITION: "position", F.LIFE_DURATION: "lifespan", F.PREVIOUS_POSITION: "velocity", F.RADIUS: "radius",
    F.ROLL: "rotation", F.ROLL_SPEED: "rotation_speed", F.COLOR: "color", F.ALPHA: "opacity",
    F.CREATION_TIME: "creation_time", F.SEQUENCE: "sequence_number", F.TRAIL_LENGTH: "trail_length",
    F.PARTICLE_ID: "particle_id", F.YAW: "yaw", F.SEQUENCE1: "sequence_number1", F.HITBOX_INDEX: "hitbox",
    F.HITBOX_OFFSET: "hitbox_offset", F.ALPHA_ALTERNATE: "alpha2", F.SCRATCH_VECTOR: "scratch_vec",
    F.SCRATCH_FLOAT: "scratch_float", F.PITCH: "pitch", F.NORMAL: "normal", F.GLOW_RGB: "glow_rgb",
    F.GLOW_ALPHA: "glow_alpha", F.FORCE_SCALE: "force_scale", F.MANUAL_FRAME: "manual_animation_frame",
    F.EXTRA1: "shader_extra_data_1", F.EXTRA2: "shader_extra_data_2", F.ROPE_SEGMENT_ID: "rope_segment_id",
}


class Snapshot:
    """``count`` points; ``arrays`` by attribute name (position, normal, radius, ... as (count,) or (count, 3));
    ``joints`` (count, 4) bone names and ``weights`` (count, 4) when skinned, else None."""

    def __init__(self, count: int, arrays: dict, joints=None, weights=None):
        self.count = count
        self.arrays = arrays
        self.joints = joints
        self.weights = weights

    @property
    def skinned(self) -> bool:
        return self.joints is not None and "position" in self.arrays


class SnapshotRead:
    """What C_INIT_InitFromCPSnapshot and C_OP_SetFromCPSnapshot share (particles.dll): the snapshot on a control
    point, the column read (default: the attribute written), the control point a position, previous position
    (a velocity in the snapshot) or normal is moved into, and random row draws."""

    def __init__(self, p):
        self.control_point = p.int("m_nControlPointNumber", 0)
        self.write = p.int("m_nAttributeToWrite", F.POSITION)
        read = p.int("m_nAttributeToRead", -1)
        self.read = self.write if read == -1 else read
        self.local_cp = p.int("m_nLocalSpaceCP", 0)
        self.random = p.bool("m_bRandom")
        self.reverse = p.bool("m_bReverse")
        self.random_seed = p.int("m_nRandomSeed")
        self.increment = p.number("m_nSnapShotIncrement", 1.0)
        # subsets are row ranges the game registers on a live snapshot; a vsnap has none
        self.subset = bool(p.str("m_strSnapshotSubset"))
        if self.subset:
            p.note("snapshot subset")
        self.counter = 0

    def column(self, sim):
        """The snapshot's values of the attribute read ((count,) or (count, 3)), or None."""
        if self.control_point < 0 or self.subset:
            return None
        snapshot = sim.snapshot(self.control_point)
        name = ATTRIBUTE_NAMES.get(self.read)
        if snapshot is None or snapshot.count == 0 or name is None:
            return None
        values = snapshot.arrays.get(name)
        if not isinstance(values, np.ndarray) or len(values) == 0:
            return None
        if (values.ndim == 2) != F.is_vector(self.write):
            return None
        return values

    def step(self, sim, v) -> int:
        return max(1, int(np.ravel(self.increment(sim, v))[0]))

    def draw(self, sim, count: int) -> int:
        if self.random_seed:
            sample = sim.random.sample(sim.random.seed + self.random_seed + self.counter)
            self.counter += 1
        else:
            sample = sim.random.next1()
        return min(int(count * sample), count - 1)

    def walk(self, sim, n: int, count: int, start: int, step: int, wrap: int = 0) -> tuple[np.ndarray, int]:
        """Rows for n particles from row ``start`` on (or drawn at random), and the row after the last. A plain walk
        (step 1, not reversed) goes back to row ``wrap`` past the end."""
        rows = np.zeros(n, np.int64)
        index = start
        plain = step == 1 and not self.reverse
        for i in range(n):
            if self.random:
                rows[i] = self.draw(sim, count)
                continue
            rows[i] = count - 1 - index if self.reverse else index
            index = (index + 1 if index + 1 < count else wrap) if plain else (index + step) % count
        return rows, index

    def local(self, sim, values):
        """Positions, velocities and normals moved into the local space control point's frame."""
        if self.local_cp < 0 or self.write not in (F.POSITION, F.PREVIOUS_POSITION, F.NORMAL):
            return values
        point = sim.control_point(self.local_cp)
        values = values @ point.rotation.T
        return values + point.position if self.write == F.POSITION else values


def read(data: bytes) -> Snapshot | None:
    resource = ResourceFile(data)
    info = resource.keyvalues("DATA")
    block = resource.block("SNAP")
    if info is None or block is None:
        return None
    raw = block_decompress(bytes(resource.block_data(block)))
    count = int(info.get("num_particles") or 0)
    strings = list(info.get("string_list") or [])
    arrays, joints, weights = {}, None, None
    for attribute in info.get("attributes") or []:
        name, kind = attribute.get("name"), attribute.get("type")
        start = int(attribute.get("data_offset") or 0)
        chunk = raw[start:start + int(attribute.get("data_size") or 0)]
        if kind in ("float3", "vector"):
            arrays[name] = np.frombuffer(chunk, "<f4", count * 3).reshape(count, 3).astype(np.float64)
        elif kind == "float":
            arrays[name] = np.frombuffer(chunk, "<f4", count).astype(np.float64)
        elif kind == "int":
            arrays[name] = np.frombuffer(chunk, "<i4", count).astype(np.float64)
        elif kind in ("string", "bone"):
            index = np.frombuffer(chunk, "<i4", count)
            arrays[name] = [strings[i] if 0 <= i < len(strings) else "" for i in index]
        elif kind == "skinning":
            rows = np.frombuffer(chunk, np.dtype([("joint", "<i2", 4), ("weight", "<f4", 4)]), count)
            joints = [[strings[j] if 0 <= j < len(strings) else "" for j in row] for row in rows["joint"]]
            weights = rows["weight"].astype(np.float64)
    return Snapshot(count, arrays, joints, weights)


def block_decompress(data: bytes) -> bytes:
    """Valve's block compression: a u32 size (top bit set = stored as is), then per 16 tokens a u16 mask whose set
    bits are back references (u16: offset - 1 in the top 12 bits, length - 3 in the low 4) and clear bits literals."""
    size = struct.unpack_from("<I", data)[0]
    if size & 0x80000000:
        size &= 0x7FFFFFFF
        return data[4:4 + size]
    out = bytearray()
    pos, mask, left = 4, 0, 0
    while len(out) < size:
        if left == 0:
            mask = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            left = 16
        if mask & 1:
            token = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            offset, length = (token >> 4) + 1, (token & 0xF) + 3
            if offset > len(out):
                raise ValueError("snapshot back reference is outside the output")
            for _ in range(length):
                out.append(out[-offset])
        else:
            out.append(data[pos])
            pos += 1
        mask >>= 1
        left -= 1
    return bytes(out[:size])
