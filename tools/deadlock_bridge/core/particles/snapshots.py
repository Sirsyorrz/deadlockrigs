"""Particle snapshots (vsnap): a fixed set of points, often traced off a model's surface and skinned to its bones,
that a system publishes on a control point (m_hSnapshot, m_nSnapshotControlPoint) for its functions and its
children to read. The SNAP block holds one array per attribute the DATA block lists, behind Valve's block
compression."""
import struct

import numpy as np

from ..resource import ResourceFile


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
