"""Vertex and index buffers (VBIB/MBUF blocks and KV3 described MVTX/MIDX buffers), decoded to numpy arrays.

Attribute decoding is vectorised over every vertex of a buffer. Packed normals and tangent frames follow
ValveResourceFormat's interpretation, as the s&box mount does.
"""
import struct

import numpy as np

from . import kv3, native

# DXGI_FORMAT -> (numpy dtype, components, normalisation): 'f' float, 'u' unorm, 's' snorm, 'i' integer
_FORMATS = {
    2: ("<f4", 4, "f"), 3: ("<u4", 4, "i"), 4: ("<i4", 4, "i"),
    6: ("<f4", 3, "f"), 7: ("<u4", 3, "i"), 8: ("<i4", 3, "i"),
    10: ("<f2", 4, "f"), 11: ("<u2", 4, "u"), 12: ("<u2", 4, "i"), 13: ("<i2", 4, "s"), 14: ("<i2", 4, "i"),
    16: ("<f4", 2, "f"), 17: ("<u4", 2, "i"), 18: ("<i4", 2, "i"),
    28: ("u1", 4, "u"), 29: ("u1", 4, "u"), 30: ("u1", 4, "i"), 31: ("i1", 4, "s"), 32: ("i1", 4, "i"),
    34: ("<f2", 2, "f"), 35: ("<u2", 2, "u"), 36: ("<u2", 2, "i"), 37: ("<i2", 2, "s"), 38: ("<i2", 2, "i"),
    41: ("<f4", 1, "f"), 42: ("<u4", 1, "i"), 43: ("<i4", 1, "i"),
    49: ("u1", 2, "u"), 50: ("u1", 2, "i"), 51: ("i1", 2, "s"), 52: ("i1", 2, "i"),
    54: ("<f2", 1, "f"), 56: ("<u2", 1, "u"), 57: ("<u2", 1, "i"), 58: ("<i2", 1, "s"), 59: ("<i2", 1, "i"),
    61: ("u1", 1, "u"), 62: ("u1", 1, "i"), 63: ("i1", 1, "s"), 64: ("i1", 1, "i"),
    87: ("u1", 4, "u"),
}

_FORMAT_NAMES = {
    "R32G32B32A32_FLOAT": 2, "R32G32B32A32_UINT": 3, "R32G32B32A32_SINT": 4, "R32G32B32_FLOAT": 6,
    "R32G32B32_UINT": 7, "R32G32B32_SINT": 8, "R16G16B16A16_FLOAT": 10, "R16G16B16A16_UNORM": 11,
    "R16G16B16A16_UINT": 12, "R16G16B16A16_SNORM": 13, "R16G16B16A16_SINT": 14, "R32G32_FLOAT": 16,
    "R32G32_UINT": 17, "R32G32_SINT": 18, "R10G10B10A2_UNORM": 24, "R10G10B10A2_UINT": 25, "R8G8B8A8_UNORM": 28,
    "R8G8B8A8_UNORM_SRGB": 29, "R8G8B8A8_UINT": 30, "R8G8B8A8_SNORM": 31, "R8G8B8A8_SINT": 32, "R16G16_FLOAT": 34,
    "R16G16_UNORM": 35, "R16G16_UINT": 36, "R16G16_SNORM": 37, "R16G16_SINT": 38, "R32_FLOAT": 41, "R32_UINT": 42,
    "R32_SINT": 43, "R8G8_UNORM": 49, "R8G8_UINT": 50, "R8G8_SNORM": 51, "R8G8_SINT": 52, "R16_FLOAT": 54,
    "R16_UNORM": 56, "R16_UINT": 57, "R16_SNORM": 58, "R16_SINT": 59, "R8_UNORM": 61, "R8_UINT": 62,
    "R8_SNORM": 63, "R8_SINT": 64, "B8G8R8A8_UNORM": 87,
}

R32_UINT = 42
R8G8B8A8_UNORM = 28


class Field:
    __slots__ = ("semantic", "index", "format", "offset", "slot")

    def __init__(self, semantic, index, fmt, offset, slot=0):
        self.semantic = semantic
        self.index = index
        self.format = fmt
        self.offset = offset
        self.slot = slot

    def __repr__(self):
        return f"{self.semantic}{self.index}:{self.format}@{self.offset}"


class Buffer:
    """A decoded vertex or index buffer."""

    def __init__(self, count: int, stride: int, fields: list[Field], data: bytes):
        self.count = count
        self.stride = stride
        self.fields = fields
        self.data = data
        self._rows = None

    def find(self, semantic: str, index: int = 0) -> Field | None:
        for field in self.fields:
            if field.semantic == semantic and field.index == index:
                return field
        return None

    def rows(self) -> np.ndarray:
        if self._rows is None:
            self._rows = np.frombuffer(self.data, np.uint8, self.count * self.stride).reshape(self.count, self.stride)
        return self._rows

    def raw(self, field: Field, start: int = 0, count: int | None = None) -> np.ndarray:
        """The attribute's components as stored, (n, components)."""
        dtype, components, _ = _FORMATS[field.format] if field.format in _FORMATS else ("<u4", 1, "i")
        dtype = np.dtype(dtype)
        end = self.count if count is None else start + count
        rows = self.rows()[start:end, field.offset:field.offset + dtype.itemsize * components]
        return np.ascontiguousarray(rows).view(dtype).reshape(-1, components)

    def read(self, field: Field, start: int = 0, count: int | None = None) -> np.ndarray:
        """The attribute as float32 (n, components), UNORM and SNORM normalised."""
        if field.format in (24, 25):  # R10G10B10A2
            packed = self.raw(Field(field.semantic, field.index, 42, field.offset), start, count)[:, 0]
            out = np.stack([packed & 0x3FF, (packed >> 10) & 0x3FF, (packed >> 20) & 0x3FF, packed >> 30], 1).astype(np.float32)
            if field.format == 24:
                out /= np.array([1023, 1023, 1023, 3], np.float32)
            return out
        if field.format == 87:  # BGRA
            values = self.raw(field, start, count).astype(np.float32) / 255.0
            return values[:, [2, 1, 0, 3]]
        if field.format not in _FORMATS:
            raise ValueError(f"vertex attribute {field.semantic} uses unsupported format {field.format}")
        _, _, kind = _FORMATS[field.format]
        values = self.raw(field, start, count)
        if kind == "f":
            return values.astype(np.float32)
        if kind == "u":
            return values.astype(np.float32) / float(np.iinfo(values.dtype).max)
        if kind == "s":
            return np.maximum(values.astype(np.float32) / float(np.iinfo(values.dtype).max), -1.0)
        return values.astype(np.float32)

    def indices(self) -> np.ndarray:
        return np.frombuffer(self.data, np.uint16 if self.stride == 2 else np.uint32, self.count)


def unpack_tangent_frames(packed: np.ndarray):
    """32 bit tangent frames (octahedral normal in 20 bits, tangent angle in 11, sign in 1) -> normals, tangents (w)."""
    packed = packed.astype(np.uint32)
    sign = packed & 1
    angle_bits = ((packed >> 1) & 0x7FF).astype(np.float32)
    x = ((packed >> 12) & 0x3FF).astype(np.float32) / 1023.0 * 2.0 - 1.0
    y = ((packed >> 22) & 0x3FF).astype(np.float32) / 1023.0 * 2.0 - 1.0
    z = 1.0 - np.abs(x) - np.abs(y)

    fold = np.clip(-z, 0.0, 1.0)
    x = x + np.where(x >= 0, -fold, fold)
    y = y + np.where(y >= 0, -fold, fold)
    length = np.sqrt(x * x + y * y + z * z)
    x /= length
    y /= length
    z /= length

    tangent_sign = np.where(z >= 0, 1.0, -1.0).astype(np.float32)
    reciprocal = 1.0 / (tangent_sign + z)
    tx = -tangent_sign * (x * x) * reciprocal + 1.0
    ty = -tangent_sign * (x * y) * reciprocal
    tz = -tangent_sign * x

    angle = angle_bits / 2047.0 * (2.0 * np.pi)
    c = np.cos(angle)
    s = np.sin(angle)
    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx

    normals = np.stack([x, y, z], 1).astype(np.float32)
    tangents = np.stack([tx * c + cx * s, ty * c + cy * s, tz * c + cz * s, np.where(sign == 0, -1.0, 1.0)], 1).astype(np.float32)
    return normals, tangents


def _unpack_byte_normals(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x = xs.astype(np.float32) - 128.0
    y = ys.astype(np.float32) - 128.0
    z_sign_bit = (x < 0).astype(np.float32)
    t_sign_bit = (y < 0).astype(np.float32)
    z_sign = -(2 * z_sign_bit - 1)
    t_sign = -(2 * t_sign_bit - 1)
    x = x * z_sign - z_sign_bit - 64
    y = y * t_sign - t_sign_bit - 64
    x_sign_bit = (x < 0).astype(np.float32)
    y_sign_bit = (y < 0).astype(np.float32)
    x_sign = -(2 * x_sign_bit - 1)
    y_sign = -(2 * y_sign_bit - 1)
    x = (x * x_sign - x_sign_bit) / 63.0
    y = (y * y_sign - y_sign_bit) / 63.0
    z = 1.0 - x - y
    inv = 1.0 / np.sqrt(x * x + y * y + z * z)
    return np.stack([x * inv * x_sign, y * inv * y_sign, z * inv * z_sign], 1).astype(np.float32)


def read_normals(buffer: Buffer, field: Field, start: int, count: int):
    """Normals (n, 3) and, when the format packs them, tangents (n, 4) with the bitangent sign in w (else None)."""
    if field.format == R32_UINT:
        return unpack_tangent_frames(buffer.raw(field, start, count)[:, 0])
    if field.format == R8G8B8A8_UNORM:
        raw = buffer.raw(field, start, count)
        normals = _unpack_byte_normals(raw[:, 0], raw[:, 1])
        tangents = np.empty((len(raw), 4), np.float32)
        tangents[:, :3] = _unpack_byte_normals(raw[:, 2], raw[:, 3])
        tangents[:, 3] = np.where(raw[:, 3] < 128, -1.0, 1.0)
        return normals, tangents
    values = buffer.read(field, start, count)
    return np.ascontiguousarray(values[:, :3]), None


def read_blend(buffer: Buffer, indices_field: Field, weights_field: Field | None, bone_weight_count: int, start: int, count: int):
    """Bone indices (n, k) int32 and weights (n, k) float32, mesh-local."""
    raw = buffer.raw(indices_field, start, count)
    fmt = indices_field.format
    if fmt in (30, 12):  # R8G8B8A8_UINT, R16G16B16A16_UINT (8 byte indices)
        bytes_ = np.ascontiguousarray(raw).view(np.uint8).reshape(len(raw), -1)
        indices = bytes_.astype(np.int32)
    elif fmt in (14, 4):  # R16G16B16A16_SINT, R32G32B32A32_SINT (16 bit indices)
        indices = np.ascontiguousarray(raw).view(np.uint16).reshape(len(raw), -1).astype(np.int32)
    elif fmt == 38:
        indices = np.ascontiguousarray(raw).view(np.uint16).reshape(len(raw), -1).astype(np.int32)
    else:
        raise ValueError(f"blend indices use unsupported format {fmt}")
    k = indices.shape[1]

    if weights_field is None:
        shared = min(max(bone_weight_count, 1), k)
        weights = np.zeros((len(raw), k), np.float32)
        weights[:, :shared] = 1.0 / shared
        return indices, weights

    wraw = buffer.raw(weights_field, start, count)
    if weights_field.format in (28, 11):
        w = np.ascontiguousarray(wraw).view(np.uint8).reshape(len(wraw), -1).astype(np.float32) / 255.0
    elif weights_field.format == 35:
        w = wraw.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"blend weights use unsupported format {weights_field.format}")
    weights = np.zeros((len(raw), k), np.float32)
    n = min(k, w.shape[1])
    weights[:, :n] = w[:, :n]
    return indices, weights


def _decode(raw, count: int, stride: int, is_vertex: bool, zstd: bool, meshopt: bool) -> bytes:
    size = count * stride
    if zstd:
        raw = kv3.zstd_decompress(raw, size) if not meshopt else _zstd_any(raw)
    if not meshopt:
        if len(raw) != size:
            raise ValueError(f"mesh buffer decompressed to {len(raw)} bytes, expected {size}")
        return bytes(raw)
    if is_vertex:
        return native.meshopt_decode_vertex(count, stride, raw)
    return native.meshopt_decode_index(count, stride, raw)


def _zstd_any(raw) -> bytes:
    import io
    import zstandard
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(bytes(raw)), read_across_frames=True)
    return reader.readall()


def read_buffer_block(block) -> tuple[list[Buffer], list[Buffer]]:
    """The vertex and index buffers of a legacy VBIB or MBUF block."""
    vertex_offset, vertex_count, index_offset, index_count = struct.unpack_from("<IIII", block, 0)
    vertices = [_read_buffer_header(block, vertex_offset + i * 24, True) for i in range(vertex_count)]
    indices = [_read_buffer_header(block, 8 + index_offset + i * 24, False) for i in range(index_count)]
    return vertices, indices


def _read_buffer_header(block, position: int, is_vertex: bool) -> Buffer:
    count, size_flags = struct.unpack_from("<Ii", block, position)
    stride = size_flags & 0x3FFFFFF
    meshopt = (size_flags & 0x4000000) == 0
    zstd = (size_flags & 0x8000000) != 0
    attributes = position + 8 + struct.unpack_from("<I", block, position + 8)[0]
    attribute_count = struct.unpack_from("<I", block, position + 12)[0]
    data_position = position + 16 + struct.unpack_from("<I", block, position + 16)[0]
    data_size = struct.unpack_from("<i", block, position + 20)[0]

    fields = []
    for i in range(attribute_count):
        a = attributes + i * 56
        name = bytes(block[a:a + 32]).split(b"\0", 1)[0].decode("ascii", "replace").upper()
        index, fmt, offset, slot = struct.unpack_from("<iIIi", block, a + 32)
        fields.append(Field(name, index, fmt, offset, slot))

    raw = block[data_position:data_position + data_size]
    data = _decode(raw, count, stride, is_vertex, zstd, meshopt) if count * stride > data_size else bytes(raw)
    return Buffer(count, stride, fields, data)


def _read_format(value) -> int:
    if isinstance(value, str):
        name = value[len("DXGI_FORMAT_"):] if value.startswith("DXGI_FORMAT_") else value
        return _FORMAT_NAMES.get(name, 0)
    return kv3.as_int(value)


def read_described_buffer(description: dict, resource, is_vertex: bool) -> Buffer:
    """A buffer described by KeyValues: inline (m_pData) or in another block (m_nBlockIndex)."""
    count = kv3.as_int(description.get("m_nElementCount"))
    stride = kv3.as_int(description.get("m_nElementSizeInBytes"))
    fields = []
    for f in description.get("m_inputLayoutFields") or []:
        name = f.get("m_pSemanticName")
        if isinstance(name, (bytes, bytearray)):
            name = bytes(name).split(b"\0", 1)[0].decode("ascii", "replace")
        fields.append(Field((name or "").upper(), kv3.as_int(f.get("m_nSemanticIndex")), _read_format(f.get("m_Format")),
                            kv3.as_int(f.get("m_nOffset")), kv3.as_int(f.get("m_nSlot"))))

    inline = description.get("m_pData")
    if isinstance(inline, (bytes, bytearray)) and inline:
        data = inline if len(inline) == count * stride else _decode(inline, count, stride, is_vertex, False, True)
    else:
        raw = resource.block_data(kv3.as_int(description.get("m_nBlockIndex")))
        meshopt = kv3.as_bool(description.get("m_bMeshoptCompressed"))
        zstd = kv3.as_bool(description.get("m_bCompressedZSTD"))
        data = _decode(raw, count, stride, is_vertex, zstd, meshopt) if meshopt or zstd else bytes(raw)

    if len(data) < count * stride:
        raise ValueError(f"mesh buffer has {len(data)} bytes, expected {count * stride}")
    return Buffer(count, stride, fields, data)
