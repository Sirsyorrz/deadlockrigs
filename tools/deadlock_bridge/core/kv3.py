"""Binary KeyValues3 (versions 1 to 5; uncompressed, LZ4 or Zstd), as found in DATA, CTRL, MDAT and similar blocks.

Objects read as dicts, arrays as lists, blobs as bytes. The layout follows ValveResourceFormat's reader.
"""
import struct

from . import native

_MAGIC = 0x4B563300
_LEGACY_MAGIC = 0x03564B56
_TRAILER = 0xFFEEDD00

_NONE, _LZ4, _ZSTD = 0, 1, 2

_I16 = struct.Struct("<h")
_U16 = struct.Struct("<H")
_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_F32 = struct.Struct("<f")
_I64 = struct.Struct("<q")
_U64 = struct.Struct("<Q")
_F64 = struct.Struct("<d")


class _Lane:
    __slots__ = ("data", "pos", "end")

    def __init__(self, data=b"", start=0, end=0):
        self.data = data
        self.pos = start
        self.end = end

    def take(self, count: int):
        pos = self.pos
        if count < 0 or count > self.end - pos:
            raise ValueError("KV3 data is truncated")
        self.pos = pos + count
        return self.data[pos:pos + count]

    def unpack(self, s: struct.Struct):
        pos = self.pos
        if pos + s.size > self.end:
            raise ValueError("KV3 data is truncated")
        self.pos = pos + s.size
        return s.unpack_from(self.data, pos)[0]

    def byte(self) -> int:
        pos = self.pos
        if pos >= self.end:
            raise ValueError("KV3 data is truncated")
        self.pos = pos + 1
        return self.data[pos]

    def cstring(self) -> str:
        end = self.data.index(0, self.pos, self.end) if isinstance(self.data, (bytes, bytearray)) else bytes(self.data[self.pos:self.end]).index(b"\0") + self.pos
        value = bytes(self.data[self.pos:end]).decode("utf-8", "replace")
        self.pos = end + 1
        return value


class _Lanes:
    __slots__ = ("b1", "b2", "b4", "b8")

    def __init__(self):
        self.b1 = self.b2 = self.b4 = self.b8 = _Lane()


class _Context:
    __slots__ = ("version", "strings", "types", "object_lengths", "blob_lengths", "blobs", "buffer", "aux")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _decompress(data, pos: int, compression: int, compressed: int, uncompressed: int):
    if compression == _NONE:
        return bytes(data[pos:pos + uncompressed]), pos + uncompressed
    if compression == _LZ4:
        out = bytearray(uncompressed)
        if native.lz4_decode(data[pos:pos + compressed], out) != uncompressed:
            raise ValueError("KV3 LZ4 buffer decoded to the wrong size")
        return out, pos + compressed
    return zstd_decompress(data[pos:pos + compressed], uncompressed), pos + compressed


def zstd_decompress(data, size: int) -> bytes:
    """Zstd data of a known decoded size, which may span several frames."""
    import io
    import zstandard
    data = bytes(data)
    out = zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    if len(out) != size:
        # Consecutive frames (KV3 before version 5 keeps its buffer and blobs as two).
        reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data), read_across_frames=True)
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = reader.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        out = b"".join(chunks)
    if len(out) != size:
        raise ValueError(f"Zstd data decoded to {len(out)} bytes, expected {size}")
    return out


def _read_lanes(buffer, offset: int, lanes: _Lanes, c1: int, c2: int, c4: int, c8: int, align_empty: bool) -> int:
    if c1 > 0:
        lanes.b1 = _Lane(buffer, offset, offset + c1)
        offset += c1
    if c2 > 0:
        offset = _align(offset, 2)
        lanes.b2 = _Lane(buffer, offset, offset + c2 * 2)
        offset += c2 * 2
    if c4 > 0:
        offset = _align(offset, 4)
        lanes.b4 = _Lane(buffer, offset, offset + c4 * 4)
        offset += c4 * 4
    if c8 > 0:
        offset = _align(offset, 8)
        lanes.b8 = _Lane(buffer, offset, offset + c8 * 8)
        offset += c8 * 8
    elif align_empty:
        offset = _align(offset, 8)
    return offset


def format_guid(data) -> bytes:
    return bytes(data[4:20])


def read(data) -> dict:
    """Reads a binary KV3 document; the root must be an object."""
    magic = _U32.unpack_from(data, 0)[0]
    if magic == _LEGACY_MAGIC:
        raise ValueError("legacy VKV3 KeyValues are not supported")
    if magic & 0xFFFFFF00 != _MAGIC:
        raise ValueError(f"not binary KeyValues3 (magic 0x{magic:08X})")
    version = magic & 0xFF
    if not 1 <= version <= 5:
        raise ValueError(f"KeyValues3 version {version} is not supported")

    root = _read_document(version, data)
    if not isinstance(root, dict):
        raise ValueError("KeyValues3 root is not an object")
    return root


def _read_document(version: int, data):
    pos = 20
    i32 = lambda: _I32.unpack_from(data, pos)[0]

    compression = i32(); pos += 4
    count2 = count_types = count_blobs = size_blobs = 0

    if version == 1:
        count1 = i32(); pos += 4
        count4 = i32(); pos += 4
        count8 = i32(); pos += 4
        size_total = i32(); pos += 4
        compressed_total = len(data) - pos
    else:
        dictionary = _U16.unpack_from(data, pos)[0]; pos += 4  # dictionary id, frame size
        count1 = i32(); pos += 4
        count4 = i32(); pos += 4
        count8 = i32(); pos += 4
        count_types = i32(); pos += 4
        pos += 4  # object count, array count
        size_total = i32(); pos += 4
        compressed_total = i32(); pos += 4
        count_blobs = i32(); pos += 4
        size_blobs = i32(); pos += 4
        if dictionary != 0:
            raise ValueError("KV3 compression dictionaries are not supported")

    if version >= 4:
        count2 = i32(); pos += 4
        pos += 4  # size of the blob compressed sizes

    size2 = compressed2 = 0
    c1b2 = c2b2 = c4b2 = c8b2 = objects_b2 = 0
    if version >= 5:
        size1 = i32(); pos += 4
        compressed1 = i32(); pos += 4
        size2 = i32(); pos += 4
        compressed2 = i32(); pos += 4
        c1b2 = i32(); pos += 4
        c2b2 = i32(); pos += 4
        c4b2 = i32(); pos += 4
        c8b2 = i32(); pos += 4
        pos += 4  # node count
        objects_b2 = i32(); pos += 4
        pos += 8  # array count, array element count
    else:
        size1 = size_total
        compressed1 = compressed_total

    ctx = _Context()
    ctx.version = version
    ctx.object_lengths = _Lane()
    ctx.blob_lengths = _Lane()
    ctx.blobs = _Lane()

    zstd_includes_blobs = version < 5 and compression == _ZSTD
    buffer1, pos = _decompress(data, pos, compression, compressed1, size1 + size_blobs if zstd_includes_blobs else size1)

    lanes1 = _Lanes()
    offset = _read_lanes(buffer1, 0, lanes1, count1, count2, count4, count8, version < 5)

    count_strings = lanes1.b4.unpack(_I32)
    ctx.aux = lanes1
    blob_sizes = None

    if version >= 5:
        ctx.strings = [lanes1.b1.cstring() for _ in range(count_strings)]
        buffer2, pos = _decompress(data, pos, compression, compressed2, size2)
        lanes2 = _Lanes()
        ctx.buffer = lanes2
        lengths_size = objects_b2 * 4
        ctx.object_lengths = _Lane(buffer2, 0, lengths_size)
        offset2 = _read_lanes(buffer2, lengths_size, lanes2, c1b2, c2b2, c4b2, c8b2, False)
        ctx.types = _Lane(buffer2, offset2, offset2 + count_types)
        offset2 += count_types
        if count_blobs == 0:
            _expect_trailer(_U32.unpack_from(buffer2, offset2)[0])
        else:
            blob_sizes = _Lane(buffer2, offset2, size2)
    else:
        ctx.buffer = lanes1
        strings_start = offset
        lane = _Lane(buffer1, offset, size1)
        strings = []
        for _ in range(count_strings):
            start = lane.pos
            strings.append(lane.cstring())
        ctx.strings = strings
        offset = lane.pos
        types_length = size_total - offset - 4 if version == 1 else count_types - (offset - strings_start)
        ctx.types = _Lane(buffer1, offset, offset + types_length)
        offset += types_length
        if count_blobs == 0:
            _expect_trailer(_U32.unpack_from(buffer1, offset)[0])
        else:
            blob_sizes = _Lane(buffer1, offset, size1)

    if count_blobs > 0:
        length_bytes = bytes(blob_sizes.take(count_blobs * 4))
        ctx.blob_lengths = _Lane(length_bytes, 0, len(length_bytes))
        _expect_trailer(blob_sizes.unpack(_U32))

        if zstd_includes_blobs:
            ctx.blobs = _Lane(buffer1, size1, size1 + size_blobs)
        elif compression != _LZ4:
            blobs, pos = _decompress(data, pos, compression, compressed_total - compressed1 - compressed2, size_blobs)
            ctx.blobs = _Lane(blobs, 0, size_blobs)
        else:
            # Each blob is split into LZ4 frames that never span two blobs; frames chain on earlier output.
            blobs = bytearray(size_blobs)
            blob_offset = 0
            for i in range(0, len(length_bytes), 4):
                blob_end = blob_offset + _I32.unpack_from(length_bytes, i)[0]
                while blob_offset < blob_end:
                    frame = blob_sizes.unpack(_U16)
                    decoded = native.lz4_decode(data[pos:pos + frame], blobs, blob_offset, blob_end)
                    pos += frame
                    if decoded < 1:
                        raise ValueError("KV3 LZ4 blob frame is empty")
                    blob_offset += decoded
            ctx.blobs = _Lane(blobs, 0, size_blobs)

        _expect_trailer(_U32.unpack_from(data, pos)[0] & 0xFFFFFFFF)

    root_type = _read_type(ctx)
    return _read_value(ctx, root_type, ctx.buffer)


def _expect_trailer(value: int):
    if value != _TRAILER:
        raise ValueError(f"KV3 trailer mismatch (0x{value:08X})")


def _read_type(ctx: _Context) -> int:
    types = ctx.types
    value = types.byte()
    if ctx.version >= 3:
        if value & 0x80:
            types.byte()
        if value & 0x40:
            types.byte()
        return value & 0x3F
    if value & 0x80:
        types.byte()
    return value & 0x7F


def _read_value(ctx: _Context, kind: int, lane: _Lanes):
    buffer = ctx.buffer

    if kind == 9:  # object
        length = ctx.object_lengths.unpack(_I32) if ctx.version >= 5 else buffer.b4.unpack(_I32)
        result = {}
        strings = ctx.strings
        b4 = buffer.b4
        for _ in range(length):
            member = _read_type(ctx)
            name_id = b4.unpack(_I32)
            name = strings[name_id] if 0 <= name_id < len(strings) else ""
            result[name] = _read_value(ctx, member, buffer)
        return result

    if kind == 8:  # array
        length = buffer.b4.unpack(_I32)
        return [_read_value(ctx, _read_type(ctx), buffer) for _ in range(length)]

    if kind in (10, 24, 25):  # typed arrays
        length = buffer.b4.unpack(_I32) if kind == 10 else buffer.b1.byte()
        element = _read_type(ctx)
        element_lane = ctx.aux if kind == 25 else buffer
        return _read_typed_array(ctx, element, element_lane, length)

    if kind == 6:  # string
        index = buffer.b4.unpack(_I32)
        strings = ctx.strings
        return strings[index] if 0 <= index < len(strings) else ""

    return _read_scalar(ctx, kind, lane)


def _read_typed_array(ctx, element: int, lane: _Lanes, length: int):
    # Plain numbers read in one go.
    simple = _SIMPLE.get(element)
    if simple is not None and length > 0:
        width, fmt, which = simple
        source = getattr(lane, which)
        raw = source.take(width * length)
        return list(struct.unpack("<%d%s" % (length, fmt), raw))
    return [_read_value(ctx, element, lane) for _ in range(length)]


# Element type -> (byte width, struct code, lane)
_SIMPLE = {
    3: (8, "q", "b8"), 4: (8, "Q", "b8"), 5: (8, "d", "b8"),
    11: (4, "i", "b4"), 12: (4, "I", "b4"), 19: (4, "f", "b4"),
    20: (2, "h", "b2"), 21: (2, "H", "b2"), 22: (1, "b", "b1"), 23: (1, "B", "b1"),
}


def _read_scalar(ctx: _Context, kind: int, lane: _Lanes):
    if kind == 1:
        return None
    if kind == 13:
        return True
    if kind == 14:
        return False
    if kind == 15:
        return 0
    if kind == 16:
        return 1
    if kind == 17:
        return 0.0
    if kind == 18:
        return 1.0
    if kind == 26:
        return 0
    if kind == 27:
        return 1
    if kind == 2:
        return lane.b1.byte() != 0
    if kind == 22:
        v = lane.b1.byte()
        return v - 256 if v > 127 else v
    if kind == 23:
        return lane.b1.byte()
    if kind == 20:
        return lane.b2.unpack(_I16)
    if kind == 21:
        return lane.b2.unpack(_U16)
    if kind == 11:
        return lane.b4.unpack(_I32)
    if kind == 12:
        return lane.b4.unpack(_U32)
    if kind == 19:
        return lane.b4.unpack(_F32)
    if kind == 3:
        return lane.b8.unpack(_I64)
    if kind == 4:
        return lane.b8.unpack(_U64)
    if kind == 5:
        return lane.b8.unpack(_F64)
    if kind == 7:  # blob
        buffer = ctx.buffer
        if ctx.version < 2:
            return bytes(buffer.b1.take(buffer.b4.unpack(_I32)))
        return bytes(ctx.blobs.take(ctx.blob_lengths.unpack(_I32)))
    raise ValueError(f"unknown KV3 node type {kind}")


# Loose conversions for KV3 values.

def as_int(value, fallback=0) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return fallback
    return fallback


def as_float(value, fallback=0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def as_bool(value, fallback=False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.lower() == "true" or value == "1"
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return fallback
