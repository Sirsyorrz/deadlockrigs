"""Compiled textures (vtex_c): header, extra data and per-mip LZ4/Zstd decompression, and DDS output.

The pixel data follows the DATA block: every mip of every face or slice, smallest mip first. Block compressed formats
are written out as DDS files as they are, which Blender (through OpenImageIO) decodes.
"""
import struct

import numpy as np

from . import kv3, native
from .resource import ResourceFile

# VTEX_FORMAT
DXT1, DXT5, I8, RGBA8888, R16, RG1616, RGBA16161616, R16F, RG1616F, RGBA16161616F = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
R32F, RG3232F, RGB323232F, RGBA32323232F = 11, 12, 13, 14
JPEG_RGBA8888, PNG_RGBA8888, JPEG_DXT5, PNG_DXT5 = 15, 16, 17, 18
BC6H, BC7, ATI2N, IA88, ETC2, ETC2_EAC, R11_EAC, RG11_EAC, ATI1N, BGRA8888 = 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
WEBP_RGBA8888, WEBP_DXT5, R32_UINT, A8, R8_UNORM = 29, 30, 31, 32, 33

FLAG_CLAMP_S, FLAG_CLAMP_T, FLAG_CLAMP_U = 1, 2, 4
FLAG_NO_LOD, FLAG_CUBE, FLAG_VOLUME, FLAG_ARRAY = 8, 16, 32, 64

BLOCK_COMPRESSED = {DXT1, DXT5, BC6H, BC7, ATI1N, ATI2N, ETC2, ETC2_EAC, R11_EAC, RG11_EAC}
UNCOMPRESSED = {RGBA8888, BGRA8888, RGBA16161616F, RGBA32323232F, RGB323232F, RG1616F, RG3232F, R16F, R32F, R16, RG1616,
                RGBA16161616, I8, R8_UNORM, A8, IA88}   # the formats decode_uncompressed reads
RAW_IMAGE = {PNG_RGBA8888, PNG_DXT5, JPEG_RGBA8888, JPEG_DXT5, WEBP_RGBA8888, WEBP_DXT5}
HDR_FORMATS = {BC6H, R16F, RG1616F, RGBA16161616F, R32F, RG3232F, RGB323232F, RGBA32323232F}

_BLOCK_SIZE = {
    DXT1: 8, ATI1N: 8, ETC2: 8, R11_EAC: 8,
    DXT5: 16, BC6H: 16, BC7: 16, ATI2N: 16, ETC2_EAC: 16, RG11_EAC: 16,
    RGBA8888: 4, BGRA8888: 4, RG1616: 4, RG1616F: 4, R32F: 4, R32_UINT: 4,
    R16: 2, R16F: 2, IA88: 2,
    RGBA16161616: 8, RGBA16161616F: 8, RG3232F: 8,
    RGB323232F: 12, RGBA32323232F: 16,
}

# DXGI formats for the DDS DX10 header.
_DXGI = {
    DXT1: 71, DXT5: 77, ATI1N: 80, ATI2N: 83, BC6H: 95, BC7: 98,
    RGBA8888: 28, BGRA8888: 87, RGBA16161616F: 10, RGBA32323232F: 2, RG1616F: 34, R16F: 54, R32F: 41,
    RG3232F: 16, RGB323232F: 6, RGBA16161616: 11, RG1616: 35, R16: 56, I8: 61, R8_UNORM: 61, A8: 65, IA88: 49,
}


class CompiledTexture:
    def __init__(self, data: bytes):
        self.resource = ResourceFile(data)
        block = self.resource.block("DATA")
        if block is None:
            raise ValueError("texture has no DATA block")
        header = self.resource.block_data(block)
        self._data_offset = block[1] + block[2]

        version, self.flags = struct.unpack_from("<HH", header, 0)
        if version != 1:
            raise ValueError(f"texture header version {version} is not supported")
        self.reflectivity = struct.unpack_from("<4f", header, 4)
        self.width, self.height, depth = struct.unpack_from("<HHH", header, 20)
        self.depth = max(depth, 1)
        self.format = header[26]
        self.mip_count = max(header[27], 1)
        self.display_width, self.display_height = self.width, self.height
        self.sheet = None
        self.radiance_sh = None
        self._mip_compression = 0
        self._compressed_sizes = None

        extra_offset = 32 + struct.unpack_from("<I", header, 32)[0]
        extra_count = struct.unpack_from("<I", header, 36)[0]
        for i in range(extra_count):
            entry = extra_offset + i * 12
            kind, rel, size = struct.unpack_from("<III", header, entry)
            offset = entry + 4 + rel
            if kind == 3:  # metadata
                w, h = struct.unpack_from("<HH", header, offset + 2)
                if 0 < w <= self.width and 0 < h <= self.height:
                    self.display_width, self.display_height = w, h
            elif kind == 2:  # sheet
                self.sheet = _read_sheet(bytes(header[offset:offset + size]))
            elif kind == 4:  # compressed mip sizes
                self._mip_compression = struct.unpack_from("<I", header, offset)[0]
                sizes_offset = offset + 4 + struct.unpack_from("<I", header, offset + 4)[0]
                count = struct.unpack_from("<I", header, offset + 8)[0]
                self._compressed_sizes = struct.unpack_from("<%di" % count, header, sizes_offset)
            elif kind == 5:  # cubemap radiance SH
                coefficients_offset, count = struct.unpack_from("<II", header, offset)
                self.radiance_sh = struct.unpack_from("<%df" % count, header, offset + coefficients_offset)

    @property
    def is_cube(self):
        return bool(self.flags & FLAG_CUBE)

    @property
    def is_volume(self):
        return bool(self.flags & FLAG_VOLUME)

    @property
    def is_array(self):
        return bool(self.flags & FLAG_ARRAY)

    @property
    def is_raw_image(self):
        return self.format in RAW_IMAGE

    @property
    def is_hdr(self):
        return self.format in HDR_FORMATS

    @property
    def block_compressed(self):
        return self.format in BLOCK_COMPRESSED

    def image_count(self, mip: int) -> int:
        depth = max(self.depth >> mip, 1) if self.is_volume else self.depth
        return depth * 6 if self.is_cube else depth

    def mip_size(self, mip: int) -> int:
        width = max(self.width >> mip, 1)
        height = max(self.height >> mip, 1)
        if self.block_compressed:
            width = (width + 3) // 4
            height = (height + 3) // 4
        return width * height * self.image_count(mip) * _BLOCK_SIZE.get(self.format, 1)

    def read_raw_image(self) -> bytes:
        return self.resource.data[self._data_offset:]

    def read_mip(self, mip: int) -> bytes:
        """One mip level: every face and slice of it, in order."""
        data = self.resource.data
        source = self._data_offset
        sizes = self._compressed_sizes
        for level in range(self.mip_count - 1, -1, -1):
            size = self.mip_size(level)
            stored = min(sizes[level], size) if sizes is not None and level < len(sizes) else size
            if level == mip:
                return self._decode(memoryview(data)[source:source + stored], size)
            source += stored
        raise IndexError(mip)

    def _decode(self, stored, size: int) -> bytes:
        if len(stored) == size or self._mip_compression == 0:
            return bytes(stored)
        if self._mip_compression == 1:
            out = bytearray(size)
            if native.lz4_decode(stored, out) != size:
                raise ValueError("texture mip decoded to the wrong size")
            return bytes(out)
        if self._mip_compression == 2:
            return kv3.zstd_decompress(stored, size)
        raise ValueError(f"texture mip compression {self._mip_compression} is not supported")

    # DDS output -------------------------------------------------------------------------------------------------

    def dds_supported(self) -> bool:
        return self.format in _DXGI

    def to_dds(self, mip: int = 0, image: int = 0, mip_count: int = 1) -> bytes:
        """A DDS file of one face/slice (``image``) from ``mip``, with ``mip_count`` levels below it."""
        width = max(self.width >> mip, 1)
        height = max(self.height >> mip, 1)
        levels = []
        for level in range(mip, min(mip + mip_count, self.mip_count)):
            data = self.read_mip(level)
            per_image = len(data) // self.image_count(level)
            levels.append(data[image * per_image:(image + 1) * per_image])

        body = b"".join(levels)
        if self.format == RGBA8888:
            return _dds_header(width, height, len(levels), 28) + body
        return _dds_header(width, height, len(levels), _DXGI[self.format]) + body

    def decode_rgba(self, mip: int = 0, image: int = 0) -> np.ndarray | None:
        """Uncompressed formats as a float32 (h, w, 4) array, top row first. None for block compressed ones."""
        width = max(self.width >> mip, 1)
        height = max(self.height >> mip, 1)
        data = self.read_mip(mip)
        per_image = len(data) // self.image_count(mip)
        raw = data[image * per_image:(image + 1) * per_image]
        return decode_uncompressed(self.format, raw, width, height)


def decode_uncompressed(fmt: int, raw: bytes, width: int, height: int) -> np.ndarray | None:
    n = width * height
    out = np.ones((n, 4), dtype=np.float32)
    if fmt == RGBA8888:
        out[:] = np.frombuffer(raw, np.uint8, n * 4).reshape(n, 4) / 255.0
    elif fmt == BGRA8888:
        a = np.frombuffer(raw, np.uint8, n * 4).reshape(n, 4) / 255.0
        out[:] = a[:, [2, 1, 0, 3]]
    elif fmt == RGBA16161616F:
        out[:] = np.frombuffer(raw, np.float16, n * 4).reshape(n, 4)
    elif fmt == RGBA32323232F:
        out[:] = np.frombuffer(raw, np.float32, n * 4).reshape(n, 4)
    elif fmt == RGB323232F:
        out[:, :3] = np.frombuffer(raw, np.float32, n * 3).reshape(n, 3)
    elif fmt == RG1616F:
        out[:, :2] = np.frombuffer(raw, np.float16, n * 2).reshape(n, 2)
        out[:, 2] = 0
    elif fmt == RG3232F:
        out[:, :2] = np.frombuffer(raw, np.float32, n * 2).reshape(n, 2)
        out[:, 2] = 0
    elif fmt == R16F:
        out[:, 0] = np.frombuffer(raw, np.float16, n)
        out[:, 1] = out[:, 2] = out[:, 0]
    elif fmt == R32F:
        out[:, 0] = np.frombuffer(raw, np.float32, n)
        out[:, 1] = out[:, 2] = out[:, 0]
    elif fmt == R16:
        out[:, 0] = np.frombuffer(raw, np.uint16, n) / 65535.0
        out[:, 1] = out[:, 2] = out[:, 0]
    elif fmt == RG1616:
        out[:, :2] = np.frombuffer(raw, np.uint16, n * 2).reshape(n, 2) / 65535.0
        out[:, 2] = 0
    elif fmt == RGBA16161616:
        out[:] = np.frombuffer(raw, np.uint16, n * 4).reshape(n, 4) / 65535.0
    elif fmt in (I8, R8_UNORM):
        v = np.frombuffer(raw, np.uint8, n) / 255.0
        out[:, 0] = out[:, 1] = out[:, 2] = v
    elif fmt == A8:
        out[:, :3] = 1
        out[:, 3] = np.frombuffer(raw, np.uint8, n) / 255.0
    elif fmt == IA88:
        a = np.frombuffer(raw, np.uint8, n * 2).reshape(n, 2) / 255.0
        out[:, 0] = out[:, 1] = out[:, 2] = a[:, 0]
        out[:, 3] = a[:, 1]
    else:
        return None
    return out.reshape(height, width, 4)


def _dds_header(width: int, height: int, mips: int, dxgi: int) -> bytes:
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | (0x20000 if mips > 1 else 0)
    header = struct.pack("<4sIIIIIII44x", b"DDS ", 124, flags, height, width, 0, 1, mips)
    pixel_format = struct.pack("<II4s20x", 32, 0x4, b"DX10")
    caps = struct.pack("<IIII4x", 0x1000 | (0x400008 if mips > 1 else 0), 0, 0, 0)
    dx10 = struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    return header + pixel_format + caps + dx10


def _read_sheet(data: bytes):
    """A sprite sheet: sequences of frames, each a list of (min u, min v, max u, max v) rectangles."""
    def offset_string(field):
        start = field + struct.unpack_from("<i", data, field)[0]
        if start <= field or start >= len(data):
            return ""
        end = data.find(b"\0", start)
        return data[start:end if end >= 0 else len(data)].decode("utf-8", "replace")

    sequences_offset, count = struct.unpack_from("<iI", data, 0)
    sequences = []
    position = sequences_offset
    for _ in range(count):
        seq_id = struct.unpack_from("<I", data, position)[0]
        clamp = data[position + 4] != 0
        frames_offset = position + 8 + struct.unpack_from("<i", data, position + 8)[0]
        frame_count = struct.unpack_from("<I", data, position + 12)[0]
        total_time = struct.unpack_from("<f", data, position + 16)[0]
        name = offset_string(position + 20)
        frames = []
        for f in range(frame_count):
            entry = frames_offset + f * 12
            display_time = struct.unpack_from("<f", data, entry)[0]
            images_offset = entry + 4 + struct.unpack_from("<i", data, entry + 4)[0]
            image_count = struct.unpack_from("<I", data, entry + 8)[0]
            images = [struct.unpack_from("<8f", data, images_offset + i * 32) for i in range(image_count)]
            frames.append((display_time, images))
        sequences.append({"id": seq_id, "clamp": clamp, "total_time": total_time, "name": name, "frames": frames})
        position += 32
    return sequences
