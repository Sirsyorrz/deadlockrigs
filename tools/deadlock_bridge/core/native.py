"""LZ4 and meshoptimizer decoding, through the bundled native library when it loads, else in Python.

The Python versions are straight ports of the same decoders. They give the same bytes but are far slower, so a map's
geometry takes minutes instead of seconds without the library.
"""
import ctypes
import os
import sys

import numpy as np

_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True

    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "native")
    names = {"win32": "dlnative.dll", "darwin": "libdlnative.dylib"}.get(sys.platform, "libdlnative.so")
    path = os.path.join(folder, names)
    if not os.path.isfile(path):
        return None

    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None

    lib.dl_lz4_decode.restype = ctypes.c_int64
    lib.dl_lz4_decode.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)
    lib.dl_meshopt_decode_vertex.restype = ctypes.c_int
    lib.dl_meshopt_decode_vertex.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64)
    lib.dl_meshopt_decode_index.restype = ctypes.c_int
    lib.dl_meshopt_decode_index.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64)
    if hasattr(lib, "dl_encode_bc"):
        lib.dl_encode_bc.restype = ctypes.c_int64
        lib.dl_encode_bc.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    _lib = lib
    return lib


def available() -> bool:
    return _load() is not None


def _pointer(buffer):
    """The address of a bytes, bytearray, memoryview or numpy buffer, and an object keeping it alive."""
    if isinstance(buffer, np.ndarray):
        return buffer.ctypes.data, buffer
    array = np.frombuffer(buffer, dtype=np.uint8)
    return array.ctypes.data, array


def lz4_decode(src, dst: bytearray, dst_offset: int = 0, dst_end: int | None = None) -> int:
    """Decodes one LZ4 block into dst from dst_offset (matches may reach back before it). Returns bytes written."""
    end = len(dst) if dst_end is None else dst_end
    lib = _load()
    if lib is not None:
        src_ptr, _keep_src = _pointer(src)
        dst_array = np.frombuffer(dst, dtype=np.uint8)
        written = lib.dl_lz4_decode(src_ptr, len(src), dst_array.ctypes.data, end, dst_offset)
        if written < 0:
            raise ValueError("LZ4 block is malformed")
        return written
    return _lz4_python(bytes(src), dst, dst_offset, end)


def _lz4_python(src: bytes, dst: bytearray, op: int, end: int) -> int:
    start = op
    ip = 0
    n = len(src)
    while ip < n:
        token = src[ip]
        ip += 1
        literal = token >> 4
        if literal == 15:
            while True:
                v = src[ip]
                ip += 1
                literal += v
                if v != 255:
                    break
        if literal > n - ip or literal > end - op:
            raise ValueError("LZ4 literal run is out of bounds")
        dst[op:op + literal] = src[ip:ip + literal]
        ip += literal
        op += literal
        if ip == n:
            break
        offset = src[ip] | (src[ip + 1] << 8)
        ip += 2
        match = token & 15
        if match == 15:
            while True:
                v = src[ip]
                ip += 1
                match += v
                if v != 255:
                    break
        match += 4
        source = op - offset
        if offset == 0 or source < 0 or match > end - op:
            raise ValueError("LZ4 match is out of bounds")
        if offset >= match:
            dst[op:op + match] = dst[source:source + match]
        else:
            for i in range(match):
                dst[op + i] = dst[source + i]
        op += match
    return op - start


def meshopt_decode_vertex(count: int, size: int, src) -> bytes:
    lib = _load()
    if lib is not None:
        out = np.empty(count * size, dtype=np.uint8)
        src_ptr, _keep = _pointer(src)
        result = lib.dl_meshopt_decode_vertex(out.ctypes.data, count, size, src_ptr, len(src))
        if result != 0:
            raise ValueError(f"meshopt vertex buffer is malformed ({result})")
        return out.tobytes()
    from . import meshopt
    return meshopt.decode_vertex(count, size, bytes(src))


def meshopt_decode_index(count: int, size: int, src) -> bytes:
    lib = _load()
    if lib is not None:
        out = np.empty(count * size, dtype=np.uint8)
        src_ptr, _keep = _pointer(src)
        result = lib.dl_meshopt_decode_index(out.ctypes.data, count, size, src_ptr, len(src))
        if result != 0:
            raise ValueError(f"meshopt index buffer is malformed ({result})")
        return out.tobytes()
    from . import meshopt
    return meshopt.decode_index(count, size, bytes(src))


def can_encode_bc() -> bool:
    lib = _load()
    return lib is not None and hasattr(lib, "dl_encode_bc")


def encode_bc(rgba: np.ndarray, with_alpha: bool) -> bytes:
    """RGBA8 (h, w, 4), top row first, sides multiples of 4 -> BC3 (with alpha) or BC1 blocks."""
    lib = _load()
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    out = np.empty((h // 4) * (w // 4) * (16 if with_alpha else 8), np.uint8)
    written = lib.dl_encode_bc(rgba.ctypes.data, w, h, 1 if with_alpha else 0, out.ctypes.data)
    return out[:written].tobytes()
