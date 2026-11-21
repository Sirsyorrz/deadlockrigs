"""Read and write compiled Source 2 resources (vmdl_c, vmat_c, vtex_c) without Valve's tools: the resource block
container, KV3 v5 documents, resource reference lists and BGRA8888 textures. Pure Python and numpy, so it runs in
Blender's bundled Python. Ported from deadlock-forge's model swap (kv3.py, resfile.py)."""

import struct
from dataclasses import dataclass

import numpy as np

NULL, BOOLEAN, INT64, UINT64, DOUBLE, STRING, BLOB, ARRAY, OBJECT, ARRAY_TYPED = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
INT32, UINT32, TRUE, FALSE, INT64_ZERO, INT64_ONE, DOUBLE_ZERO, DOUBLE_ONE = 11, 12, 13, 14, 15, 16, 17, 18
FLOAT, INT16, UINT16, INT8, UINT8, ARRAY_BYTE_LEN, ARRAY_AUX = 19, 20, 21, 22, 23, 24, 25
UINT32_ZERO, UINT32_ONE = 26, 27
TYPED_ARRAYS = (ARRAY_TYPED, ARRAY_BYTE_LEN, ARRAY_AUX)
MAGIC5 = 0x4B563305
TRAILER = 0xFFEEDD00

LANE_FMT = {
    BOOLEAN: (1, '<B'), INT8: (1, '<b'), UINT8: (1, '<B'),
    INT16: (2, '<h'), UINT16: (2, '<H'),
    INT32: (4, '<i'), UINT32: (4, '<I'), FLOAT: (4, '<f'),
    INT64: (8, '<q'), UINT64: (8, '<Q'), DOUBLE: (8, '<d'),
}
CONST_VALUE = {NULL: None, TRUE: True, FALSE: False, INT64_ZERO: 0, INT64_ONE: 1, DOUBLE_ZERO: 0.0, DOUBLE_ONE: 1.0,
               UINT32_ZERO: 0, UINT32_ONE: 1}


def lz4_block(src, size):
    """Decompress one raw LZ4 block (no frame header) of known output size."""
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        length = token >> 4
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        out += src[i:i + length]
        i += length
        if i >= n:
            break
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        length = token & 15
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        length += 4
        start = len(out) - offset
        if offset >= length:
            out += out[start:start + length]
        else:
            for k in range(length):
                out.append(out[start + k])
    if len(out) != size:
        raise ValueError(f'lz4 block gave {len(out)} bytes, expected {size}')
    return bytes(out)


def _zstd_decompress(data, size):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)


@dataclass
class Node:
    t: int
    v: object = None
    flag: int = 0
    et: int = 0  # element type of typed arrays
    eflag: int = 0

    def __getitem__(self, key):
        if self.t == OBJECT:
            for k, n in self.v:
                if k == key:
                    return n
            raise KeyError(key)
        return self.v[key]

    def __setitem__(self, key, node):
        for i, (k, _) in enumerate(self.v):
            if k == key:
                self.v[i] = (k, node)
                return
        self.v.append((key, node))

    def __delitem__(self, key):
        self.v = [(k, n) for k, n in self.v if k != key]

    def __contains__(self, key):
        return any(k == key for k, _ in self.v)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def py(self):
        if self.t == OBJECT:
            return {k: n.py() for k, n in self.v}
        if self.t == ARRAY or self.t in TYPED_ARRAYS:
            return [n.py() for n in self.v]
        return self.v


def integer(value):
    """An int node: the zero and one constants where Valve's files use them, else int32."""
    return Node(INT64_ZERO if value == 0 else INT64_ONE if value == 1 else INT32, value)


def boolean(value):
    return Node(TRUE if value else FALSE, bool(value))


@dataclass
class Doc:
    fmt: bytes
    root: Node


class _Lanes:
    def __init__(self, b1=b'', b2=b'', b4=b'', b8=b''):
        self.buf = {1: b1, 2: b2, 4: b4, 8: b8}
        self.pos = {1: 0, 2: 0, 4: 0, 8: 0}

    def read(self, t):
        size, fmt = LANE_FMT[t]
        p = self.pos[size]
        self.pos[size] = p + size
        return struct.unpack_from(fmt, self.buf[size], p)[0]

    def i32(self):
        p = self.pos[4]
        self.pos[4] = p + 4
        return struct.unpack_from('<i', self.buf[4], p)[0]

    def u8(self):
        p = self.pos[1]
        self.pos[1] = p + 1
        return self.buf[1][p]


def _align(o, a):
    return (o + a - 1) & ~(a - 1)


def _split_lanes(buf, off, c1, c2, c4, c8):
    b1 = buf[off:off + c1]
    off += c1
    if c2:
        off = _align(off, 2)
    b2 = buf[off:off + c2 * 2]
    off += c2 * 2
    if c4:
        off = _align(off, 4)
    b4 = buf[off:off + c4 * 4]
    off += c4 * 4
    if c8:
        off = _align(off, 8)
    b8 = buf[off:off + c8 * 8]
    off += c8 * 8
    return b1, b2, b4, b8, off


def kv3_decode(data):
    magic, = struct.unpack_from('<I', data, 0)
    if magic != MAGIC5:
        raise ValueError(f'not KV3 v5: {magic:#x}')
    fmt = data[4:20]
    (comp, dict_id, _frame, c1, c4, c8, ctypes, _cobj, _carr, _unc_total, comp_total,
     nblobs, blob_bytes, c2, _blk_sizes) = struct.unpack_from('<IHHiiiiHHiiiiii', data, 20)
    (u1, k1, u2, k2, c1b, c2b, c4b, c8b, _nodes, cobjb, _carrb, _elems) = struct.unpack_from('<12i', data, 72)
    if dict_id:
        raise ValueError('dictionary-compressed KV3 is not supported')
    pos = 120
    if comp == 0:
        buf1 = data[pos:pos + u1]
        pos += u1
        buf2 = data[pos:pos + u2]
        pos += u2
    elif comp == 1:
        buf1 = lz4_block(data[pos:pos + k1], u1)
        pos += k1
        buf2 = lz4_block(data[pos:pos + k2], u2)
        pos += k2
    elif comp == 2:
        buf1 = _zstd_decompress(data[pos:pos + k1], u1)
        pos += k1
        buf2 = _zstd_decompress(data[pos:pos + k2], u2)
        pos += k2
    else:
        raise ValueError(f'unknown KV3 compression {comp}')

    a1, a2, a4, a8, _ = _split_lanes(buf1, 0, c1, c2, c4, c8)
    nstr, = struct.unpack_from('<i', a4, 0)
    strings, sp = [], 0
    for _ in range(nstr):
        e = a1.index(b'\x00', sp)
        strings.append(a1[sp:e].decode('utf-8'))
        sp = e + 1
    aux = _Lanes(a1[sp:], a2, a4[4:], a8)

    objlens = buf2[:cobjb * 4]
    m1, m2, m4, m8, off = _split_lanes(buf2, cobjb * 4, c1b, c2b, c4b, c8b)
    types = buf2[off:off + ctypes]
    off += ctypes
    main = _Lanes(m1, m2, m4, m8)

    blob_lens = []
    if nblobs:
        blob_lens = list(struct.unpack_from(f'<{nblobs}i', buf2, off))
        off += nblobs * 4
    if struct.unpack_from('<I', buf2, off)[0] != TRAILER:
        raise ValueError('KV3 trailer missing')
    blobs = b''
    if nblobs and comp == 1:
        raise ValueError('lz4 KV3 with binary blobs is not supported')
    elif nblobs:
        blobs = data[pos:] if comp == 0 else _zstd_decompress(data[pos:pos + comp_total - k1 - k2], blob_bytes)

    st = {'t': 0, 'o': 0, 'b': 0, 'bl': 0}

    def rtype():
        b = types[st['t']]
        st['t'] += 1
        flag = 0
        if b & 0x80:
            flag = types[st['t']]
            st['t'] += 1
        if b & 0x40:
            st['t'] += 1
        return b & 0x3F, flag

    def rstr(i):
        return strings[i] if 0 <= i < len(strings) else ''

    def value(t, flag, lane):
        if t in CONST_VALUE:
            return Node(t, CONST_VALUE[t], flag)
        if t in LANE_FMT:
            return Node(t, lane.read(t), flag)
        if t == STRING:
            return Node(t, rstr(main.i32()), flag)
        if t == BLOB:
            n = blob_lens[st['bl']]
            st['bl'] += 1
            v = blobs[st['b']:st['b'] + n]
            st['b'] += n
            return Node(t, v, flag)
        if t == ARRAY:
            n = main.i32()
            items = []
            for _ in range(n):
                et, ef = rtype()
                items.append(value(et, ef, main))
            return Node(t, items, flag)
        if t in TYPED_ARRAYS:
            n = main.i32() if t == ARRAY_TYPED else main.u8()
            et, ef = rtype()
            el = aux if t == ARRAY_AUX else main
            return Node(t, [value(et, ef, el) for _ in range(n)], flag, et, ef)
        if t == OBJECT:
            n, = struct.unpack_from('<i', objlens, st['o'] * 4)
            st['o'] += 1
            members = []
            for _ in range(n):
                name = rstr(main.i32())
                mt, mf = rtype()
                members.append((name, value(mt, mf, main)))
            return Node(t, members, flag)
        raise ValueError(f'unknown KV3 type {t}')

    rt, rf = rtype()
    return Doc(fmt, value(rt, rf, main))


class _W:
    def __init__(self):
        self.lanes = {s: bytearray() for s in (1, 2, 4, 8)}

    def put(self, t, v):
        size, fmt = LANE_FMT[t]
        self.lanes[size] += struct.pack(fmt, v)


def kv3_encode(doc):
    """Uncompressed KV3 v5: the engine reads it like its own lz4 or zstd documents."""
    strings = {}
    main, aux = _W(), _W()
    types = bytearray()
    objlens = bytearray()
    blobs = bytearray()
    blob_lens = []
    cnt = {'nodes': 0, 'obj': 0, 'arr': 0, 'elems': 0, 'arr_all': 0}

    def sid(s):
        if not s:
            return -1
        if s not in strings:
            strings[s] = len(strings)
        return strings[s]

    def wtype(t, flag):
        if flag:
            types.append(t | 0x80)
            types.append(flag)
        else:
            types.append(t)

    def body(n, lane):
        t = n.t
        if t in CONST_VALUE:
            return
        if t in LANE_FMT:
            lane.put(t, n.v)
        elif t == STRING:
            main.lanes[4] += struct.pack('<i', sid(n.v))
        elif t == BLOB:
            blob_lens.append(len(n.v))
            blobs.extend(n.v)
        elif t == ARRAY:
            cnt['arr_all'] += 1
            cnt['arr'] += 1
            cnt['elems'] += max(len(n.v), 1)
            main.lanes[4] += struct.pack('<i', len(n.v))
            for it in n.v:
                cnt['nodes'] += 1
                wtype(it.t, it.flag)
                body(it, main)
        elif t in TYPED_ARRAYS:
            cnt['arr_all'] += 1
            # Valve's allocation hints skip aux-buffer arrays and typed elements.
            if t != ARRAY_AUX:
                cnt['arr'] += 1
                cnt['elems'] += max(len(n.v), 1)
            if t == ARRAY_TYPED:
                main.lanes[4] += struct.pack('<i', len(n.v))
            else:
                if len(n.v) >= 256:
                    raise ValueError('short typed array over 255 items')
                main.lanes[1].append(len(n.v))
            wtype(n.et, n.eflag)
            el = aux if t == ARRAY_AUX else main
            for it in n.v:
                body(it, el)
        elif t == OBJECT:
            cnt['obj'] += 1
            objlens.extend(struct.pack('<i', len(n.v)))
            for name, it in n.v:
                cnt['nodes'] += 1
                main.lanes[4] += struct.pack('<i', sid(name))
                wtype(it.t, it.flag)
                body(it, main)
        else:
            raise ValueError(f'cannot encode KV3 type {t}')

    cnt['nodes'] += 1
    wtype(doc.root.t, doc.root.flag)
    body(doc.root, main)

    def pad(b, a):
        while len(b) % a:
            b.append(0)

    b1 = bytearray(b''.join(s.encode('utf-8') + b'\x00' for s in strings)) + aux.lanes[1]
    c1 = len(b1)
    if aux.lanes[2]:
        pad(b1, 2)
        b1 += aux.lanes[2]
    pad(b1, 4)
    b1 += struct.pack('<i', len(strings)) + aux.lanes[4]
    if aux.lanes[8]:
        pad(b1, 8)
        b1 += aux.lanes[8]

    b2 = bytearray(objlens)
    for size in (1, 2, 4, 8):
        if main.lanes[size]:
            pad(b2, size)
            b2 += main.lanes[size]
    b2 += types
    for n in blob_lens:
        b2 += struct.pack('<i', n)
    b2 += struct.pack('<I', TRAILER)

    hdr = struct.pack('<I', MAGIC5) + doc.fmt
    hdr += struct.pack(
        '<IHHiiiiHHiiiiii', 0, 0, 0, c1, 1 + len(aux.lanes[4]) // 4, len(aux.lanes[8]) // 8, len(types),
        min(cnt['obj'], 0xFFFF), min(cnt['arr_all'], 0xFFFF), len(b1) + len(b2), 0,
        len(blob_lens), len(blobs), len(aux.lanes[2]) // 2, 0)
    hdr += struct.pack(
        '<12i', len(b1), 0, len(b2), 0,
        len(main.lanes[1]), len(main.lanes[2]) // 2, len(main.lanes[4]) // 4, len(main.lanes[8]) // 8,
        cnt['nodes'], cnt['obj'], cnt['arr'], cnt['elems'])
    out = hdr + bytes(b1) + bytes(b2)
    if blob_lens:
        out += bytes(blobs) + struct.pack('<I', TRAILER)
    return out


@dataclass
class Block:
    kind: str
    data: bytes


@dataclass
class Resource:
    header_version: int
    version: int
    blocks: list
    tail: bytes = b''  # data after the blocks (a texture's pixels)

    def index(self, kind, nth=0):
        return [i for i, b in enumerate(self.blocks) if b.kind == kind][nth]

    def block(self, kind):
        return self.blocks[self.index(kind)]

    def kv3(self, kind):
        return kv3_decode(self.block(kind).data)

    def set_kv3(self, kind, doc):
        self.block(kind).data = kv3_encode(doc)

    def drop(self, kind):
        self.blocks = [b for b in self.blocks if b.kind != kind]


def read_resource(raw):
    size, hv, ver, bo, bc = struct.unpack_from('<IHHII', raw, 0)
    base = 8 + bo
    blocks = []
    for i in range(bc):
        e = base + i * 12
        kind = raw[e:e + 4].decode('ascii')
        rel, length = struct.unpack_from('<II', raw, e + 4)
        start = e + 4 + rel
        blocks.append(Block(kind, raw[start:start + length]))
    return Resource(hv, ver, blocks, raw[size:])


def write_resource(res):
    """The header's size field is where the blocks end; a texture's pixels follow it."""
    n = len(res.blocks)
    out = bytearray(struct.pack('<IHHII', 0, res.header_version, res.version, 8, n))
    table = len(out)
    out += bytes(12 * n)
    for i, b in enumerate(res.blocks):
        while len(out) % 16:
            out.append(0)
        start = len(out)
        out += b.data
        e = table + i * 12
        out[e:e + 4] = b.kind.encode('ascii')
        struct.pack_into('<II', out, e + 4, start - (e + 4), len(b.data))
    struct.pack_into('<I', out, 0, len(out))
    return bytes(out) + res.tail


def murmur64(data, seed=0xEDABCDEF):
    """MurmurHash64B: a resource's id, of its path as written in references ('materials/x.vmat')."""
    m = 0x5bd1e995
    mask = 0xffffffff

    def mix(k):
        k = (k * m) & mask
        k ^= k >> 24
        return (k * m) & mask

    n = len(data)
    h1, h2, i = (seed ^ n) & mask, 0, 0
    while n - i >= 8:
        k1, k2 = struct.unpack_from('<II', data, i)
        i += 8
        h1 = ((h1 * m) & mask) ^ mix(k1)
        h2 = ((h2 * m) & mask) ^ mix(k2)
    if n - i >= 4:
        k1, = struct.unpack_from('<I', data, i)
        i += 4
        h1 = ((h1 * m) & mask) ^ mix(k1)
    rem = n - i
    if rem == 3:
        h2 ^= data[i + 2] << 16
    if rem >= 2:
        h2 ^= data[i + 1] << 8
    if rem >= 1:
        h2 ^= data[i]
        h2 = (h2 * m) & mask
    h1 ^= h2 >> 18
    h1 = (h1 * m) & mask
    h2 ^= h1 >> 22
    h2 = (h2 * m) & mask
    h1 ^= h2 >> 17
    h1 = (h1 * m) & mask
    h2 ^= h1 >> 19
    h2 = (h2 * m) & mask
    return (h1 << 32) | h2


def reference_list(paths):
    """An RERL block: the resources a file names, which the engine loads before it."""
    paths = sorted(set(paths))
    out = bytearray(struct.pack('<II', 8, len(paths)) if paths else struct.pack('<II', 0, 0))
    strings = bytearray()
    first = 8 + 16 * len(paths)
    for i, path in enumerate(paths):
        entry = 8 + 16 * i
        out += struct.pack('<QII', murmur64(path.encode()), first + len(strings) - (entry + 8), 0)
        strings += path.encode() + b'\0'
    return bytes(out + strings)


VTEX_BGRA8888 = 28


def _half_size(px):
    h, w = px.shape[:2]
    if h > 1 and w > 1:
        px = px[:h // 2 * 2, :w // 2 * 2]
        return (px[0::2, 0::2] + px[1::2, 0::2] + px[0::2, 1::2] + px[1::2, 1::2]) / 4
    if h > 1:
        px = px[:h // 2 * 2]
        return (px[0::2] + px[1::2]) / 2
    px = px[:, :w // 2 * 2]
    return (px[:, 0::2] + px[:, 1::2]) / 2


def texture(rgba):
    """A vtex_c of an (h, w, 4) uint8 RGBA image, top row first: uncompressed BGRA8888 with a full mip chain. No
    compile-input (RED2) block: the engine doesn't need one."""
    rgba = np.asarray(rgba, np.uint8)
    h, w = rgba.shape[:2]
    mips = [rgba.astype(np.float32)]
    while mips[-1].shape[0] > 1 or mips[-1].shape[1] > 1:
        mips.append(_half_size(mips[-1]))
    pixels = b''.join(np.ascontiguousarray(np.clip(np.round(m), 0, 255).astype(np.uint8)[..., [2, 1, 0, 3]]).tobytes()
                      for m in reversed(mips))  # smallest mip first
    reflectivity = (rgba[..., :3].reshape(-1, 3).mean(0) / 255).tolist()
    data = struct.pack('<HH4fHHHBBIII', 1, 0, *reflectivity, 0.0, w, h, 1, VTEX_BGRA8888, len(mips), 0, 0, 0)
    return write_resource(Resource(12, 1, [Block('DATA', data)], pixels))
