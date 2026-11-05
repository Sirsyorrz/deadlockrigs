"""meshoptimizer's vertex and index codecs (versions 0 and 1) in Python, for when the native library isn't there.

Ported from ValveResourceFormat's port of meshoptimizer (both MIT licensed). Slow; see native.py.
"""
import struct

_BITS_V0 = (0, 2, 4, 8)
_BITS_V1 = (0, 1, 2, 4, 8)


def _decode_bytes_group(data: bytes, pos: int, out: bytearray, at: int, bits: int) -> int:
    if bits == 0:
        out[at:at + 16] = bytes(16)
        return pos
    if bits == 8:
        out[at:at + 16] = data[pos:pos + 16]
        return pos + 16
    header_bytes = 16 * bits // 8
    extra = pos + header_bytes
    per_byte = 8 // bits
    sentinel = (1 << bits) - 1
    for i in range(16):
        shift = i % 8 if bits == 1 else 8 - bits * (i % per_byte + 1)
        value = (data[pos + i // per_byte] >> shift) & sentinel
        if value == sentinel:
            out[at + i] = data[extra]
            extra += 1
        else:
            out[at + i] = value
    return extra


def _decode_bytes(data: bytes, pos: int, out: bytearray, at: int, size: int, bits) -> int:
    header_size = (size // 16 + 3) // 4
    header = pos
    pos += header_size
    for i in range(0, size, 16):
        if len(data) - pos < 24:
            raise ValueError("meshopt byte group is truncated")
        group = i // 16
        index = (data[header + group // 4] >> (group % 4 * 2)) & 3
        pos = _decode_bytes_group(data, pos, out, at + i, bits[index])
    return pos


def decode_vertex(count: int, size: int, data: bytes) -> bytes:
    if data[0] & 0xF0 != 0xA0:
        raise ValueError("meshopt vertex header is invalid")
    version = data[0] & 0x0F
    tail_size = size + (0 if version == 0 else size // 4)
    tail_padded = max(tail_size, 32 if version == 0 else 24)
    end = len(data)
    last = bytearray(data[end - tail_size:end - tail_size + size])
    channels = None if version == 0 else data[end - tail_size + size:end - tail_size + size + size // 4]

    block_size = min((8192 // size) & ~15, 256)
    result = bytearray(count * size)
    scratch = bytearray(256 * 4 + 16)
    pos = 1

    for offset in range(0, count, block_size):
        n = min(block_size, count - offset)
        aligned = (n + 15) & ~15
        control_size = 0 if version == 0 else size // 4
        control = data[pos:pos + control_size]
        pos += control_size
        transposed = bytearray(n * size)

        for k in range(0, size, 4):
            control_byte = 0 if version == 0 else control[k // 4]
            buf = bytearray(4 * n + 16)
            for j in range(4):
                mode = (control_byte >> (j * 2)) & 3
                if mode == 3:
                    buf[j * n:j * n + n] = data[pos:pos + n]
                    pos += n
                elif mode == 2:
                    pass
                else:
                    bits = _BITS_V0 if version == 0 else _BITS_V1[mode:]
                    tmp = bytearray(aligned)
                    pos = _decode_bytes(data, pos, tmp, 0, aligned, bits)
                    buf[j * n:j * n + n] = tmp[:n]

            channel = 0 if version == 0 else channels[k // 4]
            width = (1, 2, 4)[channel & 3] if channel & 3 < 3 else None
            if width is None:
                raise ValueError("meshopt channel is invalid")
            rotation = (32 - (channel >> 4)) & 31

            for kk in range(0, 4, width):
                previous = int.from_bytes(last[k + kk:k + kk + width], "little")
                src = buf[kk * n:]
                for i in range(n):
                    v = 0
                    for b in range(width):
                        v |= src[i + n * b] << (8 * b)
                    if width == 1:
                        v = ((-(v & 1) ^ (v >> 1)) + previous) & 0xFF
                    elif width == 2:
                        v = ((-(v & 1) ^ (v >> 1)) + previous) & 0xFFFF
                    else:
                        if rotation:
                            v = ((v << rotation) | (v >> (32 - rotation))) & 0xFFFFFFFF
                        v ^= previous
                    transposed[i * size + k + kk:i * size + k + kk + width] = v.to_bytes(width, "little")
                    previous = v

        result[offset * size:(offset + n) * size] = transposed
        last = bytearray(transposed[(n - 1) * size:n * size])

    if len(data) - pos != tail_padded:
        raise ValueError("meshopt vertex buffer has an unexpected tail")
    return bytes(result)


def _vbyte(data: bytes, pos: int):
    lead = data[pos]
    pos += 1
    if lead < 128:
        return lead, pos
    result = lead & 127
    shift = 7
    for _ in range(4):
        group = data[pos]
        pos += 1
        result |= (group & 127) << shift
        shift += 7
        if group < 128:
            break
    return result, pos


def decode_index(count: int, size: int, buffer: bytes) -> bytes:
    if buffer[0] & 0xF0 != 0xE0:
        raise ValueError("meshopt index header is invalid")
    version = buffer[0] & 0x0F
    data_offset = 1 + count // 3
    codes = buffer[1:data_offset]
    data = buffer[data_offset:]
    safe_end = len(data) - 16
    aux_table = data[safe_end:]
    fec_max = 13 if version >= 1 else 15

    edges = [(0, 0)] * 16
    fifo = [0] * 16
    eo = vo = 0
    nxt = last = 0
    pos = 0
    out = []

    def index(last_value, p):
        v, p = _vbyte(data, p)
        delta = ((v >> 1) ^ -(v & 1)) & 0xFFFFFFFF
        return (last_value + delta) & 0xFFFFFFFF, p

    for code in codes:
        if code < 0xF0:
            fe = code >> 4
            a, b = edges[(eo - 1 - fe) & 15]
            fec = code & 15
            if fec < fec_max:
                c = nxt if fec == 0 else fifo[(vo - 1 - fec) & 15]
                if fec == 0:
                    nxt += 1
                fifo[vo] = c
                vo = (vo + (1 if fec == 0 else 0)) & 15
            else:
                if fec != 15:
                    c = (last + fec * 2 - 27) & 0xFFFFFFFF
                else:
                    c, pos = index(last, pos)
                last = c
                fifo[vo] = c
                vo = (vo + 1) & 15
            edges[eo] = (c, b)
            eo = (eo + 1) & 15
            edges[eo] = (a, c)
            eo = (eo + 1) & 15
            out += (a, b, c)
        elif code < 0xFE:
            aux = aux_table[code & 15]
            feb, fec = aux >> 4, aux & 15
            a = nxt
            nxt += 1
            b = nxt if feb == 0 else fifo[(vo - feb) & 15]
            if feb == 0:
                nxt += 1
            c = nxt if fec == 0 else fifo[(vo - fec) & 15]
            if fec == 0:
                nxt += 1
            out += (a, b, c)
            fifo[vo] = a
            vo = (vo + 1) & 15
            fifo[vo] = b
            vo = (vo + (1 if feb == 0 else 0)) & 15
            fifo[vo] = c
            vo = (vo + (1 if fec == 0 else 0)) & 15
            for e in ((b, a), (c, b), (a, c)):
                edges[eo] = e
                eo = (eo + 1) & 15
        else:
            aux = data[pos]
            pos += 1
            fea = 0 if code == 0xFE else 15
            feb, fec = aux >> 4, aux & 15
            if aux == 0:
                nxt = 0
            if fea == 0:
                a = nxt
                nxt += 1
            else:
                a = 0
            if feb == 0:
                b = nxt
                nxt += 1
            else:
                b = fifo[(vo - feb) & 15]
            if fec == 0:
                c = nxt
                nxt += 1
            else:
                c = fifo[(vo - fec) & 15]
            if fea == 15:
                a, pos = index(last, pos)
                last = a
            if feb == 15:
                b, pos = index(last, pos)
                last = b
            if fec == 15:
                c, pos = index(last, pos)
                last = c
            out += (a, b, c)
            fifo[vo] = a
            vo = (vo + 1) & 15
            fifo[vo] = b
            vo = (vo + (1 if feb in (0, 15) else 0)) & 15
            fifo[vo] = c
            vo = (vo + (1 if fec in (0, 15) else 0)) & 15
            for e in ((b, a), (c, b), (a, c)):
                edges[eo] = e
                eo = (eo + 1) & 15

    if pos != safe_end:
        raise ValueError("meshopt index data is malformed")
    fmt = "<%d%s" % (len(out), "H" if size == 2 else "I")
    return struct.pack(fmt, *[v & (0xFFFF if size == 2 else 0xFFFFFFFF) for v in out])
