"""Compiled Source 2 resources (files ending in ``_c``): a small header followed by typed blocks."""
import struct

from . import kv3


class ResourceFile:
    def __init__(self, data: bytes):
        if len(data) < 16:
            raise ValueError("resource is too small for its header")
        self.data = data
        header_version, self.version = struct.unpack_from("<HH", data, 4)
        if header_version != 12:
            raise ValueError(f"resource has unexpected header version {header_version}")

        position = 8 + struct.unpack_from("<I", data, 8)[0]
        count = struct.unpack_from("<I", data, 12)[0]
        self.blocks: list[tuple[str, int, int]] = []
        for _ in range(count):
            kind = data[position:position + 4].decode("ascii", "replace")
            offset, size = struct.unpack_from("<II", data, position + 4)
            self.blocks.append((kind, position + 4 + offset, size))
            position += 12

    def block(self, kind: str):
        for block in self.blocks:
            if block[0] == kind:
                return block
        return None

    def block_data(self, block) -> memoryview:
        if isinstance(block, int):
            block = self.blocks[block]
        _, offset, size = block
        return memoryview(self.data)[offset:offset + size]

    def keyvalues(self, kind) -> dict | None:
        """The KeyValues3 document in the first block of a type (or a block index), or None."""
        block = self.blocks[kind] if isinstance(kind, int) else self.block(kind)
        if block is None:
            return None
        return kv3.read(self.block_data(block))
