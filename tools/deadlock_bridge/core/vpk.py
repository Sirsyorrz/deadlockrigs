"""Valve Pak (VPK) directories and the archive chunks they index."""
import os
import struct
import threading

_SIGNATURE = 0x55AA1234
_EMBEDDED = 0x7FFF


class VpkArchive:
    """A VPK directory (``*_dir.vpk``, or a single-file VPK such as a map's) and its archive chunks.

    Paths are lower case with forward slashes. Reads are positional and thread safe.
    """

    def __init__(self, path: str):
        self.path = path
        self._entries: dict[str, tuple[int, int, int, int, int]] = {}
        self._handles: dict[int, object] = {}
        self._lock = threading.Lock()

        base = path[:-8] if path.lower().endswith("_dir.vpk") else path[:-4]
        self._chunk_format = base + "_{:03d}.vpk"

        with open(path, "rb") as f:
            header = f.read(12)
            signature, version, tree_size = struct.unpack("<III", header)
            if signature != _SIGNATURE:
                raise ValueError(f"'{path}' is not a VPK")
            if version not in (1, 2):
                raise ValueError(f"'{path}' has unsupported VPK version {version}")
            if version == 2:
                f.read(16)
            self._tree = f.read(tree_size)
            self._embedded_offset = f.tell()

        self._read_tree()

    def _read_tree(self):
        tree = self._tree
        pos = 0
        entries = self._entries
        unpack = struct.Struct("<HHII").unpack_from

        def string():
            nonlocal pos
            end = tree.index(b"\0", pos)
            value = tree[pos:end].decode("utf-8", "replace")
            pos = end + 1
            return value

        while True:
            extension = string()
            if not extension:
                break
            suffix = "" if extension == " " else "." + extension
            while True:
                directory = string()
                if not directory:
                    break
                prefix = "" if directory == " " else directory.replace("\\", "/").strip("/") + "/"
                while True:
                    name = string()
                    if not name:
                        break
                    preload, archive, offset, length = unpack(tree, pos + 4)
                    pos += 18
                    entries[(prefix + name + suffix).lower()] = (archive, offset, length, pos, preload)
                    pos += preload

    @property
    def files(self):
        return self._entries.keys()

    def __contains__(self, path: str) -> bool:
        return path.lower() in self._entries

    def size(self, path: str) -> int:
        archive, offset, length, preload_offset, preload = self._entries[path.lower()]
        return length + preload

    def read(self, path: str) -> bytes:
        archive, offset, length, preload_offset, preload = self._entries[path.lower()]
        data = self._tree[preload_offset:preload_offset + preload]
        if length == 0:
            return data

        if archive == _EMBEDDED:
            offset += self._embedded_offset

        handle = self._handle(archive)
        if hasattr(os, "pread"):
            body = os.pread(handle.fileno(), length, offset)
        else:
            with self._lock:
                handle.seek(offset)
                body = handle.read(length)
        if len(body) != length:
            raise IOError(f"'{path}' is truncated in '{self.path}'")
        return data + body if data else body

    def _handle(self, archive: int):
        with self._lock:
            handle = self._handles.get(archive)
            if handle is None:
                path = self.path if archive == _EMBEDDED else self._chunk_format.format(archive)
                handle = open(path, "rb")
                self._handles[archive] = handle
            return handle

    def close(self):
        with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()
