"""Deadlock textures as Blender images.

Block compressed textures are written to the cache folder as DDS files without decoding and loaded from there, so
Blender keeps them compressed on the GPU. Embedded PNG/JPEG/WebP images are written as they are, uncompressed
formats as PNG or EXR (Blender doesn't open those as DDS). Files are named after the texture and a checksum of its compiled data, so a game update brings new ones.
"""
import os
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import bpy

import numpy as np

from ..core import native
from ..core import texture as vtex
from ..core.content import compiled, normalize

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
# Uncompressed 8 bit formats, written as PNG.
_BYTE_FORMATS = {vtex.RGBA8888, vtex.BGRA8888, vtex.I8, vtex.R8_UNORM, vtex.A8, vtex.IA88}


class TextureCache:
    """Converts and loads a map's textures. With ``parallel``, conversions run on a thread pool while the import
    goes on (decoding, mip building and the DXT encoder release Python's lock) and the images, made at once
    pointing at their files, load when ``finish()`` has waited for them."""

    def __init__(self, content, cache_dir: str, max_size: int = 0, compact: bool = True, parallel: bool = False):
        self.content = content
        self._pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) if parallel else None
        self._pending: dict[str, object] = {}     # file -> its conversion
        self._deferred: list[tuple] = []          # (image, conversion, texture path)
        self.compact = compact and native.can_encode_bc()
        self.folder = os.path.join(cache_dir, "textures")
        if os.name == "nt" and len(os.path.abspath(self.folder)) > 180:
            # Blender can't open image files past Windows' 260 character limit, so a deep cache moves somewhere short.
            self.folder = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "DeadlockBridge", "textures")
        os.makedirs(self.folder, exist_ok=True)
        self.max_size = max_size
        self._images: dict[tuple, bpy.types.Image | None] = {}
        self.missing: set[str] = set()
        self.failed: dict[str, str] = {}

    def compiled_texture(self, path: str):
        data = self.content.try_read(compiled(path))
        if data is None:
            self.missing.add(normalize(path))
            return None, None
        return vtex.CompiledTexture(data), data

    def image(self, path: str | None, color: bool = False, image: int = 0, full_size: bool = False):
        """The texture as a Blender image (colour data read as sRGB), or None if it's missing or unreadable."""
        if not path:
            return None
        key = (normalize(path), color, image, full_size)
        if key in self._images:
            return self._images[key]
        try:
            result = self._load(path, color, image, full_size)
        except Exception as error:  # a broken or unusual texture must not stop a map import
            self.failed[normalize(path)] = str(error)
            result = None
        self._images[key] = result
        return result

    def file_for(self, path: str, data: bytes, suffix: str) -> str:
        # Short names: Windows paths stop at 260 characters and the cache may sit deep in a user folder.
        path = normalize(path)
        stem = _SAFE.sub("_", path.rsplit("/", 1)[-1].rsplit(".", 1)[0])[:28]
        return os.path.join(self.folder, f"{stem}_{zlib.crc32(path.encode()):08x}{zlib.crc32(data):08x}{suffix}")

    def _load(self, path: str, color: bool, image: int, full_size: bool):
        tex, data = self.compiled_texture(path)
        if tex is None:
            return None

        mip = 0
        if self.max_size and not full_size and not tex.is_raw_image:
            while mip + 1 < tex.mip_count and max(tex.width >> mip, tex.height >> mip) > self.max_size:
                mip += 1

        convert = None
        if tex.is_raw_image:
            raw = bytes(tex.read_raw_image())
            suffix = ".png" if raw[:4] == b"\x89PNG" else ".webp" if raw[8:12] == b"WEBP" else ".jpg"
            filepath = self.file_for(path, data, suffix)
            if not os.path.isfile(filepath):
                _write(filepath, raw)
        elif tex.format in (vtex.DXT1, vtex.DXT5):
            # With every mip level, so Blender keeps them compressed on the GPU.
            tag = f"_m{mip}" + (f"_i{image}" if image else "")
            filepath = self.file_for(path, data, tag + "_mips.dds")
            convert = lambda: _write(filepath, tex.to_dds(mip, image, tex.mip_count - mip))
        elif self.compact and tex.format in (vtex.BC7, vtex.ATI1N, vtex.ATI2N) and not full_size:
            tag = f"_m{mip}" + (f"_i{image}" if image else "")
            filepath = self.file_for(path, data, tag + "_dxt.dds")
            convert = lambda: _write(filepath, self._transcode(tex, mip, image))
        elif not tex.block_compressed and tex.format in vtex.UNCOMPRESSED:
            # Blender doesn't open uncompressed DX10 DDS files (the BRDF lookup is RGBA16F): 8 bit formats as PNG,
            # the rest as EXR.
            tag = f"_m{mip}" + (f"_i{image}" if image else "")
            byte = tex.format in _BYTE_FORMATS
            filepath = self.file_for(path, data, tag + (".png" if byte else ".exr"))
            pixel_type = "uint8" if byte else \
                "float" if tex.format in (vtex.R32F, vtex.RG3232F, vtex.RGB323232F, vtex.RGBA32323232F) else "half"
            convert = lambda: _write_image(filepath, tex.decode_rgba(mip, image), pixel_type)
        elif tex.dds_supported():
            tag = f"_m{mip}" + (f"_i{image}" if image else "")
            filepath = self.file_for(path, data, tag + ".dds")
            convert = lambda: _write(filepath, tex.to_dds(mip, image))
        else:
            self.failed[normalize(path)] = f"unsupported texture format {tex.format}"
            return None

        if convert is not None and not os.path.isfile(filepath):
            if self._pool is None:
                convert()
            else:
                # Converted in the background; the image points at the file and loads once it's written.
                conversion = self._pending.get(filepath)
                if conversion is None:
                    conversion = self._pending[filepath] = self._pool.submit(convert)
                img = bpy.data.images.new(os.path.basename(normalize(path)).rsplit(".", 1)[0][:60], 1, 1)
                img.filepath = filepath
                img.source = "FILE"
                self._deferred.append((img, conversion, normalize(path)))
                return self._finish_image(img, path, tex, color)

        img = bpy.data.images.load(filepath, check_existing=True)
        return self._finish_image(img, path, tex, color)

    def _finish_image(self, img, path, tex, color):
        img.name = os.path.basename(normalize(path)).rsplit(".", 1)[0][:60]
        img["dl_path"] = normalize(path)
        if tex.is_hdr:
            img.colorspace_settings.name = "Non-Color"
        else:
            img.colorspace_settings.name = "sRGB" if color else "Non-Color"
        img.alpha_mode = "CHANNEL_PACKED"
        return img

    def finish(self, progress=None):
        """Waits for the background conversions and loads their images (a failed one stays missing, reported).
        ``progress(done, total)`` is told as they come in."""
        if self._pool is None:
            return
        for index, (img, conversion, path) in enumerate(self._deferred):
            if progress is not None:
                progress(index, len(self._deferred))
            try:
                conversion.result()
            except Exception as error:
                self.failed[path] = str(error)
                continue
            img.reload()
        self._deferred.clear()
        self._pending.clear()
        self._pool.shutdown(wait=True)
        self._pool = None

    def _transcode(self, tex, mip: int, image: int) -> bytes:
        """A BC7/BC4/BC5 texture as DXT1 (opaque) or DXT5 with a full mip chain: Blender keeps only those compressed
        on the GPU, where the original formats would be uploaded at four to eight times their size."""
        import OpenImageIO as oiio
        tmp = os.path.join(self.folder, f"_transcode_{os.getpid()}_{threading.get_ident()}.dds")
        _write(tmp, tex.to_dds(mip, image))
        inp = oiio.ImageInput.open(tmp)
        pixels = inp.read_image("uint8")
        inp.close()
        os.remove(tmp)
        h, w, channels = pixels.shape
        rgba = np.full((h, w, 4), 255, np.uint8)
        if channels == 1:
            rgba[..., :3] = pixels[..., :1]
        else:
            rgba[..., :min(channels, 4)] = pixels[..., :min(channels, 4)]
        with_alpha = channels == 4 and bool((rgba[..., 3] < 250).any())

        levels = []
        level = rgba
        while True:
            lh, lw = level.shape[:2]
            ph, pw = (lh + 3) // 4 * 4, (lw + 3) // 4 * 4
            padded = level if (ph, pw) == (lh, lw) else np.pad(level, ((0, ph - lh), (0, pw - lw), (0, 0)), mode="edge")
            levels.append(native.encode_bc(padded, with_alpha))
            if lh == 1 and lw == 1:
                break
            # Box filter to the next level (odd sides keep their last row or column).
            nh, nw = max(lh // 2, 1), max(lw // 2, 1)
            f = level.astype(np.float32)
            if lh > 1:
                f = (f[0:nh * 2:2] + f[1:nh * 2:2]) * 0.5
            if lw > 1:
                f = (f[:, 0:nw * 2:2] + f[:, 1:nw * 2:2]) * 0.5
            level = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        return vtex._dds_header(w, h, len(levels), 77 if with_alpha else 71) + b"".join(levels)


def _long(path: str) -> str:
    """Windows' extended path form, so a cache deep in a user folder isn't stopped at 260 characters."""
    prefix = "\\\\?\\"
    if os.name == "nt" and len(path) > 240 and not path.startswith(prefix):
        return prefix + os.path.abspath(path)
    return path


def _write(path: str, data: bytes):
    tmp = _long(path + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, _long(path))


def _write_image(path: str, pixels: np.ndarray, pixel_type: str):
    """Float (h, w, 4) pixels, top row first, as a PNG or EXR file."""
    import OpenImageIO as oiio
    root, ext = os.path.splitext(path)
    tmp = f"{root}_{os.getpid()}.tmp{ext}"
    h, w, channels = pixels.shape
    spec = oiio.ImageSpec(w, h, channels, pixel_type)
    if ext == ".exr":
        spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(tmp)
    if out is None or not out.open(tmp, spec):
        raise OSError(f"can't write {tmp}: {oiio.geterror()}")
    out.write_image(np.ascontiguousarray(pixels, np.float32))
    out.close()
    os.replace(tmp, path)


def texture_flags(content, path: str) -> int:
    data = content.try_read(compiled(path)) if path else None
    if data is None:
        return 0
    return vtex.CompiledTexture(data).flags
