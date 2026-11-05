"""Light probe volumes (env_combined_light_probe_volume), sampled on the CPU.

The game looks the probes' ambient cubes up per pixel. Placed objects here get one ambient cube each, sampled at their
lighting origin through the same volume blending as Deadlock's deferred pass (volumes front to back by indoor level
and array index, faded at their edges, the sky light filling what they leave).
"""
import math

import numpy as np

from . import texture as vtex


def rotation_matrix(angles) -> np.ndarray:
    """Source angles (pitch, yaw, roll in degrees) as a matrix whose columns are forward, left and up."""
    pitch, yaw, roll = (math.radians(a) for a in angles)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    forward = np.array([cp * cy, cp * sy, -sp])
    left = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp])
    up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
    return np.stack([forward, left, up], 1)


class ProbeVolume:
    def __init__(self, origin, rotation, mins, maxs, fade, priority, size, offset, cubes, shadows=None):
        self.origin = np.asarray(origin, np.float64)
        self.axes = rotation.T  # rows: forward, left, up
        self.mins = np.asarray(mins, np.float64)
        self.maxs = np.asarray(maxs, np.float64)
        self.fade = np.asarray(fade, np.float64)
        self.priority = priority
        self.size = np.asarray(size, np.int64)      # voxels
        self.offset = np.asarray(offset, np.int64)  # in the atlas
        self.cubes = cubes                           # (6, depth, height, width, 3) atlas faces
        self.shadows = shadows                       # (depth, height, width, 4): baked shadows of stationary lights

    def weight(self, p) -> float:
        local = self.axes @ (np.asarray(p) - self.origin)
        lo = np.clip((local - (self.mins - 0.02)) * self.fade, 0, 1)
        hi = np.clip(((self.maxs + 0.02) - local) * self.fade, 0, 1)
        edge = min(lo.min(), hi.min())
        return edge * edge * (3 - 2 * edge)

    def _trilinear(self, p):
        """The eight voxels around a point (atlas z, y, x) and their weights, like the GPU's trilinear filter."""
        local = self.axes @ (np.asarray(p) - self.origin)
        t = (local - self.mins) / np.maximum(self.maxs - self.mins, 1e-6)
        # Voxel centres sit half a voxel in from the box's faces.
        v = np.clip(t * self.size - 0.5, 0, self.size - 1)
        i0 = np.floor(v).astype(np.int64)
        i1 = np.minimum(i0 + 1, self.size - 1)
        f = v - i0
        for dz, wz in ((i0[2], 1 - f[2]), (i1[2], f[2])):
            for dy, wy in ((i0[1], 1 - f[1]), (i1[1], f[1])):
                for dx, wx in ((i0[0], 1 - f[0]), (i1[0], f[0])):
                    w = wx * wy * wz
                    if w != 0:
                        yield (self.offset[2] + dz, self.offset[1] + dy, self.offset[0] + dx), w

    def sample(self, p) -> np.ndarray:
        """The six ambient cube faces (+x, +y, +z, -x, -y, -z), (6, 3)."""
        result = np.zeros((6, 3))
        for (z, y, x), w in self._trilinear(p):
            result += w * self.cubes[:, z, y, x, :3]
        return result

    def sample_shadows(self, p) -> np.ndarray:
        """The baked shadows of the stationary lights (one per bakedshadowindex channel, 1 in full shadow)."""
        result = np.zeros(4)
        for (z, y, x), w in self._trilinear(p):
            result += w * self.shadows[z, y, x, :4]
        return result


def _volume_many(volume: "ProbeVolume", p: np.ndarray):
    """Weights (n,), ambient cubes (n, 6, 3) and baked shadows (n, 4) of one volume at many points."""
    local = (p - volume.origin) @ volume.axes.T
    lo = np.clip((local - (volume.mins - 0.02)) * volume.fade, 0, 1)
    hi = np.clip(((volume.maxs + 0.02) - local) * volume.fade, 0, 1)
    edge = np.minimum(lo.min(1), hi.min(1))
    weight = edge * edge * (3 - 2 * edge)
    t = (local - volume.mins) / np.maximum(volume.maxs - volume.mins, 1e-6)
    v = np.clip(t * volume.size - 0.5, 0, volume.size - 1)
    i0 = np.floor(v).astype(np.int64)
    i1 = np.minimum(i0 + 1, volume.size - 1)
    f = v - i0
    cubes = np.zeros((len(p), 6, 3))
    shadows = np.zeros((len(p), 4)) if volume.shadows is not None else None
    for dz in (0, 1):
        z = (i1 if dz else i0)[:, 2] + volume.offset[2]
        wz = f[:, 2] if dz else 1 - f[:, 2]
        for dy in (0, 1):
            y = (i1 if dy else i0)[:, 1] + volume.offset[1]
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dx in (0, 1):
                x = (i1 if dx else i0)[:, 0] + volume.offset[0]
                w = (wz * wy * (f[:, 0] if dx else 1 - f[:, 0]))[:, None]
                cubes += w[:, :, None] * np.transpose(volume.cubes[:, z, y, x, :3], (1, 0, 2))
                if shadows is not None:
                    shadows += w * volume.shadows[z, y, x, :4]
    return weight, cubes, shadows


class ProbeField:
    """Every probe volume of a map plus the sky light, for ambient cube lookups."""

    def __init__(self, volumes: list[ProbeVolume], ambient_constant, ambient_vertical):
        self.volumes = sorted(volumes, key=lambda v: v.priority)
        self.ambient_constant = np.asarray(ambient_constant, np.float64)
        self.ambient_vertical = np.asarray(ambient_vertical, np.float64)

    def ambient_cube(self, p) -> np.ndarray:
        """(6, 3): +x, +y, +z, -x, -y, -z irradiance at a point (game units)."""
        total = np.zeros((6, 3))
        covered = 0.0
        for volume in self.volumes:
            w = volume.weight(p)
            if w <= 0:
                continue
            contribution = w * (1 - covered)
            total += volume.sample(p) * contribution
            covered += contribution
            if covered > 0.99:
                break
        sky = np.tile(self.ambient_constant, (6, 1))
        sky[2] += self.ambient_vertical
        sky[5] -= self.ambient_vertical
        total += np.maximum(sky, 0) * (1 - covered)
        return total

    def sample_many(self, points: np.ndarray, chunk: int = 131072):
        """Ambient cubes (n, 6, 3) and baked shadows (n, 4) at many points (game units), blended across the
        volumes as ambient_cube and baked_shadows do for one."""
        points = np.asarray(points, np.float64)
        cubes = np.zeros((len(points), 6, 3), np.float32)
        shadows = np.zeros((len(points), 4), np.float32)
        sky = np.tile(self.ambient_constant, (6, 1))
        sky[2] += self.ambient_vertical
        sky[5] -= self.ambient_vertical
        sky = np.maximum(sky, 0)
        for start in range(0, len(points), chunk):
            p = points[start:start + chunk]
            total = np.zeros((len(p), 6, 3))
            covered = np.zeros(len(p))
            shadow_total = np.zeros((len(p), 4))
            shadow_covered = np.zeros(len(p))
            for volume in self.volumes:
                weight, cube, shadow = _volume_many(volume, p)
                contribution = weight * (1 - covered)
                total += cube * contribution[:, None, None]
                covered += contribution
                if shadow is not None:
                    contribution = weight * (1 - shadow_covered)
                    shadow_total += shadow * contribution[:, None]
                    shadow_covered += contribution
            total += sky[None] * (1 - covered)[:, None, None]
            cubes[start:start + chunk] = total
            shadows[start:start + chunk] = shadow_total
        return cubes, shadows

    def baked_shadows(self, p) -> np.ndarray:
        """(4,): the stationary lights' baked shadows at a point, blended across the volumes like the irradiance;
        unshadowed outside them."""
        total = np.zeros(4)
        covered = 0.0
        for volume in self.volumes:
            w = volume.weight(p) if volume.shadows is not None else 0.0
            if w <= 0:
                continue
            contribution = w * (1 - covered)
            total += volume.sample_shadows(p) * contribution
            covered += contribution
            if covered > 0.99:
                break
        return total


def decode_volume_bc(tex: vtex.CompiledTexture, decode_2d, channels: int = 4, dtype=np.float32) -> np.ndarray:
    """A block compressed volume texture's slices, (depth, height, width, channels): the slices' blocks laid end to
    end are a tall 2D texture, which ``decode_2d(dds_bytes)`` (OpenImageIO) decodes in one go. Light probe atlases
    keep 16 bit floats, all their formats hold (BC6H halves, BC7 bytes): a dl_midtown's would take gigabytes as
    32 bit."""
    data = tex.read_mip(0)
    width, height, depth = tex.width, tex.height, tex.depth
    per_slice = len(data) // depth
    out = np.ones((depth, height, width, channels), dtype)
    # Decoders cap images at 32768 rows, so slices go in batches.
    batch = max(1, 16384 // max(height, 1))
    for first in range(0, depth, batch):
        count = min(batch, depth - first)
        chunk = data[first * per_slice:(first + count) * per_slice]
        tall = vtex._dds_header(width, height * count, 1, vtex._DXGI[tex.format]) + chunk
        pixels = decode_2d(tall)
        have = pixels.shape[-1]
        out[first:first + count, ..., :min(have, channels)] =             pixels.reshape(count, height, width, have)[..., :channels]
    return out


def read_probe_volumes(content, entities, decode_2d, log=print) -> list[ProbeVolume]:
    """The light probe volumes of a map's entities, with their atlases decoded once."""
    atlases = {}
    volumes = []

    def atlas(path, channels):
        """The light in three channels, the four stationary lights' shadows in four."""
        if path not in atlases:
            data = content.try_read(path + "_c")
            atlases[path] = None
            if data is not None:
                try:
                    tex = vtex.CompiledTexture(data)
                    atlases[path] = (decode_volume_bc(tex, decode_2d, channels, np.float16) if tex.block_compressed
                                     else _decode_volume_raw(tex)[..., :channels].astype(np.float16))
                except Exception as error:
                    log(f"light probe atlas '{path}' failed: {error}")
        return atlases[path]

    for e in entities:
        if e.classname not in ("env_combined_light_probe_volume", "env_light_probe_volume"):
            continue
        if e.get_bool("startdisabled") or not e.has("light_probe_atlas_x"):
            continue
        path = e.get_str("lightprobetexture")
        slices = atlas(path, 3) if path else None
        if slices is None:
            continue
        faces = slices.shape[0] // 6
        cubes = slices[: faces * 6].reshape(6, faces, slices.shape[1], slices.shape[2], slices.shape[3])
        shadow_path = e.get_str("lightprobetexture_dlshd")
        shadows = atlas(shadow_path, 4) if shadow_path else None
        if shadows is not None and shadows.shape[:3] != cubes.shape[1:4]:
            shadows = None

        origin = e.get_vector("origin", 3, [0, 0, 0])
        rotation = rotation_matrix(e.get_vector("angles", 3, [0, 0, 0]))
        mins = e.get_vector("box_mins", 3, [-64, -64, -64])
        maxs = e.get_vector("box_maxs", 3, [64, 64, 64])
        fade = [1.0 / (f + 0.02) for f in e.get_vector("edge_fade_dists", 3, [0, 0, 0])]
        size = [max(1, e.get_int(f"light_probe_size_{a}", 1)) for a in "xyz"]
        offset = [e.get_int(f"light_probe_atlas_{a}", 0) for a in "xyz"]
        priority = (-e.get_int("indoor_outdoor_level"), e.get_int("array_index"))
        volumes.append(ProbeVolume(origin, rotation, mins, maxs, fade, priority, size, offset, cubes, shadows))
    return volumes


def _decode_volume_raw(tex: vtex.CompiledTexture) -> np.ndarray:
    data = tex.read_mip(0)
    per = len(data) // tex.depth
    return np.stack([vtex.decode_uncompressed(tex.format, data[i * per:(i + 1) * per], tex.width, tex.height)
                     for i in range(tex.depth)])


def sky_ambient(sun_entity, linear) -> tuple:
    """light_environment's sky light: (constant, vertical) per colour, as the mount's AddSunLighting computes it."""
    if sun_entity is None:
        return (0, 0, 0), (0, 0, 0)
    forward = rotation_matrix(sun_entity.get_vector("angles", 3, [0, 0, 0]))[:, 0]
    color = np.array([linear(c / 255) for c in sun_entity.get_vector("color", 3, [255, 255, 255])])
    color *= sun_entity.get_float("brightness", 1.0) * sun_entity.get_float("brightnessscale", 1.0)
    sky = np.array([linear(c / 255) for c in sun_entity.get_vector("skycolor", 3, [255, 255, 255])])
    sky *= sun_entity.get_float("skyintensity", 1.0)
    bounce = np.array([linear(c / 255) for c in sun_entity.get_vector("skyambientbounce", 3, [0, 0, 0])])
    sun_down = forward[2]
    sun_bounce = color * bounce * 0.5
    sky_bounce = sky * bounce * 0.5
    vertical = sky * 0.5 - sky_bounce + sun_bounce * sun_down
    constant = sky * 0.5 + sky_bounce + sun_bounce * abs(sun_down)
    return tuple(constant), tuple(vertical)
