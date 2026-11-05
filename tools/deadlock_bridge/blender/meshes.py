"""Draw call geometry (core.model.Part) to Blender meshes, with the attributes the materials read."""
import numpy as np

import bpy

from . import materials as matlib


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055).astype(np.float32)


class Geometry:
    """Parts gathered for one Blender mesh: arrays appended part by part, merged once."""

    def __init__(self):
        self.positions = []
        self.normals = []
        self.uv0 = []
        self.uv1 = []
        self.vcolor = []
        self.blend = []
        self.lod_ranges = []
        self.lightmap = []
        self.indices = []
        self.material_indices = []
        self.materials: list[str] = []
        self._material_slots: dict[str, int] = {}
        self.count = 0
        self.has_uv1 = False
        self.has_blend = False
        self.has_lightmap = False

    def add(self, part, material_path: str, mode: str, positions=None, normals=None, uv0=None, uv1=None,
            tint=(255, 255, 255), flip: bool = False, lightmap_scale=(1.0, 1.0), lod_range=None):
        """Appends a part, optionally placed (positions/normals/uvs already transformed) and tinted (bytes).
        ``lod_range``: the camera distances (metres) between which the game draws it, for a level of detail."""
        n = len(part.positions)
        if n == 0 or len(part.indices) < 3:
            return
        self.lod_ranges.append(None if lod_range is None else np.tile(np.asarray(lod_range, np.float32), (n, 1)))
        positions = part.positions if positions is None else positions
        normals = part.normals if normals is None else normals
        uv0 = part.uv0 if uv0 is None else uv0
        uv1 = part.uv1 if uv1 is None else uv1

        slot = self._material_slots.get(material_path)
        if slot is None:
            slot = self._material_slots[material_path] = len(self.materials)
            self.materials.append(material_path)

        self.positions.append(positions.astype(np.float32, copy=False))
        self.normals.append(normals.astype(np.float32, copy=False))
        self.uv0.append(uv0)
        if uv1 is not None:
            self.has_uv1 = True
        self.uv1.append(uv1)

        # Object tint and vertex paint combine in gamma space: the fragment tint is stored as gamma bytes, the draw
        # call's tint linear (ValveResourceFormat turns it to gamma the same way).
        tint_rgb = _linear_to_srgb(np.asarray(part.tint[:3], np.float32)) * (np.asarray(tint, np.float32) / 255.0)
        tint_alpha = float(part.tint[3])
        paint = part.color
        if mode == "pbr":
            rgb = _srgb_to_linear(tint_rgb)[None, :] * _srgb_to_linear(paint[:, :3])
            alpha = tint_alpha * paint[:, 3]
        else:
            linear_tint = np.tile(_srgb_to_linear(tint_rgb)[None, :], (n, 1))
            if mode == "simple" and part.blend_params is not None:
                w = part.blend_params[:, 3:4]
                linear_tint = 1.0 + (linear_tint - 1.0) * w
            painted = 1.0 + (_srgb_to_linear(paint[:, :3]) - 1.0) * paint[:, 3:4]
            rgb = linear_tint * painted
            alpha = np.full(n, tint_alpha, np.float32)
        vcolor = np.empty((n, 4), np.float32)
        vcolor[:, :3] = rgb
        vcolor[:, 3] = alpha
        self.vcolor.append(vcolor)

        if part.blend_params is not None:
            self.has_blend = True
        self.blend.append(part.blend_params)

        if part.lightmap_uv is not None:
            self.has_lightmap = True
            lm = np.empty((n, 2), np.float32)
            lm[:, 0] = part.lightmap_uv[:, 0] * lightmap_scale[0] + 1.0
            lm[:, 1] = (1.0 - part.lightmap_uv[:, 1] * lightmap_scale[1]) + 1.0
            self.lightmap.append(lm)
        else:
            self.lightmap.append(None)

        indices = part.indices.reshape(-1, 3)
        if flip:
            indices = indices[:, [0, 2, 1]]
        self.indices.append(indices + self.count)
        self.material_indices.append(np.full(len(indices), slot, np.int32))
        self.count += n

    def __bool__(self):
        return self.count > 0

    def build(self, name: str, material_lookup, scale: float) -> bpy.types.Mesh:
        """The Blender mesh; positions scaled from game units by ``scale``."""
        positions = np.concatenate(self.positions) * scale
        normals = np.concatenate(self.normals)
        triangles = np.concatenate(self.indices).astype(np.int32)
        material_indices = np.concatenate(self.material_indices)
        n = len(positions)
        t = len(triangles)

        me = bpy.data.meshes.new(name)
        me.vertices.add(n)
        me.vertices.foreach_set("co", positions.ravel())
        me.loops.add(t * 3)
        me.loops.foreach_set("vertex_index", triangles.ravel())
        me.polygons.add(t)
        me.polygons.foreach_set("loop_start", np.arange(0, t * 3, 3, dtype=np.int32))
        me.polygons.foreach_set("material_index", material_indices)
        me.polygons.foreach_set("use_smooth", np.ones(t, bool))

        for path in self.materials:
            me.materials.append(material_lookup(path))

        corner = triangles.ravel()
        uv0 = np.concatenate(self.uv0)
        uv = me.attributes.new("UVMap", "FLOAT2", "CORNER")
        flipped = uv0[corner].copy()
        flipped[:, 1] = 1.0 - flipped[:, 1]
        uv.data.foreach_set("vector", flipped.ravel())

        if self.has_uv1:
            uv1 = np.concatenate([u if u is not None else np.zeros((len(p), 2), np.float32) for u, p in zip(self.uv1, self.positions)])
            attr = me.attributes.new("UV2", "FLOAT2", "CORNER")
            flipped = uv1[corner].copy()
            flipped[:, 1] = 1.0 - flipped[:, 1]
            attr.data.foreach_set("vector", flipped.ravel())

        if any(r is not None for r in self.lod_ranges):
            # (near, far); (0, 0) for parts drawn at every distance. The lighting group hides the rest.
            ranges = np.concatenate([r if r is not None else np.zeros((len(p), 2), np.float32)
                                     for r, p in zip(self.lod_ranges, self.positions)])
            me.attributes.new("dl_lod_range", "FLOAT2", "POINT").data.foreach_set("vector", ranges.ravel())

        vcolor = np.concatenate(self.vcolor)
        attr = me.color_attributes.new("dl_vcolor", "BYTE_COLOR", "POINT")
        attr.data.foreach_set("color", np.clip(vcolor, 0, 1).ravel())

        if self.has_blend:
            blend = np.concatenate([b if b is not None else np.tile(np.array([0, 0, 0, 1], np.float32), (len(p), 1))
                                    for b, p in zip(self.blend, self.positions)])
            attr = me.color_attributes.new("dl_blend", "FLOAT_COLOR", "POINT")
            attr.data.foreach_set("color", blend.ravel())

        if self.has_lightmap:
            lm = np.concatenate([l if l is not None else np.zeros((len(p), 2), np.float32) for l, p in zip(self.lightmap, self.positions)])
            attr = me.attributes.new("dl_lightmap", "FLOAT2", "POINT")
            attr.data.foreach_set("vector", lm.ravel())

        me.validate(clean_customdata=False)
        me.update()
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lengths > 1e-8, normals / np.maximum(lengths, 1e-8), np.array([0, 0, 1], np.float32))
        if len(me.vertices) == n:
            me.normals_split_custom_set_from_vertices(normals)
        return me


def place_fragment(part, matrix: np.ndarray, uv_scale=None):
    """A part moved by a fragment's 3x4 transform: positions, normals (inverse transpose), texture coordinates scaled by
    the transform's axis scales where the material asks, and whether the winding flips."""
    linear = matrix[:3, :3]
    positions = part.positions.astype(np.float64) @ linear.T + matrix[:3, 3]
    determinant = np.linalg.det(linear)
    try:
        normal_matrix = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        normal_matrix = linear
    normals = part.normals.astype(np.float64) @ normal_matrix.T
    uv0, uv1 = scaled_uvs(part, linear, uv_scale)
    return positions.astype(np.float32), normals.astype(np.float32), uv0, uv1, determinant < 0


def scaled_uvs(part, linear: np.ndarray, uv_scale=None):
    """A part's texture coordinates scaled by a transform's axis scales where its material asks."""
    uv0, uv1 = part.uv0, part.uv1
    if uv_scale is None or not uv_scale.any:
        return uv0, uv1
    axis_scales = [1.0, *np.linalg.norm(linear, axis=0)]

    def scaled(values, axis, origin):
        if axis == 0:
            return values
        return np.where(values >= 0, (values - origin) * axis_scales[axis] + origin, values)

    uv0 = np.stack([scaled(uv0[:, 0], uv_scale.u, uv_scale.ou), scaled(uv0[:, 1], uv_scale.v, uv_scale.ov)], 1).astype(np.float32)
    if uv1 is not None:
        uv1 = np.stack([scaled(uv1[:, 0], uv_scale.u2, uv_scale.ou2), scaled(uv1[:, 1], uv_scale.v2, uv_scale.ov2)], 1).astype(np.float32)
    return uv0, uv1


class UvScale:
    """A material's g_nScaleTexCoord*ByModelScale* parameters (axis 0 none, 1-3 x/y/z) and their origins."""
    __slots__ = ("u", "v", "ou", "ov", "u2", "v2", "ou2", "ov2")

    def __init__(self, m):
        get = (lambda k, d: m.ints.get(k, m.floats.get(k, d))) if m is not None else (lambda k, d: d)
        self.u = int(get("g_nScaleTexCoordUByModelScaleAxis", 0))
        self.v = int(get("g_nScaleTexCoordVByModelScaleAxis", 0))
        self.ou = get("g_nScaleTexCoordUByModelScaleOrigin", 0) / 2
        self.ov = 1 - get("g_nScaleTexCoordVByModelScaleOrigin", 2) / 2
        self.u2 = int(get("g_nScaleTexCoord2UByModelScaleAxis", 0))
        self.v2 = int(get("g_nScaleTexCoord2VByModelScaleAxis", 0))
        self.ou2 = get("g_nScaleTexCoord2UByModelScaleOrigin", 0) / 2
        self.ov2 = get("g_nScaleTexCoord2VByModelScaleOrigin", 2) / 2

    @property
    def any(self):
        return bool(self.u or self.v or self.u2 or self.v2)
