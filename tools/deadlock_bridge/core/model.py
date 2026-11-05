"""Compiled models (vmdl_c): skeleton, LOD and mesh group tables, embedded meshes and their draw calls.

``read_part`` turns one draw call into numpy arrays, with the defaults the s&box mount gives missing attributes.
"""
import numpy as np

from . import kv3, mesh as meshlib
from .resource import ResourceFile


class DrawCall:
    __slots__ = ("material", "primitive", "base_vertex", "vertex_count", "start_index", "index_count",
                 "vertex_buffers", "vertex_offsets", "index_buffer", "index_offset", "tint", "alpha", "lightmapped")

    def __init__(self, d: dict):
        self.material = (d.get("m_material") or d.get("m_pMaterial") or "").replace("\\", "/")
        primitive = d.get("m_nPrimitiveType")
        self.primitive = primitive if isinstance(primitive, str) else str(kv3.as_int(primitive))
        self.base_vertex = kv3.as_int(d.get("m_nBaseVertex"))
        self.vertex_count = kv3.as_int(d.get("m_nVertexCount"))
        self.start_index = kv3.as_int(d.get("m_nStartIndex"))
        self.index_count = kv3.as_int(d.get("m_nIndexCount"))
        vbs = d.get("m_vertexBuffers") or []
        self.vertex_buffers = [kv3.as_int(v.get("m_hBuffer")) for v in vbs]
        self.vertex_offsets = [kv3.as_int(v.get("m_nBindOffsetBytes")) for v in vbs]
        ib = d.get("m_indexBuffer") or {}
        self.index_buffer = kv3.as_int(ib.get("m_hBuffer"))
        self.index_offset = kv3.as_int(ib.get("m_nBindOffsetBytes"))
        tint = d.get("m_vTintColor") or [1, 1, 1]
        self.tint = [kv3.as_float(x) for x in tint[:3]] + [1.0] * (3 - len(tint[:3]))
        self.alpha = kv3.as_float(d.get("m_flAlpha"), 1.0)
        self.lightmapped = kv3.as_bool(d.get("m_bHasBakedLightingFromLightMap"))

    @property
    def is_triangles(self):
        return self.primitive in ("RENDER_PRIM_TRIANGLES", "5")


class CompiledMesh:
    def __init__(self, resource: ResourceFile, name: str, index: int, data: dict, buffer_block: int,
                 vertex_descriptions, index_descriptions):
        self.resource = resource
        self.name = name
        self.index = index
        self.data = data or {}
        self._buffer_block = buffer_block
        self._vertex_descriptions = vertex_descriptions
        self._index_descriptions = index_descriptions
        self._vertex_buffers = None
        self._index_buffers = None

    @property
    def bone_weight_count(self) -> int:
        skeleton = self.data.get("m_skeleton") or {}
        return kv3.as_int(skeleton.get("m_nBoneWeightCount"))

    def _decode(self):
        if self._vertex_buffers is not None:
            return
        if self._buffer_block >= 0:
            self._vertex_buffers, self._index_buffers = meshlib.read_buffer_block(self.resource.block_data(self._buffer_block))
            return
        vds = self._vertex_descriptions if self._vertex_descriptions is not None else self.data.get("m_vertexBuffers") or []
        ids = self._index_descriptions if self._index_descriptions is not None else self.data.get("m_indexBuffers") or []
        self._vertex_buffers = [meshlib.read_described_buffer(d, self.resource, True) for d in vds]
        self._index_buffers = [meshlib.read_described_buffer(d, self.resource, False) for d in ids]

    @property
    def vertex_buffers(self):
        self._decode()
        return self._vertex_buffers

    @property
    def index_buffers(self):
        self._decode()
        return self._index_buffers

    def draw_calls(self) -> list[DrawCall]:
        calls = []
        for scene_object in self.data.get("m_sceneObjects") or []:
            for d in scene_object.get("m_drawCalls") or []:
                calls.append(DrawCall(d))
        return calls


class Part:
    """One draw call's geometry, its vertices cut down to the range its indices use."""
    __slots__ = ("material", "positions", "normals", "tangents", "uv0", "uv1", "color", "tint", "lightmap_uv",
                 "blend_params", "indices", "bones", "weights", "first_vertex", "lod_mask", "mesh_index", "draw_call")

    def __init__(self):
        self.tangents = None
        self.uv1 = None
        self.lightmap_uv = None
        self.blend_params = None
        self.bones = None
        self.weights = None
        self.lod_mask = -1


class CompiledModel:
    def __init__(self, data: bytes):
        self.resource = ResourceFile(data)
        self.data = self.resource.keyvalues("DATA")
        if self.data is None:
            raise ValueError("model has no DATA block")
        d = self.data
        self.name = d.get("m_name", "")

        skeleton = d.get("m_modelSkeleton") or {}
        names = skeleton.get("m_boneName") or []
        parents = skeleton.get("m_nParent") or []
        positions = skeleton.get("m_bonePosParent") or []
        rotations = skeleton.get("m_boneRotParent") or []
        self.bones = []
        for i, name in enumerate(names):
            p = positions[i] if i < len(positions) else [0, 0, 0]
            r = rotations[i] if i < len(rotations) else [0, 0, 0, 1]
            self.bones.append((name, kv3.as_int(parents[i], -1) if i < len(parents) else -1,
                               [kv3.as_float(x) for x in p[:3]], [kv3.as_float(x) for x in r[:4]]))

        self.lod_switch_distances = [kv3.as_float(x) for x in d.get("m_lodGroupSwitchDistances") or []]
        self.mesh_groups = list(d.get("m_meshGroups") or [])
        self.default_mesh_group_mask = kv3.as_int(d.get("m_nDefaultMeshGroupMask"), -1) & 0xFFFFFFFFFFFFFFFF
        self._lod_masks = [kv3.as_int(x) for x in d.get("m_refLODGroupMasks") or []]
        self._group_masks = [kv3.as_int(x) & 0xFFFFFFFFFFFFFFFF for x in d.get("m_refMeshGroupMasks") or []]
        self._remap = [kv3.as_int(x) for x in d.get("m_remappingTable") or []]
        self._remap_starts = [kv3.as_int(x) for x in d.get("m_remappingTableStarts") or []]
        self.material_groups = [(g.get("m_name", ""), [m.replace("\\", "/") for m in g.get("m_materials") or []])
                                for g in d.get("m_materialGroups") or []]
        self.meshes = self._read_meshes()

    def _read_meshes(self) -> list[CompiledMesh]:
        meshes = []
        control = self.resource.keyvalues("CTRL")
        if control is None:
            return meshes
        r = self.resource
        for entry in control.get("embedded_meshes") or []:
            if "vbib_block" in entry:
                meshes.append(CompiledMesh(r, entry.get("name", ""), kv3.as_int(entry.get("mesh_index")),
                                           r.keyvalues(kv3.as_int(entry.get("data_block"))),
                                           kv3.as_int(entry.get("vbib_block")), None, None))
            else:
                meshes.append(CompiledMesh(r, entry.get("m_Name", ""), kv3.as_int(entry.get("m_nMeshIndex")),
                                           r.keyvalues(kv3.as_int(entry.get("m_nDataBlock"))), -1,
                                           entry.get("m_vertexBuffers") or [], entry.get("m_indexBuffers") or []))
        return meshes

    def lod_mask(self, mesh_index: int) -> int:
        if mesh_index < len(self._lod_masks) and self._lod_masks[mesh_index] != 0:
            return self._lod_masks[mesh_index]
        return -1

    def group_mask(self, mesh_index: int) -> int:
        return self._group_masks[mesh_index] if mesh_index < len(self._group_masks) else 0xFFFFFFFFFFFFFFFF

    def visible_by_default(self, mesh_index: int) -> bool:
        return len(self.mesh_groups) <= 1 or (self.group_mask(mesh_index) & self.default_mesh_group_mask) != 0

    def bone_remap(self, mesh_index: int):
        if mesh_index >= len(self._remap_starts):
            return None
        start = self._remap_starts[mesh_index]
        end = self._remap_starts[mesh_index + 1] if mesh_index + 1 < len(self._remap_starts) else len(self._remap)
        return np.array(self._remap[start:end], np.int32)

    def lowest_lod(self) -> int:
        """The most detailed LOD any mesh is in."""
        combined = 0
        for m in self._lod_masks:
            combined |= m
        if combined == 0:
            return 0
        return (combined & -combined).bit_length() - 1

    def default_parts(self, lod: int | None = None, skinned: bool = False) -> list[Part]:
        """Every triangle draw call of the meshes shown by default at a LOD (the most detailed one if None)."""
        lod = self.lowest_lod() if lod is None else lod
        parts = []
        for m in self.meshes:
            if not self.visible_by_default(m.index):
                continue
            mask = self.lod_mask(m.index)
            if mask != -1 and not (mask >> lod) & 1:
                continue
            remap = self.bone_remap(m.index) if skinned else None
            for i, call in enumerate(m.draw_calls()):
                if not call.is_triangles or call.index_count < 3 or not call.vertex_buffers:
                    continue
                part = read_part(m, call, remap, skinned and bool(self.bones) and m.bone_weight_count > 0)
                part.lod_mask = mask
                part.mesh_index = m.index
                part.draw_call = i
                parts.append(part)
        return parts


def read_part(mesh: CompiledMesh, call: DrawCall, remap=None, skinned: bool = False) -> Part:
    index_buffer = mesh.index_buffers[call.index_buffer]
    start = call.start_index + call.index_offset // index_buffer.stride
    indices = index_buffer.indices()[start:start + call.index_count].astype(np.int64)
    first = int(indices.min())
    last = int(indices.max())
    indices = (indices - first).astype(np.int32)
    n = last - first + 1

    part = Part()
    part.material = call.material
    part.indices = indices
    part.first_vertex = first
    part.positions = np.zeros((n, 3), np.float32)
    part.normals = np.tile(np.array([0, 0, 1], np.float32), (n, 1))
    part.uv0 = np.zeros((n, 2), np.float32)
    part.color = np.ones((n, 4), np.float32)

    has_uv = has_color = has_tangent = has_lightmap = has_blend = False

    for b, buffer_index in enumerate(call.vertex_buffers):
        buffer = mesh.vertex_buffers[buffer_index]
        offset = call.base_vertex + call.vertex_offsets[b] // buffer.stride + first
        if offset < 0 or offset + n > buffer.count:
            raise ValueError(f"draw call in mesh '{mesh.name}' references vertices outside its buffer")

        position = buffer.find("POSITION")
        if position is not None:
            part.positions = np.ascontiguousarray(buffer.read(position, offset, n)[:, :3])

        normal = buffer.find("NORMAL")
        if normal is not None:
            normals, tangents = meshlib.read_normals(buffer, normal, offset, n)
            part.normals = normals
            if tangents is not None:
                part.tangents = tangents
                has_tangent = True

        tangent = buffer.find("TANGENT")
        if tangent is not None and not has_tangent:
            t = buffer.read(tangent, offset, n)
            tangents = np.ones((n, 4), np.float32)
            tangents[:, :t.shape[1]] = t[:, :4]
            part.tangents = tangents

        if not has_uv:
            uv0 = buffer.find("TEXCOORD", 0)
            if uv0 is not None:
                part.uv0 = np.ascontiguousarray(buffer.read(uv0, offset, n)[:, :2])
                uv1 = buffer.find("TEXCOORD", 1)
                if uv1 is not None:
                    part.uv1 = np.ascontiguousarray(buffer.read(uv1, offset, n)[:, :2])
                has_uv = True

        if not has_color:
            color = buffer.find("COLOR")
            if color is not None:
                c = buffer.read(color, offset, n)
                out = np.ones((n, 4), np.float32)
                out[:, :c.shape[1]] = c[:, :4]
                part.color = out
                has_color = True

        # Lightmap coordinates only mean something on draw calls baked into the lightmaps.
        if not has_lightmap and call.lightmapped:
            lightmap = buffer.find("TEXCOORD", 3)
            if lightmap is not None:
                part.lightmap_uv = np.ascontiguousarray(buffer.read(lightmap, offset, n)[:, :2])
                has_lightmap = True

        if not has_blend:
            blend = buffer.find("TEXCOORD", 4)
            if blend is not None:
                v = buffer.read(blend, offset, n)
                out = np.zeros((n, 4), np.float32)
                out[:, 3] = 1
                out[:, :v.shape[1]] = v[:, :4]
                part.blend_params = out
                has_blend = True

        if skinned and part.bones is None:
            blend_indices = buffer.find("BLENDINDICES")
            if blend_indices is not None:
                weights_field = buffer.find("BLENDWEIGHT") or buffer.find("BLENDWEIGHTS")
                bones, weights = meshlib.read_blend(buffer, blend_indices, weights_field, mesh.bone_weight_count, offset, n)
                if remap is not None and len(remap):
                    bones = remap[np.clip(bones, 0, len(remap) - 1)]
                part.bones = bones
                part.weights = weights

    part.tint = (call.tint[0], call.tint[1], call.tint[2], call.alpha)
    return part
