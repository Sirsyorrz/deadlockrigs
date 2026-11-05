"""A map's world (vwrld_c), its world nodes (vwnod_c) and entity lumps (vents_c)."""
import numpy as np

from . import kv3
from .resource import ResourceFile

# ObjectTypeFlags_t
FLAG_NO_SHADOWS = 0x20
FLAG_OVERLAY = 0x2000
FLAG_DECAL = 0x4

_FLAG_NAMES = {
    "OBJECT_TYPE_IMAGE_LOD": 0x1, "OBJECT_TYPE_GEOMETRY_LOD": 0x2, "OBJECT_TYPE_DECAL": 0x4,
    "OBJECT_TYPE_MODEL": 0x8, "OBJECT_TYPE_BLOCK_LIGHT": 0x10, "OBJECT_TYPE_NO_SHADOWS": 0x20,
    "OBJECT_TYPE_WORLDSPACE_TEXURE_BLEND": 0x40, "OBJECT_TYPE_DISABLED_IN_LOW_QUALITY": 0x80,
    "OBJECT_TYPE_NO_SUN_SHADOWS": 0x100, "OBJECT_TYPE_RENDER_WITH_DYNAMIC": 0x200,
    "OBJECT_TYPE_RENDER_TO_CUBEMAPS": 0x400, "OBJECT_TYPE_MODEL_HAS_LODS": 0x800, "OBJECT_TYPE_OVERLAY": 0x2000,
    "OBJECT_TYPE_PRECOMPUTED_VISMEMBERS": 0x4000,
}


def norm(path: str) -> str:
    return (path or "").replace("\\", "/")


def read_flags(value) -> int:
    if isinstance(value, str):
        flags = 0
        for name in value.split("|"):
            name = name.strip()
            if name in _FLAG_NAMES:
                flags |= _FLAG_NAMES[name]
            elif name.isdigit():
                flags |= int(name)
        return flags
    return kv3.as_int(value) & 0xFFFFFFFF


def read_matrix(value) -> np.ndarray:
    """A 3x4 row major transform (three rows of rotation/scale terms then a translation) as a 4x4 matrix."""
    m = np.eye(4, dtype=np.float64)
    if isinstance(value, list) and len(value) == 3 and isinstance(value[0], list):
        for r in range(3):
            row = value[r]
            for c in range(min(4, len(row))):
                m[r, c] = kv3.as_float(row[c])
    elif isinstance(value, list) and len(value) >= 12:
        m[:3, :4] = np.array([kv3.as_float(x) for x in value[:12]]).reshape(3, 4)
    return m


class World:
    def __init__(self, data: bytes):
        world = ResourceFile(data).keyvalues("DATA")
        if world is None:
            raise ValueError("world has no DATA block")
        self.node_prefixes = [norm(n.get("m_worldNodePrefix", "")) for n in world.get("m_worldNodes") or []]
        self.entity_lumps = [norm(p) for p in world.get("m_entityLumps") or []]
        self.lighting = world.get("m_worldLightingInfo") or {}


class SceneObject:
    __slots__ = ("index", "model", "matrix", "tint", "skin", "flags", "lod_override", "layer", "lighting_origin")


class Fragment:
    __slots__ = ("draw_call", "matrix", "tint", "lod_mask", "lod_setup", "flags")


class Aggregate:
    __slots__ = ("model", "all_flags", "any_flags", "layer", "fragments", "lod_setups")


class WorldNode:
    def __init__(self, data: bytes):
        node = ResourceFile(data).keyvalues("DATA")
        if node is None:
            raise ValueError("world node has no DATA block")
        self.layer_names = list(node.get("m_layerNames") or [])
        layer_indices = node.get("m_sceneObjectLayerIndices") or []

        self.scene_objects = []
        for i, d in enumerate(node.get("m_sceneObjects") or []):
            o = SceneObject()
            o.index = i
            o.model = norm(d.get("m_renderableModel", ""))
            o.matrix = read_matrix(d.get("m_vTransform"))
            o.tint = [kv3.as_float(x) for x in (d.get("m_vTintColor") or [1, 1, 1, 1])]
            o.skin = d.get("m_skin", "") or ""
            o.flags = read_flags(d.get("m_nObjectTypeFlags"))
            o.lod_override = kv3.as_int(d.get("m_nLODOverride"), -1)
            layer = kv3.as_int(layer_indices[i], -1) if i < len(layer_indices) else -1
            o.layer = self.layer_names[layer] if 0 <= layer < len(self.layer_names) else None
            origin = d.get("m_vLightingOrigin")
            o.lighting_origin = [kv3.as_float(x) for x in origin] if isinstance(origin, list) else None
            self.scene_objects.append(o)

        self.aggregates = [self._read_aggregate(a) for a in node.get("m_aggregateSceneObjects") or []]

        self.material_overrides = []
        for d in node.get("m_materialOverrides") or []:
            self.material_overrides.append((kv3.as_int(d.get("m_nSceneObjectIndex")), kv3.as_int(d.get("m_nSubSceneObject")),
                                            kv3.as_int(d.get("m_nDrawCallIndex")) & 0xFFFFFFFF, norm(d.get("m_pMaterial", ""))))

    def _read_aggregate(self, d: dict) -> Aggregate:
        transforms = d.get("m_fragmentTransforms") or []
        t = 0
        fragments = []
        for f in d.get("m_aggregateMeshes") or []:
            fr = Fragment()
            fr.matrix = None
            if kv3.as_bool(f.get("m_bHasTransform")) and t < len(transforms):
                fr.matrix = read_matrix(transforms[t])
                t += 1
            fr.draw_call = kv3.as_int(f.get("m_nDrawCallIndex"))
            tint = f.get("m_vTintColor") or [255, 255, 255]
            fr.tint = tuple(kv3.as_int(x) for x in tint[:3]) if len(tint) >= 3 else (255, 255, 255)
            fr.lod_mask = kv3.as_int(f.get("m_nLODGroupMask")) & 0xFFFFFFFF
            fr.lod_setup = kv3.as_int(f.get("m_nLODSetupIndex"), -1)
            fr.flags = read_flags(f.get("m_objectFlags"))
            fragments.append(fr)

        a = Aggregate()
        a.model = norm(d.get("m_renderableModel", ""))
        a.all_flags = read_flags(d.get("m_allFlags"))
        a.any_flags = read_flags(d.get("m_anyFlags"))
        layer = kv3.as_int(d.get("m_nLayer"), -1)
        a.layer = self.layer_names[layer] if 0 <= layer < len(self.layer_names) else None
        a.fragments = fragments
        a.lod_setups = [([kv3.as_float(x) for x in s.get("m_vLODOrigin") or [0, 0, 0]],
                         kv3.as_float(s.get("m_fMaxObjectScale"), 1.0),
                         [kv3.as_float(x) for x in s.get("m_fSwitchDistances") or []]) for s in d.get("m_lodSetups") or []]
        return a


class Entity:
    """A map entity: its key values (KV3 typed, or strings in older maps) and outputs."""

    __slots__ = ("values", "connections", "lump")

    def __init__(self, values: dict, connections: list, lump: str):
        self.values = values
        self.connections = connections
        self.lump = lump

    @property
    def classname(self) -> str:
        return (self.get_str("classname") or "").lower()

    @property
    def targetname(self) -> str:
        return self.get_str("targetname") or ""

    def has(self, key: str) -> bool:
        return key.lower() in self.values

    def get(self, key: str, fallback=None):
        return self.values.get(key.lower(), fallback)

    def get_str(self, key: str, fallback=None):
        value = self.values.get(key.lower())
        if value is None:
            return fallback
        return format_value(value)

    def get_float(self, key: str, fallback: float = 0.0) -> float:
        value = self.values.get(key.lower())
        if value is None or isinstance(value, (list, dict)):
            return fallback
        return kv3.as_float(value, fallback)

    def get_int(self, key: str, fallback: int = 0) -> int:
        value = self.values.get(key.lower())
        if value is None or isinstance(value, (list, dict)):
            return fallback
        return kv3.as_int(value, fallback)

    def get_bool(self, key: str, fallback: bool = False) -> bool:
        value = self.values.get(key.lower())
        if value is None or isinstance(value, (list, dict)):
            return fallback
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            return kv3.as_int(value, int(fallback)) != 0
        return kv3.as_bool(value, fallback)

    def get_vector(self, key: str, count: int = 3, fallback=None):
        value = self.values.get(key.lower())
        if value is None:
            return fallback
        if isinstance(value, list):
            out = [kv3.as_float(x) for x in value[:count]]
            return out + [0.0] * (count - len(out))
        if isinstance(value, str):
            parts = value.split()
            if not parts:
                return fallback
            out = []
            for p in parts[:count]:
                try:
                    out.append(float(p))
                except ValueError:
                    out.append(0.0)
            return out + [0.0] * (count - len(out))
        return fallback


def format_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return " ".join(format_value(v) for v in value)
    if isinstance(value, dict):
        return ""
    return str(value)


class EntityLump:
    def __init__(self, data: bytes):
        lump = ResourceFile(data).keyvalues("DATA")
        if lump is None:
            raise ValueError("entity lump has no DATA block")
        self.name = lump.get("m_name", "")
        self.child_lumps = [norm(p) for p in lump.get("m_childLumps") or []]
        self.entities = []
        self.legacy = 0
        for e in lump.get("m_entityKeyValues") or []:
            values = (e.get("keyValues3Data") or {}).get("values")
            if not isinstance(values, dict):
                self.legacy += 1
                continue
            connections = [(c.get("m_outputName", ""), c.get("m_targetName", ""), c.get("m_inputName", ""),
                            c.get("m_overrideParam", ""), kv3.as_float(c.get("m_flDelay")), kv3.as_int(c.get("m_nTimesToFire"), -1))
                           for c in e.get("m_connections") or []]
            self.entities.append(Entity({k.lower(): v for k, v in values.items()}, connections, self.name))
