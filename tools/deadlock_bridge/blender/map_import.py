"""Imports a Deadlock map as the game builds it: the world nodes' merged geometry (aggregates) and placed models,
the entity lumps' props and lights, and the baked lighting, sky, fog and post processing the materials need."""
import math
import os
import re
import time

import numpy as np

import bpy
from mathutils import Matrix

from ..core import model as vmdl
from ..core import world as vworld
from ..core import texture as vtex
from ..core import cubemap as cubelib
from ..core import probes as probelib
from ..core.content import compiled, normalize
from . import lighting, lights, materials as matlib, post
from .meshes import Geometry, UvScale, place_fragment, scaled_uvs
from .textures import TextureCache

# Entities whose models aren't drawn (volumes, markup) even when they name one.
_UNDRAWN_PREFIXES = ("trigger_", "func_nav", "func_clip", "env_", "info_", "point_", "logic_", "light_", "snd_",
                     "post_processing", "citadel_trigger", "citadel_snd", "func_regenerate", "citadel_minimap")


def _no_shadow_materials(mesh) -> bool:
    """Every material of the mesh stays out of the game's shadow maps."""
    materials = [m for m in mesh.materials if m is not None]
    return bool(materials) and all(m.get("dl_no_shadow") or m.get("dl_outline_shell") for m in materials)


# How far decals are lifted off the surface they lie on, in game units: clear of Blender's depth precision a few
# hundred units away, too little to see.
OVERLAY_LIFT = 0.5


class Options:
    def __init__(self, **kwargs):
        self.unit_scale = 0.0254
        self.max_texture_size = 0
        self.props = True
        self.lights = True
        self.markers = True
        self.reflection_probes = True
        self.setup_render = True
        self.post_processing = True
        self.volumetric_fog = True
        self.compact_textures = True
        self.particles = True
        # Every level of detail, switched by camera distance as the game does; off, only the most detailed.
        self.detail_levels = False
        # Map effects have been running since the map loaded when you look at it in the game: a few seconds in,
        # looping ones are past their start-up (the vent steam grows in over its first five).
        self.particle_time = 6.0
        self.__dict__.update(kwargs)


def _decode_dds(cache: TextureCache, data: bytes, tag: str) -> np.ndarray:
    """A DDS (any block format) through OpenImageIO, (h, w, c) float32, top row first."""
    import OpenImageIO as oiio
    path = os.path.join(cache.folder, f"_decode_{tag}.dds")
    with open(path, "wb") as f:
        f.write(data)
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(oiio.geterror())
    pixels = inp.read_image("float")
    inp.close()
    try:
        os.remove(path)
    except OSError:
        pass
    return np.asarray(pixels, np.float32)


def _float_image(name: str, pixels: np.ndarray, cache: TextureCache, file_stem: str) -> bpy.types.Image:
    """A float image saved as an EXR in the cache (bottom row first, as Blender stores pixels)."""
    import OpenImageIO as oiio
    h, w = pixels.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :min(pixels.shape[2], 3)] = pixels[..., :3]
    path = os.path.join(cache.folder, file_stem + ".exr")
    out = oiio.ImageOutput.create(path)
    spec = oiio.ImageSpec(w, h, 4, "half")
    spec.attribute("compression", "zip")
    out.open(path, spec)
    out.write_image(np.ascontiguousarray(rgba[::-1]))
    out.close()
    image = bpy.data.images.load(path, check_existing=True)
    image.name = name
    image.colorspace_settings.name = "Linear Rec.709"
    return image


def _memory_gb() -> float | None:
    """This process's committed memory (Windows), for the import log."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                      "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                      "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PagefileUsage / 2 ** 30
    except Exception:
        return None


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


class MapImporter:
    def __init__(self, context, content, map_name: str, cache_dir: str, options: Options, progress=None):
        self.context = context
        self.content = content
        self.map = map_name.lower()
        self.options = options
        self.scale = options.unit_scale
        self._progress_callback = progress or (lambda *_: None)
        # Progress, warnings and totals go to import_log.txt in the cache folder as the import runs.
        self.log_path = os.path.join(cache_dir, "import_log.txt")
        self._log_file = None
        self._started = time.time()
        self._last_line = (0.0, "")
        self.log: list[str] = []
        self.textures = TextureCache(content, cache_dir, options.max_texture_size, options.compact_textures,
                                     parallel=True)
        self.stats = {}
        self._model_cache: dict[str, vmdl.CompiledModel | None] = {}
        self._mesh_cache: dict[tuple, bpy.types.Mesh | None] = {}
        self._model_known: dict[str, bool] = {}
        self._uv_scales: dict[str, UvScale] = {}

    def warn(self, message: str):
        self.log.append(message)
        print(f"[Deadlock] {self.map}: {message}")
        self._write_log(f"warning: {message}")

    def progress(self, fraction: float, text: str):
        """Reports progress to the caller, and to the log at each new phase and every couple of seconds within one,
        with the time taken, memory and an estimate of what's left."""
        self._progress_callback(fraction, text)
        now = time.time()
        phase = re.sub(r"\d+/\d+", "", text).strip()
        last_time, last_phase = self._last_line
        if phase == last_phase and now - last_time < 2.0:
            return
        self._last_line = (now, phase)
        elapsed = now - self._started
        memory = _memory_gb()
        line = f"+{_clock(elapsed)}  {fraction * 100:3.0f}%  {text}"
        if memory is not None:
            line += f"   RAM {memory:.1f} GB"
        if fraction >= 0.05 and elapsed >= 20:
            line += f"   ~{_clock(elapsed * (1 - fraction) / fraction)} left"
        self._write_log(line)

    def _write_log(self, line: str):
        if self._log_file is None:
            return
        try:
            self._log_file.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
            self._log_file.flush()
        except OSError:
            pass

    def _open_log(self):
        try:
            self._log_file = open(self.log_path, "w", encoding="utf-8")
        except OSError:
            self._log_file = None
        self._write_log(f"Importing {self.map} (texture limit {self.options.max_texture_size or 'full size'}, "
                        f"detail levels {'on' if self.options.detail_levels else 'off'})")
        print(f"[Deadlock] progress log: {self.log_path}")

    def _close_log(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        self._started = time.time()
        self._open_log()
        try:
            stats = self._run()
        except Exception as error:
            self._write_log(f"FAILED: {error}")
            raise
        finally:
            if self._log_file is not None:
                self._write_log(f"Finished in {_clock(time.time() - self._started)}, RAM "
                                f"{(_memory_gb() or 0):.1f} GB: " + ", ".join(f"{k} {v}" for k, v in self.stats.items()
                                                                       if k != "shaders"))
            self._close_log()
        return stats

    def _run(self):
        start = time.time()
        self.progress(0.0, "Reading the map")
        content = self.content
        content.prefer_map(self.map)
        world_path = f"maps/{self.map}/world.vwrld_c"
        if not content.exists(world_path):
            raise RuntimeError(f"'{self.map}' has no world")
        self.world = vworld.World(content.read(world_path))
        self.entities = self._read_entities()
        self.progress(0.05, "Lighting")

        scene = self.context.scene
        self.root = bpy.data.collections.new(self.map)
        scene.collection.children.link(self.root)
        self.root["dl_map"] = self.map
        self.root["dl_detail_levels"] = bool(self.options.detail_levels)
        self.collections = {}
        for name in ("World", "Props", "Lights", "Reflection Probes", "Markers"):
            c = bpy.data.collections.new(f"{name}")
            self.root.children.link(c)
            self.collections[name] = c

        self._setup_lighting()
        self.progress(0.1, "World geometry")
        self._import_world()
        if self.options.props:
            self.progress(0.6, "Props")
            self._import_entity_props()
        self.progress(0.8, "Lights")
        if self.options.lights:
            self._import_lights()
        if self.options.markers:
            self._import_markers()
        if self.options.reflection_probes:
            self._import_reflection_probes()
        self.progress(0.82, "Decals")
        self._lift_overlays()
        self._light_overlays()
        self.progress(0.85, "Light probes")
        self._apply_probes()
        self._add_outlines()
        if self.options.volumetric_fog:
            self.progress(0.88, "Volumetric fog")
            try:
                from .volumetric_fog import build_volumetric_fog
                build_volumetric_fog(self)
            except Exception as error:
                self.warn(f"volumetric fog failed: {error}")
        if self.options.particles:
            self.progress(0.9, "Particles")
            self._import_particles()
        self.progress(0.95, "Render settings")
        self._setup_scene()
        self.textures.finish(lambda done, total: self.progress(0.96 + 0.04 * done / max(total, 1),
                                                               f"Texture conversions {done}/{total}"))

        for path, error in self.materials.failed.items():
            self.warn(f"material '{path}' failed: {error.splitlines()[0]}")
        for path, error in self.textures.failed.items():
            self.warn(f"texture '{path}' failed: {error}")
        if self.textures.missing:
            self.warn(f"{len(self.textures.missing)} textures are missing from the game's content")
        self.stats["seconds"] = round(time.time() - start, 1)
        self.stats["shaders"] = dict(self.materials.shaders)
        return self.stats

    # Entities ------------------------------------------------------------------------------------------------------------

    def _read_entities(self):
        entities = []
        seen = set()

        def load(path):
            if path in seen:
                return
            seen.add(path)
            data = self.content.try_read(path + "_c")
            if data is None:
                self.warn(f"entity lump '{path}' is missing")
                return
            lump = vworld.EntityLump(data)
            entities.extend(lump.entities)
            for child in lump.child_lumps:
                load(child)

        for lump in self.world.entity_lumps:
            load(lump)
        self.stats["entities"] = len(entities)
        return entities

    def _first(self, classname, predicate=None):
        for e in self.entities:
            if e.classname == classname and not e.get_bool("startdisabled") and (predicate is None or predicate(e)):
                return e
        return None

    # Lighting resources ----------------------------------------------------------------------------------------------

    def _setup_lighting(self):
        info = self.world.lighting
        images = {"dc": None, "lum": None, "brdf": None, "shadows": None}
        if info and info.get("m_bHasLightmaps"):
            for path in info.get("m_lightMaps") or []:
                stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if stem == "directional_irradiance_sh2_dc":
                    images["dc"] = self.textures.image(path, full_size=True)
                elif stem == "directional_irradiance_sh2_lum":
                    images["lum"] = self.textures.image(path, full_size=True)
                elif stem == "direct_light_shadows":
                    images["shadows"] = self.textures.image(path, full_size=True)
                elif stem == "directional_irradiance_sh2_r" and images["lum"] is None:
                    # Older maps keep a direction per colour; the red one stands in for the shared one.
                    images["lum"] = self.textures.image(path, full_size=True)
            for key in ("dc", "lum"):
                if images[key] is not None:
                    images[key].use_half_precision = True
            if images["dc"] is None:
                self.warn("the map's lightmaps aren't second order harmonics; world geometry falls back to the probes")
        scale = (info or {}).get("m_vLightmapUvScale") or [1.0, 1.0]
        self.lightmap_scale = (float(scale[0]), float(scale[1]))
        images["brdf"] = self.textures.image("textures/dev/brdf_lut.vtex", image=1, full_size=True)

        # Sky and fog.
        sun = self._first("light_environment")
        self.sun = sun
        sky_entity = self._first("env_sky")
        self.sky_image, self.sky_exposure, self.fog_sky_image = None, 1.0, None
        if sky_entity is not None:
            self._setup_sky(sky_entity)

        fog = lighting.FogSettings()
        gradient = self._first("env_gradient_fog")
        if gradient is not None:
            fog.gradient = lighting.gradient_fog_settings(gradient, lighting.srgb_to_linear)
        cube = self._first("env_cubemap_fog")
        if cube is not None:
            exposure = self.sky_exposure
            source = cube.get_int("cubemapfogsource")
            image = self.fog_sky_image
            if source == 2 and cube.get_str("cubemapfogskymaterial"):
                image, exposure = self._sky_images(cube.get_str("cubemapfogskymaterial"), fog_only=True)
            fog.cubemap = lighting.cubemap_fog_settings(cube, exposure)
            fog.cubemap_image = image
        self.fog_settings = fog

        self.fog_group = lighting.build_fog_group(self.map, fog, self.scale)
        self.suns = self._stationary_suns() if images["dc"] is not None else []
        self.lighting_group = lighting.build_lighting_group(self.map, images, self.scale, self.fog_group, self.suns)
        unlit = lighting.build_unlit_group(self.map, self.fog_group)
        additive = lighting.build_additive_group(self.map, self.fog_group)
        self.post_entity = self._first("post_processing_volume", lambda e: e.get_bool("master")) or self._first("post_processing_volume")
        npr = lighting.build_npr_group(self.map, self.fog_group, post.exposure_scale(self.content, self.post_entity))
        self.materials = matlib.MaterialContext(self.content, self.textures, self.map, self.lighting_group, unlit,
                                                additive, self.scale, npr)
        self.lighting_images = images

    def _stationary_suns(self) -> list:
        """The stationary directional lights (the light_environment, parallel light_barn lights) as the lighting group
        lights them on lightmapped surfaces: towards the light, linear colour times brightness, their channel of the
        baked shadows (bakedshadowindex) and a barn's box. Dynamic ones have no baked shadows; Blender lights them."""
        suns = []
        for e in self.entities:
            parallel_barn = e.classname == "light_barn" and e.get_vector("size_params", 3, [0, 0, 1])[2] <= 0
            if e.classname != "light_environment" and not parallel_barn:
                continue
            index = e.get_int("bakedshadowindex", e.get_int("bakelightindex", -1))
            if not lights.is_realtime(e) or not 0 <= index < 4 or not e.get_bool("enabled", True) or e.get_bool("startdisabled"):
                continue
            r = probelib.rotation_matrix(e.get_vector("angles", 3, [0, 0, 0]))
            forward = r[:, 0]
            if parallel_barn:
                color, box = lights.barn_box(e)
                label = e.targetname or "Moon barn"
            else:
                brightness = e.get_float("brightness", 1.0) * e.get_float("brightnessscale", 1.0)
                color = [lighting.srgb_to_linear(c / 255.0) * brightness for c in e.get_vector("color", 3, [255, 255, 255])]
                box, label = None, "Sun"
            if max(color) <= 0:
                continue
            e.values["dl_in_shader"] = True
            suns.append({"direction": tuple(-forward), "color": tuple(color), "label": label, "box": box,
                         "mask": tuple(1.0 if i == index else 0.0 for i in range(4))})
        return suns

    def _sky_images(self, material_path: str, fog_only: bool = False):
        m = self.materials.compiled(material_path) if hasattr(self, "materials") else None
        if m is None:
            from ..core import material as vmat
            data = self.content.try_read(compiled(material_path))
            m = vmat.CompiledMaterial(data) if data else None
        if m is None or not m.texture("g_tSkyTexture"):
            return None, 1.0
        exposure = 2.0 ** (m.float("g_flBrightnessExposureBias") + m.float("g_flRenderOnlyExposureBias"))
        tex, data = self.textures.compiled_texture(m.texture("g_tSkyTexture"))
        if tex is None or not tex.is_cube:
            return None, exposure
        key = f"{normalize(m.texture('g_tSkyTexture')).rsplit('/', 1)[-1]}"

        def faces(mip):
            return [_decode_dds(self.textures, tex.to_dds(mip, f), f"sky{f}") for f in range(6)]

        fog_mip = max(0, min(tex.mip_count - 1, int(math.log2(max(tex.width, 1))) - 3))
        fog_faces = faces(fog_mip)
        fog_image = _float_image(f"{key} fog", cubelib.to_equirect(fog_faces, 128), self.textures, f"{key}_fog")
        if fog_only:
            return fog_image, exposure
        sky_faces = faces(0)
        width = min(4096, max(1024, tex.width * 4))
        sky_image = _float_image(key, cubelib.to_equirect(sky_faces, width), self.textures, f"{key}_sky_{width}")
        self.fog_sky_image = fog_image
        return sky_image, exposure

    def _setup_sky(self, sky_entity):
        try:
            image, exposure = self._sky_images(sky_entity.get_str("skyname") or "")
        except Exception as error:
            self.warn(f"sky failed: {error}")
            return
        tint = [c / 255.0 * sky_entity.get_float("brightnessscale", 1.0) for c in sky_entity.get_vector("tint_color", 3, [255, 255, 255])]
        self.sky_image = image
        self.sky_exposure = exposure
        self.sky_tint = tint

    # World geometry ----------------------------------------------------------------------------------------------------

    def model(self, path: str) -> vmdl.CompiledModel | None:
        key = normalize(path)
        if key not in self._model_cache:
            data = self.content.try_read(compiled(path)) if path else None
            try:
                self._model_cache[key] = vmdl.CompiledModel(data) if data else None
            except Exception as error:
                self.warn(f"model '{path}' failed: {error}")
                self._model_cache[key] = None
        return self._model_cache[key]

    def uv_scale(self, material: str) -> UvScale:
        if material not in self._uv_scales:
            self._uv_scales[material] = UvScale(self.materials.compiled(material))
        return self._uv_scales[material]

    def _import_world(self):
        nodes = []
        for prefix in self.world.node_prefixes:
            data = self.content.try_read(prefix + ".vwnod_c")
            if data is None:
                self.warn(f"world node '{prefix}' is missing")
                continue
            nodes.append(vworld.WorldNode(data))

        aggregates = [a for n in nodes for a in n.aggregates]
        total = max(len(aggregates), 1)
        vertices = 0
        for index, aggregate in enumerate(aggregates):
            self.progress(0.1 + 0.4 * index / total, f"World geometry {index}/{total}")
            try:
                vertices += self._import_aggregate(aggregate)
            except Exception as error:
                self.warn(f"aggregate '{aggregate.model}' failed: {error}")
        self.stats["aggregates"] = len(aggregates)
        self.stats["world_vertices"] = vertices

        placed = 0
        objects_total = max(sum(len(n.scene_objects) for n in nodes), 1)
        done = 0
        for node in nodes:
            overrides = {}
            for scene_object, mesh_index, draw_call, material in node.material_overrides:
                overrides.setdefault(scene_object, []).append((mesh_index, draw_call, material))
            for o in node.scene_objects:
                self.progress(0.5 + 0.1 * done / objects_total, f"Placed models {done}/{objects_total}")
                done += 1
                if self._place_scene_object(o, tuple(overrides.get(o.index, ()))):
                    placed += 1
        self.stats["scene_objects"] = placed

    def _import_aggregate(self, aggregate) -> int:
        try:
            return self._build_aggregate(aggregate)
        finally:
            # Each aggregate has a model of its own: its decoded data isn't needed again (gigabytes on dl_midtown).
            self._model_cache.pop(normalize(aggregate.model), None)

    def _instance_key(self, fragment, call):
        """What a placed fragment's geometry depends on besides its transform, when it can be a linked copy of a
        shared mesh: not on the lightmaps (their coordinates are per placement), not a decal (decals take the
        lightmap under them), its draw call, tint and (for materials scaling texture coordinates by model scale)
        axis scales. None when it can't."""
        if fragment.matrix is None or call.lightmapped:
            return None
        compiled_material = self.materials.compiled(call.material)
        if compiled_material is not None and compiled_material.shader == "citadel_overlay.vfx":
            return None
        scales = None
        if self.uv_scale(call.material).any:
            scales = tuple(np.round(np.linalg.norm(np.asarray(fragment.matrix)[:3, :3], axis=0), 4))
        return fragment.draw_call, tuple(fragment.tint), scales

    def _build_aggregate(self, aggregate) -> int:
        compiled_model = self.model(aggregate.model)
        if compiled_model is None or not compiled_model.meshes:
            return 0
        mesh = compiled_model.meshes[0]
        calls = mesh.draw_calls()
        parts = {}
        geometry = Geometry()
        has_lods = False

        def drawn(fragment):
            if not self.options.detail_levels and aggregate.lod_setups and fragment.lod_mask and not fragment.lod_mask & 1:
                return None  # only the most detailed level
            if not 0 <= fragment.draw_call < len(calls):
                return None
            call = calls[fragment.draw_call]
            if not call.is_triangles or call.index_count < 3 or not call.vertex_buffers:
                return None
            return call

        # Fragments placing the same geometry several times become linked copies of one mesh (props: trees, lamps,
        # crates) rather than copies of it. Levels of detail switch per vertex, so with them everything is merged.
        keys = {}
        if not self.options.detail_levels:
            for index, fragment in enumerate(aggregate.fragments):
                call = drawn(fragment)
                key = self._instance_key(fragment, call) if call is not None else None
                if key is not None:
                    keys[index] = key
        counts = {}
        for key in keys.values():
            counts[key] = counts.get(key, 0) + 1
        instances = {}

        for index, fragment in enumerate(aggregate.fragments):
            call = drawn(fragment)
            if call is None:
                continue
            key = keys.get(index)
            if key is not None and counts[key] > 1:
                instances.setdefault(key, []).append(fragment)
                continue
            part = parts.get(fragment.draw_call)
            if part is None:
                part = parts[fragment.draw_call] = vmdl.read_part(mesh, call)
            mode = matlib.vertex_color_mode(self.materials.compiled(part.material))
            lod_range = self._lod_range(aggregate, fragment) if self.options.detail_levels else None
            has_lods = has_lods or lod_range is not None
            if fragment.matrix is None:
                geometry.add(part, part.material, mode, tint=fragment.tint, lightmap_scale=self.lightmap_scale,
                             lod_range=lod_range)
            else:
                positions, normals, uv0, uv1, flip = place_fragment(part, fragment.matrix, self.uv_scale(part.material))
                geometry.add(part, part.material, mode, positions, normals, uv0, uv1, fragment.tint, flip,
                             self.lightmap_scale, lod_range)
        name = aggregate.model.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        placed = self._place_instances(aggregate, name, mesh, calls, parts, instances)
        if not geometry:
            return placed
        me = geometry.build(name, self.materials.material, self.scale)
        obj = bpy.data.objects.new(name, me)
        obj["dl_model"] = aggregate.model
        if aggregate.all_flags & vworld.FLAG_NO_SHADOWS or _no_shadow_materials(me):
            obj.visible_shadow = False
        if has_lods:
            # Hidden levels of detail must not shadow either.
            for material in me.materials:
                if material is not None:
                    material.use_transparent_shadow = True
        self.collections["World"].objects.link(obj)
        return geometry.count + placed

    def _place_instances(self, aggregate, name, mesh, calls, parts, instances) -> int:
        """Linked copies of one mesh per repeated fragment geometry, lit as props (their own light probes and the
        realtime lights). Returns the vertices placed."""
        placed = 0
        for (draw_call, tint, scales), fragments in instances.items():
            part = parts.get(draw_call)
            if part is None:
                part = parts[draw_call] = vmdl.read_part(mesh, calls[draw_call])
            mode = matlib.vertex_color_mode(self.materials.compiled(part.material))
            uv0, uv1 = scaled_uvs(part, np.asarray(fragments[0].matrix)[:3, :3], self.uv_scale(part.material))
            geometry = Geometry()
            geometry.add(part, part.material, mode, uv0=uv0, uv1=uv1, tint=tint, lightmap_scale=self.lightmap_scale)
            if not geometry:
                continue
            shared = geometry.build(f"{name}_{draw_call}", self.materials.material, self.scale)
            for fragment in fragments:
                obj = self._object(shared, shared.name, np.asarray(fragment.matrix), self.collections["Props"], None,
                                   bool(fragment.flags & vworld.FLAG_NO_SHADOWS))
                obj["dl_model"] = aggregate.model
                placed += geometry.count
            self.stats["instanced_fragments"] = self.stats.get("instanced_fragments", 0) + len(fragments)
        return placed

    def _lod_range(self, aggregate, fragment):
        """The camera distances (metres) between which the game draws a fragment's levels of detail, or None when it
        always does. Each level runs from its switch distance to the next one's, scaled by the setup's object
        scale (trees switch to their round canopy blob 200 units out)."""
        if not fragment.lod_mask or not 0 <= fragment.lod_setup < len(aggregate.lod_setups):
            return None
        _, object_scale, switches = aggregate.lod_setups[fragment.lod_setup]
        levels = [i for i in range(32) if fragment.lod_mask >> i & 1]
        if not switches or not levels:
            return None
        near = switches[levels[0]] if levels[0] < len(switches) else switches[-1]
        far = switches[levels[-1] + 1] if levels[-1] + 1 < len(switches) else math.inf
        if near <= 0 and far == math.inf:
            return None
        far = far * object_scale * self.scale if far != math.inf else 1e9
        return (near * object_scale * self.scale, far)

    # Placed models -----------------------------------------------------------------------------------------------------

    def model_mesh(self, path: str, skin: str = "", overrides: tuple = ()) -> bpy.types.Mesh | None:
        """A model's default meshes at its most detailed LOD, with a skin and draw call material overrides."""
        key = (normalize(path), (skin or "").lower(), overrides)
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        result = None
        compiled_model = self.model(path)
        if compiled_model is not None:
            try:
                result = self._build_model_mesh(compiled_model, path, skin, overrides)
            except Exception as error:
                self.warn(f"model '{path}' failed: {error}")
            # The mesh is kept; the decoded model is read again only for another skin or override.
            self._model_known[normalize(path)] = True
            self._model_cache.pop(normalize(path), None)
        self._mesh_cache[key] = result
        return result

    def has_model(self, path: str) -> bool:
        key = normalize(path)
        if key not in self._model_known:
            self._model_known[key] = self.model(path) is not None
        return self._model_known[key]

    def _build_model_mesh(self, m: vmdl.CompiledModel, path: str, skin: str, overrides: tuple):
        remap = {}
        if skin and m.material_groups:
            default = m.material_groups[0][1]
            chosen = None
            if skin.isdigit() and int(skin) < len(m.material_groups):
                chosen = m.material_groups[int(skin)][1]
            else:
                for name, mats in m.material_groups:
                    if name.lower() == skin.lower():
                        chosen = mats
            if chosen:
                remap = {normalize(a): b for a, b in zip(default, chosen)}
        whole = next((mat for mesh_index, call, mat in overrides if call == 0xFFFFFFFF), None)
        per_call = {(mesh_index, call): mat for mesh_index, call, mat in overrides if call != 0xFFFFFFFF}

        geometry = Geometry()
        for part in m.default_parts():
            material = part.material
            material = remap.get(normalize(material), material)
            material = per_call.get((part.mesh_index, part.draw_call), material)
            if whole:
                material = whole
            mode = matlib.vertex_color_mode(self.materials.compiled(material))
            geometry.add(part, material, mode, lightmap_scale=self.lightmap_scale)
        if not geometry:
            return None
        name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        me = geometry.build(name, self.materials.material, self.scale)
        me["dl_model"] = normalize(path)
        return me

    def _object(self, mesh, name, matrix: np.ndarray, collection, tint=None, no_shadows=False):
        obj = bpy.data.objects.new(name, mesh)
        m = Matrix([list(matrix[0]), list(matrix[1]), list(matrix[2]), [0, 0, 0, 1]])
        m.translation = (matrix[0][3] * self.scale, matrix[1][3] * self.scale, matrix[2][3] * self.scale)
        obj.matrix_world = m
        if tint is not None:
            obj.color = tint
        if no_shadows or (mesh is not None and _no_shadow_materials(mesh)):
            obj.visible_shadow = False
        collection.objects.link(obj)
        return obj

    def _place_scene_object(self, o, overrides) -> bool:
        if not o.model:
            return False
        mesh = self.model_mesh(o.model, o.skin, overrides)
        if mesh is None:
            return False
        tint = None
        if len(o.tint) >= 4 and o.tint[3] != 0:
            tint = tuple(lighting.srgb_to_linear(c) for c in o.tint[:3]) + (o.tint[3],)
        name = o.model.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        # Placed models baked into the lightmaps are lit like the world geometry.
        collection = self.collections["World"] if "dl_lightmap" in mesh.attributes else self.collections["Props"]
        obj = self._object(mesh, name, o.matrix, collection, tint, bool(o.flags & vworld.FLAG_NO_SHADOWS))
        obj["dl_model"] = o.model
        origin = o.lighting_origin or list(o.matrix[:3, 3])
        obj["dl_lighting_origin"] = origin
        return True

    def _entity_matrix(self, e) -> np.ndarray:
        origin = e.get_vector("origin", 3, [0, 0, 0])
        angles = e.get_vector("angles", 3, [0, 0, 0])
        scales = e.get_vector("scales", 3, [1, 1, 1])
        scales = [s if s != 0 else 1.0 for s in scales]
        r = probelib.rotation_matrix(angles)
        m = np.eye(4)
        m[:3, :3] = r * np.array(scales)[None, :]
        m[:3, 3] = origin
        return m

    def _import_entity_props(self):
        placed = 0
        missing = set()
        for index, e in enumerate(self.entities):
            self.progress(0.6 + 0.2 * index / max(len(self.entities), 1), f"Props {index}/{len(self.entities)}")
            cls = e.classname
            path = e.get_str("model")
            if not path or cls.startswith(_UNDRAWN_PREFIXES):
                continue
            if e.get_bool("startdisabled") or (e.get_str("rendermode") or "").lower() == "krendernone":
                continue
            if cls == "func_brush" and e.targetname.lower().endswith("spawn_block_brush"):
                continue
            if not self.has_model(path):
                missing.add(path)
                continue
            mesh = self.model_mesh(path, e.get_str("skin") or "")
            if mesh is None:
                continue
            color = e.get_vector("rendercolor", 3, [255, 255, 255])
            alpha = e.get_float("renderamt", 255.0) / 255.0
            tint = None
            if color != [255, 255, 255] or alpha != 1.0:
                tint = tuple(lighting.srgb_to_linear(c / 255.0) for c in color) + (alpha,)
            matrix = self._entity_matrix(e)
            name = e.targetname or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            obj = self._object(mesh, name, matrix, self.collections["Props"], tint, e.get_bool("disableshadows"))
            obj["dl_class"] = cls
            obj["dl_model"] = path
            placed += 1
        self.stats["entity_props"] = placed
        if missing:
            self.warn(f"{len(missing)} entity models aren't in the game's content")

    # Lights, markers, reflection probes --------------------------------------------------------------------------------

    def _cookie(self, path):
        return self.textures.image(path, color=True)

    def _import_lights(self):
        made = baked = 0
        collection = self.collections["Lights"]
        for e in self.entities:
            if not e.classname.startswith("light_"):
                continue
            if not lights.is_realtime(e):
                baked += 1
                continue
            try:
                obj = lights.create_light(e, self.scale, collection, self._cookie)
                if obj is not None:
                    made += 1
                    if e.get("dl_in_shader"):
                        obj.light_linking.receiver_collection = self.collections["Props"]
                        obj["dl_note"] = "Lights the props; world geometry gets it through its baked shadows (DL Lighting)"

            except Exception as error:
                self.warn(f"{e.classname} '{e.targetname}' failed: {error}")
        self.stats["lights"] = made
        self.stats["baked_lights"] = baked

    def _import_markers(self):
        collection = self.collections["Markers"]
        count = 0
        for e in self.entities:
            if e.classname != "info_particle_system":
                continue
            obj = bpy.data.objects.new(e.targetname or e.get_str("effect_name", "particles").rsplit("/", 1)[-1], None)
            obj.empty_display_type = "SPHERE"
            obj.empty_display_size = 8 * self.scale
            m = self._entity_matrix(e)
            mm = Matrix([list(m[0]), list(m[1]), list(m[2]), [0, 0, 0, 1]])
            mm.translation = tuple(m[:3, 3] * self.scale)
            obj.matrix_world = mm
            obj["dl_effect"] = e.get_str("effect_name", "")
            obj["dl_start_active"] = e.get_bool("start_active")
            collection.objects.link(obj)
            count += 1
        self.stats["particle_systems"] = count

    def _import_reflection_probes(self):
        """EEVEE sphere probes where the game has its cubemap volumes, with their parallax boxes, so EEVEE's
        reflections come from the same places as the game's cubemaps."""
        collection = self.collections["Reflection Probes"]
        for e in self.entities:
            if e.classname not in ("env_combined_light_probe_volume", "env_cubemap_box", "env_cubemap"):
                continue
            if not e.get_str("cubemaptexture"):
                continue
            probe = bpy.data.lightprobes.new(e.targetname or e.classname, "SPHERE")
            obj = bpy.data.objects.new(probe.name, probe)
            m = self._entity_matrix(e)
            if e.classname == "env_cubemap":
                radius = e.get_float("influenceradius", 512.0)
                probe.influence_type = "ELIPSOID"
                probe.influence_distance = radius * self.scale
                center = np.zeros(3)
                half = np.array([radius, radius, radius])
            else:
                mins = np.array(e.get_vector("box_mins", 3, [-64, -64, -64]))
                maxs = np.array(e.get_vector("box_maxs", 3, [64, 64, 64]))
                center = (mins + maxs) * 0.5
                half = np.maximum((maxs - mins) * 0.5, 1.0)
                probe.influence_type = "BOX"
                probe.influence_distance = 1.0
                fades = e.get_vector("edge_fade_dists", 3, [0, 0, 0])
                probe.falloff = max(0.0, min(1.0, max(fades) / max(half.min(), 1.0)))
                probe.use_custom_parallax = True
                probe.parallax_type = "BOX"
                probe.parallax_distance = 1.0
            mm = Matrix([list(m[0]), list(m[1]), list(m[2]), [0, 0, 0, 1]])
            world_center = m[:3, :3] @ center + m[:3, 3]
            mm.translation = tuple(world_center * self.scale)
            obj.matrix_world = mm
            obj.scale = tuple(half * self.scale)
            collection.objects.link(obj)

    def _lift_overlays(self):
        """Lifts citadel_overlay decals (signs, posters, graffiti, road paint) off the surfaces they lie on: they're
        modelled exactly on them, which the game draws with a depth bias and Blender would z-fight."""
        lift = OVERLAY_LIFT * self.scale
        meshes = {obj.data for c in (self.collections["World"], self.collections["Props"]) for obj in c.objects
                  if obj.type == "MESH"}
        for me in meshes:
            overlay = [i for i, m in enumerate(me.materials) if m is not None and m.get("dl_shader") == "citadel_overlay.vfx"]
            if not overlay or me.get("dl_lifted") or not len(me.polygons):
                continue
            material_index = np.empty(len(me.polygons), np.int32)
            me.polygons.foreach_get("material_index", material_index)
            selected = np.isin(material_index, overlay)
            loop_total = np.empty(len(me.polygons), np.int32)
            me.polygons.foreach_get("loop_total", loop_total)
            loop_vertices = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get("vertex_index", loop_vertices)
            vertices = np.unique(loop_vertices[np.repeat(selected, loop_total)])
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            normals = np.empty(len(me.vertices) * 3, np.float32)
            me.vertex_normals.foreach_get("vector", normals)
            co, normals = co.reshape(-1, 3), normals.reshape(-1, 3)
            co[vertices] += normals[vertices] * lift
            me.vertices.foreach_set("co", co.ravel())
            me["dl_lifted"] = True
            me.update()

    def _light_overlays(self):
        """Gives decals the lightmap coordinates of the surface under them. The game blends citadel_overlay into the
        G-buffer, so the deferred pass lights it with that surface's lightmap; it has no lighting of its own. Each
        decal face takes the nearest lightmapped triangle's mapping (extended across the face, so it stays on one
        chart) and the object joins the world geometry.

        Only the world triangles near decals are searched, gathered with numpy into one BVH: querying the merged
        world meshes themselves makes Blender build a search tree over each of them (gigabytes on dl_midtown)."""
        from mathutils.bvhtree import BVHTree
        from mathutils.geometry import barycentric_transform
        from mathutils import Vector
        reach = (OVERLAY_LIFT + 4.0) * self.scale
        cell = 256.0 * self.scale

        # The decals: world positions, face centres, corners.
        decals = []
        for collection in (self.collections["World"], self.collections["Props"]):
            for obj in list(collection.objects):
                me = obj.data if obj.type == "MESH" else None
                if me is None or "dl_lightmap" in me.attributes or me.users > 1 or not len(me.polygons):
                    continue
                if not any(m is not None and m.get("dl_shader") == "citadel_overlay.vfx" for m in me.materials):
                    continue
                m = np.array(obj.matrix_world, np.float64)
                co = np.empty(len(me.vertices) * 3, np.float64)
                me.vertices.foreach_get("co", co)
                co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
                centres = np.empty(len(me.polygons) * 3, np.float64)
                me.polygons.foreach_get("center", centres)
                centres = centres.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
                decals.append((collection, obj, me, co, centres))
        if not decals:
            return

        # Grid cells around the decal faces.
        all_centres = np.concatenate([d[4] for d in decals])
        keys = set()
        for offset in ((-reach, -reach, -reach), (reach, reach, reach), (-reach, reach, 0), (reach, -reach, 0)):
            for key in map(tuple, np.floor((all_centres + offset) / cell).astype(np.int64)):
                keys.add(key)

        def packed(c):
            # One int64 per cell (cells within +-65536 per axis, far beyond any map).
            return (c[..., 0] + 2 ** 16) * 2 ** 36 + (c[..., 1] + 2 ** 16) * 2 ** 18 + (c[..., 2] + 2 ** 16)

        wanted = np.array(sorted(packed(np.array(list(keys), np.int64))), np.int64)

        # The lightmapped world triangles in those cells.
        positions, lightmaps = [], []
        for obj in self.collections["World"].objects:
            me = obj.data if obj.type == "MESH" else None
            if me is None:
                continue
            attr = me.attributes.get("dl_lightmap")
            if attr is None or attr.domain != "POINT" or not len(me.polygons):
                continue
            m = np.array(obj.matrix_world, np.float64)
            co = np.empty(len(me.vertices) * 3, np.float64)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            uv = np.empty(len(me.vertices) * 2, np.float32)
            attr.data.foreach_get("vector", uv)
            uv = uv.reshape(-1, 2)
            me.calc_loop_triangles()
            tris = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tris)
            tris = tris.reshape(-1, 3)
            corners = co[tris]
            low = np.floor(corners.min(1) / cell).astype(np.int64)
            high = np.floor(corners.max(1) / cell).astype(np.int64)
            big = (high - low > 1).any(1)  # spans several cells: kept without the test (rare, a few large floors)
            near = big.copy()
            for pick in range(8):
                c = np.where([(pick >> axis) & 1 for axis in range(3)], high, low)
                near |= np.isin(packed(c), wanted)
            if near.any():
                positions.append(corners[near].reshape(-1, 3))
                lightmaps.append(uv[tris[near]].reshape(-1, 2))
        if not positions:
            return
        positions = np.concatenate(positions)
        lightmaps = np.concatenate(lightmaps)
        bvh = BVHTree.FromPolygons(positions.tolist(), np.arange(len(positions)).reshape(-1, 3).tolist())

        lit = 0
        for collection, obj, me, co, centres in decals:
            loop_vertex = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get("vertex_index", loop_vertex)
            loop_start = np.empty(len(me.polygons), np.int32)
            me.polygons.foreach_get("loop_start", loop_start)
            loop_total = np.empty(len(me.polygons), np.int32)
            me.polygons.foreach_get("loop_total", loop_total)
            corner_uv = np.zeros((len(me.loops), 2), np.float32)
            found = 0
            for face, centre in enumerate(centres):
                hit = bvh.find_nearest(Vector(centre), reach)
                if hit[2] is None:
                    continue
                t = hit[2] * 3
                a, b, c = (Vector(positions[t + k]) for k in range(3))
                ua, ub, uc = (Vector((*lightmaps[t + k], 0.0)) for k in range(3))
                if min(ua.x, ub.x, uc.x) < 0.5:
                    continue  # off the lightmaps there
                for loop in range(loop_start[face], loop_start[face] + loop_total[face]):
                    uv = barycentric_transform(Vector(co[loop_vertex[loop]]), a, b, c, ua, ub, uc)
                    corner_uv[loop] = (uv.x, uv.y)
                found += 1
            if not found:
                continue
            me.attributes.new("dl_lightmap", "FLOAT2", "CORNER").data.foreach_set("vector", corner_uv.ravel())
            if collection is not self.collections["World"]:
                collection.objects.unlink(obj)
                self.collections["World"].objects.link(obj)
            lit += 1
        self.stats["overlays_on_lightmaps"] = lit

    def _add_outlines(self):
        """pbr.vfx's NPR outline pass as an inverted hull: the mesh pushed out along its normals by the outline
        thickness (game units), only its back faces drawn, in the game's dark outline colour or the material's solid
        outline tint."""
        outlined = {}
        for obj in self.collections["Props"].objects:
            if obj.type != "MESH":
                continue
            thickness = outlined.get(obj.data.name)
            if thickness is None:
                values = [m["dl_npr_outline"] for m in obj.data.materials if m is not None and m.get("dl_npr_outline")]
                thickness = outlined[obj.data.name] = max(values) if values else 0.0
                if thickness and not any(m.get("dl_outline_shell") for m in obj.data.materials if m):
                    tints = [m.get("dl_npr_outline_tint") for m in obj.data.materials if m is not None and m.get("dl_npr_outline_tint")]
                    obj.data.materials.append(self._outline_material(tuple(tints[0]) if tints else None))
            if not thickness:
                continue
            modifier = obj.modifiers.new("NPR Outline", "SOLIDIFY")
            modifier.thickness = -thickness * self.scale
            modifier.offset = -1.0
            modifier.use_flip_normals = True
            modifier.use_rim = False
            modifier.material_offset = 32767
            modifier.use_even_offset = False

    def _outline_material(self, tint=None):
        key = f"DL NPR Outline {tint}" if tint else "DL NPR Outline"
        mat = bpy.data.materials.get(key)
        if mat is not None:
            return mat
        mat = bpy.data.materials.new(key)
        mat["dl_outline_shell"] = True
        mat.use_backface_culling = True
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()
        from .nodes import Builder
        b = Builder(tree)
        color = tint if tint else (0.003035, 0.0003035, 0.0003035)
        # The hull's normals are flipped: its near side faces away and is dropped (Cycles ignores backface culling).
        geometry = b.node("ShaderNodeNewGeometry")
        node = b.group(self.materials.unlit, Color=color, Alpha=b.sub(1.0, geometry.outputs["Backfacing"]))
        b.column(2)
        out = b.node("ShaderNodeOutputMaterial")
        out.hide = False
        b.set(out.inputs["Surface"], node.outputs[0])
        return mat

    def _import_particles(self):
        try:
            from .particles import build_particle_systems
        except ImportError:
            return
        try:
            self.stats["particles_built"] = build_particle_systems(self, self.options.particle_time)
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.warn(f"particles failed: {error}")

    # Probes ------------------------------------------------------------------------------------------------------------

    def _apply_probes(self):
        try:
            volumes = probelib.read_probe_volumes(self.content, self.entities,
                                                  lambda dds: _decode_dds(self.textures, dds, "probe"), self.warn)
        except Exception as error:
            self.warn(f"light probes failed: {error}")
            volumes = []
        constant, vertical = probelib.sky_ambient(self.sun, lighting.srgb_to_linear)
        field = probelib.ProbeField(volumes, constant, vertical)
        self.probe_field = field
        self.stats["probe_volumes"] = len(volumes)
        names = ("px", "py", "pz", "nx", "ny", "nz")
        for obj in self.collections["Props"].objects:
            origin = obj.get("dl_lighting_origin")
            if origin is None:
                corners = [obj.matrix_world @ v.co for v in obj.data.vertices[:1]] if obj.type == "MESH" and obj.data.vertices else []
                if obj.type == "MESH" and len(obj.data.vertices):
                    from mathutils import Vector
                    box = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
                    center = sum(box, Vector()) / 8.0
                    origin = [c / self.scale for c in center]
                else:
                    origin = [c / self.scale for c in obj.matrix_world.translation]
            cube = field.ambient_cube(origin)
            for i, name in enumerate(names):
                obj[f"dl_amb_{name}"] = [float(x) for x in cube[i]]
        # World geometry off the lightmaps (trees and other merged foliage, spanning the map) gets the probes and
        # their baked shadows per vertex, as the game samples them per pixel; the object's own cube (at its centre)
        # stays as the fallback.
        from mathutils import Vector
        sampled = 0
        world = list(self.collections["World"].objects)
        for index, obj in enumerate(world):
            self.progress(0.85 + 0.03 * index / max(len(world), 1), f"Light probes {index}/{len(world)}")
            if obj.type != "MESH":
                continue
            box = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            center = sum(box, Vector()) / 8.0
            cube = field.ambient_cube([c / self.scale for c in center])
            for i, name in enumerate(names):
                obj[f"dl_amb_{name}"] = [float(x) for x in cube[i]]
            sampled += self._probes_per_vertex(obj, field)
        self.stats["probe_lit_vertices"] = sampled

    def _probes_per_vertex(self, obj, field) -> int:
        """Bakes the probes into a mesh off the lightmaps: the ambient cube along each corner's normal (dl_vamb), the
        stationary lights' baked shadows per vertex (dl_vshd) and where they apply (dl_vprobe). One colour rather
        than six faces keeps materials within EEVEE's attribute limit."""
        me = obj.data
        if me.users > 1 or not len(me.vertices) or "dl_vamb" in me.attributes:
            return 0
        lightmap = me.attributes.get("dl_lightmap")
        if lightmap is None:
            off = np.ones(len(me.vertices), bool)
        elif lightmap.domain == "POINT":
            uv = np.empty(len(me.vertices) * 2, np.float32)
            lightmap.data.foreach_get("vector", uv)
            off = uv[0::2] < 0.5
        else:
            return 0  # decals on the lightmaps
        if not off.any():
            return 0
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(obj.matrix_world, np.float64)
        points = (co.reshape(-1, 3)[off] @ m[:3, :3].T + m[:3, 3]) / self.scale
        cubes, shadows = field.sample_many(points)
        all_cubes = np.zeros((len(me.vertices), 6, 3), np.float32)
        all_cubes[off] = cubes

        # Each corner's normal in the world, the cube evaluated along it as the lighting group would.
        loop_vertex = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_vertex)
        normals = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get("vector", normals)
        normals = normals.reshape(-1, 3) @ np.linalg.inv(m[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        corner_cubes = all_cubes[loop_vertex]
        ambient = np.zeros((len(me.loops), 3), np.float32)
        for axis in range(3):
            n = normals[:, axis:axis + 1]
            face = np.where(n > 0, corner_cubes[:, axis], corner_cubes[:, axis + 3])
            ambient += face * n * n
        values = np.zeros((len(me.loops), 4), np.float32)
        values[:, :3] = ambient
        values[:, 3] = 1.0
        me.attributes.new("dl_vamb", "FLOAT_COLOR", "CORNER").data.foreach_set("color", values.ravel())
        me.attributes.new("dl_vprobe", "FLOAT", "POINT").data.foreach_set("value", off.astype(np.float32))
        values = np.zeros((len(me.vertices), 4), np.float32)
        values[off] = shadows
        me.attributes.new("dl_vshd", "FLOAT_COLOR", "POINT").data.foreach_set("color", values.ravel())
        return int(off.sum())

    # Scene -------------------------------------------------------------------------------------------------------------

    def _setup_scene(self):
        scene = self.context.scene
        if self.sky_image is not None:
            world = lighting.sky_world(f"{self.map} sky", self.sky_image, self.sky_exposure, getattr(self, "sky_tint", (1, 1, 1)))
        else:
            world = bpy.data.worlds.new(f"{self.map} sky")
            world.use_nodes = True
            world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        world["dl_map"] = self.map
        scene.world = world
        if self.options.setup_render:
            post.setup_render(scene, self.scale)
        if self.options.post_processing:
            try:
                post.setup_post_processing(scene, self.content, self.post_entity)
            except Exception as error:
                self.warn(f"post processing failed: {error}")
        self._place_view(scene)

    def _place_view(self, scene):
        """Points the viewport and a camera at a spawn point, so the map is in view after import."""
        spawn = self._first("info_team_spawn") or self._first("info_player_start") or self._first("info_player_teamspawn")
        if spawn is None:
            return
        origin = np.array(spawn.get_vector("origin", 3, [0, 0, 0]))
        yaw = math.radians(spawn.get_vector("angles", 3, [0, 0, 0])[1])
        eye = (origin + np.array([0, 0, 64.0])) * self.scale
        cam_data = bpy.data.cameras.new(f"{self.map} camera")
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(90.0)
        cam_data.clip_start = 4 * self.scale
        cam_data.clip_end = 40000 * self.scale
        camera = bpy.data.objects.new(f"{self.map} camera", cam_data)
        camera.location = tuple(eye)
        camera.rotation_euler = (math.radians(90.0), 0.0, yaw - math.radians(90.0))
        self.root.objects.link(camera)
        scene.camera = camera
