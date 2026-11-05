"""A map's particle systems (info_particle_system) as Blender objects: each system simulated a moment into its life
(core.particles) and what its renderers draw then built as meshes, turned to face the scene's camera by Geometry
Nodes and shaded by a port of the game's spritecard shader.

    sprites   a point per particle; ``DL Particle Sprites`` puts a card on each (realized, so the material reads the
              particles' attributes) facing the active camera, or along the axes the particle stores
    ropes     a ribbon along the particles' spline, trails a card per particle along its motion; ``DL Particle
              Ribbons`` pushes each side of them out across the strip, in the plane facing the camera

Which systems are built: those set to start active, and inactive ones the map's own logic starts as it loads (a
logic_auto firing Start, directly or through a logic_relay). Systems that start disabled are left out, and so are
inactive ones the map never starts: game code plays those when something happens (in the hideout: the matchmaking
status effects, the logo reveal, a holiday reward's snow globe, the painting's glowing eyes).

Point attributes the meshes carry, read by the node groups and materials:
    dl_color                linear colour (lit: self illumination plus diffuse times the light probes at the system)
                            and alpha
    dl_rect<L>, _next       a sheet animated layer's frame on its texture (min u, min v, max u, max v, V down), and
                            the next frame's, blended by dl_frame_blend
    dl_offset<L>            a layer's per particle texture offset, for layers that randomize theirs
    dl_card                 where on the card a vertex is (u across, v down/along), written by the node groups for
                            sprites
    sprites:  dl_half_width, dl_roll, dl_yaw, dl_card_offset, dl_card_size (half widths), dl_right/dl_up (base axes
              for cards that don't face the camera)
    ribbons:  dl_axis (along the strip), dl_width (signed half width), dl_facing (the strip's plane normal when it
              doesn't face the camera)
"""
import time as _time
import zlib

import numpy as np

import bpy
from mathutils import Matrix

from ..core import particles as plib
from ..core import probes as probelib
from ..core.particles import renderers as R
from . import lights as lightlib
from .nodes import Builder

_LUMA = (0.2125, 0.7154, 0.0721)
_LUMA_601 = (0.299, 0.587, 0.114)

# Bumped when the node groups change, so files made by an older version get new ones.
_GROUP_VERSION = 1


def _srgb(c: float) -> float:
    """A gamma space colour component as linear."""
    c = min(max(float(c), 0.0), 1.0)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def build_particle_systems(importer, time: float = 2.0) -> int:
    """Builds every shown info_particle_system of the map as it looks ``time`` seconds after it starts. Returns how
    many systems were built."""
    started = _time.time()
    library = plib.library(importer.content)
    collection = importer.collections.get("Particles")
    if collection is None:
        collection = bpy.data.collections.new("Particles")
        importer.root.children.link(collection)
        importer.collections["Particles"] = collection
    target = _Target(importer.textures, importer.scale, importer.fog_group, collection, importer.warn)
    target.model_mesh = importer.model_mesh
    target.lit_collection = importer.collections.get("Props")

    by_name = {}
    for e in importer.entities:
        if e.targetname:
            by_name.setdefault(e.targetname.lower(), e)
    started_by_map = _started_at_spawn(importer.entities)
    lights = _ProbeLight(importer)
    target.probe_cube = lights.field.ambient_cube
    builder = _ObjectBuilder(target)

    built = left_off = failed = 0
    systems = sum(1 for e in importer.entities if e.classname == "info_particle_system")
    seen = 0
    for index, entity in enumerate(importer.entities):
        if entity.classname != "info_particle_system":
            continue
        if hasattr(importer, "progress"):
            importer.progress(0.9 + 0.05 * seen / max(systems, 1), f"Particles {seen}/{systems}")
        seen += 1
        path = entity.get_str("effect_name") or ""
        if not path or entity.get_bool("startdisabled"):
            continue
        definition = library.get(path)
        if definition is None:
            continue
        if not entity.get_bool("start_active", True) and entity.targetname.lower() not in started_by_map:
            left_off += 1
            continue
        try:
            points = _control_points(importer, entity, by_name)
            seed = zlib.crc32(f"{entity.get_str('hammeruniqueid') or index}".encode()) & 0x7FFFFFFF
            simulation = plib.simulate(definition, points, time, seed)
            holder = builder.build(entity.targetname or _stem(path), definition.path, simulation, points, time, seed,
                                   lights.at(points[0][0]))
            if holder is not None:
                holder["dl_start_active"] = entity.get_bool("start_active", True)
                built += 1
        except Exception as error:  # one broken effect must not lose the rest
            failed += 1
            importer.warn(f"particle system '{path}' failed: {error}")

    for what, count in sorted(library.unsupported.items()):
        importer.warn(f"particles: skipped {what} ({count})")
    for path in sorted(library.missing):
        importer.warn(f"particle system '{path}' is missing from the game's content")
    importer.stats["particle_systems_built"] = built
    importer.stats["particle_systems_left_off"] = left_off
    importer.stats["particle_seconds"] = round(_time.time() - started, 1)
    if failed:
        importer.stats["particle_systems_failed"] = failed
    return built


def spawn_particle_system(context, path: str, matrix_world: Matrix, time: float | None = 2.0, seed: int = 0):
    """Places a particle system from the game's content at a transform (Blender units) as it looks ``time`` seconds
    in (None: at the moment it draws the most), lit and fogged like the Deadlock map in the scene if there is one.

    It plays as the particle editor previews it: at the origin with the control points of its preview setup, and
    control points nothing sets reading 1 (the game code that sets scales and counts isn't running), then the empty
    places it. Returns the effect's empty (its "dl_time" the moment shown), or None when it draws nothing; raises
    LookupError when the game has no such system."""
    from .. import session
    target = _Target.for_scene(context)
    library = plib.library(session.content())
    definition = library.get(path)
    if definition is None:
        raise LookupError(f"'{path}' isn't one of the game's particle systems")
    points = plib.preview_control_points(definition)
    if time is None:
        time = plib.busiest_moment(definition, points, seed)
        if time is None:
            return None
    simulation = plib.simulate(definition, points, time, seed, unset_value=1.0)
    ambient = _scene_light(context, np.array(matrix_world.translation, np.float64) / target.scale, target.scale)
    holder = _ObjectBuilder(target).build(_stem(definition.path), definition.path, simulation, points, time, seed,
                                          ambient, local=True)
    if holder is not None:
        holder.matrix_world = matrix_world
    return holder


def unsupported_renderers(path: str) -> list[str]:
    """The renderers of an effect's tree the add-on doesn't draw (lights, projected decals, screen shake...)."""
    from .. import session
    library = plib.library(session.content())
    found, seen, queue = set(), set(), [path]
    while queue:
        definition = library.get(queue.pop())
        if definition is None or definition.path in seen:
            continue
        seen.add(definition.path)
        for block in definition.params.functions("m_Renderers"):
            kind = block.str("_class")
            if kind not in R.RENDERERS:
                found.add(kind.replace("C_OP_Render", ""))
        queue.extend(child.path for child in definition.children if child.path)
    return sorted(found)


def rebuild_particle_system(context, holder, time: float | None = None, seed: int | None = None) -> bool:
    """Builds an effect's particles again where its empty is now, at another moment (and seed) if given. Control
    points beyond the first move and turn with the empty. Returns whether anything was drawn."""
    from .. import session
    path = holder.get("dl_effect")
    if not path:
        return False
    time = float(holder.get("dl_time", 2.0)) if time is None else time
    seed = int(holder.get("dl_seed", 0)) if seed is None else seed
    target = _Target.for_scene(context, float(holder.get("dl_unit_scale", 0.0)) or None)
    if holder.users_collection:
        target.collection = holder.users_collection[0]
    library = plib.library(session.content())
    definition = library.get(path)
    if definition is None:
        return False
    if holder.get("dl_local"):
        # Spawned from the sidebar: played at the origin as previewed, the empty placing it.
        points = plib.preview_control_points(definition)
        simulation = plib.simulate(definition, points, time, seed, unset_value=1.0)
        ambient = _scene_light(context, np.array(holder.matrix_world.translation, np.float64) / target.scale,
                               target.scale)
        _clear_children(holder)
        _ObjectBuilder(target).build(holder.name, path, simulation, points, time, seed, ambient, holder, local=True)
        return bool(holder.children)
    rotation = np.array(holder.matrix_world.to_3x3().normalized(), np.float64)
    origin = np.array(holder.matrix_world.translation, np.float64) / target.scale
    points = {0: (origin, rotation)}
    for key, values in (holder.get("dl_control_points") or {}).items():
        values = [float(x) for x in values]
        position = origin + rotation @ np.array(values[0:3])
        points[int(key)] = (position, rotation @ np.array(values[3:12]).reshape(3, 3) if len(values) >= 12 else None)
    simulation = plib.simulate(definition, points, time, seed)
    built_at = np.array(holder.get("dl_origin", origin), np.float64)
    ambient = holder.get("dl_ambient")
    if ambient is None or np.linalg.norm(built_at - origin) > 1.0:
        ambient = _scene_light(context, origin, target.scale)

    _clear_children(holder)
    _ObjectBuilder(target).build(holder.name, path, simulation, points, time, seed, np.asarray(ambient), holder)
    return bool(holder.children)


def _clear_children(holder):
    for child in list(holder.children):
        data = child.data
        bpy.data.objects.remove(child)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)


def effect_paths(content) -> list[str]:
    """Every particle system the game ships (paths without the compiled suffix), sorted."""
    return sorted(path[:-2] for path in content.files(".vpcf_c"))


def _started_at_spawn(entities) -> set:
    """Names of the entities the map starts as it loads: the targets of a Start input fired by a logic_auto, or by a
    logic_relay a logic_auto triggers."""
    relays = {e.targetname.lower(): e for e in entities if e.classname == "logic_relay" and e.targetname}
    started = set()
    for e in entities:
        if e.classname != "logic_auto":
            continue
        for _, target, action, *_ in e.connections:
            target = (target or "").lower()
            if action.lower() == "start":
                started.add(target)
            elif action.lower() == "trigger" and target in relays:
                for _, relay_target, relay_action, *_ in relays[target].connections:
                    if relay_action.lower() == "start":
                        started.add((relay_target or "").lower())
    return started


def _control_points(importer, entity, by_name: dict) -> dict:
    """The system's control points in game units: CP0 is the entity (or what cpoint0 names), cpointN the entities
    they name (falling back to CP0), then the literal values data_cp, tint_cp and cpointN_value set."""
    def transform(e):
        m = importer._entity_matrix(e)
        rotation = m[:3, :3] / np.maximum(np.linalg.norm(m[:3, :3], axis=0), 1e-9)
        return m[:3, 3].copy(), rotation

    def named(name):
        if not name or (name.startswith("!") and name.lower() != "!self"):
            return None
        target = entity if name.lower() == "!self" else by_name.get(name.lower())
        return transform(target) if target is not None else None

    points = {0: named(entity.get_str("cpoint0") or "") or transform(entity)}
    for i in range(1, 64):
        name = entity.get_str(f"cpoint{i}") or ""
        if name:
            points[i] = named(name) or points[0]

    def literal(index, value):
        if 0 <= index < 64 and value is not None:
            rotation = points[index][1] if index in points else None
            points[index] = (np.array(value, np.float64), rotation)

    literal(entity.get_int("data_cp", -1), entity.get_vector("data_cp_value", 3))
    literal(entity.get_int("tint_cp", -1), entity.get_vector("tint_cp_color", 3, [255, 255, 255]))
    for i in range(64):
        if entity.get_str(f"cpoint{i}_value"):
            literal(i, entity.get_vector(f"cpoint{i}_value", 3))
    return points


# Light -------------------------------------------------------------------------------------------------------------
# The game lights a particle at one point, facing nowhere in particular: the light probes along the diagonal both ways
# (for an ambient cube, the mean of its six faces) plus the sun and the realtime light2 lights, the stationary ones
# behind the probes' baked shadows.

class _ProbeLight:
    """The map's light probes (read again unless the importer kept its probe field) and its realtime lights."""

    def __init__(self, importer):
        self.field = getattr(importer, "probe_field", None)
        if self.field is None:
            self.field = self._read(importer)
        self.lights = [e for e in importer.entities if e.classname.startswith("light_") and lightlib.is_realtime(e)]

    @staticmethod
    def _read(importer):
        try:
            from .map_import import _decode_dds
            volumes = probelib.read_probe_volumes(importer.content, importer.entities,
                                                  lambda dds: _decode_dds(importer.textures, dds, "particle_probe"),
                                                  importer.warn)
        except Exception as error:
            importer.warn(f"particles: light probes failed ({error}); particles are lit by the sky only")
            volumes = []
        from .lighting import srgb_to_linear
        sun = importer._first("light_environment") if hasattr(importer, "_first") else None
        constant, vertical = probelib.sky_ambient(sun, srgb_to_linear)
        return probelib.ProbeField(volumes, constant, vertical)

    def at(self, position) -> np.ndarray:
        light = np.asarray(self.field.ambient_cube(position), np.float64).mean(axis=0)
        shadows = self.field.baked_shadows(position)
        for e in self.lights:
            light = light + lightlib.light_at(e, position, shadows)
        return light


def _scene_light(context, position, scale: float) -> np.ndarray:
    """The light at a point (game units) from the nearest imported prop's ambient cube (the probes there), or full
    light, as the game draws particles without a map's lighting."""
    best, best_distance = None, None
    for obj in context.scene.objects:
        if "dl_amb_px" not in obj:
            continue
        distance = np.linalg.norm(np.array(obj.matrix_world.translation) / scale - position)
        if best_distance is None or distance < best_distance:
            best, best_distance = obj, distance
    if best is None:
        return np.ones(3)
    faces = [list(best[f"dl_amb_{name}"]) for name in ("px", "py", "pz", "nx", "ny", "nz")]
    return np.asarray(faces, np.float64).mean(axis=0)


# Objects -----------------------------------------------------------------------------------------------------------

class _Target:
    """What effects are built with and into: the texture cache, game unit size, the map's fog group (None without a
    map), the collection, and where warnings go."""

    def __init__(self, textures, scale: float, fog_group, collection, warn=print):
        self.textures = textures
        self.scale = scale
        self.fog_group = fog_group
        self.collection = collection
        self.warn = warn
        # With a map import: its model meshes (for model renderers), the collection its realtime lights light, and
        # the light probes' ambient cube at a point.
        self.model_mesh = None
        self.lit_collection = None
        self.probe_cube = None

    @classmethod
    def for_scene(cls, context, scale: float | None = None) -> "_Target":
        """Building into a scene: the Deadlock map in it (its fog and Particles collection), the add-on's texture
        cache and unit size."""
        from .. import session
        from ..preferences import cache_dir, get_prefs
        from .textures import TextureCache
        prefs = get_prefs()
        textures = TextureCache(session.content(), cache_dir(), int(prefs.texture_limit) if prefs else 0)
        scale = scale or (prefs.unit_scale if prefs else 0.0254)
        fog_group, collection = None, None
        for child in context.scene.collection.children:
            name = child.get("dl_map")
            if name:
                fog_group = bpy.data.node_groups.get(f"DL Fog {name}")
                collection = next((c for c in child.children if c.name.startswith("Particles")), None)
                break
        if collection is None:
            collection = bpy.data.collections.get("Deadlock Particles") or bpy.data.collections.new("Deadlock Particles")
            if collection.name not in context.scene.collection.children:
                context.scene.collection.children.link(collection)
        return cls(textures, scale, fog_group, collection)


class _ObjectBuilder:
    def __init__(self, target: _Target):
        self.target = target
        self.scale = target.scale
        self.materials = _Materials(target)
        self.sprite_group = _shared_group("DL Particle Sprites", _sprite_group)
        self.ribbon_group = _shared_group("DL Particle Ribbons", _ribbon_group)

    def build(self, name, effect, simulation, points, time, seed, ambient, holder=None, local=False):
        """An empty at the system's first control point holding a mesh object per renderer of every system in the
        tree that draws something (into the given empty, when rebuilding). Returns the empty, or None when nothing
        is drawn (no new empty is made then). The empty keeps what a rebuild needs. ``local``: the simulation ran at
        the origin and the objects sit in the empty's space, following it wherever it goes."""
        drawn = list(plib.draw(simulation, ambient))
        if not drawn and holder is None:
            return None
        origin, rotation = points[0]
        if holder is None:
            holder = bpy.data.objects.new(name, None)
            holder.empty_display_type = "SPHERE"
            holder.empty_display_size = 4 * self.scale
            world = Matrix(np.asarray(rotation).tolist()).to_4x4()
            world.translation = tuple(np.asarray(origin) * self.scale)
            holder.matrix_world = world
            self.target.collection.objects.link(holder)
        holder["dl_effect"] = effect
        holder["dl_local"] = bool(local)
        holder["dl_time"] = float(time)
        holder["dl_seed"] = int(seed)
        holder["dl_unit_scale"] = float(self.scale)
        holder["dl_origin"] = [float(x) for x in origin]
        holder["dl_ambient"] = [float(x) for x in ambient]
        # The other control points in the first one's frame, so a rebuild carries them along with the empty.
        relative = {}
        for index, (position, cp_rotation) in points.items():
            if index == 0:
                continue
            local = list(np.asarray(rotation).T @ (np.asarray(position) - origin))
            if cp_rotation is not None:
                local += list((np.asarray(rotation).T @ np.asarray(cp_rotation)).ravel())
            relative[str(index)] = [float(x) for x in local]
        holder["dl_control_points"] = relative

        inverse = Matrix.Identity(4) if local else holder.matrix_world.inverted_safe()
        # Each object sits at the system with its vertices around it: EEVEE sorts blended objects by their origins.
        for system, renderer, batch in drawn:
            if isinstance(batch, R.Models):
                self._models(_stem(system.definition.path), batch, holder, inverse, system.definition.path, renderer)
                continue
            material = self.materials.get(system.definition, renderer, batch.shading)
            if isinstance(batch, R.Sprites):
                obj = self._sprites(f"{_stem(system.definition.path)} sprites", batch, origin, material)
            else:
                obj = self._ribbons(f"{_stem(system.definition.path)} {renderer.kind}", batch, origin, material)
            obj["dl_effect"] = system.definition.path
            obj["dl_renderer"] = renderer.index
            obj.visible_shadow = False
            obj.parent = holder
            obj.matrix_parent_inverse = inverse
            obj.matrix_basis = Matrix.Translation(tuple(np.asarray(origin) * self.scale))
            self.target.collection.objects.link(obj)
        return holder

    def _models(self, name, batch: R.Models, holder, inverse, effect, renderer):
        """A linked duplicate of the model per particle, lit like the map's props."""
        if self.target.model_mesh is None:
            self.target.warn(f"particles: model renderer of '{effect}' needs a map import")
            return
        overrides = ((0, 0xFFFFFFFF, batch.material),) if batch.material else ()
        meshes = [self.target.model_mesh(path, batch.skin, overrides) for path in batch.paths]
        names = ("px", "py", "pz", "nx", "ny", "nz")
        for i in range(batch.count):
            mesh = meshes[int(batch.model[i])]
            if mesh is None:
                continue
            obj = bpy.data.objects.new(f"{name} model", mesh)
            m = batch.matrix[i].copy()
            m[:3, 3] *= self.scale
            obj.parent = holder
            obj.matrix_parent_inverse = inverse
            obj.matrix_basis = Matrix(m.tolist())
            obj.color = tuple(_srgb(c) for c in batch.tint[i]) + (float(np.clip(batch.alpha[i], 0.0, 1.0)),)
            obj.visible_shadow = False
            obj["dl_effect"] = effect
            obj["dl_renderer"] = renderer.index
            if self.target.probe_cube is not None:
                cube = self.target.probe_cube(batch.matrix[i][:3, 3])
                for face, values in zip(names, cube):
                    obj[f"dl_amb_{face}"] = [float(x) for x in values]
            self.target.collection.objects.link(obj)
            if self.target.lit_collection is not None:
                self.target.lit_collection.objects.link(obj)

    def _sprites(self, name, batch: R.Sprites, origin, material):
        n = batch.count
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(n)
        mesh.vertices.foreach_set("co", ((batch.position - origin) * self.scale).astype(np.float32).ravel())
        _attribute(mesh, "dl_half_width", "FLOAT", batch.half_width * self.scale)
        _attribute(mesh, "dl_roll", "FLOAT", batch.roll)
        _attribute(mesh, "dl_yaw", "FLOAT", batch.yaw)
        zeros = np.zeros((n, 1))
        _attribute(mesh, "dl_card_offset", "FLOAT_VECTOR", np.hstack([batch.card[:, 0:2], zeros]))
        _attribute(mesh, "dl_card_size", "FLOAT_VECTOR", np.hstack([batch.card[:, 2:4], zeros]))
        if batch.orientation in (R.FACE_STORED, R.FACE_SCREEN_NORMAL):
            _attribute(mesh, "dl_right", "FLOAT_VECTOR", batch.right)
            _attribute(mesh, "dl_up", "FLOAT_VECTOR", batch.up)
        self._shading_attributes(mesh, batch)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        modifier = obj.modifiers.new("Camera facing cards", "NODES")
        modifier.node_group = self.sprite_group
        _set_input(modifier, "Facing", batch.orientation)
        _set_input(modifier, "Material", material)
        mesh.materials.append(material)
        return obj

    def _ribbons(self, name, batch: R.Ribbons, origin, material):
        mesh = bpy.data.meshes.new(name)
        count = len(batch.position)
        quads = len(batch.quads)
        mesh.vertices.add(count)
        mesh.vertices.foreach_set("co", ((batch.position - origin) * self.scale).astype(np.float32).ravel())
        mesh.loops.add(quads * 4)
        mesh.loops.foreach_set("vertex_index", batch.quads.astype(np.int32).ravel())
        mesh.polygons.add(quads)
        mesh.polygons.foreach_set("loop_start", np.arange(0, quads * 4, 4, dtype=np.int32))
        _attribute(mesh, "dl_axis", "FLOAT_VECTOR", batch.axis)
        _attribute(mesh, "dl_width", "FLOAT", batch.width * self.scale)
        if batch.facing is not None:
            _attribute(mesh, "dl_facing", "FLOAT_VECTOR", batch.facing)
        _attribute(mesh, "dl_card", "FLOAT_VECTOR", np.hstack([batch.card, np.zeros((count, 1))]))
        self._shading_attributes(mesh, batch)
        mesh.update(calc_edges=True)
        mesh.materials.append(material)
        obj = bpy.data.objects.new(name, mesh)
        modifier = obj.modifiers.new("Camera facing strips", "NODES")
        modifier.node_group = self.ribbon_group
        _set_input(modifier, "Face Camera", batch.facing is None)
        return obj

    @staticmethod
    def _shading_attributes(mesh, batch):
        _attribute(mesh, "dl_color", "FLOAT_COLOR", batch.color)
        if np.any(batch.frame_blend):
            _attribute(mesh, "dl_frame_blend", "FLOAT", batch.frame_blend)
        for layer, (rect, following) in enumerate(zip(batch.rects, batch.next_rects)):
            if rect is not None:
                _attribute(mesh, f"dl_rect{layer}", "FLOAT_COLOR", rect)
                _attribute(mesh, f"dl_rect{layer}_next", "FLOAT_COLOR", following)
        for layer, offset in enumerate(batch.offsets):
            if offset is not None:
                _attribute(mesh, f"dl_offset{layer}", "FLOAT_VECTOR", offset)


def _attribute(mesh, name, kind, values):
    attribute = mesh.attributes.new(name, kind, "POINT")
    key = "value" if kind == "FLOAT" else "vector" if kind == "FLOAT_VECTOR" else "color"
    attribute.data.foreach_set(key, np.ascontiguousarray(values, np.float32).ravel())


def _set_input(modifier, name, value):
    """Sets a geometry nodes modifier input: through ``properties.inputs`` (Blender 5.2), or as the ID property
    earlier versions keep it in."""
    for item in modifier.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == name:
            properties = getattr(modifier, "properties", None)
            socket = getattr(getattr(properties, "inputs", None), item.identifier, None)
            if socket is not None:
                socket.value = value
            else:
                modifier[item.identifier] = value
            return


def _stem(path: str) -> str:
    return (path or "particles").replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0][:56]


# Geometry node groups ----------------------------------------------------------------------------------------------

def _shared_group(name: str, build) -> bpy.types.NodeTree:
    """A node group every effect shares: the file's own when it is this version's, else built anew (taking over
    the old one's users)."""
    old = bpy.data.node_groups.get(name)
    if old is not None and old.get("dl_version") == _GROUP_VERSION:
        return old
    tree = build()
    if old is not None:
        old.user_remap(tree)
        bpy.data.node_groups.remove(old)
        tree.name = name
    return tree


def _geometry_group(name: str, inputs: list[tuple]) -> tuple[bpy.types.NodeTree, Builder, dict, object]:
    """A new geometry node group with a Geometry input and output plus the given (name, socket type, default)
    inputs. Returns the tree, a builder, the group input sockets by name and the output node."""
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree["dl_version"] = _GROUP_VERSION
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for socket_name, socket_type, default in inputs:
        item = tree.interface.new_socket(socket_name, in_out="INPUT", socket_type=socket_type)
        if default is not None:
            item.default_value = default
    b = Builder(tree)
    b.column(-1)
    group_in = b.node("NodeGroupInput")
    group_in.hide = False
    b.column(14)
    group_out = b.node("NodeGroupOutput")
    group_out.hide = False
    b.column(0)
    return tree, b, {s.name: s for s in group_in.outputs if s.name}, group_out


def _named(b: Builder, name: str, kind: str = "FLOAT_VECTOR"):
    node = b.node("GeometryNodeInputNamedAttribute", data_type=kind)
    node.inputs["Name"].default_value = name
    return node.outputs["Attribute"]


def _camera(b: Builder):
    """The active camera's location and its right, up and forward axes, in the object's space (so the cards keep
    facing it wherever the object is moved)."""
    camera = b.node("GeometryNodeInputActiveCamera")
    info = b.node("GeometryNodeObjectInfo", transform_space="RELATIVE")
    b.set(info.inputs["Object"], camera.outputs[0])
    axes = []
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)):
        rotate = b.node("FunctionNodeRotateVector")
        b.set(rotate.inputs["Vector"], axis)
        b.set(rotate.inputs["Rotation"], info.outputs["Rotation"])
        axes.append(rotate.outputs[0])
    return info.outputs["Location"], axes[0], axes[1], axes[2]


def _rotate(b: Builder, vector, axis, angle):
    rotation = b.node("FunctionNodeAxisAngleToRotation")
    b.set(rotation.inputs["Axis"], axis)
    b.set(rotation.inputs["Angle"], angle)
    rotate = b.node("FunctionNodeRotateVector")
    b.set(rotate.inputs["Vector"], vector)
    b.set(rotate.inputs["Rotation"], rotation.outputs[0])
    return rotate.outputs[0]


def _index_switch(b: Builder, index, options):
    node = b.node("GeometryNodeIndexSwitch", data_type="VECTOR")
    while len([s for s in node.inputs if s.type == "VECTOR"]) < len(options):
        node.index_switch_items.new()
    b.set(node.inputs["Index"], index)
    for i, option in enumerate(options):
        b.set(node.inputs[str(i)], option)
    return node.outputs[0]


def _sprite_group() -> bpy.types.NodeTree:
    """``DL Particle Sprites``: a card on every point, realized, its corners placed as the spritecard vertex shader
    places them. Base axes by Facing (0 the camera's, 1 the camera's turned about world Z, 2 stored, 3 stored right with
    up towards the camera), turned by the particle's roll about the card's normal and its yaw about the card's up;
    the card offset and sized (in half widths) along them."""
    tree, b, i, out = _geometry_group("DL Particle Sprites", [("Facing", "NodeSocketInt", 0),
                                                            ("Material", "NodeSocketMaterial", None)])
    b.frame("Card")
    grid = b.node("GeometryNodeMeshGrid")
    for key, value in (("Size X", 2.0), ("Size Y", 2.0), ("Vertices X", 2), ("Vertices Y", 2)):
        grid.inputs[key].default_value = value
    position = b.node("GeometryNodeInputPosition").outputs[0]
    corner_store = b.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="POINT")
    corner_store.inputs["Name"].default_value = "dl_corner"
    b.set(corner_store.inputs["Geometry"], grid.outputs["Mesh"])
    b.set(corner_store.inputs["Value"], position)
    b.frame(None)

    b.column(1)
    instances = b.node("GeometryNodeInstanceOnPoints")
    b.set(instances.inputs["Points"], i["Geometry"])
    b.set(instances.inputs["Instance"], corner_store.outputs[0])
    realized = b.node("GeometryNodeRealizeInstances")
    b.set(realized.inputs["Geometry"], instances.outputs[0])

    b.column(2)
    b.frame("Camera")
    _, camera_right, camera_up, camera_forward = _camera(b)
    b.frame(None)

    b.column(4)
    b.frame("Base axes")
    stored_right = _named(b, "dl_right")
    stored_up = _named(b, "dl_up")
    z_right = b.cross((0.0, 0.0, 1.0), camera_forward)
    normal_up = b.normalize(b.cross(stored_right, camera_forward))
    right = _index_switch(b, i["Facing"], [camera_right, z_right, stored_right, stored_right])
    up = _index_switch(b, i["Facing"], [camera_up, (0.0, 0.0, 1.0), stored_up, normal_up])
    b.frame(None)

    b.column(6)
    b.frame("Roll and yaw")
    normal = b.normalize(b.cross(right, up))
    roll = _named(b, "dl_roll", "FLOAT")
    yaw = _named(b, "dl_yaw", "FLOAT")
    right = _rotate(b, right, normal, roll)
    up = _rotate(b, up, normal, roll)
    right = _rotate(b, right, b.normalize(up), yaw)
    b.frame(None)

    b.column(8)
    b.frame("Corner")
    corner = _named(b, "dl_corner")
    cx, cy, _ = b.separate(corner)
    ox, oy, _ = b.separate(_named(b, "dl_card_offset"))
    sx, sy, _ = b.separate(_named(b, "dl_card_size"))
    half_width = _named(b, "dl_half_width", "FLOAT")
    across = b.mul(b.madd(cx, sx, ox), half_width)
    along = b.mul(b.madd(cy, sy, oy), half_width)
    offset = b.vadd(b.vscale(right, across), b.vscale(up, along))
    center = b.vsub(position, corner)
    b.frame(None)

    b.column(10)
    placed = b.node("GeometryNodeSetPosition")
    b.set(placed.inputs["Geometry"], realized.outputs[0])
    b.set(placed.inputs["Position"], b.vadd(center, offset))
    card = b.combine(b.mul(b.add(cx, 1.0), 0.5), b.mul(b.sub(1.0, cy), 0.5), 0.0)
    card_store = b.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="POINT")
    card_store.inputs["Name"].default_value = "dl_card"
    b.set(card_store.inputs["Geometry"], placed.outputs[0])
    b.set(card_store.inputs["Value"], card)
    material = b.node("GeometryNodeSetMaterial")
    b.set(material.inputs["Geometry"], card_store.outputs[0])
    b.set(material.inputs["Material"], i["Material"])
    b.set(out.inputs[0], material.outputs[0])
    return tree


def _ribbon_group() -> bpy.types.NodeTree:
    """``DL Particle Ribbons``: pushes each vertex of a strip sideways by its signed width, across the strip's
    direction in the plane facing the camera (or the plane with the stored normal)."""
    tree, b, i, out = _geometry_group("DL Particle Ribbons", [("Face Camera", "NodeSocketBool", True)])
    position = b.node("GeometryNodeInputPosition").outputs[0]
    b.frame("Camera")
    location, _, _, _ = _camera(b)
    b.frame(None)
    b.column(2)
    switch = b.node("GeometryNodeSwitch", input_type="VECTOR")
    b.set(switch.inputs["Switch"], i["Face Camera"])
    b.set(switch.inputs["False"], _named(b, "dl_facing"))
    b.set(switch.inputs["True"], b.vsub(location, position))
    side = b.normalize(b.cross(_named(b, "dl_axis"), switch.outputs[0]))
    b.column(4)
    moved = b.node("GeometryNodeSetPosition")
    b.set(moved.inputs["Geometry"], i["Geometry"])
    b.set(moved.inputs["Offset"], b.vscale(side, _named(b, "dl_width", "FLOAT")))
    b.set(out.inputs[0], moved.outputs[0])
    return tree


# Materials ---------------------------------------------------------------------------------------------------------

class _Materials:
    """Spritecard materials, one per renderer, settings and fog (systems placed many times share theirs, as do
    effects built later into the same file)."""

    def __init__(self, target: _Target):
        self.target = target
        self._cache = {m["dl_particle_key"]: m for m in bpy.data.materials if "dl_particle_key" in m}

    def get(self, definition, renderer, shading: dict):
        fog = self.target.fog_group.name if self.target.fog_group is not None else ""
        settings = zlib.crc32(repr(sorted(shading.items())).encode())
        key = f"{definition.path}|{renderer.index}|{fog}|{settings:08x}"
        material = self._cache.get(key)
        if material is None:
            material = self._cache[key] = self._build(definition, renderer, shading)
            material["dl_particle_key"] = key
        return material

    def gradient(self, stops) -> bpy.types.Image:
        """A 256 texel ramp of a layer's gradient stops (gamma space, as they are authored)."""
        name = f"particle gradient {zlib.crc32(repr(tuple(stops)).encode()):08x}"
        image = bpy.data.images.get(name)
        if image is not None:
            return image
        width = 256
        pixels = np.ones((width, 4), np.float32)
        t = np.arange(width) / (width - 1)
        if stops:
            positions = np.array([s[0] for s in stops])
            colors = np.array([s[1] for s in stops])
            for c in range(3):
                pixels[:, c] = np.interp(t, positions, colors[:, c])
        image = bpy.data.images.new(name, width, 1, alpha=True)
        image.colorspace_settings.name = "sRGB"
        image.pixels.foreach_set(pixels.ravel())
        image.pack()
        return image

    def _build(self, definition, renderer, shading: dict) -> bpy.types.Material:
        name = f"{_stem(definition.path)} {renderer.kind}"
        material = bpy.data.materials.new(name)
        material["dl_effect"] = definition.path
        material["dl_renderer"] = renderer.index
        material.use_nodes = True
        material.node_tree.nodes.clear()
        _SpritecardShader(self, material, renderer, shading).build()
        additive = shading["blend_mode"] == R.BLEND_ADD
        if shading["opaque"]:
            material.surface_render_method = "DITHERED"
        else:
            material.surface_render_method = "BLENDED"
        material.use_backface_culling = False
        material.use_transparency_overlap = True
        material["dl_additive"] = additive
        try:
            material.cycles.emission_sampling = "NONE"
        except AttributeError:
            pass
        return material


class _SpritecardShader:
    """deadlock_particle.shader as nodes: each texture layer sampled in its sheet frame (cross-faded to the next),
    mixed into the ones before it by channel and blend mode; times the vertex colour and alpha; scaled by the
    overbright factor; added over the scene (dimmed by the fog in front) or alpha blended (fogged)."""

    def __init__(self, materials: _Materials, material, renderer, shading: dict):
        self.materials = materials
        self.target = materials.target
        self.material = material
        self.renderer = renderer
        self.shading = shading
        self.b = Builder(material.node_tree)
        self._previous = None  # the last layer sampled: image, extension, coordinate (distortion resamples it)

    def build(self):
        b = self.b
        b.column(0)
        b.frame("Particle")
        card = b.attribute("dl_card").outputs["Vector"]
        vertex = b.attribute("dl_color")
        frame_blend = b.attribute("dl_frame_blend").outputs["Fac"]
        b.frame(None)

        accum_rgb, accum_a = None, None  # None: still the starting (1, 1, 1, 1)
        for index, layer in enumerate(self.shading["layers"]):
            b.column(1 + index * 6)
            b.frame(f"Layer {index}")
            rgb, alpha = self._layer(index, layer, card, frame_blend, accum_rgb, accum_a)
            accum_rgb, accum_a = self._mix(layer, accum_rgb, accum_a, rgb, alpha)
            b.frame(None)

        column = 1 + len(self.shading["layers"]) * 6
        b.column(column)
        b.frame("Colour and alpha")
        color = vertex.outputs["Color"] if accum_rgb is None else b.vmul(vertex.outputs["Color"], accum_rgb)
        alpha = 1.0 if accum_a is None else accum_a
        remap = self.shading["alpha_remap"]
        if remap is not None:
            alpha = b.smoothstep(remap[0], remap[1], alpha)
        desaturation = self.shading["desaturation"]
        if desaturation:
            gray = b.dot(color, _LUMA)
            color = b.vlerp(color, b.combine(gray, gray, gray), desaturation)
        alpha = b.mul(alpha, vertex.outputs["Alpha"])
        b.frame(None)

        b.column(column + 2)
        b.frame("Output")
        blend_mode = self.shading["blend_mode"]
        transparent = b.node("ShaderNodeBsdfTransparent")
        if blend_mode == R.BLEND_MOD2X:
            # Modulate: what's behind times twice the colour, grey (0.5) where transparent.
            tint = b.vlerp((0.5, 0.5, 0.5), b.vlerp((0.5, 0.5, 0.5), color, vertex.outputs["Color"]), alpha)
            b.set(transparent.inputs["Color"], b.vscale(b.vmath("MINIMUM", b.vmath("MAXIMUM", tint, (0, 0, 0)), (1, 1, 1)), 2.0))
            shader = transparent.outputs[0]
        else:
            overbright = b.value(self.shading["overbright"], "Overbright")
            emission = b.node("ShaderNodeEmission")
            b.set(emission.inputs["Color"], b.vscale(color, overbright))
            fog_group = self.target.fog_group
            fog = b.group(fog_group, Amount=1.0) if fog_group is not None else None
            if blend_mode == R.BLEND_ADD:
                # Drawn over a fogged scene: dimmed by the fog in front of it rather than fogged again.
                b.set(emission.inputs["Strength"], b.mul(alpha, fog.outputs["Attenuation"]) if fog else alpha)
                add = b.node("ShaderNodeAddShader")
                b.set(add.inputs[0], transparent.outputs[0])
                b.set(add.inputs[1], emission.outputs[0])
                shader = add.outputs[0]
            else:
                surface = emission.outputs[0]
                if fog is not None:
                    b.set(fog.inputs["Shader"], surface)
                    surface = fog.outputs["Shader"]
                if self.shading["opaque"]:
                    alpha = b.math("GREATER_THAN", alpha, 0.5)
                mix = b.node("ShaderNodeMixShader")
                b.set(mix.inputs[0], alpha)
                b.set(mix.inputs[1], transparent.outputs[0])
                b.set(mix.inputs[2], surface)
                shader = mix.outputs[0]
        b.frame(None)
        b.column(column + 4)
        output = b.node("ShaderNodeOutputMaterial")
        output.hide = False
        b.set(output.inputs["Surface"], shader)

    # Layers ---------------------------------------------------------------------------------------------------

    def _placed(self, index, card, layer: dict):
        """The card coordinate turned, scaled and offset by the layer's transform (around the card's centre), plus
        the particle's own offset when the layer randomizes them."""
        b = self.b
        scale, offset, rotation = layer["scale"], layer["offset"], layer["rotation"]
        if layer["random_offsets"]:
            card = b.vadd(card, b.attribute(f"dl_offset{index}").outputs["Vector"])
        if scale == (1.0, 1.0) and offset == (0.0, 0.0) and rotation == 0.0 and not layer["clamp"]:
            return card
        centred = b.vsub(card, (0.5, 0.5, 0.0))
        if rotation:
            rotate = b.node("ShaderNodeVectorRotate", rotation_type="Z_AXIS")
            b.set(rotate.inputs["Vector"], centred)
            b.set(rotate.inputs["Angle"], rotation)
            centred = rotate.outputs[0]
        origin = (((offset[0] + 0.5) % 1.0), ((offset[1] + 0.5) % 1.0), 0.0)
        inverse = tuple(1.0 / s if s else 0.0 for s in scale) + (1.0,)
        placed = b.vmath("MULTIPLY_ADD", centred, inverse, origin)
        if layer["clamp"]:
            placed = b.vmath("MINIMUM", b.vmath("MAXIMUM", placed, (0.0, 0.0, 0.0)), (1.0, 1.0, 1.0))
        return placed

    def _in_rect(self, placed, rect):
        """A placed coordinate in a sheet frame's rectangle, as a Blender image coordinate (V up)."""
        b = self.b
        if rect is not None:
            r, g, blue = b.separate_color(rect.outputs["Color"])
            low = b.combine(r, g, 0.0)
            size = b.combine(b.sub(blue, r), b.sub(rect.outputs["Alpha"], g), 0.0)
            placed = b.vmath("MULTIPLY_ADD", placed, size, low)
        return b.vmath("MULTIPLY_ADD", placed, (1.0, -1.0, 1.0), (0.0, 1.0, 0.0))

    def _image(self, layer: dict):
        if layer["gradient"] is not None:
            return self.materials.gradient(layer["gradient"]), "EXTEND"
        image = self.target.textures.image(layer["texture"], color=True)
        return image, "REPEAT"

    def _layer(self, index, layer, card, frame_blend, accum_rgb, accum_a):
        """A layer's sample: (rgb, alpha) sockets."""
        b = self.b
        image, extension = self._image(layer)
        if layer["kind"] == R.TEXTURE_COLOR_LOOKUP:
            return self._color_lookup(index, layer, image, accum_rgb, accum_a)
        placed = self._placed(index, card, layer)
        rect = b.attribute(f"dl_rect{index}") if layer["sheet"] else None
        uv = self._in_rect(placed, rect)
        sample = b.image(image, uv, extension=extension, label=f"Layer {index}")
        rgb, alpha = sample.outputs["Color"], sample.outputs["Alpha"]
        if layer["sheet"] and self.shading["blend_frames"]:
            following = b.attribute(f"dl_rect{index}_next")
            sample_next = b.image(image, self._in_rect(placed, following), extension=extension, label=f"Layer {index} next frame")
            next_rgb, next_alpha = sample_next.outputs["Color"], sample_next.outputs["Alpha"]
            if index == 0 and self.shading["max_luminance_blend"]:
                weight = b.mul(b.sub(1.0, frame_blend), b.math("SQRT", b.math("MAXIMUM", b.dot(rgb, _LUMA), 0.001)))
                weight_next = b.mul(frame_blend, b.math("SQRT", b.math("MAXIMUM", b.dot(next_rgb, _LUMA), 0.001)))
                total = b.add(weight, weight_next)
                rgb = b.vscale(b.vadd(b.vscale(rgb, weight), b.vscale(next_rgb, weight_next)), b.div(1.0, total))
                alpha = b.div(b.add(b.mul(alpha, weight), b.mul(next_alpha, weight_next)), total)
            else:
                rgb = b.vlerp(rgb, next_rgb, frame_blend)
                alpha = b.lerp(alpha, next_alpha, frame_blend)
        if layer["kind"] in (R.TEXTURE_ZOOM, R.TEXTURE_DISTORTION_ZOOM):
            rgb, alpha = self._zoom(index, image, extension, uv, layer["zoom"])
        if layer["kind"] in (R.TEXTURE_DISTORTION, R.TEXTURE_DISTORTION_ZOOM) and self._previous is not None:
            rgb, alpha = self._distort(index, rgb, alpha, layer["distortion"])
        self._previous = (image, extension, uv)
        return rgb, alpha

    def _zoom(self, index, image, extension, uv, rate: float):
        """Zooms in on the card's centre, between a wide and a tight sample, by how far through its cycle the
        layer's zoom rate is."""
        b = self.b
        phase = rate % 1.0
        eased = phase / (-0.4 * (1.0 - phase) + 1.0)
        samples = []
        for scale in (4.0 + (1.0 - 4.0) * eased, 1.0 + (0.0625 - 1.0) * eased):
            zoomed = b.vmath("MULTIPLY_ADD", b.vsub(uv, (0.5, 0.5, 0.0)), (scale, scale, 1.0), (0.5, 0.5, 0.0))
            samples.append(b.image(image, zoomed, extension=extension, label=f"Layer {index} zoom"))
        wide, tight = samples
        amount = eased * eased
        return (b.vlerp(tight.outputs["Color"], wide.outputs["Color"], amount),
                b.lerp(tight.outputs["Alpha"], wide.outputs["Alpha"], amount))

    def _distort(self, index, rgb, alpha, strength: float):
        """The layer before, sampled again where this layer's red and green (in gamma space, -1 to 1) push it."""
        b = self.b
        image, extension, uv = self._previous
        gamma = b.node("ShaderNodeGamma")
        b.set(gamma.inputs["Color"], rgb)
        b.set(gamma.inputs["Gamma"], 1.0 / 2.2)
        push = b.vmath("MULTIPLY_ADD", gamma.outputs[0], (2.0, 2.0, 0.0), (-1.0, -1.0, 0.0))
        push = b.vscale(push, b.mul(alpha, strength * 0.125))
        # V runs up in Blender's image coordinates: a push down the card moves the sample up the image.
        moved = b.vmath("MULTIPLY_ADD", push, (-1.0, 1.0, 0.0), uv)
        sample = b.image(image, moved, extension=extension, label=f"Layer {index} distorted")
        return sample.outputs["Color"], sample.outputs["Alpha"]

    def _color_lookup(self, index, layer, image, accum_rgb, accum_a):
        """A one dimensional colour lookup by the luminance (in gamma space) of the layers below."""
        b = self.b
        mode = layer["channels"]
        prev_rgb = accum_rgb if accum_rgb is not None else (1.0, 1.0, 1.0)
        prev_a = accum_a if accum_a is not None else 1.0
        straight = mode in (0, 4, 5, 9, 10, 11)
        from_alpha = mode in (2, 7)
        if from_alpha:
            source = b.combine(prev_a, prev_a, prev_a)
        elif straight:
            source = prev_rgb
        else:
            source = b.vscale(prev_rgb, prev_a)
        gamma = b.node("ShaderNodeGamma")
        b.set(gamma.inputs["Color"], source)
        b.set(gamma.inputs["Gamma"], 1.0 / 2.2)
        u = b.dot(gamma.outputs[0], _LUMA)
        ramp = b.image(image, b.combine(u, 0.5, 0.0), extension="EXTEND", label=f"Layer {index} lookup")
        alpha = b.dot(ramp.outputs["Color"], _LUMA_601) if from_alpha else prev_a
        return ramp.outputs["Color"], alpha

    def _mix(self, layer, accum_rgb, accum_a, rgb, alpha):
        """Mixes a layer into what's below it: by its channel mask, then its blend mode, by its blend amount."""
        b = self.b
        mode, blend = layer["channels"], layer["blend_mode"]
        subtract = blend == 4
        acc_rgb = accum_rgb if accum_rgb is not None else (1.0, 1.0, 1.0)
        acc_a = accum_a if accum_a is not None else 1.0
        zero_rgb = (0.0, 0.0, 0.0)

        def luma601(c):
            return b.dot(c, _LUMA_601)

        def channel(i):
            return b.separate_color(rgb)[i]

        def spread(value, keep_index):
            parts = [value if (j == keep_index or not subtract) else 0.0 for j in range(3)]
            return b.combine(*parts)

        if mode == 0:
            m_rgb, m_a = rgb, (0.0 if subtract else acc_a)
        elif mode == 2:
            m_rgb, m_a = (zero_rgb if subtract else acc_rgb), alpha
        elif mode == 3:
            m_rgb, m_a = rgb, b.mul(acc_a, alpha)
        elif mode == 4:
            m_rgb, m_a = b.vlerp(acc_rgb, rgb, alpha), acc_a
        elif mode == 5:
            m_rgb, m_a = b.vlerp(acc_rgb, rgb, luma601(rgb)), acc_a
        elif mode == 6:
            m_rgb, m_a = rgb, luma601(rgb)
        elif mode == 7:
            m_rgb, m_a = (zero_rgb if subtract else acc_rgb), luma601(rgb)
        elif mode == 8:
            m_rgb, m_a = acc_rgb, b.mul(acc_a, luma601(rgb))
        elif mode in (9, 10, 11):
            value = channel(mode - 9)
            m_rgb, m_a = spread(value, mode - 9), (0.0 if subtract else acc_a)
        elif mode in (12, 13, 14):
            value = channel(mode - 12)
            m_rgb, m_a = spread(value, mode - 12), value
        else:
            m_rgb, m_a = rgb, alpha

        if accum_rgb is None and accum_a is None and blend in (0, 2) and layer["blend"] == 1.0:
            return m_rgb, m_a  # multiplying (or replacing) the starting white
        if blend == 1:
            out_rgb, out_a = b.vscale(b.vmul(acc_rgb, m_rgb), 2.0), b.mul(b.mul(acc_a, m_a), 2.0)
        elif blend == 2:
            out_rgb, out_a = m_rgb, m_a
        elif blend == 3:
            out_rgb, out_a = b.vadd(acc_rgb, m_rgb), b.add(acc_a, m_a)
        elif blend == 4:
            out_rgb, out_a = b.vsub(acc_rgb, m_rgb), b.sub(acc_a, m_a)
        elif blend == 5:
            out_rgb, out_a = b.vscale(b.vadd(acc_rgb, m_rgb), 0.5), b.mul(b.add(acc_a, m_a), 0.5)
        elif blend == 6:
            out_rgb, out_a = b.vscale(acc_rgb, b.dot(m_rgb, _LUMA)), m_a
        else:
            out_rgb, out_a = b.vmul(acc_rgb, m_rgb), b.mul(acc_a, m_a)
        amount = layer["blend"]
        if amount != 1.0:
            out_rgb = b.vlerp(acc_rgb, out_rgb, amount)
            out_a = b.lerp(acc_a, out_a, amount)
        return b.vmath("MAXIMUM", out_rgb, zero_rgb), b.math("MAXIMUM", out_a, 0.0)
