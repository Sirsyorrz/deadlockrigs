"""A map's particle effects (candle flames, the fireplace, street steam, crows) into its built scene, through the
vendored Deadlock Bridge (tools/deadlock_bridge, by Divined): every effect the map plays as it loads, simulated a
few seconds in and frozen, its sprites and ropes facing the camera.

The bridge places them in game space; the map's glTF export is a quarter turn about Z from it, so the effects are
built into that frame. Effects that draw models (Midtown's steam blobs) get the bridge's materials, lit by the light
probes through a lighting group without lightmaps, and the map's own fog.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from deadlock_bridge.blender import lighting, nodes, particles, post
from deadlock_bridge.blender import materials as matlib
from deadlock_bridge.blender.map_import import MapImporter
from deadlock_bridge.blender.textures import TextureCache
from deadlock_bridge.core import content as contentlib
from deadlock_bridge.core import world as vworld

UNIT = 0.0254
# game space -> the map's Blender space: Source2Viewer's glTF export, re-imported, is turned -90 degrees about Z
FRAME = Matrix.Rotation(-math.pi / 2, 4, 'Z')
# what the game shows a while after the map loads: looping effects are past their start-up
SIM_TIME = 6.0


class Models:
    """Model meshes with the bridge's materials, for particle model renderers; the parts of its MapImporter that
    make them."""

    _first = MapImporter._first
    model = MapImporter.model
    model_mesh = MapImporter.model_mesh
    _build_model_mesh = MapImporter._build_model_mesh

    def __init__(self, content, textures, name, fog_group, entities=()):
        self.content = content
        self.textures = textures
        self.map = name
        self.scale = UNIT
        self.entities = list(entities)
        self.log = []
        self.stats = {}
        self._model_cache, self._mesh_cache, self._model_known = {}, {}, {}
        self.lightmap_scale = (1.0, 1.0)
        self.fog_group = bridge_fog(name, fog_group)
        images = {'dc': None, 'lum': None, 'shadows': None,
                  'brdf': textures.image('textures/dev/brdf_lut.vtex', image=1, full_size=True)}
        self.lighting_group = lighting.build_lighting_group(name, images, UNIT, self.fog_group, [])
        unlit = lighting.build_unlit_group(name, self.fog_group)
        additive = lighting.build_additive_group(name, self.fog_group)
        post_entity = (self._first('post_processing_volume', lambda e: e.get_bool('master'))
                       or self._first('post_processing_volume'))
        npr = lighting.build_npr_group(name, self.fog_group, post.exposure_scale(content, post_entity))
        self.materials = matlib.MaterialContext(content, textures, name, self.lighting_group, unlit, additive, UNIT,
                                                npr)

    def warn(self, message):
        self.log.append(message)


class Entities:
    """A map's entities, as the bridge's light probe reader wants them."""

    _first = MapImporter._first
    _read_entities = MapImporter._read_entities

    def __init__(self, content, slug, textures):
        self.content = content
        self.textures = textures
        self.log, self.stats = [], {}
        content.prefer_map(slug)
        self.world = vworld.World(content.read(f'maps/{slug}/world.vwrld_c'))
        self.entities = self._read_entities()

    def warn(self, message):
        self.log.append(message)


class MapSource(Models):
    """The parts of the bridge's MapImporter its particle builder reads."""

    _entity_matrix = MapImporter._entity_matrix

    def __init__(self, content, slug, cache, collection, fog_group):
        textures = TextureCache(content, cache)
        entities = Entities(content, slug, textures)
        super().__init__(content, textures, slug, fog_group, entities.entities)
        self.log = entities.log
        self.frame = FRAME
        self.root = collection
        self.collections = {}
        self.sun = self._first('light_environment')


def bridge_fog(name, fog_group):
    """The bridge's fog interface (Shader, Amount -> Shader, Attenuation) over the map's own fog group, or no fog."""
    tree, b, i, o = nodes.new_group(f'DL Fog {name}', [('Shader', 'shader'), ('Amount', 'float', 1.0)],
                                    [('Shader', 'shader'), ('Attenuation', 'float')])
    if fog_group is None:
        b.set(o['Shader'], i['Shader'])
        b.set(o['Attenuation'], 1.0)
        return tree
    fog = b.group(fog_group, Shader=i['Shader'], Alpha=i['Amount'])
    b.set(o['Shader'], fog.outputs['Shader'])
    attenuation = fog.outputs.get('Attenuation')
    b.set(o['Attenuation'], attenuation if attenuation is not None else 1.0)
    return tree


def add_particles(coll, slug, install, cache, fog_group=None):
    """Builds the map's effects into a 'particles' collection under coll. Returns (systems built, warnings)."""
    if not contentlib.is_deadlock_folder(install):
        raise RuntimeError(f'no Deadlock install at {install}')
    content = contentlib.GameContent(install)
    try:
        parts = bpy.data.collections.new('particles')
        coll.children.link(parts)
        source = MapSource(content, slug, cache, coll, fog_group)
        source.collections['Particles'] = parts
        built = particles.build_particle_systems(source, SIM_TIME)
        for ob in parts.objects:
            ob['dlr_particles'] = True
        return built, source.log
    finally:
        content.close()


_scene_models = None


def scene_models(context, target):
    """For effects placed in a scene: model meshes already in the file (an appended map's), else new ones with the
    bridge's materials, fogged by the scene's map. Kept while the file's groups for them are."""
    def model_mesh(path, skin='', overrides=()):
        global _scene_models
        if not skin and not overrides:
            mesh = next((m for m in bpy.data.meshes if m.get('dl_model') == contentlib.normalize(path)), None)
            if mesh is not None:
                return mesh
        from deadlock_bridge import session
        try:
            if _scene_models is not None:
                _scene_models.lighting_group.name
        except ReferenceError:  # a new file, or the group was deleted
            _scene_models = None
        if _scene_models is None:
            _scene_models = Models(session.content(), target.textures, 'placed effects', target.fog_group)
        return _scene_models.model_mesh(path, skin, overrides)

    return model_mesh


_probe_lights = {}


def scene_light(context, point):
    """The ambient light and cube at a point of the scene from its map's light probes and realtime lights, read from
    the game's files (once per map and session), or None without a map."""
    from deadlock_bridge import session
    from deadlock_bridge.preferences import cache_dir
    slug = next((c['dlr_map'] for c in context.scene.collection.children_recursive if c.get('dlr_map')), None)
    if slug is None:
        return None
    light = _probe_lights.get(slug)
    if light is None:
        content = session.content()
        light = _probe_lights[slug] = particles._ProbeLight(Entities(content, slug, TextureCache(content, cache_dir())))
    game = np.array(FRAME.inverted() @ Vector(point), np.float64) / UNIT
    cube = particles._cube_in_frame(light.field.ambient_cube(game), FRAME)
    return light.at(game), cube
