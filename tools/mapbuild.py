"""Builds a Blender scene of a Deadlock map from the user's install. Runs on Windows, Linux and macOS; the add-on
calls it too.

python tools/mapbuild.py <map> [options]

  export    the map's world as glTF with textures (Source2Viewer)
  props     the props that play an idle animation, each as its own glTF with that clip; the rebatch leaves them out
  rebatch   one mesh per grid cell and material instead of ~30,000 objects (map_rebatch.py)
  lighting  the baked lighting (lightmaps, light probes) and the sky, sun, fog and tonemapping the map uses
  import    <out>/maps/<map>/<map>.blend, textures in textures/ and lighting files in lightmaps/ next to it, with
            the map's particle effects

The map is rebuilt only when the game's version or the map's archive changed since the last build (or with --force).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from pipeline import Builder, BuildError, find_pak, log  # noqa: E402
import map_environment as env  # noqa: E402
import map_npcs  # noqa: E402

MAPS = {'dl_midtown': 'Midtown', 'dl_hideout': 'Hideout'}
# the export, rebatched copy and saved scene of Midtown take about 11 GB together
NEEDED_GB = 12
# meshes Source2Viewer writes for each map's export, for its progress
EXPORT_MESHES = {'dl_midtown': 32000, 'dl_hideout': 1060}
# each step's share of a Midtown build's time, for the progress bar
# map_import.py's GLTF_END; it can't report progress during Blender's glTF import, so its log lines are counted
GLTF_IMPORT_END = 0.22
STEP_WEIGHTS = {'export': 0.25, 'props': 0.005, 'rebatch': 0.015, 'lighting': 0.08, 'import': 0.65}


class MapFiles:
    def __init__(self, slug, work, out):
        self.slug = slug
        self.title = MAPS[slug]
        self.dir = os.path.join(work, 'maps', slug)
        self.out = os.path.join(out, 'maps', slug)
        self.export = os.path.join(self.dir, 'export')
        self.world = os.path.join(self.export, 'maps', slug, 'world.gltf')
        self.lighting = os.path.join(self.dir, 'lighting')
        self.rebatched = os.path.join(self.dir, 'rebatched', slug + '.gltf')
        self.props = os.path.join(self.dir, 'props')
        self.props_file = os.path.join(self.props, 'props.json')
        self.blend = os.path.join(self.out, slug + '.blend')
        self.stamp_file = os.path.join(self.out, 'map.json')
        self.npcs_file = os.path.join(self.out, 'npcs.json')
        self.ziplines_file = os.path.join(self.out, 'ziplines.json')


def install_dir(pak):
    """The Deadlock folder holding game/citadel/pak01_dir.vpk."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(pak))))


def map_vpk(pak, slug):
    path = os.path.join(os.path.dirname(pak), 'maps', slug + '.vpk')
    if not os.path.exists(path):
        raise BuildError(f'no {slug}.vpk in {os.path.dirname(path)}')
    return path


def game_version(pak):
    try:
        with open(os.path.join(os.path.dirname(pak), 'steam.inf'), encoding='utf-8') as f:
            for line in f:
                if line.startswith('ClientVersion='):
                    return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return ''


# raised when a change to the build makes earlier builds wrong, so they show as needing a rebuild
BUILDER = 25


def game_stamp(pak, slug, mods=True):
    vpk = os.stat(map_vpk(pak, slug))
    stamp = {'game_version': game_version(pak), 'vpk_size': vpk.st_size, 'vpk_mtime': int(vpk.st_mtime),
             'builder': BUILDER}
    if not mods:
        stamp['mods'] = False
    return stamp


def vanilla_vpk(pak, slug, folder):
    """The map's vpk linked into <folder>, and a gameinfo.gi that mounts only the game's own files. Opened from
    the game folder, Source2Viewer reads the game's gameinfo.gi, whose search paths put mod managers' addon folders
    in front of the game, so mods that replace map textures would end up in the export. The gameinfo.gi is in a
    subfolder, where Source2Viewer doesn't find it by itself and only loads it once, from --game."""
    citadel = os.path.dirname(pak)
    os.makedirs(folder, exist_ok=True)
    vpk = os.path.join(folder, slug + '.vpk')
    if os.path.exists(vpk):
        os.remove(vpk)
    try:
        os.link(map_vpk(pak, slug), vpk)
    except OSError:
        shutil.copyfile(map_vpk(pak, slug), vpk)
    # forward slashes: KeyValues reads a backslash as an escape
    paths = [p.replace(os.sep, '/') for p in (citadel, os.path.join(os.path.dirname(citadel), 'core'))]
    gameinfo = os.path.join(folder, 'game', 'gameinfo.gi')
    os.makedirs(os.path.dirname(gameinfo), exist_ok=True)
    with open(gameinfo, 'w', encoding='utf-8') as f:
        f.write('"GameInfo"\n{\n\tgame "citadel"\n\tFileSystem\n\t{\n\t\tSearchPaths\n\t\t{\n'
                + ''.join(f'\t\t\tGame "{p}"\n' for p in paths) + '\t\t}\n\t}\n}\n')
    return vpk, gameinfo


def prop_nodes(j, model, origin):
    """The nodes Source2Viewer wrote for a model placed at a game-space origin: the root named after the model and
    its descendants, and the skinned meshes those joints move. Returns (root node, sorted node indices) or None."""
    stem = os.path.splitext(os.path.basename(model))[0]
    want = [origin[1] * env.UNIT, origin[2] * env.UNIT, origin[0] * env.UNIT]
    children = {c for n in j['nodes'] for c in n.get('children', ())}
    for i, node in enumerate(j['nodes']):
        if i in children or node.get('name') != stem or 'mesh' in node:
            continue
        at = node['matrix'][12:15] if 'matrix' in node else node.get('translation', (0, 0, 0))
        if max(abs(a - b) for a, b in zip(at, want)) > 0.01:
            continue
        found, stack = set(), [i]
        while stack:
            k = stack.pop()
            found.add(k)
            stack += j['nodes'][k].get('children', ())
        skins = {s for s, skin in enumerate(j.get('skins', ())) if set(skin['joints']) <= found}
        found |= {k for k, n in enumerate(j['nodes']) if n.get('skin') in skins}
        return node, sorted(found)
    return None


def idle_anim(ent):
    """The clip a prop loops in game: its starting clip when that loops (fans, bounce pads, the shops' keepers; it
    never hands over to the idle), else its idle."""
    if not ent.get('model'):
        return None
    if ent.get('startinganim') and ent.get('startinganimationloopmode') == 'ANIM_LOOP_MODE_LOOPING':
        return ent['startinganim']
    return ent.get('idleanim') or ent.get('defaultanim')


def built_stamp(out, slug):
    """The stamp of the last build in <out>, or None."""
    m = MapFiles(slug, out, out)
    if not os.path.exists(m.blend):
        return None
    try:
        return json.load(open(m.stamp_file, encoding='utf-8'))
    except (OSError, ValueError):
        return None


class MapBuilder(Builder):
    def __init__(self, args):
        super().__init__(args)
        self.mods = getattr(args, 'mods', True)

    def export(self, m):
        if os.path.exists(m.world):
            return
        os.makedirs(m.dir, exist_ok=True)
        free = shutil.disk_usage(m.dir).free / 1e9
        if free < NEEDED_GB:
            raise BuildError(f'building {m.title} needs about {NEEDED_GB} GB free, {free:.0f} GB left on the drive '
                             f'with {m.dir}')
        vanilla = os.path.join(m.dir, 'vanilla')
        if self.mods:
            vpk, game = map_vpk(self.pak, m.slug), []
        else:
            vpk, gameinfo = vanilla_vpk(self.pak, m.slug, vanilla)
            game = ['--game', gameinfo]
        try:
            self.run([self.s2v, '-i', vpk, '-f', f'maps/{m.slug}/world.vwrld_c', '-o', m.export, *game,
                      '--gltf_export_format', 'gltf', '--gltf_textures_adapt', '--gltf_export_extras'], m, show=False,
                     count=('Creating mesh', EXPORT_MESHES.get(m.slug, 10000), 1.0))
        finally:
            shutil.rmtree(vanilla, ignore_errors=True)
        if not os.path.exists(m.world):
            raise BuildError('Source2Viewer wrote no ' + m.world)

    def entities(self, m, vpks):
        world = self.dump(m, f'maps/{m.slug}/world.vwrld', vpks[:1])
        return [e for lump in env.resources(world, 'vents') for e in env.entities(self.dump(m, lump, vpks[:1]))]

    def props(self, m):
        shutil.rmtree(m.props, ignore_errors=True)
        os.makedirs(m.props)
        vpks = [map_vpk(self.pak, m.slug), self.pak]
        ents = [e for e in self.entities(m, vpks) if idle_anim(e)]
        world = json.load(open(m.world, encoding='utf-8'))
        vanilla = os.path.join(m.dir, 'vanilla')
        game = [] if self.mods else ['--game', vanilla_vpk(self.pak, m.slug, vanilla)[1]]
        props, exclude, exported = [], set(), {}
        try:
            for ent in ents:
                found = prop_nodes(world, ent['model'], ent['origin'])
                if not found:
                    continue
                root, nodes = found
                key = (ent['model'], idle_anim(ent))
                if key not in exported:
                    exported[key] = self.export_prop(m, *key, len(exported), game)
                if not exported[key]:
                    continue
                props.append({'model': ent['model'], 'anim': idle_anim(ent), 'gltf': exported[key],
                              'transform': {k: root[k] for k in ('matrix', 'translation', 'rotation', 'scale')
                                            if k in root}})
                exclude.update(nodes)
        finally:
            shutil.rmtree(vanilla, ignore_errors=True)
        json.dump({'props': props, 'exclude': sorted(exclude)}, open(m.props_file, 'w', encoding='utf-8'), indent=1)
        log(f'  props: {len(props)} animated, {len(exported)} models')

    def export_prop(self, m, model, anim, n, game):
        """The model with only its idle clip as glTF; the path relative to the props folder, or None if the model
        has no such clip."""
        rel = os.path.join(str(n), os.path.splitext(os.path.basename(model))[0] + '.gltf')
        path = os.path.join(m.props, rel)
        try:
            self.run([self.s2v, '-i', self.pak, '-f', model + '_c', '-o', path, *game, '--gltf_export_format', 'gltf',
                      '--gltf_textures_adapt', '--gltf_export_extras', '--gltf_animation_list', anim], m, show=False)
        except BuildError:
            return None
        try:
            anims = json.load(open(path, encoding='utf-8')).get('animations', ())
        except (OSError, ValueError):
            return None
        return rel if any(a.get('name') == anim for a in anims) else None

    def dump(self, m, path, vpks):
        """Source2Viewer's text dump of a resource's data, from the first of the archives that has it."""
        out = os.path.join(m.dir, 'dump.txt')
        for vpk in vpks:
            try:
                self.run([self.s2v, '-i', vpk, '-f', path + '_c', '-b', 'DATA'], m, show=False, stdout_file=out)
            except BuildError:
                continue
            text = open(out, encoding='utf-8').read()
            if 'Resource Type' in text:
                return text
        return ''

    def decompile(self, m, path, vpks, dest=None):
        dest = dest or m.lighting

        def count():
            return sum(len(fs) for _, _, fs in os.walk(dest))
        before = count()
        for vpk in vpks:
            # Source2Viewer exits 0 when the archive doesn't have the file
            try:
                self.run([self.s2v, '-i', vpk, '-f', path + '_c', '-o', dest, '-d'], m, show=False)
            except BuildError:
                continue
            if count() > before:
                return True
        return False

    def lighting(self, m):
        shutil.rmtree(m.lighting, ignore_errors=True)
        os.makedirs(m.lighting)
        vpks = [map_vpk(self.pak, m.slug), self.pak]
        self.run([self.s2v, '-i', vpks[0], '-f', f'maps/{m.slug}/lightmaps/directional_irradiance_sh2_',
                  '-o', m.lighting, '-d'], m, show=False)
        ents = self.entities(m, vpks)
        self.write_npcs(m, ents)
        settings = {'sun': env.sun(ents), 'lights': env.lights(ents), 'gradient_fog': env.gradient_fog(ents)}
        cube = env.cubemap_fog(ents)
        settings['cubemap_fog'] = cube
        sky = env.sky(ents, cube and cube['sky_entity'])
        if sky:
            texture, bias = env.sky_texture(self.dump(m, sky['material'], vpks))
            sky_dir = os.path.join(m.lighting, 'sky')
            if texture and self.decompile(m, texture, vpks, sky_dir):
                found = [os.path.join(d, f) for d, _, fs in os.walk(sky_dir) for f in fs if f.endswith('.exr')]
                if found:
                    os.replace(found[0], os.path.join(m.lighting, 'sky.exr'))
                    shutil.rmtree(sky_dir, ignore_errors=True)
            settings['sky'] = {'strength': sky['brightness'] * 2 ** bias}
        vpost, exposure = env.post_processing(ents)
        settings['tonemap'] = env.tonemap(self.dump(m, vpost, vpks)) if vpost else None
        settings['exposure'] = exposure
        volumes = env.probe_volumes(ents)
        textures = {v['texture'] for v in volumes}
        # the game's shaders read one atlas; volumes with textures of their own aren't brought over
        if len(textures) > 1:
            atlas = next((t for t in textures if t.endswith('_atlas.vtex')), sorted(textures)[0])
            volumes = [v for v in volumes if v['texture'] == atlas]
        if volumes and self.decompile(m, volumes[0]['texture'], vpks[:1]):
            settings['probe_volumes'] = volumes
        env_maps = env.env_maps(ents)
        if env_maps:
            # every cubemap of a map is a slice of one array texture
            texture = max({e['texture'] for e in env_maps}, key=[e['texture'] for e in env_maps].count)
            env_maps = [e for e in env_maps if e['texture'] == texture]
            core = os.path.join(os.path.dirname(os.path.dirname(self.pak)), 'core', 'pak01_dir.vpk')
            if self.decompile(m, texture, vpks[:1], os.path.join(m.lighting, 'cubemaps')) and \
                    self.decompile(m, 'textures/dev/brdf_lut.vtex', [core, self.pak], os.path.join(m.lighting, 'brdf')):
                settings['env_maps'] = env_maps
                # Source2Viewer decompiles only the top mip; the reflections read the game's blurred mips from this
                self.run([self.s2v, '-i', vpks[0], '-f', texture + '_c', '-o', os.path.join(m.lighting, 'cubemaps')],
                         m, show=False)
        fog = env.volumetric_fog(ents)
        if fog and self.decompile(m, fog['texture'], vpks[:1], os.path.join(m.lighting, 'volumetric_fog')):
            settings['volumetric_fog'] = fog
        settings['skins'] = self.skins(m, ents, vpks)
        settings['glass_masks'] = self.glass_masks(m, vpks)
        settings['layer_textures'] = self.layer_textures(m, vpks)
        settings['anim_textures'] = self.anim_textures(m, vpks)
        os.remove(os.path.join(m.dir, 'dump.txt'))
        json.dump(settings, open(os.path.join(m.lighting, 'settings.json'), 'w', encoding='utf-8'), indent=1)
        log(f'  lighting: sun {"baked" if settings["sun"] and settings["sun"]["baked"] else "live"}, '
            f'{len(settings["lights"])} runtime lights, {len(settings.get("probe_volumes") or [])} light probe volumes, '
            f'{len(settings.get("env_maps") or [])} cubemaps, '
            f'sky {"yes" if "sky" in settings else "no"}, fog {"cubemap " if cube else ""}'
            f'{"gradient " if settings["gradient_fog"] else ""}'
            f'{"volumetric" if settings.get("volumetric_fog") else ""}')

    def skins(self, m, ents, vpks):
        out = []
        for e in env.prop_skins(ents):
            swap = env.skin_materials(env.material_groups(self.dump(m, e['model'], vpks)), e['skin'])
            if swap:
                out.append({'model': e['model'], 'origin': e['origin'], 'materials': swap})
        return out

    def write_npcs(self, m, ents):
        os.makedirs(m.out, exist_ok=True)
        npcs = map_npcs.spawns(ents)
        json.dump({'npcs': npcs}, open(m.npcs_file, 'w', encoding='utf-8'), indent=1)
        log(f'  npcs: {len(npcs)} spawns')
        colors = env.lane_colors(self.dump(m, 'scripts/generic_data.vdata', [self.pak]))
        ziplines = env.ziplines(ents, colors)
        json.dump({'ziplines': ziplines}, open(m.ziplines_file, 'w', encoding='utf-8'))
        log(f'  ziplines: {len(ziplines)}')

    def npcs_only(self, slug):
        """Just <map>/npcs.json and ziplines.json, for a map built before the build wrote them."""
        work = os.environ.get('DLRIG_WORK') or os.path.join(os.path.dirname(TOOLS), 'heroes')
        out = os.environ.get('DLRIG_OUT') or os.path.join(work, 'out')
        m = MapFiles(slug, work, out)
        os.makedirs(m.dir, exist_ok=True)
        self.write_npcs(m, self.entities(m, [map_vpk(self.pak, slug), self.pak]))
        os.remove(os.path.join(m.dir, 'dump.txt'))
        log(f'DONE {slug} {m.npcs_file}')

    def glass_masks(self, m, vpks):
        """The glass mask textures (g_tGlass) of F_GLASS materials that don't give it as a constant; the glTF export
        leaves them out, no glTF input takes them. Texture path -> file name in the lighting folder."""
        paths = set()
        for mat in json.load(open(m.world, encoding='utf-8')).get('materials', []):
            vmat = (mat.get('extras') or {}).get('vmat') or {}
            if (vmat.get('IntParams') or {}).get('F_GLASS') and \
                    'TextureGlassMask1' not in (vmat.get('VectorParams') or {}):
                paths.add((vmat.get('TextureParams') or {}).get('g_tGlass'))
        paths.discard(None)
        dest = os.path.join(m.lighting, 'glass')
        return {p: os.path.splitext(os.path.basename(p))[0] + '.png' for p in sorted(paths)
                if self.decompile(m, p, vpks, dest)}

    def layer_textures(self, m, vpks):
        """The textures of environment_blend's upper layers, reveal masks and overlays, of the parallax windows'
        interiors and of the cosmic veils, which the glTF export leaves out. Texture path -> file under the
        lighting folder."""
        paths = set()
        for mat in json.load(open(m.world, encoding='utf-8')).get('materials', []):
            vmat = (mat.get('extras') or {}).get('vmat') or {}
            if vmat.get('ShaderName') == 'environment_blend.vfx':
                textures = vmat.get('TextureParams') or {}
                paths.update(textures[k] for k in env.BLEND_TEXTURES if textures.get(k))
            elif vmat.get('ShaderName') == 'environment_parallax_layered.vfx':
                textures = vmat.get('TextureParams') or {}
                paths.update(textures[k] for k in env.PARALLAX_TEXTURES if textures.get(k))
            elif vmat.get('ShaderName') == 'cosmic_veil.vfx':
                textures = vmat.get('TextureParams') or {}
                paths.add(textures.get('g_tMasks'))
        paths.discard(None)
        return self.decompile_textures(m, vpks, paths, 'layers')

    def anim_textures(self, m, vpks):
        """The jitter masks and second translucency textures of materials that move (map_anim.py), of the world and
        the animated props. Texture path -> file under the lighting folder."""
        gltfs = [m.world]
        if os.path.exists(m.props_file):
            gltfs += [os.path.join(m.props, p['gltf'])
                      for p in json.load(open(m.props_file, encoding='utf-8'))['props']]
        paths = set()
        for gltf in gltfs:
            for mat in json.load(open(gltf, encoding='utf-8')).get('materials', []):
                vmat = (mat.get('extras') or {}).get('vmat') or {}
                ints, textures = vmat.get('IntParams') or {}, vmat.get('TextureParams') or {}
                if ints.get('F_JITTER_VERTICES'):
                    paths.add(textures.get('g_tJitterMask'))
                if ints.get('F_ADVANCED_TRANSLUCENCY'):
                    paths.add(textures.get('g_tAltTranslucency'))
        paths.discard(None)
        return self.decompile_textures(m, vpks, paths, 'anim')

    def decompile_textures(self, m, vpks, paths, folder):
        dest = os.path.join(m.lighting, folder)
        wanted = sorted(paths)

        # Source2Viewer writes the game's folder tree, which in a deep work folder goes past Windows' 260 character
        # MAX_PATH; Python reads those but Blender's image loader can't, so each file moves to a short name
        def flat(p):
            return os.path.join(dest, hashlib.sha1(p.encode()).hexdigest()[:16] + '.png')
        for vpk in vpks:
            missing = [p for p in wanted if not os.path.exists(flat(p))]
            if not missing:
                break
            # Source2Viewer exits 0 when the archive doesn't have a file
            try:
                self.run([self.s2v, '-i', vpk, '-f', ','.join(p + '_c' for p in missing), '-o', dest, '-d',
                          '--threads', str(os.cpu_count() or 4), '-q'], m, show=False)
            except BuildError:
                continue
            for p in missing:
                tree = os.path.join(dest, os.path.splitext(p)[0] + '.png')
                if os.path.exists(tree):
                    os.replace(tree, flat(p))
        return {p: os.path.relpath(flat(p), m.lighting) for p in wanted if os.path.exists(flat(p))}

    def rebatch(self, m):
        self.run([self.python, os.path.join(TOOLS, 'map_rebatch.py'), m.world, m.rebatched,
                  '--exclude', m.props_file], m)

    def import_(self, m):
        os.makedirs(m.out, exist_ok=True)
        tmp = m.blend[:-len('.blend')] + '.new.blend'
        meshes = sum('mesh' in n for n in json.load(open(m.rebatched, encoding='utf-8'))['nodes'])
        self.run([self.blender, '-b', '--factory-startup', '--python-exit-code', '1',
                  '-P', os.path.join(TOOLS, 'map_import.py'), '--', m.rebatched, tmp, m.slug, m.title,
                  game_version(self.pak), m.lighting, m.props_file, install_dir(self.pak)], m, env=self.blender_env(m),
                 count=('Blender create Mesh node', meshes, GLTF_IMPORT_END))
        if not os.path.exists(tmp):
            raise BuildError('map_import.py wrote no ' + tmp)
        os.replace(tmp, m.blend)
        for f in (tmp + '1', m.blend + '1'):
            if os.path.exists(f):
                os.remove(f)

    def build_map(self, slug, force=False):
        work = os.environ.get('DLRIG_WORK') or os.path.join(os.path.dirname(TOOLS), 'heroes')
        out = os.environ.get('DLRIG_OUT') or os.path.join(work, 'out')
        m = MapFiles(slug, work, out)
        stamp = game_stamp(self.pak, slug, self.mods)
        if not force and built_stamp(out, slug) == stamp:
            log(f'{m.title} is up to date (game version {stamp["game_version"]})')
            log(f'DONE {slug} {m.blend}')
            return
        os.makedirs(m.dir, exist_ok=True)
        open(os.path.join(m.dir, 'build.log'), 'w').close()
        log(f'== {m.title} ({slug}), game version {stamp["game_version"]}'
            + ('' if self.mods else ', without mods'))
        # a game update can leave an older export behind
        shutil.rmtree(m.export, ignore_errors=True)
        steps = [('export', self.export), ('props', self.props), ('rebatch', self.rebatch),
                 ('lighting', self.lighting), ('import', self.import_)]
        done = 0.0
        for i, (name, step) in enumerate(steps):
            self.begin_step(i, len(steps), 'map_' + name, done, done + STEP_WEIGHTS[name])
            step(m)
            done += STEP_WEIGHTS[name]
        json.dump(stamp, open(m.stamp_file, 'w', encoding='utf-8'))
        if not self.keep:
            shutil.rmtree(os.path.join(m.dir, 'export'), ignore_errors=True)
            shutil.rmtree(m.lighting, ignore_errors=True)
            shutil.rmtree(m.props, ignore_errors=True)
            shutil.rmtree(os.path.dirname(m.rebatched), ignore_errors=True)
        log(f'DONE {slug} {m.blend}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map', choices=sorted(MAPS))
    ap.add_argument('--blender')
    ap.add_argument('--s2v')
    ap.add_argument('--s2v-cache', help='where a downloaded Source2Viewer-CLI is kept')
    ap.add_argument('--pak', help="Deadlock's folder or its game/citadel/pak01_dir.vpk")
    ap.add_argument('--python', help='Python with numpy (default: this one)')
    ap.add_argument('--work', help='build files (default: heroes/)')
    ap.add_argument('--out', help='the map goes in <out>/maps/<map>/ (default: heroes/out/)')
    ap.add_argument('--keep', action='store_true', help='keep the export and the rebatched copy')
    ap.add_argument('--force', action='store_true', help='rebuild even if the map is up to date')
    ap.add_argument('--npcs', action='store_true', help="only write the map's npcs.json and ziplines.json")
    ap.add_argument('--no-mods', dest='mods', action='store_false',
                    help="leave out the mods in Deadlock's addons folders (they are in by default, as in the game)")
    args = ap.parse_args(argv)
    args.exports = False
    args.backup = False
    try:
        if args.npcs:
            MapBuilder(args).npcs_only(args.map)
        else:
            MapBuilder(args).build_map(args.map, args.force)
    except (BuildError, subprocess.CalledProcessError, OSError) as e:
        log(f'ERROR {args.map}: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
