"""Builds a Blender scene of a Deadlock map from the user's install. Runs on Windows, Linux and macOS; the add-on
calls it too.

python tools/mapbuild.py <map> [options]

  export    the map's world as glTF with textures (Source2Viewer)
  rebatch   one mesh per grid cell and material instead of ~30,000 objects (map_rebatch.py)
  lighting  the baked lighting (lightmaps, light probes) and the sky, sun, fog and tonemapping the map uses
  import    <out>/maps/<map>/<map>.blend, textures in textures/ and lighting files in lightmaps/ next to it

The map is rebuilt only when the game's version or the map's archive changed since the last build (or with --force).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from pipeline import Builder, BuildError, find_pak, log  # noqa: E402
import map_environment as env  # noqa: E402

MAPS = {'dl_midtown': 'Midtown', 'dl_hideout': 'Hideout'}
# the export, rebatched copy and saved scene of Midtown take about 11 GB together
NEEDED_GB = 12


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
        self.blend = os.path.join(self.out, slug + '.blend')
        self.stamp_file = os.path.join(self.out, 'map.json')


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


def game_stamp(pak, slug, mods=True):
    vpk = os.stat(map_vpk(pak, slug))
    stamp = {'game_version': game_version(pak), 'vpk_size': vpk.st_size, 'vpk_mtime': int(vpk.st_mtime)}
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
                      '--gltf_export_format', 'gltf', '--gltf_textures_adapt', '--gltf_export_extras'], m, show=False)
        finally:
            shutil.rmtree(vanilla, ignore_errors=True)
        if not os.path.exists(m.world):
            raise BuildError('Source2Viewer wrote no ' + m.world)

    def dump(self, m, path, vpks):
        """Source2Viewer's text dump of a resource's data, from the first of the archives that has it."""
        out = os.path.join(m.lighting, 'dump.txt')
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
        world = self.dump(m, f'maps/{m.slug}/world.vwrld', vpks[:1])
        ents = [e for lump in env.resources(world, 'vents') for e in env.entities(self.dump(m, lump, vpks[:1]))]
        settings = {'sun': env.sun(ents), 'gradient_fog': env.gradient_fog(ents)}
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
        os.remove(os.path.join(m.lighting, 'dump.txt'))
        json.dump(settings, open(os.path.join(m.lighting, 'settings.json'), 'w', encoding='utf-8'), indent=1)
        log(f'  lighting: sun {"baked" if settings["sun"] and settings["sun"]["baked"] else "live"}, '
            f'{len(settings.get("probe_volumes") or [])} light probe volumes, '
            f'sky {"yes" if "sky" in settings else "no"}, fog {"cubemap " if cube else ""}'
            f'{"gradient" if settings["gradient_fog"] else ""}')

    def rebatch(self, m):
        self.run([self.python, os.path.join(TOOLS, 'map_rebatch.py'), m.world, m.rebatched], m)

    def import_(self, m):
        os.makedirs(m.out, exist_ok=True)
        tmp = m.blend[:-len('.blend')] + '.new.blend'
        self.run([self.blender, '-b', '--factory-startup', '--python-exit-code', '1',
                  '-P', os.path.join(TOOLS, 'map_import.py'), '--', m.rebatched, tmp, m.slug, m.title,
                  game_version(self.pak), m.lighting], m, env=self.blender_env(m))
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
        steps = [('export', self.export), ('rebatch', self.rebatch), ('lighting', self.lighting),
                 ('import', self.import_)]
        for i, (name, step) in enumerate(steps):
            log(f'STEP {i + 1}/{len(steps)} map_{name}')
            step(m)
        json.dump(stamp, open(m.stamp_file, 'w', encoding='utf-8'))
        if not self.keep:
            shutil.rmtree(os.path.join(m.dir, 'export'), ignore_errors=True)
            shutil.rmtree(m.lighting, ignore_errors=True)
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
    ap.add_argument('--no-mods', dest='mods', action='store_false',
                    help="leave out the mods in Deadlock's addons folders (they are in by default, as in the game)")
    args = ap.parse_args(argv)
    args.exports = False
    args.backup = False
    try:
        MapBuilder(args).build_map(args.map, args.force)
    except (BuildError, subprocess.CalledProcessError, OSError) as e:
        log(f'ERROR {args.map}: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
