"""Builds Deadlock hero rigs from the game files. Runs on Windows, Linux and macOS; the Blender add-on calls it too.

python tools/pipeline.py <hero|all> [steps...] [options]

steps (default: extract data base rig):
  extract       model, textures and the decompiled vmdl from the game (Source2Viewer)
  data          cloth/physics data and the face keys
  base          the model in Blender with face keys (build/<hero>.blend)
  rig           the rig (<out>/<hero>_rig.blend)
  anims         the game clips as an action library next to the rig (<hero>_anims.blend), for the rig's Game
                Animations panel and the tests; saves the grips and twist weights it samples to rigdata/
  test          checks the rig (needs anims for most checks); writes <work>/<hero>/test.log
  export_anims  the game clips as one glb (<out>/<hero>_animations.glb) and as the rig's action library (anims)
  clean         deletes the hero's build files
  dev           extract data base anims rig test

Found automatically unless given: Blender (--blender, $BLENDER), Deadlock's pak01_dir.vpk (--pak, $PAK) and
Source2Viewer-CLI (--s2v, $S2V; downloaded once otherwise).
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import hero as hero_mod  # noqa: E402

# Source2Viewer releases (up to 20.1) leave the vmdl's cloth chains and capsules out of the decompile, so builds use a
# pinned master commit. release.py publishes its CLI for each platform under this tag.
S2V_COMMIT = 'f030654ff81b352fd1bcb44d631c563f51c7dcd1'
S2V_TAG = 's2v-' + S2V_COMMIT[:9]
S2V_URL = os.environ.get('S2V_URL', f'https://github.com/Sirsyorrz/deadlockrigs/releases/download/{S2V_TAG}/{{rid}}.zip')
S2V_REPO = 'https://github.com/ValveResourceFormat/ValveResourceFormat'

DEFAULT_STEPS = ['extract', 'data', 'base', 'rig']
DEV_STEPS = ['extract', 'data', 'base', 'anims', 'rig', 'test']
ALL_STEPS = DEV_STEPS + ['export_anims', 'clean']
# Blender output worth showing; the rest goes to the build log
SHOW = re.compile(r'^(meshes|baked light|reflections:|fog:|materials:|shape keys|wrote|mesh without|collar|IK |skipping|WARNING|twist|mirror|clips|'
                  r'(morph|bone) keys|dumped|tails|PASS|FAIL)|Error|Traceback')


class BuildError(Exception):
    pass


def log(msg):
    print(msg, flush=True)


def exe(name):
    return name + '.exe' if os.name == 'nt' else name


def runtime_id():
    machine = platform.machine().lower()
    arch = 'arm64' if machine in ('arm64', 'aarch64') else 'x64'
    osname = {'Windows': 'win', 'Darwin': 'osx'}.get(platform.system(), 'linux')
    return f'{osname}-{arch}'


def user_data_dir():
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~/AppData/Local')
    elif platform.system() == 'Darwin':
        base = os.path.expanduser('~/Library/Application Support')
    else:
        base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
    return os.path.join(base, 'deadlockrigs')


# ---------------------------------------------------------------- finding things

def steam_roots():
    roots = []
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Valve\Steam') as k:
                roots.append(winreg.QueryValueEx(k, 'SteamPath')[0])
        except OSError:
            pass
        roots += [r'C:\Program Files (x86)\Steam', r'C:\Program Files\Steam']
    elif platform.system() == 'Darwin':
        roots.append(os.path.expanduser('~/Library/Application Support/Steam'))
    else:
        roots += [os.path.expanduser(p) for p in ('~/.local/share/Steam', '~/.steam/steam', '~/.steam/root',
                                                  '~/.var/app/com.valvesoftware.Steam/.local/share/Steam',
                                                  '~/snap/steam/common/.local/share/Steam')]
    return [r for r in roots if os.path.isdir(r)]


def steam_libraries():
    libs = []
    for root in steam_roots():
        libs.append(root)
        vdf = os.path.join(root, 'steamapps', 'libraryfolders.vdf')
        if os.path.exists(vdf):
            text = open(vdf, encoding='utf-8', errors='replace').read()
            libs += [p.replace('\\\\', '\\') for p in re.findall(r'"path"\s+"([^"]+)"', text)]
    seen, out = set(), []
    for lib in libs:
        key = os.path.normcase(os.path.realpath(lib))
        if key not in seen and os.path.isdir(lib):
            seen.add(key)
            out.append(lib)
    return out


def find_pak(given=None):
    if given:
        if os.path.isdir(given):
            given = os.path.join(given, 'game', 'citadel', 'pak01_dir.vpk')
        if not os.path.exists(given):
            raise BuildError(f'no pak01_dir.vpk at {given}')
        return given
    for lib in steam_libraries():
        pak = os.path.join(lib, 'steamapps', 'common', 'Deadlock', 'game', 'citadel', 'pak01_dir.vpk')
        if os.path.exists(pak):
            return pak
    raise BuildError('Deadlock not found in any Steam library; give its folder with --pak')


def find_blender(given=None):
    cands = [given, os.environ.get('BLENDER'), shutil.which('blender')]
    for lib in steam_libraries():
        cands.append(os.path.join(lib, 'steamapps', 'common', 'Blender', exe('blender')))
    if os.name == 'nt':
        cands += sorted(glob.glob(r'C:\Program Files\Blender Foundation\Blender*\blender.exe'), reverse=True)
    elif platform.system() == 'Darwin':
        cands.append('/Applications/Blender.app/Contents/MacOS/Blender')
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise BuildError('Blender not found; give it with --blender')


def find_s2v(given=None, cache=None):
    given = given or os.environ.get('S2V')
    if given:
        if not os.path.isfile(given):
            raise BuildError(f'no Source2Viewer-CLI at {given}')
        return given
    local = os.path.join(ROOT, 's2v', 'bin', exe('Source2Viewer-CLI'))
    if os.path.isfile(local):
        return local
    cache = cache or os.path.join(user_data_dir(), 's2v')
    target = os.path.join(cache, S2V_TAG)
    cli = os.path.join(target, exe('Source2Viewer-CLI'))
    if os.path.isfile(cli):
        return cli
    url = S2V_URL.format(rid=runtime_id())
    log(f'downloading Source2Viewer-CLI ({runtime_id()}) from {url}')
    os.makedirs(target, exist_ok=True)
    zpath = target + '.zip'
    try:
        urllib.request.urlretrieve(url, zpath)
    except OSError as e:
        if shutil.which('dotnet'):
            return build_s2v(target)
        raise BuildError(f'could not download Source2Viewer-CLI from {url} ({e}). Set the path to an existing '
                         'Source2Viewer-CLI in the add-on preferences (or --s2v)') from e
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    os.remove(zpath)
    if not os.path.isfile(cli):
        raise BuildError(f'{url} has no {exe("Source2Viewer-CLI")}')
    os.chmod(cli, os.stat(cli).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def build_s2v(target):
    """From source at the pinned commit (needs the .NET SDK)."""
    src = os.path.join(os.path.dirname(target), 'src')
    log(f'building Source2Viewer-CLI {S2V_COMMIT[:9]} from source')
    if not os.path.isdir(os.path.join(src, '.git')):
        subprocess.run(['git', 'clone', '-q', '--filter=blob:none', S2V_REPO, src], check=True)
    subprocess.run(['git', '-C', src, 'fetch', '-q', 'origin', S2V_COMMIT], check=True)
    subprocess.run(['git', '-C', src, 'checkout', '-q', '--detach', S2V_COMMIT], check=True)
    subprocess.run(['dotnet', 'publish', os.path.join(src, 'CLI', 'CLI.csproj'), '-c', 'Release', '--self-contained',
                    '-r', runtime_id(), '-p:RunAnalyzersDuringBuild=false', '-o', target],
                   check=True, stdout=subprocess.DEVNULL)
    return os.path.join(target, exe('Source2Viewer-CLI'))


# ---------------------------------------------------------------- running things

class Builder:
    def __init__(self, args):
        self.args = args
        self.keep = args.keep
        self._blender = args.blender
        self._s2v = args.s2v
        self._pak = args.pak
        self.python = args.python or sys.executable
        if args.work:
            os.environ['DLRIG_WORK'] = os.path.abspath(args.work)
        if args.out:
            os.environ['DLRIG_OUT'] = os.path.abspath(args.out)
        if args.exports:
            os.environ['DLRIG_EXPORTS'] = '1'

    @property
    def blender(self):
        if not self._blender or not os.path.isfile(self._blender):
            self._blender = find_blender(self._blender)
        return self._blender

    @property
    def s2v(self):
        if not self._s2v or not os.path.isfile(self._s2v):
            self._s2v = find_s2v(self._s2v, self.args.s2v_cache)
        return self._s2v

    @property
    def pak(self):
        if not self._pak or not os.path.isfile(self._pak):
            self._pak = find_pak(self._pak)
        return self._pak

    def run(self, cmd, h, show=True, stdout_file=None, env=None):
        """Runs a tool; its output goes to <work>/<hero>/build.log, the lines worth seeing to the console."""
        os.makedirs(h.dir, exist_ok=True)
        with open(os.path.join(h.dir, 'build.log'), 'a', encoding='utf-8') as logf:
            logf.write('$ ' + ' '.join(cmd) + '\n')
            logf.flush()
            # Windows' default encoding would trip over the game's text files
            env = dict(os.environ, PYTHONUTF8='1', **(env or {}))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                 errors='replace', cwd=ROOT, env=env,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            out = open(stdout_file, 'w', encoding='utf-8') if stdout_file else None
            tail = []
            for line in p.stdout:
                logf.write(line)
                if out:
                    out.write(line)
                tail = (tail + [line.rstrip()])[-20:]
                if show and SHOW.search(line):
                    log('  ' + line.rstrip())
            if out:
                out.close()
            if p.wait() != 0:
                raise BuildError(f'{os.path.basename(cmd[0])} failed (exit {p.returncode}); last output:\n' +
                                 '\n'.join(tail) + f'\nfull log: {os.path.join(h.dir, "build.log")}')

    def blender_env(self, h):
        """An empty user folder for the build's Blender: one sharing the user's would sync extensions on start and,
        with factory settings, uninstall the add-on's wheels (scipy) from under the running Blender."""
        d = os.path.join(os.path.dirname(h.dir), '.blender')
        os.makedirs(d, exist_ok=True)
        for k in [k for k in os.environ if k.startswith('BLENDER_USER_')]:
            del os.environ[k]
        return {'BLENDER_USER_RESOURCES': d}

    def blender_script(self, h, script, blend=None, show=True):
        cmd = [self.blender, '-b', '--factory-startup']
        if blend:
            cmd.append(blend)
        cmd += ['--python-exit-code', '1', '-P', os.path.join(TOOLS, script), '--', h.slug]
        self.run(cmd, h, show, env=self.blender_env(h))

    def python_script(self, h, script, *args):
        self.run([self.python, os.path.join(TOOLS, script), *args], h)

    def s2v_export(self, h, out_dir, *flags, stdout_file=None):
        cmd = [self.s2v, '-i', self.pak, '-f', h.model + '_c']
        if out_dir:
            cmd += ['-o', out_dir]
        self.run(cmd + list(flags), h, show=False, stdout_file=stdout_file)

    # ------------------------------------------------------------ steps

    def extract(self, h):
        if not os.path.exists(h.raw_glb):
            self.s2v_export(h, h.path('raw'), '--gltf_export_format', 'glb', '--gltf_export_materials',
                            '--gltf_textures_adapt', '--gltf_export_extras')
        if not os.path.exists(h.vmdl):
            self.s2v_export(h, h.path('dec'), '-d')
            if not self.keep:
                for dirpath, _dirs, files in os.walk(h.path('dec')):
                    for f in files:
                        if not f.endswith('.vmdl'):
                            os.remove(os.path.join(dirpath, f))
        for f in (h.raw_glb, h.vmdl):
            if not os.path.exists(f):
                raise BuildError(f'Source2Viewer wrote no {f}')

    def data(self, h):
        os.makedirs(h.data, exist_ok=True)
        gamedata, gamerig = os.path.join(h.data, 'gamedata.json'), os.path.join(h.data, 'gamerig.json')
        self.python_script(h, 'vmdl_extract.py', h.vmdl, gamedata)
        self.python_script(h, 'gamerig.py', gamedata, h.raw_glb, gamerig)
        phys = os.path.join(h.data, 'phys.txt')
        self.s2v_export(h, None, '-b', 'PHYS', stdout_file=phys)
        self.python_script(h, 'fe_extract.py', phys, os.path.join(h.data, 'cloth.json'))
        os.remove(phys)
        if h.face:
            builder = h.face.get('builder', 'keys.py')
            if builder == 'keys.py':
                self.blender_script(h, 'dump.py')
                self.python_script(h, 'keys.py', os.path.join(h.data, 'head.npz'), os.path.join(h.data, 'keys.npz'))
            else:
                self.blender_script(h, builder)

    def base(self, h):
        if os.path.exists(h.base_blend):
            os.remove(h.base_blend)
        self.blender_script(h, 'build.py')
        if not os.path.exists(h.base_blend):
            raise BuildError('build.py wrote no ' + h.base_blend)

    def rig(self, h):
        os.makedirs(h.out, exist_ok=True)
        if os.path.exists(h.rig_blend):
            if self.args.backup:
                old = h.rig_blend[:-len('.blend')] + '.old.blend'
                os.replace(h.rig_blend, old)
                log(f'  previous rig kept as {os.path.basename(old)}')
            else:
                os.remove(h.rig_blend)
        self.blender_script(h, 'rig.py', h.base_blend)
        if not os.path.exists(h.rig_blend):
            raise BuildError('rig.py wrote no ' + h.rig_blend)
        if not self.keep:
            for f in glob.glob(os.path.join(h.out, '*.blend1')):
                os.remove(f)

    def anim_glb(self, h):
        if not os.path.exists(h.anim_glb):
            self.s2v_export(h, h.path('anim'), '--gltf_export_format', 'glb', '--gltf_export_animations',
                            '--gltf_compose_additive')
        if not os.path.exists(h.anim_glb):
            raise BuildError('Source2Viewer wrote no ' + h.anim_glb)

    def anims(self, h):
        self.anim_glb(h)
        os.makedirs(h.out, exist_ok=True)
        if os.path.exists(h.anims_blend):
            os.remove(h.anims_blend)
        self.blender_script(h, 'anims.py')
        if not os.path.exists(h.anims_blend):
            raise BuildError('anims.py wrote no ' + h.anims_blend)
        log('  wrote ' + h.anims_blend)
        if not self.keep:
            shutil.rmtree(h.path('anim'), ignore_errors=True)

    def test(self, h):
        test_log = os.path.join(h.dir, 'test.log')
        cmd = [self.blender, '-b', '--factory-startup', h.rig_blend, '--python-exit-code', '1',
               '-P', os.path.join(TOOLS, 'test_rig.py')]
        before = os.path.getsize(os.path.join(h.dir, 'build.log')) if os.path.exists(os.path.join(h.dir, 'build.log')) else 0
        self.run(cmd, h, env=self.blender_env(h))
        with open(os.path.join(h.dir, 'build.log'), encoding='utf-8') as f:
            f.seek(before)
            lines = [ln for ln in f.read().splitlines() if ln.startswith(('PASS', 'FAIL', 'clips'))]
        open(test_log, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    def export_anims(self, h):
        self.anim_glb(h)
        os.makedirs(h.out, exist_ok=True)
        dest = os.path.join(h.out, f'{h.slug}_animations.glb')
        shutil.copyfile(h.anim_glb, dest)
        log('  wrote ' + dest)
        self.anims(h)

    def clean(self, h):
        shutil.rmtree(h.dir, ignore_errors=True)

    def build(self, slug, steps):
        h = hero_mod.Hero(slug)
        os.makedirs(h.dir, exist_ok=True)
        open(os.path.join(h.dir, 'build.log'), 'w').close()
        log(f'== {h.name} ({h.model}): {" ".join(steps)}')
        for i, step in enumerate(steps):
            log(f'STEP {i + 1}/{len(steps)} {step}')
            getattr(self, step)(h)
        log(f'DONE {slug} ' + (h.rig_blend if os.path.exists(h.rig_blend) else h.out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('hero', help="hero slug from heroes.json, or 'all'")
    ap.add_argument('steps', nargs='*', help='steps to run (default: extract data base rig)')
    ap.add_argument('--blender')
    ap.add_argument('--s2v')
    ap.add_argument('--s2v-cache', help='where a downloaded Source2Viewer-CLI is kept')
    ap.add_argument('--pak', help="Deadlock's folder or its game/citadel/pak01_dir.vpk")
    ap.add_argument('--python', help='Python with numpy and scipy for the face keys (default: this one)')
    ap.add_argument('--work', help='build files (default: heroes/)')
    ap.add_argument('--out', help='finished files go in <out>/<hero>/ (default: heroes/<hero>/out/)')
    ap.add_argument('--exports', action='store_true', help='also write the model as .glb and .fbx')
    ap.add_argument('--keep', action='store_true', help='keep intermediate files')
    ap.add_argument('--backup', action='store_true', help='rename an existing rig to <hero>_rig.old.blend')
    ap.add_argument('--list', action='store_true', help='print the heroes as json and exit')
    args = ap.parse_args(argv)
    if args.list:
        print(json.dumps([{'slug': h['slug'], 'name': h['name']} for h in hero_mod.heroes()]))
        return 0
    steps = args.steps or DEFAULT_STEPS
    if steps == ['dev']:
        steps = DEV_STEPS
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        ap.error(f'unknown step {bad[0]}; steps: {" ".join(ALL_STEPS)}')
    slugs = [h['slug'] for h in hero_mod.heroes()] if args.hero == 'all' else [args.hero]
    b = Builder(args)
    failed = []
    for slug in slugs:
        try:
            b.build(slug, steps)
        except (BuildError, subprocess.CalledProcessError, OSError) as e:
            log(f'ERROR {slug}: {e}')
            failed.append(slug)
    if failed:
        log('failed: ' + ' '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
