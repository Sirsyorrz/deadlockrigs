"""Packages the Blender add-on and the Source2Viewer-CLI builds it downloads.

python tools/release.py addon [patch|minor|major|X.Y.Z]
                                dist/deadlock_rigs-<version>-<platform>.zip, one per platform (scipy wheels differ);
                                bumps the version in addon/blender_manifest.toml first if given one
python tools/release.py s2v     dist/s2v/<runtime>.zip for each platform, to upload to the GitHub release
                                tagged pipeline.S2V_TAG (needs git and the .NET SDK)

Both need Blender ($BLENDER or found like pipeline.py does) only for the addon target."""
import os
import re
import shutil
import subprocess
import sys
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import pipeline  # noqa: E402

DIST = os.path.join(ROOT, 'dist')
# what the build runs; the rest of tools/ is for developing and checking rigs
ADDON_TOOLS = ['pipeline.py', 'hero.py', 'vmdl_extract.py', 'gamerig.py', 'fe_extract.py', 'dump.py', 'keys.py',
               'facegeo.py', 'morphs.py', 'bonekeys.py', 'build.py', 'bodygroups.py', 'collar.py', 'tongue_mesh.py',
               'rig.py', 'rig_ui.py', 'expressions.py', 'hands.py', 'mirror.py', 'anims.py', 'mapbuild.py',
               'map_rebatch.py', 'map_import.py', 'map_lighting.py', 'map_environment.py', 'map_blend.py', 'map_fog.py']
SCIPY = 'scipy>=1.16,<1.17'
# Blender 5.1+ runs Python 3.13
WHEEL_PLATFORMS = {
    'linux-x64': ['manylinux_2_17_x86_64', 'manylinux2014_x86_64'],
    'windows-x64': ['win_amd64'],
    'macos-arm64': ['macosx_14_0_arm64'],
    'macos-x64': ['macosx_10_14_x86_64'],
}
S2V_RIDS = ['linux-x64', 'win-x64', 'osx-arm64', 'osx-x64']


MANIFEST = os.path.join(ROOT, 'addon', 'blender_manifest.toml')
VERSION_LINE = re.compile(r'^version = "([^"]+)"', re.M)


def version():
    text = open(MANIFEST, encoding='utf-8').read()
    return VERSION_LINE.search(text).group(1)


def bump(how):
    old = version()
    if re.fullmatch(r'\d+\.\d+\.\d+', how):
        new = how
    elif how in ('major', 'minor', 'patch'):
        major, minor, patch = map(int, old.split('.'))
        new = {'major': f'{major + 1}.0.0', 'minor': f'{major}.{minor + 1}.0',
               'patch': f'{major}.{minor}.{patch + 1}'}[how]
    else:
        sys.exit(f'bad version bump {how!r}: use patch, minor, major or X.Y.Z')
    text = open(MANIFEST, encoding='utf-8').read()
    open(MANIFEST, 'w', encoding='utf-8').write(VERSION_LINE.sub(f'version = "{new}"', text, count=1))
    print(f'version {old} -> {new}')


def download_wheels(dest):
    os.makedirs(dest, exist_ok=True)
    pip = [sys.executable, '-m', 'pip']
    if subprocess.run(pip + ['--version'], capture_output=True).returncode != 0:
        pip = ['uvx', 'pip']
    names = []
    for plats in WHEEL_PLATFORMS.values():
        before = set(os.listdir(dest))
        cmd = pip + ['download', SCIPY, '--only-binary=:all:', '--python-version', '3.13', '--no-deps', '-q',
                     '-d', dest]
        for p in plats:
            cmd += ['--platform', p]
        subprocess.run(cmd, check=True)
        new = sorted(set(os.listdir(dest)) - before) or [f for f in os.listdir(dest) if any(p in f for p in plats)]
        names += new
    return sorted(set(names))


def addon():
    stage = os.path.join(ROOT, 'build', 'addon')
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, 'tools'))
    for f in ('__init__.py', 'crop.py', 'look.py'):
        shutil.copy(os.path.join(ROOT, 'addon', f), stage)
    for f in ADDON_TOOLS:
        shutil.copy(os.path.join(TOOLS, f), os.path.join(stage, 'tools'))
    shutil.copy(os.path.join(ROOT, 'heroes.json'), stage)
    shutil.copy(os.path.join(ROOT, 'LICENSE'), stage)
    shutil.copytree(os.path.join(ROOT, 'rigdata'), os.path.join(stage, 'rigdata'))
    wheels = download_wheels(os.path.join(stage, 'wheels'))
    manifest = open(MANIFEST, encoding='utf-8').read()
    # top-level keys have to come before the first [table]
    top = ('platforms = [' + ', '.join(f'"{p}"' for p in WHEEL_PLATFORMS) + ']\n'
           'wheels = [\n' + ''.join(f'  "./wheels/{w}",\n' for w in wheels) + ']\n\n')
    i = manifest.index('\n[') + 1
    manifest = manifest[:i] + top + manifest[i:]
    manifest += '\n[build]\npaths_exclude_pattern = ["__pycache__/", "*.pyc"]\n'
    open(os.path.join(stage, 'blender_manifest.toml'), 'w', encoding='utf-8').write(manifest)
    os.makedirs(DIST, exist_ok=True)
    blender = pipeline.find_blender()
    subprocess.run([blender, '--factory-startup', '--command', 'extension', 'build', '--source-dir', stage,
                    '--output-dir', DIST, '--split-platforms'], check=True)
    for f in sorted(os.listdir(DIST)):
        if f.endswith('.zip') and version() in f:
            print('wrote', os.path.join(DIST, f))


def s2v():
    src = os.path.join(ROOT, 'build', 's2v-src')
    if not os.path.isdir(os.path.join(src, '.git')):
        subprocess.run(['git', 'clone', '-q', '--filter=blob:none', pipeline.S2V_REPO, src], check=True)
    subprocess.run(['git', '-C', src, 'fetch', '-q', 'origin', pipeline.S2V_COMMIT], check=True)
    subprocess.run(['git', '-C', src, 'checkout', '-q', '--detach', pipeline.S2V_COMMIT], check=True)
    out = os.path.join(DIST, 's2v')
    os.makedirs(out, exist_ok=True)
    for rid in S2V_RIDS:
        pub = os.path.join(ROOT, 'build', 's2v-' + rid)
        shutil.rmtree(pub, ignore_errors=True)
        subprocess.run(['dotnet', 'publish', os.path.join(src, 'CLI', 'CLI.csproj'), '-c', 'Release',
                        '--self-contained', '-r', rid, '-p:RunAnalyzersDuringBuild=false', '-p:DebugType=none',
                        '-o', pub], check=True, stdout=subprocess.DEVNULL)
        zpath = os.path.join(out, rid + '.zip')
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(pub)):
                info = zipfile.ZipInfo.from_file(os.path.join(pub, f), f)
                # keep the executable bit for Linux/macOS unzip tools; pipeline.py sets it again itself
                info.external_attr = (0o755 if not f.endswith(('.txt', '.dll')) else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, open(os.path.join(pub, f), 'rb').read())
        print('wrote', zpath)
    print(f'upload them to the release tagged {pipeline.S2V_TAG}:\n'
          f'  gh release create {pipeline.S2V_TAG} {out}/*.zip --title "Source2Viewer-CLI {pipeline.S2V_COMMIT[:9]}"')


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else ''
    if target == 'addon':
        if len(sys.argv) > 2:
            bump(sys.argv[2])
        addon()
    elif target == 's2v':
        s2v()
    else:
        sys.exit(__doc__)
