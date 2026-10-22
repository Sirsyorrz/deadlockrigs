"""Hero registry (heroes.json) and the per-hero file layout.

Build files go under DLRIG_WORK/<slug>/ (default heroes/<slug>/), finished files under DLRIG_OUT/<slug>/ when
DLRIG_OUT is set, else heroes/<slug>/out/."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# values the rig takes from the game clips (held-prop grips, twist weights), kept so builds without clips match
RIGDATA = os.path.join(ROOT, 'rigdata')


def heroes():
    return json.load(open(os.path.join(ROOT, 'heroes.json')))


class Hero:
    def __init__(self, slug):
        cfg = next((h for h in heroes() if h['slug'] == slug), None)
        if cfg is None:
            sys.exit(f'unknown hero {slug!r}; see heroes.json')
        self.cfg = cfg
        self.slug = slug
        self.name = cfg['name']
        self.model = cfg['model']
        self.dir = os.path.join(os.environ.get('DLRIG_WORK') or os.path.join(ROOT, 'heroes'), slug)
        stem = self.model[:-len('.vmdl')]
        self.raw_glb = os.path.join(self.dir, 'raw', stem + '.glb')
        self.anim_glb = os.path.join(self.dir, 'anim', stem + '.glb')
        self.vmdl = os.path.join(self.dir, 'dec', self.model)
        self.data = os.path.join(self.dir, 'data')
        out_root = os.environ.get('DLRIG_OUT')
        self.out = os.path.join(out_root, slug) if out_root else os.path.join(self.dir, 'out')
        self.face = cfg.get('face')
        self.rename = cfg.get('rename', {})
        self.drop = set(cfg.get('drop', []))
        self.cloth_links = [tuple(p) for p in cfg.get('cloth_links', [])]

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    @property
    def base_blend(self):
        return os.path.join(self.dir, 'build', f'{self.slug}.blend')

    @property
    def rig_blend(self):
        return os.path.join(self.out, f'{self.slug}_rig.blend')

    @property
    def anims_blend(self):
        return os.path.join(self.out, f'{self.slug}_anims.blend')

    @property
    def rigdata(self):
        return os.path.join(RIGDATA, f'{self.slug}.json')


def from_argv():
    """The hero slug: first argument after '--' when run inside Blender, else argv[1]."""
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    return Hero(args[0]), args[1:]
