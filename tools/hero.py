"""Hero registry (heroes.json) and the per-hero file layout under heroes/<slug>/."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Hero:
    def __init__(self, slug):
        heroes = json.load(open(os.path.join(ROOT, 'heroes.json')))
        cfg = next((h for h in heroes if h['slug'] == slug), None)
        if cfg is None:
            sys.exit(f'unknown hero {slug!r}; see heroes.json')
        self.cfg = cfg
        self.slug = slug
        self.name = cfg['name']
        self.model = cfg['model']
        self.dir = os.path.join(ROOT, 'heroes', slug)
        stem = self.model[:-len('.vmdl')]
        self.raw_glb = os.path.join(self.dir, 'raw', stem + '.glb')
        self.anim_glb = os.path.join(self.dir, 'anim', stem + '.glb')
        self.vmdl = os.path.join(self.dir, 'dec', self.model)
        self.data = os.path.join(self.dir, 'data')
        self.out = os.path.join(self.dir, 'out')
        self.face = cfg.get('face')
        self.rename = cfg.get('rename', {})
        self.drop = set(cfg.get('drop', []))
        self.cloth_links = [tuple(p) for p in cfg.get('cloth_links', [])]

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    @property
    def base_blend(self):
        return os.path.join(self.out, f'{self.slug}.blend')

    @property
    def rig_blend(self):
        return os.path.join(self.out, f'{self.slug}_rig.blend')

    @property
    def anims_blend(self):
        return os.path.join(self.out, f'{self.slug}_anims.blend')


def from_argv():
    """The hero slug: first argument after '--' when run inside Blender, else argv[1]."""
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    return Hero(args[0]), args[1:]
