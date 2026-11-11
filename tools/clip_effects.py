"""python3 clip_effects.py <hero slug | model path> [clip ...]

The particle effects a hero's animation clips start in the game, read from the user's install (nothing extracted):

- the clip's own particle events (CNmParticleEvent: reloads, dashes, Paige's Excalibur, Solomon's pieces), started
  at a fraction of the clip and placed by the effect's "preview" control point setup, or by the event's attachment;
- effects the game code spawns for the hero's abilities (abilities.vdata, e.g. Dynamo's singularity, Mina's bat
  swarm). The code isn't readable, but the effect artists previewed each one on the clip it plays with
  (m_previewState.m_sequenceName) and with the control points it needs (Dynamo's CP1 holds the singularity's radius).

Control points are given as the effect's drivers: attachment name and type, offset. attachments() turns attachment
names into a bone and an offset (game units, the bone's space).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deadlock_bridge.core import content as contentlib
from deadlock_bridge.core import resource

# slots holding the hero's own abilities and weapon; the rest (jump, slide, zipline, dash, parry) are shared and
# preview their effects on other models, whose sequence names can collide with the hero's clips
OWN_SLOTS = ('ESlot_Signature_', 'ESlot_Weapon_')
# vdata keys and effect names for what enemies see, what shows on targets, or what's drawn only while aiming
SKIP = ('enemy', 'preview', 'status', 'tgt', 'target', 'impact', 'tracer', 'projectile', 'miss')
# artists also preview buffs and debuffs on idles and runs; only an ability's own clips say when it plays
ABILITY_CLIP = ('ability', 'ult')


def resource_data(content, path, block='DATA'):
    data = content.try_read(contentlib.compiled(path))
    if data is None:
        return None
    res = resource.ResourceFile(data)
    if block == 'DATA':
        return res.keyvalues('DATA')
    found = res.block(block)
    return res.keyvalues(res.blocks.index(found)) if found else None


def external_refs(content, path):
    """The resource paths a compiled resource lists in its RERL block."""
    data = content.try_read(contentlib.compiled(path))
    if data is None:
        return []
    res = resource.ResourceFile(data)
    block = res.block('RERL')
    if block is None:
        return []
    raw = bytes(res.block_data(block))
    offset, count = struct.unpack_from('<II', raw, 0)
    out = []
    for i in range(count):
        at = offset + i * 16
        start = at + 8 + struct.unpack_from('<q', raw, at + 8)[0]
        out.append(raw[start:raw.index(b'\0', start)].decode())
    return out


def _walk(node, key=''):
    """(key, string) for every string value under a vdata block."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)
    elif isinstance(node, str):
        yield key, node


def _config(definition, name):
    configs = (definition or {}).get('m_controlPointConfigurations') or []
    return next((c for c in configs if (c.get('m_name') or '').lower() == name.lower()), None)


def _drivers(cfg):
    out = []
    for d in (cfg or {}).get('m_drivers') or []:
        out.append({'cp': int(d.get('m_iControlPoint', 0)),
                    'type': d.get('m_iAttachType', 'PATTACH_ABSORIGIN_FOLLOW'),
                    'attachment': d.get('m_attachmentName', ''),
                    'offset': tuple(d.get('m_vecOffset', (0.0, 0.0, 0.0))),
                    'angles': tuple(d.get('m_angOffset', (0.0, 0.0, 0.0))),
                    'entity': d.get('m_entityName', 'self')})
    return out


def _skipped(*names):
    return any(s in n.lower() for n in names for s in SKIP)


class ClipEffects:
    def __init__(self, content, model):
        self.content = content
        self.model = contentlib.normalize(model)
        self.skeleton = next((r for r in external_refs(content, self.model) if r.endswith('.vnmskel')), None)
        self._clips = None
        self._ability = None
        self._effects = {}

    def clip_file(self, name):
        """The compiled clip called name (as the rig's actions are, game_<name>) on this model's skeleton. Clips
        aren't always next to the model (Lash's, Ivy's live in older folders)."""
        if self._clips is None:
            self._clips = {}
            for f in self.content.files('.vnmclip_c'):
                base = f.rsplit('/', 1)[-1][:-len('.vnmclip_c')]
                if '+' not in base:
                    self._clips.setdefault(base, []).append(f)
        here = os.path.dirname(self.model) + '/'
        for f in sorted(self._clips.get(name, ()), key=lambda f: not f.startswith(here)):
            data = resource_data(self.content, f)
            if data and (self.skeleton is None or data.get('m_skeleton') == self.skeleton):
                return f, data
        return None, None

    def effect(self, path):
        if path not in self._effects:
            self._effects[path] = resource_data(self.content, path)
        return self._effects[path]

    def hero_abilities(self):
        """{slot: ability name} of the hero whose model this is (heroes.vdata), or {}."""
        heroes = resource_data(self.content, 'scripts/heroes.vdata') or {}
        for hero in heroes.values():
            if isinstance(hero, dict) and contentlib.normalize(hero.get('m_strModelName', '')) == self.model:
                return dict(hero.get('m_mapBoundAbilities') or {})
        return {}

    def ability_effects(self):
        """{sequence name: [effect]} for the effects of the hero's own abilities (and the abilities and modifiers
        they name) whose preview plays one of its ability clips."""
        if self._ability is not None:
            return self._ability
        self._ability = {}
        abilities = resource_data(self.content, 'scripts/abilities.vdata') or {}
        modifiers = resource_data(self.content, 'scripts/modifiers.vdata') or {}
        queue = [name for slot, name in self.hero_abilities().items() if slot.startswith(OWN_SLOTS)]
        named, seen = set(), set()
        while queue:
            name = queue.pop()
            if name in named:
                continue
            named.add(name)
            block = abilities.get(name) or modifiers.get(name)
            if not isinstance(block, dict):
                continue
            for key, value in _walk(block):
                if value in abilities or value in modifiers:
                    queue.append(value)
                if not value.endswith('.vpcf') or value in seen or _skipped(key, value.rsplit('/', 1)[-1]):
                    continue
                seen.add(value)
                cfg = _config(self.effect(value), 'preview')
                sequence = ((cfg or {}).get('m_previewState') or {}).get('m_sequenceName') or ''
                # some name the clip file (models\heroes_wip\baba\clips\x.vnmclip) instead of a sequence
                sequence = sequence.replace('\\', '/').rsplit('/', 1)[-1].split('.', 1)[0]
                if not any(s in sequence.lower() for s in ABILITY_CLIP):
                    continue
                self._ability.setdefault(sequence, []).append({
                    'path': value, 'source': 'ability', 'ability': name, 'key': key, 'start': 0.0, 'loop': True,
                    'drivers': _drivers(cfg)})
        return self._ability

    def clip_events(self, name):
        _, data = self.clip_file(name)
        if not data:
            return []
        out, seen = [], set()
        for e in data.get('m_events') or []:
            if e.get('_class') != 'CNmParticleEvent' or not e.get('m_hParticleSystem'):
                continue
            path = e['m_hParticleSystem']
            start = float(e.get('m_flStartTime', {}).get('m_flValue', 0.0))
            if (path, round(start, 4)) in seen or e.get('m_type', 'Create') not in ('Create', 'Create_CFG'):
                continue
            seen.add((path, round(start, 4)))
            if e.get('m_type') == 'Create_CFG':
                drivers = _drivers(_config(self.effect(path), e.get('m_config') or 'preview'))
            else:
                drivers = [{'cp': i, 'type': e.get(f'm_attachmentType{i}'), 'attachment': e.get(f'm_attachmentPoint{i}'),
                            'offset': (0.0, 0.0, 0.0), 'angles': (0.0, 0.0, 0.0), 'entity': 'self'}
                           for i in (0, 1) if e.get(f'm_attachmentType{i}', 'PATTACH_INVALID') != 'PATTACH_INVALID']
            out.append({'path': path, 'source': 'clip', 'start': start, 'loop': False,
                        'detach': bool(e.get('m_bDetachFromOwner')), 'drivers': drivers})
        return out

    def duration(self, name):
        _, data = self.clip_file(name)
        return float(data.get('m_flDuration', 0.0)) if data else 0.0

    def effects(self, name):
        """Every effect clip `name` starts: [{path, source, start (fraction of the clip), loop, drivers}]. Ability
        previews name the sequence without the hero prefix some clips have (vampirebat_ability_ult_loop)."""
        found = self.clip_events(name)
        for sequence, effects in self.ability_effects().items():
            if name == sequence or name.endswith('_' + sequence):
                found += [e for e in effects if all(e['path'] != f['path'] for f in found)]
        return found

    def attachments(self):
        """{attachment: (bone, offset, rotation xyzw)} from the model, game units in the bone's space."""
        mdat = resource_data(self.content, self.model, 'MDAT') or {}
        out = {}
        for item in mdat.get('m_attachments') or []:
            a = item.get('value') or {}
            names = a.get('m_influenceNames') or ['']
            if a.get('m_nInfluences', 1) and names[0]:
                out[a.get('m_name', item.get('key'))] = (names[0], tuple(a['m_vInfluenceOffsets'][0]),
                                                         tuple(a['m_vInfluenceRotations'][0]))
        return out

    def hitbox_sets(self):
        """{set name: [{bone, mins, maxs, radius, shape}]} of the model's hitboxes, game units in the bone's space."""
        mdat = resource_data(self.content, self.model, 'MDAT') or {}
        return {item['key']: [{'bone': h['m_sBoneName'], 'mins': tuple(h['m_vMinBounds']),
                               'maxs': tuple(h['m_vMaxBounds']), 'radius': float(h.get('m_flShapeRadius', 0.0)),
                               'shape': int(h.get('m_nShapeType', 0))}
                              for h in (item.get('value') or {}).get('m_HitBoxes') or []]
                for item in mdat.get('m_hitboxsets') or []}

    def bind_pose(self):
        """{bone: (rotation 3x3, position)} of the model's skeleton in model space, game units."""
        import numpy as np
        skel = (resource_data(self.content, self.model) or {}).get('m_modelSkeleton') or {}
        out, order = {}, []
        for i, name in enumerate(skel.get('m_boneName') or []):
            r, t = quat_matrix(skel['m_boneRotParent'][i]), np.array(skel['m_bonePosParent'][i], np.float64)
            parent = skel['m_nParent'][i]
            if parent >= 0:
                pr, pt = out[order[parent]]
                r, t = pr @ r, pt + pr @ t
            out[name] = (r, t)
            order.append(name)
        return out


def quat_matrix(q):
    import numpy as np
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_of(slug):
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'heroes.json')) as f:
        heroes = json.load(f)
    return next(h['model'] for h in heroes if h['slug'] == slug)


def main(argv):
    target, clips = argv[0], argv[1:]
    content = contentlib.GameContent(contentlib.find_deadlock())
    fx = ClipEffects(content, target if '/' in target else model_of(target))
    if not clips:
        names = {f.rsplit('/', 1)[-1][:-len('.vnmclip_c')] for f in content.files('.vnmclip_c')
                 if f.startswith(os.path.dirname(fx.model) + '/clips/') and '+' not in f}
        clips = sorted(names | set(fx.ability_effects()))
    attachments = fx.attachments()
    for clip in clips:
        found = fx.effects(clip)
        if not found:
            continue
        print(f'{clip} ({fx.duration(clip):.2f} s)')
        for e in found:
            cps = ', '.join(f"cp{d['cp']} {d['type'].replace('PATTACH_', '').lower()}"
                            + (f" {d['attachment']}" if d['attachment'] else '')
                            + ('' if d['attachment'] in attachments or not d['attachment'] else ' (missing)')
                            for d in e['drivers'])
            print(f"  {e['start']:.2f} {e['source']:7} {e['path']}  [{cps}]")


if __name__ == '__main__':
    main(sys.argv[1:])
