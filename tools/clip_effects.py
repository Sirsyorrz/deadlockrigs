"""python3 clip_effects.py <hero slug | model path> [clip ...]

The particle effects a hero's animation clips start in the game, read from the user's install (nothing extracted):

- the clip's own particle events (CNmParticleEvent: reloads, dashes, Paige's Excalibur, Solomon's pieces), started
  at a fraction of the clip and placed by the effect's "preview" control point setup, or by the event's attachment;
- the same events of the model's legacy animations (AE_CL_CREATE_PARTICLE_EFFECT), which the rig's actions also
  come from under their old names (bebop_hyperbeam_charge for the clip hyperbeam_charge);
- effects the game code spawns for the hero's abilities (abilities.vdata, e.g. Dynamo's singularity, Mina's bat
  swarm). The code isn't readable, but the effect artists previewed many on the clip they play with
  (m_previewState.m_sequenceName) and with the control points they need (Dynamo's CP1 holds the singularity's
  radius). The rest, when placed on the hero, go on their ability's clips: the hero's animgraphs pick those by the
  ability's name or casting action.

Control points are given as the effect's drivers: attachment name and type, offset. attachments() turns attachment
names into a bone and an offset (game units, the bone's space).
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deadlock_bridge.core import content as contentlib
from deadlock_bridge.core import resource

# slots holding the hero's own abilities and weapon; the rest (jump, slide, zipline, dash, parry) are shared and
# preview their effects on other models, whose sequence names can collide with the hero's clips
OWN_SLOTS = ('ESlot_Signature_', 'ESlot_Weapon_')
ATTACHED = ('PATTACH_POINT', 'PATTACH_POINT_FOLLOW')
# where a cast's flash goes when the effect itself doesn't say (its preview sits at the world origin)
CAST_DRIVER = {'cp': 0, 'type': 'PATTACH_POINT_FOLLOW', 'attachment': 'ability_cast', 'offset': (0.0, 0.0, 0.0),
               'angles': (0.0, 0.0, 0.0), 'entity': 'self'}
# a legacy animation event's attachment types
LEGACY_ATTACH = {'point': 'PATTACH_POINT', 'point_follow': 'PATTACH_POINT_FOLLOW', 'absorigin': 'PATTACH_ABSORIGIN',
                 'absorigin_follow': 'PATTACH_ABSORIGIN_FOLLOW', 'worldorigin': 'PATTACH_WORLDORIGIN'}
# vdata keys and effect names for what enemies see, what shows on targets, what's drawn only while aiming, or only
# for the hero's own player (first person beams, screen overlays)
SKIP = ('enemy', 'preview', 'status', 'tgt', 'target', 'impact', 'tracer', 'projectile', 'miss', 'local', '_player',
        'owner', 'screen', 'debug')
# artists also preview buffs and debuffs on idles, runs and the bind pose: those say nothing about when it plays
ABILITY_WORDS = {'ability', 'ult'}
GENERIC_WORDS = {'idle', 'idle1', 'run', 'walk', 'sprint', 'stand', 'crouch', 'bindpose', 'bind', 'pose', 'ref',
                 'tools', 'animsetup', 'animbindpose', 'ui', 'shop', 'flinch', 'reload', 'melee', 'jump', 'slide',
                 'mantle', 'outofcombat', 'combat'}
# the parts of a state's node path its conditions sit under (the state is the path before them)
SM_PARTS = ('Entry Overrides', 'Global Transitions', 'Transition Conduit')
FILLER_WORDS = {'weapon', 'arm', 'item', 'zoomed', 'open', 'close', 'center', 'centre', 'e', 'n', 's', 'w', 'ne', 'nw',
                'se', 'sw', 'l', 'r', 'left', 'right', 'fwd', 'back', 'add', 'air', 'additive', 'ability', 'hero'}
# the moments of an ability an effect's name and its clips' names share
STAGES = {'cast', 'charge', 'fire', 'loop', 'attached', 'channel', 'start', 'end', 'windup', 'land', 'exit', 'intro',
          'throw', 'hold', 'release', 'jump', 'dash', 'activate'}
STAGE_ALIASES = {'precast': 'cast', 'casting': 'cast', 'charging': 'charge', 'channeling': 'channel',
                 'channelling': 'channel', 'landing': 'land', 'activation': 'activate'}
# effects that show on whoever an ability hits, or away from the caster; generic/ and modifiers/ hold the shared
# slows, stuns and heals. A buff is the caster's own often enough (Vindicta's flight) to keep
STATUS = ('debuff', 'stun', 'slow', 'particles/generic/', 'particles/modifiers/', 'victim', 'explode', 'explosion',
          'hit', 'death', 'bullet', 'link', 'chain', 'tether', 'ground')


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


def _status(key, path):
    return any(s in key.lower() or s in path.lower() for s in STATUS)


def _on_hero(cfg, attachments):
    """Whether a preview setup puts the effect on the hero: control points on its origin or attachments only."""
    placed = [d for d in _drivers(cfg) if d['type'] != 'PATTACH_WORLDORIGIN']
    return bool(placed) and all(
        d['entity'] in ('self', '') and (d['type'] not in ATTACHED or d['attachment'] in attachments) for d in placed)


def _words(text):
    return set(re.sub(r'([a-z])([A-Z])', r'\1_\2', text).lower().replace('.', '_').split('_'))


def _stages(text):
    return {STAGE_ALIASES.get(w, w) for w in _words(text)} & STAGES


def _core(clip):
    """The words naming what a clip is of (hyperbeam, hook), not how it's played."""
    return {w for w in _words(clip) if w not in GENERIC_WORDS | STAGES | FILLER_WORDS and not w.isdigit()}


def _graph_ids(name, block, graph):
    """The animgraph IDs an ability goes by: its name, its casting actions, and IDs named after an action without its
    stage (ability_flight_casting: ability_flight_channeling, vindicta_ability_flight)."""
    actions = [a for a in (block.get('m_strAG2CastingAction'), block.get('m_strAG2CastCompletedAction')) if a]
    ids = {i for i in [name, *actions] if i in graph}
    for action in actions:
        stem = re.sub(r'_(casting|cast|channeling|channel|start)$', '', action)
        if stem != action and (stem.count('_') or len(stem) >= 6):
            ids |= {i for i in graph if i.startswith(stem + '_') or i.endswith('_' + stem)}
    return ids


def _cores(clips):
    return set().union(*map(_core, clips))


def _same_stage(key, path, clips):
    """The clips sharing a stage word with the effect's key or name (m_ChargeParticle: the charge clips), or all."""
    stages = _stages(key) | _stages(path.rsplit('/', 1)[-1])
    return {c for c in clips if _stages(c) & stages} or set(clips)


_shared = {}
_titles = {}


def titles(content):
    """{ability: its English name} from the game's localization files (outside the VPKs)."""
    if content.install not in _titles:
        path = os.path.join(content.game, 'citadel', 'resource', 'localization', 'citadel_heroes',
                            'citadel_heroes_english.txt')
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                _titles[content.install] = dict(re.findall(r'^\s*"([^"]+)"\s+"([^"]*)"', f.read(), re.M))
        except OSError:
            _titles[content.install] = {}
    return _titles[content.install]


def for_model(content, model):
    """One ClipEffects per model and game install: reading a hero's animgraphs takes a second or two."""
    key = (id(content), contentlib.normalize(model))
    if key not in _shared:
        _shared[key] = ClipEffects(content, model)
    return _shared[key]


class ClipEffects:
    def __init__(self, content, model):
        self.content = content
        self.model = contentlib.normalize(model)
        self.skeleton = next((r for r in external_refs(content, self.model) if r.endswith('.vnmskel')), None)
        self._clips = None
        self._ability = None
        self._effects = {}
        self._legacy = None
        self._graph = None
        self._nm = None
        self._catalog = None

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

    def legacy_anims(self):
        """{name: anim} of the model's own pre-animgraph-2 animations: rig actions not named after a clip file
        (bebop_hyperbeam_charge_loop next to the clip hyperbeam_charge_loop)."""
        if self._legacy is None:
            anim = resource_data(self.content, self.model, 'ANIM') or {}
            self._legacy = {a['m_name']: a for a in anim.get('m_animArray') or [] if a.get('m_name')}
        return self._legacy

    @staticmethod
    def names(name):
        """name and, for one of three words or more, the name without its first (a hero prefix:
        vampirebat_ability_ult_loop, bebop_hyperbeam_charge)."""
        parts = name.split('_')
        return [name, '_'.join(parts[1:])] if len(parts) > 2 else [name]

    def graph_actions(self):
        """{graph ID: {clip}}: the clips the hero's animgraphs play in a state picked by that ID (an ability's name
        or its m_strAG2CastingAction). The deepest picking state wins: a top-level selector over every ability says
        nothing about one."""
        if self._graph is not None:
            return self._graph
        self._graph = {}
        for path in self.content.files('.vnmgraph_c'):
            data = resource_data(self.content, path[:-2])
            if not data or data.get('m_skeleton') != self.skeleton:
                continue
            nodes, paths = data.get('m_nodes') or [], data.get('m_nodePaths') or []
            resources = data.get('m_resources') or []
            scopes = []
            for node in nodes:
                ids = node.get('m_comparisionIDs')
                if ids and node.get('m_comparison') == 'Matches':
                    parts = paths[node['m_nNodeIdx']].split('/')
                    cut = next((i for i, p in enumerate(parts) if p in SM_PARTS), len(parts) - 1)
                    scopes.append(('/'.join(parts[:cut]) + '/', ids))
            for node in nodes:
                slot = node.get('m_nDataSlotIdx', -1)
                if ('ClipNode' not in node.get('_class', '') or not 0 <= slot < len(resources)
                        or not resources[slot].endswith('.vnmclip')):
                    continue
                clip = resources[slot].rsplit('/', 1)[-1][:-len('.vnmclip')]
                inside = [s for s in scopes if paths[node['m_nNodeIdx']].startswith(s[0])]
                deepest = max((len(s[0]) for s in inside), default=0)
                for scope, ids in inside:
                    if len(scope) == deepest:
                        for i in ids:
                            self._graph.setdefault(i, set()).add(clip)
        return self._graph

    def ability_clips(self, abilities, own):
        """{ability: {clip}} of the hero's own abilities, from the animgraph IDs that name one ability only (a hero's
        throws often share one)."""
        graph = self.graph_actions()
        users = {}
        for name in own:
            for i in _graph_ids(name, abilities.get(name) or {}, graph):
                users.setdefault(i, set()).add(name)
        out = {}
        for i, names in users.items():
            if len(names) == 1:
                out.setdefault(next(iter(names)), set()).update(graph[i])
        for name, clips in out.items():
            # aim and locomotion graphs pick their sets by the ability too: keep the runs of the ability's own
            # (hyperbeam_run_n), not the shared ones (item_run_n)
            own_clips = {c for c in clips if not _words(c) & GENERIC_WORDS}
            out[name] = {c for c in clips if _core(c) & _cores(own_clips)}
        # a clip in two abilities' sets (an overlay graph, one state for two IDs) goes to the one it's named after;
        # words in several abilities' names (the hero's codename) tell none apart
        named = {name: _words(name) | _words((abilities.get(name) or {}).get('m_strAG2CastingAction') or '')
                 for name in own}
        common = {w for w in set().union(*named.values()) if sum(w in ws for ws in named.values()) > 1}
        shared = {c for name in out for c in out[name] if sum(c in out[o] for o in out) > 1}
        for name, clips in out.items():
            out[name] = {c for c in clips if c not in shared or _core(c) & (named[name] - common)
                         or not any(_core(c) & (named[o] - common) for o in out if c in out[o])}
        return out

    def clip_names(self):
        """The hero's clip files (its folder's and the ones its animgraphs play) and legacy animations."""
        if self._nm is None:
            folder = os.path.dirname(self.model) + '/clips/'
            self._nm = {f[len(folder):-len('.vnmclip_c')] for f in self.content.files('.vnmclip_c')
                        if f.startswith(folder) and '+' not in f}
            self._nm.update(*self.graph_actions().values())
            self._nm.update(self.legacy_anims())
        return self._nm

    def ability_effects(self):
        """[effect with its clips] for the effects of the hero's own abilities (and the abilities and modifiers they
        name): on the clip an artist previewed it with, or, when placed on the hero and previewed on no clip that
        says when, on its ability's animgraph clips of the same stage (cast, charge, loop...)."""
        if self._ability is not None:
            return self._ability
        self._ability = []
        abilities = resource_data(self.content, 'scripts/abilities.vdata') or {}
        modifiers = resource_data(self.content, 'scripts/modifiers.vdata') or {}
        own = [name for slot, name in self.hero_abilities().items() if slot.startswith(OWN_SLOTS)]
        # every block an ability reaches, with the ability (not modifier) it hangs off: a transformed form's own
        # abilities (Silver's) play on their own clips
        reached = {}
        for root in own:
            queue = [(root, root)]
            while queue:
                name, owner = queue.pop()
                if (root, name) in reached:
                    continue
                reached[root, name] = owner
                block = abilities.get(name) or modifiers.get(name)
                for _, value in _walk(block if isinstance(block, dict) else {}):
                    if value in abilities or value in modifiers:
                        queue.append((value, value if value in abilities else owner))
        owners = list(dict.fromkeys(reached.values()))
        by_ability = self.ability_clips(abilities, owners)
        # effects of the hero's own folders; others reached through shared modifiers (item barriers, another hero's
        # buff reused) show on whoever has them
        folders = {os.path.dirname(v) for root in own for _, v in _walk(abilities.get(root) or {})
                   if v.endswith('.vpcf')}
        attachments = self.attachments()
        seen = set()
        for (root, name), owner in reached.items():
            block = abilities.get(name) or modifiers.get(name)
            if not isinstance(block, dict):
                continue
            if owner in by_ability or owner == root or (abilities.get(owner) or {}).get('m_strAG2CastingAction'):
                clips = by_ability.get(owner, set())
            else:
                clips = by_ability.get(root, set())
            for key, value in _walk(block):
                if not value.endswith('.vpcf') or value in seen or _skipped(key, value.rsplit('/', 1)[-1]):
                    continue
                if name != root and os.path.dirname(value) not in folders:
                    continue
                cfg = _config(self.effect(value), 'preview')
                sequence = ((cfg or {}).get('m_previewState') or {}).get('m_sequenceName') or ''
                # some name the clip file (models\heroes_wip\baba\clips\x.vnmclip) instead of a sequence
                sequence = sequence.replace('\\', '/').rsplit('/', 1)[-1].split('.', 1)[0]
                drivers, guessed, extra = _drivers(cfg), False, set()
                if self._specific(sequence, clips):
                    found = set(self.names(sequence))
                    if clips and not found & clips and _cores(found) & _cores(clips):
                        # previewed on an older clip of the ability the animgraph no longer plays
                        # (bebop_hyperbeam_charge_loop)
                        extra = _same_stage(key, value, clips)
                elif not clips or _status(key, value):
                    continue
                elif _on_hero(cfg, attachments):
                    found = _same_stage(key, value, clips)
                elif (not any(d['type'] != 'PATTACH_WORLDORIGIN' for d in drivers) and 'ability_cast' in attachments
                      and _stages(key) | _stages(value.rsplit('/', 1)[-1])):
                    # placed by the game code alone (preview at the world origin): a cast's flash goes where
                    # the model says casts come from
                    drivers = [CAST_DRIVER] + [d for d in drivers if d['cp'] != 0]
                    found, guessed = _same_stage(key, value, clips), True
                else:
                    continue
                seen.add(value)
                self._ability.append({
                    'path': value, 'source': 'ability', 'ability': name, 'key': key, 'start': 0.0, 'loop': True,
                    'drivers': drivers, 'clips': found, 'guessed': guessed, 'root': root, 'extra': extra})
        # a clip another of the ability's effects was previewed on is that one's (the charge's, not the beam's)
        for e in self._ability:
            claimed = set().union(*(o['clips'] for o in self._ability
                                    if o is not e and o['root'] == e['root'] and not o['guessed']))
            e['clips'] = e['clips'] | (e.pop('extra') - claimed)
        return self._ability

    def catalog(self):
        """[(ability, title, [effect])]: every effect of each of the hero's abilities (and what they name), for
        adding by hand where no clip says when it plays. Each effect's 'kind' says where it's put: 'hero' (on the
        attachments it was previewed on), 'cast' (a flash with no placement, on the ability_cast attachment),
        'target' (what shows on whoever is hit or buffed: put on this rig), 'area' (laid out around the rig's
        origin as previewed: domes, walls, beams)."""
        if self._catalog is not None:
            return self._catalog
        abilities = resource_data(self.content, 'scripts/abilities.vdata') or {}
        modifiers = resource_data(self.content, 'scripts/modifiers.vdata') or {}
        names = titles(self.content)
        attachments = self.attachments()
        self._catalog = []
        slots = sorted(self.hero_abilities().items(), key=lambda s: (not s[0].startswith('ESlot_Signature_'), s[0]))
        for slot, root in slots:
            if not slot.startswith(OWN_SLOTS):
                continue
            effects, queue, named = {}, [root], set()
            while queue:
                name = queue.pop()
                if name in named:
                    continue
                named.add(name)
                block = abilities.get(name) or modifiers.get(name)
                for key, value in _walk(block if isinstance(block, dict) else {}):
                    if value in abilities or value in modifiers:
                        queue.append(value)
                    if not value.endswith('.vpcf') or value in effects or _skipped(key, value.rsplit('/', 1)[-1]):
                        continue
                    if self.effect(value) is None:
                        continue
                    cfg = _config(self.effect(value), 'preview')
                    drivers, kind = _drivers(cfg), 'area'
                    if _status(key, value):
                        kind = 'target'
                    elif _on_hero(cfg, attachments):
                        kind = 'hero'
                    elif (not any(d['type'] != 'PATTACH_WORLDORIGIN' for d in drivers)
                          and 'ability_cast' in attachments and _stages(key) | _stages(value.rsplit('/', 1)[-1])):
                        kind = 'cast'
                        drivers = [CAST_DRIVER] + [d for d in drivers if d['cp'] != 0]
                    effects[value] = {'path': value, 'source': 'ability', 'ability': name, 'key': key, 'kind': kind,
                                      'start': 0.0, 'loop': True, 'drivers': drivers, 'guessed': kind == 'cast'}
            if effects:
                title = names.get(root) or root.replace('citadel_', '').replace('ability_', '').replace('_', ' ')
                self._catalog.append((root, title, list(effects.values())))
        return self._catalog

    def _specific(self, sequence, ability_clips):
        """Whether a preview's sequence says when the effect plays: a clip of this hero that isn't one artists
        preview buffs on (idles, runs, the bind pose)."""
        if not sequence or not set(self.names(sequence)) & self.clip_names():
            return False
        words = set(sequence.lower().split('_'))
        return (any(n in ability_clips for n in self.names(sequence)) or bool(words & ABILITY_WORDS)
                or not words & GENERIC_WORDS)

    def nm_clip(self, name):
        """The clip file (path, data) an action name stands for, under any of its names."""
        for n in self.names(name):
            path, data = self.clip_file(n)
            if data:
                return path, data
        return None, None

    def legacy_events(self, name):
        anim = self.legacy_anims().get(name)
        out = []
        for e in (anim or {}).get('m_eventArray') or []:
            kind, data = e.get('m_sEventName', ''), e.get('m_EventData') or {}
            path = data.get('name') or ''
            if not kind.startswith('AE_CL_CREATE_PARTICLE_EFFECT') or not path or _skipped(path.rsplit('/', 1)[-1]):
                continue
            if kind.endswith('_CFG'):
                drivers = _drivers(_config(self.effect(path), data.get('config') or 'preview'))
            else:
                drivers = [{'cp': cp, 'type': LEGACY_ATTACH.get(data.get(f'attachment_type{s}') or '',
                                                                'PATTACH_ABSORIGIN_FOLLOW'),
                            'attachment': data.get(f'attachment_point{s}') or '', 'offset': (0.0, 0.0, 0.0),
                            'angles': (0.0, 0.0, 0.0), 'entity': 'self'}
                           for cp, s in ((0, ''), (1, '_cp1')) if cp == 0 or data.get(f'attachment_point{s}')]
            out.append({'path': path, 'source': 'clip', 'start': float(e.get('m_flCycle', 0.0)), 'loop': False,
                        'detach': bool(data.get('detach_from_owner')), 'drivers': drivers})
        return out

    def clip_events(self, name):
        out = self.legacy_events(name)
        _, data = self.nm_clip(name)
        if not data:
            return out
        seen = {(e['path'], round(e['start'], 4)) for e in out}
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
        anim = self.legacy_anims().get(name)
        if anim:
            frames = (anim.get('m_pData') or {}).get('m_nFrames', 1)
            return max(frames - 1, 1) / float(anim.get('fps') or 30.0)
        _, data = self.nm_clip(name)
        return float(data.get('m_flDuration', 0.0)) if data else 0.0

    def effects(self, name):
        """Every effect clip `name` starts: [{path, source, start (fraction of the clip), loop, drivers}]. The rig's
        action may go by the legacy name (bebop_hyperbeam_charge) or the clip file's (hyperbeam_charge)."""
        found = self.clip_events(name)
        names = set(self.names(name))
        for e in self.ability_effects():
            if names & e['clips'] and all(e['path'] != f['path'] for f in found):
                found.append(e)
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
    clips = clips or sorted(fx.clip_names())
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
            print(f"  {e['start']:.2f} {e['source']:7} {e['path']}  [{cps}]"
                  + (f"  {e['ability']}.{e['key']}" if e['source'] == 'ability' else '')
                  + ('  (placement guessed)' if e.get('guessed') else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
