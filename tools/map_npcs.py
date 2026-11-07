"""Where a map's NPCs stand in the game, from its entities (mapbuild.py writes them to <map>/npcs.json; the add-on
places rigs there).

Each spawn: the NPC's slug in heroes.json, its group (the add-on's NPCs menu), a label, Blender position (metres)
and heading (radians about Z: the game's yaw, since the rigs face the way the game's models face), and ground when
the point is in the air and the NPC should stand on the floor under it.

The Lane Guardians have no entity: the game puts them where the lanes' info_super_trooper_spawn points are (the
super troopers come out where a Guardian fell), under the zipline nodes that name them as their guard.
"""
import math
from itertools import groupby

from map_environment import num, to_blender

TEAMS = {2: 'amber', 3: 'sapphire'}
LANES = {1: 'yellow', 4: 'blue', 6: 'purple'}
GROUPS = (
    ('PATRONS', 'Patrons', 'The Hidden King and the Archmother over their bases'),
    ('WALKERS', 'Walkers', 'One per lane for each team'),
    ('GUARDIANS', 'Lane Guardians', 'One per lane for each team'),
    ('BASE_GUARDIANS', 'Base Guardians', 'Two at each base entrance'),
    ('MIDBOSS', 'Mid-boss', 'In its pit under the middle of the map'),
    ('TROOPERS', 'Troopers', "A wave at each lane's trooper spawn in the bases"),
)
# a wave at each lane spawn's four points; the game sends ranged, melee and medic troopers
WAVE = ('trooper', 'trooper', 'melee_trooper', 'medic')
# the base's own spawn points sit far behind the lanes; the lane ones are within this of the middle (inches)
LANE_SPAWN_Y = 4000


def team_of(e):
    return TEAMS.get(int(num(e, 'teamnumber')))


def lane_of(e):
    return LANES.get(int(num(e, 'lanenum')), '')


def spawn(slug, group, label, e):
    return {'slug': slug, 'group': group, 'label': label, 'location': to_blender([float(v) for v in e['origin']]),
            'yaw': math.radians(float((e.get('angles') or (0, 0, 0))[1]))}


def spawns(ents):
    out = []
    troopers = {}
    for e in ents:
        cls = e.get('classname')
        team, lane = team_of(e), lane_of(e)
        if cls == 'npc_boss_tier3' and team:
            out.append(spawn({'amber': 'hidden_king', 'sapphire': 'archmother'}[team], 'PATRONS',
                             {'amber': 'Hidden King', 'sapphire': 'Archmother'}[team], e))
        elif cls == 'npc_boss_tier2' and team:
            out.append(spawn('walker', 'WALKERS', f'Walker {team} {lane}', e))
        elif cls == 'info_super_trooper_spawn' and team and abs(float(e['origin'][1])) < LANE_SPAWN_Y:
            out.append(spawn('guardian', 'GUARDIANS', f'Guardian {team} {lane}', e))
        elif cls == 'npc_barrack_boss' and team:
            out.append(spawn(f'{team}_base_guardian', 'BASE_GUARDIANS', f'Base Guardian {team} {lane}', e))
        elif cls == 'info_neutral_trooper_camp' and e.get('subclass_name') == 'neutral_camp_midboss':
            out.append(spawn('midboss', 'MIDBOSS', 'Mid-boss', e))
        elif cls == 'info_trooper_spawn' and team:
            troopers.setdefault((team, lane), []).append(e)
    for (team, lane), points in sorted(troopers.items()):
        # nearest the lanes first, so the ranged troopers lead
        points.sort(key=lambda p: abs(float(p['origin'][1])))
        for kind, e in zip(WAVE, points):
            out.append(spawn(f'{team}_{kind}', 'TROOPERS', f'{team.title()} {kind.replace("_", " ")} {lane}', e))
            # the points start the troopers' ziplines, up to 5 m over the floor: the add-on stands them on it
            out[-1]['ground'] = True
    for s in out:
        s['label'] = s['label'].replace('  ', ' ').strip().title()
    # two Base Guardians at each entrance, two ranged troopers in a wave
    for label, same in groupby(sorted(out, key=lambda s: s['label']), key=lambda s: s['label']):
        same = list(same)
        if len(same) > 1:
            for i, s in enumerate(same):
                s['label'] = f'{label} {i + 1}'
    return out
