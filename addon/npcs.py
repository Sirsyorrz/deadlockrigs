# SPDX-License-Identifier: GPL-3.0-or-later
"""The game's NPCs on a map: the Patrons, Walkers, Guardians, Base Guardians, the Mid-boss and trooper waves as
rigs where the game has them, each playing its idle.

The spots come from the map's entities (tools/map_npcs.py), saved as <map>/npcs.json by the map build; a map built
before that gets it in a few seconds. Rigs and animations that aren't built yet are built first, one after another.
Each NPC's rig is appended once and copied for its other spots: the copies share the meshes and materials (a face's
shape keys would follow the first rig, none of the NPCs has one), and pose and animate on their own.
"""
import json
import math
import os
import random

import bpy
from mathutils import Euler, Matrix, Vector

ROOT = 'dlr_npcs'
# heroes.json "idle" names each NPC's clip; these are tried for the others
IDLE_CLIPS = ('game_idle', 'game_primary_idle', 'game_stand_idle')
# every copy plays from its own point in the loop, the same each time it is spawned
SEED = 7
# NLA repeats are capped at 1000; this many frames are covered
LOOP_FRAMES = 20000
# how far under an airborne spawn point the floor is looked for (metres)
GROUND_REACH = 10


def scene_map(scene):
    from . import crop
    maps = crop.map_collections(scene) + crop.map_collections(scene, linked=True)
    return maps[0]['dlr_map'] if maps else None


def spawns_path(slug, context=None):
    from . import output_dir
    return os.path.join(output_dir(context), 'maps', slug, 'npcs.json')


def wanted(scene, slug, context=None):
    """The map's spawns in the groups picked, or None until npcs.json is there."""
    path = spawns_path(slug, context)
    if not os.path.exists(path):
        return None
    groups = scene.dlr_npc_groups
    return [s for s in json.load(open(path, encoding='utf-8'))['npcs'] if s['group'] in groups]


def anims_path(slug, context=None):
    from . import rig_path
    rig = rig_path(slug, context)
    return os.path.join(os.path.dirname(rig), f'{slug}_anims.blend')


def missing_builds(scene, spawns, context=None):
    from . import rig_path
    jobs = []
    for slug in dict.fromkeys(s['slug'] for s in spawns):
        if not os.path.exists(rig_path(slug, context)):
            jobs.append((slug, 'RIG'))
        if scene.dlr_npc_anims and not os.path.exists(anims_path(slug, context)):
            jobs.append((slug, 'ANIMS'))
    return jobs


def npc_root(scene):
    return next((c for c in scene.collection.children_recursive if c.get(ROOT)), None)


def remove(scene):
    root = npc_root(scene)
    if root is None:
        return
    colls = [root] + list(root.children_recursive)
    obs = {o for c in colls for o in c.objects}
    datas = {o.data for o in obs if o.data is not None}
    acts = {s.action for o in obs if o.animation_data for t in o.animation_data.nla_tracks for s in t.strips
            if s.action}
    # the bone widgets are in no collection
    shapes = {pb.custom_shape for o in obs if o.pose for pb in o.pose.bones if pb.custom_shape} - obs
    bpy.data.batch_remove(list(obs))
    for c in colls:
        bpy.data.collections.remove(c)
    shapes = [o for o in shapes if o.users == 0]
    datas |= {o.data for o in shapes}
    bpy.data.batch_remove(shapes)
    for d in datas:
        if d.users == 0:
            if isinstance(d, bpy.types.Mesh):
                bpy.data.meshes.remove(d)
            elif isinstance(d, bpy.types.Armature):
                bpy.data.armatures.remove(d)
    for a in acts:
        if a.users == 0:
            bpy.data.actions.remove(a)


def remap_ids(owner, new):
    """Points owner's object references (parent, modifiers, constraints, drivers, ID properties) that are in new at
    their copies."""
    def swap(holder, attr):
        v = getattr(holder, attr, None)
        if v in new:
            setattr(holder, attr, new[v])

    swap(owner, 'parent')
    for m in getattr(owner, 'modifiers', ()):
        swap(m, 'object')
    cons = list(getattr(owner, 'constraints', ()))
    if owner.pose:
        cons += [c for pb in owner.pose.bones for c in pb.constraints]
    for c in cons:
        for attr in ('target', 'pole_target', 'space_object'):
            swap(c, attr)
        for t in getattr(c, 'targets', ()):
            swap(t, 'target')
    if owner.animation_data:
        for fc in owner.animation_data.drivers:
            for v in fc.driver.variables:
                for t in v.targets:
                    swap(t, 'id')
    for k in owner.keys():
        if isinstance(owner[k], bpy.types.Object) and owner[k] in new:
            owner[k] = new[owner[k]]


def copy_rig(coll):
    """A copy of a rig's collection whose objects share the original's data."""
    obs = list(coll.all_objects)
    new = {o: o.copy() for o in obs}

    def copy_coll(c):
        out = bpy.data.collections.new(c.name)
        for o in c.objects:
            out.objects.link(new[o])
        for ch in c.children:
            out.children.link(copy_coll(ch))
        return out
    out = copy_coll(coll)
    for o in new.values():
        remap_ids(o, new)
    return out


def rig_of(coll):
    return next((o for o in coll.all_objects if o.type == 'ARMATURE' and 'dlrig' in o), None)


def idle_clip(slug, names):
    from . import heroes
    cfg = next((h for h in heroes() if h['slug'] == slug), {})
    names = set(names)
    if cfg.get('idle') in names:
        return cfg['idle']
    return next((c for c in IDLE_CLIPS if c in names), next((n for n in sorted(names) if 'idle' in n), None))


def load_idle(arm, slug):
    path = bpy.path.abspath(arm.get('anim_library', ''))
    if not os.path.exists(path):
        return None
    with bpy.data.libraries.load(path) as (src, dst):
        clip = idle_clip(slug, src.actions)
        dst.actions = [clip] if clip else []
    if not dst.actions:
        return None
    act = dst.actions[0]
    act.use_fake_user = False
    cb = act.layers[0].strips[0].channelbags[0]
    # as the rig's Game Animations do: twist and cloth-follow bones are rebuilt by their constraints
    driven = {b.name for b in arm.pose.bones if 'Twist' in b.constraints or 'Cloth Follow' in b.constraints}
    for f in [f for f in cb.fcurves if f.data_path.startswith('pose.bones["') and f.data_path.split('"')[1] in driven]:
        cb.fcurves.remove(f)
    act.name = f'{arm["dlrig"]} idle'
    return act


def play(arm, act, offset):
    """The clip on repeat from the scene's start, offset frames into its loop; arms and legs go to FK for it."""
    for p in arm.get('clip_off', ['weapon_hold']):
        if p in arm:
            arm[p] = 0.0
    for p in [k for k in arm.keys() if k.startswith('ik_')]:
        arm[p] = 0.0
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = None
    track = ad.nla_tracks.new()
    track.name = 'Game idle'
    f0, f1 = act.frame_range
    length = max(1.0, f1 - f0)
    start = bpy.context.scene.frame_start - offset * length
    strip = track.strips.new(act.name, int(start), act)
    strip.action_slot = act.slots[0]
    strip.frame_start_ui = start
    strip.repeat = min(1000.0, math.ceil(LOOP_FRAMES / length))
    strip.extrapolation = 'HOLD'


def ground(scene, spot):
    """The floor under an airborne spawn point, else the point. Walls and slopes steeper than 45 degrees don't
    count."""
    loc = Vector(spot['location'])
    if not spot.get('ground'):
        return loc
    dg = bpy.context.evaluated_depsgraph_get()
    start = loc + Vector((0, 0, 0.3))
    for _ in range(8):
        hit, at, normal, *_ = scene.ray_cast(dg, start, Vector((0, 0, -1)), distance=GROUND_REACH)
        if not hit:
            break
        if normal.z > 0.7:
            return at
        start = at - Vector((0, 0, 0.01))
    return loc


def spawn(context):
    from . import hero_name, load_rig, register_rig_ui, riglight, rig_path
    scene = context.scene
    slug = scene_map(scene)
    spawns = wanted(scene, slug, context) if slug else None
    if not spawns:
        return 0
    remove(scene)
    # before any NPC is there to be hit
    for spot in spawns:
        spot['location'] = ground(scene, spot)
    from . import tools_module
    title = tools_module('mapbuild').MAPS.get(slug, slug)
    root = bpy.data.collections.new(f'{title} NPCs')
    root[ROOT] = slug
    scene.collection.children.link(root)
    rnd = random.Random(SEED)
    count = 0
    for npc in dict.fromkeys(s['slug'] for s in spawns):
        path = rig_path(npc, context)
        if not os.path.exists(path):
            continue
        first = load_rig(path, npc)
        if not first:
            continue
        first = first[0]
        if sum(s['slug'] == npc for s in spawns) > 1:
            group = bpy.data.collections.new(hero_name(npc) + 's')
            root.children.link(group)
        else:
            group = root
        act = load_idle(rig_of(first), npc) if scene.dlr_npc_anims else None
        for i, spot in enumerate(s for s in spawns if s['slug'] == npc):
            coll = first if i == 0 else copy_rig(first)
            coll.name = spot['label']
            group.children.link(coll)
            arm = rig_of(coll)
            arm.matrix_world = Matrix.LocRotScale(spot['location'], Euler((0, 0, spot['yaw'])), None)
            if act:
                play(arm, act, rnd.random())
            count += 1
    register_rig_ui()
    riglight.light_rigs(scene)
    return count


def toggle(self, context):
    from . import queue_position, start_queue
    scene = context.scene
    if not scene.dlr_npcs:
        remove(scene)
        return
    slug = scene_map(scene)
    if not slug or queue_position() is not None:
        # a queue already running spawns them when it is done
        return

    def ready(ok):
        sc = bpy.context.scene
        if not ok:
            sc.dlr_npcs_status = 'failed'
            return
        spawns = wanted(sc, slug)
        jobs = missing_builds(sc, spawns or [])
        if jobs:
            start_queue(bpy.context, jobs, built)
        else:
            built(True)

    def built(ok):
        sc = bpy.context.scene
        sc.dlr_npcs_status = 'failed' if not ok else ''
        if ok and sc.dlr_npcs:
            # appending the rigs is done in a timer so the panel shows the build finishing first
            bpy.app.timers.register(lambda: spawn(bpy.context) and None, first_interval=0.1)

    scene.dlr_npcs_status = ''
    if wanted(scene, slug, context) is None:
        start_queue(context, [(slug, 'NPCS')], ready)
    else:
        ready(True)


def regroup(self, context):
    if context.scene.dlr_npcs:
        toggle(self, context)


def draw(layout, context):
    from . import _job, queue_position, hero_name
    sc = context.scene
    box = layout.box()
    row = box.row(align=True)
    pos = queue_position()
    busy = _job is not None and not _job.done
    row.enabled = not busy or pos is not None
    row.prop(sc, 'dlr_npcs', text='NPCs', icon='COMMUNITY')
    row.prop_menu_enum(sc, 'dlr_npc_groups', text='Which', icon='DOWNARROW_HLT')
    row = box.row()
    row.enabled = not busy
    row.prop(sc, 'dlr_npc_anims')
    if pos is not None and _job is not None:
        what = {'NPCS': "Finding the map's NPCs", 'RIG': f'Building {hero_name(_job.slug)}',
                'ANIMS': f"Exporting {hero_name(_job.slug)}'s animations"}.get(_job.kind, _job.slug)
        box.label(text=f'{what} ({pos[0]}/{pos[1]})')
        box.progress(factor=_job.progress, type='BAR', text=f'{_job.status}  {_job.progress:.0%}')
        box.operator('dlr.cancel', icon='X')
    elif sc.dlr_npcs_status == 'failed':
        box.label(text='Building the NPCs failed', icon='ERROR')
        if _job is not None and _job.error:
            box.label(text=_job.error[:80])
        box.operator('dlr.show_log', icon='TEXT')


def register():
    from . import tools_module
    S, P = bpy.types.Scene, bpy.props
    groups = tools_module('map_npcs').GROUPS
    S.dlr_npcs = P.BoolProperty(
        name='NPCs', default=False, update=toggle,
        description=("Put the game's NPCs on the map where the game has them, as rigs playing their idle. "
                     "Rigs and animations not built yet are built first (a few minutes each, the first time)"))
    S.dlr_npc_groups = P.EnumProperty(
        name='NPCs', items=groups, options={'ENUM_FLAG'}, update=regroup, default={g[0] for g in groups},
        description='Which NPCs to put on the map')
    S.dlr_npc_anims = P.BoolProperty(
        name='Play Idles', default=True, update=regroup,
        description="Each NPC plays its game idle on repeat (needs its animations, exported when missing)")
    S.dlr_npcs_status = P.StringProperty(options={'SKIP_SAVE'})


def unregister():
    for p in ('dlr_npcs', 'dlr_npc_groups', 'dlr_npc_anims', 'dlr_npcs_status'):
        delattr(bpy.types.Scene, p)
