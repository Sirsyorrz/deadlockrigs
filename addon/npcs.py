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
from mathutils.bvhtree import BVHTree

from . import ziplines

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


def npc_effects(slug):
    """The particle effects that play on an NPC's extra models (heroes.json extra_models "particles": the patrons'
    effigy flames, the cores' glow) -> [(extra key, effect path, offset from the rig)]."""
    from . import heroes
    cfg = next((h for h in heroes() if h['slug'] == slug), {})
    out = []
    for name, extra in cfg.get('extra_models', {}).items():
        key = ''.join(c if c.isalnum() else '_' for c in name.lower()).strip('_')
        out += [(key, path, Vector(extra.get('offset', (0, 0, 0)))) for path in extra.get('particles', [])]
    return out


def add_effects(arm, slug, cache):
    """The NPC's effects at its spot, parented to the rig and shown with what they belong to (the effigy's flames
    with the patron, the core's glow once it's weakened). Simulated once per kind (cache: effect path -> holder) and copied for the other
    spots. Skipped quietly without the game's files."""
    from . import particles
    for key, path, offset in npc_effects(slug):
        try:
            if path not in cache:
                bridge = particles.bridge_particles()
                cache[path] = bridge.spawn_particle_system(bpy.context, path, Matrix(), time=2.0,
                                                           playback={'loop': True, 'length': None, 'start': 0.0})
            source = cache[path]
        except Exception as e:
            print(f'NPC effect {path}: {e}')
            cache[path] = None
            continue
        if source is None:
            continue
        holder = source if source.parent is None and source.get('dlr_npc_effect') is None else copy_tree(source)
        holder['dlr_npc_effect'] = key
        holder.parent = arm
        holder.matrix_parent_inverse = Matrix()
        holder.matrix_basis = Matrix.Translation(offset)
        coll = arm.users_collection[0]
        stages = {**arm.get('stage_models', {}), **arm.get('stage_extras', {})}
        prop, hidden = ('stage', f'v != {stages[key]}') if key in stages else (f'show_{key}', 'v < 0.5')
        for ob in [holder] + list(holder.children_recursive):
            for c in list(ob.users_collection):
                c.objects.unlink(ob)
            coll.objects.link(ob)
            if prop not in arm:
                continue
            for path in ('hide_viewport', 'hide_render'):
                d = ob.driver_add(path).driver
                d.type = 'SCRIPTED'
                v = d.variables.new()
                v.name = 'v'
                v.targets[0].id = arm
                v.targets[0].data_path = f'["{prop}"]'
                d.expression = hidden


def copy_tree(root):
    """A copy of an effect's empty and its objects (data shared)."""
    copies = {}
    for ob in [root] + list(root.children_recursive):
        copies[ob] = ob.copy()
        for c in ob.users_collection:
            c.objects.link(copies[ob])
    for ob, new in copies.items():
        if ob.parent in copies:
            new.parent = copies[ob.parent]
        for mod in getattr(new, 'modifiers', ()):
            for attr in ('object', 'target'):
                if getattr(mod, attr, None) in copies:
                    setattr(mod, attr, copies[getattr(mod, attr)])
    return copies[root]


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


class Floor:
    """The floor under airborne spawn points. scene.ray_cast builds a tree of every map object on each cast (minutes
    on Midtown), so only the objects whose bounds are under the point are cast against, each tree built once."""

    def __init__(self, scene):
        self.dg = bpy.context.evaluated_depsgraph_get()
        self.boxes = []
        for o in scene.objects:
            if o.type != 'MESH' or not o.visible_get():
                continue
            pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
            self.boxes.append((o, Vector([min(p[i] for p in pts) for i in range(3)]),
                               Vector([max(p[i] for p in pts) for i in range(3)])))
        self.trees = {}

    def tree(self, o):
        if o.name not in self.trees:
            self.trees[o.name] = (BVHTree.FromObject(o, self.dg), o.matrix_world.copy())
        return self.trees[o.name]

    def under(self, loc):
        """The highest upward-facing surface within GROUND_REACH under loc; walls and slopes over 45 degrees don't
        count."""
        top, best = loc.z + 0.3, None
        for o, lo, hi in self.boxes:
            if not (lo.x <= loc.x <= hi.x and lo.y <= loc.y <= hi.y and lo.z <= top and hi.z >= loc.z - GROUND_REACH):
                continue
            tree, mw = self.tree(o)
            inv = mw.inverted()
            start = inv @ Vector((loc.x, loc.y, top))
            down = (inv.to_3x3() @ Vector((0, 0, -1))).normalized()
            for _ in range(8):
                at, normal, _i, _d = tree.ray_cast(start, down)
                if at is None:
                    break
                world = mw @ at
                if world.z < loc.z - GROUND_REACH:
                    break
                if (mw.to_3x3().inverted().transposed() @ normal).normalized().z > 0.7:
                    if best is None or world.z > best.z:
                        best = world
                    break
                start = at + down * 0.01
        return best


def ground(floor, spot):
    loc = Vector(spot['location'])
    if not spot.get('ground'):
        return loc
    return floor.under(loc) or loc


def placing(context):
    """Puts the picked NPCs on the scene's map, one NPC (its rig and copies) per step; yields (done, how many, the
    next NPC's name) before each. Appending a rig into a file holding a map takes seconds per rig (Blender's append
    work grows with the file), so the panel shows the progress between them."""
    from . import hero_name, load_rig, register_rig_ui, riglight, rig_path, tools_module
    scene = context.scene
    slug = scene_map(scene)
    spawns = wanted(scene, slug, context) if slug else None
    if not spawns:
        return
    remove(scene)
    # before any NPC is there to be hit
    floor = Floor(scene) if any(s.get('ground') for s in spawns) else None
    for spot in spawns:
        spot['location'] = ground(floor, spot)
    title = tools_module('mapbuild').MAPS.get(slug, slug)
    root = bpy.data.collections.new(f'{title} NPCs')
    root[ROOT] = slug
    scene.collection.children.link(root)
    rnd = random.Random(SEED)
    effects = {}
    kinds = [n for n in dict.fromkeys(s['slug'] for s in spawns) if os.path.exists(rig_path(n, context))]
    for k, npc in enumerate(kinds):
        yield k, len(kinds), hero_name(npc)
        first = load_rig(rig_path(npc, context), npc)
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
            add_effects(arm, npc, effects)
    register_rig_ui()
    riglight.light_rigs(scene)


def spawn(context):
    """All at once; returns the number of NPCs placed."""
    for _ in placing(context):
        pass
    root = npc_root(context.scene)
    return sum(1 for o in root.all_objects if o.type == 'ARMATURE') if root else 0


_placing = {'steps': None, 'at': None}


def stop_placing():
    if _placing['steps'] is not None:
        _placing['steps'].close()
    _placing.update(steps=None, at=None)


def place_later():
    """Runs placing() a step per timer tick so the panel redraws between rigs."""
    stop_placing()
    steps = _placing['steps'] = placing(bpy.context)

    def step():
        from . import redraw
        if _placing['steps'] is not steps:
            return None
        try:
            _placing['at'] = next(steps)
        except StopIteration:
            _placing.update(steps=None, at=None)
            redraw()
            return None
        redraw()
        return 0.05
    bpy.app.timers.register(step, first_interval=0.1)


def toggle(self, context):
    from . import queue_position, start_queue
    scene = context.scene
    stop_placing()
    if not scene.dlr_npcs:
        remove(scene)
        return
    slug = scene_map(scene)
    if not slug or queue_position() is not None:
        # a queue already running places them when it is done
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
            place_later()

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
    row.popover('DLR_PT_npc_groups', text=f'Which ({len(sc.dlr_npc_groups)}/{len(GROUP_IDS)})')
    row = box.row()
    row.enabled = not busy
    row.prop(sc, 'dlr_npc_anims')
    ziplines.draw(box, context)
    if _placing['at'] is not None:
        done, total, name = _placing['at']
        box.label(text=f'Placing {name} ({done + 1}/{total})')
        box.progress(factor=done / total, type='BAR', text=f'{done}/{total}')
    elif pos is not None and _job is not None:
        what = {'NPCS': "Finding the map's NPCs and ziplines", 'RIG': f'Building {hero_name(_job.slug)}',
                'ANIMS': f"Exporting {hero_name(_job.slug)}'s animations"}.get(_job.kind, _job.slug)
        box.label(text=f'{what} ({pos[0]}/{pos[1]})')
        box.progress(factor=_job.progress, type='BAR', text=f'{_job.status}  {_job.progress:.0%}')
        box.operator('dlr.cancel', icon='X')
    elif sc.dlr_npcs_status == 'failed':
        box.label(text='Building the NPCs failed', icon='ERROR')
        if _job is not None and _job.error:
            box.label(text=_job.error[:80])
        box.operator('dlr.show_log', icon='TEXT')


GROUP_IDS = []


class DLR_PT_npc_groups(bpy.types.Panel):
    # a popover rather than a menu: a menu closes on the first click, this stays open for picking several
    bl_label = 'Which NPCs'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'

    def draw(self, context):
        col = self.layout.column(align=True)
        for g in GROUP_IDS:
            col.prop_enum(context.scene, 'dlr_npc_groups', g)


def register():
    from . import tools_module
    S, P = bpy.types.Scene, bpy.props
    groups = tools_module('map_npcs').GROUPS
    GROUP_IDS[:] = [g[0] for g in groups]
    bpy.utils.register_class(DLR_PT_npc_groups)
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
    stop_placing()
    for p in ('dlr_npcs', 'dlr_npc_groups', 'dlr_npc_anims', 'dlr_npcs_status'):
        delattr(bpy.types.Scene, p)
    bpy.utils.unregister_class(DLR_PT_npc_groups)
