# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock Rigs: builds a hero's rig from the user's own Deadlock install (tools/pipeline.py in a background
process) and opens or appends it."""
import importlib.util
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading

import bpy
from bpy.app.handlers import persistent

from . import crop, effects, look, npcs, particles, riglight, textures, ziplines

HERE = os.path.dirname(os.path.abspath(__file__))


def tools_dir():
    # packaged: tools/ ships inside the add-on; in the repo it sits next to addon/
    for d in (os.path.join(HERE, 'tools'), os.path.join(os.path.dirname(HERE), 'tools')):
        if os.path.exists(os.path.join(d, 'pipeline.py')):
            return d
    raise RuntimeError('tools/pipeline.py not found next to the add-on')


_heroes = []
_hero_items = []


def heroes():
    if not _heroes:
        path = os.path.join(os.path.dirname(tools_dir()), 'heroes.json')
        _heroes.extend(sorted(json.load(open(path, encoding='utf-8')), key=lambda h: h['name']))
    return _heroes


def hero_items(self, context):
    # kept in a module list: Blender reads the item strings after the callback returns
    if not _hero_items:
        _hero_items.extend((h['slug'], h['name'], '') for h in heroes() if not h.get('npc') and not h.get('form'))
        # an empty identifier is a heading in the menu
        for heading, key in (('Forms', 'form'), ('NPCs', 'npc')):
            group = [(h['slug'], h['name'], '') for h in heroes() if h.get(key)]
            if group:
                _hero_items.extend([None, ('', heading, '')] + group)
    return _hero_items


def hero_name(slug):
    return next((h['name'] for h in heroes() if h['slug'] == slug), slug)


def prefs(context=None):
    return (context or bpy.context).preferences.addons[__package__].preferences


def user_dir(sub):
    try:
        return bpy.utils.extension_path_user(__package__, path=sub, create=True)
    except (ValueError, AttributeError):
        # installed as a legacy add-on
        d = os.path.join(bpy.utils.user_resource('DATAFILES', create=True), 'deadlock_rigs', sub)
        os.makedirs(d, exist_ok=True)
        return d


def output_dir(context=None):
    return bpy.path.abspath(prefs(context).output_dir) or default_output()


def default_output():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'DeadlockRigs')


def rig_path(slug, context=None):
    return os.path.join(output_dir(context), slug, f'{slug}_rig.blend')


def tools_module(name):
    """A module of tools/, imported as <add-on>.tools.<name>. Blender's extension policy forbids adding to sys.path
    or importing top-level modules from the extension's folder; in the repo tools/ isn't inside the add-on at all."""
    package = f'{__package__}.tools'
    if package not in sys.modules:
        d = tools_dir()
        spec = importlib.util.spec_from_file_location(package, os.path.join(d, '__init__.py'),
                                                      submodule_search_locations=[d])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f'{package}.{name}')


def map_path(slug, context=None):
    return os.path.join(output_dir(context), 'maps', slug, f'{slug}.blend')


_map_state = {}


def map_state(slug, context):
    """'missing', 'current', 'outdated' (the game or the map changed since the build), 'old' (built by an earlier
    add-on with a bug fixed since), 'mods' (built with the other Include Mods setting), or 'unknown' when the game
    isn't found. Cached: it reads the Steam library list."""
    if slug not in _map_state:
        mapbuild = tools_module('mapbuild')
        built = mapbuild.built_stamp(output_dir(context), slug)
        if built is None:
            state = 'missing'
        else:
            try:
                given = prefs(context).deadlock_dir
                pak = mapbuild.find_pak(bpy.path.abspath(given) if given else None)
                mods = prefs(context).map_mods
                if mapbuild.game_stamp(pak, slug, mods) == built:
                    state = 'current'
                elif mapbuild.game_stamp(pak, slug, not mods) == built:
                    state = 'mods'
                elif built.get('builder', 1) < mapbuild.BUILDER:
                    state = 'old'
                else:
                    state = 'outdated'
            except Exception:
                state = 'unknown'
        _map_state[slug] = (state, (built or {}).get('game_version', ''))
    return _map_state[slug]


# ---------------------------------------------------------------- preferences

class DLR_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    output_dir: bpy.props.StringProperty(
        name='Output Folder', subtype='DIR_PATH', default=default_output(),
        description='Each hero is saved in its own folder here')
    deadlock_dir: bpy.props.StringProperty(
        name='Deadlock Folder', subtype='DIR_PATH', default='',
        description="Deadlock's install folder. Leave empty to find it through Steam")
    s2v_path: bpy.props.StringProperty(
        name='Source2Viewer-CLI', subtype='FILE_PATH', default='',
        description='An existing Source2Viewer-CLI to use. Leave empty to download it on the first build')
    exports: bpy.props.BoolProperty(
        name='Also save the model as .glb and .fbx', default=False,
        description='For other programs (Unity, Unreal, Maya). The rig itself is Blender only')
    map_mods: bpy.props.BoolProperty(
        name='Include Mods', default=True, update=lambda self, context: _map_state.clear(),
        description="Build maps with the mods in Deadlock's addons folders (such as replaced posters and signs), "
                    "as the game shows them. Off: the map as Valve ships it")
    keep_build_files: bpy.props.BoolProperty(
        name='Keep build files', default=False,
        description='Keep the extracted game files and intermediate data (for reporting problems)')

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, 'output_dir')
        col.prop(self, 'deadlock_dir')
        col.prop(self, 's2v_path')
        col.prop(self, 'exports')
        col.prop(self, 'keep_build_files')
        col.label(text=f'Build files: {user_dir("work")}', icon='INFO')


# ---------------------------------------------------------------- the build job

class Job:
    """One pipeline.py run in a background process; its output is read on a thread and drained by a timer."""

    def __init__(self, slug, kind, cmd, env):
        self.slug, self.kind = slug, kind
        self.lines = queue.Queue()
        self.status = 'Starting'
        self.progress = 0.0
        self.error = ''
        self.log = []
        self.done = False
        self.ok = False
        self.quiet = False
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding='utf-8', errors='replace', env=env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip())
        self.lines.put(None)

    STEP_TEXT = {'extract': 'Extracting from the game', 'data': 'Reading cloth and face data',
                 'base': 'Importing the model', 'rig': 'Building the rig', 'export_anims': 'Exporting animations',
                 'anims': 'Importing animations into Blender', 'clean': 'Cleaning up',
                 'map_export': 'Exporting the map from the game', 'map_props': 'Exporting the animated props',
                 'map_rebatch': 'Merging the map objects',
                 'map_lighting': 'Exporting the lighting, sky and fog', 'map_import': 'Importing the map into Blender'}

    def poll(self):
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.proc.wait()
                self.done = True
                self.ok = self.proc.returncode == 0 and not self.error
                if self.ok:
                    self.status, self.progress = 'Done', 1.0
                return
            if line.startswith('PROGRESS '):
                self.progress = float(line.split()[1])
                continue
            self.log = (self.log + [line])[-200:]
            print('[Deadlock Rigs]', line)
            if line.startswith('STEP '):
                step = line.split()[2]
                self.status = self.STEP_TEXT.get(step, step)
            elif line.startswith('downloading'):
                self.status = 'Downloading Source2Viewer (first build only)'
            elif line.startswith('ERROR'):
                self.error = line.split(':', 1)[-1].strip()

    def cancel(self):
        _queue.clear()
        if not self.done:
            self.proc.kill()
            self.error = 'Cancelled'


_job = None


def redraw():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def scipy_path():
    try:
        import scipy
    except ImportError:
        return None
    return os.path.dirname(os.path.dirname(scipy.__file__))


# builds run one after another (the NPCs a map needs): what is still to run, how many there were, and what to call
# with the outcome (True when all of them worked) once the last one is done
_queue = []
_queue_info = {'total': 0, 'done': None}


def start_queue(context, jobs, done):
    _queue[:] = jobs
    _queue_info.update(total=len(jobs), done=done)
    if not jobs:
        _finish_queue(True)
        return
    start(context, *_queue.pop(0), quiet=True)


def queue_position():
    """(this job's number, how many) while a queue runs, else None."""
    if _queue_info['done'] is None:
        return None
    return _queue_info['total'] - len(_queue), _queue_info['total']


def _finish_queue(ok):
    done = _queue_info['done']
    _queue_info.update(total=0, done=None)
    _queue.clear()
    if done:
        done(ok)


def tick():
    global _job
    if _job is None:
        return None
    _job.poll()
    redraw()
    if not _job.done:
        return 0.25
    _map_state.clear()
    if _job.quiet:
        if _job.ok and _queue:
            start(bpy.context, *_queue.pop(0), quiet=True)
        else:
            _finish_queue(_job.ok)
        return None
    if _job.ok and _job.kind == 'RIG':
        after = bpy.context.scene.dlr_after
        path = rig_path(_job.slug)
        if after == 'OPEN':
            bpy.app.timers.register(lambda: open_rig(path) and None, first_interval=0.1)
        elif after == 'APPEND':
            append_rig(path, _job.slug)
    return None


def start(context, slug, kind, quiet=False):
    """quiet: part of a queue, nothing is opened or appended when it is done."""
    global _job
    p = prefs(context)
    if kind == 'MAP':
        cmd = [sys.executable, os.path.join(tools_dir(), 'mapbuild.py'), slug, '--force']
    elif kind == 'NPCS':
        cmd = [sys.executable, os.path.join(tools_dir(), 'mapbuild.py'), slug, '--npcs']
    else:
        steps = ['export_anims'] if kind == 'ANIMS' else \
            ['extract', 'data', 'base', 'rig'] + ([] if p.keep_build_files else ['clean'])
        cmd = [sys.executable, os.path.join(tools_dir(), 'pipeline.py'), slug, *steps, '--backup']
    cmd += ['--blender', bpy.app.binary_path, '--python', sys.executable,
            '--work', user_dir('work'), '--out', output_dir(context), '--s2v-cache', user_dir('s2v')]
    if p.deadlock_dir:
        cmd += ['--pak', bpy.path.abspath(p.deadlock_dir)]
    if p.s2v_path:
        cmd += ['--s2v', bpy.path.abspath(p.s2v_path)]
    if p.exports and kind not in ('MAP', 'NPCS'):
        cmd.append('--exports')
    if kind == 'MAP' and not p.map_mods:
        cmd.append('--no-mods')
    if p.keep_build_files:
        cmd.append('--keep')
    env = dict(os.environ)
    sp = scipy_path()
    if sp:
        env['PYTHONPATH'] = os.pathsep.join([sp] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env['PYTHONUNBUFFERED'] = '1'
    env['DLRIG_ADDON'] = '1'
    _job = Job(slug, kind, cmd, env)
    _job.quiet = quiet
    bpy.app.timers.register(tick, first_interval=0.25)


# ---------------------------------------------------------------- opening a rig

def has_rigs():
    return any('dlrig' in o for o in bpy.data.objects if o.type == 'ARMATURE')


def register_rig_ui():
    """Registers the add-on's own copy of the rig panel. It serves every rig in the file and replaces the copy saved
    in a rig's text block, which in older rigs only knows its own hero. Returns its namespace."""
    path = os.path.join(tools_dir(), 'rig_ui.py')
    ns = {'__name__': 'dlrig_ui'}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    effects.attach_panel()
    return ns


def idle_pose(ui, objects):
    """The panel's Idle Pose on each rig among objects (rigs built before it existed stay at rest)."""
    for ob in objects:
        if ob.type == 'ARMATURE' and 'idle_pose' in ob:
            ui['idle_pose'](bpy.context, ob)
    ui['update'](bpy.context)


# a file opened with Start in Idle Pose on: the option is a property of the scene that is being replaced
_idle_on_load = False


def pick_open_hero(scene):
    """Points the hero menu at the file's rig (the active one if there are several); a newly opened file's scene
    starts at the first hero, so Export Animations would otherwise export Abrams."""
    rigs = [o for o in scene.objects if o.type == 'ARMATURE' and 'dlrig' in o]
    active = bpy.context.view_layer.objects.active
    rig = active if active in rigs else next(iter(rigs), None)
    if rig is not None and any(item and item[0] == rig['dlrig'] for item in hero_items(scene, bpy.context)):
        scene.dlr_hero = rig['dlrig']


@persistent
def on_load(_dummy):
    global _idle_on_load
    if has_rigs():
        pick_open_hero(bpy.context.scene)
        ui = register_rig_ui()
        if _idle_on_load:
            idle_pose(ui, bpy.data.objects)
    _idle_on_load = False


def open_rig(path):
    global _idle_on_load
    _idle_on_load = bpy.context.scene.dlr_idle_pose
    bpy.ops.wm.open_mainfile(filepath=path)
    return True


def append_rig(path, slug):
    idle = bpy.context.scene.dlr_idle_pose
    collections = load_rig(path, slug)
    for c in collections:
        bpy.context.scene.collection.children.link(c)
    ui = register_rig_ui()
    if idle:
        idle_pose(ui, [o for c in collections for o in c.all_objects])
    riglight.light_rigs(bpy.context.scene)


def load_rig(path, slug):
    """Appends the rig's collection without putting it in a scene."""
    name = hero_name(slug)
    with bpy.data.libraries.load(path) as (src, dst):
        dst.collections = [c for c in src.collections if c == name]
        dst.texts = [t for t in src.texts if t.startswith('dlrig_')]
    for c in dst.collections:
        # the rig looks for its clip library next to the open file, which is no longer the rig's own
        for ob in c.all_objects:
            if ob.type == 'ARMATURE' and 'dlrig' in ob:
                ob['anim_library'] = os.path.join(os.path.dirname(path), f'{slug}_anims.blend')
                # older rigs don't point to their texts, and a second rig's texts arrive renamed (.001)
                for t in dst.texts:
                    prop = next((p for p in ('dlrig_ui', 'dlrig_physics') if t.name.startswith(p)), None)
                    if prop and not isinstance(ob.get(prop), bpy.types.Text):
                        ob[prop] = t
    return dst.collections


def append_map(path, slug, link=False):
    """Appends or links the map's collection, makes its sky the scene's world and turns on the game's tonemapping."""
    title = tools_module('mapbuild').MAPS[slug]
    with bpy.data.libraries.load(path, link=link) as (src, dst):
        dst.collections = [c for c in src.collections if c == title]
        dst.worlds = [w for w in src.worlds if w == f'{title} sky']
        # in the same load as the collection, so an appended map's group uses its own (movable) origin empty
        dst.node_groups = [g for g in src.node_groups if g == riglight.PROBE_GROUP]
    scene = bpy.context.scene
    for c in dst.collections:
        scene.collection.children.link(c)
    if dst.worlds:
        scene.world = dst.worlds[0]
    look.enable(scene)
    look.show_in_viewport(bpy.context)
    riglight.light_rigs(scene)
    textures.apply(scene)
    return dst.collections


# ---------------------------------------------------------------- operators and panel

class DLR_OT_build(bpy.types.Operator):
    bl_idname = 'dlr.build'
    bl_label = 'Build Rig'
    bl_description = "Build the hero's rig from your Deadlock install"

    kind: bpy.props.EnumProperty(items=[('RIG', 'Rig', ''), ('ANIMS', 'Animations', ''), ('MAP', 'Map', '')])

    @classmethod
    def description(cls, context, props):
        if props.kind == 'MAP':
            return ('Export the map from your Deadlock install into a .blend (about 10 minutes and 12 GB of free '
                    'space while building). Rebuilt only after a game update')
        if props.kind == 'ANIMS':
            return ("Save the hero's game animations as a .glb, and as a .blend next to the rig that its "
                    'Game Animations panel loads from')
        return cls.bl_description

    @classmethod
    def poll(cls, context):
        return _job is None or _job.done

    def execute(self, context):
        slug = context.scene.dlr_map if self.kind == 'MAP' else context.scene.dlr_hero
        try:
            start(context, slug, self.kind)
        except (OSError, RuntimeError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class DLR_OT_cancel(bpy.types.Operator):
    bl_idname = 'dlr.cancel'
    bl_label = 'Cancel'

    def execute(self, context):
        if _job:
            _job.cancel()
        return {'FINISHED'}


class DLR_OT_open(bpy.types.Operator):
    bl_idname = 'dlr.open'
    bl_label = 'Open Rig'
    bl_description = 'Open the built rig (unsaved changes in this file are lost)'

    append: bpy.props.BoolProperty()

    def execute(self, context):
        slug = context.scene.dlr_hero
        path = rig_path(slug, context)
        if not os.path.exists(path):
            self.report({'ERROR'}, f'No rig built yet at {path}')
            return {'CANCELLED'}
        if self.append:
            append_rig(path, slug)
        else:
            open_rig(path)
        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.append and bpy.data.is_dirty:
            return context.window_manager.invoke_confirm(self, event, message='Unsaved changes will be lost')
        return self.execute(context)


class DLR_OT_append_map(bpy.types.Operator):
    bl_idname = 'dlr.append_map'
    bl_label = 'Append Map'

    link: bpy.props.BoolProperty()

    @classmethod
    def description(cls, context, props):
        if props.link:
            return ('Link the built map into this scene (about 10 seconds). It lights, renders and crops the same '
                    "but can't be edited, and the file keeps pointing at the map in the output folder")
        return ('Copy the built map into this scene so it can be edited. Blender freezes for several '
                'minutes while it appends (Midtown takes about 5); Link Map is much faster')

    def invoke(self, context, event):
        if self.link:
            return self.execute(context)
        return context.window_manager.invoke_confirm(
            self, event, title='Append Map',
            message='Appending takes several minutes and Blender stops responding until it finishes. '
                    'Use Link Map unless you need to edit the map.',
            confirm_text='Append')

    def execute(self, context):
        slug = context.scene.dlr_map
        path = map_path(slug, context)
        if not os.path.exists(path):
            self.report({'ERROR'}, f'No map built yet at {path}')
            return {'CANCELLED'}
        if any(c.get('dlr_map') == slug for c in context.scene.collection.children_recursive):
            self.report({'WARNING'}, 'This scene already has the map; adding another copy')
        append_map(path, slug, self.link)
        return {'FINISHED'}


class DLR_OT_show_log(bpy.types.Operator):
    bl_idname = 'dlr.show_log'
    bl_label = 'Show Build Log'
    bl_description = 'Put the last build output in a text block (Text Editor)'

    def execute(self, context):
        text = bpy.data.texts.get('deadlock_rigs_build.log') or bpy.data.texts.new('deadlock_rigs_build.log')
        text.from_string('\n'.join(_job.log) if _job else '')
        # a new window turned into a Text Editor showing the log
        before = set(context.window_manager.windows)
        bpy.ops.wm.window_new()
        win = next((w for w in context.window_manager.windows if w not in before), None)
        if win is None:
            self.report({'INFO'}, 'Build log is in the Text Editor as deadlock_rigs_build.log')
            return {'FINISHED'}
        area = win.screen.areas[0]
        area.type = 'TEXT_EDITOR'
        area.spaces[0].text = text
        area.spaces[0].show_word_wrap = True
        area.spaces[0].top = max(0, len(text.lines) - 40)
        return {'FINISHED'}


class DLR_OT_open_folder(bpy.types.Operator):
    bl_idname = 'dlr.open_folder'
    bl_label = 'Show Output Folder'
    bl_description = "Open the hero's output folder in the file manager"

    def execute(self, context):
        hero_dir = os.path.join(output_dir(context), context.scene.dlr_hero)
        path = hero_dir if os.path.isdir(hero_dir) else output_dir(context)
        os.makedirs(path, exist_ok=True)
        bpy.ops.wm.path_open(filepath=path)
        return {'FINISHED'}


class DLR_PT_main(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Deadlock Rigs'
    bl_label = 'Deadlock Rigs'

    def draw(self, context):
        layout = self.layout
        sc = context.scene
        col = layout.column(align=True)
        col.prop(sc, 'dlr_hero', text='')
        busy = _job is not None and not _job.done
        row = col.row(align=True)
        row.enabled = not busy
        row.operator('dlr.build', text='Build Rig', icon='ARMATURE_DATA').kind = 'RIG'
        row.prop(sc, 'dlr_after', text='')
        sub = col.row()
        sub.enabled = not busy
        sub.operator('dlr.build', text='Export Animations', icon='ACTION').kind = 'ANIMS'
        col.prop(sc, 'dlr_idle_pose')

        if _job is not None and _job.kind != 'MAP' and not _job.quiet and _job.slug == sc.dlr_hero:
            draw_job(layout, context, hero_name(_job.slug))

        path = rig_path(sc.dlr_hero, context)
        if os.path.exists(path):
            row = layout.row(align=True)
            row.operator('dlr.open', text='Open Rig', icon='FILE_FOLDER').append = False
            row.operator('dlr.open', text='Append to Scene', icon='APPEND_BLEND').append = True
        layout.operator('dlr.open_folder', icon='FOLDER_REDIRECT')


def draw_job(layout, context, name):
    box = layout.box()
    if not _job.done:
        box.progress(factor=_job.progress, type='BAR', text=f'{_job.status}  {_job.progress:.0%}')
        box.operator('dlr.cancel', icon='X')
    elif _job.ok:
        box.label(text=f'{name}: done', icon='CHECKMARK')
    else:
        box.label(text=f'{name}: failed', icon='ERROR')
        # labels cut long text, so wrap it to the sidebar's width
        width = max(20, int(context.region.width / (7 * context.preferences.system.ui_scale)))
        lines = [w for ln in (_job.error or 'see the build log').split('\n')[:4] for w in textwrap.wrap(ln, width)]
        col = box.column(align=True)
        for line in lines[:8]:
            col.label(text=line)
        box.operator('dlr.show_log', icon='TEXT')


MAP_STATE_TEXT = {'missing': ('Not built yet', 'INFO'), 'current': ('Up to date (game {})', 'CHECKMARK'),
                  'outdated': ('Game updated since build {}, rebuild', 'ERROR'),
                  'mods': ('Built with mods {}, rebuild', 'ERROR'),
                  'old': ('Built by an older add-on, rebuild', 'ERROR'),
                  'unknown': ('Built for game {}', 'INFO')}


class DLR_PT_map(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Deadlock Rigs'
    bl_label = 'Map'
    bl_parent_id = 'DLR_PT_main'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        sc = context.scene
        slug = sc.dlr_map
        col = layout.column(align=True)
        col.prop(sc, 'dlr_map', text='')
        busy = _job is not None and not _job.done
        state, version = map_state(slug, context)
        row = col.row()
        row.enabled = not busy
        row.operator('dlr.build', text='Rebuild Map' if state in ('outdated', 'mods', 'old') else 'Build Map',
                     icon='WORLD').kind = 'MAP'
        row = col.row()
        row.enabled = not busy
        row.prop(prefs(context), 'map_mods')
        text, icon = MAP_STATE_TEXT[state]
        if state == 'mods':
            version = 'off' if prefs(context).map_mods else 'on'
        layout.label(text=text.format(version), icon=icon)
        if _job is not None and _job.kind == 'MAP' and _job.slug == slug:
            draw_job(layout, context, tools_module('mapbuild').MAPS[slug])
        if state != 'missing':
            row = layout.row(align=True)
            row.operator('dlr.append_map', text='Link Map', icon='LINK_BLEND').link = True
            row.operator('dlr.append_map', text='Append Map', icon='APPEND_BLEND').link = False
        look.draw(layout, context)
        if crop.any_map(sc):
            npcs.draw(layout, context)
            riglight.draw(layout, context)
            textures.draw(layout, context)
            crop.draw(layout, context)
        particles.draw(layout, context)


_map_items = []


def map_items(self, context):
    if not _map_items:
        _map_items.extend((slug, title, '') for slug, title in tools_module('mapbuild').MAPS.items())
    return _map_items


CLASSES = (DLR_Preferences, DLR_OT_build, DLR_OT_cancel, DLR_OT_open, DLR_OT_append_map, DLR_OT_show_log,
           DLR_OT_open_folder, DLR_PT_main, DLR_PT_map)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.dlr_hero = bpy.props.EnumProperty(name='Hero', items=hero_items)
    bpy.types.Scene.dlr_map = bpy.props.EnumProperty(name='Map', items=map_items)
    bpy.types.Scene.dlr_after = bpy.props.EnumProperty(
        name='After building', default='OPEN',
        items=[('OPEN', 'Then Open', 'Open the rig when it is built'),
               ('APPEND', 'Then Append', 'Append the rig into this scene when it is built'),
               ('NONE', 'Just Save', 'Only save it to the output folder')])
    bpy.types.Scene.dlr_idle_pose = bpy.props.BoolProperty(
        name='Start in Idle Pose', default=False,
        description="Open or append the hero standing as in the game's idle (the pose its weapon hold comes from) "
                    "instead of the rest pose. The rig panel's Idle Pose button does the same later")
    bpy.app.handlers.load_post.append(on_load)
    crop.register()
    textures.register()
    riglight.register()
    particles.register()
    effects.register()
    npcs.register()
    ziplines.register()


def unregister():
    if _job:
        _job.cancel()
    ziplines.unregister()
    npcs.unregister()
    effects.unregister()
    particles.unregister()
    riglight.unregister()
    textures.unregister()
    crop.unregister()
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    del bpy.types.Scene.dlr_hero
    del bpy.types.Scene.dlr_map
    del bpy.types.Scene.dlr_after
    del bpy.types.Scene.dlr_idle_pose
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
