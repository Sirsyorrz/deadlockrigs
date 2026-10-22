# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock Rigs: builds a hero's rig from the user's own Deadlock install (tools/pipeline.py in a background
process) and opens or appends it."""
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading

import bpy
from bpy.app.handlers import persistent

HERE = os.path.dirname(os.path.abspath(__file__))


def tools_dir():
    # packaged: tools/ ships inside the add-on; in the repo it sits next to addon/
    for d in (os.path.join(HERE, 'tools'), os.path.join(os.path.dirname(HERE), 'tools')):
        if os.path.exists(os.path.join(d, 'pipeline.py')):
            return d
    raise RuntimeError('tools/pipeline.py not found next to the add-on')


_heroes = []


def heroes():
    if not _heroes:
        path = os.path.join(os.path.dirname(tools_dir()), 'heroes.json')
        _heroes.extend(sorted(json.load(open(path, encoding='utf-8')), key=lambda h: h['name']))
    return _heroes


def hero_items(self, context):
    return [(h['slug'], h['name'], '') for h in heroes()]


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
                 'anims': 'Importing animations into Blender', 'clean': 'Cleaning up'}

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
            self.log = (self.log + [line])[-200:]
            print('[Deadlock Rigs]', line)
            if line.startswith('STEP '):
                count, step = line.split()[1:3]
                i, n = (int(x) for x in count.split('/'))
                self.progress = (i - 1) / n
                self.status = self.STEP_TEXT.get(step, step)
            elif line.startswith('downloading'):
                self.status = 'Downloading Source2Viewer (first build only)'
            elif line.startswith('ERROR'):
                self.error = line.split(':', 1)[-1].strip()

    def cancel(self):
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


def tick():
    global _job
    if _job is None:
        return None
    _job.poll()
    redraw()
    if not _job.done:
        return 0.25
    if _job.ok and _job.kind == 'RIG':
        after = bpy.context.scene.dlr_after
        path = rig_path(_job.slug)
        if after == 'OPEN':
            bpy.app.timers.register(lambda: open_rig(path) and None, first_interval=0.1)
        elif after == 'APPEND':
            append_rig(path, _job.slug)
    return None


def start(context, slug, kind):
    global _job
    p = prefs(context)
    if kind == 'RIG':
        steps = ['extract', 'data', 'base', 'rig'] + ([] if p.keep_build_files else ['clean'])
    else:
        steps = ['export_anims']
    cmd = [sys.executable, os.path.join(tools_dir(), 'pipeline.py'), slug, *steps,
           '--blender', bpy.app.binary_path, '--python', sys.executable,
           '--work', user_dir('work'), '--out', output_dir(context), '--s2v-cache', user_dir('s2v'), '--backup']
    if p.deadlock_dir:
        cmd += ['--pak', bpy.path.abspath(p.deadlock_dir)]
    if p.s2v_path:
        cmd += ['--s2v', bpy.path.abspath(p.s2v_path)]
    if p.exports:
        cmd.append('--exports')
    if p.keep_build_files:
        cmd.append('--keep')
    env = dict(os.environ)
    sp = scipy_path()
    if sp:
        env['PYTHONPATH'] = os.pathsep.join([sp] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env['PYTHONUNBUFFERED'] = '1'
    env['DLRIG_ADDON'] = '1'
    _job = Job(slug, kind, cmd, env)
    bpy.app.timers.register(tick, first_interval=0.25)


# ---------------------------------------------------------------- opening a rig

def register_rig_ui():
    """Runs the rig's panel script. Only for rigs in the output folder, which this add-on built."""
    text = bpy.data.texts.get('dlrig_ui.py')
    if text is None or hasattr(bpy.types, 'DLRIG_PT_main'):
        return
    exec(compile(text.as_string(), text.name, 'exec'), {'__name__': 'dlrig_ui'})


@persistent
def on_load(_dummy):
    # with "Auto Run Python Scripts" off Blender skips the rig's panel script; rigs this add-on built are trusted
    if bpy.context.preferences.filepaths.use_scripts_auto_execute or not bpy.data.filepath:
        return
    out = os.path.normcase(os.path.realpath(output_dir()))
    if os.path.normcase(os.path.realpath(bpy.data.filepath)).startswith(out + os.sep):
        register_rig_ui()


def open_rig(path):
    bpy.ops.wm.open_mainfile(filepath=path)
    return True


def append_rig(path, slug):
    name = hero_name(slug)
    if bpy.data.texts.get('dlrig_ui.py') or any('dlrig' in o for o in bpy.data.objects if o.type == 'ARMATURE'):
        print('[Deadlock Rigs] this file already has a rig; its panel serves the first one')
    with bpy.data.libraries.load(path) as (src, dst):
        dst.collections = [c for c in src.collections if c == name]
        dst.texts = [t for t in src.texts if t.startswith('dlrig_')]
    for c in dst.collections:
        bpy.context.scene.collection.children.link(c)
        # the rig looks for its clip library next to the open file, which is no longer the rig's own
        for ob in c.all_objects:
            if ob.type == 'ARMATURE' and 'dlrig' in ob:
                ob['anim_library'] = os.path.join(os.path.dirname(path), f'{slug}_anims.blend')
    register_rig_ui()


# ---------------------------------------------------------------- operators and panel

class DLR_OT_build(bpy.types.Operator):
    bl_idname = 'dlr.build'
    bl_label = 'Build Rig'
    bl_description = "Build the hero's rig from your Deadlock install"

    kind: bpy.props.EnumProperty(items=[('RIG', 'Rig', ''), ('ANIMS', 'Animations', '')])

    @classmethod
    def description(cls, context, props):
        if props.kind == 'ANIMS':
            return ("Save the hero's game animations as a .glb, and as a .blend next to the rig that its "
                    'Game Animations panel loads from')
        return cls.bl_description

    @classmethod
    def poll(cls, context):
        return _job is None or _job.done

    def execute(self, context):
        try:
            start(context, context.scene.dlr_hero, self.kind)
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

        if _job is not None and _job.slug == sc.dlr_hero:
            box = layout.box()
            if busy:
                box.progress(factor=_job.progress, type='BAR', text=_job.status)
                box.operator('dlr.cancel', icon='X')
            elif _job.ok:
                box.label(text=f'{hero_name(_job.slug)}: done', icon='CHECKMARK')
            else:
                box.label(text=f'{hero_name(_job.slug)}: failed', icon='ERROR')
                # labels cut long text, so wrap it to the sidebar's width
                width = max(20, int(context.region.width / (7 * context.preferences.system.ui_scale)))
                lines = [w for ln in (_job.error or 'see the build log').split('\n')[:4]
                         for w in textwrap.wrap(ln, width)]
                col = box.column(align=True)
                for line in lines[:8]:
                    col.label(text=line)
                box.operator('dlr.show_log', icon='TEXT')

        path = rig_path(sc.dlr_hero, context)
        if os.path.exists(path):
            row = layout.row(align=True)
            row.operator('dlr.open', text='Open Rig', icon='FILE_FOLDER').append = False
            row.operator('dlr.open', text='Append to Scene', icon='APPEND_BLEND').append = True
        layout.operator('dlr.open_folder', icon='FOLDER_REDIRECT')


CLASSES = (DLR_Preferences, DLR_OT_build, DLR_OT_cancel, DLR_OT_open, DLR_OT_show_log, DLR_OT_open_folder,
           DLR_PT_main)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.dlr_hero = bpy.props.EnumProperty(name='Hero', items=hero_items)
    bpy.types.Scene.dlr_after = bpy.props.EnumProperty(
        name='After building', default='OPEN',
        items=[('OPEN', 'Then Open', 'Open the rig when it is built'),
               ('APPEND', 'Then Append', 'Append the rig into this scene when it is built'),
               ('NONE', 'Just Save', 'Only save it to the output folder')])
    bpy.app.handlers.load_post.append(on_load)


def unregister():
    if _job:
        _job.cancel()
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    del bpy.types.Scene.dlr_hero
    del bpy.types.Scene.dlr_after
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
