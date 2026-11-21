"""Game model browser: a grid of thumbnails of the game's models (search, folder, sort) for Add Game Model.

Thumbnails are made on demand for the page on screen and cached: a grey pass first (S2V geometry-only export,
well under a second for a page), then a textured pass (S2V with materials, ~10 s for 30 models) that replaces
them. Rendering happens in one background Blender kept running (model_thumbs.py), so pages after the first don't
pay Blender's start-up."""
import importlib
import json
import os
import shutil
import subprocess
import tempfile
import threading

import bpy
import bpy.utils.previews
import numpy as np

from . import gamelaunch, gamerender

THUMB_SIZE = 160
COLUMNS = 8
PAGE_SIZE = COLUMNS * 4
HERE = os.path.dirname(os.path.abspath(__file__))
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

_previews = None
_worker = None


def _addon():
    return importlib.import_module(__package__)


_thumb_dir = None


def thumb_dir():
    global _thumb_dir
    if _thumb_dir is None:
        _thumb_dir = _addon().user_dir('model_thumbs')
    return _thumb_dir


def thumb_paths(model):
    """(textured png, grey png) of a short model path like 'props/barrel_03'."""
    base = os.path.join(thumb_dir(), model.replace('/', '__'))
    return base + '.png', base + '.g.png'


def _done(png):
    return os.path.exists(png) or os.path.exists(png + '.none')


class ThumbWorker(threading.Thread):
    """Makes the thumbnails of the newest page asked for: grey ones first, then textured."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.wanted = []
        self.running = True
        self.busy = 0
        self.error = ''
        self.blender = None
        self.s2v = None
        # resolved on the main thread: bpy isn't safe to call from this one
        self.vpk = os.path.join(gamerender.game_dir() or '', 'citadel', 'pak01_dir.vpk')
        self.s2v_cache = _addon().user_dir('s2v')
        self.pipeline = _addon().tools_module('pipeline')
        self.blender_path = bpy.app.binary_path
        thumb_dir()

    def want(self, models):
        with self.lock:
            if models == self.wanted:
                return
            self.wanted = list(models)
        self.event.set()
        if not bpy.app.timers.is_registered(_redraw):
            bpy.app.timers.register(_redraw, first_interval=0.5)

    def pending(self):
        with self.lock:
            wanted = list(self.wanted)
        return self.running and any(not _done(thumb_paths(m)[0]) for m in wanted) and not (
            self.error and not self.busy)

    def run(self):
        while self.running:
            self.event.wait()
            self.event.clear()
            while self.running:
                with self.lock:
                    wanted = list(self.wanted)
                grey = [m for m in wanted if not _done(thumb_paths(m)[0]) and not _done(thumb_paths(m)[1])]
                textured = [m for m in wanted if not _done(thumb_paths(m)[0])]
                todo, textures = (grey, False) if grey else (textured, True)
                if not todo:
                    break
                self.busy = len(textured)
                try:
                    self.make(todo, textures)
                    self.error = ''
                except Exception as error:  # noqa: BLE001 (shown in the browser; a page must not kill the thread)
                    self.error = str(error)
                    gamelaunch.log(f'model thumbnails: {error}')
                    for m in todo:
                        png = thumb_paths(m)[0 if textures else 1]
                        open(png + '.none', 'w').close()
            self.busy = 0

    def make(self, models, textures):
        vpk = self.vpk
        if not os.path.exists(vpk):
            raise FileNotFoundError('Deadlock not found (set its folder in the add-on preferences)')
        if self.s2v is None:
            self.s2v = self.pipeline.find_s2v(cache=self.s2v_cache)
        tmp = tempfile.mkdtemp(prefix='export_', dir=thumb_dir())
        try:
            cmd = [self.s2v, '-i', vpk, '-f', ','.join(f'models/{m}.vmdl_c' for m in models), '-o', tmp,
                   '--gltf_export_format', 'glb', '--threads', str(os.cpu_count() or 4), '-q']
            if textures:
                cmd.append('--gltf_export_materials')
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
                           creationflags=NO_WINDOW)
            items = []
            for m in models:
                glb = os.path.join(tmp, 'models', *m.split('/')) + '.glb'
                png = thumb_paths(m)[0 if textures else 1]
                if os.path.exists(glb):
                    items.append([glb, png])
                else:
                    open(png + '.none', 'w').close()
            if items:
                self.render(items, textures)
            if textures:
                for m in models:
                    grey = thumb_paths(m)[1]
                    if os.path.exists(thumb_paths(m)[0]) and os.path.exists(grey):
                        os.remove(grey)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def render(self, items, textures):
        if self.blender is None or self.blender.poll() is not None:
            self.blender = subprocess.Popen(
                [self.blender_path, '-b', '--factory-startup', '-P', os.path.join(HERE, 'model_thumbs.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=NO_WINDOW)
        self.blender.stdin.write(json.dumps({'items': items, 'size': THUMB_SIZE, 'textured': textures}) + '\n')
        self.blender.stdin.flush()
        for line in self.blender.stdout:
            if line.startswith('DONE'):
                return
        raise RuntimeError('the thumbnail renderer (background Blender) quit')

    def stop(self):
        self.running = False
        self.event.set()
        if self.blender and self.blender.poll() is None:
            self.blender.kill()


def worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = ThumbWorker()
        _worker.start()
    return _worker


_loaded = {}  # model -> thumbnail file its preview holds ('' = the placeholder)
PENDING = (0.25, 0.25, 0.25, 0.4)


def best_thumb(model):
    return next((png for png in thumb_paths(model) if os.path.exists(png)), '')


def icon(model):
    """Preview icon id for a model. One preview per model whose pixels are swapped in place when a better
    thumbnail lands: the open dialog can't be made to rebuild from a timer, but it redraws its icons."""
    if model not in _previews:
        _previews.new(model)
        _loaded[model] = None
    _update(model)
    return _previews[model].icon_id


def _update(model):
    png = best_thumb(model)
    if _loaded.get(model) == png:
        return False
    preview = _previews[model]
    if png:
        try:
            image = bpy.data.images.load(png, check_existing=False)
        except RuntimeError:
            return False
        try:
            pixels = np.empty(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(pixels)
            preview.image_size = image.size[:]
            preview.image_pixels_float.foreach_set(pixels)
        finally:
            bpy.data.images.remove(image)
    else:
        preview.image_size = (2, 2)
        preview.image_pixels_float.foreach_set(np.tile(np.array(PENDING, np.float32), 4))
    _loaded[model] = png
    return True


_folder_items = []


def folder_items(self, context):
    if not _folder_items:
        counts = {}
        for model in gamerender.game_models():
            folder = model.split('/', 1)[0] if '/' in model else ''
            counts[folder] = counts.get(folder, 0) + 1
        _folder_items.append(('ALL', f'All folders ({sum(counts.values())})', ''))
        _folder_items.extend((f, f'{f} ({n})', '') for f, n in sorted(counts.items()) if f)
    return _folder_items


def results(search, folder, sort):
    models = gamerender.search_game_models(search, limit=None) if search.strip() else list(gamerender.game_models())
    if folder != 'ALL':
        models = [m for m in models if m.startswith(folder + '/')]
    if sort == 'NAME':
        models.sort(key=lambda m: m.rsplit('/', 1)[-1])
    elif sort == 'FOLDER' or not search.strip():
        models.sort()
    return models


def _search_changed(self, context):
    self.page = 1


class DLR_OT_game_model_browser(bpy.types.Operator):
    bl_idname = 'dlr.game_model_browser'
    bl_label = 'Add Game Model'
    bl_description = ('Pick a game model from a grid of thumbnails: an empty at the 3D cursor that the game draws '
                      'the model at. Thumbnails are made for each page the first time it is shown (grey first, '
                      'textured a few seconds later) and kept')

    search: bpy.props.StringProperty(name='Search', options={'TEXTEDIT_UPDATE', 'SKIP_SAVE'}, update=_search_changed,
                                     description='Words of the model path; hero names find their models too '
                                                 '(holliday -> astro)')
    folder: bpy.props.EnumProperty(name='Folder', items=folder_items, update=_search_changed)
    sort: bpy.props.EnumProperty(name='Sort', items=(
        ('BEST', 'Best Match', 'Models whose file name matches first'),
        ('NAME', 'Name', 'By file name'),
        ('FOLDER', 'Folder', 'By folder, then name'),
    ), update=_search_changed)
    page: bpy.props.IntProperty(name='Page', default=1, min=1, options={'SKIP_SAVE'})

    def invoke(self, context, event):
        if not gamerender.game_models():
            self.report({'ERROR'}, 'Deadlock not found (set its folder in the add-on preferences)')
            return {'CANCELLED'}
        # previews are ~100 KB each: start each browse with an empty set
        _previews.clear()
        _loaded.clear()
        return context.window_manager.invoke_props_dialog(self, width=900, title='Game Models', confirm_text='Close')

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.prop(self, 'search', text='', icon='VIEWZOOM')
        row.prop(self, 'folder', text='')
        row.prop(self, 'sort', text='')
        models = results(self.search, self.folder, self.sort)
        pages = max(1, -(-len(models) // PAGE_SIZE))
        page = min(self.page, pages)
        shown = models[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
        work = worker()
        work.want(shown)
        row = layout.row()
        row.label(text=f'{len(models)} models')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.prop(self, 'page', text=f'Page (of {pages})')
        if work.error:
            layout.label(text=work.error, icon='ERROR')
        if not shown:
            layout.label(text='No model matches')
            return
        grid = layout.grid_flow(row_major=True, columns=COLUMNS, even_columns=True, even_rows=True, align=False)
        for model in shown:
            cell = grid.column(align=True)
            cell.template_icon(icon_value=icon(model), scale=5.0)
            cell.operator_context = 'EXEC_DEFAULT'
            name = model.rsplit('/', 1)[-1]
            cell.operator(gamerender.DLR_OT_game_add_model.bl_idname, text=name).model = model
            cell.label(text=model.rsplit('/', 1)[0] if '/' in model else '')


def _redraw():
    """Thumbnails land from the worker thread: copy them into the shown previews."""
    if _worker is None or _previews is None:
        return None
    with _worker.lock:
        wanted = list(_worker.wanted)
    if any([_update(m) for m in wanted if m in _previews]):
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return 0.5 if _worker.pending() else None


def register():
    global _previews
    _previews = bpy.utils.previews.new()
    bpy.utils.register_class(DLR_OT_game_model_browser)


def unregister():
    global _previews, _worker
    bpy.utils.unregister_class(DLR_OT_game_model_browser)
    if bpy.app.timers.is_registered(_redraw):
        bpy.app.timers.unregister(_redraw)
    if _worker:
        _worker.stop()
        _worker = None
    if _previews:
        bpy.utils.previews.remove(_previews)
        _previews = None
