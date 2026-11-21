"""Game model browser: a window with a scrolling grid of thumbnails of the game's models (search, folder, sort) for
Add Game Model.

The window is a Text Editor area turned into the browser (Python can't make a new editor type): its sidebar holds the
controls and the main region is drawn here. Thumbnails are made on demand for the models in view and cached: a grey pass first (S2V geometry-only
export, about a second per 100 models), then a textured pass (S2V with materials, ~10 s per 30 models) that
replaces them. Rendering happens in one background Blender kept running (model_thumbs.py), so later batches
don't pay Blender's start-up."""
import importlib
import json
import os
import shutil
import subprocess
import tempfile
import threading

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

from . import gamelaunch, gamerender, uitext

THUMB_SIZE = 160
GREY_BATCH = 96
TEXTURED_BATCH = 32
HERE = os.path.dirname(os.path.abspath(__file__))
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

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
    """Makes the thumbnails of the newest list asked for: grey ones first, then textured, top of the list first."""

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
                # top of the list first: the grey pass is cheap, so it runs ahead in bigger batches
                todo, textures = (grey[:GREY_BATCH], False) if grey else (textured[:TEXTURED_BATCH], True)
                if not todo:
                    break
                self.busy = len(textured)
                try:
                    self.make(todo, textures)
                    self.error = ''
                except Exception as error:  # noqa: BLE001 (shown in the browser; a batch must not kill the thread)
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


def best_thumb(model):
    return next((png for png in thumb_paths(model) if os.path.exists(png)), '')


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


_results = [None, []]


def results(search, folder, sort):
    key = (search, folder, sort)
    if _results[0] == key:
        return _results[1]
    models = gamerender.search_game_models(search, limit=None) if search.strip() else list(gamerender.game_models())
    if folder != 'ALL':
        models = [m for m in models if m.startswith(folder + '/')]
    if sort == 'NAME':
        models.sort(key=lambda m: m.rsplit('/', 1)[-1])
    elif sort == 'FOLDER' or not search.strip():
        models.sort()
    _results[:] = [key, models]
    return models


class Browser:
    """The one open browser window: which area it is, scroll and hover, and the grid layout of the last draw (the
    click and scroll operators hit-test against it)."""

    def __init__(self, window, area):
        self.window = window.as_pointer()
        self.area = area.as_pointer()
        self.scroll = 0.0
        self.hover = -1
        self.size = (1, 1)
        self.visible = []


_browser = None
_textures = {}  # model -> (thumbnail file, GPUTexture); oldest dropped past TEXTURE_CAP
TEXTURE_CAP = 600
LOADS_PER_TICK = 24


def browser_area(context):
    return _browser is not None and context.area is not None and context.area.as_pointer() == _browser.area


def _props(context):
    return context.window_manager.dlr_model_browser


class Grid:
    """Cell geometry for a region size and thumbnail size, in region pixels; row 0 at the top."""

    def __init__(self, width, height, thumb, count):
        s = self.s = bpy.context.preferences.system.ui_scale
        self.pad = 10 * s
        self.status = 24 * s
        self.bar = 10 * s
        self.thumb = thumb * s
        self.text = 32 * s
        self.width, self.height = width, height
        inner = width - 2 * self.pad - self.bar
        self.cols = max(1, int(inner // (self.thumb + 10 * s)))
        self.cell_w = inner / self.cols
        self.cell_h = self.thumb + self.text + 10 * s
        self.rows = -(-count // self.cols)
        self.view = height - self.status
        self.max_scroll = max(0.0, self.rows * self.cell_h + 2 * self.pad - self.view)

    def top(self, row, scroll):
        return self.height - self.pad - row * self.cell_h + scroll

    def hit(self, x, y, scroll, count):
        if y < self.status or x < self.pad or x >= self.width - self.pad - self.bar:
            return -1
        row = int((self.top(0, scroll) - y) // self.cell_h)
        col = int((x - self.pad) // self.cell_w)
        index = row * self.cols + col
        return index if row >= 0 and 0 <= index < count else -1

    def knob(self):
        """Scroll bar track and knob lengths."""
        track = self.view - 2 * self.pad
        return track, max(30 * self.s, track * self.view / (self.view + self.max_scroll))

    def on_bar(self, x):
        return x >= self.width - self.pad - self.bar * 2

    def rows_in_view(self, scroll):
        first = int(max(0.0, scroll - self.pad) // self.cell_h)
        return first, int((scroll + self.view) // self.cell_h) + 1


def _grid(context, models):
    return Grid(*_browser.size, _props(context).thumb_size, len(models))


def _clamp_scroll(grid):
    _browser.scroll = min(max(_browser.scroll, 0.0), grid.max_scroll)


BACK = (0.13, 0.13, 0.13, 1.0)
CELL = (0.2, 0.2, 0.2, 1.0)
CELL_HOVER = (0.27, 0.27, 0.27, 1.0)
SELECTED = (0.28, 0.45, 0.75, 1.0)
PENDING = (0.25, 0.25, 0.25, 1.0)
TEXT = (0.9, 0.9, 0.9, 1.0)
TEXT_DIM = (0.55, 0.55, 0.55, 1.0)


def _rect(shader, x, y, w, h, color):
    batch = batch_for_shader(shader, 'TRI_FAN', {'pos': ((x, y), (x + w, y), (x + w, y + h), (x, y + h))})
    shader.uniform_float('color', color)
    batch.draw(shader)


def _image(shader, tex, x, y, w, h):
    batch = batch_for_shader(shader, 'TRI_FAN', {'pos': ((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
                                                 'texCoord': ((0, 0), (1, 0), (1, 1), (0, 1))})
    shader.uniform_sampler('image', tex)
    batch.draw(shader)


def _fit(text, width):
    if blf.dimensions(0, text)[0] <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if blf.dimensions(0, text[:mid] + '…')[0] <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + '…'


def _text(text, x, y, width, color, centre=True):
    text = _fit(text, width)
    w = blf.dimensions(0, text)[0]
    blf.position(0, x + (width - w) / 2 if centre else x, y, 0)
    blf.color(0, *color)
    blf.draw(0, text)


def _draw():
    context = bpy.context
    if not browser_area(context) or context.region is None:
        return
    region = context.region
    _browser.size = (region.width, region.height)
    props = _props(context)
    models = results(props.search, props.folder, props.sort)
    grid = _grid(context, models)
    _clamp_scroll(grid)
    s = context.preferences.system.ui_scale
    flat = gpu.shader.from_builtin('UNIFORM_COLOR')
    image = gpu.shader.from_builtin('IMAGE')
    gpu.state.blend_set('ALPHA')
    _rect(flat, 0, 0, region.width, region.height, BACK)
    first, last = grid.rows_in_view(_browser.scroll)
    visible = models[first * grid.cols:(last + 1) * grid.cols]
    _browser.visible = visible
    blf.size(0, 11 * s)
    for i, model in enumerate(visible, first * grid.cols):
        row, col = divmod(i, grid.cols)
        x = grid.pad + col * grid.cell_w
        top = grid.top(row, _browser.scroll)
        cell_y = top - grid.cell_h + 4 * s
        if model == props.selected:
            _rect(flat, x + 2 * s, cell_y, grid.cell_w - 4 * s, grid.cell_h - 4 * s, SELECTED)
        elif i == _browser.hover:
            _rect(flat, x + 2 * s, cell_y, grid.cell_w - 4 * s, grid.cell_h - 4 * s, CELL_HOVER)
        tx = x + (grid.cell_w - grid.thumb) / 2
        ty = top - 5 * s - grid.thumb
        _rect(flat, tx, ty, grid.thumb, grid.thumb, CELL)
        tex = _textures.get(model)
        if tex:
            gpu.state.blend_set('ALPHA')  # blf.draw turns blending off
            image.bind()
            _image(image, tex[1], tx, ty, grid.thumb, grid.thumb)
            flat.bind()
        name, folder = model.rsplit('/', 1)[-1], model.rsplit('/', 1)[0] if '/' in model else ''
        _text(name, x + 4 * s, ty - 14 * s, grid.cell_w - 8 * s, TEXT)
        _text(folder, x + 4 * s, ty - 27 * s, grid.cell_w - 8 * s, TEXT_DIM)
    if grid.max_scroll:
        track, knob = grid.knob()
        pos = (track - knob) * _browser.scroll / grid.max_scroll
        bx = region.width - grid.pad - grid.bar + 3 * s
        _rect(flat, bx, grid.status + grid.pad, grid.bar - 4 * s, track, (0.18, 0.18, 0.18, 1))
        _rect(flat, bx, grid.status + grid.pad + track - pos - knob, grid.bar - 4 * s, knob, (0.4, 0.4, 0.4, 1))
    _rect(flat, 0, 0, region.width, grid.status, (0.1, 0.1, 0.1, 1))
    hovered = models[_browser.hover] if 0 <= _browser.hover < len(models) else ''
    status = f'{len(models)} models'
    if hovered:
        status += f'   models/{hovered}.vmdl'
    _text(status, grid.pad, 8 * s, region.width - 2 * grid.pad, TEXT_DIM, centre=False)
    if not models:
        _text('No model matches', 0, region.height / 2, region.width, TEXT_DIM)
    gpu.state.blend_set('NONE')


def _load_texture(model, png):
    try:
        img = bpy.data.images.load(png, check_existing=False)
    except RuntimeError:
        return
    try:
        w, h = img.size
        pixels = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(img)
    tex = gpu.types.GPUTexture((w, h), format='RGBA8', data=gpu.types.Buffer('FLOAT', w * h * 4, pixels))
    _textures.pop(model, None)
    _textures[model] = (png, tex)
    while len(_textures) > TEXTURE_CAP:
        _textures.pop(next(iter(_textures)))


def _find_area():
    for window in bpy.context.window_manager.windows:
        if window.as_pointer() == _browser.window:
            for area in window.screen.areas:
                if area.as_pointer() == _browser.area and area.type == 'TEXT_EDITOR':
                    return window, area
    return None, None


def _tick():
    """Hands the models in view (and a screen below) to the thumbnail worker and loads finished thumbnails."""
    global _browser
    if _browser is None:
        return None
    window, area = _find_area()
    if area is None:
        _browser = None
        _textures.clear()
        return None
    sidebar = next((r for r in area.regions if r.type == 'UI'), None)
    if sidebar is not None and sidebar.active_panel_category != 'Models':
        try:
            sidebar.active_panel_category = 'Models'
        except (TypeError, ValueError, AttributeError):
            pass
    props = bpy.context.window_manager.dlr_model_browser
    models = results(props.search, props.folder, props.sort)
    visible = list(_browser.visible)
    if visible:
        end = models.index(visible[-1]) + 1 if visible[-1] in models else 0
        worker().want(visible + models[end:end + len(visible)])
    changed = False
    loads = 0
    for model in visible:
        png = best_thumb(model)
        if png and _textures.get(model, ('',))[0] != png:
            _load_texture(model, png)
            changed = True
            loads += 1
            if loads >= LOADS_PER_TICK:
                break
    if changed or (_worker and _worker.busy):
        area.tag_redraw()
    return 0.1 if loads >= LOADS_PER_TICK else 0.3


def _event_xy(context, event):
    region = context.region
    return event.mouse_x - region.x, event.mouse_y - region.y


def _models(context):
    props = _props(context)
    return results(props.search, props.folder, props.sort)


def _reveal(context, index):
    models = _models(context)
    grid = _grid(context, models)
    row = index // grid.cols
    top = grid.pad + row * grid.cell_h
    if top < _browser.scroll:
        _browser.scroll = top - grid.pad
    elif top + grid.cell_h > _browser.scroll + grid.view:
        _browser.scroll = top + grid.cell_h - grid.view + grid.pad
    _clamp_scroll(grid)


class BrowserOperator:
    @classmethod
    def poll(cls, context):
        return browser_area(context) and context.region is not None and context.region.type == 'WINDOW'


class DLR_OT_model_browser_click(BrowserOperator, bpy.types.Operator):
    bl_idname = 'dlr.model_browser_click'
    bl_label = 'Select Model'
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        x, y = _event_xy(context, event)
        models = _models(context)
        grid = _grid(context, models)
        if grid.max_scroll and grid.on_bar(x):
            self.grid = grid
            self.drag(context, y)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        index = grid.hit(x, y, _browser.scroll, len(models))
        _props(context).selected = models[index] if index >= 0 else ''
        context.area.tag_redraw()
        return {'FINISHED'}

    def drag(self, context, y):
        grid = self.grid
        track, knob = grid.knob()
        _browser.scroll = (grid.status + grid.pad + track - y - knob / 2) / max(track - knob, 1) * grid.max_scroll
        _clamp_scroll(grid)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self.drag(context, _event_xy(context, event)[1])
        elif event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'ESC'} and event.value == 'RELEASE':
            return {'FINISHED'}
        return {'RUNNING_MODAL'}


class DLR_OT_model_browser_hover(BrowserOperator, bpy.types.Operator):
    bl_idname = 'dlr.model_browser_hover'
    bl_label = 'Hover Model'
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        models = _models(context)
        index = _grid(context, models).hit(*_event_xy(context, event), _browser.scroll, len(models))
        if index != _browser.hover:
            _browser.hover = index
            context.area.tag_redraw()
        return {'PASS_THROUGH'}


class DLR_OT_model_browser_scroll(BrowserOperator, bpy.types.Operator):
    bl_idname = 'dlr.model_browser_scroll'
    bl_label = 'Scroll Models'
    bl_options = {'INTERNAL'}

    rows: bpy.props.FloatProperty()
    pages: bpy.props.FloatProperty()
    to: bpy.props.EnumProperty(items=(('NONE', 'None', ''), ('TOP', 'Top', ''), ('END', 'End', '')))
    zoom: bpy.props.IntProperty()

    def invoke(self, context, event):
        models = _models(context)
        grid = _grid(context, models)
        if self.zoom:
            props = _props(context)
            anchor = _browser.scroll / max(grid.cell_h, 1)
            props.thumb_size = min(max(props.thumb_size + self.zoom, 64), 320)
            grid = _grid(context, models)
            _browser.scroll = anchor * grid.cell_h
        elif event.type == 'TRACKPADPAN':
            _browser.scroll -= event.mouse_y - event.mouse_prev_y
        elif self.to == 'TOP':
            _browser.scroll = 0
        elif self.to == 'END':
            _browser.scroll = grid.max_scroll
        else:
            _browser.scroll += self.rows * grid.cell_h + self.pages * grid.view
        _clamp_scroll(grid)
        _browser.hover = -1
        context.area.tag_redraw()
        return {'FINISHED'}


class DLR_OT_model_browser_step(BrowserOperator, bpy.types.Operator):
    bl_idname = 'dlr.model_browser_step'
    bl_label = 'Move Selection'
    bl_options = {'INTERNAL'}

    dx: bpy.props.IntProperty()
    dy: bpy.props.IntProperty()

    def invoke(self, context, event):
        models = _models(context)
        if not models:
            return {'CANCELLED'}
        props = _props(context)
        cols = _grid(context, models).cols
        index = models.index(props.selected) if props.selected in models else -1
        index = 0 if index < 0 else min(max(index + self.dx + self.dy * cols, 0), len(models) - 1)
        props.selected = models[index]
        _reveal(context, index)
        context.area.tag_redraw()
        return {'FINISHED'}


class DLR_OT_model_browser_import(bpy.types.Operator):
    bl_idname = 'dlr.model_browser_import'
    bl_label = 'Import'
    bl_description = 'Add the selected model at the 3D cursor (double-click or Enter in the grid does the same)'
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return bool(_props(context).selected)

    def invoke(self, context, event):
        if event.type == 'LEFTMOUSE' and context.region and context.region.type == 'WINDOW':
            # a double-click imports what it lands on
            models = _models(context)
            index = _grid(context, models).hit(*_event_xy(context, event), _browser.scroll, len(models))
            if index < 0:
                return {'CANCELLED'}
            _props(context).selected = models[index]
        return self.execute(context)

    def execute(self, context):
        model = _props(context).selected
        result = bpy.ops.dlr.game_add_model('EXEC_DEFAULT', True, model=model)
        if 'FINISHED' in result:
            self.report({'INFO'}, f'Added {model.rsplit("/", 1)[-1]}')
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return result


def _search_changed(self, context):
    if _browser:
        _browser.scroll = 0.0
        _browser.hover = -1


class DLR_PG_model_browser(bpy.types.PropertyGroup):
    search: bpy.props.StringProperty(name='Search', options={'TEXTEDIT_UPDATE', 'SKIP_SAVE'}, update=_search_changed,
                                     description='Words of the model path; hero names find their models too '
                                                 '(holliday -> astro)')
    folder: bpy.props.EnumProperty(name='Folder', items=folder_items, update=_search_changed)
    sort: bpy.props.EnumProperty(name='Sort', items=(
        ('BEST', 'Best Match', 'Models whose file name matches first'),
        ('NAME', 'Name', 'By file name'),
        ('FOLDER', 'Folder', 'By folder, then name'),
    ), update=_search_changed)
    thumb_size: bpy.props.IntProperty(name='Thumbnail Size', default=128, min=64, max=320, subtype='PIXEL',
                                      description='Size of the thumbnails in the grid (Ctrl+wheel there too)')
    selected: bpy.props.StringProperty(name='Model', options={'SKIP_SAVE'})


class DLR_PT_model_browser(bpy.types.Panel):
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Models'
    bl_label = 'Game Models'
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, context):
        return browser_area(context)

    def draw(self, context):
        layout = self.layout
        props = _props(context)
        name = props.selected.rsplit('/', 1)[-1]
        col = layout.column()
        col.scale_y = 1.6
        col.operator('dlr.model_browser_import', text=f'Import {name}' if name else 'Select a Model', icon='IMPORT')
        if props.selected:
            uitext.label(context, layout, f'models/{props.selected}.vmdl')
        layout.separator()
        layout.prop(props, 'search', text='', icon='VIEWZOOM')
        layout.prop(props, 'folder', text='')
        layout.prop(props, 'sort', text='')
        layout.prop(props, 'thumb_size', text='Size')
        layout.separator()
        work = _worker
        if work and work.error:
            col = layout.column()
            col.alert = True
            uitext.label(context, col, work.error, icon='ERROR')
        elif work and work.busy:
            uitext.label(context, layout, f'Making thumbnails ({work.busy} to go)', icon='TIME')
        col = layout.column(align=True)
        col.enabled = False
        col.label(text='Click: select')
        col.label(text='Double-click, Enter: import')
        col.label(text='Arrows: move selection')
        col.label(text='Ctrl+wheel: thumbnail size')


def _not_browser(cls, context):
    return not browser_area(context)


def _text_panels():
    """The Text Editor's own sidebar panels, kept out of the browser window."""
    import bl_ui.space_text as space_text
    return [getattr(space_text, n) for n in ('TEXT_PT_find', 'TEXT_PT_properties') if hasattr(space_text, n)]


def _hide_text_panels(hide):
    for cls in _text_panels():
        if hide == ('poll' in cls.__dict__):
            continue
        try:
            bpy.utils.unregister_class(cls)
            if hide:
                cls.poll = classmethod(_not_browser)
            else:
                del cls.poll
            bpy.utils.register_class(cls)
        except (RuntimeError, ValueError):
            pass


class DLR_OT_game_model_browser(bpy.types.Operator):
    bl_idname = 'dlr.game_model_browser'
    bl_label = 'Add Game Model'
    bl_description = ('Open a window with a grid of thumbnails of the game\'s models to add one: an empty at the 3D '
                      'cursor that the game draws the model at. Thumbnails are made for the models in view the first '
                      'time they are shown (grey first, textured a few seconds later) and kept')

    def execute(self, context):
        global _browser
        if not gamerender.game_models():
            self.report({'ERROR'}, 'Deadlock not found (set its folder in the add-on preferences)')
            return {'CANCELLED'}
        wm = context.window_manager
        if _browser is not None:
            old, _area = _find_area()
            if old is not None:
                with context.temp_override(window=old):
                    bpy.ops.wm.window_close()
            _browser = None
        before = {w.as_pointer() for w in wm.windows}
        # The only way to open a small separate window from Python; its area is then turned into the browser.
        bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        window = next((w for w in wm.windows if w.as_pointer() not in before), None)
        if window is None:
            window = next((w for w in wm.windows if w.screen.areas[0].type == 'PREFERENCES'), None)
        if window is None:
            self.report({'ERROR'}, 'could not open a window')
            return {'CANCELLED'}
        area = window.screen.areas[0]
        area.ui_type = 'TEXT_EDITOR'
        space = area.spaces.active
        space.text = None
        space.show_region_header = False
        space.show_region_footer = False
        space.show_region_ui = True
        _browser = Browser(window, area)
        _textures.clear()
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=0.1)
        return {'FINISHED'}


@bpy.app.handlers.persistent
def _reopen(*_args):
    """A browser window saved with the file comes back with it: make it the browser again."""
    global _browser
    _browser = None
    _textures.clear()
    for window in bpy.context.window_manager.windows:
        areas = window.screen.areas
        if window.screen.is_temporary and len(areas) == 1 and areas[0].type == 'TEXT_EDITOR':
            space = areas[0].spaces.active
            if space.text is None and not space.show_region_header:
                _browser = Browser(window, areas[0])
                if not bpy.app.timers.is_registered(_tick):
                    bpy.app.timers.register(_tick, first_interval=0.5)
                return


KEYS = (
    ('dlr.model_browser_click', 'LEFTMOUSE', 'PRESS', {}, {}),
    ('dlr.model_browser_import', 'LEFTMOUSE', 'DOUBLE_CLICK', {}, {}),
    ('dlr.model_browser_import', 'RET', 'PRESS', {}, {}),
    ('dlr.model_browser_import', 'NUMPAD_ENTER', 'PRESS', {}, {}),
    ('dlr.model_browser_hover', 'MOUSEMOVE', 'ANY', {}, {}),
    ('dlr.model_browser_scroll', 'WHEELUPMOUSE', 'PRESS', {}, {'rows': -0.5}),
    ('dlr.model_browser_scroll', 'WHEELDOWNMOUSE', 'PRESS', {}, {'rows': 0.5}),
    ('dlr.model_browser_scroll', 'WHEELUPMOUSE', 'PRESS', {'ctrl': True}, {'zoom': 16}),
    ('dlr.model_browser_scroll', 'WHEELDOWNMOUSE', 'PRESS', {'ctrl': True}, {'zoom': -16}),
    ('dlr.model_browser_scroll', 'TRACKPADPAN', 'ANY', {}, {}),
    ('dlr.model_browser_scroll', 'PAGE_UP', 'PRESS', {}, {'pages': -0.9}),
    ('dlr.model_browser_scroll', 'PAGE_DOWN', 'PRESS', {}, {'pages': 0.9}),
    ('dlr.model_browser_scroll', 'HOME', 'PRESS', {}, {'to': 'TOP'}),
    ('dlr.model_browser_scroll', 'END', 'PRESS', {}, {'to': 'END'}),
    ('dlr.model_browser_step', 'LEFT_ARROW', 'PRESS', {}, {'dx': -1}),
    ('dlr.model_browser_step', 'RIGHT_ARROW', 'PRESS', {}, {'dx': 1}),
    ('dlr.model_browser_step', 'UP_ARROW', 'PRESS', {}, {'dy': -1}),
    ('dlr.model_browser_step', 'DOWN_ARROW', 'PRESS', {}, {'dy': 1}),
)
CLASSES = (DLR_PG_model_browser, DLR_PT_model_browser, DLR_OT_game_model_browser, DLR_OT_model_browser_click,
           DLR_OT_model_browser_hover, DLR_OT_model_browser_scroll, DLR_OT_model_browser_step,
           DLR_OT_model_browser_import)
_keymap_items = []
_draw_handle = None


def register():
    global _draw_handle
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.dlr_model_browser = bpy.props.PointerProperty(type=DLR_PG_model_browser)
    _draw_handle = bpy.types.SpaceTextEditor.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')
    _hide_text_panels(True)
    bpy.app.handlers.load_post.append(_reopen)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:
        keymap = keyconfig.keymaps.new('Text', space_type='TEXT_EDITOR')
        for idname, key, value, mods, props in KEYS:
            item = keymap.keymap_items.new(idname, key, value, **mods)
            for name, v in props.items():
                setattr(item.properties, name, v)
            _keymap_items.append((keymap, item))


def unregister():
    global _browser, _worker, _draw_handle
    for keymap, item in _keymap_items:
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _keymap_items.clear()
    _hide_text_panels(False)
    if _reopen in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reopen)
    if _draw_handle:
        bpy.types.SpaceTextEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _browser = None
    _textures.clear()
    if _worker:
        _worker.stop()
        _worker = None
    del bpy.types.WindowManager.dlr_model_browser
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
