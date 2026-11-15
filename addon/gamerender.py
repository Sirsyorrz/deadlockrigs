# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock as a Blender render engine (spike): the viewport's camera drives a running Deadlock and the game's frame
is drawn back into the viewport.

The game is an -insecure dev session with -netconport (worklog notes "Game capture"). Everything goes through its
console: no injected code yet, so the frame comes back as a jpeg_screenshot file (about 0.17 s at 4K).

Camera facts measured on build 6861:
- Blender world = game units x 0.0254 turned -90 degrees about Z (built maps and rigs): Blender (x, y) =
  (game y, -game x). Checked on the hideout's playButtonFogandLight (ent_text vs the built map's light).
- With camera overrides on and every pivot/offset at 0, the camera sits exactly at the pawn's origin, so
  setpos_exact/setang_exact place it; noclip keeps it from falling.
- citadel_camera_override_fov is the vertical FOV in degrees and clamps below 50. Narrower views are a centre crop
  of a 50 degree frame (lower resolution).
"""
import math
import os
import shutil
import socket
import struct
import threading
import time

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

UNIT = 0.0254
GAME_FROM_BLENDER = Matrix.Diagonal((1 / UNIT, 1 / UNIT, 1 / UNIT, 1)) @ Matrix.Rotation(math.pi / 2, 4, 'Z')
MIN_VFOV = 50.0
MAX_VFOV = 120.0
PORT = 29000
SETTLE = 0.06  # after a teleport, before the shot: the client needs a frame or two to catch up
ENGINE_ID = 'DLR_DEADLOCK_GAME'
SETUP = (
    'citadel_camera_free_roam 0', 'noclip 1', 'citadel_camera_overrides_enabled 1',
    'citadel_camera_override_pivot_x_offset 0', 'citadel_camera_override_pivot_y_offset 0',
    'citadel_camera_override_pivot_z_offset 0', 'citadel_camera_override_x_offset 0',
    'citadel_camera_override_y_offset 0', 'citadel_camera_override_z_offset 0',
    'citadel_camera_override_x_worst_case_offset 0', 'citadel_camera_override_y_worst_case_offset 0',
    'citadel_camera_override_z_worst_case_offset 0', 'citadel_hud_visible 0', 'cl_hud_telemetry_frametime_show 0',
    'cl_hud_telemetry_ping_show 0', 'engine_no_focus_sleep 0', 'jpeg_quality 95',
)

_HEADER = struct.Struct('!4sHIH')


def _steam_game_dir():
    roots = [os.path.expanduser(p) for p in ('~/.local/share/Steam', '~/.steam/steam',
                                             'C:/Program Files (x86)/Steam')]
    for root in roots:
        candidate = os.path.join(root, 'steamapps', 'common', 'Deadlock', 'game')
        if os.path.isdir(candidate):
            return candidate
    return None


def game_dir():
    given = os.environ.get('DLR_DEADLOCK_GAME')
    if given:
        return given
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.deadlock_dir:
            return os.path.join(bpy.path.abspath(prefs.deadlock_dir), 'game')
    except (KeyError, AttributeError, TypeError):
        pass
    return _steam_game_dir()


class Netcon:
    """Send-only VConsole2 client. A reader thread drains the socket: the game logs constantly and a full buffer
    would stall it."""

    def __init__(self, port=PORT):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=2.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(0.5)
        self.alive = True
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while self.alive:
            try:
                if not self.sock.recv(1 << 20):
                    break
            except socket.timeout:
                continue
            except OSError:
                break
        self.alive = False

    def send(self, *commands):
        packets = []
        for c in commands:
            payload = c.encode('utf-8') + b'\x00'
            packets.append(_HEADER.pack(b'CMND', 0x00D4, _HEADER.size + len(payload), 0) + payload)
        self.sock.sendall(b''.join(packets))

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class View:
    """A game camera: position in game units, pitch/yaw in degrees (Source convention: +pitch looks down), vertical
    FOV in degrees as asked of the game."""
    __slots__ = ('pos', 'pitch', 'yaw', 'vfov')

    def __init__(self, pos, pitch, yaw, vfov):
        self.pos, self.pitch, self.yaw, self.vfov = pos, pitch, yaw, vfov

    def close_to(self, other):
        return (other is not None and max(abs(a - b) for a, b in zip(self.pos, other.pos)) < 0.05
                and abs(self.pitch - other.pitch) < 0.01 and abs((self.yaw - other.yaw + 180) % 360 - 180) < 0.01
                and abs(self.vfov - other.vfov) < 0.01)

    def commands(self):
        x, y, z = self.pos
        return (f'setpos_exact {x:.3f} {y:.3f} {z:.3f}', f'setang_exact {self.pitch:.4f} {self.yaw:.4f} 0',
                f'citadel_camera_override_fov {self.vfov:.3f}')


def game_view(cam_matrix, region_vfov, region_aspect, game_aspect):
    """The game camera that covers a Blender view: same eye and direction, a vertical FOV wide enough that the
    game's frame (game_aspect) covers the region in both directions. Roll is dropped (setang ignores it)."""
    g = GAME_FROM_BLENDER @ cam_matrix
    eye = g.translation
    fx, fy, fz = (-g.col[2].xyz).normalized()
    yaw = math.degrees(math.atan2(fy, fx))
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, fz))))
    t = math.tan(math.radians(region_vfov) / 2)
    need = max(t, t * region_aspect / game_aspect)
    vfov = min(MAX_VFOV, max(MIN_VFOV, math.degrees(2 * math.atan(need))))
    return View(tuple(eye), pitch, yaw, vfov)


class Link:
    """One game connection. view_draw posts the wanted view; a worker thread places the game camera, takes a
    jpeg screenshot and hands the file back. Latest wins: views posted while a shot is in flight are skipped."""

    def __init__(self, port=PORT):
        self.port = port
        self.con = None
        self.error = ''
        self.wanted = None
        self.sent = None
        self.frame = None  # (path, View it was taken with, sequence)
        self.seq = 0
        self.timing = 0.0
        self.event = threading.Event()
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.con = Netcon(self.port)
        self.con.send(*SETUP)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        self.event.set()
        if self.con:
            self.con.close()
            self.con = None

    def want(self, view):
        with self.lock:
            if view.close_to(self.wanted):
                return
            self.wanted = view
        self.event.set()

    def _run(self):
        shots = os.path.join(game_dir() or '', 'citadel', 'screenshots')
        cache = os.path.join(bpy.app.tempdir or '/tmp', 'dlr_game_frames')
        os.makedirs(cache, exist_ok=True)
        while self.running:
            self.event.wait(0.5)
            self.event.clear()
            with self.lock:
                view = self.wanted
            if view is None or view.close_to(self.sent):
                continue
            if not (self.con and self.con.alive):
                self.error = 'lost the game console'
                self.running = False
                break
            try:
                start = time.time()
                self.con.send(*view.commands())
                time.sleep(SETTLE)
                path = self._shoot(shots, start)
            except OSError as error:
                self.error = str(error)
                continue
            if path is None:
                self.error = 'no screenshot appeared (is the game window minimised?)'
                continue
            self.seq += 1
            # Two alternating files: Blender may still be reading the previous one.
            dest = os.path.join(cache, f'frame{self.seq % 2}.jpg')
            shutil.move(path, dest)
            self.sent = view
            self.frame = (dest, view, self.seq)
            self.timing = time.time() - start
            self.error = ''

    def _shoot(self, folder, since, timeout=3.0):
        before = set(os.listdir(folder)) if os.path.isdir(folder) else set()
        self.con.send('jpeg_screenshot')
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.01)
            try:
                new = [n for n in os.listdir(folder) if n.endswith('.jpg') and n not in before]
            except FileNotFoundError:
                continue
            if new:
                path = os.path.join(folder, new[0])
                size = -1
                # The engine writes asynchronously: wait until the size settles.
                while size != os.path.getsize(path):
                    size = os.path.getsize(path)
                    time.sleep(0.01)
                return path
        return None


_link = None
_shown = None  # (View, sequence) of the frame loaded into the image; set by the timer, read by view_draw


def link():
    return _link


def _image(path):
    img = bpy.data.images.get('.dlr_game_frame')
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = '.dlr_game_frame'
    else:
        img.filepath = path
        img.reload()
    return img


class DeadlockGameEngine(bpy.types.RenderEngine):
    bl_idname = ENGINE_ID
    bl_label = 'Deadlock (game)'
    bl_use_preview = False
    bl_use_shading_nodes_custom = False

    def render(self, depsgraph):
        self.report({'WARNING'}, 'Deadlock (game): only the viewport works in this spike')

    def view_update(self, context, depsgraph):
        pass

    def view_draw(self, context, depsgraph):
        region, rv3d = context.region, context.region_data
        lk = _link
        if lk is None or not lk.running or not rv3d.is_perspective:
            return
        wm = rv3d.window_matrix
        region_vfov = math.degrees(2 * math.atan(1 / wm[1][1]))
        aspect = region.width / max(region.height, 1)
        img = bpy.data.images.get('.dlr_game_frame')
        game_aspect = img.size[0] / img.size[1] if img and img.size[1] else 16 / 9
        lk.want(game_view(rv3d.view_matrix.inverted(), region_vfov, aspect, game_aspect))
        if _shown is None or img is None:
            return
        shot = _shown[0]
        # The frame's corners in tangent space, through the region's projection (keeps camera-view shift/zoom).
        tv = math.tan(math.radians(shot.vfov) / 2)
        th = tv * game_aspect
        x0, x1 = -wm[0][0] * th - wm[0][2], wm[0][0] * th - wm[0][2]
        y0, y1 = -wm[1][1] * tv - wm[1][2], wm[1][1] * tv - wm[1][2]
        texture = gpu.texture.from_image(img)
        shader = gpu.shader.from_builtin('IMAGE')
        batch = batch_for_shader(shader, 'TRI_FAN', {
            'pos': ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            'texCoord': ((0, 0), (1, 0), (1, 1), (0, 1)),
        })
        with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
            gpu.matrix.load_identity()
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            shader.uniform_sampler('image', texture)
            batch.draw(shader)


def _poll_frames():
    """Load a new frame and redraw Deadlock viewports. Here, not in view_draw: Blender forbids writing data-blocks
    while drawing, and the worker thread can't touch Blender at all."""
    global _shown
    lk = _link
    frame = lk.frame if lk else None
    if frame and (_shown is None or frame[2] != _shown[1]):
        _image(frame[0])
        _shown = (frame[1], frame[2])
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED':
                    area.tag_redraw()
    return 0.03


class DLR_OT_game_connect(bpy.types.Operator):
    bl_idname = 'dlr.game_connect'
    bl_label = 'Connect to Deadlock'
    bl_description = 'Connect to a running -insecure Deadlock session (netcon port) and drive its camera'

    port: bpy.props.IntProperty(default=PORT)

    def execute(self, context):
        global _link
        if _link:
            _link.stop()
        _link = Link(self.port)
        try:
            _link.start()
        except OSError as error:
            _link = None
            self.report({'ERROR'}, f'no game console on port {self.port}: {error}')
            return {'CANCELLED'}
        return {'FINISHED'}


class DLR_OT_game_disconnect(bpy.types.Operator):
    bl_idname = 'dlr.game_disconnect'
    bl_label = 'Disconnect'

    def execute(self, context):
        global _link
        if _link:
            _link.stop()
            _link = None
        return {'FINISHED'}


class DLR_PT_game_render(bpy.types.Panel):
    bl_label = 'Deadlock Game'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    COMPAT_ENGINES = {ENGINE_ID}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        col = self.layout.column()
        lk = _link
        if lk and lk.running:
            col.label(text=f'Connected (port {lk.port})', icon='LINKED')
            if lk.frame:
                col.label(text=f'Frame {lk.frame[2]}, {lk.timing * 1000:.0f} ms')
            col.operator(DLR_OT_game_disconnect.bl_idname)
        else:
            col.operator(DLR_OT_game_connect.bl_idname, icon='UNLINKED')
        if lk and lk.error:
            col.label(text=lk.error, icon='ERROR')


CLASSES = (DeadlockGameEngine, DLR_OT_game_connect, DLR_OT_game_disconnect, DLR_PT_game_render)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    if not bpy.app.timers.is_registered(_poll_frames):
        bpy.app.timers.register(_poll_frames, persistent=True)


def unregister():
    global _link
    if _link:
        _link.stop()
        _link = None
    if bpy.app.timers.is_registered(_poll_frames):
        bpy.app.timers.unregister(_poll_frames)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
