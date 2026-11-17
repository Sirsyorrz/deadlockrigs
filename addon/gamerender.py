# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock as a Blender render engine (spike): the viewport's camera drives a running Deadlock and the game's frame
is drawn back into the viewport, with the scene's own meshes drawn solid on top. F12 renders the scene camera the
same way, from a full quality screenshot.

The game is an -insecure dev session with -netconport (worklog notes "Game capture"). Everything goes through its
console: no injected code yet, so frames come back as screenshot files (jpeg for the viewport, png for F12).

Camera facts measured on build 6861:
- Blender world = game units x 0.0254 turned -90 degrees about Z (built maps and rigs): Blender (x, y) =
  (game y, -game x). Checked on the hideout's playButtonFogandLight (ent_text vs the built map's light).
- With camera overrides on and every pivot/offset at 0, the camera sits exactly at the pawn's origin, so
  setpos_exact/setang_exact place it; noclip keeps it from falling. The camera is the player's own hero.
- citadel_camera_override_fov is the vertical FOV in degrees and clamps below 50. Narrower views are a centre crop
  of a 50 degree frame (lower resolution).
"""
import atexit
import json
import math
import mmap
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Quaternion, Vector

UNIT = 0.0254
GAME_FROM_BLENDER = Matrix.Diagonal((1 / UNIT, 1 / UNIT, 1 / UNIT, 1)) @ Matrix.Rotation(math.pi / 2, 4, 'Z')
BLENDER_FROM_GAME = GAME_FROM_BLENDER.inverted()
MIN_VFOV = 50.0
MAX_VFOV = 120.0
# Game units the camera may go to: past these the game kills the hero for leaving the play area. Probed on the
# hideout (x/y 14000 fine, 16200 out; z -3000 fine, -4500 out; 6000 fine, 8000 out), kept a little inside.
PLAY_AREA = ((-13500, 13500), (-13500, 13500), (-2800, 5500))
PORT = 29000
SETTLE = 0.06  # after a teleport, before the shot: the client needs a frame or two to catch up
# Camera sent to the x11grab stream showing it, at 1080p: console setang 65-105 ms, bridge 28-34 ms. A frame is
# paired with the last view sent at least this long before it arrived.
STREAM_LATENCY = 0.08
STREAM_LATENCY_BRIDGE = 0.025
SPAWNED = os.path.expanduser('~/.cache/dlr-game/spawned.json')  # models this Blender put in the game
SEND_INTERVAL = 1 / 60
FRAME = '.dlr_game_frame'
ENGINE_ID = 'DLR_DEADLOCK_GAME'
BRIDGE_PORT = 29001
_BRIDGE_COMMAND = struct.Struct('<4sII3f3fff')
_BRIDGE_STATUS = struct.Struct('<4sIIIIii3f3fff96s')
_POSE_HEAD = struct.Struct('<4sIIII64s')  # slot, entity index (0 = newest unclaimed by model), flags, count, model
_POSE_REPLY = struct.Struct('<4sIII3f96s')
_BIND_HEAD = struct.Struct('<4sIII')
_BIND_BONE = struct.Struct('<8fi60s')
_TIME_REPLY = struct.Struct('<4sIIId')


def _time_state(data):
    f = _TIME_REPLY.unpack(data[:_TIME_REPLY.size])
    game_time = struct.unpack_from('<f', data, _TIME_REPLY.size)[0] if len(data) >= _TIME_REPLY.size + 4 else -1.0
    return {'available': f[1], 'frozen': f[2], 'step_done': f[3], 'framerate': f[4], 'game_time': game_time}
POSE_SLOTS = 8
PAWN_SLOT = POSE_SLOTS - 1  # holds the player's own hero folded away
# A puppet's entity follows its pose (the game culls a model by its entity's bounds) once the pose has moved this
# far, in game units.
FOLLOW_DISTANCE = 16.0
# Close enough to the pose that the model's bounds still cover it (a fresh spawn starts metres away).
ARRIVED = 48.0
# The server only sends entities near the player (puppets and map props far from the hidden hero are missing or
# half drawn): the hero follows the bridge camera once it is this far away (game units), and a capture right
# after a move waits PAWN_STREAM_IN seconds for the entities to arrive.
PAWN_FOLLOW = 512.0
PAWN_STREAM_IN = 0.5
SETUP_COMMON = ('citadel_hud_visible 0', 'cl_hud_telemetry_frametime_show 0', 'cl_hud_telemetry_ping_show 0',
                'engine_no_focus_sleep 0', 'jpeg_quality 95')
# Console camera (no bridge): the camera is the hero's own eye.
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
    """VConsole2 client. A reader thread drains the socket (the game logs constantly and a full buffer would stall
    it) and keeps recent console lines for request().

    The game only serves one console client at a time, and drops commands sent before its connect handshake
    (the replay of its buffered log) is over: send() waits for that."""

    HANDSHAKE_END = 'End VConsole Buffered Messages'

    def __init__(self, port=PORT):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=2.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(0.5)
        self.alive = True
        self.ready = threading.Event()
        self.lines = []
        self.lines_lock = threading.Lock()
        self.send_lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True).start()
        if not self.ready.wait(5.0):
            self.ready.set()

    def _drain(self):
        raw = bytearray()
        while self.alive:
            try:
                data = self.sock.recv(1 << 20)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            raw.extend(data)
            while len(raw) >= _HEADER.size:
                kind, _, length, _ = _HEADER.unpack_from(raw)
                if length < _HEADER.size or len(raw) < length:
                    break
                payload = bytes(raw[_HEADER.size:length])
                del raw[:length]
                if kind == b'PRNT' and len(payload) > 28:
                    text = payload[28:].rstrip(b'\x00').decode('utf-8', errors='replace')
                    if self.HANDSHAKE_END in text:
                        self.ready.set()
                    with self.lines_lock:
                        self.lines.extend(text.splitlines())
                        del self.lines[:-400]
        self.alive = False
        self.ready.set()

    def request(self, command, starts_with, timeout=3.0):
        """Send command and return the first console line after it that starts with starts_with, or None."""
        with self.lines_lock:
            self.lines.clear()
        self.send(command)
        end = time.time() + timeout
        while time.time() < end:
            with self.lines_lock:
                for line in self.lines:
                    if line.strip().startswith(starts_with):
                        return line.strip()
            time.sleep(0.02)
        return None

    def send(self, *commands):
        self.ready.wait(5.0)
        packets = []
        for c in commands:
            payload = c.encode('utf-8') + b'\x00'
            packets.append(_HEADER.pack(b'CMND', 0x00D4, _HEADER.size + len(payload), 0) + payload)
        with self.send_lock:
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

    def inside(self):
        return all(lo <= v <= hi for v, (lo, hi) in zip(self.pos, PLAY_AREA))

    def commands(self):
        x, y, z = self.pos
        return (f'setpos_exact {x:.3f} {y:.3f} {z:.3f}', f'setang_exact {self.pitch:.4f} {self.yaw:.4f} 0',
                f'citadel_camera_override_fov {self.vfov:.3f}')


class Bridge:
    """The camera bridge DLL in the game (native/dlr_bridge): sets the main view's eye, angles and horizontal FOV
    each frame, independent of the hero. UDP on localhost; every packet is answered with the bridge's status."""

    def __init__(self, port=BRIDGE_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.addr = ('127.0.0.1', port)
        # The game keeps the last camera number it applied across Blender sessions: start above any earlier
        # session's (they send at most ~60 a second), so a frame tagged with an old number never passes for ours.
        self.seq = int(time.time() * 100) & 0x3fffffff
        self.lock = threading.Lock()
        self.size = (1920, 1080)

    def _exchange(self, packet):
        with self.lock:
            self.sock.sendto(packet, self.addr)
            data, _ = self.sock.recvfrom(4096)
        f = _BRIDGE_STATUS.unpack(data[:_BRIDGE_STATUS.size])
        status = {'state': f[2], 'views': f[3], 'applied': f[4], 'size': (f[5], f[6]), 'origin': f[7:10],
                  'angles': f[10:13], 'fov': f[13], 'message': f[15].split(b'\0')[0].decode(errors='replace')}
        if status['size'][1] > 0:
            self.size = status['size']
        return status

    def status(self):
        return self._exchange(b'ping')

    def place(self, view):
        """Hand the main view to this camera; returns the sequence number the bridge acknowledges once applied."""
        w, h = self.size
        hfov = math.degrees(2 * math.atan(math.tan(math.radians(view.vfov) / 2) * w / h))
        self.seq += 1
        self._exchange(_BRIDGE_COMMAND.pack(b'DLRC', self.seq, 1, *view.pos, view.pitch, view.yaw, 0.0, hfov, 0.0))
        return self.seq

    def wait_applied(self, seq, frames=3, timeout=2.0):
        """Until the game has drawn a few main views with the camera of this sequence number."""
        end = time.time() + timeout
        first = None
        while time.time() < end:
            status = self.status()
            if status['applied'] == seq:
                if first is None:
                    first = status['views']
                elif status['views'] - first >= frames:
                    return True
            time.sleep(0.005)
        return False

    def release(self):
        self.seq += 1
        try:
            self._exchange(_BRIDGE_COMMAND.pack(b'DLRC', self.seq, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        except OSError:
            pass

    def close(self):
        self.sock.close()


class PoseChannel:
    """The bridge's pose slots: a slot binds a game model entity, and while Blender keeps talking to the bridge the
    game draws that entity with the world-space bone transforms last sent for it. Own socket: the camera's replies
    must not mix with these."""

    def __init__(self, port=BRIDGE_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.addr = ('127.0.0.1', port)
        self.lock = threading.Lock()

    def _exchange(self, packet):
        with self.lock:
            self.sock.sendto(packet, self.addr)
            return self.sock.recvfrom(65536)[0]

    def pose(self, slot, model, pose=b'', count=0, entity=0, active=True):
        """Bind the slot (to `entity`, or the newest unclaimed entity whose model path contains `model`) and set its
        pose (count x 32 bytes: pos xyz, scale, quat xyzw, game world space). Returns the bound entity or None."""
        head = _POSE_HEAD.pack(b'DLRP', slot, entity, 1 if active and count else 0, count, model.encode()[:63])
        f = _POSE_REPLY.unpack(self._exchange(head + pose)[:_POSE_REPLY.size])
        if not f[1]:
            return None
        return {'entity': f[2], 'count': f[3], 'origin': f[4:7], 'model': f[7].split(b'\0')[0].decode()}

    def time(self, frozen, step=0, seconds=0.0):
        """Freeze the game clock, or with a new `step` number advance it once by `seconds` (exactly one game frame).
        Returns the bridge's time state."""
        data = self._exchange(struct.pack('<4sIIf', b'DLRT', 1 if frozen else 0, step, seconds))
        return _time_state(data)

    def entities(self, model):
        """Indices of the game's entities whose model path contains `model`."""
        data = self._exchange(b'DLRL' + model.encode()[:63].ljust(64, b'\0'))
        count = struct.unpack_from('<I', data, 4)[0]
        return set(struct.unpack_from(f'<{count}I', data, 8))

    def skeleton(self, slot):
        """Bone names, parents and inverse binds (game model space) of the slot's model."""
        names, parents, inv_binds = [], [], []
        while True:
            data = self._exchange(struct.pack('<4sII', b'DLRB', slot, len(names)))
            _, total, first, count = _BIND_HEAD.unpack(data[:_BIND_HEAD.size])
            if not total or first != len(names):
                raise ConnectionError('the bridge did not return the model skeleton')
            for i in range(count):
                f = _BIND_BONE.unpack_from(data, _BIND_HEAD.size + i * _BIND_BONE.size)
                inv_binds.append(f[:8])
                parents.append(f[8])
                names.append(f[9].split(b'\0')[0].decode())
            if len(names) >= total or not count:
                return names, parents, inv_binds

    def close(self):
        self.sock.close()


def _transform_matrix(t):
    """pos xyz, uniform scale, quat xyzw -> 4x4."""
    x, y, z, s, qx, qy, qz, qw = t
    m = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4() @ Matrix.Diagonal((s, s, s, 1))
    m.translation = (x, y, z)
    return m


class Puppet:
    """A game model entity posed from a Blender armature through a bridge pose slot.

    Rigs share the game's bone names, and their rest heads are the game bind positions in Blender's frame. Skinning
    agrees when each game bone's world transform is G * ArmatureWorld * Pose * Rest^-1 * K * Bind (K: game model
    space to Blender, G: Blender world to game world), so no per-bone axis convention is needed. Game bones the rig
    lacks follow their nearest ancestor it has."""

    def __init__(self, arm_name, slot, entity, model, names, parents, inv_binds, arm):
        self.arm_name, self.slot, self.entity, self.model = arm_name, slot, entity, model
        self.origin = None
        bones = arm.data.bones
        self.bones = []  # (blender bone name, Rest^-1 * K * Bind) per game bone
        for i, name in enumerate(names):
            own = i
            while own >= 0 and names[own] not in bones:
                own = parents[own]
            if own < 0:
                own_name, rest = None, Matrix.Identity(4)
            else:
                own_name = names[own]
                rest = bones[own_name].matrix_local.inverted()
            bind = _transform_matrix(inv_binds[i]).inverted()
            self.bones.append((own_name, rest @ BLENDER_FROM_GAME @ bind))

    def pose(self, arm):
        """The packed pose of an evaluated armature, and a point to keep the entity at (game units)."""
        world = GAME_FROM_BLENDER @ arm.matrix_world
        pose_bones = arm.pose.bones
        cache = {}
        out = np.empty((len(self.bones), 8), np.float32)
        for i, (name, local) in enumerate(self.bones):
            moved = cache.get(name)
            if moved is None:
                moved = cache[name] = world @ (pose_bones[name].matrix if name else Matrix.Identity(4))
            m = moved @ local
            loc, rot, scale = m.decompose()
            out[i] = (*loc, (scale.x + scale.y + scale.z) / 3, rot.x, rot.y, rot.z, rot.w)
        pos = out[:, :3]
        # Near the feet, robust to stray helper bones (Abrams' muzzle_fx hangs metres below the floor in some clips).
        anchor = (float(np.median(pos[:, 0])), float(np.median(pos[:, 1])), float(np.percentile(pos[:, 2], 10)))
        return out.tobytes(), anchor


def rig_models():
    """Every hero, form and NPC model the add-on has rigs for, with their extra models. A map only loads the models
    its match uses (The Doorman isn't in Midtown's), and a model spawned without being loaded has no skeleton."""
    from . import heroes
    models = set()
    for hero in heroes():
        models.add(hero.get('model'))
        models.update(extra.get('model') for extra in (hero.get('extra_models') or {}).values())
    return sorted(m for m in models if m)


def _arm_model(arm):
    if arm.get('dlr_model'):
        return arm['dlr_model']
    from . import heroes
    slug = arm.get('dlrig')
    return next((h['model'] for h in heroes() if h['slug'] == slug), None)


def puppet_armature(obj):
    """The armature a selected object belongs to (itself, its Armature modifier's or its parent), or None."""
    if obj is None:
        return None
    if obj.type == 'ARMATURE':
        return obj
    found = obj.find_armature()
    if found:
        return found
    return obj.parent if obj.parent and obj.parent.type == 'ARMATURE' else None


def tangent_extents(proj):
    """The view's edges in tangent space (x / -z, y / -z): ((left, right), (bottom, top)). Blender's projection
    gives ndc = P00 * u - P02, so the edges are where that is -1 and 1 (handles lens shift and camera-view zoom)."""
    return (((-1 + proj[0][2]) / proj[0][0], (1 + proj[0][2]) / proj[0][0]),
            ((-1 + proj[1][2]) / proj[1][1], (1 + proj[1][2]) / proj[1][1]))


def game_view(cam_matrix, proj, game_aspect, min_vfov=MIN_VFOV):
    """The game camera that covers a Blender view: same eye and direction, a vertical FOV wide enough that the
    game's centred frame (game_aspect) covers the whole view. Roll is dropped (the bridge could take it; the
    console can't)."""
    g = GAME_FROM_BLENDER @ cam_matrix
    fx, fy, fz = (-g.col[2].xyz).normalized()
    yaw = math.degrees(math.atan2(fy, fx))
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, fz))))
    (u0, u1), (v0, v1) = tangent_extents(proj)
    need = max(abs(v0), abs(v1), abs(u0) / game_aspect, abs(u1) / game_aspect)
    vfov = min(MAX_VFOV, max(min_vfov, math.degrees(2 * math.atan(need))))
    return View(tuple(g.translation), pitch, yaw, vfov)


def blender_view(pos, pitch, yaw):
    """Blender camera matrix for a game camera (game units, degrees)."""
    to_blender = Matrix.Rotation(-math.pi / 2, 4, 'Z')
    rot = to_blender @ Euler((math.radians(90 - pitch), 0, math.radians(yaw - 90))).to_matrix().to_4x4()
    return Matrix.Translation((to_blender @ Vector(pos)) * UNIT) @ rot


HOST = '/run/host'
HOST_LOADER = HOST + '/usr/lib/ld-linux-x86-64.so.2'
HOST_LIBS = ('/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib/pulseaudio',
             '/usr/lib/x86_64-linux-gnu/pulseaudio')


def host_tool(name):
    """Command prefix for a host program, or None. Steam's Blender runs in Steam's runtime container
    (pressure-vessel), which has none of ffmpeg/xprop/xwininfo: there the host's own copy runs through the host's
    loader and libraries (mounted under /run/host)."""
    found = shutil.which(name)
    if found:
        return [found]
    if os.environ.get('container') == 'pressure-vessel' and os.path.exists(HOST_LOADER):
        for d in ('/usr/bin', '/bin', '/usr/local/bin'):
            path = HOST + d + '/' + name
            if os.path.exists(path):
                libs = ':'.join(HOST + lib for lib in HOST_LIBS if os.path.isdir(HOST + lib))
                return [HOST_LOADER, '--library-path', libs, path]
    return None


def _run_tool(name, *args, **kwargs):
    tool = host_tool(name)
    if tool is None:
        raise FileNotFoundError(name)
    return subprocess.run([*tool, *args], **kwargs)


def find_game_window():
    """X11 id of the game window (Proton runs it under XWayland), or None."""
    try:
        out = _run_tool('xprop', '-root', '_NET_CLIENT_LIST', capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for wid in out.split('#')[-1].replace(',', ' ').split():
        try:
            cls = _run_tool('xprop', '-id', wid, 'WM_CLASS', capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if 'steam_app_1422450' in cls:
            return wid
    return None


def screen_size():
    """The X desktop's size (all monitors side by side: only the height says much), or None."""
    if not os.environ.get('DISPLAY'):
        return None
    try:
        out = _run_tool('xprop', '-root', '_NET_DESKTOP_GEOMETRY', capture_output=True, text=True, timeout=2).stdout
        w, h = (int(v) for v in out.split('=')[1].split(','))
        return w, h
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def keep_window_below(wid):
    """Ask the window manager to keep the game window under every other window (EWMH _NET_WM_STATE_BELOW). The
    stream reads covered windows fine; a minimised one stops it (the game also drops to ~10 frames a second)."""
    import ctypes
    try:
        x = ctypes.CDLL('libX11.so.6')
    except OSError:
        return False
    x.XOpenDisplay.restype = ctypes.c_void_p
    x.XInternAtom.restype = ctypes.c_ulong
    x.XDefaultRootWindow.restype = ctypes.c_ulong

    class ClientMessage(ctypes.Structure):
        _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                    ('display', ctypes.c_void_p), ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong),
                    ('format', ctypes.c_int), ('data', ctypes.c_long * 5)]

    class Event(ctypes.Union):
        _fields_ = [('xclient', ClientMessage), ('pad', ctypes.c_long * 24)]

    dpy = x.XOpenDisplay(None)
    if not dpy:
        return False
    dpy = ctypes.c_void_p(dpy)
    try:
        event = Event()
        m = event.xclient
        m.type, m.send_event, m.display, m.window, m.format = 33, 1, dpy, int(wid, 16), 32  # ClientMessage
        m.message_type = x.XInternAtom(dpy, b'_NET_WM_STATE', 0)
        m.data[0], m.data[1], m.data[3] = 1, x.XInternAtom(dpy, b'_NET_WM_STATE_BELOW', 0), 1  # add, from an app
        root = ctypes.c_ulong(x.XDefaultRootWindow(dpy))
        x.XSendEvent(dpy, root, 0, (1 << 20) | (1 << 19), ctypes.byref(event))  # SubstructureRedirect|Notify
        x.XFlush(dpy)
    finally:
        x.XCloseDisplay(dpy)
    return True


def _window_size(wid):
    try:
        out = _run_tool('xwininfo', '-id', wid, capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        out = ''
    size = {}
    for line in out.splitlines():
        key, _, value = line.strip().partition(':')
        if key in ('Width', 'Height'):
            size[key] = int(value)
    return (size['Width'], size['Height']) if len(size) == 2 else None


_CAPTURE_STATUS = struct.Struct('<4sIIIIIQQQ64s')
_DEPTH_STATUS = struct.Struct('<4sIIIIIQ64s')  # magic, state, done request, width, height, format, present, message
_DEPTH_HEADER = struct.Struct('<4sIIIIIIIIIQ')  # magic, version, seq, w, h, format, stride, camera seq, step, views,
#                                                 present; texels from DEPTH_DATA
DEPTH_DATA = 4096
# The game's near plane (r_nearz default) with an infinite reversed-Z projection: stored depth = NEAR_Z / view z.
# Measured through the depth copy against the hideout floor (r_nearz 10 scaled the fit exactly).
NEAR_Z = 7.0
_FRAME_HEADER = struct.Struct('<IIIIIIIIQQQQ')  # seq, w, h, format, stride, camera seq, clock step, views, present,
                                                # data offset, capacity, time
_BGRA_FORMATS = (44, 50)  # VK_FORMAT_B8G8R8A8_UNORM / _SRGB
_RGBA_FORMATS = (37, 43)
_SRGB_TO_LINEAR = None


def srgb_to_linear(pixels):
    """uint8 sRGB values -> float32 linear, through a table (a pow over a 4K frame takes ~0.3 s)."""
    global _SRGB_TO_LINEAR
    if _SRGB_TO_LINEAR is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        _SRGB_TO_LINEAR = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return _SRGB_TO_LINEAR[pixels]


def frames_path(port=BRIDGE_PORT):
    if sys.platform == 'win32':
        return os.path.join(tempfile.gettempdir(), f'dlr-frames-{port}')
    return f'/dev/shm/dlr-frames-{port}'  # the same tmpfs in Steam's containers (game and Blender) and the host


class BridgeFrames:
    """The game's own frames: the bridge copies each presented image on the GPU into a shared file, tagged with
    the camera and clock step it shows. Exact pixels at the game's render size, no window grab, no guessed
    latency. Same interface as Stream (latest, seq, size, alive, stop); latest carries the tags as a 4th item."""
    bgr = True

    def __init__(self, port=BRIDGE_PORT):
        self.port = port
        self.path = frames_path(port)
        self.game_path = self.path if sys.platform == 'win32' else 'Z:' + self.path.replace('/', '\\')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.map = None
        self.map_size = 0
        self.size = None
        self.error = ''
        self.cached = None
        self.pinged = 0.0
        self.last_status = None
        self.lock = threading.Lock()

    def request(self, want=True):
        packet = struct.pack('<4sI256s', b'DLRF', 1 if want else 0, self.game_path.encode()[:255])
        with self.lock:
            self.sock.sendto(packet, ('127.0.0.1', self.port))
            data = self.sock.recv(4096)
        f = _CAPTURE_STATUS.unpack(data[:_CAPTURE_STATUS.size])
        self.pinged = time.time()
        self.last_status = {'state': f[1], 'size': (f[2], f[3]), 'format': f[4], 'presents': f[6],
                            'copied': f[7], 'published': f[8], 'message': f[9].split(b'\0')[0].decode()}
        return self.last_status

    def start(self, timeout=6.0):
        # The first request makes the game rebuild its swapchain (the copy needs it made after the bridge loaded).
        end = time.time() + timeout
        status = self.request()
        while not status['published'] and time.time() < end:
            if status['state'] == 3:
                break
            time.sleep(0.05)
            status = self.request()
        if not status['published']:
            raise OSError(f"no frames from the bridge ({status['message'] or 'none published'})")
        self._open()
        if self.read() is None:
            raise OSError('the bridge frame file is unreadable')

    def _open(self):
        if self.map is not None:
            self.map.close()
        with open(self.path, 'rb') as f:
            self.map_size = os.fstat(f.fileno()).st_size
            self.map = mmap.mmap(f.fileno(), self.map_size, prot=mmap.PROT_READ) if sys.platform != 'win32' \
                else mmap.mmap(f.fileno(), self.map_size, access=mmap.ACCESS_READ)

    def _header(self):
        if struct.unpack_from('<Q', self.map, 16)[0] != self.map_size:
            self._open()  # the bridge grew the file for a bigger frame
        index = struct.unpack_from('<I', self.map, 8)[0]
        if index > 1:
            return None, None
        return index, _FRAME_HEADER.unpack_from(self.map, 64 + 64 * index)

    @property
    def seq(self):
        if self.map is None:
            return 0
        if time.time() - self.pinged > 1.0:
            self._ping()
        _, header = self._header()
        return header[8] if header else 0

    def _ping(self):
        try:
            self.request()
        except OSError:
            pass

    def read(self):
        """(bytes, arrival time, present number, tags) of the newest frame, copied out consistently."""
        for _ in range(50):
            index, header = self._header()
            if header is None:
                return None
            if self.cached and self.cached[2] == header[8]:
                return self.cached
            if header[0] & 1:
                time.sleep(0.001)
                continue
            _, w, h, fmt, stride, camera, step, views, present, offset, _, _ = header
            data = self.map[offset:offset + stride * h]
            if _FRAME_HEADER.unpack_from(self.map, 64 + 64 * index)[0] != header[0]:
                continue  # overwritten while copying
            if fmt not in _BGRA_FORMATS + _RGBA_FORMATS:
                self.error = f'unsupported game frame format {fmt}'
                return None
            self.bgr = fmt in _BGRA_FORMATS
            self.size = (w, h)
            self.cached = (data, time.time(), present, {'camera': camera, 'step': step, 'views': views})
            return self.cached
        return None

    @property
    def latest(self):
        if self.map is None:
            return None
        if time.time() - self.pinged > 1.0:
            self._ping()
        return self.read()

    @property
    def alive(self):
        if time.time() - self.pinged > 1.0:
            self._ping()
        status = self.last_status
        return bool(status) and status['state'] == 1 and os.path.exists(self.path)

    def wait(self, ok, timeout=3.0):
        """The first frame whose tags satisfy ok(tags)."""
        end = time.time() + timeout
        while time.time() < end:
            frame = self.latest
            if frame is not None and ok(frame[3]):
                return frame
            time.sleep(0.002)
        return None

    def depth(self, timeout=3.0):
        """One snapshot of the scene depth (the bridge copies it where the game copies it, mid-frame): view-axis
        distance in game units (inf for sky), rows bottom-up, and its tags."""
        self.depth_request = (getattr(self, 'depth_request', int(time.time()) & 0xffffff) + 1) & 0xffffffff
        end = time.time() + timeout
        while True:
            with self.lock:
                self.sock.sendto(struct.pack('<4sI', b'DLRD', self.depth_request), ('127.0.0.1', self.port))
                data = self.sock.recv(4096)
            f = _DEPTH_STATUS.unpack(data[:_DEPTH_STATUS.size])
            if f[1] == 3:
                raise OSError('no game depth: ' + f[7].split(b'\0')[0].decode(errors='replace'))
            if f[1] == 0 and f[2] == self.depth_request:
                break
            if time.time() > end:
                raise TimeoutError('the game did not copy its depth')
            self._ping()  # keeps the bridge capturing (it stops a while after the last packet)
            time.sleep(0.01)
        with open(self.path + '-depth', 'rb') as file:
            data = file.read()
        _, _, seq, w, h, fmt, stride, camera, step, views, present = _DEPTH_HEADER.unpack_from(data)
        if seq & 1:
            raise OSError('the depth file was being written')
        raw = np.frombuffer(data, np.uint32, h * stride // 4, DEPTH_DATA).reshape(h, stride // 4)[::-1, :w]
        if fmt == 129:  # D24_UNORM_S8: depth in the low 24 bits
            stored = (raw & 0xffffff).astype(np.float32) * np.float32(1 / 0xffffff)
        else:
            stored = raw.view(np.float32)
        with np.errstate(divide='ignore'):
            z = np.where(stored > 0, np.float32(NEAR_Z) / stored, np.float32(np.inf))
        return z, {'camera': camera, 'step': step, 'views': views}

    def stop(self):
        try:
            self.request(False)
        except OSError:
            pass
        if self.map is not None:
            self.map.close()
            self.map = None
        self.sock.close()


class Stream:
    """The game window as a live stream: ffmpeg x11grab writes raw RGBX frames to a pipe, a thread keeps the
    latest. Frames stay bytes: the shader unpacks them, so no per-frame conversion on the CPU."""

    def __init__(self, wid):
        self.wid = wid
        self.size = None
        self.latest = None  # (bytes, arrival time, sequence)
        self.seq = 0
        self.bgr = False
        self.error = ''
        self.proc = None

    def start(self):
        self.size = _window_size(self.wid) or self._probe_size()
        w, h = self.size
        self.proc = subprocess.Popen(
            [*host_tool('ffmpeg'), '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-framerate', '60',
             '-window_id', self.wid, '-i', os.environ.get('DISPLAY', ':0'), '-f', 'rawvideo', '-pix_fmt', 'rgb0',
             '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, args=(w * h * 4,), daemon=True).start()

    def _probe_size(self):
        out = subprocess.run(
            [*host_tool('ffmpeg'), '-loglevel', 'error', '-f', 'x11grab', '-window_id', self.wid, '-i',
             os.environ.get('DISPLAY', ':0'), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
            capture_output=True, timeout=10)
        # x11grab has no size query; one grey frame of the window has w * h bytes and games are 16:9.
        n = len(out.stdout)
        h = int(round(math.sqrt(n * 9 / 16)))
        if not n or n % h:
            raise OSError('could not read the game window: ' + out.stderr.decode(errors='replace').strip())
        return n // h, h

    def _read(self, n):
        pipe = self.proc.stdout
        while True:
            buf = pipe.read(n)
            if len(buf) < n:
                break
            self.seq += 1
            self.latest = (buf, time.time(), self.seq)
        err = self.proc.stderr.read().decode(errors='replace').strip() if self.proc.stderr else ''
        self.error = 'game window stream ended' + (f': {err}' if err else '')

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


class Link:
    """One game connection. view_draw posts the wanted view; a worker thread places the game camera. The frame
    comes back as a live window stream when possible (X11/XWayland and ffmpeg), else as jpeg screenshots (latest
    view wins; views posted while a shot is in flight are skipped). F12 renders call capture() directly; the shot
    lock keeps them from interleaving with the viewport."""

    def __init__(self, port=PORT, setup=(), launched=False):
        self.port = port
        self.launch_setup = tuple(setup)  # console commands for a session started from here (map)
        self.launched = launched  # the add-on started this game: its window may be kept below
        self.con = None
        self.error = ''
        self.wanted = None
        self.sent = None
        self.frame = None  # (path, View it was taken with, sequence)
        self.seq = 0
        self.timing = 0.0
        self.event = threading.Event()
        self.running = False
        self.outside = False
        self.lock = threading.Lock()
        self.shot_lock = threading.Lock()
        self.cache = os.path.join(bpy.app.tempdir or '/tmp', 'dlr_game_frames')
        self.stream = None
        self.bridge = None
        self.start_camera = None
        self.pinged = 0.0
        self.history = deque(maxlen=120)  # (time sent, View), for pairing stream frames with views
        self.poses = None
        self.puppets = {}  # armature name -> Puppet
        self.declined = set()  # rigs the user took out of the game, or that failed to spawn
        self.render_frame = None  # last frame rendered while the game clock is frozen
        self.stream_restarted = 0.0
        self.time_step = 0
        self.frames_error = ''
        self.pawn_at = None
        self.pawn_moved = 0.0

    @property
    def min_vfov(self):
        return 1.0 if self.bridge else MIN_VFOV

    def allowed(self, view):
        # Only the console camera moves the hero (the game kills it outside the play area).
        return self.bridge is not None or view.inside()

    def follow(self, pos):
        """Move the hidden hero to the bridge camera when it has gone far (noclip: it floats there). Inside the play
        area only (the game kills a hero outside it). Returns how long entities still need to stream in."""
        if not (self.bridge and self.con):
            return 0.0
        target = tuple(min(max(v, lo), hi) for v, (lo, hi) in zip(pos, PLAY_AREA))
        if self.pawn_at is None or math.dist(target, self.pawn_at) > PAWN_FOLLOW:
            commands = ['setpos_exact {:.1f} {:.1f} {:.1f}'.format(*target)]
            if self.pawn_at is None:
                commands.insert(0, 'noclip 1')
            self.con.send(*commands)
            self.pawn_at = target
            self.pawn_moved = time.time()
        return self.streaming()

    def streaming(self):
        """Seconds until entities near the hero's last move have arrived."""
        return max(0.0, self.pawn_moved + PAWN_STREAM_IN - time.time())

    def follow_render(self, pos):
        """follow() for a render frame. True when the hero moved: a cut far away mid-animation thaws the game clock
        (entities only stream in while it runs, so effects jump ahead by about PAWN_STREAM_IN there)."""
        if self.follow(pos) <= 0:
            return False
        if self.render_frame is not None:
            self.release_time()
        return True

    def _place(self, view):
        if self.bridge:
            self.follow(view.pos)
            return self.bridge.place(view)
        self.con.send(*view.commands())
        return None

    def stream_view(self, arrived):
        """The view a stream frame that arrived at this time most likely shows."""
        chosen = None
        for sent, view, _ in list(self.history):
            if sent > arrived - (STREAM_LATENCY_BRIDGE if self.bridge else STREAM_LATENCY):
                break
            chosen = view
        if chosen is None and self.history:
            chosen = self.history[0][1]
        return chosen

    def tagged_view(self, camera_seq):
        """The view a bridge frame tagged with this camera sequence number shows."""
        chosen = None
        for _, view, seq in list(self.history):
            if seq is None or seq > camera_seq:
                break
            chosen = view
        if chosen is None and self.history:
            chosen = self.history[0][1]
        return chosen

    def game_camera(self):
        """Where the game camera is now: ((x, y, z), pitch, yaw) in game units and degrees, or None."""
        if self.bridge:
            try:
                status = self.bridge.status()
            except OSError:
                return None
            return tuple(status['origin']), status['angles'][0], status['angles'][1]
        line = self.con.request('getpos', 'setpos') if self.con else None
        if not line:
            return None
        try:
            pos_part, ang_part = line.split(';')[:2]
            pos = tuple(float(v) for v in pos_part.split()[1:4])
            pitch, yaw = (float(v) for v in ang_part.split()[1:3])
        except ValueError:
            return None
        return pos, pitch, yaw

    def start(self):
        if self.running:
            return
        os.makedirs(self.cache, exist_ok=True)
        self.con = Netcon(self.port)
        if self.launch_setup:
            self.con.send(*self.launch_setup)
            # The hero spawns before the camera and the stream start.
            time.sleep(2.0)
        bridge = Bridge()
        try:
            if bridge.status()['state'] == 1:
                self.bridge = bridge
        except OSError:
            pass
        if self.bridge is None:
            bridge.close()
        else:
            self.poses = PoseChannel()
        # Read the game's own camera before taking it over (Connect jumps the viewports there).
        self.start_camera = self.game_camera()
        self.con.send(*SETUP_COMMON)
        if self.poses:
            self.remove_leftovers()
            self.hide_player()
        if self.bridge is None:
            self.con.send(*SETUP)
        if self.bridge:
            frames = BridgeFrames()
            try:
                frames.start()
                self.stream = frames
            except OSError as error:
                frames.stop()
                self.frames_error = str(error)
        wid = None
        if self.stream:
            wid = find_game_window() if os.environ.get('DISPLAY') and host_tool('xprop') else None
            if wid and self.launched:
                keep_window_below(wid)
            wid = None
        elif not os.environ.get('DISPLAY'):
            self.error = 'screenshot mode: no X display for the window stream'
        elif host_tool('ffmpeg') is None:
            self.error = 'screenshot mode: ffmpeg not found (install it for the live stream)'
        else:
            wid = find_game_window()
            if not wid:
                self.error = 'screenshot mode: game window not found (needs xprop)'
        if wid:
            stream = Stream(wid)
            try:
                stream.start()
                self.stream = stream
                if self.launched:
                    keep_window_below(wid)
            except (OSError, subprocess.SubprocessError) as error:
                self.error = f'screenshot mode: {error}'
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        self.event.set()
        if self.stream:
            self.stream.stop()
            self.stream = None
        had_puppets = bool(self.puppets)
        for puppet in list(self.puppets.values()):
            self.remove_puppet(puppet)
        if self.poses:
            try:
                self.poses.pose(PAWN_SLOT, '', active=False)
            except OSError:
                pass
        self.release_time()
        if had_puppets and self.con:
            # Closing the console right after sending can drop the ent_remove commands.
            time.sleep(0.3)
        if self.poses:
            self.poses.close()
            self.poses = None
        if self.bridge:
            self.bridge.release()
            self.bridge.close()
            self.bridge = None
        if self.con:
            self.con.close()
            self.con = None

    def add_puppet(self, arm):
        """Spawn the armature's game model and bind it to a free pose slot."""
        if not self.poses:
            raise ConnectionError('posing needs the bridge (start the session with --bridge)')
        model = _arm_model(arm)
        if not model:
            raise ValueError(f'{arm.name} is not a Deadlock rig (no game model known for it)')
        used = {p.slot for p in self.puppets.values()} | {PAWN_SLOT}
        slot = next((i for i in range(POSE_SLOTS) if i not in used), None)
        if slot is None:
            raise ValueError(f'all {POSE_SLOTS} pose slots are in use')
        before = self.poses.entities(model)
        # prop_dynamic_create prepends models/ itself.
        self.con.send(f'prop_dynamic_create {model[len("models/"):] if model.startswith("models/") else model}')
        bound = None
        end = time.time() + 5.0
        while bound is None and time.time() < end:
            time.sleep(0.1)
            new = self.poses.entities(model) - before
            if new:
                bound = self.poses.pose(slot, model, entity=max(new), active=False)
        if bound is None:
            raise TimeoutError(f'the game did not spawn {model}')
        if not bound['count']:
            self.con.send(f'ent_remove {bound["entity"]}')
            raise ValueError('the map did not load this model: quit Deadlock and start it again from Blender '
                             '(Start Deadlock loads every rig\'s model)')
        names, parents, inv_binds = self.poses.skeleton(slot)
        puppet = Puppet(arm.name, slot, bound['entity'], model, names, parents, inv_binds, arm)
        self.puppets[arm.name] = puppet
        self._save_spawned()
        return puppet

    def hide_player(self):
        """Fold the player's own hero (the session needs one, and it stands where the shots are) into a point: every
        bone of its pose at scale 0, mesh and shadow gone. The game shows it again 3 s after Blender goes quiet."""
        line = self.con.request('ent_find player', "'player'", timeout=2.0)
        try:
            entity = int(line.rsplit('entindex', 1)[1].strip(' )'))
        except (AttributeError, IndexError, ValueError):
            return False
        bound = self.poses.pose(PAWN_SLOT, '', entity=entity, active=False)
        if bound is None or not bound['count']:
            return False
        folded = struct.pack('<3ff4f', *bound['origin'], 0.0, 0.0, 0.0, 0.0, 1.0) * bound['count']
        return self.poses.pose(PAWN_SLOT, '', folded, bound['count'], entity=entity) is not None

    def _save_spawned(self):
        try:
            os.makedirs(os.path.dirname(SPAWNED), exist_ok=True)
            with open(SPAWNED, 'w') as f:
                json.dump([[p.entity, p.model] for p in self.puppets.values()], f)
        except OSError:
            pass

    def remove_leftovers(self):
        """Remove models an earlier Blender spawned and never removed (it crashed or quit while connected). Only
        entities that still have that model: the game reuses entity indices."""
        try:
            with open(SPAWNED) as f:
                spawned = json.load(f)
        except (OSError, ValueError):
            return
        for entity, model in spawned:
            try:
                if entity in self.poses.entities(model):
                    self.con.send(f'ent_remove {entity}')
            except OSError:
                pass
        self._save_spawned()

    def sync_puppets(self, depsgraph):
        """Every visible Deadlock rig in the scene is posed in the game (rigs taken out stay out)."""
        if not (self.running and self.poses):
            return
        for obj in depsgraph.objects:
            arm = obj.original
            if (obj.type != 'ARMATURE' or not arm.get('dlrig') or arm.name in self.puppets
                    or arm.name in self.declined or not arm.visible_get()):
                continue
            try:
                self.add_puppet(arm)
            except (OSError, ValueError, TimeoutError) as error:
                self.declined.add(arm.name)
                self.error = f'{arm.name}: {error}'
        if self.puppets:
            self.send_poses(depsgraph)

    def remove_puppet(self, puppet):
        self.puppets.pop(puppet.arm_name, None)
        try:
            self.poses.pose(puppet.slot, puppet.model, entity=puppet.entity, active=False)
        except OSError:
            pass
        if self.con and self.con.alive:
            self.con.send(f'ent_remove {puppet.entity}')
        self._save_spawned()

    def send_poses(self, depsgraph, settle=0.0):
        """Send every puppet's current pose (evaluated armatures of this depsgraph). With settle, wait up to that long
        until every moved entity has arrived (ent_setpos goes through the server; until then the pose may be culled)."""
        end = time.time() + settle
        while True:
            if self._send_poses(depsgraph) or time.time() > end:
                return
            time.sleep(0.05)

    def _send_poses(self, depsgraph):
        arrived = True
        for puppet in list(self.puppets.values()):
            arm = bpy.data.objects.get(puppet.arm_name)
            if arm is None or arm.type != 'ARMATURE':
                self.remove_puppet(puppet)
                continue
            data, anchor = puppet.pose(arm.evaluated_get(depsgraph))
            # The entity's bounds decide culling: keep the entity with the pose.
            if puppet.origin is None or math.dist(anchor, puppet.origin) > FOLLOW_DISTANCE:
                self.con.send('ent_setpos {} {:.2f} {:.2f} {:.2f}'.format(puppet.entity, *anchor))
                puppet.origin = anchor
            try:
                bound = self.poses.pose(puppet.slot, puppet.model, data, len(puppet.bones), entity=puppet.entity)
            except OSError as error:
                self.error = f'pose: {error}'
                continue
            if bound is None or math.dist(bound['origin'], puppet.origin) > ARRIVED:
                arrived = False
        return arrived

    def reshoot(self):
        with self.lock:
            self.sent = None
        self.event.set()

    def want(self, view):
        with self.lock:
            if view.close_to(self.wanted):
                return
            self.wanted = view
        self.event.set()

    def capture_stream(self, view):
        """Place the game camera and take the next stream frame that shows it: linear float RGBA, rows bottom-up."""
        with self.shot_lock:
            time.sleep(self.follow(view.pos))
            seq = self.bridge.place(view)
            if isinstance(self.stream, BridgeFrames):
                # Tagged frames: the first one drawn with this camera (and the clock step, while rendering).
                step = self.time_step if self.render_frame is not None else None
                latest = self.stream.wait(
                    lambda tags: tags['camera'] == seq and (step is None or tags['step'] >= step))
                if latest is None:
                    raise TimeoutError('no game frame with the camera from the bridge')
            else:
                if not self.bridge.wait_applied(seq):
                    raise TimeoutError('the game did not take the camera')
                # The camera is drawn already: wait for x11grab's next full frame after now.
                ready = time.time() + 0.04
                end = ready + 2.0
                while time.time() < end:
                    latest = self.stream.latest
                    if latest is not None and latest[1] >= ready:
                        break
                    time.sleep(0.005)
                else:
                    raise TimeoutError('no frame from the game window stream')
            self.sent = view
            w, h = self.stream.size
            bgr = self.stream.bgr
        px = np.frombuffer(latest[0], np.uint8).reshape(h, w, 4)[::-1]
        rgb = np.empty((h, w, 4), np.float32)
        rgb[..., :3] = srgb_to_linear(px[..., 2::-1] if bgr else px[..., :3])
        rgb[..., 3] = 1.0
        return rgb

    def advance(self, frame, fps):
        """Game clock for a render frame: frozen while rendering, moved on by exactly the time between this and the
        last rendered frame, so effects play at the animation's speed however long each capture takes."""
        if not self.poses:
            return
        if self.render_frame is None or frame <= self.render_frame:
            state = self.poses.time(True, self.time_step)
        else:
            self.time_step += 1
            state = self.poses.time(True, self.time_step, (frame - self.render_frame) / fps)
            end = time.time() + 2.0
            while state['step_done'] != self.time_step and time.time() < end:
                time.sleep(0.005)
                state = self.poses.time(True, self.time_step)
        self.render_frame = frame
        if not state['available']:
            self.error = 'game clock not available (bridge): effects run in real time during renders'

    def release_time(self):
        self.render_frame = None
        if self.poses:
            try:
                self.poses.time(False, self.time_step)
            except OSError:
                pass

    def capture(self, view, kind='jpeg_screenshot', settle=SETTLE):
        """Place the game camera and take a screenshot; returns the file path (the caller moves or deletes it)."""
        if not (self.con and self.con.alive):
            raise ConnectionError('lost the game console')
        # The console camera is the player's own hero: past the play area the game kills it (and with god mode on
        # it gets stuck there, flooding the console).
        if not self.allowed(view):
            raise ValueError('the camera is outside the game\'s play area')
        ext = '.png' if kind == 'png_screenshot' else '.jpg'
        folder = os.path.join(game_dir() or '', 'citadel', 'screenshots')
        with self.shot_lock:
            time.sleep(self.follow(view.pos))
            seq = self._place(view)
            if seq is None or not self.bridge.wait_applied(seq):
                time.sleep(settle)
            before = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            self.con.send(kind)
            end = time.time() + 10.0
            while time.time() < end:
                time.sleep(0.01)
                try:
                    new = [n for n in os.listdir(folder) if n.endswith(ext) and n not in before]
                except FileNotFoundError:
                    continue
                if new:
                    path = os.path.join(folder, new[0])
                    size = -1
                    # The engine writes asynchronously: wait until the size settles.
                    while size != os.path.getsize(path):
                        size = os.path.getsize(path)
                        time.sleep(0.02)
                    self.sent = view
                    return path
        raise TimeoutError('no screenshot appeared (is the game window minimised?)')

    def _run(self):
        if self.stream:
            self._run_stream()
        else:
            self._run_shots()

    def _keepalive(self):
        # The bridge gives the camera back to the game after 3 s without packets (Blender closed or crashed).
        if self.bridge and time.time() - self.pinged > 1.0:
            self.pinged = time.time()
            try:
                self.bridge.status()
            except OSError:
                pass

    def _restart_stream(self):
        """ffmpeg ends when the window is unmapped or resized (minimised, fullscreen toggle): grab it again."""
        if time.time() - self.stream_restarted < 1.0:
            return
        self.stream_restarted = time.time()
        old = self.stream
        if isinstance(old, BridgeFrames):
            old.stop()
            stream = BridgeFrames()
            try:
                stream.start()
            except OSError as error:
                stream.stop()
                self.error = f'bridge frames stopped; restart failed: {error}'
                return
            self.stream = stream
            return
        wid = find_game_window() or old.wid
        stream = Stream(wid)
        stream.seq = old.seq  # the viewport redraws on a new sequence number
        try:
            stream.start()
        except (OSError, subprocess.SubprocessError) as error:
            self.error = f'{old.error or "game window stream ended"}; restart failed: {error}'
            return
        self.stream = stream

    def _run_stream(self):
        while self.running:
            self.event.wait(0.5)
            self.event.clear()
            self._keepalive()
            if self.stream and not self.stream.alive:
                self._restart_stream()
            with self.lock:
                view = self.wanted
            if view is None or view.close_to(self.sent):
                continue
            self.outside = not self.allowed(view)
            if self.outside:
                continue
            if not (self.con and self.con.alive):
                self.error = 'lost the game console'
                self.running = False
                break
            try:
                with self.shot_lock:
                    seq = self._place(view)
            except OSError as error:
                self.error = str(error)
                continue
            self.history.append((time.time(), view, seq))
            self.sent = view
            time.sleep(SEND_INTERVAL)

    def _run_shots(self):
        while self.running:
            self.event.wait(0.5)
            self.event.clear()
            self._keepalive()
            with self.lock:
                view = self.wanted
            if view is None or view.close_to(self.sent):
                continue
            self.outside = not self.allowed(view)
            if self.outside:
                continue
            start = time.time()
            try:
                path = self.capture(view)
            except ConnectionError as error:
                self.error = str(error)
                self.running = False
                break
            except (OSError, ValueError, TimeoutError) as error:
                self.error = str(error)
                continue
            self.seq += 1
            # Two alternating files: Blender may still be reading the previous one.
            dest = os.path.join(self.cache, f'frame{self.seq % 2}.jpg')
            shutil.move(path, dest)
            self.frame = (dest, view, self.seq)
            self.timing = time.time() - start
            self.error = ''


_link = None
_shown = None  # (View, sequence) of the frame loaded into the image; set by the timer, read by view_draw
_stream_tex = None  # (stream sequence, GPUTexture, View) of the stream frame on the GPU
_stream_drawn = 0


def link():
    return _link


def _image(path):
    img = bpy.data.images.get(FRAME)
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = FRAME
    else:
        img.filepath = path
        img.reload()
    return img


_solid_shader = None


def solid_shader():
    """Headlit solid shading for the scene's own meshes, drawn over the game frame."""
    global _solid_shader
    if _solid_shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4', 'mvp')
        info.push_constant('MAT4', 'model')
        info.push_constant('VEC4', 'color')
        info.push_constant('VEC3', 'light')
        info.vertex_in(0, 'VEC3', 'pos')
        info.vertex_in(1, 'VEC3', 'nor')
        iface = gpu.types.GPUStageInterfaceInfo('dlr_solid_iface')
        iface.smooth('VEC3', 'world_nor')
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', 'frag')
        info.vertex_source('void main() { world_nor = mat3(model) * nor; gl_Position = mvp * vec4(pos, 1.0); }')
        info.fragment_source('void main() { float d = abs(dot(normalize(world_nor), light));'
                             ' frag = vec4(color.rgb * (0.2 + 0.8 * d), color.a); }')
        _solid_shader = gpu.shader.create_from_info(info)
    return _solid_shader


_packed_shaders = {}


def packed_shader(bgr=False):
    """Draws a stream frame uploaded as raw bytes: each R32F texel holds one RGBX pixel's four bytes. Unpacks
    them, flips rows (X11 is top-down) and decodes sRGB to linear, like an sRGB image texture would."""
    if bgr not in _packed_shaders:
        info = gpu.types.GPUShaderCreateInfo()
        info.sampler(0, 'FLOAT_2D', 'image')
        info.vertex_in(0, 'VEC2', 'pos')
        info.vertex_in(1, 'VEC2', 'texCoord')
        iface = gpu.types.GPUStageInterfaceInfo('dlr_packed_iface')
        iface.smooth('VEC2', 'uv')
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', 'frag')
        info.vertex_source('void main() { uv = texCoord; gl_Position = vec4(pos, 0.0, 1.0); }')
        info.fragment_source(
            'vec3 texel(ivec2 p) {'
            '  uint b = floatBitsToUint(texelFetch(image, p, 0).r);'
            '  vec3 c = vec3(float(b & 255u), float((b >> 8u) & 255u), float((b >> 16u) & 255u)) / 255.0;'
            + ('  c = c.bgr;' if bgr else '') +
            '  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));'
            '}'
            'void main() {'
            '  ivec2 size = textureSize(image, 0);'
            '  vec2 p = vec2(uv.x, 1.0 - uv.y) * vec2(size) - 0.5;'
            '  ivec2 i = ivec2(floor(p));'
            '  vec2 f = p - vec2(i);'
            '  ivec2 hi = size - 1;'
            '  vec3 a = texel(clamp(i, ivec2(0), hi)), b = texel(clamp(i + ivec2(1, 0), ivec2(0), hi));'
            '  vec3 c = texel(clamp(i + ivec2(0, 1), ivec2(0), hi)), d = texel(clamp(i + ivec2(1), ivec2(0), hi));'
            '  frag = vec4(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), 1.0);'
            '}')
        _packed_shaders[bgr] = gpu.shader.create_from_info(info)
    return _packed_shaders[bgr]


def _stream_texture(lk, region_width):
    """The latest stream frame as a texture, uploaded once per new frame. Bilinear is done in the shader: the
    texels are packed bytes, hardware filtering would blend their bit patterns. A frame twice the region's width
    goes up as every other pixel (the upload is the cost: ~16 ms for a 4K frame)."""
    global _stream_tex
    latest = lk.stream.latest
    if latest is None:
        return None
    buf, arrived, seq = latest[:3]
    if _stream_tex is None or _stream_tex[0] != seq:
        w, h = lk.stream.size
        texels = np.frombuffer(buf, np.float32)
        if w >= 2 * region_width:
            texels = np.ascontiguousarray(texels.reshape(h, w)[::2, ::2])
            h, w = texels.shape
        data = gpu.types.Buffer('FLOAT', w * h, texels)
        view = lk.tagged_view(latest[3]['camera']) if len(latest) > 3 else lk.stream_view(arrived)
        _stream_tex = (seq, gpu.types.GPUTexture((w, h), format='R32F', data=data), view, lk.stream.bgr)
    return _stream_tex


def _is_map_object(obj):
    """Objects of a built map: the game already draws them."""
    data = obj.data
    return (obj.get('dl_model') is not None or (data is not None and data.get('dl_model') is not None)
            or obj.name.startswith('cell_'))


def _object_color(obj):
    for slot in obj.material_slots:
        if slot.material:
            return tuple(slot.material.diffuse_color)
    return (0.8, 0.8, 0.8, 1.0)


class DeadlockGameEngine(bpy.types.RenderEngine):
    bl_idname = ENGINE_ID
    bl_label = 'Deadlock (game)'
    bl_use_preview = False
    bl_use_shading_nodes_custom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = {}  # object name -> (batch, matrix_world, colour)

    def view_update(self, context, depsgraph):
        """Rebuild the solid batches of meshes whose geometry changed; transforms only update the matrix."""
        shader = solid_shader()
        rebuild = {u.id.original.name for u in depsgraph.updates
                   if isinstance(u.id, bpy.types.Object) and u.is_updated_geometry}
        lk = _link
        if lk:
            lk.sync_puppets(depsgraph)
        puppets = set(lk.puppets) if lk else set()
        if lk and puppets and not lk.stream and any(
                isinstance(u.id, bpy.types.Object) and u.id.original.name in puppets for u in depsgraph.updates):
            # Screenshot mode only shoots when the view moves: a new pose needs a new shot too.
            lk.reshoot()
        seen = set()
        for inst in depsgraph.object_instances:
            obj = inst.object
            if inst.is_instance or obj.type not in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
                continue
            if _is_map_object(obj.original):
                continue
            # The game draws posed rigs itself.
            owner = puppet_armature(obj.original)
            if owner is not None and owner.name in puppets:
                continue
            name = obj.original.name
            seen.add(name)
            old = self.batches.get(name)
            if old is None or name in rebuild:
                mesh = obj.to_mesh()
                mesh.calc_loop_triangles()
                n = len(mesh.loop_triangles)
                if not n:
                    obj.to_mesh_clear()
                    continue
                co = np.empty(len(mesh.vertices) * 3, np.float32)
                mesh.vertices.foreach_get('co', co)
                tri = np.empty(n * 3, np.int32)
                mesh.loop_triangles.foreach_get('vertices', tri)
                normals = np.empty(n * 9, np.float32)
                mesh.loop_triangles.foreach_get('split_normals', normals)
                pos = co.reshape(-1, 3)[tri]
                batch = batch_for_shader(shader, 'TRIS', {'pos': pos, 'nor': normals.reshape(-1, 3)})
                obj.to_mesh_clear()
            else:
                batch = old[0]
            self.batches[name] = (batch, obj.matrix_world.copy(), _object_color(obj.original))
        for name in set(self.batches) - seen:
            del self.batches[name]

    def view_draw(self, context, depsgraph):
        region, rv3d = context.region, context.region_data
        lk = _link
        if not rv3d.is_perspective:
            _draw_message(region, 'Deadlock (game) needs a perspective view')
            return
        if lk is None or not lk.running:
            _draw_message(region, 'Not connected: Render Properties > Deadlock Game > Connect')
        elif lk.stream:
            w, h = lk.stream.size
            lk.want(game_view(rv3d.view_matrix.inverted(), rv3d.window_matrix, w / h, lk.min_vfov))
            frame = _stream_tex if _stream_tex is not None else _stream_texture(lk, region.width)
            if lk.outside:
                _draw_message(region, 'Outside the game\'s play area: the game camera stays put '
                                      '(View > Jump to Game View to go back)')
            elif frame is not None and frame[2] is not None:
                self._draw_frame(rv3d.window_matrix, frame[1], w / h, frame[2], packed_shader(frame[3]))
        else:
            img = bpy.data.images.get(FRAME)
            game_aspect = img.size[0] / img.size[1] if img and img.size[1] else 16 / 9
            lk.want(game_view(rv3d.view_matrix.inverted(), rv3d.window_matrix, game_aspect, lk.min_vfov))
            if lk.outside:
                _draw_message(region, 'Outside the game\'s play area: the game camera stays put '
                                      '(View > Jump to Game View to go back)')
            elif _shown is not None and img is not None:
                self._draw_frame(rv3d.window_matrix, gpu.texture.from_image(img), game_aspect, _shown[0],
                                 gpu.shader.from_builtin('IMAGE'))
        self._draw_objects(rv3d)

    def _draw_frame(self, wm, texture, game_aspect, shot, shader):
        # The frame's corners in tangent space, through the region's projection (keeps camera-view shift/zoom).
        tv = math.tan(math.radians(shot.vfov) / 2)
        th = tv * game_aspect
        x0, x1 = -wm[0][0] * th - wm[0][2], wm[0][0] * th - wm[0][2]
        y0, y1 = -wm[1][1] * tv - wm[1][2], wm[1][1] * tv - wm[1][2]
        batch = batch_for_shader(shader, 'TRI_FAN', {
            'pos': ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            'texCoord': ((0, 0), (1, 0), (1, 1), (0, 1)),
        })
        with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
            gpu.matrix.load_identity()
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            shader.uniform_sampler('image', texture)
            batch.draw(shader)

    def _draw_objects(self, rv3d):
        if not self.batches:
            return
        shader = solid_shader()
        # Headlight: light along the view direction, so every face toward the camera is lit.
        light = (rv3d.view_matrix.inverted().to_3x3() @ Vector((0, 0, 1))).normalized()
        persp = rv3d.perspective_matrix
        framebuffer = gpu.state.active_framebuffer_get()
        framebuffer.clear(depth=1.0)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('NONE')
        shader.bind()
        shader.uniform_float('light', light)
        for batch, matrix, color in self.batches.values():
            shader.uniform_float('mvp', persp @ matrix)
            shader.uniform_float('model', matrix)
            shader.uniform_float('color', color)
            batch.draw(shader)
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('NONE')

    def render(self, depsgraph):
        scene = depsgraph.scene
        render = scene.render
        scale = render.resolution_percentage / 100
        width, height = int(render.resolution_x * scale), int(render.resolution_y * scale)
        result = self.begin_result(0, 0, width, height)
        try:
            pixels = self._render_pixels(depsgraph, scene, width, height)
        except Exception as error:
            self.report({'ERROR'}, f'Deadlock (game): {error}')
            pixels = np.zeros((width * height, 4), np.float32)
        # Assigning the array converts it pixel by pixel (~1 s per HD frame); foreach_set copies the buffer.
        result.layers[0].passes['Combined'].rect.foreach_set(np.ascontiguousarray(pixels, np.float32).ravel())
        self.end_result(result)

    def _render_pixels(self, depsgraph, scene, width, height):
        lk = _link
        if lk is None or not lk.running:
            raise ConnectionError('not connected to the game (Render Properties > Deadlock Game > Connect)')
        if scene.camera is None:
            raise ValueError('the scene has no camera')
        # The original camera isn't moved to the frame in animation renders (own depsgraph): use the evaluated one.
        camera = scene.camera.evaluated_get(depsgraph)
        render = scene.render
        proj = camera.calc_matrix_camera(depsgraph, x=width, y=height, scale_x=render.pixel_aspect_x,
                                         scale_y=render.pixel_aspect_y)
        w, h = lk.bridge.size if lk.bridge else (16, 9)
        view = game_view(camera.matrix_world, proj, w / h, lk.min_vfov)
        # Before the puppets: their entities only reach the client near the hero.
        frame_number = scene.frame_current + scene.frame_subframe
        first = lk.render_frame is None
        moved = lk.follow_render(view.pos)
        lk.sync_puppets(depsgraph)
        if lk.puppets:
            # Entities move through server ticks: wait before the clock stops; later frames move during the step.
            lk.send_poses(depsgraph, settle=2.0 if first or moved else 0.0)
        time.sleep(lk.streaming())
        lk.advance(frame_number, render.fps / render.fps_base)
        if lk.puppets and not first:
            lk.send_poses(depsgraph, settle=1.0)
        if lk.stream and lk.bridge:
            frame = lk.capture_stream(view)
        else:
            path = lk.capture(view, 'png_screenshot', settle=0.25)
            try:
                frame = _load_linear(path)
            finally:
                os.remove(path)
        return _resample(frame, view, proj, width, height)


def _load_linear(path):
    """A screenshot as linear float RGBA, rows bottom-up (Blender's order)."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        buf = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(buf)
    finally:
        bpy.data.images.remove(img)
    rgb = buf.reshape(h, w, 4)
    # Byte images keep their sRGB encoding in .pixels; the render result wants linear.
    c = rgb[..., :3]
    rgb[..., :3] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    rgb[..., 3] = 1.0
    return rgb


def _resample(frame, view, proj, width, height):
    """The scene camera's view out of a centred game frame, bilinear."""
    fh, fw = frame.shape[:2]
    tv = math.tan(math.radians(view.vfov) / 2)
    th = tv * fw / fh
    (u0, u1), (v0, v1) = tangent_extents(proj)
    # Game pixels per output pixel: from 2 up, average blocks first (supersampling; bilinear alone would skip
    # most of the game's pixels and alias).
    k = int(min(fw * (u1 - u0) / (2 * th) / width, fh * (v1 - v0) / (2 * tv) / height))
    if k == 2:
        fh, fw = fh // 2, fw // 2
        f = frame[:fh * 2, :fw * 2]
        frame = (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) * np.float32(0.25)
    elif k > 2:
        fh, fw = fh // k, fw // k
        frame = frame[:fh * k, :fw * k].reshape(fh, k, fw, k, 4).mean(axis=(1, 3), dtype=np.float32)
    u = u0 + (np.arange(width) + 0.5) / width * (u1 - u0)
    v = v0 + (np.arange(height) + 0.5) / height * (v1 - v0)
    x = (u + th) / (2 * th) * fw - 0.5
    y = (v + tv) / (2 * tv) * fh - 0.5
    if (fw, fh) == (width, height) and np.abs(x - np.arange(width)).max() < 0.01 and \
            np.abs(y - np.arange(height)).max() < 0.01:
        return frame.reshape(-1, 4)  # the game drew exactly this view: no resampling blur
    x = np.clip(x, 0, fw - 1.001)
    y = np.clip(y, 0, fh - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = (x - x0).astype(np.float32)[None, :, None], (y - y0).astype(np.float32)[:, None, None]
    rows = np.unique(np.concatenate([y0, y0 + 1]))
    # Separable: the needed rows across, then down.
    across = frame[rows][:, x0] * (1 - fx) + frame[rows][:, x0 + 1] * fx
    at = np.searchsorted(rows, y0)
    return (across[at] * (1 - fy) + across[at + 1] * fy).reshape(-1, 4)


def _resample_nearest(values, view, proj, width, height):
    """The scene camera's view out of a centred game image, nearest texel (depth must not blend across edges)."""
    fh, fw = values.shape[:2]
    tv = math.tan(math.radians(view.vfov) / 2)
    th = tv * fw / fh
    (u0, u1), (v0, v1) = tangent_extents(proj)
    u = u0 + (np.arange(width) + 0.5) / width * (u1 - u0)
    v = v0 + (np.arange(height) + 0.5) / height * (v1 - v0)
    x = np.clip(((u + th) / (2 * th) * fw).astype(np.int32), 0, fw - 1)
    y = np.clip(((v + tv) / (2 * tv) * fh).astype(np.int32), 0, fh - 1)
    return values[y][:, x]


def _draw_message(region, text):
    blf.size(0, 16)
    width, _ = blf.dimensions(0, text)
    blf.position(0, max(10, (region.width - width) / 2), region.height / 2, 0)
    blf.color(0, 1.0, 0.75, 0.3, 1.0)
    blf.draw(0, text)


_outside_shown = False


def camera_frame(context, width, height):
    """The game's view of the scene camera as linear float RGBA (rows bottom-up), width x height."""
    lk = _link
    if lk is None or not lk.running or not (lk.stream and lk.bridge):
        raise ConnectionError('not connected to a game session with the camera bridge')
    scene = context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError('the scene has no camera')
    depsgraph = context.evaluated_depsgraph_get()
    proj = camera.calc_matrix_camera(depsgraph, x=width, y=height, scale_x=scene.render.pixel_aspect_x,
                                     scale_y=scene.render.pixel_aspect_y)
    w, h = lk.bridge.size
    view = game_view(camera.matrix_world, proj, w / h, lk.min_vfov)
    lk.capture_stream(view)
    # Again once the game has drawn the new view a while: streamed textures, temporal AA and fog settle.
    time.sleep(0.5)
    return _resample(lk.capture_stream(view), view, proj, width, height)


def camera_plate(scene, depsgraph, width, height, settle=0.5):
    """The game's view of the scene camera as a plate for compositing: linear float RGBA (rows bottom-up) and the
    view-axis depth in Blender units (inf for sky), both width x height."""
    lk = _link
    if lk is None or not lk.running or not isinstance(lk.stream, BridgeFrames):
        raise ConnectionError('needs a game session with the bridge\'s frame capture')
    if scene.camera is None:
        raise ValueError('the scene has no camera')
    camera = scene.camera.evaluated_get(depsgraph)
    proj = camera.calc_matrix_camera(depsgraph, x=width, y=height, scale_x=scene.render.pixel_aspect_x,
                                     scale_y=scene.render.pixel_aspect_y)
    w, h = lk.bridge.size
    view = game_view(camera.matrix_world, proj, w / h, lk.min_vfov)
    lk.capture_stream(view)
    time.sleep(settle)  # streamed textures, temporal AA and fog settle
    for _ in range(3):
        rgb = lk.capture_stream(view)
        z, tags = lk.stream.depth()
        if tags['camera'] == lk.bridge.seq:
            break
    else:
        raise TimeoutError('the game depth never showed this camera')
    return (_resample(rgb, view, proj, width, height),
            _resample_nearest(z, view, proj, width, height) * np.float32(UNIT))


PLATE_IMAGE = 'Deadlock Game Plate'
PLATE_DEPTH = 'Deadlock Game Depth'
PLATE_GROUP = 'Deadlock Game Plate'


def _plate_image(name, pixels, width, height, non_color):
    img = bpy.data.images.get(name)
    if img is not None and (tuple(img.size) != (width, height) or not img.is_float):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color' if non_color else 'Linear Rec.709'
    img.pixels.foreach_set(pixels.ravel())
    img.update()
    img.pack()  # survives a save; the plate is one grab, not a file
    return img


def plate_compositor(scene):
    """Compositor: the render over the game plate by depth (Z Combine), so the game's walls and props hide what
    is behind them. The render needs its Z pass and a transparent film."""
    tree = bpy.data.node_groups.get(PLATE_GROUP)
    if tree is None:
        tree = bpy.data.node_groups.new(PLATE_GROUP, 'CompositorNodeTree')
    scene.render.film_transparent = True
    for layer in scene.view_layers:
        layer.use_pass_z = True  # before the Render Layers node: its Depth output only exists with the pass
    tree.nodes.clear()
    if not tree.interface.items_tree:
        tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.scene = scene
    plate = tree.nodes.new('CompositorNodeImage')
    plate.image = bpy.data.images[PLATE_IMAGE]
    depth = tree.nodes.new('CompositorNodeImage')
    depth.image = bpy.data.images[PLATE_DEPTH]
    split = tree.nodes.new('CompositorNodeSeparateColor')
    combine = tree.nodes.new('CompositorNodeZcombine')
    combine.inputs['Use Alpha'].default_value = True
    out = tree.nodes.new('NodeGroupOutput')
    for i, node in enumerate((layers, plate, depth, split, combine, out)):
        node.location = (300 * (i // 3 + (node is combine) + 2 * (node is out)), -300 * (i % 3))
    links = tree.links
    links.new(layers.outputs['Image'], combine.inputs['A'])
    links.new(layers.outputs['Depth'], combine.inputs['Depth A'])
    links.new(plate.outputs['Image'], combine.inputs['B'])
    links.new(depth.outputs['Image'], split.inputs['Image'])
    links.new(split.outputs['Red'], combine.inputs['Depth B'])
    links.new(combine.outputs['Result'], out.inputs[0])
    scene.compositing_node_group = tree


class DLR_OT_game_plate(bpy.types.Operator):
    bl_idname = 'dlr.game_plate'
    bl_label = 'Game Plate for Camera'
    bl_description = ('Grab the game\'s view of the scene camera with its depth and composite renders over it '
                      '(the game\'s walls and props hide what is behind them). Turns on Transparent film and the Z '
                      'pass; the plate matches the game best with the Standard view transform')
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _link is not None and _link.running and context.scene.camera is not None

    def execute(self, context):
        try:
            w, h = write_plate(context.scene, context.evaluated_depsgraph_get())
        except (OSError, ValueError, TimeoutError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        plate_compositor(context.scene)
        self.report({'INFO'}, f'game plate {w}x{h} with depth; compositor set to "{PLATE_GROUP}"')
        return {'FINISHED'}


def write_plate(scene, depsgraph, settle=0.5):
    """The game plate and its depth for the scene camera into the plate images, at the render size."""
    r = scene.render
    w, h = r.resolution_x * r.resolution_percentage // 100, r.resolution_y * r.resolution_percentage // 100
    rgb, z = camera_plate(scene, depsgraph, w, h, settle)
    depth = np.empty((h, w, 4), np.float32)
    # Blender's own background depth: far past anything, but finite (inf breaks the comparison's anti-aliasing)
    depth[..., :3] = np.minimum(z.reshape(h, w), np.float32(1e10))[..., None]
    depth[..., 3] = 1.0
    _plate_image(PLATE_IMAGE, rgb, w, h, False)
    _plate_image(PLATE_DEPTH, depth, w, h, True)
    return w, h


FINAL_ENGINES = (('BLENDER_EEVEE', 'EEVEE', 'Render with EEVEE'), ('CYCLES', 'Cycles', 'Render with Cycles'))
FINAL_BACKGROUNDS = (
    ('MAP', 'Imported Map', 'Everything from Blender: rigs and the map imported into the scene (best quality; faces '
     'and all shape keys show)'),
    ('PLATE', 'Game Plate', 'The game draws the background each frame (with its depth, so walls still hide rigs '
     'behind them); Blender renders only the scene\'s own objects over it. No map import needed'),
)
# What the final render changed, put back when it ends: scene name, engine, compositor group, film, lock, rigs
_final = None


def _final_plate_frame(scene, *_args):
    """render_pre for each frame of a plate render: the game clock to this frame, then its plate (render thread,
    interface locked)."""
    if _final is None or scene.name != _final['scene'] or _link is None:
        return
    lk = _link
    depsgraph = scene.view_layers[0].depsgraph
    if scene.camera:
        lk.follow_render(tuple(GAME_FROM_BLENDER @ scene.camera.evaluated_get(depsgraph).matrix_world.translation))
        time.sleep(lk.streaming())
    first = lk.render_frame is None
    lk.advance(scene.frame_current + scene.frame_subframe, scene.render.fps / scene.render.fps_base)
    try:
        write_plate(scene, depsgraph, settle=0.5 if first else 0.05)
    except (OSError, ValueError, TimeoutError) as error:
        print(f'Deadlock game plate, frame {scene.frame_current}: {error}')


def _layer_collection(root, name):
    if root.collection.name == name:
        return root
    for child in root.children:
        found = _layer_collection(child, name)
        if found:
            return found
    return None


def _hold_out_maps(scene):
    """An imported map stays in the scene for its lights and probe light on the rigs, but the plate shows the map:
    its collections render as holdouts (they still cast shadows and hide rigs behind them). Returns what to restore."""
    changed = []
    for layer in scene.view_layers:
        stack = [layer.layer_collection]
        while stack:
            lc = stack.pop()
            if any(k in lc.collection for k in ('dlr_map', 'dlr_crop_of', 'dlr_npcs')):
                changed.append((layer.name, lc.collection.name, lc.holdout))
                lc.holdout = True
            else:
                stack.extend(lc.children)
    return changed


@bpy.app.handlers.persistent
def _final_restore(*_args):
    # Not from the render thread's handler itself: changing the engine there races the render job's own teardown.
    def restore():
        global _final
        state, _final = _final, None
        if state is None:
            return None
        if _final_plate_frame in bpy.app.handlers.render_pre:
            bpy.app.handlers.render_pre.remove(_final_plate_frame)
        scene = bpy.data.scenes.get(state['scene'])
        if scene is not None:
            scene.render.engine = state['engine']
            scene.render.use_lock_interface = state['lock']
            if 'compositor' in state:
                scene.compositing_node_group = state['compositor']
                scene.render.film_transparent = state['film']
                for layer_name, collection_name, holdout in state['holdouts']:
                    layer = scene.view_layers.get(layer_name)
                    found = layer and _layer_collection(layer.layer_collection, collection_name)
                    if found:
                        found.holdout = holdout
        if _link is not None:
            _link.release_time()
            _link.declined -= state['rigs']
        return None
    bpy.app.timers.register(restore, first_interval=0.1)


class DLR_OT_game_final_render(bpy.types.Operator):
    bl_idname = 'dlr.game_final_render'
    bl_label = 'Final Render'
    bl_description = ('Render with EEVEE or Cycles instead of the game (the game stays the viewport preview): over '
                      'the imported map, or over game plates drawn per frame')
    animation: bpy.props.BoolProperty(name='Animation', default=False)

    @classmethod
    def poll(cls, context):
        return _final is None and context.scene.camera is not None

    def execute(self, context):
        global _final
        scene = context.scene
        background = scene.dlr_final_background
        state = {'scene': scene.name, 'engine': scene.render.engine, 'lock': scene.render.use_lock_interface,
                 'rigs': set()}
        lk = _link
        if background == 'PLATE':
            if lk is None or not lk.running or not isinstance(lk.stream, BridgeFrames):
                self.report({'ERROR'}, 'Game Plate needs a connected game session with the bridge\'s frame capture')
                return {'CANCELLED'}
            # The rigs come from Blender: out of the game while it draws the plates.
            for puppet in list(lk.puppets.values()):
                lk.remove_puppet(puppet)
                lk.declined.add(puppet.arm_name)
                state['rigs'].add(puppet.arm_name)
            for obj in scene.objects:
                if obj.type == 'ARMATURE' and obj.get('dlrig') and obj.name not in lk.declined:
                    lk.declined.add(obj.name)
                    state['rigs'].add(obj.name)
            lk.render_frame = None
            try:
                write_plate(scene, context.evaluated_depsgraph_get())
            except (OSError, ValueError, TimeoutError) as error:
                lk.declined -= state['rigs']
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            state.update(compositor=scene.compositing_node_group, film=scene.render.film_transparent,
                         holdouts=_hold_out_maps(scene))
            scene.render.engine = scene.dlr_final_engine  # the Depth output of Render Layers is the engine's
            plate_compositor(scene)
            bpy.app.handlers.render_pre.append(_final_plate_frame)
        _final = state
        scene.render.engine = scene.dlr_final_engine
        scene.render.use_lock_interface = True  # the plate handler writes images during the render
        result = bpy.ops.render.render('INVOKE_DEFAULT', animation=self.animation)
        if 'CANCELLED' in result:
            _final_restore()
            return {'CANCELLED'}
        return {'FINISHED'}


def _poll_frames():
    """Load a new frame and redraw Deadlock viewports. Here, not in view_draw: Blender forbids writing data-blocks
    while drawing, and the worker thread can't touch Blender at all. Never raises: Blender drops a timer that does
    (after an F12 render the viewport then never updated again)."""
    try:
        return _poll_frames_once()
    except Exception:
        traceback.print_exc()
        return 0.25


def _poll_frames_once():
    global _shown, _outside_shown, _stream_drawn
    lk = _link
    frame = lk.frame if lk else None
    changed = bool(lk) and lk.outside != _outside_shown
    if changed:
        _outside_shown = lk.outside
    stream = lk.stream if lk else None
    areas = [area for window in bpy.context.window_manager.windows if window.screen
             for area in window.screen.areas
             if area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED']
    if stream and areas and stream.seq != _stream_drawn:
        # Uploaded here, not in view_draw: a texture made while drawing costs ~5x (a 4K frame 15 ms, here 3).
        width = max((r.width for area in areas for r in area.regions if r.type == 'WINDOW'), default=0)
        try:
            _stream_texture(lk, width)
        except RuntimeError:
            # No GPU context in this tick (Blender's render job around an F12 render): next tick
            return 1 / 60
        _stream_drawn = stream.seq
        changed = True
    if frame and (_shown is None or frame[2] != _shown[1]):
        _image(frame[0])
        _shown = (frame[1], frame[2])
        changed = True
    if changed:
        for area in areas:
            area.tag_redraw()
    return 1 / 60 if stream else 0.03


def jump_viewports(context, pos, pitch, yaw):
    """Put every 3D viewport's view camera where the game camera is (orbiting 1 m in front of it)."""
    cam = blender_view(pos, pitch, yaw)
    forward = (cam.to_3x3() @ Vector((0, 0, -1))).normalized()
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        rv3d = area.spaces.active.region_3d
        rv3d.view_perspective = 'PERSP'
        rv3d.view_rotation = cam.to_quaternion()
        rv3d.view_distance = 1.0
        rv3d.view_location = cam.translation + forward * 1.0
        area.tag_redraw()


class DLR_OT_game_jump(bpy.types.Operator):
    bl_idname = 'dlr.game_jump'
    bl_label = 'Jump to Game View'
    bl_description = 'Move the 3D viewports to where the game camera is now'

    @classmethod
    def poll(cls, context):
        return _link is not None and _link.running

    def execute(self, context):
        cam = _link.game_camera()
        if cam is None:
            self.report({'ERROR'}, 'the game did not answer getpos')
            return {'CANCELLED'}
        jump_viewports(context, *cam)
        return {'FINISHED'}


class DLR_OT_game_connect(bpy.types.Operator):
    bl_idname = 'dlr.game_connect'
    bl_label = 'Connect to Deadlock'
    bl_description = 'Connect to a running -insecure Deadlock session (netcon port) and drive its camera'

    port: bpy.props.IntProperty(default=PORT)

    def execute(self, context):
        try:
            connect(context, self.port)
        except OSError as error:
            self.report({'ERROR'}, f'no game console on port {self.port}: {error}')
            return {'CANCELLED'}
        return {'FINISHED'}


def connect(context, port=PORT, setup=(), launched=False):
    global _link
    from . import gamelaunch
    # Poses, camera and frames are written into the game's memory: never into a VAC-secured game.
    if gamelaunch.game_insecure() is False:
        raise ConnectionRefusedError('Deadlock is running without -insecure: quit it and use Start Deadlock')
    if _link:
        _link.stop()
    _link = Link(port, setup, launched)
    try:
        _link.start()
    except OSError:
        _link = None
        raise
    if not bpy.app.timers.is_registered(_poll_frames):
        bpy.app.timers.register(_poll_frames, persistent=True)
    cam = _link.start_camera
    if cam and context.screen:
        jump_viewports(context, *cam)
    show_game_in_viewports(context)


def show_game_in_viewports(context):
    """Rendered shading (the game frame) without the floor grid and axes, which would draw over it."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            space.shading.type = 'RENDERED'
            overlay = space.overlay
            overlay.show_floor = overlay.show_axis_x = overlay.show_axis_y = overlay.show_axis_z = False
            area.tag_redraw()


_session = None  # the gamelaunch.Session this Blender started


def _session_poll():
    """Connects once a session started from here is ready."""
    global _session
    s = _session
    if s is None:
        return None
    if s.error or not s.ready.is_set():
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()
        return None if s.error else 0.5
    try:
        connect(bpy.context, s.port, s.setup_commands(), launched=True)
    except OSError as error:
        s.error = f'started, but could not connect: {error}'
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None


class DLR_OT_game_start(bpy.types.Operator):
    bl_idname = 'dlr.game_start'
    bl_label = 'Start Deadlock'
    bl_description = ('Launch Deadlock through Steam as a local -insecure session (no matchmaking, no VAC) with the '
                      'camera bridge, and connect when it is up (1-2 minutes). Your game settings are put back '
                      'when it quits')

    map: bpy.props.EnumProperty(name='Map', items=(
        ('dl_hideout', 'Hideout', 'The hideout the game starts in (loads fastest)'),
        ('dl_midtown', 'Midtown', 'The match map (needs more memory)'),
    ))
    resolution: bpy.props.EnumProperty(name='Game Resolution', default='2160', items=(
        ('2160', '3840 x 2160', 'Sharpest: renders come from 4K frames (averaged down for smaller outputs). Capped at '
                                'the screen size'),
        ('1440', '2560 x 1440', ''),
        ('1080', '1920 x 1080', 'Lightest on the viewport'),
    ))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        global _session
        from . import gamelaunch
        game = game_dir()
        if not game or not os.path.isdir(game):
            self.report({'ERROR'}, 'Deadlock not found (set its folder in the add-on preferences)')
            return {'CANCELLED'}
        height = int(self.resolution)
        screen = screen_size()
        if screen and height > screen[1]:
            height = screen[1]
        session = gamelaunch.Session(game, PORT, map_name=self.map, size=(height * 16 // 9, height),
                                     borderless=bool(screen) and height >= screen[1], precache=rig_models())
        try:
            session.start()
        except (OSError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        _session = session
        bpy.app.timers.register(_session_poll, first_interval=1.0)
        return {'FINISHED'}


class DLR_OT_game_quit(bpy.types.Operator):
    bl_idname = 'dlr.game_quit'
    bl_label = 'Quit Deadlock'
    bl_description = 'Quit the Deadlock session started from Blender and restore your game settings'

    def execute(self, context):
        global _link, _session
        con = None
        if _link:
            con = _link.con
            _link.running = False
            for puppet in list(_link.puppets.values()):
                _link.remove_puppet(puppet)
        if _session:
            _session.stop(con)
            _session = None
        if _link:
            _link.stop()
            _link = None
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


class DLR_OT_game_pose_rig(bpy.types.Operator):
    bl_idname = 'dlr.game_pose_rig'
    bl_label = 'Pose in Game'
    bl_description = ('Spawn the selected rig\'s hero in the game and pose it from the rig: the game draws and lights '
                      'it (viewport and F12), and it follows keyframes and posing. Visible rigs are added by '
                      'themselves; this brings back one taken out of the game')

    @classmethod
    def poll(cls, context):
        arm = puppet_armature(context.active_object)
        return (_link is not None and _link.running and _link.poses is not None and arm is not None
                and arm.name not in _link.puppets)

    def execute(self, context):
        arm = puppet_armature(context.active_object)
        _link.declined.discard(arm.name)
        try:
            _link.add_puppet(arm)
            _link.send_poses(context.evaluated_depsgraph_get())
        except (OSError, ValueError, TimeoutError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        # Hide the rig's own meshes from the overlay.
        arm.update_tag()
        return {'FINISHED'}


class DLR_OT_game_unpose_rig(bpy.types.Operator):
    bl_idname = 'dlr.game_unpose_rig'
    bl_label = 'Remove from Game'
    bl_description = 'Remove the rig\'s hero from the game'

    armature: bpy.props.StringProperty()

    def execute(self, context):
        puppet = _link.puppets.get(self.armature) if _link else None
        if puppet:
            _link.declined.add(self.armature)
            _link.remove_puppet(puppet)
            arm = bpy.data.objects.get(self.armature)
            if arm:
                arm.update_tag()
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
            col.label(text='Camera: bridge (free camera)' if lk.bridge else
                      'Camera: console (moves the hero; start the session with --bridge for a free camera)')
            if lk.stream:
                w, h = lk.stream.size
                col.label(text=f'Streaming the game window ({w}x{h})')
            elif lk.frame:
                col.label(text=f'Screenshot mode: frame {lk.frame[2]}, {lk.timing * 1000:.0f} ms')
            if lk.outside:
                col.label(text='View is outside the play area', icon='ERROR')
            col.operator(DLR_OT_game_jump.bl_idname, icon='VIEW_CAMERA')
            if lk.poses:
                col.label(text='Visible Deadlock rigs are posed in the game (Rendered shading or F12)')
                arm = puppet_armature(context.active_object)
                if arm is not None and arm.name in lk.declined:
                    col.operator(DLR_OT_game_pose_rig.bl_idname, icon='ARMATURE_DATA')
                for name, puppet in lk.puppets.items():
                    row = col.row()
                    row.label(text=f'{name}: entity {puppet.entity}', icon='OUTLINER_OB_ARMATURE')
                    row.operator(DLR_OT_game_unpose_rig.bl_idname, text='', icon='X').armature = name
            row = col.row()
            row.operator(DLR_OT_game_disconnect.bl_idname)
            if _session:
                row.operator(DLR_OT_game_quit.bl_idname, icon='QUIT')
        elif _session and not _session.error and not _session.ready.is_set():
            col.label(text=_session.status + '...', icon='TIME')
        else:
            row = col.row()
            row.operator(DLR_OT_game_start.bl_idname, icon='PLAY')
            row.operator(DLR_OT_game_connect.bl_idname, icon='UNLINKED')
        if _session and _session.error:
            col.label(text=_session.error, icon='ERROR')
        if lk and lk.error:
            col.label(text=lk.error, icon='ERROR')


def _view_menu(self, context):
    if context.scene.render.engine == ENGINE_ID:
        self.layout.operator(DLR_OT_game_jump.bl_idname, icon='VIEW_CAMERA')
    elif _link is not None and _link.running:
        self.layout.operator(DLR_OT_game_plate.bl_idname, icon='IMAGE_BACKGROUND')


CLASSES = (DeadlockGameEngine, DLR_OT_game_connect, DLR_OT_game_disconnect, DLR_OT_game_start, DLR_OT_game_quit,
           DLR_OT_game_jump,
           DLR_OT_game_pose_rig, DLR_OT_game_unpose_rig, DLR_PT_game_render,
           DLR_OT_game_plate, DLR_OT_game_final_render)


# Blender panels list the engines they show for; a custom engine has to add itself. Output settings (format,
# frame range, output path, encoding) and object data (camera lens, mesh, light) are engine independent.
SHARED_PANEL_CONTEXTS = {'output', 'data'}


def _shared_panels():
    for name in dir(bpy.types):
        panel = getattr(bpy.types, name)
        engines = getattr(panel, 'COMPAT_ENGINES', None)
        if (isinstance(engines, set) and 'BLENDER_RENDER' in engines
                and getattr(panel, 'bl_context', None) in SHARED_PANEL_CONTEXTS):
            yield panel


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    for panel in _shared_panels():
        panel.COMPAT_ENGINES.add(ENGINE_ID)
    bpy.types.Scene.dlr_final_engine = bpy.props.EnumProperty(name='Final Engine', items=FINAL_ENGINES)
    bpy.types.Scene.dlr_final_background = bpy.props.EnumProperty(name='Final Background', items=FINAL_BACKGROUNDS)
    for handlers in (bpy.app.handlers.render_complete, bpy.app.handlers.render_cancel):
        handlers.append(_final_restore)
    if not bpy.app.timers.is_registered(_poll_frames):
        bpy.app.timers.register(_poll_frames, persistent=True)
    bpy.types.VIEW3D_MT_view.append(_view_menu)
    for handlers in (bpy.app.handlers.render_complete, bpy.app.handlers.render_cancel):
        handlers.append(_render_finished)


@atexit.register
def _quit():
    if _link:
        _link.stop()


@bpy.app.handlers.persistent
def _render_finished(*_args):
    if _link:
        _link.release_time()


def unregister():
    global _link
    for handlers in (bpy.app.handlers.render_complete, bpy.app.handlers.render_cancel):
        for handler in (_render_finished, _final_restore):
            if handler in handlers:
                handlers.remove(handler)
    if _final_plate_frame in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.remove(_final_plate_frame)
    del bpy.types.Scene.dlr_final_engine, bpy.types.Scene.dlr_final_background
    if _link:
        _link.stop()
        _link = None
    if bpy.app.timers.is_registered(_poll_frames):
        bpy.app.timers.unregister(_poll_frames)
    bpy.types.VIEW3D_MT_view.remove(_view_menu)
    for panel in _shared_panels():
        panel.COMPAT_ENGINES.discard(ENGINE_ID)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
