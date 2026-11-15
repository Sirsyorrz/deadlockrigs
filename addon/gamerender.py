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
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
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
POSE_SLOTS = 8
# A puppet's entity follows its pose (the game culls a model by its entity's bounds) once the pose has moved this
# far, in game units.
FOLLOW_DISTANCE = 16.0
# Close enough to the pose that the model's bounds still cover it (a fresh spawn starts metres away).
ARRIVED = 48.0
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
        self.seq = 0
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
        f = _TIME_REPLY.unpack(data[:_TIME_REPLY.size])
        return {'available': f[1], 'frozen': f[2], 'step_done': f[3], 'framerate': f[4]}

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


class Stream:
    """The game window as a live stream: ffmpeg x11grab writes raw RGBX frames to a pipe, a thread keeps the
    latest. Frames stay bytes: the shader unpacks them, so no per-frame conversion on the CPU."""

    def __init__(self, wid):
        self.wid = wid
        self.size = None
        self.latest = None  # (bytes, arrival time, sequence)
        self.seq = 0
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

    @property
    def min_vfov(self):
        return 1.0 if self.bridge else MIN_VFOV

    def allowed(self, view):
        # Only the console camera moves the hero (the game kills it outside the play area).
        return self.bridge is not None or view.inside()

    def _place(self, view):
        if self.bridge:
            return self.bridge.place(view)
        self.con.send(*view.commands())
        return None

    def stream_view(self, arrived):
        """The view a stream frame that arrived at this time most likely shows."""
        chosen = None
        for sent, view in list(self.history):
            if sent > arrived - (STREAM_LATENCY_BRIDGE if self.bridge else STREAM_LATENCY):
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
        if self.bridge is None:
            self.con.send(*SETUP)
        wid = None
        if not os.environ.get('DISPLAY'):
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
        used = {p.slot for p in self.puppets.values()}
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
        names, parents, inv_binds = self.poses.skeleton(slot)
        puppet = Puppet(arm.name, slot, bound['entity'], model, names, parents, inv_binds, arm)
        self.puppets[arm.name] = puppet
        self._save_spawned()
        return puppet

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
        """Place the game camera and take the next stream frame that shows it: linear float RGBA, rows bottom-up.
        Same pixels as a png screenshot (mean diff 0.4/255) without the ~2 s the engine takes to write one."""
        with self.shot_lock:
            seq = self.bridge.place(view)
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
        rgb = np.frombuffer(latest[0], np.uint8).reshape(h, w, 4)[::-1].astype(np.float32) / 255.0
        c = rgb[..., :3]
        rgb[..., :3] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
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
                    self._place(view)
            except OSError as error:
                self.error = str(error)
                continue
            self.history.append((time.time(), view))
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


_packed_shader = None


def packed_shader():
    """Draws a stream frame uploaded as raw bytes: each R32F texel holds one RGBX pixel's four bytes. Unpacks
    them, flips rows (X11 is top-down) and decodes sRGB to linear, like an sRGB image texture would."""
    global _packed_shader
    if _packed_shader is None:
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
        _packed_shader = gpu.shader.create_from_info(info)
    return _packed_shader


def _stream_texture(lk):
    """The latest stream frame as a texture, uploaded once per new frame. Bilinear is done in the shader: the
    texels are packed bytes, hardware filtering would blend their bit patterns."""
    global _stream_tex
    latest = lk.stream.latest
    if latest is None:
        return None
    buf, arrived, seq = latest
    if _stream_tex is None or _stream_tex[0] != seq:
        w, h = lk.stream.size
        data = gpu.types.Buffer('FLOAT', w * h, np.frombuffer(buf, np.float32))
        _stream_tex = (seq, gpu.types.GPUTexture((w, h), format='R32F', data=data), lk.stream_view(arrived))
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
            frame = _stream_texture(lk)
            if lk.outside:
                _draw_message(region, 'Outside the game\'s play area: the game camera stays put '
                                      '(View > Jump to Game View to go back)')
            elif frame is not None and frame[2] is not None:
                self._draw_frame(rv3d.window_matrix, frame[1], w / h, frame[2], packed_shader())
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
        camera = scene.camera
        if camera is None:
            raise ValueError('the scene has no camera')
        render = scene.render
        proj = camera.calc_matrix_camera(depsgraph, x=width, y=height, scale_x=render.pixel_aspect_x,
                                         scale_y=render.pixel_aspect_y)
        w, h = lk.bridge.size if lk.bridge else (16, 9)
        view = game_view(camera.matrix_world, proj, w / h, lk.min_vfov)
        lk.sync_puppets(depsgraph)
        frame_number = scene.frame_current + scene.frame_subframe
        first = lk.render_frame is None
        if lk.puppets:
            # Entities move through server ticks: wait before the clock stops; later frames move during the step.
            lk.send_poses(depsgraph, settle=2.0 if first else 0.0)
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
    u = u0 + (np.arange(width) + 0.5) / width * (u1 - u0)
    v = v0 + (np.arange(height) + 0.5) / height * (v1 - v0)
    x = np.clip((u + th) / (2 * th) * fw - 0.5, 0, fw - 1.001)
    y = np.clip((v + tv) / (2 * tv) * fh - 0.5, 0, fh - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = (x - x0)[None, :, None], (y - y0)[:, None, None]
    top = frame[y0][:, x0] * (1 - fx) + frame[y0][:, x0 + 1] * fx
    bottom = frame[y0 + 1][:, x0] * (1 - fx) + frame[y0 + 1][:, x0 + 1] * fx
    return (top * (1 - fy) + bottom * fy).reshape(-1, 4).astype(np.float32)


def _draw_message(region, text):
    blf.size(0, 16)
    width, _ = blf.dimensions(0, text)
    blf.position(0, max(10, (region.width - width) / 2), region.height / 2, 0)
    blf.color(0, 1.0, 0.75, 0.3, 1.0)
    blf.draw(0, text)


_outside_shown = False


def _poll_frames():
    """Load a new frame and redraw Deadlock viewports. Here, not in view_draw: Blender forbids writing data-blocks
    while drawing, and the worker thread can't touch Blender at all."""
    global _shown, _outside_shown, _stream_drawn
    lk = _link
    frame = lk.frame if lk else None
    changed = bool(lk) and lk.outside != _outside_shown
    if changed:
        _outside_shown = lk.outside
    stream = lk.stream if lk else None
    if stream and stream.seq != _stream_drawn:
        _stream_drawn = stream.seq
        changed = True
    if frame and (_shown is None or frame[2] != _shown[1]):
        _image(frame[0])
        _shown = (frame[1], frame[2])
        changed = True
    if changed:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED':
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
        cam = _link.start_camera
        if cam:
            jump_viewports(context, *cam)
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


class DLR_OT_game_standard_view(bpy.types.Operator):
    bl_idname = 'dlr.game_standard_view'
    bl_label = 'Use Standard View Transform'
    bl_description = ('Show renders as the game shows them: the game frame is already tonemapped, so any other '
                      'view transform (AgX, Filmic) changes its look')

    def execute(self, context):
        context.scene.view_settings.view_transform = 'Standard'
        context.scene.view_settings.look = 'None'
        context.scene.view_settings.exposure = 0.0
        context.scene.view_settings.gamma = 1.0
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
            col.operator(DLR_OT_game_disconnect.bl_idname)
        else:
            col.operator(DLR_OT_game_connect.bl_idname, icon='UNLINKED')
        if lk and lk.error:
            col.label(text=lk.error, icon='ERROR')
        vs = context.scene.view_settings
        if vs.view_transform != 'Standard' or vs.look != 'None' or vs.exposure != 0 or vs.gamma != 1:
            box = col.box()
            box.label(text=f'F12 renders get the {vs.view_transform} look on top of the game\'s', icon='INFO')
            box.operator(DLR_OT_game_standard_view.bl_idname)


def _view_menu(self, context):
    if context.scene.render.engine == ENGINE_ID:
        self.layout.operator(DLR_OT_game_jump.bl_idname, icon='VIEW_CAMERA')


CLASSES = (DeadlockGameEngine, DLR_OT_game_connect, DLR_OT_game_disconnect, DLR_OT_game_jump,
           DLR_OT_game_pose_rig, DLR_OT_game_unpose_rig, DLR_OT_game_standard_view, DLR_PT_game_render)


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
        if _render_finished in handlers:
            handlers.remove(_render_finished)
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
