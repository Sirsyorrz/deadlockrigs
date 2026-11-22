"""Start and stop the Deadlock session the game render engine drives: the user's own Steam install, launched with
-insecure (no VAC, local server only) and the camera bridge loaded as the session's server.dll.

The bridge goes in through one extra gameinfo.gi search path that exists only while the game loads: the original
file is restored byte-exact as soon as the game's console is up, so a normal Steam launch never sees it. The game
saves archived cvars (HUD, camera) and its window mode on exit; the user's cfgs are copied before the launch and put
back after the session quits, by Blender or, when Blender is gone by then, by a detached watcher process.
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib

APPID = '1422450'
OVERLAY = 'citadel/dlrigs_bridge'
MARK = '// dlrigs session'
MOD_PATHS = ('citadel/deadworks_mods', 'citadel/addons', 'citadel/deadworks_addons')
HERE = os.path.dirname(os.path.abspath(__file__))
# packaged: native/ inside the add-on; in the repo the build output next to addon/
BRIDGE_DLLS = (os.path.join(HERE, 'native', 'dlr_bridge.dll'),
               os.path.join(os.path.dirname(HERE), 'native', 'dlr_bridge', 'build', 'dlr_bridge.dll'))
# The build watermark (Deadlock logo, top left) and the connection/matchmaking notes live in the overlay layer, not
# the HUD, so no cvar hides them. The session mounts a copy of their stylesheet with every entry kept invisible.
# Same-length edit of the compiled file: offsets and block sizes stay valid. It has to be packed: the engine searches
# every path's pak01 VPK before any loose folder, so a loose copy in the overlay never wins over citadel's VPK.
CLIENT_STATUS_CSS = 'panorama/styles/citadel_client_status.vcss_c'
CLIENT_STATUS_EDIT = (b'.StatusEntry.ShowStatus{opacity: 1.0;}', b'.StatusEntry.ShowStatus{opacity: 0.0;}')
# Launch to console: ~60 s, plus a stall of ~95 s that some launches sit through after "Game started".
READY_DRAWING = 8.0  # seconds of steady drawing before the first console client
CONSOLE_TIMEOUT = 300
# The watcher restores once the game has been gone this long (Proton can drop the process briefly while starting).
WATCH_GONE = 10.0
WATCH_POLL = 2.0


def state_dir():
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'dlr-game', 'session')


def _claim(path):
    """Rename path to a name only this caller has, or None when it's gone (Blender and the watcher can both be
    restoring the same session)."""
    claimed = f'{path}.claim{os.getpid()}.{threading.get_ident()}'
    try:
        os.rename(path, claimed)
    except FileNotFoundError:
        return None
    return claimed


def log_path():
    return os.path.join(os.path.dirname(state_dir()), 'game.log')


def log(text):
    """One line in the session log users can send with a bug report (errors only show briefly in the panel)."""
    try:
        os.makedirs(os.path.dirname(log_path()), exist_ok=True)
        with open(log_path(), 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')
    except OSError:
        pass


class LoggedText:
    """A text attribute (an error) that also goes to the session log whenever it changes to something new."""

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, owner=None):
        return self if obj is None else obj.__dict__.get(self.name, '')

    def __set__(self, obj, text):
        if text and text != obj.__dict__.get(self.name):
            log(f'{type(obj).__name__}: {text}')
        obj.__dict__[self.name] = text


def bridge_dll():
    return next((p for p in BRIDGE_DLLS if os.path.exists(p)), None)


def steam_root():
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Valve\Steam') as k:
                return winreg.QueryValueEx(k, 'SteamPath')[0]
        except OSError:
            return None
    for p in ('~/.steam/root', '~/.local/share/Steam', '~/.steam/steam'):
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            return os.path.realpath(p)
    return None


def _game_commandlines():
    """Command lines (lists of arguments) of running deadlock.exe processes; None where they can't be read."""
    if os.name == 'nt':
        try:
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 "Get-CimInstance Win32_Process -Filter \"Name='deadlock.exe'\" | ForEach-Object { $_.CommandLine }"],
                capture_output=True, text=True, errors='replace', creationflags=0x08000000)
        except OSError:
            return None
        if out.returncode or out.stdout is None:
            return None
        return [line.split() for line in out.stdout.splitlines() if line.strip()]
    found = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read()
        except OSError:
            continue
        args = cmd.split(b'\0')
        # Wine shows the Windows path as argv[0]; launcher wrappers only mention it further along.
        if args[0].replace(b'\\', b'/').lower().endswith(b'win64/deadlock.exe'):
            found.append([a.decode('utf-8', 'replace') for a in args if a])
    return found


def game_running(game):
    """The game's process. Never probe the console port for this: a console client that connects and leaves while
    the game replays its log to it can hang the game for good."""
    if os.name == 'nt':
        # tasklist's output can come back as None (seen from Blender on Windows); the command lines are the fallback.
        try:
            out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq deadlock.exe', '/NH'], capture_output=True,
                                 text=True, errors='replace', creationflags=0x08000000).stdout
        except OSError:
            out = None
        if out:
            return 'deadlock.exe' in out.lower()
    return bool(_game_commandlines())


def session_running():
    """A session game (launched with -insecure) is running. The user's own Steam launch never is, so a normal
    launch started right after the session quit isn't taken for the session."""
    return any('-insecure' in args for args in _game_commandlines() or ())


def _shut_down(console_log, since):
    """The session's console log (rewritten each launch) shows the engine finished shutting down: the cfgs are
    saved by then, though Proton can keep the process around for a while longer."""
    try:
        if os.path.getmtime(console_log) < since:
            return False
        with open(console_log, 'rb') as f:
            f.seek(max(0, os.path.getsize(console_log) - 4096))
            return b'ShutdownSource2Logging' in f.read()
    except OSError:
        return False


def game_insecure():
    """True when every running Deadlock is an -insecure session, False when one is not (a VAC-secured game: the
    add-on must not touch it), None when no game runs or its command line can't be read."""
    lines = _game_commandlines()
    if not lines:
        return None
    return all('-insecure' in args for args in lines)


class Session:
    """One launch. start() returns at once; a thread runs the launch and reports through .status / .error / .ready.
    Everything it changed is undone by stop() or, after a crash, by recover() (run before every start)."""

    error = LoggedText()

    def __init__(self, game, port, size=(1920, 1080), mods=False, extra=(), map_name='dl_hideout', borderless=False,
                 precache=()):
        self.game = game
        self.port = port
        self.size = size
        self.map_name = map_name
        self.borderless = borderless
        self.mods = mods
        self.extra = list(extra)
        # Resource paths added to the map's load list by the bridge (models/... .vmdl, particles/... .vpcf): ones
        # the map doesn't use would otherwise draw as the ERROR model.
        self.precache = list(precache)
        self.status = ''
        self.error = ''
        self.ready = threading.Event()
        self.done = threading.Event()
        self.state = state_dir()
        self.gameinfo = os.path.join(game, 'citadel', 'gameinfo.gi')
        self.backup = os.path.join(self.state, 'gameinfo.gi.orig')
        self.cfg_backup = os.path.join(self.state, 'addon_cfg')  # dlsession.py's cfg/ has another layout

    # gameinfo.gi ----------------------------------------------------------------------------------------------
    def _patched(self, original):
        nl = '\r\n' if '\r\n' in original else '\n'
        lines = original.split(nl)
        block = _search_paths_block(lines)
        keys = {lines[i].split()[0].lower() for i in block if lines[i].split()}
        out, inserted = [], False
        for i, line in enumerate(lines):
            parts = line.split()
            key = parts[0].lower() if parts else ''
            if i not in block:
                out.append(line)
                continue
            if key == 'game' and not inserted:
                indent = line[:len(line) - len(line.lstrip())]
                # The stock file (since the 2026-09-29 update) has no Mod/Write keys, so the first Game path
                # becomes MOD and the write path: the overlay would take both from citadel and the game dies with
                # "Unable to read default keybinding configuration (user_keys_default)".
                out += [f'{indent}{k}\t\t\t\tcitadel\t{MARK}' for k in ('Mod', 'Write') if k.lower() not in keys]
                out.append(f'{indent}Game\t\t\t\t{OVERLAY}\t{MARK}')
                inserted = True
            # Without mods: Deadlock Mod Manager's mounts are left out, so the session looks like the vanilla game.
            if key in ('game', 'addonroot') and not self.mods and len(parts) > 1 and parts[1] in MOD_PATHS:
                out.append(f'{MARK} {line.strip()}')
                continue
            out.append(line)
        if not inserted:
            raise RuntimeError('gameinfo.gi has no Game search path')
        return nl.join(out)

    def restore_gameinfo(self):
        backup = _claim(self.backup)
        if not backup:
            return
        original = open(backup, 'rb').read()
        current = open(self.gameinfo, 'rb').read()
        if current != original:
            if MARK.encode() in current:
                with open(self.gameinfo, 'wb') as f:
                    f.write(original)
            else:
                # Steam verify or a mod manager rewrote it after our edit: theirs wins, keep ours aside.
                shutil.copyfile(backup, f'{self.backup}.{int(time.time())}')
        os.remove(backup)

    # user cfgs ------------------------------------------------------------------------------------------------
    def _user_cfgs(self):
        root = steam_root()
        found = []
        if root:
            userdata = os.path.join(root, 'userdata')
            for user in (os.listdir(userdata) if os.path.isdir(userdata) else ()):
                app = os.path.join(userdata, user, APPID)
                for sub, keep in (('local/cfg', lambda n: '.vcfg' in n or n.startswith('video.txt')),
                                  ('remote', lambda n: n.endswith('.vcfg'))):
                    d = os.path.join(app, sub)
                    if os.path.isdir(d):
                        found += [os.path.join(d, n) for n in os.listdir(d) if keep(n)]
        d = os.path.join(self.game, 'citadel', 'cfg')
        found += [os.path.join(d, n) for n in os.listdir(d) if n.startswith('video.txt')] if os.path.isdir(d) else []
        return found

    def _backup_name(self, path):
        return os.path.join(self.cfg_backup, hashlib.sha1(path.encode()).hexdigest()[:16])

    def snapshot_cfgs(self):
        shutil.rmtree(self.cfg_backup, ignore_errors=True)
        os.makedirs(self.cfg_backup)
        index = {}
        for path in self._user_cfgs():
            name = self._backup_name(path)
            shutil.copy2(path, name)
            index[path] = name
        with open(os.path.join(self.cfg_backup, 'index.json'), 'w') as f:
            json.dump(index, f)

    def restore_cfgs(self):
        index_path = _claim(os.path.join(self.cfg_backup, 'index.json'))
        if not index_path:
            return
        with open(index_path) as f:
            index = json.load(f)
        for path, name in index.items():
            shutil.copy2(os.path.join(self.cfg_backup, os.path.basename(name)), path)
        for path in self._user_cfgs():
            if path not in index:
                os.remove(path)
        shutil.rmtree(self.cfg_backup, ignore_errors=True)

    def _session_video(self):
        """The session's render size, put back with the other cfgs. A window is the render size but can't be
        bigger than the desktop's work area (a 4K window on a 4K screen loses the panel's height); borderless
        fullscreen renders at defaultres into a screen-sized swapchain, so it is used for screen-sized sessions
        only (smaller ones would be upscaled into it: 4x the pixels to copy for nothing)."""
        w, h = self.size
        want = {'setting.fullscreen': '0', 'setting.coop_fullscreen': '1' if self.borderless else '0',
                'setting.defaultres': str(w),
                'setting.defaultresheight': str(h)}
        for path in self._user_cfgs():
            if os.path.basename(path) != 'video.txt':
                continue
            lines = open(path, encoding='utf-8', errors='replace').read().splitlines(keepends=True)
            for i, line in enumerate(lines):
                parts = line.split('"')
                if len(parts) >= 5 and parts[1] in want:
                    parts[3] = want[parts[1]]
                    lines[i] = '"'.join(parts)
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

    def recover(self):
        """Undo a session that never got to stop (Blender crashed, or the game was closed by hand)."""
        self.restore_gameinfo()
        if not game_running(self.game):
            self.restore_cfgs()
            self.remove_bridge()

    def bridge_path(self):
        return bridge_path(self.game)

    def console_log(self):
        return os.path.join(self.game, 'citadel', 'console.log')

    def precache_path(self):
        return os.path.join(os.path.dirname(self.bridge_path()), 'precache.txt')

    def remove_bridge(self):
        """The bridge stays inert without -insecure, but it is only on disk while a session runs, so a normal
        (VAC-secured) launch can never find a foreign server.dll even if gameinfo.gi was left patched."""
        for path in (self.bridge_path(), self.precache_path(), self.client_status_path()):
            try:
                os.remove(path)
            except OSError:
                pass
        # Meshes sent from Blender (gamemodel): a new file per edit, only of use to the session that made them.
        shutil.rmtree(os.path.join(self.game, *OVERLAY.split('/'), 'models', 'dlrigs'), ignore_errors=True)

    def client_status_path(self):
        return os.path.join(self.game, *OVERLAY.split('/'), 'pak01_dir.vpk')

    def hide_client_status(self):
        """Mount the client status stylesheet with its entries invisible. Skipped (logged) when the game's copy no
        longer has the rule: the watermark then shows, nothing else changes."""
        try:
            data = vpk_read(os.path.join(self.game, 'citadel', 'pak01_dir.vpk'), CLIENT_STATUS_CSS)
        except (OSError, ValueError, struct.error) as error:
            log(f'client status style not read: {error}')
            return
        old, new = CLIENT_STATUS_EDIT
        if data is None or data.count(old) != 1:
            log('client status style: rule not found, the watermark stays')
            return
        try:
            vpk_write(self.client_status_path(), {CLIENT_STATUS_CSS: data.replace(old, new)})
        except OSError as error:
            log(f'client status style not written: {error}')

    # launch ---------------------------------------------------------------------------------------------------
    def args(self):
        w, h = self.size
        args = ['-dev', '-insecure', '-console', '-novid', '-windowed', '-condebug', '-netconport', str(self.port),
                '-w', str(w), '-h', str(h)]
        if self.map_name != 'dl_hideout':
            args += ['+map', self.map_name]
        return args + self.extra

    def setup_commands(self):
        """Console commands for the first connection."""
        if self.map_name == 'dl_hideout':
            return []
        # A map loaded from the command line starts without a hero; the camera bridge needs the player in the game
        # (the hero is folded away in shots).
        return ['jointeam 2', 'selecthero hero_atlas']

    def start(self):
        if game_running(self.game):
            raise RuntimeError('Deadlock is already running: quit it first (the session needs its own launch '
                               'options), or Connect if it is a session started earlier')
        dll = bridge_dll()
        if not dll:
            raise RuntimeError('the camera bridge (dlr_bridge.dll) is missing from the add-on')
        self.recover()
        os.makedirs(self.state, exist_ok=True)
        target = self.bridge_path()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(dll, target)
        with open(self.precache_path(), 'w') as f:
            f.write(''.join(path + '\n' for path in self.precache))
        self.hide_client_status()
        stale = os.path.join(os.path.dirname(target), 'dlr_unlocker.dll')
        if os.path.exists(stale):
            os.remove(stale)
        self.snapshot_cfgs()
        self._session_video()
        original = open(self.gameinfo, 'rb').read()
        with open(self.backup, 'wb') as f:
            f.write(original)
        try:
            with open(self.gameinfo, 'wb') as f:
                f.write(self._patched(original.decode('utf-8')).encode('utf-8'))
            launch_steam(['-applaunch', APPID, *self.args()])
        except Exception:
            self.restore_gameinfo()
            self.restore_cfgs()
            self.remove_bridge()
            raise
        self.start_watcher()
        self.status = 'Starting Deadlock'
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        try:
            start = time.time()
            end = start + CONSOLE_TIMEOUT
            # The bridge answers once the map's server has loaded it, then counts drawn views. A console client
            # that connects while the map is still loading can hang the game for good (twice, at 2 s into the
            # map): wait until it has drawn steadily for a while.
            steady_since, last_views, seen, gone_since = None, -1, False, None
            while True:
                views = _bridge_views()
                if views is not None and views > last_views >= 0:
                    steady_since = steady_since or time.time()
                    if time.time() - steady_since >= READY_DRAWING:
                        break
                else:
                    steady_since = None
                last_views = -1 if views is None else views
                if time.time() > end:
                    self.error = ('the game did not come up with the camera bridge' if game_running(self.game)
                                  else 'the game did not start (is Steam running?)')
                    return
                if game_running(self.game):
                    seen, gone_since = True, None
                elif seen and time.time() - (gone_since or time.time()) < 3:
                    gone_since = gone_since or time.time()
                elif seen:
                    # Put everything back now: the user's next launch (from Steam) must not see the session's edits.
                    self.error = f'the game quit while starting (its log: {self.console_log()})'
                    self.restore_gameinfo()
                    self.restore_cfgs()
                    self.remove_bridge()
                    return
                if game_insecure() is False:
                    self.error = ('Deadlock started without -insecure (was it already open, or launched by '
                                  'something else?): not connecting')
                    return
                self.status = f'Starting Deadlock ({time.time() - start:.0f} s)'
                time.sleep(0.5)
            self.status = 'Ready'
            self.ready.set()
        finally:
            self.restore_gameinfo()
            self.done.set()

    def token_path(self):
        return os.path.join(self.state, 'session_id')

    def start_watcher(self):
        """A process outside Blender that restores everything when the game quits, so closing Blender first (or
        Blender crashing) doesn't leave the session's window mode and cvars in the user's next normal launch."""
        token = f'{os.getpid()}-{time.time():.0f}'
        with open(self.token_path(), 'w') as f:
            f.write(token)
        python = sys.executable
        if not python or 'python' not in os.path.basename(python).lower():
            log('restore watcher not started: no Python executable')
            return
        args = [python, '-I', os.path.abspath(__file__), '--watch', self.game, token]
        try:
            if os.name == 'nt':
                subprocess.Popen(args, creationflags=0x00000008 | 0x00000200 | 0x08000000,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(args, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError as error:
            log(f'restore watcher not started: {error}')

    def stop(self, con=None):
        """Quit the game (through the given console client: the game serves only one) and put the cfgs back."""
        try:
            if con is None:
                from .gamerender import Netcon
                con = Netcon(self.port)
            con.send('quit')
            time.sleep(0.5)
        except OSError:
            pass
        end = time.time() + 30
        while time.time() < end and game_running(self.game):
            time.sleep(0.5)
        self.restore_gameinfo()
        if not game_running(self.game):
            self.restore_cfgs()
            self.remove_bridge()


def watch(game, token):
    """The detached watcher: waits for the session's game to come and go, then undoes the session. Gives up
    without touching anything once a newer session has started (its token replaced ours)."""
    session = Session(game, 0)

    def current():
        try:
            with open(session.token_path()) as f:
                return f.read() == token
        except OSError:
            return False

    started = time.time()
    give_up = started + CONSOLE_TIMEOUT + 60
    seen, gone_since = False, None
    while current():
        if seen and _shut_down(session.console_log(), started):
            break
        if session_running():
            seen, gone_since = True, None
        elif seen or time.time() > give_up:
            gone_since = gone_since or time.time()
            if time.time() - gone_since >= WATCH_GONE:
                break
        time.sleep(WATCH_POLL)
    else:
        return
    if not current():
        return
    restoring = os.path.exists(os.path.join(session.cfg_backup, 'index.json'))
    session.restore_gameinfo()
    session.restore_cfgs()
    if restoring:
        log('watcher: the game quit without Blender, settings restored')
    # The bridge dll can't be deleted on Windows while the exiting process still has it loaded.
    end = time.time() + 120
    while session_running() and time.time() < end:
        time.sleep(WATCH_POLL)
    session.remove_bridge()


def _search_paths_block(lines):
    """Indices of the lines inside gameinfo.gi's SearchPaths { } block."""
    start = next((i for i, line in enumerate(lines) if line.split()[:1] and line.split()[0].lower() == 'searchpaths'),
                 None)
    if start is None:
        return range(0)
    depth = 0
    for i in range(start + 1, len(lines)):
        code = lines[i].split('//')[0]
        depth += code.count('{') - code.count('}')
        if depth <= 0 and '}' in code:
            return range(start + 1, i)
    return range(start + 1, len(lines))


def bridge_path(game):
    return os.path.join(game, *OVERLAY.split('/'), 'bin', 'win64', 'server.dll')


def _vpk_entries(dir_path):
    """(ext, folder, name, preload bytes, archive, offset, length) per file of a VPK v2 directory, and the offset
    of data stored in the _dir file itself."""
    with open(dir_path, 'rb') as f:
        signature, version, tree_size = struct.unpack('<III', f.read(12))
        if signature != 0x55AA1234 or version != 2:
            raise ValueError(f'{dir_path} is not a VPK v2')
        f.read(16)
        tree = f.read(tree_size)
    data_start = 28 + tree_size
    pos = 0

    def string():
        nonlocal pos
        end = tree.index(b'\0', pos)
        text = tree[pos:end].decode('utf-8', 'replace')
        pos = end + 1
        return text

    def entries():
        nonlocal pos
        while True:
            e = string()
            if not e:
                return
            while True:
                p = string()
                if not p:
                    break
                while True:
                    n = string()
                    if not n:
                        break
                    _crc, preload, archive, offset, length, _end = struct.unpack_from('<IHHIIH', tree, pos)
                    pos += 18 + preload
                    yield e, p, n, tree[pos - preload:pos], archive, offset, length

    return entries(), data_start


def vpk_read(dir_path, wanted):
    """One file's bytes from a VPK v2 (pak01_dir.vpk + pak01_NNN.vpk), or None when it isn't in there."""
    stem, ext = wanted.rsplit('.', 1)
    folder, name = stem.rsplit('/', 1) if '/' in stem else (' ', stem)
    entries, data_start = _vpk_entries(dir_path)
    for e, p, n, body, archive, offset, length in entries:
        if (e, p, n) != (ext, folder, name):
            continue
        if length:
            if archive == 0x7FFF:
                path, offset = dir_path, data_start + offset
            else:
                path = dir_path.replace('_dir.vpk', f'_{archive:03d}.vpk')
            with open(path, 'rb') as f:
                f.seek(offset)
                body += f.read(length)
        return body
    return None


def vpk_paths(dir_path, ext):
    """Every file path in a VPK v2 with that extension, e.g. 'models/props/barrel_03.vmdl_c'."""
    entries, _ = _vpk_entries(dir_path)
    return [f'{n}.{e}' if p == ' ' else f'{p}/{n}.{e}' for e, p, n, *_ in entries if e == ext]


def vpk_write(path, files):
    """A VPK v2 with its data inside the _dir file, holding {path inside: bytes}."""
    tree, body = bytearray(), bytearray()
    by_ext = {}
    for inner, data in files.items():
        stem, ext = inner.rsplit('.', 1)
        folder, name = stem.rsplit('/', 1) if '/' in stem else (' ', stem)
        by_ext.setdefault(ext, {}).setdefault(folder, []).append((name, data))
    for ext, folders in by_ext.items():
        tree += ext.encode() + b'\0'
        for folder, entries in folders.items():
            tree += folder.encode() + b'\0'
            for name, data in entries:
                tree += name.encode() + b'\0'
                tree += struct.pack('<IHHIIH', zlib.crc32(data), 0, 0x7FFF, len(body), len(data), 0xFFFF)
                body += data
            tree += b'\0'
        tree += b'\0'
    tree += b'\0'
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIIIIII', 0x55AA1234, 2, len(tree), len(body), 0, 0, 0))
        f.write(tree)
        f.write(body)


def bridge_log(game):
    """The bridge's own log of its last session (hooks, capture setup), or ''."""
    try:
        with open(os.path.join(os.path.dirname(bridge_path(game)), 'dlr_bridge.log'), encoding='utf-8',
                  errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def _bridge_views():
    """Main views the bridge has seen, or None while it isn't up."""
    from .gamerender import Bridge
    bridge = Bridge()
    try:
        status = bridge.status()
        return status['views'] if status['state'] == 1 else None
    except OSError:
        return None
    finally:
        bridge.close()


def _steam_pipe_line(args):
    """The line `steam <args>` hands a running Steam through ~/.steam/steam.pipe (as Steam's log shows it)."""
    quote = lambda a: "'" + a.replace("'", "'\\''") + "'"
    return ' '.join(quote(a) for a in ['~/.steam/root/ubuntu12_32/steam', *args]) + '\n'


def launch_steam(args):
    if os.name == 'nt':
        root = steam_root()
        exe = os.path.join(root, 'steam.exe') if root else None
        if not exe or not os.path.exists(exe):
            raise RuntimeError('Steam not found')
        subprocess.Popen([exe, *args], creationflags=0x00000008 | 0x00000200)  # DETACHED, new process group
        return
    if sys.platform == 'darwin':
        raise RuntimeError('Deadlock does not run on macOS')
    steam = shutil.which('steam')
    if steam:
        subprocess.Popen([steam, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return
    # Steam's own Blender runs in a container without the steam command; the running client listens on its pipe.
    pipe = os.path.expanduser('~/.steam/steam.pipe')
    try:
        fd = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        raise RuntimeError('Steam is not running (start Steam first)')
    try:
        os.write(fd, _steam_pipe_line(args).encode())
    finally:
        os.close(fd)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--watch":
    watch(sys.argv[2], sys.argv[3])
