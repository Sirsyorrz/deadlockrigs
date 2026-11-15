"""Start and stop the Deadlock session the game render engine drives: the user's own Steam install, launched with
-insecure (no VAC, local server only) and the camera bridge loaded as the session's server.dll.

The bridge goes in through one extra gameinfo.gi search path that exists only while the game loads: the original
file is restored byte-exact as soon as the game's console is up, so a normal Steam launch never sees it. The game
saves archived cvars (HUD, camera) and its window mode on exit; the user's cfgs are copied before the launch and put
back after the session quits.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

APPID = '1422450'
OVERLAY = 'citadel/dlrigs_bridge'
MARK = '// dlrigs session'
MOD_PATHS = ('citadel/deadworks_mods', 'citadel/addons', 'citadel/deadworks_addons')
HERE = os.path.dirname(os.path.abspath(__file__))
# packaged: native/ inside the add-on; in the repo the build output next to addon/
BRIDGE_DLLS = (os.path.join(HERE, 'native', 'dlr_bridge.dll'),
               os.path.join(os.path.dirname(HERE), 'native', 'dlr_bridge', 'build', 'dlr_bridge.dll'))
# Launch to console: ~60 s, plus a stall of ~95 s that some launches sit through after "Game started".
READY_DRAWING = 8.0  # seconds of steady drawing before the first console client
CONSOLE_TIMEOUT = 300


def state_dir():
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'dlr-game', 'session')


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
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='deadlock.exe'\" | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, creationflags=0x08000000)
        if out.returncode:
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
        out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq deadlock.exe', '/NH'], capture_output=True,
                             text=True, creationflags=0x08000000).stdout
        return 'deadlock.exe' in out.lower()
    return bool(_game_commandlines())


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

    def __init__(self, game, port, size=(1920, 1080), mods=False, extra=(), map_name='dl_hideout', borderless=False):
        self.game = game
        self.port = port
        self.size = size
        self.map_name = map_name
        self.borderless = borderless
        self.mods = mods
        self.extra = list(extra)
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
        out, inserted, in_search = [], False, False
        for line in original.split(nl):
            parts = line.split()
            key = parts[0].lower() if parts else ''
            if key == 'searchpaths':
                in_search = True
            if not in_search:
                out.append(line)
                continue
            if key == 'game' and not inserted:
                indent = line[:len(line) - len(line.lstrip())]
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
        if not os.path.exists(self.backup):
            return
        original = open(self.backup, 'rb').read()
        current = open(self.gameinfo, 'rb').read()
        if current != original:
            if MARK.encode() in current:
                with open(self.gameinfo, 'wb') as f:
                    f.write(original)
            else:
                # Steam verify or a mod manager rewrote it after our edit: theirs wins, keep ours aside.
                shutil.copyfile(self.backup, f'{self.backup}.{int(time.time())}')
        os.remove(self.backup)

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
        index_path = os.path.join(self.cfg_backup, 'index.json')
        if not os.path.exists(index_path):
            return
        index = json.load(open(index_path))
        for path, name in index.items():
            shutil.copy2(os.path.join(self.cfg_backup, os.path.basename(name)), path)
        for path in self._user_cfgs():
            if path not in index:
                os.remove(path)
        shutil.rmtree(self.cfg_backup)

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
        return os.path.join(self.game, *OVERLAY.split('/'), 'bin', 'win64', 'server.dll')

    def remove_bridge(self):
        """The bridge stays inert without -insecure, but it is only on disk while a session runs, so a normal
        (VAC-secured) launch can never find a foreign server.dll even if gameinfo.gi was left patched."""
        try:
            os.remove(self.bridge_path())
        except OSError:
            pass

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
        self.status = 'Starting Deadlock'
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        try:
            start = time.time()
            end = start + CONSOLE_TIMEOUT
            # The bridge answers once the map's server has loaded it, then counts drawn views. A console client
            # that connects while the map is still loading can hang the game for good (twice, at 2 s into the
            # map): wait until it has drawn steadily for a while.
            steady_since, last_views = None, -1
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
