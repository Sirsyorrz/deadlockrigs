"""Finds Deadlock through Steam and mounts its VPKs: game/citadel and game/core, then the map VPKs.

Files are looked up by their compiled path (``materials/foo.vmat_c``), case insensitively, in mount order.
"""
import os
import re
import threading

from .vpk import VpkArchive

DEADLOCK_APP_ID = 1422450

# Content folders under game/, highest priority first, as in Deadlock's gameinfo.
SEARCH_PATHS = ("citadel", "core")


def _steam_path_from_registry() -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
        try:
            with winreg.OpenKey(hive, key) as k:
                for name in ("SteamPath", "InstallPath"):
                    try:
                        value, _ = winreg.QueryValueEx(k, name)
                        if value and os.path.isdir(value):
                            return os.path.normpath(value)
                    except OSError:
                        pass
        except OSError:
            pass
    return None


def _steam_roots() -> list[str]:
    roots = []
    registry = _steam_path_from_registry()
    if registry:
        roots.append(registry)
    for path in (r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam",
                 os.path.expanduser("~/.local/share/Steam"), os.path.expanduser("~/.steam/steam")):
        if os.path.isdir(path) and os.path.normcase(path) not in map(os.path.normcase, roots):
            roots.append(path)
    return roots


def _library_folders(steam_root: str) -> list[str]:
    folders = [os.path.join(steam_root, "steamapps")]
    vdf = os.path.join(steam_root, "steamapps", "libraryfolders.vdf")
    try:
        with open(vdf, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return folders
    for match in re.finditer(r'"path"\s+"([^"]+)"', text):
        path = os.path.join(match.group(1).replace("\\\\", "\\"), "steamapps")
        if os.path.isdir(path) and os.path.normcase(path) not in map(os.path.normcase, folders):
            folders.append(path)
    return folders


def find_deadlock() -> str | None:
    """The Deadlock install folder (the one holding game/citadel), or None."""
    for root in _steam_roots():
        for library in _library_folders(root):
            manifest = os.path.join(library, f"appmanifest_{DEADLOCK_APP_ID}.acf")
            install_dir = "Deadlock"
            try:
                with open(manifest, "r", encoding="utf-8", errors="replace") as f:
                    found = re.search(r'"installdir"\s+"([^"]+)"', f.read())
                    if found:
                        install_dir = found.group(1)
            except OSError:
                pass
            path = os.path.join(library, "common", install_dir)
            if os.path.isfile(os.path.join(path, "game", "citadel", "pak01_dir.vpk")):
                return path
    return None


def is_deadlock_folder(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "game", "citadel", "pak01_dir.vpk"))


class GameContent:
    """Deadlock's mounted VPKs. Base content wins over map VPKs that bundle a file under the same path, except for
    the files of the map being loaded (see ``prefer_map``)."""

    def __init__(self, install: str):
        self.install = install
        self.game = os.path.join(install, "game")
        self.archives: list[VpkArchive] = []
        self.maps: dict[str, VpkArchive] = {}
        self._files: dict[str, VpkArchive] = {}
        self._preferred: VpkArchive | None = None
        self._lock = threading.Lock()

        for search in SEARCH_PATHS:
            path = os.path.join(self.game, search, "pak01_dir.vpk")
            if os.path.isfile(path):
                self._add(VpkArchive(path))

        maps_dir = os.path.join(self.game, "citadel", "maps")
        if os.path.isdir(maps_dir):
            for name in sorted(os.listdir(maps_dir), key=str.lower):
                if not name.lower().endswith(".vpk"):
                    continue
                try:
                    archive = VpkArchive(os.path.join(maps_dir, name))
                except (OSError, ValueError):
                    continue
                map_name = name[:-4].lower()
                if f"maps/{map_name}/world.vwrld_c" in archive:
                    self.maps[map_name] = archive
                self._add(archive)

    def _add(self, archive: VpkArchive):
        self.archives.append(archive)
        files = self._files
        for path in archive.files:
            if path not in files:
                files[path] = archive

    def prefer_map(self, map_name: str | None):
        """Read files a map VPK bundles from it first: community maps ship their own versions of shared paths."""
        self._preferred = self.maps.get(map_name.lower()) if map_name else None

    def exists(self, path: str) -> bool:
        path = normalize(path)
        if self._preferred is not None and path in self._preferred:
            return True
        return path in self._files

    def read(self, path: str) -> bytes:
        path = normalize(path)
        if self._preferred is not None and path in self._preferred:
            return self._preferred.read(path)
        archive = self._files.get(path)
        if archive is None:
            raise FileNotFoundError(path)
        return archive.read(path)

    def try_read(self, path: str) -> bytes | None:
        try:
            return self.read(path)
        except FileNotFoundError:
            return None

    def files(self, extension: str | None = None):
        if extension is None:
            return list(self._files)
        extension = extension.lower()
        return [p for p in self._files if p.endswith(extension)]

    def close(self):
        for archive in self.archives:
            archive.close()


def normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("/").lower()


def compiled(path: str) -> str:
    """A resource path with its compiled suffix: ``materials/x.vmat`` -> ``materials/x.vmat_c``."""
    path = normalize(path)
    return path if path.endswith("_c") else path + "_c"
