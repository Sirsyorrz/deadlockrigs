"""The mounted game content, opened once per Blender session."""
from .core import content as contentlib

_content = None
_install = None
# Set by a host add-on (Deadlock Rigs) that has its own Deadlock folder setting and cache folder.
install_override = None
cache_override = None


def install_path() -> str | None:
    if install_override and contentlib.is_deadlock_folder(install_override):
        return install_override
    try:
        from .preferences import get_prefs
        prefs = get_prefs()
    except Exception:
        prefs = None
    if prefs and prefs.game_path and contentlib.is_deadlock_folder(prefs.game_path):
        return prefs.game_path
    return contentlib.find_deadlock()


def content() -> contentlib.GameContent:
    global _content, _install
    install = install_path()
    if install is None:
        raise RuntimeError("Deadlock was not found. Set its folder in the add-on's preferences.")
    if _content is None or _install != install:
        if _content is not None:
            _content.close()
        _content = contentlib.GameContent(install)
        _install = install
    return _content


def reset():
    global _content, _install
    if _content is not None:
        _content.close()
    _content = None
    _install = None
