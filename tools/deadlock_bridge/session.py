"""The mounted game content, opened once per Blender session."""
from .core import content as contentlib

_content = None
_install = None


def install_path() -> str | None:
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
