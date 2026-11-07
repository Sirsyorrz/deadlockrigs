"""Smaller copies of a built map's textures, for renders that would otherwise run out of memory (the add-on's Map
Textures setting points the map's images at them).

blender -b --factory-startup -P tools/map_textures.py -- <map folder> [--jobs N]

Writes <map folder>/textures_<size>/ for each of SIZES, holding the textures larger than that size scaled down to
it (longest side). Textures already that small aren't copied: the add-on falls back to the full one. Lightmaps
aren't touched, so the lighting stays as sharp.
"""
import os
import struct
import subprocess
import sys

SIZES = (1024, 512)


def tier_dir(map_dir, size):
    return os.path.join(map_dir, f'textures_{size}')


def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return struct.unpack('>II', head[16:24])


def wanted(map_dir):
    """The textures needing a smaller copy that doesn't exist yet."""
    src = os.path.join(map_dir, 'textures')
    out = []
    for name in sorted(os.listdir(src)):
        size = png_size(os.path.join(src, name)) if name.lower().endswith('.png') else None
        if size and any(max(size) > s and not os.path.exists(os.path.join(tier_dir(map_dir, s), name))
                        for s in SIZES):
            out.append(name)
    return out


def shrink(map_dir, names):
    """In Blender: writes each texture's smaller copies, scaling the largest first."""
    import bpy
    for s in SIZES:
        os.makedirs(tier_dir(map_dir, s), exist_ok=True)
    for name in names:
        img = bpy.data.images.load(os.path.join(map_dir, 'textures', name), check_existing=False)
        img.colorspace_settings.is_data = True
        img.alpha_mode = 'CHANNEL_PACKED'
        w, h = img.size
        for s in sorted(SIZES, reverse=True):
            if max(w, h) <= s:
                continue
            k = s / max(w, h)
            img.scale(max(1, round(w * k)), max(1, round(h * k)))
            img.save(filepath=os.path.join(tier_dir(map_dir, s), name))
        bpy.data.images.remove(img)


def make_tiers(map_dir, jobs=None, blender=None):
    """Runs shrink over `jobs` Blender processes. Returns how many textures it shrank."""
    names = wanted(map_dir)
    if not names:
        return 0
    jobs = max(1, min(jobs or os.cpu_count() or 4, len(names), 16))
    blender = blender or __import__('bpy').app.binary_path
    lists, procs = [], []
    for i in range(jobs):
        part = names[i::jobs]
        lst = os.path.join(map_dir, f'.shrink_{i}.txt')
        with open(lst, 'w', encoding='utf-8') as f:
            f.write('\n'.join(part))
        lists.append(lst)
        procs.append(subprocess.Popen([blender, '-b', '--factory-startup', '--python-exit-code', '1', '-P',
                                       os.path.abspath(__file__), '--', map_dir, '--list', lst],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    failed = [p.wait() for p in procs]
    for lst in lists:
        os.remove(lst)
    if any(failed):
        raise RuntimeError('shrinking the map textures failed')
    return len(names)


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    map_dir = os.path.abspath(args[0])
    if '--list' in args:
        with open(args[args.index('--list') + 1], encoding='utf-8') as f:
            shrink(map_dir, [n for n in f.read().split('\n') if n])
        return
    jobs = int(args[args.index('--jobs') + 1]) if '--jobs' in args else None
    print(f'shrank {make_tiers(map_dir, jobs)} textures')


if __name__ == '__main__':
    main()
