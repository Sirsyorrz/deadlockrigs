"""The map's reflections (used by map_import.py), as the game's deferred lighting and pbr.vfx do them (decompiled
from game version 6821): box projected cubemaps (env_combined_light_probe_volume, env_cubemap_box) blended by their
edge fade in priority order, sampled in the cubemap's own frame, blurred by sqrt(roughness), and lit by the game's
BRDF lookup table. Callers scale the radiance by the env map normalization, the shaded point's diffuse light over
the luminance the cubemaps saw there, so reflections follow the local light.

Blender has no cube arrays or mips, so images stand in:
  reflection_atlas_<k>.exr  every cubemap as an equirect tile, at three sizes standing in for the mips (box
                            filtered from the top mip, not the game's GGX prefilter); one texel of padding,
                            wrapped across the seam
  reflection_maps.exr       per cubemap a row of PARAM_TEXELS texels, see map_params
  reflection_index.png      a coarse map space grid with up to CANDIDATES cubemaps per cell in blending order;
                            unused slots point at an empty row. Outside the grid the map-wide cubemaps apply
  brdf_lut.exr              the game's textures/dev/brdf_lut.vtex, GGX layer
"""
import glob
import math
import os
import re

import bpy
import numpy as np
from map_lighting import UNIT, Graph
from map_probes import SOURCE_FROM_BLENDER, save_image, tile_slices

GROUP = 'Deadlock reflections'
PARAM_TEXELS = 10
CANDIDATES = 4
MAX_CELLS = 2_000_000
# the game's VRF counterpart widens boxes by this much (Source units)
BOUNDS_EXTEND = 0.02
TOP_MIP = 6
LUMA = (0.2125, 0.7154, 0.0721)
Y00, Y1 = 0.282095, 0.488603


def slice_files(cube_dir):
    files = {}
    for f in glob.glob(os.path.join(cube_dir, '**', '*.exr'), recursive=True):
        m = re.search(r'_f(\d+)\.exr$', f)
        if m:
            files[int(m.group(1))] = f
    return files


def load_pixels(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = 'Non-Color'
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[..., :3]


def equirect_directions(h, w):
    """Source2Viewer's equirect in the cubemap's (Source) axes, as the GPU addresses the cube: +X at the left edge,
    turning toward -Y, rows bottom up as Blender reads them. Directions and texel solid angles."""
    lat = ((np.arange(h) + 0.5) / h - 0.5) * math.pi
    lon = (np.arange(w) + 0.5) / w * 2 * math.pi
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    d = np.stack([np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    return d, np.cos(lat) * (2 * math.pi / w) * (math.pi / h)


def luma_harmonic(rgb):
    """The cubemap's luminance as the L1 harmonic the game dots with (normal, 1): irradiance over pi, in its axes."""
    d, area = equirect_directions(*rgb.shape[:2])
    lum = rgb @ np.array(LUMA)
    l00 = (lum * area).sum() * Y00
    l1 = (lum[..., None] * d * area[..., None]).sum((0, 1)) * Y1
    return l1 * Y1 * 2 / 3, l00 * Y00


def downsample(rgb, w):
    h0, w0 = rgb.shape[:2]
    f = w0 // w
    return rgb.reshape(h0 // f, f, w0 // f, f, 3).mean((1, 3))


def pad(tile):
    """One texel around: wrapped left and right across the seam, repeated at the poles."""
    t = np.concatenate([tile[:, -1:], tile, tile[:, :1]], 1)
    return np.concatenate([t[:1], t, t[-1:]], 0)


def level_widths(count):
    return (512 if count <= 48 else 256, 64, 16)


def build_atlases(slices, widths, cols):
    """slices: equirects in tile order. Returns per level the atlas (rows bottom up)."""
    rows = math.ceil(len(slices) / cols)
    atlases = []
    for w in widths:
        th, tw = w // 2 + 2, w + 2
        out = np.zeros((rows * th, cols * tw, 4), np.float32)
        out[..., 3] = 1
        for k, rgb in enumerate(slices):
            y, x = (k // cols) * th, (k % cols) * tw
            out[y:y + th, x:x + tw, :3] = pad(downsample(rgb, w))
        atlases.append(out)
    return atlases


def map_params(maps, tiles, harmonics, cols):
    """Rows of PARAM_TEXELS RGBA texels, the last an empty map:
      0-2  map space -> the cubemap's local Source units (xyz, translation in w)
      3-5  map directions -> local
      6    box mins, tile column     7  box maxs, tile row
      8    1 / edge fade width       9  luminance harmonic in map axes (xyz) and its constant (w)"""
    out = np.zeros((len(maps) + 1, PARAM_TEXELS, 4), np.float32)
    for i, m in enumerate(maps):
        rot = np.array(m['rotation'], float)
        to_local = rot.T @ SOURCE_FROM_BLENDER
        out[i, 0:3, :3] = to_local / UNIT
        out[i, 0:3, 3] = -rot.T @ np.array(m['origin'], float)
        out[i, 3:6, :3] = to_local
        out[i, 6, :3] = np.array(m['mins']) - BOUNDS_EXTEND
        out[i, 7, :3] = np.array(m['maxs']) + BOUNDS_EXTEND
        out[i, 6, 3], out[i, 7, 3] = tiles[i] % cols, tiles[i] // cols
        out[i, 8, :3] = 1 / (np.maximum(np.array(m['fade']), 0) + BOUNDS_EXTEND)
        l1, l0 = harmonics[tiles[i]]
        out[i, 9, :3] = SOURCE_FROM_BLENDER.T @ (rot @ l1)
        out[i, 9, 3] = l0
    out[-1, 6, :3], out[-1, 7, :3] = 1e9, -1e9
    return out


def box_corners(m):
    local = np.array([[(m['mins'], m['maxs'])[(k >> a) & 1][a] for a in range(3)] for k in range(8)], float)
    return (np.array(m['origin']) + local @ np.array(m['rotation']).T) @ SOURCE_FROM_BLENDER * UNIT


def index_grid(maps):
    """Cells' candidate cubemaps: the ones reaching into the cell in blending order, up to the first whose full
    weight covers all of it; empty slots are len(maps). Map-wide cubemaps (a quarter of everything's bounds or
    more) are left to the outside of the grid, so its cells stay small. Returns (cells z, y, x, CANDIDATES),
    origin, cell size, the outside candidates."""
    corners = [box_corners(m) for m in maps]
    size = np.array([np.prod(np.ptp(c, 0)) for c in corners])
    all_size = np.prod(np.max([c.max(0) for c in corners], 0) - np.min([c.min(0) for c in corners], 0))
    wide = [i for i in range(len(maps)) if size[i] >= all_size / 4][:CANDIDATES]
    local_maps = [i for i in range(len(maps)) if i not in wide] or list(range(len(maps)))
    lo = np.min([corners[i].min(0) for i in local_maps], 0)
    hi = np.max([corners[i].max(0) for i in local_maps], 0)
    cell = max(0.5, (np.prod(hi - lo) / MAX_CELLS) ** (1 / 3))
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    gz, gy, gx = np.meshgrid(*(np.arange(d) for d in dims[::-1]), indexing='ij')
    centres = lo + (np.stack([gx, gy, gz], -1).reshape(-1, 3) + 0.5) * cell
    source = centres @ SOURCE_FROM_BLENDER.T / UNIT
    radius = cell * math.sqrt(3) / 2 / UNIT
    n = len(centres)
    cand = np.full((n, CANDIDATES), len(maps), np.int32)
    count = np.zeros(n, np.int32)
    full = np.zeros(n, bool)
    for i, m in enumerate(maps):
        local = (source - np.array(m['origin'])) @ np.array(m['rotation'])
        mins, maxs = np.array(m['mins']), np.array(m['maxs'])
        below, above = mins - local, local - maxs
        dist = np.linalg.norm(np.maximum(np.maximum(below, above), 0), axis=1)
        reach = (dist <= radius) & ~full & (count < CANDIDATES)
        cand[reach, count[reach]] = i
        count[reach] += 1
        inner = np.maximum(below, above) + np.maximum(np.array(m['fade']), 0)
        full |= reach & np.all(inner <= -radius, axis=1)
    outside = (wide + [len(maps)] * CANDIDATES)[:CANDIDATES]
    return cand.reshape(*dims[::-1], CANDIDATES), lo, cell, outside


def save_half(name, pixels, path):
    h, w = pixels.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(pixels.reshape(-1))
    scene = bpy.context.scene
    s = scene.render.image_settings
    s.file_format, s.color_mode, s.exr_codec, s.color_depth = 'OPEN_EXR', 'RGB', 'ZIP', '16'
    img.save_render(path, scene=scene)
    bpy.data.images.remove(img)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    return img


def brdf_image(lut_dir, light_dir):
    files = sorted(glob.glob(os.path.join(lut_dir, '**', '*_z001.exr'), recursive=True))
    if not files:
        return None
    rgb = load_pixels(files[0])
    return save_half('brdf_lut', np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1),
                     os.path.join(light_dir, 'brdf_lut.exr'))


class Texels:
    def __init__(self, g, img):
        self.g, self.img = g, img
        self.w, self.h = img.size

    def fetch(self, x, y):
        g = self.g
        n = g.node('ShaderNodeTexImage', image=self.img, interpolation='Closest', extension='EXTEND')
        g.put(n.inputs['Vector'], g.combine(g.math('DIVIDE', g.math('ADD', x, 0.5), self.w),
                                            g.math('DIVIDE', g.math('ADD', y, 0.5), self.h), 0.0))
        return n.outputs['Color'], n.outputs['Alpha']


def reflection_group(anchor, atlases, widths, cols, params_img, index_img, index_scale, grid, lut_img):
    """Node group: Normal, Roughness, Specular Color (F0) -> Radiance (the blended cubemaps), Baked Luma (what they
    saw at the normal, for the normalization), Single Scatter and Multi Scatter (the split sum BRDF and its energy
    compensation, which the game lights with the diffuse irradiance)."""
    origin, cell, dims, grid_cols, outside = grid
    tree = bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    for name, kind in (('Normal', 'NodeSocketVector'), ('Roughness', 'NodeSocketFloat'),
                       ('Specular Color', 'NodeSocketColor')):
        tree.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
    for name, kind in (('Radiance', 'NodeSocketColor'), ('Baked Luma', 'NodeSocketFloat'),
                       ('Single Scatter', 'NodeSocketColor'), ('Multi Scatter', 'NodeSocketColor')):
        tree.interface.new_socket(name=name, in_out='OUTPUT', socket_type=kind)
    g = Graph(tree)
    inp, out = g.node('NodeGroupInput'), g.node('NodeGroupOutput')
    pos = g.node('ShaderNodeTexCoord', object=anchor).outputs['Object']
    normal = g.vec('NORMALIZE', inp.outputs['Normal'])
    rough = g.math('MINIMUM', g.math('MAXIMUM', inp.outputs['Roughness'], 0.0), 1.0)
    to_camera = g.node('ShaderNodeNewGeometry').outputs['Incoming']
    a2 = g.math('MULTIPLY', rough, rough)
    smooth = g.math('MAXIMUM', g.math('SUBTRACT', 1.0, a2), 0.0)
    bend = g.math('MULTIPLY', smooth, g.math('ADD', g.math('SQRT', smooth), a2))
    mirror = g.vec('REFLECT', g.vec('SCALE', to_camera, -1.0), normal)
    r = g.vec('NORMALIZE', g.vec('ADD', normal, g.vec('SCALE', g.vec('SUBTRACT', mirror, normal), bend)))
    mip = g.math('MULTIPLY', g.math('SQRT', rough), TOP_MIP)
    mips = [math.log2(256 / (w / 4)) for w in widths]
    params, index = Texels(g, params_img), Texels(g, index_img)

    def clamp(x, lo, hi):
        return g.math('MINIMUM', g.math('MAXIMUM', x, lo), hi)

    def pick(f, a, b):
        return g.math('ADD', g.math('MULTIPLY', f, a), g.math('MULTIPLY', g.math('SUBTRACT', 1.0, f), b))

    gx, gy, _ = dims
    c = g.split(g.vec('SCALE', g.vec('SUBTRACT', pos, tuple(float(x) for x in origin)), 1 / cell))
    cx, cy, cz = (clamp(g.math('FLOOR', c[a]), 0, int(d) - 1) for a, d in zip(range(3), dims))
    col = g.math('MODULO', cz, grid_cols)
    row = g.math('FLOOR', g.math('DIVIDE', g.math('ADD', cz, 0.5), grid_cols))
    color, alpha = index.fetch(g.math('ADD', cx, g.math('MULTIPLY', col, int(gx))),
                               g.math('ADD', cy, g.math('MULTIPLY', row, int(gy))))
    in_grid = None
    for a, d in enumerate(dims):
        ok = g.math('MULTIPLY', g.math('GREATER_THAN', c[a], 0.0), g.math('LESS_THAN', c[a], float(d)))
        in_grid = ok if in_grid is None else g.math('MULTIPLY', in_grid, ok)
    stored = (*g.split(color)[:3], alpha)[:CANDIDATES]
    if index_scale != 1:
        stored = [g.math('FLOOR', g.math('ADD', g.math('MULTIPLY', x, index_scale), 0.5)) for x in stored]
    cand = [pick(in_grid, x, outside[k]) for k, x in enumerate(stored)]

    def rows3(vol, first, vector, offset):
        parts = []
        for k in range(3):
            row_c, row_a = params.fetch(first + k, vol)
            d = g.vec('DOT_PRODUCT', row_c, vector)
            parts.append(g.math('ADD', d, row_a) if offset else d)
        return g.combine(*parts)

    def sample(level, tile_x, tile_y, d):
        img, w = atlases[level], widths[level]
        x, y, z = g.split(d)
        u = g.math('MODULO', g.math('ADD', g.math('DIVIDE', g.math('ARCTAN2', g.math('MULTIPLY', y, -1.0), x), 2 * math.pi),
                                        1.0), 1.0)
        v = g.math('ADD', g.math('DIVIDE', g.math('ARCSINE', clamp(z, -1.0, 1.0)), math.pi), 0.5)
        aw, ah = img.size
        px = g.math('ADD', g.math('MULTIPLY', tile_x, w + 2), g.math('ADD', g.math('MULTIPLY', u, w), 1.0))
        py = g.math('ADD', g.math('MULTIPLY', tile_y, w // 2 + 2), g.math('ADD', g.math('MULTIPLY', v, w // 2), 1.0))
        n = g.node('ShaderNodeTexImage', image=img, interpolation='Linear', extension='EXTEND')
        g.put(n.inputs['Vector'], g.combine(g.math('DIVIDE', px, aw), g.math('DIVIDE', py, ah), 0.0))
        return n.outputs['Color']

    radiance, harmonic, total = None, None, 0.0
    for vol in cand:
        local = rows3(vol, 0, pos, True)
        rl = rows3(vol, 3, r, False)
        mins_c, tile_x = params.fetch(6, vol)
        maxs_c, tile_y = params.fetch(7, vol)
        inv, _ = params.fetch(8, vol)
        sh_c, sh_a = params.fetch(9, vol)
        near_min = g.split(g.vec('MULTIPLY', g.vec('SUBTRACT', local, mins_c), inv))
        near_max = g.split(g.vec('MULTIPLY', g.vec('SUBTRACT', maxs_c, local), inv))
        f = None
        for comp in (*near_min, *near_max):
            comp = clamp(comp, 0.0, 1.0)
            f = comp if f is None else g.math('MINIMUM', f, comp)
        smooth_f = g.math('MULTIPLY', g.math('MULTIPLY', f, f), g.math('SUBTRACT', 3.0, g.math('MULTIPLY', f, 2.0)))
        w = g.math('MULTIPLY', smooth_f, g.math('SUBTRACT', 1.0, total))
        total = g.math('ADD', total, w) if not isinstance(total, float) else w
        # where the ray leaves the box, as seen from the cubemap's origin; Blender's divide gives 0 for x / 0
        rx = g.split(rl)
        safe = g.combine(*(g.math('ADD', a, g.math('MULTIPLY', g.math('SUBTRACT', g.math('MULTIPLY', g.math(
            'GREATER_THAN', a, 0.0), 2.0), 1.0), 1e-6)) for a in rx))
        t_far = g.split(g.vec('MAXIMUM', g.vec('DIVIDE', g.vec('SUBTRACT', maxs_c, local), safe),
                                g.vec('DIVIDE', g.vec('SUBTRACT', mins_c, local), safe)))
        t = g.math('ABSOLUTE', g.math('MINIMUM', t_far[0], g.math('MINIMUM', t_far[1], t_far[2])))
        corrected = g.vec('NORMALIZE', g.vec('ADD', local, g.vec('SCALE', rl, t)))
        d = g.vec('NORMALIZE', g.vec('ADD', corrected, g.vec('SCALE', g.vec('SUBTRACT', rl, corrected), rough)))
        levels = [sample(k, tile_x, tile_y, d) for k in range(len(atlases))]
        color = levels[0]
        for k in range(1, len(levels)):
            blend = clamp(g.math('DIVIDE', g.math('SUBTRACT', mip, mips[k - 1]), mips[k] - mips[k - 1]), 0.0, 1.0)
            color = g.vec('ADD', color, g.vec('SCALE', g.vec('SUBTRACT', levels[k], color), blend))
        part = g.vec('SCALE', color, w)
        radiance = part if radiance is None else g.vec('ADD', radiance, part)
        luma = g.vec('SCALE', g.combine(*g.split(sh_c)), w)
        luma_w = g.math('MULTIPLY', sh_a, w)
        harmonic = (luma, luma_w) if harmonic is None else (g.vec('ADD', harmonic[0], luma),
                                                             g.math('ADD', harmonic[1], luma_w))
    baked = g.math('DIVIDE', g.math('ADD', g.vec('DOT_PRODUCT', harmonic[0], normal), harmonic[1]),
                   g.math('MAXIMUM', total, 1e-4))

    n_dot_v = clamp(g.vec('DOT_PRODUCT', normal, to_camera), 0.0, 1.0)
    lu = g.math('ADD', g.math('MULTIPLY', rough, 0.984375), 0.0078125)
    # the table's first row is N.V = 1; Blender reads the EXR bottom up
    lv = g.math('SUBTRACT', 1.0, g.math('ADD', g.math('MULTIPLY', g.math('SQRT', g.math('SUBTRACT', 1.0, n_dot_v)),
                                                          0.984375), 0.0078125))
    lut = g.node('ShaderNodeTexImage', image=lut_img, interpolation='Linear', extension='EXTEND')
    g.put(lut.inputs['Vector'], g.combine(lu, lv, 0.0))
    lx, ly, _ = g.split(lut.outputs['Color'])
    f0 = inp.outputs['Specular Color']
    single = g.vec('ADD', g.combine(lx, lx, lx), g.vec('SCALE', f0, g.math('SUBTRACT', ly, lx)))
    missing = g.math('SUBTRACT', 1.0, ly)
    average = g.vec('ADD', f0, g.vec('SCALE', g.vec('SUBTRACT', (1.0, 1.0, 1.0), f0), 1 / 21))
    multi = g.vec('SCALE', g.vec('DIVIDE', g.vec('MULTIPLY', single, average),
                                 g.vec('SUBTRACT', (1.0, 1.0, 1.0), g.vec('SCALE', average, missing))), missing)
    g.links.new(radiance, out.inputs['Radiance'])
    g.links.new(baked, out.inputs['Baked Luma'])
    g.links.new(single, out.inputs['Single Scatter'])
    g.links.new(multi, out.inputs['Multi Scatter'])
    tree.use_fake_user = True
    return tree


def add_reflections(coll, title, maps, cube_dir, lut_dir, light_dir):
    """Packs the cubemaps next to the map and makes the group; None when there are none."""
    files = slice_files(cube_dir)
    maps = [m for m in maps if m['index'] in files]
    lut_img = brdf_image(lut_dir, light_dir) if maps else None
    if not maps or lut_img is None:
        return None
    os.makedirs(light_dir, exist_ok=True)
    order = sorted({m['index'] for m in maps})
    tile_of = {s: k for k, s in enumerate(order)}
    slices, harmonics = [], []
    for s in order:
        rgb = load_pixels(files[s])
        slices.append(rgb)
        harmonics.append(luma_harmonic(rgb))
    widths = level_widths(len(slices))
    cols = max(1, math.ceil(math.sqrt(len(slices) / 2)))
    atlases = [save_half(f'reflection_atlas_{k}', a, os.path.join(light_dir, f'reflection_atlas_{k}.exr'))
               for k, a in enumerate(build_atlases(slices, widths, cols))]
    del slices
    params_img = save_image('reflection_maps', map_params(maps, [tile_of[m['index']] for m in maps], harmonics,
                                                             cols), os.path.join(light_dir, 'reflection_maps.exr'))
    grid, origin, cell, outside = index_grid(maps)
    index_scale = 255 if len(maps) + 1 <= 256 else 1
    tiled, grid_cols = tile_slices(grid.astype(np.float32) / index_scale)
    index_img = save_image('reflection_index', tiled, os.path.join(light_dir, 'reflection_index' +
                                                                     ('.png' if index_scale != 1 else '.exr')))
    anchor = bpy.data.objects.get(coll.get('dlr_anchor', ''))
    if anchor is None:
        anchor = bpy.data.objects.new(f'{title} origin', None)
        anchor.empty_display_type = 'PLAIN_AXES'
        coll.objects.link(anchor)
        coll['dlr_anchor'] = anchor.name
    dims = grid.shape[:3][::-1]
    print(f'reflections: {len(maps)} cubemaps, tiles {widths}, index {dims[0]}x{dims[1]}x{dims[2]} cells of '
          f'{cell:.2f} m')
    return reflection_group(anchor, atlases, widths, cols, params_img, index_img, index_scale,
                            (origin, cell, dims, grid_cols, outside), lut_img)
