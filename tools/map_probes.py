"""The map's light probes for rigs (used by map_import.py): the node group 'Deadlock probe light' gives the probe
irradiance at the shaded point for its normal, per pixel, the way the game lights characters (SampleProbeAmbientCube in
ValveResourceFormat's lighting.lpv.slang). Map props get theirs baked per vertex instead (map_lighting.bake_probes).

Blender has no 3D textures, so three images stand in for the game's atlas:
  probe_atlas.png    each volume's 6 ambient cube faces, z slices tiled in a grid, shelf packed; 8 bit, see encode
  probe_volumes.exr  per volume a row of PARAM_TEXELS texels: map space -> volume coordinates, probe counts, the
                     normal's rotation into the volume and where its block is in the atlas
  probe_index.png    a coarse map space grid with up to CANDIDATES volumes per cell in the game's priority order;
                     the shader takes the first that holds the point, else the last (the nearest). The grid
                     leaves out the last volume, the map-wide one, so its cells stay small; outside it the
                     point takes that volume
All three are read with Closest interpolation (filtering is done in the nodes): linear filtering and mipmaps would
blend across volume and slice edges.

Positions are relative to the '<map> origin' empty, so an appended map that is moved keeps lighting rigs right.
"""
import math
import os

import bpy
import numpy as np
from map_lighting import UNIT, Graph

GROUP = 'Deadlock probe light'
PARAM_TEXELS = 8
CANDIDATES = 4
MAX_CELLS = 4_000_000
MAX_TEXTURE = 16384
# the atlas is 8 bit: RGB over its largest channel, which the alpha holds as log2 in [LOG_MIN, LOG_MAX]
LOG_MIN, LOG_MAX = -14.0, 6.0
FACES = 6
# Source = SOURCE_FROM_BLENDER @ Blender, the inverse of the glTF import's (x, y, z) -> (y, -x, z)
SOURCE_FROM_BLENDER = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


def shelf_pack(shapes, width):
    """Rows of blocks, tallest first; block positions and the total height."""
    pos = [None] * len(shapes)
    x = y = shelf = 0
    for i in sorted(range(len(shapes)), key=lambda i: -shapes[i][1]):
        w, h = shapes[i][:2]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        pos[i] = (x, y)
        x += w
        shelf = max(shelf, h)
    return pos, y + shelf


def layout(volumes):
    """Each volume's block: z slices in `cols` columns per face, the 6 faces stacked. Tries a few block shapes and
    atlas widths and keeps the smallest packing within MAX_TEXTURE. Returns the blocks (x, y, cols, face height)
    and the atlas size."""
    best = None
    for k in (0.25, 0.5, 1, 2, 4, math.inf):
        shapes = []
        for v in volumes:
            sx, sy, sz = v['size']
            cols = max(1, min(sz, round(math.sqrt(FACES * sz * sy / sx) * k) if k != math.inf else sz))
            face_h = math.ceil(sz / cols) * sy
            shapes.append((cols * sx, FACES * face_h, cols, face_h))
        widest = max(w for w, _, _, _ in shapes)
        for width in range(widest, MAX_TEXTURE + 1, 256):
            pos, height = shelf_pack(shapes, width)
            if height <= MAX_TEXTURE and (best is None or width * height < best[0]):
                best = (width * height, [(x, y, c, f) for (x, y), (_, _, c, f) in zip(pos, shapes)], (width, height))
    if best is None:
        raise ValueError('the light probes do not fit in one texture')
    return best[1], best[2]


def pack_atlas(atlas, volumes, blocks, size):
    face_depth = atlas.shape[0] // FACES
    out = np.zeros((size[1], size[0], 3), np.float32)
    for v, (bx, by, cols, face_h) in zip(volumes, blocks):
        sx, sy, sz = v['size']
        ox, oy, oz = v['offset']
        for face in range(FACES):
            for k in range(sz):
                x = bx + (k % cols) * sx
                y = by + face * face_h + (k // cols) * sy
                out[y:y + sy, x:x + sx] = atlas[face * face_depth + oz + k, oy:oy + sy, ox:ox + sx]
    return out


def encode(rgb):
    """HDR RGB -> 8 bit RGBA, the largest channel's log2 in alpha, rounded up so RGB stays in range."""
    peak = rgb.max(-1)
    level = np.ceil((np.log2(np.maximum(peak, 2 ** LOG_MIN)) - LOG_MIN) / (LOG_MAX - LOG_MIN) * 255)
    level = np.clip(level, 0, 255)
    scale = 2 ** (LOG_MIN + level / 255 * (LOG_MAX - LOG_MIN))
    out = np.empty(rgb.shape[:2] + (4,), np.float32)
    out[..., :3] = np.round(np.clip(rgb / scale[..., None], 0, 1) * 255)
    out[..., 3] = level
    return out / 255


def volume_params(volumes, blocks):
    """Rows of PARAM_TEXELS RGBA texels; see the module docstring."""
    out = np.zeros((len(volumes), PARAM_TEXELS, 4), np.float32)
    for i, (v, (bx, by, cols, face_h)) in enumerate(zip(volumes, blocks)):
        rot_t = np.array(v['rotation'], float).T
        mins, maxs = np.array(v['mins'], float), np.array(v['maxs'], float)
        inv_ext = 1 / (maxs - mins)
        # local = rot^T (p_source - origin); normalized = (local - mins) / extent
        m = inv_ext[:, None] * (rot_t @ SOURCE_FROM_BLENDER) / UNIT
        c = -inv_ext * (rot_t @ np.array(v['origin'], float) + mins)
        out[i, 0:3, :3], out[i, 0:3, 3] = m, c
        out[i, 3] = (*v['size'], cols)
        out[i, 4:7, :3] = rot_t @ SOURCE_FROM_BLENDER
        out[i, 7] = (bx, by, face_h, 0)
    return out


def volume_corners(v):
    mins, maxs = v['mins'], v['maxs']
    local = np.array([[(mins, maxs)[(k >> a) & 1][a] for a in range(3)] for k in range(8)], float)
    source = np.array(v['origin'], float) + local @ np.array(v['rotation'], float).T
    return source @ SOURCE_FROM_BLENDER * UNIT


def index_grid(volumes):
    """The candidate volumes of each grid cell: the ones reaching into it in priority order, up to the first that
    holds all of it, then the nearest if none does. Returns (cells z, y, x, CANDIDATES), origin, cell size."""
    corners = np.concatenate([volume_corners(v) for v in volumes[:-1] or volumes])
    lo, hi = corners.min(0), corners.max(0)
    cell = max(1.0, (np.prod(hi - lo) / MAX_CELLS) ** (1 / 3))
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    gz, gy, gx = np.meshgrid(*(np.arange(d) for d in dims[::-1]), indexing='ij')
    centres = lo + (np.stack([gx, gy, gz], -1).reshape(-1, 3) + 0.5) * cell
    source = centres @ SOURCE_FROM_BLENDER.T / UNIT
    radius = cell * math.sqrt(3) / 2 / UNIT
    n = len(centres)
    cand = np.full((n, CANDIDATES), -1, np.int32)
    count = np.zeros(n, np.int32)
    full = np.zeros(n, bool)
    nearest = np.zeros(n, np.int32)
    nearest_dist = np.full(n, np.inf, np.float32)
    for i, v in enumerate(volumes):
        local = (source - np.array(v['origin'])) @ np.array(v['rotation'])
        below, above = np.array(v['mins']) - local, local - np.array(v['maxs'])
        dist = np.linalg.norm(np.maximum(np.maximum(below, above), 0), axis=1)
        closer = dist < nearest_dist
        nearest[closer], nearest_dist[closer] = i, dist[closer]
        reach = (dist <= radius) & ~full & (count < CANDIDATES)
        cand[reach, count[reach]] = i
        count[reach] += 1
        full |= reach & np.all(np.maximum(below, above) <= -radius, axis=1)
    # outside every candidate the game takes the nearest volume
    add = ~full & (count < CANDIDATES) & ~np.any(cand == nearest[:, None], axis=1)
    cand[add, count[add]] = nearest[add]
    count[add] += 1
    for k in range(1, CANDIDATES):
        pad = count <= k
        cand[pad, k] = cand[pad, k - 1]
    return cand.reshape(*dims[::-1], CANDIDATES), lo, cell


def tile_slices(grid):
    """(z, y, x, c) -> 2D image with the z slices in a grid; returns it and the column count."""
    dz, dy, dx = grid.shape[:3]
    cols = max(1, math.ceil(math.sqrt(dz * dy / dx)))
    rows = math.ceil(dz / cols)
    out = np.zeros((rows * dy, cols * dx, 4), np.float32)
    out[..., 3] = 1
    for z in range(dz):
        y, x = (z // cols) * dy, (z % cols) * dx
        out[y:y + dy, x:x + dx, :grid.shape[3]] = grid[z]
    return out, cols


def save_image(name, pixels, path):
    """Rows of `pixels` go bottom up, as Blender's do. 8 bit PNG for .png paths, else 32 bit float EXR."""
    h, w = pixels.shape[:2]
    exr = not path.endswith('.png')
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=exr)
    # a generated image is colour managed on save unless it is Non-Color
    img.colorspace_settings.name = 'Non-Color'
    img.alpha_mode = 'CHANNEL_PACKED'
    img.pixels.foreach_set(pixels.reshape(-1))
    if exr:
        scene = bpy.context.scene
        settings = scene.render.image_settings
        settings.file_format, settings.color_mode, settings.exr_codec, settings.color_depth = \
            'OPEN_EXR', 'RGBA', 'ZIP', '32'
        # Image.save writes EXRs uncompressed whatever the settings
        img.save_render(path, scene=scene)
    else:
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    # the alpha channel holds data; straight alpha would divide the colour by it
    img.alpha_mode = 'CHANNEL_PACKED'
    img.use_half_precision = False
    return img


class Texels:
    """Closest lookups of whole texels in an image."""

    def __init__(self, g, img):
        self.g, self.img = g, img
        self.w, self.h = img.size

    def fetch(self, x, y):
        g = self.g
        u = g.math('DIVIDE', g.math('ADD', x, 0.5), self.w)
        v = g.math('DIVIDE', g.math('ADD', y, 0.5), self.h)
        n = g.node('ShaderNodeTexImage', image=self.img, interpolation='Closest', extension='EXTEND')
        g.put(n.inputs['Vector'], g.combine(u, v, 0.0))
        return n.outputs['Color'], n.outputs['Alpha']


def probe_group(anchor, atlas_img, params_img, index_img, index_scale, grid_origin, cell, grid_dims, grid_cols,
                len_volumes):
    """Node group: Normal (world) -> Irradiance at the shaded point."""
    tree = bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    tree.interface.new_socket(name='Normal', in_out='INPUT', socket_type='NodeSocketVector')
    tree.interface.new_socket(name='Irradiance', in_out='OUTPUT', socket_type='NodeSocketColor')
    g = Graph(tree)
    inp, out = g.node('NodeGroupInput'), g.node('NodeGroupOutput')
    coords = g.node('ShaderNodeTexCoord', object=anchor)
    pos = coords.outputs['Object']
    normal = g.vec('NORMALIZE', inp.outputs['Normal'])
    atlas, params, index = Texels(g, atlas_img), Texels(g, params_img), Texels(g, index_img)

    def clamp(x, lo, hi):
        return g.math('MINIMUM', g.math('MAXIMUM', x, lo), hi)

    def pick(f, a, b):
        return g.math('ADD', g.math('MULTIPLY', f, a), g.math('MULTIPLY', g.math('SUBTRACT', 1.0, f), b))

    def tile(k, cols):
        """Column and row of slice k in a grid of `cols` columns (k is a whole number)."""
        return g.math('MODULO', k, cols), g.math('FLOOR', g.math('DIVIDE', g.math('ADD', k, 0.5), cols))

    gx, gy, _ = grid_dims
    c = g.split(g.vec('SCALE', g.vec('SUBTRACT', pos, tuple(float(x) for x in grid_origin)), 1 / cell))
    cx, cy, cz = (clamp(g.math('FLOOR', c[a]), 0, int(d) - 1) for a, d in zip(range(3), grid_dims))
    col, row_ = tile(cz, grid_cols)
    ix = g.math('ADD', cx, g.math('MULTIPLY', col, int(gx)))
    iy = g.math('ADD', cy, g.math('MULTIPLY', row_, int(gy)))
    color, alpha = index.fetch(ix, iy)
    in_grid = None
    for a, d in enumerate(grid_dims):
        ok = g.math('MULTIPLY', g.math('GREATER_THAN', c[a], 0.0), g.math('LESS_THAN', c[a], float(d)))
        in_grid = ok if in_grid is None else g.math('MULTIPLY', in_grid, ok)
    stored = (*g.split(color)[:3], alpha)[:CANDIDATES]
    if index_scale != 1:
        stored = [g.math('FLOOR', g.math('ADD', g.math('MULTIPLY', x, index_scale), 0.5)) for x in stored]
    cand = [pick(in_grid, x, len_volumes - 1) for x in stored]

    def row(vol, texel):
        return params.fetch(texel, vol)

    def affine(vol, first, vector, offset=True):
        parts = []
        for k in range(3):
            color, alpha = row(vol, first + k)
            d = g.vec('DOT_PRODUCT', color, vector)
            parts.append(g.math('ADD', d, alpha) if offset else d)
        return parts

    def inside(norm):
        m = None
        for x in norm:
            edge = g.math('MINIMUM', x, g.math('SUBTRACT', 1.0, x))
            m = edge if m is None else g.math('MINIMUM', m, edge)
        return g.math('GREATER_THAN', m, -1e-4)

    # first candidate holding the point, else the last
    vol = cand[CANDIDATES - 1]
    norm = affine(vol, 0, pos)
    for k in reversed(range(CANDIDATES - 1)):
        nk = affine(cand[k], 0, pos)
        f = inside(nk)
        vol = pick(f, cand[k], vol)
        norm = [pick(f, a, b) for a, b in zip(nk, norm)]

    size_c, cols = row(vol, 3)
    size = g.split(size_c)
    nl = affine(vol, 4, normal, offset=False)
    block_c, _ = row(vol, 7)
    block = g.split(block_c)
    bx, by, face_h = block[0], block[1], block[2]

    # texel centre coordinates, kept half a probe in from the edges as the game does
    t = [clamp(g.math('SUBTRACT', g.math('MULTIPLY', norm[a], size[a]), 0.5), 0.0,
               g.math('SUBTRACT', size[a], 1.0)) for a in range(3)]
    lo = [g.math('FLOOR', x) for x in t]
    frac = [g.math('SUBTRACT', x, f) for x, f in zip(t, lo)]
    hi = [g.math('MINIMUM', g.math('ADD', f, 1.0), g.math('SUBTRACT', size[a], 1.0)) for a, f in enumerate(lo)]
    sx, sy = size[0], size[1]
    slices = []
    for zk in (lo[2], hi[2]):
        col, row_ = tile(zk, cols)
        sx0 = g.math('ADD', bx, g.math('MULTIPLY', col, sx))
        sy0 = g.math('ADD', by, g.math('MULTIPLY', row_, sy))
        slices.append((sx0, sy0))

    def lerp(a, b, f):
        return g.mix('MIX', a, b, f)

    def probe(x, y):
        color, level = atlas.fetch(x, y)
        scale = g.math('POWER', 2.0, g.math('ADD', g.math('MULTIPLY', level, LOG_MAX - LOG_MIN), LOG_MIN))
        return g.vec('SCALE', color, scale)

    total = None
    for axis in range(3):
        positive = g.math('GREATER_THAN', nl[axis], 0.0)
        face = g.math('ADD', axis, g.math('MULTIPLY', g.math('SUBTRACT', 1.0, positive), 3))
        face_y = g.math('MULTIPLY', face, face_h)
        planes = []
        for sx0, sy0 in slices:
            ybase = g.math('ADD', sy0, face_y)
            rows_ = []
            for yk in (lo[1], hi[1]):
                y = g.math('ADD', ybase, yk)
                a = probe(g.math('ADD', sx0, lo[0]), y)
                b = probe(g.math('ADD', sx0, hi[0]), y)
                rows_.append(lerp(a, b, frac[0]))
            planes.append(lerp(rows_[0], rows_[1], frac[1]))
        sample = lerp(planes[0], planes[1], frac[2])
        weighted = g.vec('SCALE', sample, g.math('MULTIPLY', nl[axis], nl[axis]))
        total = weighted if total is None else g.vec('ADD', total, weighted)
    g.links.new(total, out.inputs['Irradiance'])
    tree.use_fake_user = True
    return tree


def add_probe_light(coll, title, atlas, volumes, light_dir):
    """Writes the probe images next to the map and makes the group; None without volumes."""
    if not volumes or atlas is None:
        return None
    blocks, size = layout(volumes)
    os.makedirs(light_dir, exist_ok=True)
    atlas_img = save_image('probe_atlas', encode(pack_atlas(atlas, volumes, blocks, size)),
                           os.path.join(light_dir, 'probe_atlas.png'))
    params_img = save_image('probe_volumes', volume_params(volumes, blocks),
                            os.path.join(light_dir, 'probe_volumes.exr'))
    grid, origin, cell = index_grid(volumes)
    # 8 bit while the volume numbers fit
    index_scale = 255 if len(volumes) <= 256 else 1
    tiled, cols = tile_slices(grid.astype(np.float32) / index_scale)
    index_img = save_image('probe_index', tiled, os.path.join(light_dir, 'probe_index' +
                                                              ('.png' if index_scale != 1 else '.exr')))
    anchor = bpy.data.objects.new(f'{title} origin', None)
    anchor.empty_display_type = 'PLAIN_AXES'
    coll.objects.link(anchor)
    coll['dlr_anchor'] = anchor.name
    dims = grid.shape[:3][::-1]
    print(f'rig light: {len(volumes)} probe volumes, atlas {size[0]}x{size[1]}, '
          f'index {dims[0]}x{dims[1]}x{dims[2]} cells of {cell:.2f} m')
    return probe_group(anchor, atlas_img, params_img, index_img, index_scale, origin, cell, dims, cols,
                       len(volumes))
