"""Merges a Source2Viewer map export into one mesh per (grid cell, material).

python tools/map_rebatch.py <world.gltf> <out.gltf> [--cell-size 100] [--max-tris 2000000]

The export has ~30,000 objects, which Blender imports and draws slowly. Every node transform (and the rest pose of
the few skinned props) is baked into the vertices; triangles, the material UVs (TEXCOORD_0), vertex colours and
materials are unchanged. The other UV sets and custom attributes, which no material reads, are left out, except the
lightmap UVs: those become _LIGHTMAP_UV (u, v flipped for Blender, 1), so a material can tell lightmapped surfaces
(z = 1) from the rest (z = 0, the attribute is missing). The output's buffers go in <out>_bins/ next to it and its
images keep the export's URIs, made relative to the output, so textures load from where Source2Viewer wrote them.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from urllib.parse import quote, unquote

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
TYPE = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}
TRIANGLES = 4
KEPT = ('NORMAL', 'TANGENT', 'TEXCOORD_0', 'COLOR_0')
LIGHTMAP = '_LIGHTMAP_UV'


class Source:
    def __init__(self, path):
        self.dir = os.path.dirname(os.path.abspath(path))
        self.j = json.load(open(path, encoding='utf-8'))
        if self.j.get('extensionsRequired'):
            sys.exit(f'unsupported glTF extensions: {self.j["extensionsRequired"]}')
        self.buffers = {}

    def buffer(self, i):
        if i not in self.buffers:
            uri = self.j['buffers'][i].get('uri', '')
            if not uri or uri.startswith('data:'):
                sys.exit('needs a .gltf with external .bin buffers')
            self.buffers[i] = np.memmap(os.path.join(self.dir, unquote(uri)), dtype=np.uint8, mode='r')
        return self.buffers[i]

    def read(self, i, rows=None):
        """Accessor i as float32 (or its own integer type for indices/joints), only the given rows."""
        a = self.j['accessors'][i]
        if 'sparse' in a:
            sys.exit('sparse accessors are not supported')
        dtype, width, count = np.dtype(COMPONENT[a['componentType']]), WIDTH[a['type']], a['count']
        if 'bufferView' not in a:
            data = np.zeros((count, width), dtype)
        else:
            view = self.j['bufferViews'][a['bufferView']]
            start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
            size = dtype.itemsize * width
            stride = view.get('byteStride') or size
            raw = np.lib.stride_tricks.as_strided(self.buffer(view['buffer'])[start:], (count, size), (stride, 1))
            data = raw if rows is None else raw[rows]
            data = np.ascontiguousarray(data).view(dtype).reshape(-1, width)
            rows = None
        if rows is not None:
            data = data[rows]
        if a.get('normalized') and dtype.kind in 'iu':
            data = np.maximum(data.astype(np.float32) / np.iinfo(dtype).max, -1.0)
        return data


def local_matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    x, y, z, w = node.get('rotation', (0, 0, 0, 1))
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = rot * np.array(node.get('scale', (1, 1, 1)), float)
    m[:3, 3] = node.get('translation', (0, 0, 0))
    return m


def world_matrices(j):
    worlds = {}
    stack = [(n, np.eye(4)) for n in j['scenes'][j.get('scene', 0)]['nodes']]
    while stack:
        i, parent = stack.pop()
        if i in worlds:
            sys.exit('node graph has a cycle or a shared child')
        worlds[i] = parent @ local_matrix(j['nodes'][i])
        stack += [(c, worlds[i]) for c in j['nodes'][i].get('children', ())]
    return worlds


def rest_skins(src, worlds):
    """Per skin, joint world @ inverse bind: the matrices that pose its vertices in the rest pose."""
    out = []
    for skin in src.j.get('skins', ()):
        joints = skin['joints']
        if 'inverseBindMatrices' in skin:
            inv = src.read(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1).astype(float)
        else:
            inv = np.tile(np.eye(4), (len(joints), 1, 1))
        out.append(np.stack([worlds.get(jn, np.eye(4)) @ inv[k] for k, jn in enumerate(joints)]))
    return out


class Piece:
    """One primitive placed in the world; holds only what grouping needs, the vertices are re-read on writing."""
    __slots__ = ('prim', 'world', 'skin', 'tris', 'verts', 'attrs')

    def __init__(self, prim, world, skin, tris, verts, attrs):
        self.prim, self.world, self.skin, self.tris, self.verts = prim, world, skin, tris, verts
        self.attrs = attrs


def lightmap_uv(src, prim):
    """The attribute holding the primitive's lightmap UVs, or None.

    Source2Viewer writes one buffer view per attribute, in order, for each vertex buffer of a mesh, and only later,
    while building a lightmapped draw call, the copy of its lightmap UVs scaled by the world's lightmap UV scale. So
    that copy is the one TEXCOORD_n whose buffer view is not in a run with the other attributes of its vertex buffer.
    Tied to how the pinned Source2Viewer writes glTF."""
    at = prim['attributes']
    views = {name: src.j['accessors'][acc].get('bufferView', -1) for name, acc in at.items()}
    taken = set(views.values())
    lo = hi = views['POSITION']
    while lo - 1 in taken:
        lo -= 1
    while hi + 1 in taken:
        hi += 1
    outside = [name for name, view in views.items() if not lo <= view <= hi]
    if any(not name.startswith(('TEXCOORD_', '_')) for name in outside):
        return None
    uvs = [name for name in outside if name.startswith('TEXCOORD_')]
    return uvs[0] if len(uvs) == 1 else None


def indices(src, prim):
    if 'indices' in prim:
        idx = src.read(prim['indices']).reshape(-1)
    else:
        idx = np.arange(src.j['accessors'][prim['attributes']['POSITION']]['count'], dtype=np.uint32)
    return idx[:len(idx) // 3 * 3]


def vertex_matrices(src, prim, used, skin_mats, fallback):
    at = prim['attributes']
    joints = np.clip(src.read(at['JOINTS_0'], used).astype(np.int64), 0, len(skin_mats) - 1)
    weights = src.read(at['WEIGHTS_0'], used).astype(float)
    total = weights.sum(1, keepdims=True)
    weights = weights / np.where(total > 0, total, 1)
    mats = np.einsum('nk,nkij->nij', weights, skin_mats[joints])
    mats[total[:, 0] <= 0] = fallback
    return mats


def place(src, prim, world, skin, skin_mats):
    """World positions of the vertices the primitive's indices use, the per-vertex (or one) matrix, and the
    remapped indices."""
    idx = indices(src, prim)
    used, remap = np.unique(idx, return_inverse=True)
    pos = src.read(prim['attributes']['POSITION'], used).astype(float)
    if skin is not None:
        mats = vertex_matrices(src, prim, used, skin_mats[skin], world)
        return np.einsum('nij,nj->ni', mats[:, :3, :3], pos) + mats[:, :3, 3], mats, used, remap
    return pos @ world[:3, :3].T + world[:3, 3], None, used, remap


def scan(src, cell_size, worlds, skin_mats):
    groups = defaultdict(list)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    skipped = 0
    for i, world in worlds.items():
        node = src.j['nodes'][i]
        if 'mesh' not in node:
            continue
        for prim in src.j['meshes'][node['mesh']]['primitives']:
            if prim.get('mode', TRIANGLES) != TRIANGLES or 'POSITION' not in prim['attributes']:
                skipped += 1
                continue
            at = prim['attributes']
            skin = node.get('skin') if 'JOINTS_0' in at and 'WEIGHTS_0' in at else None
            pos, _mats, used, remap = place(src, prim, world, skin, skin_mats)
            if not len(remap):
                continue
            a, b = pos.min(0), pos.max(0)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
            centre = (a + b) / 2
            # glTF is Y up: the grid is on X and Z
            cell = (math.floor(centre[0] / cell_size), math.floor(centre[2] / cell_size))
            attrs = {name: at[name] for name in KEPT if name in at}
            lm = lightmap_uv(src, prim)
            if lm:
                attrs[LIGHTMAP] = at[lm]
            groups[cell, prim.get('material', -1), bool(lm)].append(
                Piece(prim, world, skin, len(remap) // 3, len(used), attrs))
    return groups, lo, hi, skipped


def split(pieces, max_tris):
    parts, part, tris = [], [], 0
    for p in pieces:
        if part and tris + p.tris > max_tris:
            parts.append(part)
            part, tris = [], 0
        part.append(p)
        tris += p.tris
    return parts + [part] if part else parts


DEFAULTS = {'NORMAL': (0, 1, 0), 'TANGENT': (1, 0, 0, 1)}


def default(name, width):
    if name in DEFAULTS:
        return DEFAULTS[name]
    return (1,) * width if 'COLOR' in name else (0,) * width


def merge(src, part, skin_mats):
    widths = {}
    for p in part:
        for name, acc in p.attrs.items():
            widths[name] = 3 if name == LIGHTMAP else max(widths.get(name, 0), WIDTH[src.j['accessors'][acc]['type']])
    nverts = sum(p.verts for p in part)
    pos = np.empty((nverts, 3), np.float32)
    idx = np.empty(sum(p.tris for p in part) * 3, np.uint32)
    attrs = {name: np.empty((nverts, w), np.float32) for name, w in widths.items()}
    v = t = 0
    for p in part:
        world_pos, mats, used, remap = place(src, p.prim, p.world, p.skin, skin_mats)
        n = len(used)
        pos[v:v + n] = world_pos
        lin = mats[:, :3, :3] if mats is not None else p.world[:3, :3]
        det = np.median(np.linalg.det(lin)) if mats is not None else np.linalg.det(lin)
        tri = remap.reshape(-1, 3).astype(np.uint32)
        if det < 0:
            tri = tri[:, ::-1]
        idx[t:t + tri.size] = tri.reshape(-1) + v
        if mats is None:
            normal_mat = np.linalg.inv(lin).T if abs(det) > 1e-12 else lin

            def xform(vec, m):
                return vec @ m.T
        else:
            normal_mat = lin

            def xform(vec, m):
                return np.einsum('nij,nj->ni', m, vec)
        for name, w in widths.items():
            dest = attrs[name][v:v + n]
            if name not in p.attrs:
                dest[:] = default(name, w)
                continue
            data = src.read(p.attrs[name], used).astype(np.float32)
            if name == LIGHTMAP:
                data = np.column_stack([data[:, 0], 1 - data[:, 1], np.ones(n, np.float32)])
            elif name == 'NORMAL':
                data = xform(data.astype(float), normal_mat)
                data /= np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)
            elif name == 'TANGENT':
                data = data.astype(float)
                data[:, :3] = xform(data[:, :3], lin)
                data[:, :3] /= np.maximum(np.linalg.norm(data[:, :3], axis=1, keepdims=True), 1e-12)
                if det < 0:
                    data[:, 3] *= -1
            if data.shape[1] < w:
                data = np.hstack([data, np.tile(np.float32(default(name, w)[data.shape[1]:]), (n, 1))])
            dest[:] = data
        v += n
        t += tri.size
    return pos, idx, attrs


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)[:60] or 'x'


def cell_name(cell):
    return 'cell_{}_{}'.format(*cell).replace('-', 'm')


def write(src, groups, out_path, max_tris, skin_mats, worlds):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    bins = os.path.splitext(os.path.basename(out_path))[0] + '_bins'
    os.makedirs(os.path.join(out_dir, bins), exist_ok=True)
    j = src.j
    out = {'asset': {'version': '2.0', 'generator': 'deadlockrigs map_rebatch'}, 'scene': 0,
           'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [], 'accessors': [], 'bufferViews': [], 'buffers': []}
    for key in ('materials', 'textures', 'samplers', 'extensions', 'extensionsUsed'):
        if key in j:
            out[key] = j[key]
    rel = os.path.relpath(src.dir, out_dir)
    out['images'] = [dict(img, uri=quote(os.path.join(rel, unquote(img['uri'])).replace(os.sep, '/')))
                     if 'uri' in img and not img['uri'].startswith('data:') else img for img in j.get('images', ())]

    # the sun (light_environment) and any other lights stay as nodes at their world transform
    for i, world in worlds.items():
        node = j['nodes'][i]
        if 'KHR_lights_punctual' in node.get('extensions', {}):
            out['nodes'].append({'name': node.get('name', 'light'), 'matrix': world.T.reshape(-1).tolist(),
                                 'extensions': node['extensions']})
            out['scenes'][0]['nodes'].append(len(out['nodes']) - 1)

    by_cell = defaultdict(list)
    for (cell, mat, lit), pieces in groups.items():
        by_cell[cell].append((mat, lit, pieces))
    materials = j.get('materials', [])
    total = 0
    for n, cell in enumerate(sorted(by_cell), 1):
        name = cell_name(cell)
        out['buffers'].append({'uri': f'{bins}/{name}.bin', 'byteLength': 0})
        buf = len(out['buffers']) - 1
        children, offset, tris = [], 0, 0
        with open(os.path.join(out_dir, bins, name + '.bin'), 'wb') as f:
            def accessor(arr, ctype, target, bounds=False):
                nonlocal offset
                data = np.ascontiguousarray(arr).tobytes()
                f.write(data)
                pad = (-len(data)) % 4
                f.write(b'\0' * pad)
                out['bufferViews'].append({'buffer': buf, 'byteOffset': offset, 'byteLength': len(data),
                                           'target': target})
                offset += len(data) + pad
                acc = {'bufferView': len(out['bufferViews']) - 1, 'componentType': ctype, 'count': len(arr),
                       'type': TYPE[1 if arr.ndim == 1 else arr.shape[1]]}
                if bounds:
                    acc['min'], acc['max'] = arr.min(0).tolist(), arr.max(0).tolist()
                out['accessors'].append(acc)
                return len(out['accessors']) - 1

            for mat, lit, pieces in sorted(by_cell[cell], key=lambda g: g[:2]):
                mat_name = materials[mat].get('name', f'mat{mat}') if 0 <= mat < len(materials) else 'none'
                parts = split(pieces, max_tris)
                for k, part in enumerate(parts):
                    pos, idx, attrs = merge(src, part, skin_mats)
                    prim = {'attributes': {'POSITION': accessor(pos, 5126, 34962, True)}, 'mode': TRIANGLES}
                    for attr, data in attrs.items():
                        prim['attributes'][attr] = accessor(data, 5126, 34962)
                    prim['indices'] = accessor(idx, 5125, 34963)
                    if mat >= 0:
                        prim['material'] = mat
                    label = (f'{name}__{safe_name(mat_name)}' + ('' if lit else '__unlit') +
                             (f'_{k}' if len(parts) > 1 else ''))
                    out['meshes'].append({'name': label, 'primitives': [prim]})
                    out['nodes'].append({'name': label, 'mesh': len(out['meshes']) - 1})
                    children.append(len(out['nodes']) - 1)
                    tris += len(idx) // 3
        out['buffers'][buf]['byteLength'] = offset
        out['nodes'].append({'name': name, 'children': children})
        out['scenes'][0]['nodes'].append(len(out['nodes']) - 1)
        total += tris
        print(f'cell {n}/{len(by_cell)} {name}: {tris:,} tris', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return len(out['meshes']), total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--cell-size', type=float, default=100.0, help='grid cell width in metres (default 100)')
    ap.add_argument('--max-tris', type=int, default=2_000_000, help='split a merged mesh above this many triangles')
    args = ap.parse_args(argv)
    src = Source(args.input)
    worlds = world_matrices(src.j)
    skin_mats = rest_skins(src, worlds)
    groups, lo, hi, skipped = scan(src, args.cell_size, worlds, skin_mats)
    pieces = sum(len(g) for g in groups.values())
    print(f'{pieces} primitives ({skipped} skipped) into {len(groups)} meshes, '
          f'bounds {np.round(lo, 1).tolist()} to {np.round(hi, 1).tolist()}', flush=True)
    meshes, tris = write(src, groups, args.output, args.max_tris, skin_mats, worlds)
    print(f'wrote {args.output}: {meshes} meshes, {tris:,} triangles')
    return 0


if __name__ == '__main__':
    sys.exit(main())
