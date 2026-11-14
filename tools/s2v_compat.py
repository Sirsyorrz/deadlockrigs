"""Source2Viewer 21 writes materials closer to the game: a self-illum tint as emissiveFactor (factor 1 on F_SELF_ILLUM
materials whose mask is a placeholder), F_UNLIT as KHR_materials_unlit and citadel_overlay decals as alpha blended. Blender's importer then lights them itself
(Calico's cat form glowed white) or builds no Principled BSDF, while the rig and map code shade self-illum and unlit
from the vmat extras and expect the materials as older exports wrote them (map_lighting.apply_overlay builds the
decals' opacity from the colour translucency ranges and skips a material whose alpha is already linked). This
takes them back out of an export."""
import json
import os
import struct


def _strip(doc):
    changed = False
    for mat in doc.get('materials', []):
        if mat.pop('emissiveFactor', None) is not None:
            changed = True
        shader = ((mat.get('extras') or {}).get('vmat') or {}).get('ShaderName')
        if shader == 'citadel_overlay.vfx' and mat.pop('alphaMode', None) is not None:
            mat.pop('alphaCutoff', None)
            changed = True
        ext = mat.get('extensions') or {}
        if ext.pop('KHR_materials_unlit', None) is not None:
            changed = True
            if not ext:
                mat.pop('extensions', None)
    for key in ('extensionsUsed', 'extensionsRequired'):
        if 'KHR_materials_unlit' in doc.get(key, []):
            doc[key].remove('KHR_materials_unlit')
            if not doc[key]:
                del doc[key]
            changed = True
    return changed


def strip_materials(path):
    """Rewrites a .gltf or .glb in place; returns whether anything changed."""
    if path.lower().endswith('.gltf'):
        doc = json.load(open(path, encoding='utf-8'))
        if not _strip(doc):
            return False
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, separators=(',', ':'))
        return True
    with open(path, 'rb') as f:
        data = f.read()
    json_len = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + json_len])
    if not _strip(doc):
        return False
    chunk = json.dumps(doc, separators=(',', ':')).encode('utf-8')
    chunk += b' ' * (-len(chunk) % 4)
    rest = data[20 + json_len:]
    out = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(chunk) + len(rest)) + struct.pack('<I4s', len(chunk), b'JSON')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(out + chunk + rest)
    os.replace(tmp, path)
    return True


def strip_tree(folder):
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(('.gltf', '.glb')):
                strip_materials(os.path.join(dirpath, name))
