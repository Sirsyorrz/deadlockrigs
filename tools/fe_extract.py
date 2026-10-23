"""Source 2 vertex cloth (the model's FeModel, as printed by Source2Viewer-CLI -b PHYS) -> json for rig.py and the
cloth bake. Only $cloth_* nodes (cloth painted on a mesh, Abrams' coat) and the nodes their rods tie them to are
kept; the bone chains in the same FeModel are the ClothChains vmdl_extract.py reads.

python fe_extract.py phys.txt cloth.json

Source units are inches (x forward, y left, z up); positions come out in the glTF import's metres (+x the hero's
left, -y forward, +z up)."""
import json
import re
import sys

IN = 0.0254
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[\[\]{}=,]|[^\s\[\]{}=,]+')


def parse(text):
    toks = TOKEN.findall(text)
    i = 0

    def value():
        nonlocal i
        t = toks[i]
        i += 1
        if t == '{':
            obj = {}
            while toks[i] != '}':
                k = toks[i]
                i += 2
                obj[k] = value()
            i += 1
            return obj
        if t == '[':
            arr = []
            while toks[i] != ']':
                if toks[i] == ',':
                    i += 1
                    continue
                arr.append(value())
            i += 1
            return arr
        if t.startswith('"'):
            return t[1:-1]
        try:
            return float(t) if any(c in t for c in '.eE') else int(t)
        except ValueError:
            return t
    return value()


def fe_block(text):
    start = text.index('m_pFeModel =')
    body = text[text.index('{', start):]
    depth = 0
    for j, c in enumerate(body):
        depth += c == '{'
        depth -= c == '}'
        if depth == 0:
            return parse(body[:j + 1])
    raise ValueError('unbalanced m_pFeModel')


def to_blender(p):
    return [p[1] * IN, -p[0] * IN, p[2] * IN]


def follow_weights(fe, names):
    """Where a node goes without simulation: its parent bone, then lerped towards each soft parent in turn (the
    game's m_CtrlSoftOffsets, alpha being the share kept), which comes down to a weighted blend of bones."""
    base = {e['nCtrlChild']: e['nCtrlParent'] for e in fe.get('m_CtrlOffsets', [])}
    soft = {}
    for e in fe.get('m_CtrlSoftOffsets', []):
        soft.setdefault(e['nCtrlChild'], []).append((e['nCtrlParent'], e['flAlpha']))
    out = {}
    for k in set(base) | set(soft):
        w = {}
        if k in base:
            w[names[base[k]]] = 1.0
        for p, a in soft.get(k, []):
            w = {b: v * a for b, v in w.items()}
            w[names[p]] = w.get(names[p], 0.0) + 1 - a
        out[k] = w
    return out


def extract(fe):
    n = fe['m_nNodeCount']
    names = fe['m_CtrlName']
    inv = fe['m_NodeInvMasses']
    integ = fe['m_NodeIntegrator']
    # per dynamic node: indexed by node minus the static node count
    radii = [0.0] * fe.get('m_nStaticNodes', 0) + (fe.get('m_NodeCollisionRadii') or [])
    pose = [to_blender(p[:3]) for p in fe['m_InitPose'][:n]]
    follow = follow_weights(fe, names)
    cloth = {k for k in range(n) if names[k].startswith('$cloth')}
    rods = [r for r in fe.get('m_Rods', []) if r['nNode'][0] in cloth or r['nNode'][1] in cloth]
    keep = sorted(cloth | {k for r in rods for k in r['nNode']})
    remap = {k: i for i, k in enumerate(keep)}
    nodes = []
    for k in keep:
        it = integ[k] if k < len(integ) else {}
        nodes.append({
            'name': names[k],
            'free': inv[k] > 0.0,
            'rest': [round(x, 6) for x in pose[k]],
            'follow': {b: round(w, 5) for b, w in follow.get(k, {names[k]: 1.0}).items() if w > 1e-4},
            'damping': it.get('flPointDamping', 0.0),
            'goal': it.get('flAnimationVertexAttraction', 0.0),
            # the share of the animation's own motion the node takes along (goal strength cubed)
            'force': it.get('flAnimationForceAttraction', 0.0),
            'gravity': it.get('flGravity', 360.0) / 360.0,
            'radius': round(radii[k] * IN, 5) if k < len(radii) else 0.0,
        })
    out_rods = [[remap[r['nNode'][0]], remap[r['nNode'][1]], round(r['flMinDist'] * IN, 6),
                 round(r['flMaxDist'] * IN, 6)] for r in rods]
    return {'nodes': nodes, 'rods': out_rods}


if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    fe = fe_block(text) if 'm_pFeModel =' in text else {}
    out = extract(fe) if 'm_nNodeCount' in fe else {'nodes': [], 'rods': []}
    out['nodes'] = out['nodes'] if any(nd['name'].startswith('$cloth') for nd in out['nodes']) else []
    out['rods'] = out['rods'] if out['nodes'] else []
    json.dump(out, open(sys.argv[2], 'w'))
    free = sum(nd['free'] for nd in out['nodes'])
    print(f'vertex cloth: {len(out["nodes"])} nodes ({free} free), {len(out["rods"])} rods')
