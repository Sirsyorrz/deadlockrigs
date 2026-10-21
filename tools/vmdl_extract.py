"""python3 vmdl_extract.py artist.vmdl out.json
Pulls cloth chains, cloth collision capsules and anim constraints out of a decompiled (KV3 text) vmdl."""
import json
import re
import sys

TOKEN = re.compile(r'\s*((?://[^\n]*|<!--.*?-->|"(?:[^"\\]|\\.)*"|[{}\[\]=,]|[^\s{}\[\]=,]+))', re.S)


def tokens(text):
    pos = 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            break
        pos = m.end()
        t = m.group(1)
        if t.startswith('//') or t.startswith('<!--'):
            continue
        yield t


def parse(text):
    toks = list(tokens(text))
    i = 0

    def value():
        nonlocal i
        t = toks[i]
        i += 1
        if t == '{':
            obj = {}
            while toks[i] != '}':
                k = toks[i].strip('"')
                assert toks[i + 1] == '=', (k, toks[i + 1])
                i += 2
                obj[k] = value()
            i += 1
            return obj
        if t == '[':
            arr = []
            while toks[i] != ']':
                arr.append(value())
                if toks[i] == ',':
                    i += 1
            i += 1
            return arr
        if t.startswith('"'):
            return t[1:-1]
        if t in ('true', 'false'):
            return t == 'true'
        if t == 'null':
            return None
        try:
            return float(t) if any(c in t for c in '.eE') else int(t)
        except ValueError:
            return t

    return value()


def walk(node, out):
    if isinstance(node, dict):
        if '_class' in node:
            keep_children = node['_class'].startswith('AnimConstraint')
            out.setdefault(node['_class'], []).append({k: v for k, v in node.items() if keep_children or k != 'children'})
        for v in node.values():
            walk(v, out)
    elif isinstance(node, list):
        for v in node:
            walk(v, out)


text = open(sys.argv[1]).read()
text = text[text.index('-->') + 3:]
found = {}
walk(parse(text), found)
keep = ['ClothParams', 'ClothChain', 'ClothShapeCapsule', 'ClothShapeSphere', 'AnimConstraintTiltTwist',
        'AnimConstraintAim', 'AnimConstraintOrient', 'AnimConstraintSlave', 'AnimConstraintBoneInput',
        'AnimConstraintPoint', 'AnimConstraintParent']
json.dump({k: found[k] for k in keep if k in found}, open(sys.argv[2], 'w'), indent=1)
for k in keep:
    print(k, len(found.get(k, [])))
