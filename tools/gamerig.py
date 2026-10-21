"""python3 gamerig.py gamedata.json artist.glb gamerig.json
Converts the game's cloth chains and collision capsules (bone-local, inches) to rest-pose armature
space in metres, Blender axes, and lists the twist constraints, ready for rig.py."""
import json
import struct
import sys

import numpy as np

INCH = 0.0254
G2B = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)

data = json.load(open(sys.argv[1]))
raw = open(sys.argv[2], 'rb').read()
n = struct.unpack('<I', raw[12:16])[0]
nodes = json.loads(raw[20:20 + n])['nodes']
parent = {c: i for i, nd in enumerate(nodes) for c in nd.get('children', [])}
index = {nd.get('name'): i for i, nd in enumerate(nodes)}


def local(nd):
    if 'matrix' in nd:
        return np.array(nd['matrix'], float).reshape(4, 4).T
    x, y, z, w = nd.get('rotation', [0, 0, 0, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * np.array(nd.get('scale', [1, 1, 1]))
    m[:3, 3] = nd.get('translation', [0, 0, 0])
    return m


def world(name):
    i = index[name]
    m = local(nodes[i])
    while i in parent:
        i = parent[i]
        m = local(nodes[i]) @ m
    return G2B @ m


def to_rest(bone, p):
    return (world(bone) @ np.array([*(np.array(p) * INCH), 1.0]))[:3].round(5).tolist()


chains = []
for c in data.get('ClothChain', []):
    joints = []
    for j in c['chain']['joints']:
        joints.append({
            'bone': j['joint_name'],
            'simulate': bool(j.get('simulate', True)),
            'goal_strength': j.get('goal_strength', 0.5),
            'goal_damping': j.get('goal_damping', 0.0),
            'gravity': j.get('gravity_z', 1.0),
            'radius': j.get('collision_radius', 0.0) * INCH,
            'stretch': j.get('stretch_spring', 1.0),
            'bend': j.get('bend_spring', 0.0),
        })
    chains.append({'name': c['name'], 'joints': joints})

capsules = [{'bone': c['parent_bone'], 'p0': to_rest(c['parent_bone'], c['point0']),
             'p1': to_rest(c['parent_bone'], c['point1']),
             'r0': c['radius0'] * INCH, 'r1': c['radius1'] * INCH} for c in data.get('ClothShapeCapsule', [])]

# AnimConstraintTiltTwist: the slave bone turns by weight x the input bone's twist
twists = []
for c in data.get('AnimConstraintTiltTwist', []):
    kids = c.get('children', [])
    slave = next((k for k in kids if k['_class'] == 'AnimConstraintSlave'), None)
    inp = next((k for k in kids if k['_class'] == 'AnimConstraintBoneInput'), None)
    if slave and inp:
        twists.append({'bone': slave['parent_bone'], 'weight': slave['weight'], 'input': inp['parent_bone']})

json.dump({'chains': chains, 'capsules': capsules, 'twists': twists}, open(sys.argv[3], 'w'), indent=1)
print(len(chains), 'chains', sum(len(c['joints']) for c in chains), 'joints', len(capsules), 'capsules',
      len(twists), 'twists')
