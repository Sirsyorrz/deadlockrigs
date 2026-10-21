"""Expression and viseme presets as ARKit shape key weights (shared by rig.py and the Blender panel)."""

ARKIT = [
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeWideLeft', 'eyeWideRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight',
    'jawOpen', 'jawForward', 'jawLeft', 'jawRight', 'mouthClose',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthLeft', 'mouthRight',
    'mouthPucker', 'mouthFunnel', 'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
]

GROUPS = {
    'Eyes': [k for k in ARKIT if k.startswith('eye')],
    'Brows': [k for k in ARKIT if k.startswith('brow')],
    'Cheeks & Nose': [k for k in ARKIT if k.startswith(('cheek', 'nose'))],
    'Jaw': [k for k in ARKIT if k.startswith('jaw')] + ['mouthClose'],
    'Mouth': [k for k in ARKIT if k.startswith('mouth') and k != 'mouthClose'],
}


def both(name, v):
    return {name + 'Left': v, name + 'Right': v}


def mirror(preset):
    out = {}
    for k, v in preset.items():
        if k.endswith('Left'):
            k = k[:-4] + 'Right'
        elif k.endswith('Right'):
            k = k[:-5] + 'Left'
        elif k in ('mouthLeft', 'jawLeft'):
            k = k.replace('Left', 'Right')
        out[k] = v
    return out


SMIRK = {'mouthSmileLeft': 0.8, 'mouthDimpleLeft': 0.3, 'cheekSquintLeft': 0.35, 'eyeSquintLeft': 0.2,
         'mouthPressRight': 0.2}
WINK = {'eyeBlinkLeft': 1, 'cheekSquintLeft': 0.4, 'mouthSmileLeft': 0.35, 'browDownLeft': 0.2}

EXPRESSIONS = {
    'Smile': {**both('mouthSmile', 0.8), **both('cheekSquint', 0.4), **both('eyeSquint', 0.2)},
    'Laugh': {**both('mouthSmile', 1), 'jawOpen': 0.35, **both('cheekSquint', 0.6), **both('eyeSquint', 0.45),
              **both('mouthUpperUp', 0.2), 'browInnerUp': 0.2},
    'Smirk L': SMIRK,
    'Smirk R': mirror(SMIRK),
    'Smug': {'mouthSmileLeft': 0.5, 'mouthSmileRight': 0.2, **both('eyeBlink', 0.35), 'browOuterUpLeft': 0.4,
             **both('mouthPress', 0.3)},
    'Sad': {**both('mouthFrown', 0.7), 'browInnerUp': 0.85, 'mouthShrugLower': 0.4, **both('mouthPress', 0.2),
            **both('eyeLookDown', 0.15)},
    'Angry': {**both('browDown', 1), **both('eyeSquint', 0.5), **both('noseSneer', 0.4), **both('mouthFrown', 0.4),
              **both('mouthPress', 0.4)},
    'Shout': {**both('browDown', 1), 'jawOpen': 0.6, **both('mouthUpperUp', 0.5), **both('mouthLowerDown', 0.4),
              **both('noseSneer', 0.6), **both('mouthStretch', 0.4), **both('eyeSquint', 0.3)},
    'Surprised': {'browInnerUp': 1, **both('browOuterUp', 1), **both('eyeWide', 1), 'jawOpen': 0.45,
                  'mouthFunnel': 0.3},
    'Scared': {'browInnerUp': 1, **both('browOuterUp', 0.4), **both('eyeWide', 0.8), **both('mouthStretch', 0.6),
               'jawOpen': 0.2, **both('mouthLowerDown', 0.3)},
    'Disgust': {'noseSneerLeft': 1, 'noseSneerRight': 0.7, 'mouthUpperUpLeft': 0.7, 'mouthUpperUpRight': 0.4,
                **both('browDown', 0.5), **both('eyeSquint', 0.4), **both('mouthFrown', 0.3),
                **both('cheekSquint', 0.3)},
    'Pout': {'mouthPucker': 0.4, 'mouthShrugLower': 0.6, **both('mouthFrown', 0.3), 'browInnerUp': 0.4},
    'Kiss': {'mouthPucker': 1, **both('eyeBlink', 0.4)},
    'Thinking': {'browDownRight': 0.5, 'browOuterUpLeft': 0.6, 'eyeSquintRight': 0.3, 'mouthLeft': 0.4,
                 'mouthPressLeft': 0.3, **both('eyeLookUp', 0.4), 'eyeLookOutLeft': 0.3, 'eyeLookInRight': 0.3},
    'Sleepy': {**both('eyeBlink', 0.7), 'browInnerUp': 0.2, 'jawOpen': 0.05},
    'Blink': both('eyeBlink', 1),
    'Wink L': WINK,
    'Wink R': mirror(WINK),
}

VISEMES = {
    'AA': {'jawOpen': 0.55, **both('mouthLowerDown', 0.2)},
    'EE': {'jawOpen': 0.15, **both('mouthStretch', 0.5), **both('mouthSmile', 0.2), **both('mouthUpperUp', 0.2)},
    'IH': {'jawOpen': 0.2, **both('mouthStretch', 0.3)},
    'OH': {'jawOpen': 0.35, 'mouthFunnel': 0.7},
    'OO': {'jawOpen': 0.1, 'mouthPucker': 0.9},
    'MBP': {**both('mouthPress', 0.6), 'mouthRollUpper': 0.3, 'mouthRollLower': 0.3},
    'FV': {'mouthRollLower': 0.7, **both('mouthUpperUp', 0.3), 'jawOpen': 0.05},
    'TH/L': {'jawOpen': 0.25, **both('mouthUpperUp', 0.15)},
    'CH/SH': {'jawOpen': 0.1, 'mouthFunnel': 0.5, 'mouthShrugUpper': 0.3},
    'Rest': {},
}
