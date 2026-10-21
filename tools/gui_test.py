"""WAYLAND_DISPLAY=wayland-1 blender --enable-autoexec out/<hero>_rig.blend -P gui_test.py -- shots_dir
Drives the real UI with timers: operators run in window context, screenshots after each step."""
import os
import sys
import traceback

import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index('--') + 1]
os.makedirs(OUT, exist_ok=True)
steps = []
log = []


def step(f):
    steps.append(f)
    return f


def win():
    return bpy.context.window_manager.windows[0]


def area(kind):
    return next(a for a in win().screen.areas if a.type == kind)


def ctx(kind='VIEW_3D'):
    a = area(kind)
    return dict(window=win(), area=a, region=next(r for r in a.regions if r.type == 'WINDOW'), screen=win().screen)


def shot(name):
    with bpy.context.temp_override(**ctx()):
        bpy.ops.screen.screenshot(filepath=f'{OUT}/{name}.png')


def say(*a):
    msg = ' '.join(str(x) for x in a)
    log.append(msg)
    print('GUI', msg)


arm = next(o for o in bpy.data.objects if 'dlrig' in o)


def select_bone(name):
    for pb in arm.pose.bones:
        pb.select = False
    arm.pose.bones[name].select = True
    arm.data.bones.active = arm.data.bones[name]


@step
def s_start():
    say('panel registered', hasattr(bpy.types, 'DLRIG_PT_main'), 'mode', bpy.context.mode)
    for a in win().screen.areas:
        if a.type == 'VIEW_3D':
            for r in a.regions:
                if r.type == 'UI':
                    say('sidebar width', r.width, 'category', getattr(r, 'active_panel_category', '?'))
    with bpy.context.temp_override(**ctx()):
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected() if False else bpy.ops.view3d.view_all()
    shot('00_start')


@step
def s_ik_hand():
    select_bone('IK_hand_L')
    before = arm.pose.bones['hand_L'].head.copy()
    with bpy.context.temp_override(**ctx()):
        bpy.ops.transform.translate(value=(-0.15, -0.3, 0.35))
    bpy.context.view_layer.update()
    after = arm.pose.bones['hand_L'].head.copy()
    goal = arm.pose.bones['IK_hand_L'].head.copy()
    say('IK hand: hand moved', round((after - before).length, 3), 'm, distance to box', round((after - goal).length * 1000, 2), 'mm')
    shot('01_ik_hand')


@step
def s_snap():
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.snap(limb='arm', side='L', to_ik=False)
    say('after snap to FK ik_arm_L =', arm['ik_arm_L'])
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.reset_pose()
    arm['ik_arm_L'] = 1.0


@step
def s_face():
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.expression(name='Laugh')
        bpy.ops.dlrig.hand(name='Fist', side='B')
    kb = bpy.data.objects['Head'].data.shape_keys.key_blocks
    say('laugh smile', round(kb['mouthSmileLeft'].value, 2), 'action', arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else None)
    shot('02_face_keys')


@step
def s_game():
    with bpy.context.temp_override(**ctx()):
        r = bpy.ops.dlrig.game_anim(clip='game_out_of_combat_run_n', mode='NEW')
    sc = bpy.context.scene
    say('game anim', r, arm.animation_data.action.name, sc.frame_start, sc.frame_end)
    with bpy.context.temp_override(**ctx()):
        bpy.ops.view3d.view_all()
    shot('03_game_anim')


@step
def s_play():
    sc = bpy.context.scene
    sc.dlrig_phys_loop = True
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.clear_physics()
    cb = arm.animation_data.action.layers[0].strips[0].channelbags[0]
    n0 = len([f for f in cb.fcurves if 'PHYS-' in f.data_path])
    with bpy.context.temp_override(**ctx()):
        bpy.ops.screen.animation_play()
    n1 = len([f for f in arm.animation_data.action.layers[0].strips[0].channelbags[0].fcurves if 'PHYS-' in f.data_path])
    say('physics curves before play', n0, 'after play started', n1)


@step
def s_playing():
    shot('04_playing')
    with bpy.context.temp_override(**ctx()):
        bpy.ops.screen.animation_cancel(restore_frame=False)
    say('frame after playing', bpy.context.scene.frame_current)


@step
def s_idle():
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.reset_pose()
        arm.animation_data.action = None
        r = bpy.ops.dlrig.idle_layer()
    bpy.context.scene.dlrig_idle_strength = 0.5
    say('idle layer', r, 'strength', bpy.context.scene.dlrig_idle_strength)
    shot('06_idle')


@step
def s_hands_ui():
    with bpy.context.temp_override(**ctx()):
        bpy.ops.dlrig.game_anim(clip='game_ui_hero_select', mode='POSE')
    shot('05_pose_from_game')


@step
def s_done():
    with open(f'{OUT}/log.txt', 'w') as f:
        f.write('\n'.join(log))
    bpy.ops.wm.quit_blender()


it = iter(steps)


def tick():
    try:
        f = next(it)
    except StopIteration:
        return None
    try:
        f()
    except Exception:
        traceback.print_exc()
        say('EXC in', f.__name__, traceback.format_exc().splitlines()[-1])
    return 2.0


bpy.app.timers.register(tick, first_interval=4)
