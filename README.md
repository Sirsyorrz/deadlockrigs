# deadlockrigs

Blender animation rigs for Deadlock heroes, built from the game files.

Each hero gets a rig with these parts:
- IK/FK arms and legs, with IK tweak rings
- the game's twist bones and cloth/hair physics
- a loader for all of the hero's in-game animations
- hand poses

Violet also has ARKit face shape keys with face controls and expressions.

Nothing extracted from the game is committed: `heroes/` is ignored and holds everything per hero.

## Build

Needs Blender (Steam), Source2Viewer-CLI (from `deadlock-forge`) and the installed game. The paths are at the
top of `make.sh` and can be overridden with `BLENDER`, `S2V` and `PAK`.

    ./make.sh violet                 # one hero, all steps
    ./make.sh all                    # every hero in heroes.json
    ./make.sh haze rig test          # only some steps

The steps run in order: `extract data base anims rig test`.

| Step | What it does | Writes |
|---|---|---|
| `extract` | Source2Viewer exports of the model (with textures), the model with animations, and the decompiled vmdl | |
| `data` | Cloth chains and collision capsules from the vmdl; face keys for heroes with a face config | `data/` |
| `base` | The model with bodygroup-hidden parts moved to "Alternate parts", meshes named after their vmdl parts | `out/<hero>.blend`, `.glb`, `.fbx` |
| `anims` | The animation library | `out/<hero>_anims.blend` |
| `rig` | The rig (needs the two files above) | `out/<hero>_rig.blend` |
| `test` | Numeric checks (IK reach, snapping, tweaks, twist vs game, physics, clip loading) | `test.log` |

`KEEP=1` keeps the big intermediate files (animation glb, decompiled dmx). Without it they are deleted after use.

`heroes.json` lists the heroes: slug, display name, codename and model. It also holds optional per-hero settings:
- `rename` / `drop`: rename or remove mesh parts
- `cloth_links`: neighbouring cloth chains to hold together
- `face`: face shape keys, only Violet so far

## Using a rig

Open `heroes/<hero>/out/<hero>_rig.blend` and allow "Run Python Scripts" once so the panel loads. Keep
`<hero>_anims.blend` next to it, because the rig loads clips from it. Everything is in Sidebar (N) > tab named after
the hero, in Pose Mode on the armature.

- Auto-Key toggle at the top: with it on, every button sets keys.
- Arms & Legs: limbs start in IK. Boxes are the hands/feet, diamonds aim the elbows/knees. The two icons next to
  each slider switch to IK / to FK without the limb jumping. In IK, the rings on the upper/lower arm and leg
  rotate those bones on top of the IK result (the upper one carries the rest of the limb). Snapping either way
  bakes that into the pose. "Convert Animation to IK" keys the IK controls on every frame of an FK animation.
- Game Animations: Load as New Action (arms/legs switch to FK), Insert at Current Frame, or Copy Pose. "Add Idle
  Motion" layers the idle's breathing and sway on top as an NLA track.
- Physics: the game's cloth chains (goal strength, bend, gravity) and body collision capsules, baked into the
  action. "Re-bake when playback starts" re-bakes every time you press play.
- Hands: Relaxed, Fist, Point, Peace, Thumbs Up, Open, Grip (also in the Asset Browser).
- Face (Violet): circles on the face, a `FACE` box with 51 ARKit sliders, expressions and mouth shapes.

## How the rig works

IK runs on hidden clean chains (`IKC-*`), because the imported bones' tails point arbitrary ways. Game clips also
key bone translations, so each `IKC-*` bone takes its deform bone's location offset through a driver, and a tiny
locked `IKC-<hand/ankle>` bone carries the end's offset; its tail is the effector, aimed at `IKE-*` on the IK
control. `IK_tweak_*` bones copy the chain's local transforms and add their own rotation. The deform bones copy
`MCH-*` children of the tweaks, so their rest pose (and the skinning) is untouched.

Twist uses `TW-*` helpers and a Transformation constraint (swing/twist about the limb axis). The weights are fitted
per hero against the twist bones baked into its clips, when the clips have them; otherwise Violet's weights are
used.

Physics keys `PHYS-*` helper bones; each cloth bone adds its helper's rotation on top of the animation.

Parts a skeleton lacks (Baba's arms, Bebop's left fingers) are skipped. The panel code lives in the .blend
(`dlrig_ui.py` text block): after changing `tools/rig_ui.py`, rebuild the rig.

## Status

All 44 heroes in `heroes.json` build and pass `test`, except Sinclair, whose twist check fails. Sinclair has no
TiltTwist constraints in the vmdl, and the best fitted weights still differ from the twist baked into his clips
by up to about 40 degrees.

Other known gaps:
- Baba has no arm or spine bones under the standard names, so only her legs get IK.
- Bebop has no left-hand fingers, and Mo & Krill have no `finger_index_meta`, so they get no hand poses (both
  sides for Mo & Krill).
- Face controls exist only for Violet. Abrams and McGinnis have real face bones, which are posable directly;
  Billy, Deadman Danny, Rem and Victor ship morphs.
- Limbs in IK use rest bone lengths plus the clip's own translation offsets, so a game clip converted to IK
  matches the clip, but stretching a limb by hand is not possible.

## Violet's face

The game model has no face bones or morphs (the face is skinned to `head` only). `tools/keys.py` therefore builds
51 ARKit-named shape keys from the mesh:
- Lids turn about the eyeball centre.
- The jaw turns about `jaw_0` with harmonic weights.
- The rest are translation "virtual bones" with geodesic falloff.

Sizes follow sot-pirate's `face_keys.gd`. The landmarks and UV island ids in `keys.py` and the control positions in
`heroes.json` are specific to Violet's head.

`mouthClose` is meant to be used with `jawOpen`, like ARKit. There are no teeth or tongue in the model, so the
mouth interior is a plain dark material.

    tools/sheet.sh shots/check.png eyeBlinkLeft,jawOpen 0 face   # render keys, angle, focus: face|mouth|L|R

## Other checks

    B=~/.local/share/Steam/steamapps/common/Blender/blender
    WAYLAND_DISPLAY=wayland-1 $B --enable-autoexec heroes/violet/out/violet_rig.blend -P tools/gui_test.py -- /tmp/gui
    tools/hand_sheet.sh heroes/violet/out/violet_rig.blend L shots/hands_L.png Rest Fist Point "Thumbs Up"
    tools/anim_sheet.sh heroes/violet/out/violet_rig.blend game_out_of_combat_run_n 0,4,8,12 hips 1 1 shots/run.png
