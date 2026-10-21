# deadlockrigs

Blender animation rigs for Deadlock heroes, built from the game files.

Each hero gets a rig with these parts:
- IK/FK arms and legs, with IK tweak rings
- the game's twist bones and cloth/hair physics
- a loader for all of the hero's in-game animations
- hand poses

Some heroes also have ARKit face shape keys with face controls and expressions; see [Faces](#faces) for who.

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
- `props`: held props (Abrams' book). The prop's bone gets a box control in the Props collection, and its default
  hold comes from the idle clip. An optional `chain` (with `chain_end` and `anchor`) stays attached to the prop
  through IK.
- `face`: face shape keys. `part` is the head mesh's vmdl part, `mesh` its object name after `rename`. Optional
  `controls` places the face controls by hand (Violet); without it they come from `data/face.json`

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
- Held props (Abrams' book): move or rotate the box around the prop. It follows the hand, and the chain to the cuff
  follows the book. "Chain follows prop" in Arms & Legs turns that off. Reset puts the book back in his hand.
- Face (heroes with face keys): circles on the face, a `FACE` box with 51 ARKit sliders, expressions and mouth
  shapes.

## How the rig works

IK runs on hidden clean chains (`IKC-*`), because the imported bones' tails point arbitrary ways. Game clips also
key bone translations, so each `IKC-*` bone takes its deform bone's location offset through a driver, and a tiny
locked `IKC-<hand/ankle>` bone carries the end's offset; its tail is the effector, aimed at `IKE-*` on the IK
control. `IK_tweak_*` bones copy the chain's local transforms and add their own rotation. IK sets the upper and
lower bones' roll from the pole, so when switching FK to IK (or converting a clip) any twist the FK pose has about
the limb axis goes into the tweaks, which keeps clothing and props parented to the arm/thigh in place. The deform bones copy
`MCH-*` children of the tweaks, so their rest pose (and the skinning) is untouched.

Twist uses `TW-*` helpers and a Transformation constraint (swing/twist about the limb axis). The weights are fitted
per hero against the twist bones baked into its clips, when the clips have them; otherwise Violet's weights are
used.

Physics keys `PHYS-*` helper bones; each cloth bone adds its helper's rotation on top of the animation.

A prop chain runs on `CHC-*` bones (head to next head) whose pose copies the deform chain's own keys through drivers,
so a clip's chain shape is the starting point; IK only bends it to keep the end on `ANC-*`, which rides the prop.

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
- Only some heroes have face controls yet; see [Faces](#faces).
- Limbs in IK use rest bone lengths plus the clip's own translation offsets, so a game clip converted to IK
  matches the clip, but stretching a limb by hand is not possible.

## Faces

| Hero | Face rig | Notes |
|---|---|---|
| Apollo | yes | |
| Drifter | yes | lower teeth follow the jaw |
| Graves | yes | dark lid make-up shows when the eyes close |
| Mina | yes | |
| Paige | yes | lower teeth follow the jaw; the glasses stay put |
| Violet | yes | controls placed by hand |
| Venator | not yet | the lips are parted, so the mouth is not closed off and no mouth bag is found |
| Nurse Harrow | not yet | separate head mesh, not checked further |
| The Doorman | not yet | separate head mesh, but the lips are one closed surface (no mouth bag), so the jaw can't open the mouth |
| Abrams, McGinnis | no keys | real face bones (lids, lips, brows, jaw), posable directly |
| Billy, Rem, Victor | no keys | the game ships face morphs; not wired to the face panel |
| everyone else | no | the face is part of the body mesh, or the hero has no eye bones |

The heads have no face bones or morphs (the face is skinned to `head` only). `tools/keys.py` therefore builds 51
ARKit-named shape keys from the head mesh:
- Lids turn about the eyeball centre.
- The jaw turns about `jaw_0` with harmonic weights.
- The rest are translation "virtual bones" with geodesic falloff.

Sizes follow sot-pirate's `face_keys.gd`, scaled by the eye spacing (Violet's 88 mm is the reference). The face
is found from the geometry, so a new hero usually only needs `"face": {"part": "head", "mesh": "head"}`:
- Eyeballs: the vertices skinned to `eye_0_L/R`, with a sphere fit for centre and radius.
- Eye sockets: skin around the eyeball where almost no rays escape. The lid margin is the first skin met walking
  out from straight ahead in each direction around the eye (skin inside the ball, where a head closes the socket
  with a cup, is skipped).
- Mouth bag: the enclosed skin around the lips. Its edge is the lip line, split into upper and lower lip by which
  way the edge faces, and the corners are its ends.
- Lashes, liner and brow cards ride the skin under them; pieces standing off the skin (hair, glasses) stay put.
- Teeth: separate pieces with a material named `*teeth*`/`*tooth*`. They block the rays when finding the mouth
  bag, and a piece below the lip line moves rigidly with the jaw.

`data/face.json` holds the control positions derived from the same landmarks. `make.sh <hero> data base rig test`
rebuilds a face; `test` checks every key's size, left/right balance, and that the controls and sliders drive the
keys.

`mouthClose` is meant to be used with `jawOpen`, like ARKit. None of the heads has a tongue, so `keys.py` builds one
on the mouth bag floor (`data/keys_tongue.npz`), sized to the bag, and `build.py` adds it to the head mesh. It
follows the jaw and has the 52nd ARKit key, `tongueOut` (meant with `jawOpen`; on a closed mouth it goes through
the lips). Heads without teeth get none; the rest of the mouth interior is a plain dark material.

Hair, hats and other meshes around the head are dumped too: skin above the eyes that lies within a few mm of them
stays put, so brow keys can't push it through.

    HERO=mina tools/sheet.sh shots/check.png eyeBlinkLeft,jawOpen 0 face   # render keys, angle, focus: face|mouth|L|R

## Other checks

    B=~/.local/share/Steam/steamapps/common/Blender/blender
    WAYLAND_DISPLAY=wayland-1 $B --enable-autoexec heroes/violet/out/violet_rig.blend -P tools/gui_test.py -- /tmp/gui
    tools/hand_sheet.sh heroes/violet/out/violet_rig.blend L shots/hands_L.png Rest Fist Point "Thumbs Up"
    tools/anim_sheet.sh heroes/violet/out/violet_rig.blend game_out_of_combat_run_n 0,4,8,12 hips 1 1 shots/run.png
