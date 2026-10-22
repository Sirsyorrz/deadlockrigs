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

Needs Blender (Steam), `git`, the .NET 10 SDK and the installed game. The paths are at the top of `make.sh` and can
be overridden with `BLENDER`, `S2V` and `PAK`.

`extract` uses Source2Viewer-CLI from [Source 2 Viewer](https://github.com/ValveResourceFormat/ValveResourceFormat).
The releases (up to 20.1) leave the cloth chains, capsules and physics joints out of the decompiled vmdl, so on
first use `make.sh` builds a pinned master commit (`S2V_COMMIT`) into `s2v/` (ignored, about 30 s). Set `S2V` to
use an existing binary instead.

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

Blender 5.x writes a .blend format that only 4.5 and newer can open. For Blender 4.4, unpack a portable 4.4 build
from download.blender.org into `older-versions/` (ignored, e.g. `older-versions/blender-4.4.3-linux-x64/`) and build
with `OLD=4.4`. The output goes to `heroes/<hero>/out-4.4/` and the test log to `test-4.4.log`, next to the normal
`out/`. `anims` needs the animation glb, so use `KEEP=1` on whichever build runs first, or `extract` again.

    KEEP=1 ./make.sh violet
    OLD=4.4 ./make.sh violet base anims rig test

`KEEP=1` keeps the big intermediate files (animation glb, decompiled dmx). Without it they are deleted after use.

`heroes.json` lists the heroes: slug, display name, codename and model. It also holds optional per-hero settings:
- `rename` / `drop`: rename or remove mesh parts
- `cloth_links`: neighbouring cloth chains to hold together
- `split`: object name -> materials to move out of their mesh into a separate object (Violet's portfolio and
  brush), so they can be hidden or deleted
- `props`: held props (Abrams' book). The prop's bone gets a box control in the Props collection, and its default
  hold comes from the idle clip. An optional `chain` (with `chain_end` and `anchor`) stays attached to the prop
  through IK. `"hold": false` is for worn props (Violet's portfolio) that already sit in place at rest.
- `face`: face shape keys. `part` is the head mesh's vmdl part, `mesh` its object name after `rename`. Optional
  `controls` places the face controls by hand (Violet); without it they come from `data/face.json`. `tune` scales
  a key (and its `_mid` in-between) for one hero, e.g. Drifter's open grin takes only 0.6 of `mouthPucker`.
  `"builder": "morphs.py"` builds the keys from the game's own face morphs (Billy, Rem, Victor), `"bonekeys.py"`
  from its FACS face bones (Abrams). `attach` lists prop bones that ride the face skin (his cigarette).

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
- Physics: the game's cloth chains (goal strength, bend, gravity), vertex cloth (Abrams' coat) and body
  collision capsules, baked into the action. "Re-bake when playback starts" re-bakes every time you press play.
- Hands: Relaxed, Fist, Point, Peace, Thumbs Up, Open, Grip (also in the Asset Browser).
- Held props (Abrams' book): move or rotate the box around the prop. It follows the hand, and the chain to the cuff
  follows the book. "Chain follows prop" in Arms & Legs turns that off. Reset puts the book back in his hand.
  Violet's portfolio works the same way (box around bag and strap, follows the chest); it and her brush are separate
  objects (`Portfolio`, `Brush`) you can hide or delete.
- Tail (Abrams, Apollo, Baba's pigeon, Calico's cat, Ivy, Mirage's djinn, Rat King, Rem, Silver, Vyper; Grey Talon's
  owl has one bone): rings along the tail for FK, and a box at the tip that rides the FK tail. Move the box to bend
  the tail towards it; left alone it changes nothing, so clips and physics play as they are. "Tail IK" in Arms & Legs
  turns the box off. Infernus' tail bones carry no skin (his tail is flames) and stay in Secondary.
- Face (heroes with face keys): circles on the face, a `FACE` box with 51 ARKit sliders, expressions and mouth
  shapes. The eye target moves both eyes; the circle around each eye moves that eye alone, and scaling it sizes
  the iris.

## How the rig works

IK runs on hidden clean chains (`IKC-*`), because the imported bones' tails point arbitrary ways. Game clips also
key bone translations, so each `IKC-*` bone takes its deform bone's location offset through a driver, and a tiny
locked `IKC-<hand/ankle>` bone carries the end's offset; its tail is the effector, aimed at `IKE-*` on the IK
control. `IK_tweak_*` bones copy the chain's local transforms and add their own rotation. IK sets the upper and
lower bones' roll from the pole, so when switching FK to IK (or converting a clip) any twist the FK pose has about
the limb axis goes into the tweaks, which keeps clothing and props parented to the arm/thigh in place. The deform bones copy
`MCH-*` children of the tweaks, so their rest pose (and the skinning) is untouched.

Many of the game's right-side bones have rest frames turned (mostly 180 degrees) against the mirrored left ones, which
made X-mirror posing and Paste Flipped bend the other side the wrong way. The rig turns those rest frames into the
mirror of the left ones (`tools/mirror.py`; positions stay, so the skinning doesn't change), and `anims.py` converts
the clips' keys to match. The clip library is therefore meant for the rig, not the plain `<hero>.blend`, and
right-side bones exported from the rig don't have the game's bone orientations.

Twist uses `TW-*` helpers and a Transformation constraint (swing/twist about the limb axis). The weights are fitted
per hero against the twist bones baked into its clips, when the clips have them; otherwise Violet's weights are
used.

Physics keys `PHYS-*` helper bones; each cloth bone adds its helper's rotation on top of the animation.

Vertex cloth (`$cloth_*` bones, Abrams' coat and 15 other heroes) is a Source 2 FeModel: particles held by rods and
pulled towards the body. `make.sh data` dumps it with Source2Viewer (`-b PHYS`) and `tools/fe_extract.py` writes
`data/cloth.json` (rest positions, which body bones each particle follows and how much, damping, rods). Each free
particle follows its body bones through an Armature constraint and adds its `PHYS-*` helper's offset; the bake runs
a position-based solve in the root's frame, so root motion doesn't drag the cloth.

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
| Billy | game morphs | 31 keys; no lip corner puller, so `mouthSmile` is empty |
| Drifter | yes | lower teeth follow the jaw; the sculpt is an open grin, so pucker, funnel and mouthLeft/Right are toned down |
| Graves | yes | her dark lid make-up shows when the eyes close |
| Mina | yes | |
| Paige | yes | lower teeth follow the jaw; the glasses stay put |
| Rem | game morphs | 21 keys; the mask's eyes have no bones (no eye look) and the beak has no lip shapes |
| Victor | game morphs | 36 keys |
| Violet | yes | controls placed by hand |
| Venator | not yet | the lips are parted, so the mouth is not closed off and no mouth bag is found |
| Nurse Harrow | not yet | separate head mesh, not checked further |
| The Doorman | not yet | separate head mesh, but the lips are one closed surface (no mouth bag), so the jaw can't open the mouth |
| Abrams | game face bones | 56 keys; eyes, teeth, tongue and cigarette follow the skin |
| McGinnis | not yet | game face bones, like Abrams |
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

`irisLarge*`/`irisSmall*` slide the front of the eyeball over the sphere, away from or towards the gaze axis.

`mouthClose` is meant to be used with `jawOpen`, like ARKit. None of the heads has a tongue, so `keys.py` builds one
on the mouth bag floor (`data/keys_tongue.npz`), sized to the bag, and `build.py` adds it to the head mesh. It
follows the jaw and has the 52nd ARKit key, `tongueOut` (meant with `jawOpen`; on a closed mouth it goes through
the lips). A driven corrective, `tongueOut_mid`, peaks at `tongueOut` 0.5 and lifts the tongue to the lips before
it slides out, so half-out tongues no longer go through the lower lip. Heads without teeth get none; the rest of the
mouth interior is a plain dark material.

Separate head meshes reach down into the collar skinned almost fully to `neck_0`, while the collar follows
`spine_3`, so bending the neck pushed the hidden neck out through the collar (Violet). `build.py` (`tools/collar.py`)
gives head vertices covered by a torso mesh that mesh's weights, blending back to their own over 3 cm above it.

Billy, Rem and Victor ship FACS morphs (`lipCornerPuller`, `jawDrop`, ...). `tools/morphs.py` builds the ARKit keys
as sums of them, so they work with the same panel and controls. A morph's `_50` in-between becomes a driven `_mid`
key like `tongueOut_mid`. Eye looks turn the eyeball about its bone, and the controls go where each morph moves the
skin most. Control sizes and travel scale with the head (`face.json` `scale`).

Abrams has the game's FACS face bones (`lip_corner_L`, `brow_inner_L`, lid pivots, `tongue_*`). `tools/bonekeys.py`
poses them per ARKit key (moves in mm, scaled by eye spacing) and bakes the skinned result into keys on every mesh
they move: `data/keys.npz` for the face and `data/keysmesh_<part>.npz` for the eyes, teeth and tongue. The head
drives those, so the panel and controls are the same as everywhere else. Props on the face (`face.attach`, his
cigarette) ride the skin nearest their bone.

Glints stand off the eyeball and would poke through a closed lid, so the upper lid closing pushes them into the
ball. Lash and liner cards lift slightly off the ball as they turn, because the lid skin bunching up above the
margin came through thick liner (Paige's).

Hair, hats and other meshes around the head are dumped too: skin above the eyes that lies within a few mm of them
stays put, so brow keys can't push it through.

    HERO=mina tools/sheet.sh shots/check.png eyeBlinkLeft,jawOpen 0 face   # render keys, angle, focus: face|mouth|L|R

## Other checks

    B=~/.local/share/Steam/steamapps/common/Blender/blender
    WAYLAND_DISPLAY=wayland-1 $B --enable-autoexec heroes/violet/out/violet_rig.blend -P tools/gui_test.py -- /tmp/gui
    tools/hand_sheet.sh heroes/violet/out/violet_rig.blend L shots/hands_L.png Rest Fist Point "Thumbs Up"
    tools/anim_sheet.sh heroes/violet/out/violet_rig.blend game_out_of_combat_run_n 0,4,8,12 hips 1 1 shots/run.png
