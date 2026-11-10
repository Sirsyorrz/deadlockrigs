# Hand-made faces

`<hero>.json` here is a face built by hand for `tools/handkeys.py` (heroes.json face `"builder": "handkeys.py"`), for
heads the generator (`keys.py`) can't read. The build turns it into the same ARKit shape keys and controls as every
other face.

Positions are mm from the head bone in Blender axes: +x the hero's left, -y forward, +z up. Named points:
`eye_L`/`eye_R` (eyeball centres), or any bone under the head (`jaw_0`, `nose_0`).

Everything with `{s}` / `{S}` in its name is written for the left side (`L` / `Left`) and mirrored for the right.

## regions

Weights 0..1 per vertex, by name in `"regions"` or inline:

| form | weight |
|---|---|
| `{"bone": "jaw_0"}` | the game's skin weight for that bone (a list adds several) |
| `{"sphere": p, "r": mm, "soft": mm}` | 1 inside r, fading to 0 over soft |
| `{"box": p, "size": [x, y, z], "soft": f}` | an ellipsoid with those radii, fading over f radii |
| `{"geo": p, "r": mm, "soft": mm, "on": region}` | like sphere, but measured along the skin from the vertex nearest p |
| `{"half": p, "n": [x, y, z], "soft": mm}` | the side of a plane the normal points to |
| `{"lid": "{s}", "part": "upper"\|"lower", "soft": deg}` | above or below the middle line of an eye's opening |
| `{"mat": name}` | a material |
| `{"piece": "skin"\|"balls"\|"other"}` | the main skin, the eyeballs, or every other piece (teeth, cards) |
| `{"pieces_near": region}` | whole separate pieces reaching into a region |
| `{"all": [...]}`, `{"any": [...]}`, `{"not": r}` | product, max, 1 - r |
| `{"times": k, "of": r}`, `{"smooth": n, "of": r}` | scaled; averaged over neighbours n times |
| `{"whole": r}` | r on the skin; each other piece (a tooth) takes its mean, so it moves without stretching |
| `{"ramp": [lo, hi], "of": r}` | 0 where r is below lo, 1 above hi (a hold along the split from the body: the head bones' weight ramping from 0.5) |
| `{"harmonic": {"one": r, "zero": r}}` | smooth 1 to 0 across the skin between two regions |

## keys

`"keys"`: ARKit key name -> list of moves, added together. Eyeballs never move unless a move says `"balls": true`.

| move | |
|---|---|
| `{"move": region, "by": [x, y, z]}` | translate, mm |
| `{"turn": region, "about": p, "axis": [x, y, z] or "lid_{s}", "deg": d}` | rotate, scaled by the weight |
| `{"lid": region, "eye": "{s}", "close": f, "part": "upper"\|"lower"}` | turn a lid by f of the eye's opening where it sits (upper 0.9 + lower 0.2 shuts it) |
| `{"scale": region, "about": p, "by": f or [x, y, z]}` | |
| `{"push": region, "by": mm}` | along the skin normal |
| `{"key": name, "times": k}` | another key's moves |

Extra fields: `"mask": region` multiplies one move, `"sym": true` repeats it mirrored in a key without `{S}`.

`"mid"`: keys that get an in-between (`<key>_mid`), so turns stay round half way. `"hold"`: a region no key moves.

`"tongue"`: a generated tongue for a head with none (keys.py's shape), lying on the mouth floor under `top` (z) from
`tip` to `root` (y), at most `width` x `height` mm, raised `gap` mm off the floor. The keys in `follow` (and their
in-betweens) move it like the floor under it; tongueOut slides it `reach` mm out, lifts it to `lift` (z) and drops
the tip `droop` mm (default 6). Part the lips for it with the head's own `"tongueOut"` moves.

`"bag"`: the dark mouth inside. Skin in `region` hidden from the front by over `deep` mm of mouth goes dark (`inner`
mm marks the lips' inner faces). `spread` (deg): skin that a ray tilted that far escapes from stays light, so lip
under an overhanging lip doesn't go dark and show as a spike at the corners. `cap` (a point, mm): closes the hole
in the skin nearest it with a dark face, for a bag open at the back (Baba's), which otherwise shows the body behind it.
`stick` (`{"region": r, "mm": d}`): bag skin in `r` within `d` mm of the face takes the nearest face vertex's
offsets in every key, for a bag lying just behind the cheek (Apollo's) that keys would otherwise poke through. Keep
`r` away from the lips, or an inner lip can glue to the other lip.

`"eyes"`: `gaze` (the left eye's look direction), `lid_axis` (the left eye's opening, corner to corner), and the look
angles `up`, `down`, `in`, `out` (degrees), `iris` (default true). The eye looks, their in-betweens and the iris keys
come from these.

## controls

`"controls"`: where the face controls sit. `landmarks` (`lid`, `lidlow`, `brow`, `cheek`, `corner`, `lipup`,
`liplow`, `nose`, left side), `brow_mid`, `mouth`, `jaw`, `eye_target` + `eye_spacing` (mm), `face_box` (the FACE
panel bone), `scale` (control size and travel, 1 = Violet's face). With `snap` (default) each point moves onto the
skin under it and `lift` mm (default 6) out along its normal.

## rig

`"rig"`: the bone face on top of the keys (`tools/facebones.py`, built into the rig by `tools/facerig.py`). Deform bones
on the skin for free posing past the presets; the shape keys stay for lip sync and expressions. Each bone takes its
share of the skin from the game bone it hangs under (`head`, `jaw_0`) and the face bones already under that bone, so
the rest pose is unchanged.

| entry | |
|---|---|
| `"chains": {"snout_{i}": {"points": [p, ...], "parent": bone, "mask": region, "blend": mm, "size": mm}}` | FK bones joint to joint, each weighted along its stretch of the line, blending over `blend` at each joint |
| `"chains": {"whisker_{n}_{i}_{s}": {"pieces": region, "root": p, "bones": 3, "parent": [bones]}}` | a chain along each separate piece in the region (whiskers, antennae), from the end nearest `root`; hangs off the nearest of `parent` |
| `"tweaks": {name: {"at": p, "parent": bone, "r": mm, "soft": mm, "mask": region, "field": region, "main": true, "size": mm, "reach": mm}}` | a control on the skin at `at` (snapped into `mask`), weighted `r` along the skin fading over `soft`, or by `field`; `main` ones carry the tweaks under them; `reach` keeps it within that distance of where the keys put it, `down` caps only its downward travel (mm); `batch` splits its weights off with the other tweaks of that batch instead of its parent's, in order (a last batch takes an equal share from everything under it) |
| `"reparent": {"nose_0": "snout_1"}` | game bones that ride a chain |
| `"show": ["ear_0_{s}"]` | game bones shown with the face bones |
| `"rigid": region` | separate pieces in it (teeth) move whole, on their mean weights |
| `"scale"` | widget size, 1 = Violet's face |

`at` can also be `{"lid": "{s}", "t": 0..1, "part": "upper"|"lower"}`, a point on a lid's margin from the inner
corner (0) to the outer (1). A tweak's control rides the skin through the shape keys (drivers on `MCH-follow_*`), and
`DEF-<name>` does the deforming.

## checking

    python tools/handkeys.py <hero> --regions
    blender -b -P tools/face_preview.py -- <hero> out_dir "" side,34 0,-300,0 500 jaw,lid_up_L   # region heat maps
    blender -b -P tools/face_preview.py -- <hero> out_dir eyeBlinkLeft,jawOpen=0.5 34 0,-300,0 500
    blender -b -P tools/head_views.py -- heroes/<hero>/build/<hero>.blend out_dir                 # mm grid views
    blender -b -P tools/control_views.py -- <hero> out.png 34,side                                # control spots

    python tools/handkeys.py <hero>; REGIONS=facerig blender -b -P tools/face_preview.py -- <hero> out_dir "" side 0,-300,0 500 lip_up_L   # bone weights
    blender -b -P tools/rig_views.py -- <hero> out.png side,34 0,-300,0 500 "Face Bones,Face Detail" pose.py              # widgets, posed

`face_preview.py` reads `data/keys.npz` straight onto the base build, so after the first `pipeline.py <hero> data
base` only `handkeys.py` needs rerunning between tries.
