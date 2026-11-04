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

`"eyes"`: `gaze` (the left eye's look direction), `lid_axis` (the left eye's opening, corner to corner), and the look
angles `up`, `down`, `in`, `out` (degrees), `iris` (default true). The eye looks, their in-betweens and the iris keys
come from these.

## controls

`"controls"`: where the face controls sit. `landmarks` (`lid`, `lidlow`, `brow`, `cheek`, `corner`, `lipup`,
`liplow`, `nose`, left side), `brow_mid`, `mouth`, `jaw`, `eye_target` + `eye_spacing` (mm), `face_box` (the FACE
panel bone), `scale` (control size and travel, 1 = Violet's face). With `snap` (default) each point moves onto the
skin under it and `lift` mm (default 6) out along its normal.

## checking

    python tools/handkeys.py <hero> --regions
    blender -b -P tools/face_preview.py -- <hero> out_dir "" side,34 0,-300,0 500 jaw,lid_up_L   # region heat maps
    blender -b -P tools/face_preview.py -- <hero> out_dir eyeBlinkLeft,jawOpen=0.5 34 0,-300,0 500
    blender -b -P tools/head_views.py -- heroes/<hero>/build/<hero>.blend out_dir                 # mm grid views
    blender -b -P tools/control_views.py -- <hero> out.png 34,side                                # control spots

`face_preview.py` reads `data/keys.npz` straight onto the base build, so after the first `pipeline.py <hero> data
base` only `handkeys.py` needs rerunning between tries.
