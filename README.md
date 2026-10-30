# Deadlock Rigs

Blender animation rigs for Deadlock heroes built into blender.

![Violet's rig in Blender](docs/violet_rig.png)

Each hero gets a rig with:
- IK/FK arms and legs, with soft IK, stretch, and switchable parents for the IK hands, feet and poles
- feet that roll on the heel, ball and toe tip, bank onto either edge and swivel (the arc behind the heel), with an
  IK ankle control that turns the foot on its own and an IK toe control
- torso, hips and chest controls on top of the FK spine
- head and FK arms that can keep their own angle instead of following the body
- the game's twist bones and cloth/hair physics, including coats
- tail controls for heroes with tails
- hand poses
- weapons held in hand by default (the `weapon_hold` property; loading a game clip switches it off); Solomon's board
  floats with his body instead
- an FK jaw (and eyes) for heroes without a face rig whose jaw is skinned (Deadman Danny, Rat King, Vyper)
- face shape keys, controls and expressions for the heroes listed under [Faces](#faces)

NPCs are in the list too, under NPCs: the Guardian, Walker, Base Guardian, Mid-boss, Hidden King, Archmother and the
ranged, melee and medic troopers of both teams (Amber and Sapphire). They get the same controls for whatever their
skeleton has (the Mid-boss' five arms, the Walker's four legs, the patrons' lower body as a tail). The Base Guardian
gets foot and knee controls that its pistons follow, and a turret aim target.

| | |
|---|---|
| ![Abrams' rig](docs/abrams_rig.png) | ![Abrams posed from a game clip](docs/abrams_pose.png) |
| ![Violet on a frame of her run clip](docs/violet_run.png) | ![Vyper's tail controls](docs/vyper_tail.png) |

## Install

Needs Blender 5.1 or newer and Deadlock installed through Steam. Works on Windows, Linux and macOS.

1. Download the zip for your platform from [Releases](https://github.com/Sirsyorrz/deadlockrigs/releases).
2. In Blender: Edit > Preferences > Get Extensions > menu at the top right > Install from Disk, and pick the zip.
3. Open the 3D View sidebar (N) > Deadlock Rigs, pick a hero and press Build Rig.

![The Deadlock Rigs panel](docs/addon_panel.png)

The first build downloads Source2Viewer-CLI (about 60 MB) to read the game files. A build takes one to three
minutes. Rigs are saved to `Documents/DeadlockRigs/<hero>/<hero>_rig.blend`, and rebuilding keeps the previous rig as
`<hero>_rig.old.blend`.

Export Animations saves all of the hero's game animations as `<hero>_animations.glb`, and as `<hero>_anims.blend`
next to the rig. With that file there, the rig gets a Game Animations panel to load clips as actions, insert them at
the current frame, copy a pose, or layer the idle's breathing on top. Keep it next to the rig file.

The rig's controls are in the sidebar under a tab named after the hero, in Pose Mode. If you open a rig file
yourself, allow "Run Python Scripts" so that panel loads.

Settings are in Preferences > Add-ons > Deadlock Rigs: output folder, Deadlock folder (found through Steam when
empty), an existing Source2Viewer-CLI, saving the model as `.glb`/`.fbx` too, and keeping build files for bug reports.

## Maps

The Map section of the panel builds Midtown or the Hideout from your own install, so it matches your game version.
Build Map exports the map with Source2Viewer and merges its ~30,000 objects into one mesh per 100 m cell and material
(same triangles, material UVs and materials), then saves `Documents/DeadlockRigs/maps/<map>/<map>.blend` with
`textures` and `lightmaps` folders next to it. Midtown takes about 11 minutes and needs about 12 GB free while
building; the result is about 3.5 GB. The panel shows when a game update has made the map outdated, and building again
skips an up-to-date map.

**Mods.** Mods you have installed for Deadlock (in its `addons` folders, as mod managers put them) that replace map
textures, such as posters and signs, are in the built map like they are in the game. Turn off Include Mods in the Map
section and rebuild for the map as Valve ships it.

Link Map and Append Map add the map's collection to the open scene, make the map's sky the scene's world and turn on
Game Look. The map is at its game position; rigs are at the origin, so move the rig or the map.

Link Map takes seconds: the map stays in its own file and your file points at it, so it can't be edited (moved,
materials, the node group values below) and the map's folder has to come along when the project moves. Append Map
copies the map into your file so all of it can be edited, but Blender stops responding for several minutes while it
appends (about 5 for Midtown).

**Lighting.** The map is lit the way the game lights it, from its own files: surfaces with lightmaps get the baked
lightmap, props get the map's light probe volumes (sampled per vertex when the map is built), unlit materials show
their colour as is, and signs and windows glow at the game's self-illumination strength. The baked light is emission
on top of each material, so lights you add still light the map; its strength is the Baked light strength value in the
`Deadlock baked light` node group. The map's sun comes in at the game's strength (the Hideout's); where the game has
none (Midtown), a Sun light from the game's sun direction lights only the rigs. The map's distance and height fog
are in the `Deadlock fog` node group (Fog strength), and the sky is the scene's world, seen by the camera only.

**Rig light.** Rigs are lit from the map's light probes per pixel, as the game lights heroes, so they pick up the
room they stand in and follow it as they move. Linking or appending a map does this for the rigs in the scene, and so
does appending a rig to a scene with a map; Light Rigs (Map section) does it by hand, next to its Strength. It is
emission on top of the rig's materials, like the map's baked light. In Cycles the map's baked light also bounces onto
rigs, so lower Strength there if they look too bright. Maps built before this need a rebuild.

**Game Look** (Map section) puts the game's tonemapping on the scene: its compositor gets the game's exposure and
filmic curve, and the view transform is set to Standard. That is most of what makes a render look like Deadlock
rather than Blender. Click it again to go back to your previous view settings.

Works in EEVEE and Cycles. Cycles loads the whole visible map with full size textures, which runs out of memory on
32 GB for Midtown: crop the map to the shot first and set Render > Simplify > Texture Limit to 1024 or 2048.

**Crop.** With a map in the scene, the Map section can cut it down to what a shot needs, which makes the viewport and
renders much faster: Add Crop Box and Crop to Boxes keep what is inside the boxes, Crop to Camera keeps what the scene
camera sees over the frame range. Objects across the edge keep the polygons that reach in. The Keep dropdown picks
map parts that are never cropped (skyline buildings, sky clouds, smoke, fog planes). Restore Map undoes it; Delete
Cropped removes the cropped parts for good to make the file smaller. A linked map crops the same way: the map's
collection is hidden and a `(cropped)` collection next to it holds what is kept, with the cut objects as local
copies.

## Help

Questions, bug reports and requests: the [DeadlockForge Discord](https://discord.gg/DkwQ2baWfb). When reporting a
build problem, turn on "Keep build files" in the add-on settings and include the build log (Show Build Log).

## Faces

| Hero | Face rig |
|---|---|
| Abrams, McGinnis | yes (from the game's face bones) |
| Apollo, Calico, Celeste, Drifter, Graves, Grey Talon, Holliday, Infernus, Kelvin, Lady Geist, Mina, Mirage, Nurse Harrow, Paige, Pocket, Silver, Solomon, The Doorman, Venator, Vindicta, Violet | yes |
| Billy, Rem, Victor | yes (from the game's face morphs) |
| everyone else | not yet |

![Violet's expressions](docs/faces_violet.png)
![Abrams' expressions](docs/faces_abrams.png)

| | |
|---|---|
| ![Violet's face controls](docs/violet_face_controls.png) | ![Abrams' face controls](docs/abrams_face_controls.png) |

## Hands

![Hand poses](docs/hands.png)

## Known gaps

- Sinclair's arm twist doesn't fully match his game animations.
- Baba only gets leg IK, and Bebop and Mo & Krill get no hand poses (their skeletons lack the bones).

## Build from source

The add-on runs `tools/pipeline.py`, which also works on its own. It needs Python 3 with numpy and scipy, Blender and
Deadlock; Blender and Deadlock are found through Steam, or pass `--blender` and `--pak`.

    python tools/pipeline.py violet             # build one hero
    python tools/pipeline.py all dev            # every hero, with the animation library and tests
    python tools/pipeline.py --help             # steps and options

Per-hero settings live in `heroes.json`. `python tools/release.py addon` packages the add-on.

## Credits

The map tools follow work by others in the Deadlock community:

- Miruneko (m1ru_neko), [Deadlock Map Optimizer](https://github.com/Miruneko0/Deadlock-Map-Optimizer): its Map Crop
  add-on showed cropping a map to boxes or the camera, which our Crop is modelled on.
- spenzo: his Blender port of Midtown, and merging the map's objects so Blender can handle it, which our map build
  does from your own install.
- [Cravvnn](https://github.com/cravvnn), [RenderLock](https://github.com/cravvnn/RenderLock): research into using
  the game's Source 2 lighting in Blender.
- [Source 2 Viewer / ValveResourceFormat contributors](https://github.com/ValveResourceFormat/ValveResourceFormat):
  the map export, and the lightmap, light probe and fog shaders our map lighting is ported from.

## License

GPL-3.0-or-later. Source2Viewer-CLI is from [Source 2 Viewer](https://github.com/ValveResourceFormat/ValveResourceFormat)
(MIT).
