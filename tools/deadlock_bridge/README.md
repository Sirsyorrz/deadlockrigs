> Vendored into Deadlock Rigs from Divined's deadlock_bridge 0.1.0 (GPL-3.0-or-later), used for map particle effects (tools/map_particles.py, addon/particles.py). Changed: the bundled Windows library is left out (the Python decoders run instead); session.py and preferences.py take the host's Deadlock and cache folders; blender/particles.py builds into a host's turned map frame, takes host model and light sources and a fog group with an Alpha input, and keeps map effects with several control points in place.

# Deadlock Bridge

A Blender 5.2 extension that loads Deadlock maps straight from the game's VPKs: geometry, props, materials, baked
lighting, realtime lights, sky, fog, volumetric fog, particles and post processing, shaded the way the game shades
them. Nothing is extracted by hand and nothing from the game ships with the add-on; it reads your own install.

The shading follows the s&box Deadlock mount (`deadlockmount`), checked against the game's own shaders decompiled
with the Source 2 Viewer CLI (`deferred_lighting`, `post_process`, `cs_volumetric_fog`).

## Install

```
blender --command extension build --source-dir deadlock_bridge --output-dir dist
```

Then **Edit > Preferences > Get Extensions > Install from Disk** and pick `dist/deadlock_bridge-0.1.0.zip`.
Deadlock is found through Steam; set its folder in the add-on's preferences if it isn't.

## Use

**File > Import > Deadlock Map**, or **Import Deadlock Map** in the 3D view's **Deadlock** sidebar tab. Pick a map
(every map VPK in `game/citadel/maps`, community maps included) and what to bring in. The import sets up the scene to
render like the game (the camera starts at a spawn point); use **Rendered** viewport shading or F12, in EEVEE or
Cycles. The sidebar has per-map controls (baked light strength, detail distance, fog opacities, volumetric fog
density and light scattering, collection visibility), **EEVEE / Cycles** buttons that redo the render setup, and
**Particle Effects**: search any of the game's ~18,000 effects and add it at the 3D cursor, played as the particle
editor previews it (its preview control points, unset ones reading 1) and shown at its busiest moment unless you set
a time. Effects that only hold lights, projected decals or screen shakes say so instead of drawing.

The first import of a map converts its textures into the cache folder (about 90 s for dl_hideout); later imports of
it take about 30 s. A **Deadlock Report** text block lists counts and anything that failed.

## What is reproduced

* **Geometry**: the world nodes' aggregates with every fragment placed (mirroring, tints, texture coordinates scaled
  by model scale where materials ask), placed models with their skins and material overrides, entity props
  (`prop_dynamic` and friends) with render colour and alpha, all at full detail. With **Game detail levels** ticked
  on import, every level of detail comes in and switches by distance from the camera as the game does (trees become
  their round canopy blob, without the trunk, a few metres out), in renders and the viewport; the sidebar's Detail
  Distance then scales that distance. Meshoptimizer and LZ4 decoding
  run in a small native library (`native/dlnative.dll`, source next to it).
* **Materials** (node trees computing what the game writes to its G-buffer): `environment_blend` (up to three layers,
  reveal masks, colour correction and tint matrices, UV transforms, world and UV overlays, the self illumination
  packing), `environment_layer`, `environment_simple`, `pbr` (tint masks, vertex colour, self illumination with
  fresnel, texture transforms, alpha test with the distance boost, translucency, additive, unlit), `pbr` glass,
  `environment_parallax_layered` windows, `citadel_overlay` decals, and NPR props (`F_USE_NPR_LIGHTING`: the banded
  bounce light along the game's bounce normal, exposure control, transmissive colour, rim light, and the outline pass
  as an inverted hull). Normal maps decode as the game's hemi-octahedral ones in its tangent frame. Parallax windows
  put their interior layers on the second UV set when the material says so (one room per window, not per pane).
  Decals (`citadel_overlay`) are lifted off the wall they lie on and lit by that wall's lightmap, as the game's
  deferred pass lights them.
* **Baked lighting**, as `deferred_lighting` does it: the second order harmonic lightmaps (Deadlock's current 8.5
  format, a BC6H DC term and a shared luminance direction, with the ZH3 sharpening), the light probes' ambient cubes
  for props (blended across probe volumes like the game; baked per vertex, with their shadows of the stationary
  lights, for world geometry off the lightmaps such as trees), weighted by the game's split sum energy from its own
  BRDF lookup table. It is emitted light, so Blender doesn't relight it: no diffuse bounces, the sky only seen.
* **Realtime lights**: the sun and the dynamic and stationary light2 lights (static ones only exist in the lightmaps,
  as in the game); parallel barn lights (moonlight over a whole map) become Blender suns. Brightness is the game's
  (`colour * 2^brightness * 100^2 / d^2`, calibrated against Blender's own units). In Cycles, light node trees add the skirt fade, cone ramp and barn lights' frustum, soft edges and cookie.
  Stationary directional lights (the sun, parallel barns) light lightmapped surfaces inside their material through
  their channel of the baked shadows, as the game does, so they come in through the windows and around the buildings
  the map was baked with; Blender's lamps for them light only the props.
* **Shadows**: glass, windows, decals and other surfaces the game keeps out of its shadow maps don't cast shadows.
* **Reflections**: Blender's own (ray traced, plus EEVEE sphere probes placed at the game's cubemap volumes with
  their parallax boxes), reflecting the baked scene.
* **Sky** (`sky.vfx` cubemap at its exposure), **gradient and cubemap fog** (Valve's formulas, per material, camera
  rays only), **volumetric fog** (the fog volumes' density and the map's baked fog irradiance volume as an OpenVDB
  grid), **post processing** in the compositor: exposure, the `.vpost`'s filmic curve (matches the game's
  `post_process` shader to four decimals) and its bloom (threshold, four blur levels with its weights).
* **Particles**: every effect the map plays at load, simulated six seconds in (past their start-up, as you see them
  in a running game) and frozen: sprites, ropes and trails facing the camera or along their normals, with the
  particle shader's layers, sheets and blend modes, and model particles (the vent steam's blobs) as instances of the
  model. Lit by the light probes and the realtime lights (stationary ones behind the probes' baked shadows).

## Memory

Blender keeps only DXT1/DXT5 textures compressed on the GPU. Deadlock's are mostly BC7, BC4 and BC5, so with
**Compact GPU textures** (preferences, on by default) they are re-encoded to DXT1/DXT5 with mips: dl_hideout's
textures take about 2 GB of video memory instead of about 8 GB (1.1 GB compressed, plus 0.75 GB for the 8192 x 8192
lightmaps, which always load in full). Without it, a second GPU-heavy program can push EEVEE out of memory, which
shows as blank or flooded frames.
dl_midtown is much bigger (37 million world vertices): set **Texture size limit** to 1024 for it, as the s&box mount
does, and expect it to need well over 16 GB of RAM.

## Known differences from the game

* Volumetric fog: the game turns density into extinction with an engine constant the map doesn't store; the default
  (Density Scale 0.0025 per unit) is an estimate. Blender's lights don't scatter in it by default (unshadowed in EEVEE
  they veil whole rooms); the baked fog irradiance holds the static light.
* Realtime light shadows are Blender's (ray traced) rather than the game's baked masks, except the sun's on
  lightmapped surfaces.
* NPR props: direct light uses Blender's BSDF, not the game's toon ramp; the rim light isn't depth occluded.
* Rough reflections come from Blender's glossy rays, not the game's cubemaps normalised by the lightmaps.
* Particles are a snapshot, not animated; light particle renderers are skipped (listed in the report). Model
  particles need a map import (effects added from the sidebar draw their sprites, ropes and trails only).
* Entities that game logic changes at runtime (the hideout's finding-match effects, lights switched by I/O) load in
  their map state.
* Heroes and other animated models aren't imported as rigs; props use their bind pose.
