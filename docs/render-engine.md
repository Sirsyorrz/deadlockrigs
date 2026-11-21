# Deadlock Render Engine

[Back to the README](../README.md)

This is a separate part of the add-on and works differently from the rigs and maps. The rigs and maps rebuild
Deadlock's models and lighting inside Blender. The render engine doesn't rebuild anything: it starts **Deadlock
itself** and uses the game as the renderer. Blender drives the game's camera, and the game draws and lights your
rigs, game models and lights exactly as it would in a match.

## Starting it

Everything is under **Deadlock Rigs > Game** in the 3D View sidebar.

1. Press **Start Deadlock**, pick the map and the game resolution. 
   The game starts through Steam and Blender connects when it's up (1 to 2 minutes).
2. Once connected, the render engine switches to **Deadlock (game)**: the viewport shows the game from your Blender
   view, and F12 renders come from the game.
3. **Quit Deadlock** when you're done. Your game settings are put back.

**Connect to Deadlock** joins a session that's already running, one started from Blender earlier (it has the
camera bridge). Disconnect puts back the camera and HUD settings the add-on changed.

The game runs as a local offline session with `-insecure`: no matchmaking and no VAC. The add-on refuses to connect
to a normal game launch.

## In the game

- **Rigs**: visible rigs are put into the game as their hero and follow your posing and keyframes. Use the X next to
  one to take it out, **Pose in Game** to bring it back.
- **Add Game Model**: opens a thumbnail browser of the game's models. The chosen model is placed at the 3D cursor as
  an empty; move, rotate and scale the empty and the game follows.
- **Meshes**: your own meshes (and curves, text...) are in the game too, with each material's base colour or base
  colour image, lit by the map. Untick **Mesh in Game** on one to keep it out. Edits are sent once the mesh has been
  still for a moment. No collision, roughness, normal maps or rigging yet.
- **Add Game Light**: a spot light the game draws, shadows included. Any spot or point light with **Light in Game**
  on works the same way.
- **Map**: choose what of the map is in the game (Patrons, Walkers, Guardians, Mid-boss, troopers, neutral camps and
  more), then **Reload Map** to apply it. Reloading brings everything back as at the start of a match (10 to 15
  seconds).
- **Pause Game**: stops the game clock so effects, animation and NPCs hold still. F12 and animation renders still
  step it by the frame time.

## Memory

The game itself uses about 7 to 9 GB on Midtown, on top of Blender. Don't build maps while it's running, and don't
expect a full Midtown `.blend` and Midtown in the game to fit together in 32 GB.

## After a game update

The render engine hooks into the running game, so a Deadlock update can break it until the add-on is updated.
Rigs and maps aren't affected.

If something goes wrong, the Game panel shows the error and the path of its log. Send that log in the
[DeadlockForge Discord](https://discord.gg/DkwQ2baWfb).
