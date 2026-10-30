"""Deadlock's volumetric fog as a Blender volume (used by map_import.py). The fog volumes' density and the indirect
light baked into the controller's fog irradiance volume are put into an OpenVDB file on the irradiance volume's
voxels; the shader absorbs and emits like the game's froxel pass (ValveResourceFormat, Renderer/Shaders/
volumetric_fog.comp.slang and Renderer/SceneEnvironment/VolumetricFog.cs, MIT licence): extinction is density *
fog strength * a scattering scale per inch, faded in with the distance from the camera, and the fog in-scatters its lighting
times the extinction. Fog lit by lights and the sun isn't done.
"""
import glob
import os

import bpy
import numpy as np
from mathutils import Matrix

from map_lighting import Graph, UNIT

# VolumetricFog.ScatteringScale is 0.0022 (the controller's fog strength to extinction per inch); that leaves the
# Hideout far foggier than the game, this matches it in screenshots
SCATTERING_SCALE = 0.0022 * 0.2
# The share of the extinction that scatters Blender lights. The world would light the fog through it too, on top of
# the baked indirect light, so it's kept small and the fog lights are made brighter to make up for it (not by much
# more: EEVEE loses lights that bright in volumes).
SCATTER_SHARE = 0.1
# GetFogDensity never goes below this inside the fog
MIN_DENSITY = 0.001


def load_irradiance(folder, dims):
    """The slices Source2Viewer writes (one EXR per z), as [x, y, z, rgb]. The files' first row is the lowest y."""
    files = sorted(glob.glob(os.path.join(folder, '**', '*_z[0-9][0-9][0-9].exr'), recursive=True))
    if len(files) != dims[2]:
        return None
    out = np.empty((dims[0], dims[1], dims[2], 3), np.float32)
    for z, f in enumerate(files):
        img = bpy.data.images.load(f)
        w, h = img.size
        if (w, h) != (dims[0], dims[1]):
            bpy.data.images.remove(img)
            return None
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        # Blender's pixels start at the bottom row
        out[:, :, z] = px.reshape(h, w, 4)[::-1, :, :3].transpose(1, 0, 2)
    return out


def bake(fog, irradiance):
    """Extinction per metre (before the camera fade-in) and in-scattered light, on the irradiance voxels."""
    lo, hi, dims = np.array(fog['mins']), np.array(fog['maxs']), np.array(fog['dims'])
    voxel = (hi - lo) / dims
    density = np.zeros(tuple(dims), np.float32)
    light_scale = np.ones(tuple(dims) + (3,), np.float32)
    axes = [lo[a] + (np.arange(dims[a]) + 0.5) * voxel[a] for a in range(3)]
    for v in fog['volumes']:
        rot = np.array(v['rotation'])
        mins, maxs = np.array(v['mins']), np.array(v['maxs'])
        corners = np.array([[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1])
                            for z in (mins[2], maxs[2])]) @ rot.T + v['origin']
        first = np.clip(np.floor((corners.min(0) - lo) / voxel).astype(int), 0, dims)
        last = np.clip(np.ceil((corners.max(0) - lo) / voxel).astype(int), 0, dims)
        if np.any(last <= first):
            continue
        sl = tuple(slice(a, b) for a, b in zip(first, last))
        grid = np.stack(np.meshgrid(*[axes[a][sl[a]] for a in range(3)], indexing='ij'), -1)
        unit = ((grid - v['origin']) @ rot - mins) / (maxs - mins)
        inside = np.all((unit >= 0) & (unit <= 1), -1)
        edge = np.abs(unit * 2 - 1)
        edge = np.linalg.norm(edge, axis=-1) if v['sphere'] else edge.max(-1)
        value = np.where(inside, np.maximum(1 - edge, 0) ** v['exponent'] * v['strength'], 0).astype(np.float32)
        region = density[sl]
        stronger = value > region
        region[stronger] = value[stronger]
        light_scale[sl][stronger] = np.array(v['tint'], np.float32) * v['indirect']
    inside_any = density > 0
    density = np.where(inside_any, np.clip(density, MIN_DENSITY, 1.0), 0.0)
    extinction = (density * fog['strength'] * SCATTERING_SCALE / UNIT).astype(np.float32)
    light = irradiance * light_scale * (np.array(fog['tint'], np.float32) * fog['indirect'])
    # left empty outside the fog, so the VDB doesn't store it
    light[~inside_any] = 0
    return extinction, light.astype(np.float32)


def write_vdb(path, extinction, light):
    import openvdb
    density = openvdb.FloatGrid()
    density.copyFromArray(extinction, tolerance=0.0)
    density.name = 'density'
    density.saveFloatAsHalf = True
    color = openvdb.Vec3SGrid()
    color.copyFromArray(light)
    color.name = 'color'
    color.saveFloatAsHalf = True
    openvdb.write(path, grids=[density, color])


def fog_material(name, fog):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    g = Graph(nt)
    out = g.node('ShaderNodeOutputMaterial')
    density = g.node('ShaderNodeAttribute', attribute_name='density').outputs['Fac']
    color = g.node('ShaderNodeAttribute', attribute_name='color').outputs['Color']
    inches = g.math('DIVIDE', g.node('ShaderNodeCameraData').outputs['View Distance'], UNIT)
    start, end = fog['fade_in']
    start = start + end if start < 0 else start
    fade = g.math('MULTIPLY', g.math('SUBTRACT', inches, start), 1 / (end - start), clamp=True) \
        if end > 0 and start < end else 1.0
    # the froxels stop at the draw distance
    fade = g.math('MULTIPLY', fade, g.math('LESS_THAN', inches, fog['draw_distance']))
    extinction = g.math('MULTIPLY', density, fade)
    absorb = g.node('ShaderNodeVolumeAbsorption')
    absorb.inputs['Color'].default_value = (1, 1, 1, 1)
    g.links.new(g.math('MULTIPLY', extinction, 1 - SCATTER_SHARE), absorb.inputs['Density'])
    scatter = g.node('ShaderNodeVolumeScatter')
    scatter.inputs['Color'].default_value = (1, 1, 1, 1)
    g.links.new(g.math('MULTIPLY', extinction, SCATTER_SHARE), scatter.inputs['Density'])
    emit = g.node('ShaderNodeEmission')
    g.links.new(color, emit.inputs['Color'])
    g.links.new(extinction, emit.inputs['Strength'])
    add = g.node('ShaderNodeAddShader')
    g.links.new(absorb.outputs[0], add.inputs[0])
    g.links.new(emit.outputs[0], add.inputs[1])
    add2 = g.node('ShaderNodeAddShader')
    g.links.new(add.outputs[0], add2.inputs[0])
    g.links.new(scatter.outputs[0], add2.inputs[1])
    g.links.new(add2.outputs[0], out.inputs['Volume'])
    return mat


def add_volumetric_fog(coll, fog, folder, vdb_path, title):
    """A volume object with the map's volumetric fog, or None."""
    irradiance = load_irradiance(folder, fog['dims'])
    if irradiance is None or not fog['volumes']:
        return None
    extinction, light = bake(fog, irradiance)
    if not extinction.any():
        return None
    write_vdb(vdb_path, extinction, light)
    vol = bpy.data.volumes.new(f'{title} volumetric fog')
    vol.filepath = vdb_path
    vol.materials.append(fog_material(f'{title} volumetric fog', fog))
    vol.display.density = 0.02
    ob = bpy.data.objects.new(vol.name, vol)
    lo, hi, dims = np.array(fog['mins']), np.array(fog['maxs']), np.array(fog['dims'])
    voxel = (hi - lo) / dims * UNIT
    centre = (lo + 0.5 * (hi - lo) / dims) * UNIT
    # index space (Source x, y, z voxels) to Blender (y, -x, z)
    ob.matrix_world = Matrix(((0, voxel[1], 0, centre[1]), (-voxel[0], 0, 0, -centre[0]), (0, 0, voxel[2], centre[2]),
                              (0, 0, 0, 1)))
    ob.hide_select = True
    coll.objects.link(ob)
    return ob


def light_fog(coll, lights):
    """Only the lights that light the game's fog light the volume: each light leaves volumes alone, and those with a
    fog strength get a copy that lights only volumes, as bright as the game's fog lighting (colour times attenuation
    times fog strength, against Blender's isotropic 1/4pi phase and the scatter share). EEVEE doesn't apply light
    linking to volumes. Returns the number of fog lights."""
    count = 0
    for ob in lights:
        ob.data.volume_factor = 0.0
        strength = float(ob.get('dlr_fog', 0.0))
        if strength <= 0:
            continue
        copy = ob.copy()
        copy.data = ob.data.copy()
        copy.name = copy.data.name = f'{ob.name} fog'
        copy.data.energy = ob.data.energy * strength * 4 * np.pi / SCATTER_SHARE
        copy.data.volume_factor = 1.0
        copy.data.diffuse_factor = copy.data.specular_factor = copy.data.transmission_factor = 0.0
        # the shadow pool is already full on big maps, and volume shadows are off by default
        copy.data.use_shadow = False
        coll.objects.link(copy)
        count += 1
    return count
