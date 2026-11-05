"""Deadlock's volumetric fog (cs_volumetric_fog) as a Blender volume.

The game fills camera froxels with a density, the largest of every env_volumetric_fog_volume's
``saturate(edge ^ falloff * strength)`` (edge 1 inside, 0 at its faces), and an in-scatter colour: the map compiler's
baked fog irradiance volume times the controller's indirect strength, plus the lights with fog enabled, times the
tint. It then integrates emission and absorption along each pixel's ray, faded in near the camera. Here the irradiance
volume becomes an OpenVDB grid on a volume object covering the controller's box, and the volume shader computes the
same density from the fog volumes' boxes, absorbing by it and emitting density x in-scatter. Blender's lights can
scatter in it too (Light Scattering in the fog material, off by default: unshadowed in EEVEE they veil the room).

The game turns density into extinction per unit with an engine constant this can't read from the map, so it is a
setting (``Density Scale`` in the fog material), chosen so dl_hideout's haze matches the game's.
"""
import math
import os

import numpy as np

import bpy

from ..core import probes as probelib
from ..core import texture as vtex
from .lighting import srgb_to_linear
from .nodes import Builder

DENSITY_SCALE = 0.0025      # extinction per game unit at density 1
LIGHT_SCATTERING = 0.0      # Blender lights scattering in it too (off: the baked fog irradiance holds the static light)


def _box_matrix(e):
    """World (game units) -> the volume's 0-1 box space, as a 3x4 matrix."""
    origin = np.array(e.get_vector("origin", 3, [0, 0, 0]))
    rotation = probelib.rotation_matrix(e.get_vector("angles", 3, [0, 0, 0]))
    mins = np.array(e.get_vector("box_mins", 3, [-64, -64, -64]))
    maxs = np.array(e.get_vector("box_maxs", 3, [64, 64, 64]))
    size = np.maximum(maxs - mins, 1e-3)
    m = np.zeros((3, 4))
    m[:, :3] = rotation.T / size[:, None]
    m[:, 3] = (-(rotation.T @ origin) - mins) / size
    return m


def build_volumetric_fog(importer) -> bpy.types.Object | None:
    controller = importer._first("env_volumetric_fog_controller")
    if controller is None:
        return None
    volumes = [e for e in importer.entities if e.classname == "env_volumetric_fog_volume" and not e.get_bool("startdisabled")]
    if not volumes:
        return None

    scale = importer.scale
    global_strength = controller.get_float("fogstrength", 1.0)
    indirect_strength = controller.get_float("indirectstrength", 1.0) if controller.get_bool("indirectenabled", True) else 0.0
    tint = [srgb_to_linear(c / 255.0) for c in controller.get_vector("tintcolor", 3, [255, 255, 255])]

    # The baked irradiance, as a grid over the controller's box.
    origin = np.array(controller.get_vector("origin", 3, [0, 0, 0]))
    box_min = origin + np.array(controller.get_vector("box_mins", 3, [-512, -512, -512]))
    box_max = origin + np.array(controller.get_vector("box_maxs", 3, [512, 512, 512]))
    path = controller.get_str("fogirradiancevolume")
    vdb_path = None
    if path:
        try:
            vdb_path = _irradiance_vdb(importer, path, box_min, box_max)
        except Exception as error:
            importer.warn(f"volumetric fog irradiance failed: {error}")

    if vdb_path is not None:
        volume = bpy.data.volumes.new(f"{importer.map} fog")
        volume.filepath = vdb_path
        obj = bpy.data.objects.new(f"{importer.map} volumetric fog", volume)
    else:
        # No baked irradiance: a cube over the box, lit only by Blender's lights.
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        for c in cube.users_collection:
            c.objects.unlink(cube)
        obj = cube
        obj.name = f"{importer.map} volumetric fog"
        centre = (box_min + box_max) * 0.5
        obj.location = tuple(centre * scale)
        obj.scale = tuple((box_max - box_min) * 0.5 * scale)

    material = _fog_material(importer, volumes, global_strength, indirect_strength, tint, vdb_path is not None,
                             controller)
    if obj.type == "VOLUME":
        obj.data.materials.append(material)
    else:
        obj.data.materials.append(material)
    obj["dl_class"] = "env_volumetric_fog_controller"
    collection = importer.collections.get("Fog")
    if collection is None:
        collection = bpy.data.collections.new("Fog")
        importer.root.children.link(collection)
        importer.collections["Fog"] = collection
    collection.objects.link(obj)
    return obj


def _irradiance_vdb(importer, path, box_min, box_max) -> str | None:
    import openvdb
    from .map_import import _decode_dds
    tex, data = importer.textures.compiled_texture(path)
    if tex is None:
        return None
    if tex.block_compressed:
        slices = probelib.decode_volume_bc(tex, lambda dds: _decode_dds(importer.textures, dds, "fog"))
    else:
        slices = probelib._decode_volume_raw(tex)
    # (z, y, x, rgba) -> (x, y, z, rgb)
    grid_data = np.ascontiguousarray(np.transpose(slices[..., :3], (2, 1, 0, 3))).astype(np.float32)
    dims = np.array(grid_data.shape[:3], np.float64)
    voxel = (box_max - box_min) / dims * importer.scale

    grid = openvdb.Vec3SGrid()
    grid.copyFromArray(grid_data)
    grid.name = "irradiance"
    # Voxel centres at the texture's texel centres across the box.
    matrix = [[voxel[0], 0, 0, 0], [0, voxel[1], 0, 0], [0, 0, voxel[2], 0],
              [box_min[0] * importer.scale + voxel[0] * 0.5, box_min[1] * importer.scale + voxel[1] * 0.5,
               box_min[2] * importer.scale + voxel[2] * 0.5, 1]]
    grid.transform = openvdb.createLinearTransform(matrix)
    out = importer.textures.file_for(path, data, ".vdb")
    openvdb.write(out, grids=[grid])
    return out


def _fog_material(importer, volumes, global_strength, indirect_strength, tint, has_grid, controller):
    mat = bpy.data.materials.new(f"{importer.map} volumetric fog")
    mat["dl_map"] = importer.map
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    b = Builder(tree)
    scale = importer.scale

    b.column(0)
    geometry = b.node("ShaderNodeNewGeometry")
    position = b.vscale(geometry.outputs["Position"], 1.0 / scale)
    camera = b.node("ShaderNodeCameraData")

    # Density: the largest of the volumes' faded strengths.
    b.column(2)
    b.frame("Fog volumes")
    density = None
    for e in volumes:
        m = _box_matrix(e)
        lx = b.add(b.dot(position, tuple(m[0, :3])), m[0, 3])
        ly = b.add(b.dot(position, tuple(m[1, :3])), m[1, 3])
        lz = b.add(b.dot(position, tuple(m[2, :3])), m[2, 3])
        local = b.combine(lx, ly, lz)
        inside = b.mul(b.mul(b.math("COMPARE", lx, 0.5, 0.5), b.math("COMPARE", ly, 0.5, 0.5)), b.math("COMPARE", lz, 0.5, 0.5))
        centred = b.vmath("ABSOLUTE", b.vsub(b.vscale(local, 2.0), (1, 1, 1)))
        cx, cy, cz = b.separate(centred)
        if e.get_int("shape") == 1:
            edge = b.saturate(b.sub(1.0, b.vmath("LENGTH", centred)))
        else:
            edge = b.saturate(b.sub(1.0, b.math("MAXIMUM", cx, b.math("MAXIMUM", cy, cz))))
        faded = b.math("POWER", edge, max(e.get_float("falloffexponent", 1.0), 1e-3))
        d = b.mul(b.saturate(b.mul(faded, e.get_float("fogstrength", 1.0))), inside)
        density = d if density is None else b.math("MAXIMUM", density, d)
    b.frame(None)

    b.column(6)
    fade_start = controller.get_float("fadeinstart", 0.0)
    fade_end = max(controller.get_float("fadeinend", 0.0), fade_start + 1.0)
    distance = b.mul(camera.outputs["View Distance"], 1.0 / scale)
    fade = b.saturate(b.div(b.sub(distance, fade_start), fade_end - fade_start))
    density_scale = b.value(DENSITY_SCALE, "Density Scale")
    density_scale.node.name = "dl_fog_density"
    sigma = b.mul(b.mul(b.mul(density, fade), b.mul(density_scale, global_strength)), 1.0 / scale)

    b.column(8)
    if has_grid:
        attr = b.node("ShaderNodeAttribute", attribute_name="irradiance")
        inscatter = b.vmul(attr.outputs["Vector"], tuple(t * indirect_strength for t in tint))
    else:
        inscatter = b.rgb((0, 0, 0))
    emission = b.node("ShaderNodeEmission")
    b.set(emission.inputs["Color"], inscatter)
    b.set(emission.inputs["Strength"], sigma)
    absorption = b.node("ShaderNodeVolumeAbsorption")
    b.set(absorption.inputs["Density"], sigma)
    scatter = b.node("ShaderNodeVolumeScatter")
    b.set(scatter.inputs["Color"], tuple(tint))
    scattering = b.value(LIGHT_SCATTERING, "Light Scattering")
    scattering.node.name = "dl_fog_scattering"
    b.set(scatter.inputs["Density"], b.mul(sigma, scattering))
    b.column(10)
    add = b.node("ShaderNodeAddShader")
    b.set(add.inputs[0], emission.outputs[0])
    b.set(add.inputs[1], absorption.outputs[0])
    add2 = b.node("ShaderNodeAddShader")
    b.set(add2.inputs[0], add.outputs[0])
    b.set(add2.inputs[1], scatter.outputs[0])
    b.column(12)
    out = b.node("ShaderNodeOutputMaterial")
    out.hide = False
    b.set(out.inputs["Volume"], add2.outputs[0])
    try:
        mat.cycles.emission_sampling = "NONE"
    except AttributeError:
        pass
    return mat
