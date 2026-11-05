"""Deadlock materials (vmat_c) as node trees that compute what the game's shaders write to their G-buffer, ending in
the map's ``DL Lighting`` group. Ports of the s&box mount's shaders, which follow the game's (decompiled) ones.

Vertex data the trees read (written by meshes.py):
    UVMap, UV2           texture coordinates (V flipped for Blender)
    dl_vcolor            byte colour: the object tint and vertex paint, combined per shader family (see
                         ``vertex_color_mode``), alpha = the tint's
    dl_blend             float colour: environment blend parameters (TEXCOORD4)
    dl_lightmap          float2: lightmap coordinates + 1, (0, 0) off the lightmaps
"""
import math

import bpy

from ..core import material as vmat
from ..core.content import compiled, normalize
from . import groups
from .nodes import Builder

TINT_STRENGTH = 0.85

ENVIRONMENT = {"environment_blend.vfx", "environment_layer.vfx", "environment_simple.vfx"}


def vertex_color_mode(m: vmat.CompiledMaterial | None) -> str:
    """How a draw call's tint and paint combine into dl_vcolor: 'env' (tint x lerp(1, paint, paint alpha)),
    'simple' (env with the tint faded by blend.w) or 'pbr' (tint x paint, alpha multiplied)."""
    if m is None:
        return "pbr"
    if m.shader == "environment_simple.vfx":
        return "simple"
    if m.shader in ENVIRONMENT or m.shader == "environment_parallax_layered.vfx":
        return "env"
    return "pbr"


class MaterialContext:
    def __init__(self, content, textures, map_name: str, lighting, unlit, additive, unit_scale: float, npr=None):
        self.content = content
        self.npr = npr
        self.textures = textures
        self.map_name = map_name
        self.lighting = lighting
        self.unlit = unlit
        self.additive = additive
        self.unit_scale = unit_scale
        self._compiled: dict[str, vmat.CompiledMaterial | None] = {}
        self._materials: dict[str, bpy.types.Material] = {}
        self.failed: dict[str, str] = {}
        self.shaders: dict[str, int] = {}

    def compiled(self, path: str) -> vmat.CompiledMaterial | None:
        key = normalize(path)
        if key not in self._compiled:
            data = self.content.try_read(compiled(path)) if path else None
            try:
                self._compiled[key] = vmat.CompiledMaterial(data) if data is not None else None
            except Exception as error:
                self.failed[key] = str(error)
                self._compiled[key] = None
        return self._compiled[key]

    def material(self, path: str) -> bpy.types.Material:
        key = normalize(path)
        mat = self._materials.get(key)
        if mat is not None:
            return mat
        source = self.compiled(path)
        name = key.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "missing"
        mat = bpy.data.materials.new(name)
        mat["dl_path"] = key
        mat["dl_map"] = self.map_name
        if source is None:
            _missing(mat)
        else:
            mat["dl_shader"] = source.shader
            self.shaders[source.shader] = self.shaders.get(source.shader, 0) + 1
            try:
                Translator(self, source, mat).build()
            except Exception as error:  # one odd material must not stop the map
                import traceback
                self.failed[key] = f"{error}\n{traceback.format_exc()}"
                mat.node_tree.nodes.clear() if mat.node_tree else None
                _missing(mat)
        self._materials[key] = mat
        return mat


def _missing(mat):
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    b = Builder(tree)
    bsdf = b.node("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 1.0, 1.0)
    b.column(2)
    out = b.node("ShaderNodeOutputMaterial")
    out.hide = False
    tree.links.new(bsdf.outputs[0], out.inputs[0])


class Translator:
    """Builds one material's node tree."""

    def __init__(self, ctx: MaterialContext, source: vmat.CompiledMaterial, mat: bpy.types.Material):
        self.ctx = ctx
        self.m = source
        self.mat = mat
        mat.use_nodes = True
        self.tree = mat.node_tree
        self.tree.nodes.clear()
        self.b = Builder(self.tree)
        self.output = None

    # Parameters ---------------------------------------------------------------------------------------------------

    def combo(self, name):
        return self.m.combo(name)

    def f(self, name, fallback=0.0):
        return self.m.floats.get(name, fallback)

    def i(self, name, fallback=0):
        return self.m.ints.get(name, fallback)

    def v(self, name, fallback=(0.0, 0.0, 0.0, 0.0)):
        value = self.m.vectors.get(name)
        if value is None:
            return tuple(fallback) + (0.0,) * (4 - len(fallback))
        return value

    def texture(self, param, color=False, uv=None, label=None, extension=None):
        """An image node for a texture parameter, or None when the material has none."""
        path = self.m.texture(param)
        if not path:
            return None
        image = self.ctx.textures.image(path, color=color)
        if image is None:
            return None
        node = self.b.image(image, uv, extension=extension or "REPEAT", label=label or param)
        return node

    # Shared pieces ------------------------------------------------------------------------------------------------

    def matrix_inputs(self, node, prefix, rows):
        """Sets a DL Color Matrix-like node's rows (3x4 numpy) on inputs '<prefix> R/G/B/Offset'."""
        node.inputs[f"{prefix} R"].default_value = tuple(rows[0, :3])
        node.inputs[f"{prefix} G"].default_value = tuple(rows[1, :3])
        node.inputs[f"{prefix} B"].default_value = tuple(rows[2, :3])
        node.inputs[f"{prefix} Offset"].default_value = (rows[0, 3], rows[1, 3], rows[2, 3])

    def color_matrices(self, layer: str, color_param: str | None):
        csb = self.v(f"g_vAlbedoContrastSaturationBrightness{layer}", (1, 1, 1, 0))
        pivot = (1.0, 1.0, 1.0)
        if csb[0] != 1.0 and color_param:
            tex, _ = self.ctx.textures.compiled_texture(self.m.texture(color_param) or "")
            if tex is not None:
                pivot = tex.reflectivity[:3]
        correct = vmat.color_correct(csb[:3], pivot)
        tint = self.v(f"g_vColorTint{layer}", (1, 1, 1, 0))
        linear = [vmat.srgb_to_linear(c) for c in tint[:3]]
        tint_rows = vmat.tint_matrix(linear, TINT_STRENGTH, self.i(f"g_nTextureColorTintMode{layer}"))
        return correct, tint_rows

    def apply_matrix(self, color, rows, label):
        node = self.b.group(groups.color_matrix(), label=label, Color=color)
        node.inputs["R"].default_value = tuple(rows[0, :3])
        node.inputs["G"].default_value = tuple(rows[1, :3])
        node.inputs["B"].default_value = tuple(rows[2, :3])
        node.inputs["Offset"].default_value = (rows[0, 3], rows[1, 3], rows[2, 3])
        return node.outputs[0]

    def vertex_color(self):
        """dl_vcolor times the object's tint (Object > Viewport Display > Color, linear; white by default)."""
        b = self.b
        attr = b.attribute("dl_vcolor")
        info = b.node("ShaderNodeObjectInfo")
        color = b.vmul(attr.outputs["Color"], info.outputs["Color"])
        alpha = b.mul(attr.outputs["Alpha"], info.outputs["Alpha"])
        return color, alpha

    def casts_shadows(self, blended: bool) -> bool:
        """Whether the game draws the material into its shadow maps: not when the material opts out, nor for what's
        see-through or laid over other surfaces (windows, glass, decals, additive and unlit effects, the black backdrop
        shells outside the windows)."""
        shader = self.m.shader
        if self.combo("F_DO_NOT_CAST_SHADOWS") or blended:
            return False
        if shader in ("environment_parallax_layered.vfx", "citadel_overlay.vfx", "black.vfx", "black_unlit.vfx",
                      "sky.vfx", "cosmic_veil.vfx", "spritecard.vfx", "projected_decals.vfx"):
            return False
        if self.combo("F_GLASS") or self.combo("F_UNLIT") or self.combo("F_ADDITIVE_BLEND"):
            return False
        return True

    def finish(self, shader_socket, blended=False, backfaces=None):
        b = self.b
        mat = self.mat
        if not self.casts_shadows(blended):
            # Shadow rays pass through (Cycles; EEVEE through transparent shadows).
            b.column(38)
            light_path = b.node("ShaderNodeLightPath")
            transparent = b.node("ShaderNodeBsdfTransparent")
            mix = b.node("ShaderNodeMixShader")
            b.set(mix.inputs[0], light_path.outputs["Is Shadow Ray"])
            b.set(mix.inputs[1], shader_socket)
            b.set(mix.inputs[2], transparent.outputs[0])
            shader_socket = mix.outputs[0]
            mat.use_transparent_shadow = True
            mat["dl_no_shadow"] = True
        b.column(40)
        out = b.node("ShaderNodeOutputMaterial")
        out.hide = False
        b.set(out.inputs["Surface"], shader_socket)
        if backfaces is None:
            backfaces = self.combo("F_RENDER_BACKFACES")
        mat.use_backface_culling = not backfaces
        mat.surface_render_method = "BLENDED" if blended else "DITHERED"
        if blended:
            mat.use_transparency_overlap = True
        try:
            mat.cycles.emission_sampling = "NONE"
        except AttributeError:
            pass
        return out

    def lit(self, albedo, metal, rough, normal, emission=None, alpha=None, ao=None, transmission=None, fog=None):
        b = self.b
        b.column(30)
        node = b.group(self.ctx.lighting, label="DL Lighting")
        node.hide = False
        b.set(node.inputs["Albedo"], albedo)
        b.set(node.inputs["Metalness"], metal)
        b.set(node.inputs["Roughness"], rough)
        b.set(node.inputs["Normal"], normal)
        b.set(node.inputs["Emission"], emission)
        b.set(node.inputs["Alpha"], alpha)
        b.set(node.inputs["Ambient Occlusion"], ao)
        b.set(node.inputs["Transmission"], transmission)
        b.set(node.inputs["Fog"], fog)
        return node.outputs[0]

    def fog_amount(self):
        return min(self.f("g_flGradientFogAmount", 1.0), self.f("g_flCubmapFogAmount", 1.0))

    # Dispatch -----------------------------------------------------------------------------------------------------

    def build(self):
        shader = self.m.shader
        translucent = self.combo("F_TRANSLUCENT")
        alpha_test = self.combo("F_ALPHA_TEST")
        if shader in ENVIRONMENT and not translucent and not alpha_test:
            self.environment()
        elif shader == "pbr.vfx" and self.combo("F_GLASS"):
            self.glass()
        elif shader == "environment_parallax_layered.vfx":
            self.parallax()
        elif shader == "citadel_overlay.vfx":
            self.overlay()
        elif shader in ("black.vfx", "black_unlit.vfx"):
            self.black()
        elif shader == "sky.vfx":
            self.black()
        else:
            self.pbr()

    # pbr.vfx --------------------------------------------------------------------------------------------------------

    def pbr(self):
        b = self.b
        m = self.m
        shader = m.shader
        unlit = self.combo("F_UNLIT") or shader.startswith("tools_") or shader in ("unlit.vfx", "spritecard.vfx")
        translucent = self.combo("F_TRANSLUCENT") or shader in ("projected_decals.vfx",)
        alpha_test = self.combo("F_ALPHA_TEST") and not translucent
        additive = translucent and self.combo("F_ADDITIVE_BLEND")
        npr = self.combo("F_USE_NPR_LIGHTING")
        environment = shader in ENVIRONMENT

        # Layer shaders drawn translucent or cut out read their first layer, like the mount.
        color_param = "g_tColor1" if m.texture("g_tColor1") else "g_tColor"
        normal_param = "g_tNormalRoughness1" if m.texture("g_tNormalRoughness1") else "g_tNormalRoughness"
        layer = "1"

        b.column(0)
        uv = b.uv("UVMap")
        uv2 = b.uv("UV2")
        albedo_uv = normal_uv = uv
        if self.combo("F_ENABLE_TEXTURE_TRANSFORMS"):
            albedo_uv = self._pbr_transform(uv, "g_vAlbedoTexcoordScale1", "g_vAlbedoTexcoordOffset1", "g_flAlbedoTexcoordRotation1")
            normal_uv = self._pbr_transform(uv, "g_vNormalTexcoordScale1", "g_vNormalTexcoordOffset1", "g_flNormalTexcoordRotation1")

        b.column(2)
        color_node = self.texture(color_param, True, albedo_uv, "Color")
        nr_node = self.texture(normal_param, False, normal_uv, "Normal Roughness")
        color = color_node.outputs["Color"] if color_node else (1.0, 1.0, 1.0)
        color_alpha = color_node.outputs["Alpha"] if color_node else 1.0

        tint_param = "g_tTintMaskRimLightMask" if npr else "g_tTintMask"
        tint_node = self.texture(tint_param, False, albedo_uv, "Tint Mask")
        if tint_node:
            tint_mask = b.separate_color(tint_node.outputs["Color"])[0]
        elif environment:
            tint_mask = color_alpha
        else:
            tint_mask = 1.0

        if alpha_test or translucent:
            metal_node = self.texture("g_tMetalness", False, albedo_uv, "Metalness")
            metal = b.separate_color(metal_node.outputs["Color"])[0] if metal_node else 0.0
        elif environment:
            packed = self.texture("g_tPacked1", False, albedo_uv, "Packed")
            metal = b.separate_color(packed.outputs["Color"])[1] if packed else 0.0
        else:
            metal = color_alpha if color_node else 0.0

        ao_node = self.texture("g_tAmbientOcclusion", False, uv2 if self.i("g_bUseSecondaryUvForAmbientOcclusion") else uv, "AO")
        ao = b.saturate(b.separate_color(ao_node.outputs["Color"])[0]) if ao_node else 1.0

        b.column(4)
        vc, vc_alpha = self.vertex_color()
        vc_strength = self.f(f"g_fVertexColorStrength{layer}", 1.0)
        amount = b.mul(b.lerp(1.0, tint_mask, float(self.i(f"g_bMaskVertexColorTint{layer}", 1) != 0)), vc_strength)
        apply_first = self.i("g_bApplyTintToVertexColors") != 0

        correct, tint_rows = self.color_matrices(layer, color_param)
        albedo = color
        if apply_first:
            albedo = b.clerp(albedo, b.clerp(albedo, vc, 1.0, "MULTIPLY"), amount)
        b.column(6)
        albedo = self.apply_matrix(albedo, correct, "Albedo Correct")
        tinted = self.apply_matrix(albedo, tint_rows, "Color Tint")
        albedo = b.clerp(albedo, tinted, b.lerp(1.0, tint_mask, float(self.i(f"g_bMaskColorTint{layer}", 1) != 0)))
        if not apply_first:
            albedo = b.clerp(albedo, b.clerp(albedo, vc, 1.0, "MULTIPLY"), amount)

        b.column(8)
        if nr_node:
            gradient = b.group(groups.normal_gradient(), Normal_Roughness=nr_node.outputs["Color"]).outputs[0]
            normal = b.group(groups.gradient_normal(), Gradient=gradient).outputs[0]
            roughness = b.separate_color(nr_node.outputs["Color"])[2]
        else:
            normal = None
            roughness = 1.0

        geometry = b.node("ShaderNodeNewGeometry")
        view = geometry.outputs["Incoming"]
        if self.combo("F_USE_FRONT_FACE_NORMALS_FOR_BACK_FACES") and self.combo("F_RENDER_BACKFACES"):
            # Blender turns the normal towards the viewer on back faces; foliage keeps the front face normal, so
            # both sides of a leaf card light alike.
            front = normal if normal is not None else geometry.outputs["Normal"]
            normal = b.vscale(front, b.madd(geometry.outputs["Backfacing"], -2.0, 1.0))
        shading_normal = normal if normal is not None else geometry.outputs["Normal"]

        # Self illumination replaces the albedo rather than adding to it.
        emission = None
        if self.combo("F_SELF_ILLUM"):
            b.column(10)
            illum_uv = uv2 if self.i("g_bUseSecondaryUvForSelfIllum1") else uv
            if self.combo("F_ENABLE_TEXTURE_TRANSFORMS"):
                illum_uv = self._pbr_transform(illum_uv, "g_vSelfIllumTexcoordScale1", "g_vSelfIllumTexcoordOffset1", "g_flSelfIllumTexcoordRotation1")
            mask_node = self.texture("g_tSelfIllumMask", False, illum_uv, "Self Illum Mask")
            mask = b.separate_color(mask_node.outputs["Color"])[0] if mask_node else 1.0
            si_tint = self.v("g_vSelfIllumTint1", (1, 1, 1))[:3]
            si_tint_linear = si_tint
            illum_tint = b.clerp(si_tint_linear, b.vmul(si_tint_linear, albedo), self.f("g_flSelfIllumAlbedoFactor1", 1.0))
            exponent = self.f("g_flSelfIllumFresnelMaskExponent", 5.0)
            fresnel = b.math("POWER", b.math("MAXIMUM", b.dot(view, shading_normal), 0.0), exponent)
            rng = self.v("g_vSelfIllumFresnelMaskRange1", (0, 0))
            span = rng[1] - rng[0]
            if span != 0:
                fresnel_blend = b.saturate(b.div(b.sub(fresnel, rng[0]), span))
            else:
                fresnel_blend = b.math("GREATER_THAN", fresnel, rng[0])
            fresnel_mask = b.clerp(self.v("g_vSelfIllumFresnelMaskTint1", (1, 1, 1))[:3], (1, 1, 1), fresnel_blend)
            illum_tint = b.vmul(illum_tint, fresnel_mask)
            albedo = b.vmul(albedo, b.clerp((1, 1, 1), fresnel_mask, mask))
            scale = self.f("g_flSelfIllumScale1", 0.0)
            albedo = b.clerp(albedo, illum_tint, b.mul(mask, max(0.0, min(1.0, scale))))
            emission = b.vscale(albedo, b.mul(mask, scale))

        # Opacity.
        b.column(12)
        alpha = None
        if translucent:
            power = self.f("g_flAlphaAnglePower1", 0.0)
            fade = 1.0
            if power != 0:
                sb = (-1.0, 1.0) if power < 0 else (1.0, 0.0)
                fade = b.math("POWER", b.math("MAXIMUM", b.madd(b.math("ABSOLUTE", b.dot(view, shading_normal)), sb[0], sb[1]), 0.001), abs(power))
            alpha = b.saturate(b.mul(b.mul(b.mul(vc_alpha, color_alpha), fade), self.f("g_flOpacityScale1", 1.0)))
        elif alpha_test:
            camera = b.node("ShaderNodeCameraData")
            distance = b.mul(camera.outputs["View Distance"], 1.0 / self.ctx.unit_scale)
            boost = b.lerp(1.0, self.f("g_flAlphaBoostStrength", 3.0),
                           b.saturate(b.div(distance, max(self.f("g_flAlphaBoostDistance", 20000.0), 1.0))))
            raw = b.mul(b.mul(color_alpha, vc_alpha), boost)
            alpha = b.math("GREATER_THAN", raw, self.f("g_flAlphaTestReference1", self.f("g_flAlphaTestReference", 0.5)))

        fog = self.fog_amount()
        if unlit or additive:
            scale = self.f("g_flUnlitAdditiveScale1", 1.0) if additive else 1.0
            color_out = b.vscale(albedo, scale) if scale != 1.0 else albedo
            if emission is not None:
                color_out = b.vadd(color_out, emission)
            b.column(30)
            group = self.ctx.additive if additive else self.ctx.unlit
            node = b.group(group, Color=color_out, Alpha=alpha if alpha is not None else 1.0, Fog=fog)
            node.hide = False
            self.finish(node.outputs[0], blended=translucent)
            return

        if npr and self.ctx.npr is not None:
            b.column(30)
            node = b.group(self.ctx.npr, label="DL NPR Lighting")
            node.hide = False
            rim_mask = b.separate_color(tint_node.outputs["Color"])[1] if tint_node else 1.0
            transmissive = self.texture("g_tNprTransmissiveColor", True, albedo_uv, "Transmissive")
            for name, value in (("Albedo", albedo), ("Metalness", metal), ("Roughness", roughness), ("Normal", normal),
                                ("Emission", emission), ("Alpha", alpha), ("Ambient Occlusion", ao), ("Rim Mask", rim_mask),
                                ("Transmissive", transmissive.outputs["Color"] if transmissive else None), ("Fog", fog)):
                b.set(node.inputs[name], value)
            if not (self.combo("F_DISABLE_NPR_OUTLINE") or self.combo("F_CLOAK") or translucent):
                thickness = self.f("g_flOverrideNprOutlineThickness", 0.2) if self.combo("F_OVERRIDE_NPR_OUTLINE") else 0.1
                self.mat["dl_npr_outline"] = thickness
                if self.combo("F_SOLID_COLOR_OUTLINE"):
                    self.mat["dl_npr_outline_tint"] = [vmat.srgb_to_linear(c) for c in self.v("g_vSolidOutlineTint", (0.1, 0.1, 0.1))[:3]]
            self.finish(node.outputs[0], blended=translucent)
            return

        shader_out = self.lit(albedo, metal, roughness, normal, emission, alpha, ao, fog=fog)
        self.finish(shader_out, blended=translucent)

    def _pbr_transform(self, uv, scale, offset, rotation):
        s = self.v(scale, (1, 1))
        o = self.v(offset, (0, 0))
        node = self.b.group(groups.pbr_uv_transform(), UV=uv)
        node.inputs["Scale"].default_value = (s[0], s[1], 0.0)
        node.inputs["Offset"].default_value = (o[0], o[1], 0.0)
        node.inputs["Rotation"].default_value = self.f(rotation, 0.0)
        return node.outputs[0]

    # environment_blend / layer / simple ------------------------------------------------------------------------------

    def uv_transform_values(self, param):
        value = self.v(param, (1, 0, 0, 0))
        scale = value[0] if value[0] != 0 else 1.0
        angle = math.radians(value[1])
        return (scale * math.sin(angle), scale * math.cos(angle), 0.0), (value[2] / scale, value[3] / scale, 0.0)

    def transformed_uv(self, uv, param):
        sin_cos, offset = self.uv_transform_values(param)
        if sin_cos == (0.0, 1.0, 0.0) and offset == (0.0, 0.0, 0.0):
            return uv
        node = self.b.group(groups.uv_transform(), UV=uv)
        node.inputs["Sin Cos"].default_value = sin_cos
        node.inputs["Offset"].default_value = offset
        return node.outputs[0]

    def environment(self):
        b = self.b
        m = self.m
        simple = m.shader == "environment_simple.vfx"
        blend = m.shader == "environment_blend.vfx"
        layers = (3 if self.combo("F_LAYER_COUNT") else 2) if blend else 1

        b.column(0)
        uv = b.uv("UVMap")
        uv2 = b.uv("UV2")
        vc, _ = self.vertex_color()
        params = b.attribute("dl_blend")
        bp_x, bp_y, bp_z = b.separate_color(params.outputs["Color"])

        if simple:
            b.column(2)
            node = self.texture("g_tColor", True, uv, "Color") or self.texture("g_tColor1", True, uv, "Color")
            color = node.outputs["Color"] if node else (1, 1, 1)
            alpha = node.outputs["Alpha"] if node else 1.0
            b.column(4)
            albedo = b.clerp(color, b.clerp(color, vc, 1.0, "MULTIPLY"), alpha)
            illum = b.vscale(albedo, bp_z)
            pack = b.group(groups.env_composite(), Albedo=albedo, Illum=illum, Self_Illum=bp_z)
            shader = self.lit(pack.outputs["Albedo"], bp_y, bp_x, None, pack.outputs["Emission"], fog=self.fog_amount())
            self.finish(shader)
            return

        results = []
        for layer in range(1, layers + 1):
            n = str(layer)
            b.column(2 + (layer - 1) * 0)
            b.frame(f"Layer {n}")
            layer_uv = self.transformed_uv(uv, f"g_vScaleRotationTranslation{n}")
            if self.combo(f"F_LAYER{n}_BIPLANAR"):
                layer_uv = self._biplanar_uv(self.f(f"g_flBiPlanarTiling{n}", self.f("g_flBiPlanarTiling", 196.0)))
            color = self.texture(f"g_tColor{n}", True, layer_uv, f"Color {n}")
            nr = self.texture(f"g_tNormalRoughness{n}", False, layer_uv, f"Normal Roughness {n}")
            packed = self.texture(f"g_tPacked{n}", False, layer_uv, f"Packed {n}")

            b.column(6)
            node = b.group(groups.env_layer(), label=f"Layer {n}")
            node.hide = False
            if color:
                b.set(node.inputs["Color"], color.outputs["Color"])
                b.set(node.inputs["Alpha"], color.outputs["Alpha"])
            if nr:
                b.set(node.inputs["Normal Roughness"], nr.outputs["Color"])
            else:
                node.inputs["Normal Roughness"].default_value = (0.5, 0.5, 1.0, 1.0)
            if packed:
                b.set(node.inputs["Packed"], packed.outputs["Color"])
            b.set(node.inputs["Vertex Color"], vc)
            correct, tint_rows = self.color_matrices(n, f"g_tColor{n}")
            self.matrix_inputs(node, "Correct", correct)
            self.matrix_inputs(node, "Tint", tint_rows)
            node.inputs["Mask Color Tint"].default_value = float(self.i(f"g_bMaskColorTint{n}", 1) != 0)
            node.inputs["Mask Vertex Color Tint"].default_value = float(self.i(f"g_bMaskVertexColorTint{n}", 1) != 0)
            node.inputs["Model Tint"].default_value = float(self.i(f"g_bModelTint{n}", 1) != 0)
            node.inputs["Vertex Color Strength"].default_value = self.f(f"g_fVertexColorStrength{n}", 1.0)
            node.inputs["Normal Strength"].default_value = self.f(f"g_fLayerNormalStrength{n}", 1.0)
            node.inputs["Self Illum Scale"].default_value = self.f(f"g_fLayerSelfIllumScale{n}", 1.0)
            for key, param in (("Roughness CB", "g_vLayerRoughnessContrastBrightness"), ("Tint Mask CB", "g_vLayerTintMaskContrastBrightness"),
                               ("Metalness CB", "g_vLayerMetalnessContrastBrightness")):
                value = self.v(f"{param}{n}", (1, 0))
                node.inputs[key].default_value = (value[0], value[1], 0.0)

            si_tint = self.v(f"g_vLayerSelfIllumTint{n}", (1, 1, 1))[:3]
            b.column(8)
            illum = b.vmul(b.vscale(node.outputs["Albedo"], node.outputs["Self Illum"]), si_tint)
            b.frame(None)
            results.append({"albedo": node.outputs["Albedo"], "gradient": node.outputs["Gradient"],
                            "roughness": node.outputs["Roughness"], "metal": node.outputs["Metalness"],
                            "self_illum": node.outputs["Self Illum"], "illum": illum, "tint_mask": node.outputs["Tint Mask"]})
            b.column(2)

        # Weights of the layers above the first: their reveal masks against the paint.
        b.column(10)
        blend2 = bp_x
        blend3 = 0.0
        if layers > 2:
            blend3 = bp_y
            blend2 = b.select(b.math("LESS_THAN", blend3, 1.0), blend2, 0.0)

        # The UV overlay's reveal pushes layer 2 in.
        overlay_uv = None
        if self.combo("F_UV_OVERLAY"):
            secondary = self.i("g_bUseSecondaryUvForOverlay", 1) != 0
            overlay_uv = self.transformed_uv(uv2 if secondary else uv, "g_vOverlayScaleRotationTranslation")
            reveal_overlay = self.texture("g_tRevealOverlay", False, overlay_uv, "Reveal Overlay")
            if reveal_overlay:
                r = b.separate_color(reveal_overlay.outputs["Color"])[0]
                if self.i("g_bRevealOverlayInvert2"):
                    r = b.sub(1.0, r)
                blend2 = b.saturate(b.add(blend2, b.mul(r, self.f("g_flRevealOverlayStrength2", 1.0))))

        weights = []
        for layer in range(2, layers + 1):
            n = str(layer)
            reveal_uv = self.transformed_uv(uv2 if self.i(f"g_bRevealUseSecondaryUv{n}") else uv, f"g_vRevealScaleRotationTranslation{n}")
            mask_node = self.texture(f"g_tRevealMask{n}", False, reveal_uv, f"Reveal Mask {n}")
            mask = b.separate_color(mask_node.outputs["Color"])[0] if mask_node else 0.5
            node = b.group(groups.reveal_weight(), label=f"Reveal {n}", Mask=mask, Blend=blend2 if layer == 2 else blend3)
            node.inputs["Invert"].default_value = float(self.i(f"g_bRevealInvert{n}") != 0)
            node.inputs["Offset"].default_value = self.f(f"g_flRevealOffset{n}", 0.0)
            node.inputs["Softness"].default_value = self.f(f"g_flRevealSoftness{n}", 0.125)
            weights.append(node.outputs[0])

        # World overlays: colour modulation projected on the world, shared by every layer.
        b.column(12)
        world_overlays = self.i("F_WORLD_OVERLAY")
        if world_overlays > 0:
            for k in range(min(world_overlays, 2)):
                param = "g_tColorWorldOverlay" if k == 0 else "g_tColorWorldOverlay2"
                if world_overlays >= 2:
                    tiling = self.v("g_vWorldOverlayTiling", (512, 512))[k]
                    bright = self.v("g_vWorldOverlayBrightnessContrast", (1, 1))[k]
                    dark = self.v("g_vWorldOverlayDarknessContrast", (1, 1))[k]
                    strengths = [self.v(f"g_vWorldOverlayStrengths{L}", (1, 0))[k] for L in (1, 2, 3)]
                else:
                    tiling = self.f("g_flWorldOverlayTiling", 512.0)
                    bright = self.f("g_flWorldOverlayBrightnessContrast", 1.0)
                    dark = self.f("g_flWorldOverlayDarknessContrast", 1.0)
                    strengths = [self.f(f"g_flWorldOverlayStrength{L}", 1.0) for L in (1, 2, 3)]
                node = self._world_overlay(param, tiling)
                if node is None:
                    continue
                factor = self._overlay_factor(node.outputs["Color"], dark, bright)
                for L, res in enumerate(results):
                    mode = self._per_layer_int("g_nWorldOverlayTintMask", L + 1, 1)
                    mask = self._overlay_mask(mode, res["tint_mask"])
                    res["albedo"] = b.clerp(res["albedo"], b.vmul(res["albedo"], factor), b.mul(mask, strengths[L]))

        # The UV overlay.
        if self.combo("F_UV_OVERLAY") and overlay_uv is not None:
            oc = self.texture("g_tColorOverlay", True, overlay_uv, "Overlay Color")
            onr = self.texture("g_tNormalRoughnessOverlay", False, overlay_uv, "Overlay Normal")
            if oc:
                factor = self._overlay_factor(oc.outputs["Color"], self.f("g_flOverlayDarknessContrast", 1.0), self.f("g_flOverlayBrightnessContrast", 1.0))
                grad = b.group(groups.normal_gradient(), Normal_Roughness=onr.outputs["Color"]).outputs[0] if onr else None
                rough = b.sub(b.mul(b.separate_color(onr.outputs["Color"])[2], 2.0), 1.0) if onr else 0.0
                for L, res in enumerate(results):
                    mode = self._per_layer_int("g_nOverlayTintMask", L + 1, 0)
                    mask = self._overlay_mask(mode, res["tint_mask"])
                    strength = self._per_layer_float("g_flOverlayStrength", L + 1, 1.0)
                    res["albedo"] = b.clerp(res["albedo"], b.vmul(res["albedo"], factor), b.mul(mask, strength))
                    if grad is not None:
                        ns = self._per_layer_float("g_flOverlayNormalStrength", L + 1, 1.0)
                        res["gradient"] = b.vadd(res["gradient"], b.vscale(grad, b.mul(mask, ns)))
                    rs = self._per_layer_float("g_flOverlayRoughnessStrength", L + 1, 0.0)
                    if rs != 0:
                        res["roughness"] = b.saturate(b.add(res["roughness"], b.mul(b.mul(rough, rs), mask)))

        # Composite.
        b.column(16)
        result = dict(results[0])
        for weight, res in zip(weights, results[1:]):
            result["albedo"] = b.clerp(result["albedo"], res["albedo"], weight)
            result["gradient"] = b.vlerp(result["gradient"], res["gradient"], weight)
            result["roughness"] = b.lerp(result["roughness"], res["roughness"], weight)
            result["metal"] = b.lerp(result["metal"], res["metal"], weight)
            result["self_illum"] = b.lerp(result["self_illum"], res["self_illum"], weight)
            result["illum"] = b.clerp(result["illum"], res["illum"], weight)

        b.column(20)
        normal = b.group(groups.gradient_normal(), Gradient=result["gradient"]).outputs[0]
        pack = b.group(groups.env_composite(), Albedo=result["albedo"], Illum=result["illum"], Self_Illum=result["self_illum"])
        shader = self.lit(pack.outputs["Albedo"], result["metal"], result["roughness"], normal, pack.outputs["Emission"],
                          fog=self.fog_amount())
        self.finish(shader)

    def _per_layer_int(self, prefix, layer, fallback):
        key = f"{prefix}{layer}"
        if key in self.m.ints:
            return self.m.ints[key]
        return int(self.m.floats.get(key, fallback))

    def _per_layer_float(self, prefix, layer, fallback):
        key = f"{prefix}{layer}"
        if key in self.m.ints:
            return float(self.m.ints[key])
        return self.m.floats.get(key, fallback)

    def _overlay_mask(self, mode, tint_mask):
        if mode == 0:
            return 1.0
        if mode == 1:
            return tint_mask
        return self.b.sub(1.0, tint_mask)

    def _overlay_factor(self, color, dark, bright):
        """OverlayFactor: a texel as a signed multiplier around 0.5, each side shaped by its contrast."""
        b = self.b
        channels = []
        for c in b.separate_color(color):
            s = b.sub(b.mul(c, 2.0), 1.0)
            brighten = b.mul(b.sub(1.0, b.math("POWER", b.math("MAXIMUM", b.sub(1.0, b.math("MAXIMUM", s, 0.0)), 1e-6), bright)), bright)
            darken = b.mul(b.sub(b.math("POWER", b.math("MAXIMUM", b.add(1.0, b.math("MINIMUM", s, 0.0)), 1e-6), dark), 1.0), dark)
            channels.append(b.math("MAXIMUM", b.add(b.add(brighten, darken), 1.0), 0.0))
        return b.combine_color(*channels)

    def _biplanar_uv(self, tiling):
        """A planar projection of the world position (game units) on the normal's dominant plane."""
        b = self.b
        geometry = b.node("ShaderNodeNewGeometry")
        scale = 1.0 / (self.ctx.unit_scale * (tiling or 1.0))
        p = b.vscale(geometry.outputs["Position"], scale)
        px, py, pz = b.separate(p)
        nx, ny, nz = b.separate(geometry.outputs["Normal"])
        ax, ay, az = b.math("ABSOLUTE", nx), b.math("ABSOLUTE", ny), b.math("ABSOLUTE", nz)
        x_major = b.mul(b.math("GREATER_THAN", ax, ay), b.math("GREATER_THAN", ax, az))
        y_major = b.mul(b.sub(1.0, x_major), b.math("GREATER_THAN", ay, az))
        # x major: (y, z); y major: (z, x) swapped to (x, z); z major: (x, y). V flipped for Blender.
        u = b.select(x_major, py, b.select(y_major, px, px))
        v = b.select(x_major, pz, b.select(y_major, pz, py))
        return b.combine(u, v, 0.0)

    def _world_overlay(self, param, tiling):
        b = self.b
        path = self.m.texture(param)
        image = self.ctx.textures.image(path, color=True) if path else None
        if image is None:
            return None
        coords = b.node("ShaderNodeTexCoord")
        space = coords.outputs["Object"] if self.combo("F_WORLD_OVERLAY_USE_OBJECT_SPACE") else None
        if space is None:
            geometry = b.node("ShaderNodeNewGeometry")
            space = geometry.outputs["Position"]
        vector = b.vscale(space, 1.0 / (self.ctx.unit_scale * (tiling or 1.0)))
        node = b.image(image, vector, label=param)
        node.projection = "BOX"
        node.projection_blend = 0.35
        return node

    # pbr F_GLASS ----------------------------------------------------------------------------------------------------

    def glass(self):
        b = self.b
        b.column(0)
        uv = b.uv("UVMap")
        b.column(2)
        color_node = self.texture("g_tColor", True, uv, "Color")
        nr = self.texture("g_tNormalRoughness", False, uv, "Normal Roughness")
        glass_node = self.texture("g_tGlass", False, uv, "Glass")
        tint_node = self.texture("g_tTintMask", False, uv, "Tint Mask")
        color = color_node.outputs["Color"] if color_node else (1, 1, 1)
        tint_mask = b.separate_color(tint_node.outputs["Color"])[0] if tint_node else 1.0

        b.column(4)
        correct, tint_rows = self.color_matrices("1", "g_tColor")
        albedo = self.apply_matrix(color, correct, "Albedo Correct")
        albedo = b.clerp(albedo, self.apply_matrix(albedo, tint_rows, "Color Tint"), tint_mask)
        vc, _ = self.vertex_color()
        albedo = b.clerp(albedo, b.vmul(albedo, vc), b.mul(tint_mask, self.f("g_fVertexColorStrength1", 1.0)))
        transmission = b.separate_color(glass_node.outputs["Color"])[0] if glass_node else 1.0

        b.column(6)
        if nr:
            gradient = b.group(groups.normal_gradient(), Normal_Roughness=nr.outputs["Color"]).outputs[0]
            normal = b.group(groups.gradient_normal(), Gradient=gradient).outputs[0]
            roughness = b.separate_color(nr.outputs["Color"])[2]
        else:
            normal, roughness = None, 0.1

        emission = None
        if self.combo("F_SELF_ILLUM"):
            mask_node = self.texture("g_tSelfIllumMask", False, uv, "Self Illum Mask")
            mask = b.separate_color(mask_node.outputs["Color"])[0] if mask_node else 1.0
            emission = b.vscale(albedo, b.mul(mask, self.f("g_flSelfIllumScale1", 0.0)))

        lit = self.lit(albedo, 0.0, roughness, normal, emission, transmission=transmission, fog=self.fog_amount())

        # What's behind shows through, tinted more the more steeply it's looked through (albedo ^ (1 / N.V)).
        b.column(34)
        geometry = b.node("ShaderNodeNewGeometry")
        n_dot_v = b.math("MAXIMUM", b.dot(geometry.outputs["Normal"], geometry.outputs["Incoming"]), 0.01)
        tint = b.vmath("POWER", b.vmath("MAXIMUM", albedo, (0.01, 0.01, 0.01)), b.combine(b.div(1.0, n_dot_v), b.div(1.0, n_dot_v), b.div(1.0, n_dot_v)))
        through = b.vscale(b.vmath("MINIMUM", tint, (1, 1, 1)), b.mul(transmission, transmission))
        transparent = b.node("ShaderNodeBsdfTransparent")
        b.set(transparent.inputs["Color"], through)
        add = b.node("ShaderNodeAddShader")
        b.set(add.inputs[0], lit)
        b.set(add.inputs[1], transparent.outputs[0])
        self.finish(add.outputs[0], blended=True)

    # citadel_overlay ------------------------------------------------------------------------------------------------

    def overlay(self):
        b = self.b
        b.column(0)
        uv = b.uv("UVMap")

        def offset_uv(param):
            o = self.v(param, (0, 0))
            if o[0] == 0 and o[1] == 0:
                return uv
            return b.vadd(uv, (o[0], -o[1], 0.0))

        b.column(2)
        color_node = self.texture("g_tColor", True, offset_uv("g_vAlbedoTexcoordUVOffset"), "Color")
        nr = self.texture("g_tNormalRoughness", False, offset_uv("g_vNormalAndRoughnessTexcoordUVOffset"), "Normal Roughness")
        mask_node = self.texture("g_tSelfIllumTintMask", False, offset_uv("g_vSelfIllumTexcoordOffset"), "Self Illum Tint Mask")
        color = color_node.outputs["Color"] if color_node else (1, 1, 1)
        alpha = color_node.outputs["Alpha"] if color_node else 0.0
        si_mask, tint_mask, _ = b.separate_color(mask_node.outputs["Color"]) if mask_node else (0.0, 1.0, 0.0)

        b.column(4)
        vc, vc_alpha = self.vertex_color()
        tinted = b.vmul(b.vmul(color, self.v("g_vColorTint", (1, 1, 1))[:3]), vc)
        albedo = b.clerp(color, tinted, tint_mask)
        translucency = b.mul(vc_alpha, alpha)

        si_tint = self.v("g_vSelfIllumTint", (1, 1, 1))[:3]
        scale = self.f("g_flSelfIllumScale", 0.0)
        illum = b.clerp(si_tint, b.vmul(si_tint, albedo), self.f("g_flSelfIllumAlbedoFactor", 1.0))
        albedo = b.clerp(albedo, illum, b.mul(si_mask, max(0.0, min(1.0, scale))))
        glow = b.mul(b.saturate(b.mul(b.mul(si_mask, scale), 0.1)), 10.0)
        emission = b.vscale(albedo, glow)

        b.column(6)
        normal = None
        roughness = 1.0
        if nr:
            gradient = b.group(groups.normal_gradient(), Normal_Roughness=nr.outputs["Color"]).outputs[0]
            normal = b.group(groups.gradient_normal(), Gradient=gradient).outputs[0]
            roughness = b.separate_color(nr.outputs["Color"])[2]

        in_range = self.v("g_vColorTranslucencyInputRange", (0, 1))
        out_range = self.v("g_vColorTranslucencyOutputRange", (0, 1))
        span = in_range[1] - in_range[0]
        opacity = b.saturate(b.div(b.sub(translucency, in_range[0]), span if span != 0 else 1e-6))
        opacity = b.add(b.mul(opacity, out_range[1] - out_range[0]), out_range[0])

        shader = self.lit(albedo, 0.0, roughness, normal, emission, alpha=opacity, fog=self.fog_amount())
        self.finish(shader, blended=True)

    # environment_parallax_layered -------------------------------------------------------------------------------------

    def parallax(self):
        b = self.b
        b.column(0)
        uv = b.uv("UVMap")
        geometry = b.node("ShaderNodeNewGeometry")
        view = geometry.outputs["Incoming"]
        normal_geo = geometry.outputs["Normal"]

        def frame(uv_map):
            """A UV map's tangent and bitangent, through the normal map node."""
            axes = []
            for color in ((1.0, 0.5, 0.5, 1.0), (0.5, 1.0, 0.5, 1.0)):
                node = b.node("ShaderNodeNormalMap", space="TANGENT", uv_map=uv_map)
                node.inputs["Color"].default_value = color
                axes.append(node.outputs[0])
            return b.dot(view, axes[0]), b.dot(view, axes[1])

        b.column(2)
        inverse_n_dot_v = b.div(1.0, b.math("MAXIMUM", b.dot(view, normal_geo), 0.02))
        view_primary = frame("UVMap")
        # Layers can sit on the second UV set (one room per window where the first tiles per pane), the view then
        # taken in that set's own tangent frame.
        secondary = {name: self.i(f"g_b{name}UseSecondaryUV") != 0
                     for name in ("Layer2", "Layer3", "ReflectionOcclusion")}
        uv_secondary = b.uv("UV2") if any(secondary.values()) else None
        view_secondary = frame("UV2") if uv_secondary is not None else None

        def clamped(v, low, high):
            return tuple(b.math("MINIMUM", b.math("MAXIMUM", c, low), high) for c in v)

        def shifted(base, v, offset_param):
            offset = b.mul(self.f(offset_param, 0.0) / 1024.0, inverse_n_dot_v)
            return b.vsub(base, b.combine(b.mul(v[0], offset), b.mul(v[1], offset), 0.0))

        # As the game chains them: layer 3 on the second set takes layer 2's clamped view, clamped again.
        limit2 = self.f("g_flParallaxViewAngleLimit2", 0.5)
        limit3 = self.f("g_flParallaxViewAngleLimit3", 0.5)
        limit_occlusion = self.f("g_flReflectionOcclusionViewAngleLimit", 0.5)
        if secondary["Layer2"]:
            view2 = clamped(view_secondary, -limit2, limit2)
            uv2 = shifted(uv_secondary, view2, "g_flLayer2Offset")
        else:
            uv2 = shifted(uv, clamped(view_primary, -limit2, limit2), "g_flLayer2Offset")
            view2 = view_secondary
        if secondary["Layer3"]:
            view3 = clamped(view2, -limit3, limit3)
            uv3 = shifted(uv_secondary, view3, "g_flLayer3Offset")
        else:
            uv3 = shifted(uv, clamped(view_primary, -limit3, limit3), "g_flLayer3Offset")
            view3 = view2
        if secondary["ReflectionOcclusion"]:
            uv_occ = shifted(uv_secondary, clamped(view3, -limit_occlusion, limit_occlusion), "g_flReflectionOcclusionOffset")
        else:
            uv_occ = shifted(uv, clamped(view_primary, -limit_occlusion, limit3), "g_flReflectionOcclusionOffset")

        b.column(4)
        vc, _ = self.vertex_color()
        colors = []
        texels = []
        for layer, (param, layer_uv) in enumerate((("g_tColor", uv), ("g_tColor2", uv2), ("g_tColor3", uv3)), start=1):
            node = self.texture(param, True, layer_uv, f"Color {layer}")
            texel = node.outputs["Color"] if node else (0, 0, 0)
            texel_alpha = node.outputs["Alpha"] if node else 0.0
            correct, _ = self.color_matrices(str(layer), param)
            c = self.apply_matrix(texel, correct, f"Correct {layer}")
            c = b.clerp(c, b.vmul(c, vc), self.f(f"g_fVertexColorStrength{layer}", 1.0))
            colors.append(c)
            texels.append(texel_alpha)

        b.column(8)
        m1 = self.texture("g_tMask", False, uv, "Mask")
        m2 = self.texture("g_tMask2", False, uv2, "Mask 2")
        mask1 = b.separate_color(m1.outputs["Color"])[0] if m1 else 1.0
        mask2 = b.separate_color(m2.outputs["Color"])[0] if m2 else 0.0
        through1 = b.sub(1.0, mask1)
        through2 = b.sub(1.0, mask2)

        def emissive_tint(tint_param, level_param):
            tint = self.v(tint_param, (1, 1, 1))[:3]
            scale = 2.0 ** self.f(level_param, 0.0)
            return tuple(vmat.srgb_to_linear(c) * scale for c in tint)

        tint2 = emissive_tint("g_vEmissiveTint2", "g_flEmissiveLevel2")
        tint3 = emissive_tint("g_vEmissiveTint3", "g_flEmissiveLevel3")
        lum = lambda t: t[0] * 0.2125 + t[1] * 0.7154 + t[2] * 0.0721
        emissive2 = b.mul(b.mul(texels[1], through1), self.f("g_flEmissiveLevel2", 0.0) * lum(tint2))
        emissive3 = b.mul(b.mul(texels[2], through2), self.f("g_flEmissiveLevel3", 0.0) * lum(tint3))

        b.column(10)
        albedo = b.clerp(b.clerp(colors[2], colors[1], mask2), colors[0], mask1)
        through_both = b.mul(through1, through2)
        albedo = b.clerp(albedo, b.vmul(albedo, tint2), b.mul(b.sub(1.0, through_both), b.saturate(emissive2)))
        self_illum = b.lerp(b.mul(through1, emissive2), emissive3, through_both)
        albedo = b.clerp(albedo, b.vmul(albedo, tint3), b.mul(through_both, b.saturate(emissive3)))

        b.column(12)
        nr1 = self.texture("g_tNormalRoughness", False, uv, "Normal Roughness")
        nr2 = self.texture("g_tNormalRoughness2", False, uv2, "Normal Roughness 2")
        occ = self.texture("g_tReflectionOcclusionMask", False, uv_occ, "Reflection Occlusion")
        occluded = b.math("LESS_THAN", b.separate_color(occ.outputs["Color"])[0], 0.5) if occ else 1.0
        blend_strength = self.f("g_bLayer2ShadingBlendStrength", 1.0)
        nrc1 = nr1.outputs["Color"] if nr1 else (0.5, 0.5, 1.0)
        nrc2 = nr2.outputs["Color"] if nr2 else (0.5, 0.5, 1.0)
        r1 = b.separate_color(nrc1)[2]
        r2 = b.separate_color(nrc2)[2]
        r1 = b.lerp(b.lerp(1.0, r1, occluded), r1, blend_strength)
        r2 = b.lerp(1.0, r2, occluded)
        t = b.mul(through1, blend_strength)
        nr = b.clerp(nrc1, nrc2, t)
        roughness = b.lerp(r1, r2, t)
        gradient = b.group(groups.normal_gradient(), Normal_Roughness=nr).outputs[0]
        normal = b.group(groups.gradient_normal(), Gradient=gradient).outputs[0]

        emission = b.vscale(albedo, b.mul(b.saturate(b.mul(self_illum, 0.1)), 10.0))
        shader = self.lit(albedo, 0.0, roughness, normal, emission, fog=self.fog_amount())
        self.finish(shader)

    # black.vfx ------------------------------------------------------------------------------------------------------

    def black(self):
        b = self.b
        node = b.group(self.ctx.unlit, Color=(0, 0, 0), Alpha=1.0)
        self.finish(node.outputs[0])
