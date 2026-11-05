import bpy

from . import session


def _value_node(tree, name=None, label=None):
    if tree is None:
        return None
    for node in tree.nodes:
        if node.type == "VALUE" and ((name and node.name == name) or (label and node.label == label)):
            return node
    return None


def _frame_values(tree, frame_label):
    """Value nodes inside a labelled frame of a node tree."""
    if tree is None:
        return []
    frame = next((n for n in tree.nodes if n.type == "FRAME" and n.label == frame_label), None)
    if frame is None:
        return []
    return [n for n in tree.nodes if n.type == "VALUE" and n.parent == frame and n.label]


class DEADLOCK_OT_setup_render(bpy.types.Operator):
    bl_idname = "deadlock.setup_render"
    bl_label = "Set Up Deadlock Render"
    bl_description = ("Standard view transform, no diffuse bounces (the maps' baked lighting is their bounce light), "
                      "glossy bounces and ray traced reflections")
    bl_options = {"REGISTER", "UNDO"}

    engine: bpy.props.EnumProperty(name="Engine", items=[("BLENDER_EEVEE", "EEVEE", ""), ("CYCLES", "Cycles", "")])

    def execute(self, context):
        from .blender import post
        from .preferences import get_prefs
        prefs = get_prefs()
        context.scene.render.engine = self.engine
        post.setup_render(context.scene, prefs.unit_scale if prefs else 0.0254)
        return {"FINISHED"}


class DEADLOCK_PT_sidebar(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Deadlock"
    bl_label = "Deadlock"

    def draw(self, context):
        layout = self.layout
        install = session.install_path()
        if install is None:
            layout.label(text="Deadlock not found", icon="ERROR")
            layout.operator("preferences.addon_show", text="Set the game folder").module = __package__
            return
        layout.operator("deadlock.import_map", icon="IMPORT")
        row = layout.row(align=True)
        row.operator("deadlock.setup_render", text="EEVEE").engine = "BLENDER_EEVEE"
        row.operator("deadlock.setup_render", text="Cycles").engine = "CYCLES"
        if context.scene.view_settings.view_transform != "Standard":
            layout.label(text="View transform isn't Standard", icon="INFO")

        for collection in context.scene.collection.children:
            name = collection.get("dl_map")
            if not name:
                continue
            box = layout.box()
            box.label(text=collection.name, icon="WORLD")

            col = box.column(align=True)
            for child in collection.children:
                row = col.row(align=True)
                row.prop(child, "hide_viewport", text="", emboss=False)
                row.prop(child, "hide_render", text="", emboss=False)
                row.label(text=f"{child.name}  ({len(child.all_objects)})")

            lighting = bpy.data.node_groups.get(f"DL Lighting {name}")
            node = _value_node(lighting, name="dl_baked_light")
            if node is not None:
                box.prop(node.outputs[0], "default_value", text="Baked Light")
            node = _value_node(lighting, name="dl_detail_distance") if collection.get("dl_detail_levels") else None
            if node is not None:
                box.prop(node.outputs[0], "default_value", text="Detail Distance")

            fog = bpy.data.node_groups.get(f"DL Fog {name}")
            for label in ("Gradient fog", "Cubemap fog"):
                values = _frame_values(fog, label)
                opacity = next((v for v in values if v.label == "Max Opacity"), None)
                if opacity is not None:
                    box.prop(opacity.outputs[0], "default_value", text=f"{label} opacity")

            volume = bpy.data.materials.get(f"{name} volumetric fog")
            if volume is not None and volume.node_tree is not None:
                for node_name, text in (("dl_fog_density", "Volumetric fog density"), ("dl_fog_scattering", "Fog light scattering")):
                    node = volume.node_tree.nodes.get(node_name)
                    if node is not None:
                        box.prop(node.outputs[0], "default_value", text=text)


classes = (DEADLOCK_OT_setup_render, DEADLOCK_PT_sidebar)
