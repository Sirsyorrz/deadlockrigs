"""A small builder for shader node trees: maths on sockets or constants, grouped, laid out in columns.

Every helper takes sockets or plain numbers/tuples and returns an output socket, so a shader port reads close to the
HLSL it comes from.
"""
import bpy

_X_STEP = 220
_Y_STEP = 190


def _is_socket(value):
    return isinstance(value, bpy.types.NodeSocket)


class Builder:
    def __init__(self, tree: bpy.types.NodeTree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links
        self._column = 0
        self._row = 0
        self._frame = None

    # Layout ---------------------------------------------------------------------------------------------------------

    def column(self, column: int):
        """Places the following nodes in another column (left to right)."""
        self._column = column
        self._row = 0

    def frame(self, label: str | None):
        """Puts the following nodes in a labelled frame (None ends it)."""
        if label is None:
            self._frame = None
            return
        frame = self.nodes.new("NodeFrame")
        frame.label = label
        frame.label_size = 14
        self._frame = frame

    def node(self, kind: str, **props):
        node = self.nodes.new(kind)
        for key, value in props.items():
            setattr(node, key, value)
        node.location = (self._column * _X_STEP, -self._row * _Y_STEP)
        self._row += 1
        if self._frame is not None:
            node.parent = self._frame
        node.hide = kind not in ("ShaderNodeTexImage", "ShaderNodeBsdfPrincipled", "ShaderNodeGroup",
                                 "ShaderNodeOutputMaterial", "ShaderNodeTexEnvironment")
        return node

    # Wiring ---------------------------------------------------------------------------------------------------------

    def set(self, socket: bpy.types.NodeSocket, value):
        """Links a socket, or sets its default value from a number or tuple."""
        if value is None:
            return
        if _is_socket(value):
            self.links.new(value, socket)
            return
        current = socket.default_value
        if isinstance(value, (int, float)):
            try:
                n = len(current)
            except TypeError:
                socket.default_value = value
                return
            socket.default_value = (value,) * n if n != 4 or socket.type != "RGBA" else (value, value, value, 1.0)
            return
        value = tuple(value)
        try:
            n = len(current)
        except TypeError:
            socket.default_value = value[0]
            return
        if len(value) < n:
            value = value + ((1.0,) if n == 4 and len(value) == 3 else (0.0,) * (n - len(value)))
        socket.default_value = value[:n]

    # Constants ------------------------------------------------------------------------------------------------------

    def value(self, v: float, label: str | None = None):
        node = self.node("ShaderNodeValue")
        node.outputs[0].default_value = v
        if label:
            node.label = label
        return node.outputs[0]

    def vector(self, v):
        node = self.node("ShaderNodeCombineXYZ")
        for i in range(3):
            node.inputs[i].default_value = v[i]
        return node.outputs[0]

    def rgb(self, v):
        node = self.node("ShaderNodeRGB")
        node.outputs[0].default_value = (v[0], v[1], v[2], 1.0)
        return node.outputs[0]

    # Scalar maths ---------------------------------------------------------------------------------------------------

    def math(self, op: str, a, b=None, c=None, clamp: bool = False):
        node = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(node.inputs[0], a)
        if b is not None:
            self.set(node.inputs[1], b)
        if c is not None:
            self.set(node.inputs[2], c)
        return node.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def div(self, a, b):
        return self.math("DIVIDE", a, b)

    def madd(self, a, b, c):
        return self.math("MULTIPLY_ADD", a, b, c)

    def saturate(self, a):
        return self.math("ADD", a, 0.0, clamp=True)

    def lerp(self, a, b, t):
        """a + (b - a) * t for floats."""
        node = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=False)
        self.set(node.inputs[0], t)
        self.set(node.inputs[2], a)
        self.set(node.inputs[3], b)
        return node.outputs[0]

    def smoothstep(self, edge0, edge1, x):
        node = self.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP", clamp=True)
        self.set(node.inputs["Value"], x)
        self.set(node.inputs["From Min"], edge0)
        self.set(node.inputs["From Max"], edge1)
        return node.outputs[0]

    def select(self, condition, if_true, if_false):
        """condition (0 or 1) ? if_true : if_false, for floats."""
        return self.lerp(if_false, if_true, condition)

    # Vector maths ---------------------------------------------------------------------------------------------------

    def vmath(self, op: str, a, b=None, c=None, scale=None):
        node = self.node("ShaderNodeVectorMath", operation=op)
        self.set(node.inputs[0], a)
        if b is not None:
            self.set(node.inputs[1], b)
        if c is not None:
            self.set(node.inputs[2], c)
        if scale is not None:
            self.set(node.inputs[3], scale)
        if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE"):
            return node.outputs[1]
        return node.outputs[0]

    def vadd(self, a, b):
        return self.vmath("ADD", a, b)

    def vsub(self, a, b):
        return self.vmath("SUBTRACT", a, b)

    def vmul(self, a, b):
        return self.vmath("MULTIPLY", a, b)

    def vscale(self, a, s):
        return self.vmath("SCALE", a, scale=s)

    def dot(self, a, b):
        return self.vmath("DOT_PRODUCT", a, b)

    def cross(self, a, b):
        return self.vmath("CROSS_PRODUCT", a, b)

    def normalize(self, a):
        return self.vmath("NORMALIZE", a)

    def vlerp(self, a, b, t):
        node = self.node("ShaderNodeMix", data_type="VECTOR", clamp_factor=False)
        node.factor_mode = "UNIFORM"
        self.set(node.inputs[0], t)
        self.set(node.inputs[4], a)
        self.set(node.inputs[5], b)
        return node.outputs[1]

    def clerp(self, a, b, t, blend: str = "MIX", clamp: bool = False):
        """Colour mix (RGBA): a to b by t, or a blend mode such as MULTIPLY."""
        node = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_factor=clamp, clamp_result=False)
        self.set(node.inputs[0], t)
        self.set(node.inputs[6], a)
        self.set(node.inputs[7], b)
        return node.outputs[2]

    def combine(self, x, y=0.0, z=0.0):
        node = self.node("ShaderNodeCombineXYZ")
        self.set(node.inputs[0], x)
        self.set(node.inputs[1], y)
        self.set(node.inputs[2], z)
        return node.outputs[0]

    def separate(self, v):
        node = self.node("ShaderNodeSeparateXYZ")
        self.set(node.inputs[0], v)
        return node.outputs[0], node.outputs[1], node.outputs[2]

    def separate_color(self, c):
        node = self.node("ShaderNodeSeparateColor")
        self.set(node.inputs[0], c)
        return node.outputs[0], node.outputs[1], node.outputs[2]

    def combine_color(self, r, g, b):
        node = self.node("ShaderNodeCombineColor")
        self.set(node.inputs[0], r)
        self.set(node.inputs[1], g)
        self.set(node.inputs[2], b)
        return node.outputs[0]

    # Inputs ---------------------------------------------------------------------------------------------------------

    def attribute(self, name: str, kind: str = "GEOMETRY"):
        """An attribute node; returns the node (outputs Color, Vector, Fac, Alpha)."""
        return self.node("ShaderNodeAttribute", attribute_type=kind, attribute_name=name)

    def image(self, image, vector=None, extension: str = "REPEAT", interpolation: str = "Linear", label: str | None = None):
        node = self.node("ShaderNodeTexImage", interpolation=interpolation, extension=extension)
        node.image = image
        if label:
            node.label = label
        if vector is not None:
            self.set(node.inputs[0], vector)
        return node

    def uv(self, name: str = "UVMap"):
        node = self.node("ShaderNodeUVMap")
        node.uv_map = name
        return node.outputs[0]

    def group(self, group_tree: bpy.types.NodeTree, label: str | None = None, **inputs):
        node = self.node("ShaderNodeGroup")
        node.node_tree = group_tree
        if label:
            node.label = label
        for name, value in inputs.items():
            socket = node.inputs.get(name.replace("_", " "))
            if socket is None:
                raise KeyError(f"group '{group_tree.name}' has no input '{name}'")
            self.set(socket, value)
        return node


# Node groups -----------------------------------------------------------------------------------------------------------

_SOCKET_TYPES = {
    "float": "NodeSocketFloat",
    "vector": "NodeSocketVector",
    "color": "NodeSocketColor",
    "shader": "NodeSocketShader",
}


def new_group(name: str, inputs: list[tuple], outputs: list[tuple]) -> tuple[bpy.types.NodeTree, Builder, dict, dict]:
    """A fresh shader node group with the given sockets: (name, type[, default]). Returns the tree, a builder and
    the group input and output sockets by name."""
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for socket in outputs:
        tree.interface.new_socket(socket[0], in_out="OUTPUT", socket_type=_SOCKET_TYPES[socket[1]])
    for socket in inputs:
        item = tree.interface.new_socket(socket[0], in_out="INPUT", socket_type=_SOCKET_TYPES[socket[1]])
        if len(socket) > 2:
            default = socket[2]
            if socket[1] == "color" and len(default) == 3:
                default = tuple(default) + (1.0,)
            item.default_value = default

    b = Builder(tree)
    b.column(-1)
    group_in = b.node("NodeGroupInput")
    group_in.hide = False
    b.column(30)
    group_out = b.node("NodeGroupOutput")
    group_out.hide = False
    b.column(0)
    ins = {s.name: s for s in group_in.outputs if s.name}
    outs = {s.name: s for s in group_out.inputs if s.name}
    return tree, b, ins, outs
