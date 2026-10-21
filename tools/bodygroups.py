"""Default visibility of a hero's render meshes, read from the decompiled (KV3 text) vmdl."""
import re


def mesh_parts(vmdl_text):
    """mesh name -> visible by default. A bodygroup shows its first choice, so meshes only reachable through a
    later choice (alternate heads, reload props) start hidden; meshes outside every bodygroup are always on."""
    meshes = [m for m in re.findall(r'_class = "RenderMeshFile"\s*name = "([^"]+)"', vmdl_text)
              if not re.search(r'_lod\d+$', m)]
    default_on, listed = set(), set()
    groups = vmdl_text[vmdl_text.find('"BodyGroupList"'):]
    for g in re.finditer(r'_class = "BodyGroup"\s*name = "([^"]+)"(.*?)(?=_class = "BodyGroup"\s*name|\Z)', groups, re.S):
        choices = [re.findall(r'"([^"]+)"', c) for c in re.findall(r'meshes =\s*\[(.*?)\]', g.group(2), re.S)]
        for i, choice in enumerate(choices):
            listed.update(choice)
            if i == 0:
                default_on.update(choice)
    return {m: m in default_on or m not in listed for m in meshes}
