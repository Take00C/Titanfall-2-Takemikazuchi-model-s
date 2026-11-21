"""Material parameters straight from a Project MIKHAIL scene.

The mikhail-toolkit locates the `Material` records but does not decode them, and
paints every unit's lit panels with one hard-coded yellow.  The scene says
otherwise: each unit has a `glow_mat` (shader type COLOR_ONLY) that carries its
own emissive colour - red-orange on the blue-violet Type-00R, blue on the white
Type-00A, cyan on the Shiranui.  This module reads those values so nothing about
the lighting is guessed.

Record layout (tag stream, field names are CRC32 hashes):

    83 03 <crc "Material">  ...  87 <crc field> <value>  87 <crc field> <value> ...

    value tags:  8d + 4 bytes       RGBA8 colour
                 8f + 1 byte        8-bit float            91 + 4 bytes  f32
                 99 91 + 12 bytes   3 x f32                9a 91 + 16    4 x f32
                 9a 8f + 4 bytes    4 x 8-bit float        a6/86 ...\\0   string
                 d0 / d1            false / true           84            alignment padding

The glow colour sits in the field that directly follows `emissive = true`.  That
field's name is not in the game's string tables (crc 0x21349838), so it is found
by position and checked by type; if the layout ever differs this raises instead
of falling back to a made-up colour.
"""
import re
import struct
import zlib

_ELEM = {0x09: ("B", 1), 0x0a: ("H", 2), 0x0b: ("H", 2), 0x0c: ("I", 4),
         0x0f: ("B", 1), 0x10: ("e", 2), 0x11: ("f", 4)}
GLOW_COLOR_KEY = 0x21349838

BODY_FIELDS = ["AlbedoColor", "UserColor_Red", "UserColor_Green", "UserColor_Blue", "UserColor_Yellow",
               "emissiveColor", "emissiveColorIntensity", "emissiveScale",
               "Camouflage_Color0", "Camouflage_Color1", "Camouflage_Color2",
               "DirtIntensity", "ScratchIntensity", "Metallic", "Roughness"]


def crc(name: str) -> int:
    return zlib.crc32(name.encode()) & 0xFFFFFFFF


def _f8(b):
    """the engine's 8-bit float (sign, 4-bit exponent bias 7, 3-bit mantissa): 0x38 = 1.0"""
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x07
    v = (m / 8.0) * 2 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def _value(rec: bytes, p: int):
    while p < len(rec) and rec[p] == 0x84:
        p += 1
    if p >= len(rec):
        return None
    t = rec[p]
    if t < 0x80:
        return t
    if t in (0xd0, 0xd1):
        return t == 0xd1
    tid = t & 0x7F
    if tid == 0x0d:
        return {"rgba8": list(rec[p + 1:p + 5])}
    if tid in (0x06, 0x26):
        e = rec.find(b"\x00", p + 1)
        return rec[p + 1:e].decode("ascii", "replace")
    if tid == 0x03 and rec[p + 1] == 0x07:
        return {"enum_crc": "%08x" % struct.unpack_from("<I", rec, p + 2)[0]}
    if tid in _ELEM:
        fmt, _ = _ELEM[tid]
        v = struct.unpack_from("<" + fmt, rec, p + 1)[0]
        return round(_f8(v), 6) if tid == 0x0f else (round(v, 6) if fmt in "ef" else v)
    if 0x16 <= tid <= 0x26 and p + 1 < len(rec) and (rec[p + 1] & 0x7F) in _ELEM:
        n, eid = tid - 0x16, rec[p + 1] & 0x7F
        fmt, _ = _ELEM[eid]
        vals = struct.unpack_from("<%d%s" % (n, fmt), rec, p + 2)
        return [round(_f8(x) if eid == 0x0f else float(x), 6) for x in vals]
    return None


def _field(rec: bytes, name: str):
    k = rec.find(b"\x87" + struct.pack("<I", crc(name)))
    return None if k < 0 else _value(rec, k + 5)


_ENUMS = {"%08x" % crc(n): n for n in
          ("ShaderType_COLOR_ONLY", "ShaderType_CUSTOM_PBR", "ShaderType_PBR", "BlendType_NONE")}


def parse_materials(scn: bytes):
    """Every named Material record in the scene, in file order."""
    mark = b"\x83\x03" + struct.pack("<I", crc("Material"))
    starts = [m.start() for m in re.finditer(re.escape(mark), scn)]
    out = []
    for i, q in enumerate(starts):
        rec = scn[q:starts[i + 1] if i + 1 < len(starts) else min(len(scn), q + 1200)]
        name = _field(rec, "name")
        if not isinstance(name, str):
            continue
        st = _field(rec, "shaderType")
        mat = {"name": name, "shaderName": _field(rec, "shaderName"),
               "shaderType": _ENUMS.get(st["enum_crc"], st["enum_crc"]) if isinstance(st, dict) else None,
               "albedoColor_rgba8": (_field(rec, "albedoColor") or {}).get("rgba8"),
               "bloomScale": _field(rec, "bloomScale")}
        if mat["shaderType"] == "ShaderType_COLOR_ONLY":
            # the lit-panel material: "glow_mat" on a mech, "<unit>_gl_mat" on a weapon
            k = rec.find(b"\x87" + struct.pack("<I", crc("emissive")))
            if k < 0:
                raise ValueError(f"{name} has no 'emissive' field - material layout changed")
            mat["emissive"] = _value(rec, k + 5)
            key = struct.unpack_from("<I", rec, k + 7)[0] if rec[k + 6] == 0x87 else None
            vec = _value(rec, k + 11) if key is not None else None
            if key != GLOW_COLOR_KEY or not (isinstance(vec, list) and len(vec) == 3):
                raise ValueError(f"{name}: expected a 3-float colour in field {GLOW_COLOR_KEY:08x} after "
                                 f"'emissive', found field {key} = {vec!r}")
            mat["emissiveColor_linear"] = vec
        else:
            for f in BODY_FIELDS:
                v = _field(rec, f)
                if v is not None:
                    mat[f] = v
        out.append(mat)
    return out


def is_glow(material):
    return material.get("shaderType") == "ShaderType_COLOR_ONLY"


def glow_of(materials):
    """The unit's lit-panel material, or None if the scene has none."""
    return next((m for m in materials if is_glow(m)), None)


def resolve_drawblocks(meshes, materials):
    """Name the material of every drawblock, and mark the glow ones.

    A drawblock's `materialIndex` counts from a per-scene base; the named Material records sit
    in the file in exactly that order (checked on the Type-00: 789 chest_mat, 790 glow_mat,
    791 head_mat ... and on the Type-74 blade: 69 bl74_00_mat, 70 bl74_00_gl_mat).  This
    replaces the toolkit's guess ("a material shared by three or more parts is the glow"),
    which cannot work for a one-part model such as a weapon.
    """
    idxs = sorted({b["materialIndex"] for m in meshes for b in (m.get("drawblocks") or [])
                   if b.get("materialIndex") is not None})
    if not idxs:
        return
    base = idxs[0]
    if idxs[-1] - base >= len(materials):
        raise ValueError(f"drawblock material indices span {idxs[-1] - base + 1} but the scene has "
                         f"{len(materials)} named materials")
    for m in meshes:
        names = []
        for b in m.get("drawblocks") or []:
            k = b.get("materialIndex")
            names.append(materials[k - base]["name"] if k is not None else None)
        m["drawblockMaterial"] = names
        m["drawblockGlow"] = [bool(n) and is_glow(materials[[x["name"] for x in materials].index(n)]) for n in names]
