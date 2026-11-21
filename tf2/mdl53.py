"""Reader for Titanfall 2 studio models (.mdl, IDST version 53).

One file holds everything: the studio header and tables, then the embedded VTX (strips),
VVD (vertices), optional VVC and PHY blobs.  `parse()` returns a JSON-friendly summary plus
the raw header dict and file bytes; the helpers at the bottom give the bind pose and a
numpy view of the vertex buffer.  Layout per public notes (rmdlconv, rsx, r2-model-compiler
docs) and checked against the retail Ronin models.
"""
import struct, sys, json, os

HDR_FIELDS = [
    ("id", "i"), ("version", "i"), ("checksum", "i"), ("sznameindex", "i"),
    ("name", "64s"), ("length", "i"),
    ("eyeposition", "3f"), ("illumposition", "3f"), ("hull_min", "3f"), ("hull_max", "3f"),
    ("view_bbmin", "3f"), ("view_bbmax", "3f"),
    ("flags", "i"),
    ("numbones", "i"), ("boneindex", "i"),
    ("numbonecontrollers", "i"), ("bonecontrollerindex", "i"),
    ("numhitboxsets", "i"), ("hitboxsetindex", "i"),
    ("numlocalanim", "i"), ("localanimindex", "i"),
    ("numlocalseq", "i"), ("localseqindex", "i"),
    ("activitylistversion", "i"), ("eventsindexed", "i"),
    ("numtextures", "i"), ("textureindex", "i"),
    ("numcdtextures", "i"), ("cdtextureindex", "i"),
    ("numskinref", "i"), ("numskinfamilies", "i"), ("skinindex", "i"),
    ("numbodyparts", "i"), ("bodypartindex", "i"),
    ("numlocalattachments", "i"), ("localattachmentindex", "i"),
    ("numlocalnodes", "i"), ("localnodeindex", "i"), ("localnodenameindex", "i"), ("localNodeUnk", "i"),
    ("flexdescindex", "i"),
    ("numflexcontrollers", "i"), ("flexcontrollerindex", "i"),
    ("numflexrules", "i"), ("flexruleindex", "i"),
    ("numikchains", "i"), ("ikchainindex", "i"),
    ("uiPanelCount", "i"), ("uiPanelOffset", "i"),
    ("numlocalposeparameters", "i"), ("localposeparamindex", "i"),
    ("surfacepropindex", "i"),
    ("keyvalueindex", "i"), ("keyvaluesize", "i"),
    ("numlocalikautoplaylocks", "i"), ("localikautoplaylockindex", "i"),
    ("mass", "f"), ("contents", "i"),
    ("numincludemodels", "i"), ("includemodelindex", "i"),
    ("virtualModel", "i"),
    ("bonetablebynameindex", "i"),
    ("constdirectionallightdot", "B"), ("rootLOD", "B"), ("numAllowedRootLODs", "B"), ("unused0", "B"),
    ("fadeDistance", "f"),
    ("numflexcontrollerui", "i"), ("flexcontrolleruiindex", "i"),
    ("flVertAnimFixedPointScale", "f"), ("surfacepropLookup", "i"),
    ("sourceFilenameOffset", "i"),
    ("numsrcbonetransform", "i"), ("srcbonetransformindex", "i"),
    ("illumpositionattachmentindex", "i"),
    ("linearboneindex", "i"),
    ("boneFlexDriverCount", "i"), ("boneFlexDriverIndex", "i"),
    ("perTriAABBIndex", "i"), ("perTriAABBNodeCount", "i"), ("perTriAABBLeafCount", "i"), ("perTriAABBVertCount", "i"),
    ("unkStringOffset", "i"),
    ("vtxOffset", "i"), ("vvdOffset", "i"), ("vvcOffset", "i"), ("phyOffset", "i"),
    ("vtxSize", "i"), ("vvdSize", "i"), ("vvcSize", "i"), ("phySize", "i"),
    ("collisionOffset", "i"), ("staticCollisionCount", "i"),
    ("boneFollowerCount", "i"), ("boneFollowerOffset", "i"),
]


def fmt(fields):
    return "<" + "".join(f for _, f in fields)


def unpack(fields, data, off):
    vals = struct.unpack_from(fmt(fields), data, off)
    out = {}
    i = 0
    for name, f in fields:
        digits = "".join(c for c in f if c.isdigit())
        kind = f[-1]
        n = int(digits) if (digits and kind != "s") else 1
        if n == 1:
            v = vals[i]
            if kind == "s":
                v = v.split(b"\0", 1)[0].decode("latin1")
            out[name] = v
            i += 1
        else:
            out[name] = list(vals[i:i + n])
            i += n
    return out


def size(fields):
    return struct.calcsize(fmt(fields))


def cstr(data, off):
    if off <= 0 or off >= len(data):
        return None
    end = data.find(b"\0", off)
    return data[off:end].decode("latin1")


BONE_FIELDS = [
    ("sznameindex", "i"), ("parent", "i"), ("bonecontroller", "6i"),
    ("pos", "3f"), ("quat", "4f"), ("rot", "3f"), ("scale", "3f"),
    ("posscale", "3f"), ("rotscale", "3f"), ("scalescale", "3f"),
    ("poseToBone", "12f"), ("qAlignment", "4f"),
    ("flags", "i"), ("proctype", "i"), ("procindex", "i"), ("physicsbone", "i"),
    ("surfacepropidx", "i"), ("contents", "i"), ("surfacepropLookup", "i"),
    ("collisionIndex", "H"), ("collisionCount", "H"), ("unused", "7i"),
]
ATTACH_FIELDS = [("sznameindex", "i"), ("flags", "I"), ("localbone", "i"), ("local", "12f"), ("unused", "8i")]
HBOXSET_FIELDS = [("sznameindex", "i"), ("numhitboxes", "i"), ("hitboxindex", "i")]
BBOX_FIELDS = [("bone", "i"), ("group", "i"), ("bbmin", "3f"), ("bbmax", "3f"), ("szhitboxnameindex", "i"),
               ("forceCritPoint", "i"), ("hitdataGroupOffset", "i"), ("unused", "6i")]
SEQDESC_FIELDS = [
    ("baseptr", "i"), ("szlabelindex", "i"), ("szactivitynameindex", "i"), ("flags", "i"),
    ("activity", "i"), ("actweight", "i"), ("numevents", "i"), ("eventindex", "i"),
    ("bbmin", "3f"), ("bbmax", "3f"), ("numblends", "i"), ("animindexindex", "i"),
    ("movementindex", "i"), ("groupsize", "2i"), ("paramindex", "2i"), ("paramstart", "2f"),
    ("paramend", "2f"), ("paramparent", "i"), ("fadeintime", "f"), ("fadeouttime", "f"),
    ("localentrynode", "i"), ("localexitnode", "i"), ("nodeflags", "i"), ("entryphase", "f"),
    ("exitphase", "f"), ("lastframe", "f"), ("nextseq", "i"), ("pose", "i"), ("numikrules", "i"),
    ("numautolayers", "i"), ("autolayerindex", "i"), ("weightlistindex", "i"), ("posekeyindex", "i"),
    ("numiklocks", "i"), ("iklockindex", "i"), ("keyvalueindex", "i"), ("keyvaluesize", "i"),
    ("cycleposeindex", "i"), ("activitymodifierindex", "i"), ("numactivitymodifiers", "i"),
    ("ikResetMask", "i"), ("unk_C4", "i"), ("unused", "8i"),
]
TEXTURE_FIELDS = [("sznameindex", "i"), ("unused_flags", "i"), ("used", "i"), ("unused", "8i")]
BODYPART_FIELDS = [("sznameindex", "i"), ("nummodels", "i"), ("base", "i"), ("modelindex", "i")]
MODEL_FIELDS = [
    ("name", "64s"), ("type", "i"), ("boundingradius", "f"), ("nummeshes", "i"), ("meshindex", "i"),
    ("numvertices", "i"), ("vertexindex", "i"), ("tangentsindex", "i"), ("numattachments", "i"),
    ("attachmentindex", "i"), ("numeyeballs", "i"), ("eyeballindex", "i"), ("pad", "4i"),
    ("colorindex", "i"), ("uv2index", "i"), ("unused", "4i"),
]
MESH_FIELDS = [
    ("material", "i"), ("modelindex", "i"), ("numvertices", "i"), ("vertexoffset", "i"),
    ("numflexes", "i"), ("flexindex", "i"), ("materialtype", "i"), ("materialparam", "i"),
    ("meshid", "i"), ("center", "3f"), ("modelvertexdata", "i"), ("numLODVertexes", "8i"),
    ("unk_54", "2i"), ("unused", "6i"),
]
IKCHAIN_FIELDS = [("sznameindex", "i"), ("linktype", "i"), ("numlinks", "i"), ("linkindex", "i"),
                  ("unk_10", "f"), ("unused", "3i")]
IKLINK_FIELDS = [("bone", "i"), ("kneeDir", "3f"), ("unused0", "3f")]
POSEPARAM_FIELDS = [("sznameindex", "i"), ("flags", "i"), ("start", "f"), ("end", "f"), ("loop", "f")]
INCLUDE_FIELDS = [("szlabelindex", "i"), ("sznameindex", "i")]
SRCBONE_FIELDS = [("sznameindex", "i"), ("pretransform", "12f"), ("posttransform", "12f")]
EVENT_FIELDS = [("cycle", "f"), ("event", "i"), ("type", "i"), ("options", "64s"), ("szeventindex", "i")]


def parse(path):
    d = open(path, "rb").read()
    h = unpack(HDR_FIELDS, d, 0)
    if h["id"] != 0x54534449:
        raise ValueError("bad id")
    info = {"path": path, "filesize": len(d), "version": h["version"], "name": h["name"], "length": h["length"],
            "flags": "0x%x" % h["flags"], "checksum": h["checksum"]}
    if h["version"] != 53:
        return info, h, d
    info["counts"] = {k: h[k] for k in ("numbones", "numbonecontrollers", "numhitboxsets", "numlocalanim", "numlocalseq",
                                         "numtextures", "numcdtextures", "numskinref", "numskinfamilies", "numbodyparts",
                                         "numlocalattachments", "numlocalnodes", "numikchains", "uiPanelCount",
                                         "numlocalposeparameters", "numincludemodels", "numsrcbonetransform",
                                         "numflexcontrollers", "numflexrules", "numlocalikautoplaylocks")}
    info["sizes"] = {k: h[k] for k in ("vtxOffset", "vvdOffset", "vvcOffset", "phyOffset", "vtxSize", "vvdSize", "vvcSize", "phySize")}
    info["mass"] = h["mass"]
    info["surfaceprop"] = cstr(d, h["surfacepropindex"])
    info["hull"] = [h["hull_min"], h["hull_max"]]
    info["eyeposition"] = h["eyeposition"]
    info["illumposition"] = h["illumposition"]
    # bones
    bones = []
    bsz = size(BONE_FIELDS)
    for i in range(h["numbones"]):
        o = h["boneindex"] + i * bsz
        b = unpack(BONE_FIELDS, d, o)
        bones.append({"i": i, "name": cstr(d, o + b["sznameindex"]), "parent": b["parent"], "flags": "0x%x" % b["flags"],
                      "proctype": b["proctype"], "pos": [round(x, 3) for x in b["pos"]],
                      "physicsbone": b["physicsbone"], "surfaceprop": cstr(d, o + b["surfacepropidx"])})
    info["bones"] = bones
    # attachments
    atts = []
    asz = size(ATTACH_FIELDS)
    for i in range(h["numlocalattachments"]):
        o = h["localattachmentindex"] + i * asz
        a = unpack(ATTACH_FIELDS, d, o)
        loc = a["local"]
        atts.append({"name": cstr(d, o + a["sznameindex"]), "bone": bones[a["localbone"]]["name"] if 0 <= a["localbone"] < len(bones) else a["localbone"],
                     "flags": "0x%x" % a["flags"], "pos": [round(loc[3], 2), round(loc[7], 2), round(loc[11], 2)],
                     # attachment space -> its bone's space (rows of the 3x4 matrix without the translation)
                     "rot": [[round(loc[r * 4 + c], 5) for c in range(3)] for r in range(3)]})
    info["attachments"] = atts
    # hitboxes
    hsets = []
    for i in range(h["numhitboxsets"]):
        o = h["hitboxsetindex"] + i * size(HBOXSET_FIELDS)
        s = unpack(HBOXSET_FIELDS, d, o)
        boxes = []
        for j in range(s["numhitboxes"]):
            bo = o + s["hitboxindex"] + j * size(BBOX_FIELDS)
            bb = unpack(BBOX_FIELDS, d, bo)
            boxes.append({"bone": bones[bb["bone"]]["name"] if 0 <= bb["bone"] < len(bones) else bb["bone"], "group": bb["group"],
                          "name": cstr(d, bo + bb["szhitboxnameindex"]) if bb["szhitboxnameindex"] else None,
                          "crit": bb["forceCritPoint"],
                          "hitdataGroup": cstr(d, bo + bb["hitdataGroupOffset"]) if bb["hitdataGroupOffset"] else None})
        hsets.append({"name": cstr(d, o + s["sznameindex"]), "hitboxes": boxes})
    info["hitboxsets"] = hsets
    # textures / skins
    tex = []
    for i in range(h["numtextures"]):
        o = h["textureindex"] + i * size(TEXTURE_FIELDS)
        t = unpack(TEXTURE_FIELDS, d, o)
        tex.append(cstr(d, o + t["sznameindex"]))
    info["textures"] = tex
    cdt = []
    for i in range(h["numcdtextures"]):
        (p,) = struct.unpack_from("<i", d, h["cdtextureindex"] + 4 * i)
        cdt.append(cstr(d, p))
    info["cdtextures"] = cdt
    skins = []
    for fi in range(h["numskinfamilies"]):
        skins.append(list(struct.unpack_from("<%dh" % h["numskinref"], d, h["skinindex"] + fi * h["numskinref"] * 2)))
    info["skinfamilies"] = skins
    # bodyparts
    bps = []
    for i in range(h["numbodyparts"]):
        o = h["bodypartindex"] + i * size(BODYPART_FIELDS)
        bp = unpack(BODYPART_FIELDS, d, o)
        models = []
        for j in range(bp["nummodels"]):
            mo = o + bp["modelindex"] + j * size(MODEL_FIELDS)
            m = unpack(MODEL_FIELDS, d, mo)
            mats = []
            for k in range(m["nummeshes"]):
                me = unpack(MESH_FIELDS, d, mo + m["meshindex"] + k * size(MESH_FIELDS))
                mats.append(me["material"])
            models.append({"name": m["name"], "nummeshes": m["nummeshes"], "numvertices": m["numvertices"], "mesh_materials": mats})
        bps.append({"name": cstr(d, o + bp["sznameindex"]), "base": bp["base"], "models": models})
    info["bodyparts"] = bps
    # ik chains
    iks = []
    for i in range(h["numikchains"]):
        o = h["ikchainindex"] + i * size(IKCHAIN_FIELDS)
        c = unpack(IKCHAIN_FIELDS, d, o)
        links = []
        for j in range(c["numlinks"]):
            l = unpack(IKLINK_FIELDS, d, o + c["linkindex"] + j * size(IKLINK_FIELDS))
            links.append(bones[l["bone"]]["name"] if 0 <= l["bone"] < len(bones) else l["bone"])
        iks.append({"name": cstr(d, o + c["sznameindex"]), "links": links})
    info["ikchains"] = iks
    # pose params
    pps = []
    for i in range(h["numlocalposeparameters"]):
        o = h["localposeparamindex"] + i * size(POSEPARAM_FIELDS)
        p = unpack(POSEPARAM_FIELDS, d, o)
        pps.append({"name": cstr(d, o + p["sznameindex"]), "start": p["start"], "end": p["end"], "loop": p["loop"]})
    info["poseparams"] = pps
    # include models
    incs = []
    for i in range(h["numincludemodels"]):
        o = h["includemodelindex"] + i * size(INCLUDE_FIELDS)
        im = unpack(INCLUDE_FIELDS, d, o)
        incs.append({"label": cstr(d, o + im["szlabelindex"]) if im["szlabelindex"] else None, "name": cstr(d, o + im["sznameindex"])})
    info["includemodels"] = incs
    # sequences
    seqs = []
    ssz = size(SEQDESC_FIELDS)
    for i in range(h["numlocalseq"]):
        o = h["localseqindex"] + i * ssz
        s = unpack(SEQDESC_FIELDS, d, o)
        evs = []
        for j in range(s["numevents"]):
            eo = o + s["eventindex"] + j * size(EVENT_FIELDS)
            e = unpack(EVENT_FIELDS, d, eo)
            evs.append({"name": cstr(d, eo + e["szeventindex"]), "options": e["options"], "cycle": round(e["cycle"], 3)})
        seqs.append({"label": cstr(d, o + s["szlabelindex"]), "activity": cstr(d, o + s["szactivitynameindex"]),
                     "numblends": s["numblends"], "flags": "0x%x" % s["flags"], "events": evs,
                     "numautolayers": s["numautolayers"], "numikrules": s["numikrules"]})
    info["sequences"] = seqs
    # nodes
    nodes = []
    for i in range(h["numlocalnodes"]):
        (p,) = struct.unpack_from("<i", d, h["localnodenameindex"] + 4 * i)
        nodes.append(cstr(d, p))
    info["nodes"] = nodes
    # keyvalues
    if h["keyvaluesize"] > 0:
        info["keyvalues"] = d[h["keyvalueindex"]:h["keyvalueindex"] + h["keyvaluesize"]].split(b"\0")[0].decode("latin1")
    # src bone transforms
    sbt = []
    for i in range(h["numsrcbonetransform"]):
        o = h["srcbonetransformindex"] + i * size(SRCBONE_FIELDS)
        s = unpack(SRCBONE_FIELDS, d, o)
        sbt.append(cstr(d, o + s["sznameindex"]))
    info["srcbonetransforms"] = sbt
    return info, h, d


# --------------------------------------------------------------------------- helpers
import numpy as np

VVD_VERTEX = np.dtype([("weight", "<f4", 3), ("bone", "u1", 3), ("numbones", "u1"),
                       ("pos", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2)])
assert VVD_VERTEX.itemsize == 48


def read_bones(d, h):
    """Full bone records, with the bind (rest) pose in model space taken from poseToBone."""
    bsz = size(BONE_FIELDS)
    out = []
    for i in range(h["numbones"]):
        o = h["boneindex"] + i * bsz
        b = unpack(BONE_FIELDS, d, o)
        m = np.asarray(b["poseToBone"], dtype=np.float64).reshape(3, 4)     # model space -> bone space
        R, t = m[:, :3], m[:, 3]
        world_R = R.T                                                       # bone space -> model space
        world_t = -R.T @ t
        out.append({"index": i, "name": cstr(d, o + b["sznameindex"]), "parent": b["parent"],
                    "local_pos": [float(x) for x in b["pos"]], "local_quat": [float(x) for x in b["quat"]],
                    "flags": b["flags"], "proctype": b["proctype"],
                    "bind_pos": [round(float(x), 4) for x in world_t],
                    "bind_rot": [[round(float(x), 6) for x in row] for row in world_R]})
    return out


def read_hitboxes(d, h):
    """Every hitbox of every set, in file order, with the two Respawn-specific fields
    (crit flag and the hit-data group string scripts use to map a hit to a bodygroup)."""
    out = []
    for i in range(h["numhitboxsets"]):
        o = h["hitboxsetindex"] + i * size(HBOXSET_FIELDS)
        s = unpack(HBOXSET_FIELDS, d, o)
        for j in range(s["numhitboxes"]):
            bo = o + s["hitboxindex"] + j * size(BBOX_FIELDS)
            bb = unpack(BBOX_FIELDS, d, bo)
            out.append({"set": cstr(d, o + s["sznameindex"]), "bone": bb["bone"], "group": bb["group"],
                        "min": [round(x, 4) for x in bb["bbmin"]], "max": [round(x, 4) for x in bb["bbmax"]],
                        "name": (cstr(d, bo + bb["szhitboxnameindex"]) or "") if bb["szhitboxnameindex"] else "",
                        "crit": bb["forceCritPoint"],
                        "hitdata": (cstr(d, bo + bb["hitdataGroupOffset"]) or "") if bb["hitdataGroupOffset"] else ""})
    return out


def vvd_header(d, h):
    o = h["vvdOffset"]
    ident, version, checksum, num_lods = struct.unpack_from("<4sIiI", d, o)
    lod_verts = struct.unpack_from("<8i", d, o + 16)
    num_fixups, fixup_start, vertex_start, tangent_start = struct.unpack_from("<4i", d, o + 48)
    if ident != b"IDSV":
        raise ValueError("embedded VVD not found at vvdOffset")
    return {"offset": o, "version": version, "checksum": checksum, "numLODs": num_lods,
            "numLODVertexes": list(lod_verts), "numFixups": num_fixups, "fixupTableStart": fixup_start,
            "vertexDataStart": vertex_start, "tangentDataStart": tangent_start}


def vvd_vertices(d, h):
    """(numpy record view of every vertex in file order, absolute file offset of the first one).
    Pass a bytearray to get a writable view."""
    v = vvd_header(d, h)
    start = v["offset"] + v["vertexDataStart"]
    n = (v["tangentDataStart"] - v["vertexDataStart"]) // VVD_VERTEX.itemsize
    return np.frombuffer(d, dtype=VVD_VERTEX, count=n, offset=start), start