"""Titanfall 2 pak files (rpak version 7): read one, and write one holding textures and materials.

Layout of an uncompressed v7 pak, in file order (reference: r-ex/RePak, src/logic/pakfile.cpp):

    header                 0x58 bytes
    streaming file paths   header.starpak_paths_size bytes (none here: every mip is stored in the pak)
    slab headers           16 bytes each   flags, alignment, total size
    page headers           12 bytes each   slab index, alignment, size
    page pointers           8 bytes each   (page, offset) of every field that holds a (page, offset) pointer
    asset descriptors      72 bytes each
    guid references         8 bytes each   (page, offset) of every field that holds another asset's guid
    dependents              4 bytes each   asset indices
    page data              the pages, back to back

An asset is a header in one page plus data in others; fields that point at data hold (page index,
offset), and fields that refer to other assets hold that asset's guid, which is a hash of its path.
"""
import struct

MAGIC = 0x6B615052                       # 'RPak'
HEADER = struct.Struct("<IHHQ8sQ8sQ8sHHHHIIIIII")
assert HEADER.size == 0x58
SLAB = struct.Struct("<iiQ")
PAGE = struct.Struct("<iii")
PTR = struct.Struct("<ii")
ASSET = struct.Struct("<Q8siiiiqHhIIIIII4s")
assert ASSET.size == 72

SF_HEAD, SF_CPU, SF_TEMP, SF_DEV = 0, 1, 2, 0x80
MAX_PAGE_MERGE_SIZE = 0xFFFF
M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF


def guid(path):
    """The engine's hash of an asset path (case-insensitive, '\\' counts as '/')."""
    data = path.encode("ascii") + b"\0" * 8
    v1, i = 0, 0
    while True:
        w = int.from_bytes(data[i:i + 4], "little")
        v4 = ~w & (w - 0x01010101) & 0x80808080 & M32          # marks the first zero byte
        v5 = (v4 ^ (v4 - 1)) & M32                             # everything up to and including it
        v6 = ((v5 & w) ^ 0x5C5C5C5C) & M32
        v7 = ~v6 & (v6 - 0x01010101) & 0x80808080 & M32        # marks backslashes
        v8 = v7 & (-v7 & M32)
        if v7 != v8:
            v9 = 0xFF000000
            while True:
                v10 = v9
                if (v9 & v6) == 0:
                    v8 |= v9 & 0x80808080
                v9 >>= 8
                if v10 < 0x100:
                    break
        v11 = (0x633D5F1 * v1) & M64
        v12 = ((0xFB8C4D96501 * (((v5 & w) - 45 * (v8 >> 7)) & 0xDFDFDFDF)) & M64) >> 24
        if v4:
            break
        i += 4
        t = (v11 + v12) & M64
        v1 = (t >> 61) ^ t
    length = i + (v5.bit_length() - 1) // 8
    return (v12 + v11 - 0xAE502812AA7333 * (length & M32)) & M64


# ---------------------------------------------------------------- reading
class Pak:
    def __init__(self, path):
        self.path = str(path)
        d = self.data = open(path, "rb").read()
        (magic, self.version, self.flags, self.file_time, _u0, self.compressed_size, _u1, self.decompressed_size, _u2,
         self.starpak_paths_size, n_slabs, n_pages, self.patch_index, n_ptrs, n_assets, n_uses, n_deps,
         self.unk7, self.unk8) = HEADER.unpack_from(d, 0)
        if magic != MAGIC or self.version != 7:
            raise ValueError(f"{path}: not a version 7 rpak (magic {magic:#x}, version {self.version})")
        if self.flags & 0x8300 or self.compressed_size != self.decompressed_size:
            raise ValueError(f"{path}: compressed pak (flags {self.flags:#x}); only uncompressed paks are read here")
        o = HEADER.size
        self.starpak_paths = [s.decode("ascii", "replace") for s in d[o:o + self.starpak_paths_size].split(b"\0") if s]
        o += self.starpak_paths_size
        self.slabs = [dict(zip(("flags", "alignment", "size"), SLAB.unpack_from(d, o + i * SLAB.size))) for i in range(n_slabs)]
        o += n_slabs * SLAB.size
        self.pages = [dict(zip(("slab", "alignment", "size"), PAGE.unpack_from(d, o + i * PAGE.size))) for i in range(n_pages)]
        o += n_pages * PAGE.size
        self.pointers = [PTR.unpack_from(d, o + i * PTR.size) for i in range(n_ptrs)]
        o += n_ptrs * PTR.size
        self.assets = []
        for i in range(n_assets):
            (g, _unk, hp, ho, cp, co, star, page_end, int_deps, dep_i, use_i, dep_n, use_n, head_size, ver,
             kind) = ASSET.unpack_from(d, o + i * ASSET.size)
            self.assets.append({"index": i, "guid": g, "head": (hp, ho), "cpu": (cp, co), "starpak": star,
                                "page_end": page_end, "internal_dependencies": int_deps, "dependents_index": dep_i,
                                "uses_index": use_i, "dependents_count": dep_n, "uses_count": use_n,
                                "head_size": head_size, "version": ver, "type": kind.decode("ascii").rstrip("\0")})
        o += n_assets * ASSET.size
        self.uses = [PTR.unpack_from(d, o + i * PTR.size) for i in range(n_uses)]
        o += n_uses * PTR.size
        self.dependents = list(struct.unpack_from(f"<{n_deps}I", d, o))
        o += n_deps * 4
        self.page_offset = []
        for p in self.pages:
            self.page_offset.append(o)
            o += p["size"]
        self.end_of_pages = o
        self.by_guid = {a["guid"]: a for a in self.assets}

    def at(self, ptr, size=None):
        """bytes at a (page, offset) pointer; to the end of the page when size is None"""
        page, off = ptr
        start = self.page_offset[page] + off
        end = self.page_offset[page] + self.pages[page]["size"] if size is None else start + size
        return self.data[start:end]

    def cstring(self, ptr):
        raw = self.at(ptr)
        return raw[:raw.index(b"\0")].decode("ascii", "replace")

    def page_flags(self, page):
        return self.slabs[self.pages[page]["slab"]]["flags"]

    def pointer_set(self):
        if not hasattr(self, "_pointer_set"):
            self._pointer_set = set(self.pointers)
        return self._pointer_set

    def asset_uses(self, a):
        return self.uses[a["uses_index"]:a["uses_index"] + a["uses_count"]]


TXTR_HEAD = struct.Struct("<QiiHHHHIBBBBBBB12sxq")
assert TXTR_HEAD.size == 56
FORMATS = {0: "BC1", 1: "BC1 sRGB", 2: "BC2", 3: "BC2 sRGB", 4: "BC3", 5: "BC3 sRGB", 6: "BC4", 7: "BC4 signed",
           8: "BC5", 9: "BC5 signed", 10: "BC6H", 11: "BC6H signed", 12: "BC7", 13: "BC7 sRGB", 31: "RGBA8",
           32: "RGBA8 sRGB", 53: "R8"}
BLOCK_BYTES = {0: 8, 1: 8, 2: 16, 3: 16, 4: 16, 5: 16, 6: 8, 7: 8, 8: 16, 9: 16, 10: 16, 11: 16, 12: 16, 13: 16}


def read_texture(pak, a):
    (g, name_page, name_off, w, h, depth, fmt, data_size, compression, opt_mips, array_size, resource_flags, usage_flags,
     mips, streamed_mips, per_mip, num_pixels) = TXTR_HEAD.unpack(pak.at(a["head"], TXTR_HEAD.size))
    has_name = (a["head"][0], a["head"][1] + 8) in pak.pointer_set()
    name = pak.cstring((name_page, name_off)) if has_name else None
    return {"guid": g, "name": name, "width": w, "height": h, "depth": depth, "format": fmt,
            "format_name": FORMATS.get(fmt, f"#{fmt}"), "data_size": data_size, "compression": compression,
            "optional_streamed_mips": opt_mips, "array_size": array_size, "resource_flags": resource_flags,
            "usage_flags": usage_flags, "mips": mips, "streamed_mips": streamed_mips, "per_mip": per_mip.hex(),
            "num_pixels": num_pixels}


MATL_HEAD_SIZE = 208
# offsets inside the 208-byte material header (MaterialAssetHeader_v12_t)
M_GUID, M_NAME, M_SURF, M_SURF2, M_PASS, M_DX, M_SHADERSET = 0x10, 0x18, 0x20, 0x28, 0x30, 0x50, 0x90
M_TEX, M_STREAMTEX, M_NUMSTREAM, M_SAMPLERS, M_UNK_AE, M_UNK_B0 = 0x98, 0xA0, 0xA8, 0xAA, 0xAE, 0xB0
M_FEATURES, M_UNK_BC, M_FLAGS, M_FLAGS2, M_WIDTH, M_HEIGHT, M_DEPTH = 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC


def read_material(pak, a):
    head = pak.at(a["head"], MATL_HEAD_SIZE)
    ptrs = pak.pointer_set()

    def ptr_at(off):
        p = PTR.unpack_from(head, off)
        return p if (a["head"][0], a["head"][1] + off) in ptrs else None

    q = lambda off: struct.unpack_from("<Q", head, off)[0]
    tex_ptr, stream_ptr = ptr_at(M_TEX), ptr_at(M_STREAMTEX)
    n_tex = 0
    if tex_ptr and stream_ptr and tex_ptr[0] == stream_ptr[0]:
        n_tex = (stream_ptr[1] - tex_ptr[1]) // 8
    textures = list(struct.unpack_from(f"<{n_tex}Q", pak.at(tex_ptr, n_tex * 8))) if n_tex else []
    cpu_ptr, cpu_size, cpu_version = struct.unpack_from("<8sII", pak.at(a["cpu"], 16))
    cpu_data_ptr = PTR.unpack(cpu_ptr)
    return {
        "guid": q(M_GUID),
        "name": pak.cstring(ptr_at(M_NAME)) if ptr_at(M_NAME) else None,
        "surface": pak.cstring(ptr_at(M_SURF)) if ptr_at(M_SURF) else "",
        "surface2": pak.cstring(ptr_at(M_SURF2)) if ptr_at(M_SURF2) else "",
        "pass_materials": [q(M_PASS + 8 * i) for i in range(4)],      # shadow, prepass, vsm, colpass
        "dx_states": head[M_DX:M_DX + 0x40],
        "shader_set": q(M_SHADERSET),
        "textures": textures,
        "num_streaming": struct.unpack_from("<h", head, M_NUMSTREAM)[0],
        "samplers": struct.unpack_from("<I", head, M_SAMPLERS)[0],
        "unk_ae": struct.unpack_from("<h", head, M_UNK_AE)[0], "unk_b0": q(M_UNK_B0),
        "features": struct.unpack_from("<I", head, M_FEATURES)[0],
        "unk_bc": struct.unpack_from("<I", head, M_UNK_BC)[0],
        "flags": struct.unpack_from("<I", head, M_FLAGS)[0], "flags2": struct.unpack_from("<I", head, M_FLAGS2)[0],
        "width": struct.unpack_from("<h", head, M_WIDTH)[0], "height": struct.unpack_from("<h", head, M_HEIGHT)[0],
        "depth": struct.unpack_from("<h", head, M_DEPTH)[0],
        "head": head,
        "uber": pak.at(cpu_data_ptr, cpu_size), "uber_version": cpu_version,
    }


def mip_sizes(fmt, width, height, count):
    """byte size of each mip, largest first (block-compressed formats only)"""
    bpb = BLOCK_BYTES[fmt]
    return [max(1, (max(1, width >> i) + 3) // 4) * max(1, (max(1, height >> i) + 3) // 4) * bpb for i in range(count)]


# ---------------------------------------------------------------- writing
class Lump:
    def __init__(self, page, offset, size):
        self.page, self.offset, self.size = page, offset, size
        self.data = bytearray(size)

    def ptr(self, offset=0):
        return (self.page, self.offset + offset)


class Builder:
    """Lays lumps out in pages and slabs the way RePak does, so the result has the shape the game is fed
    by every other Northstar skin."""

    def __init__(self):
        self.slabs = []          # {"flags", "alignment"}
        self.pages = []          # {"flags", "slab", "alignment", "size", "lumps": [(offset, Lump)]}
        self.pointers = []
        self.assets = []

    def _slab(self, flags, align):
        best, diff = None, None
        for s in self.slabs:
            if s["flags"] != flags:
                continue
            if s["alignment"] == align:
                best = s
                break
            if diff is None or abs(s["alignment"] - align) < diff:
                best, diff = s, abs(s["alignment"] - align)
        if best is None:
            if len(self.slabs) >= 20:
                raise ValueError("more than 20 slabs")
            best = {"flags": flags, "alignment": align, "index": len(self.slabs)}
            self.slabs.append(best)
        best["alignment"] = max(best["alignment"], align)
        return best

    def lump(self, size, flags, align):
        aligned = -(-size // align) * align
        slab = self._slab(flags, align)
        best, diff = None, None
        for p in self.pages:
            if p["flags"] != flags:
                continue
            a = max(p["alignment"], align)
            if -(-p["size"] // a) * a + aligned > MAX_PAGE_MERGE_SIZE:
                continue
            if p["alignment"] == align:
                best = p
                break
            if diff is None or abs(p["alignment"] - align) < diff:
                best, diff = p, abs(p["alignment"] - align)
        if best is None:
            best = {"flags": flags, "slab": slab["index"], "alignment": align, "size": 0, "lumps": [],
                    "index": len(self.pages)}
            self.pages.append(best)
        best["alignment"] = max(best["alignment"], align)
        best["size"] = -(-best["size"] // align) * align          # pad what is there up to this lump's alignment
        lump = Lump(best["index"], best["size"], size)
        best["lumps"].append(lump)
        best["size"] += aligned
        return lump

    def pointer(self, lump, offset, target=None, target_offset=0):
        """register a (page, offset) pointer field, and fill it in when a target is given"""
        self.pointers.append(lump.ptr(offset))
        if target is not None:
            PTR.pack_into(lump.data, offset, *target.ptr(target_offset))

    def begin(self, asset_guid, name):
        if any(a["guid"] == asset_guid for a in self.assets):
            raise ValueError(f"asset {name} ({asset_guid:#x}) added twice")
        a = {"guid": asset_guid, "name": name, "uses": [], "dependents": []}
        self.assets.append(a)
        return a

    def use(self, asset, lump, offset, target_guid):
        """a field that holds another asset's guid"""
        struct.pack_into("<Q", lump.data, offset, target_guid)
        if target_guid:
            asset["uses"].append((lump.ptr(offset), target_guid))

    def finish(self, asset, head, cpu, version, kind):
        asset.update(head=head.ptr(), head_size=head.size, cpu=cpu.ptr() if cpu is not None else (-1, 0),
                     version=version, type=kind, page_end=len(self.pages))

    def write(self, path, file_time=None):
        index = {a["guid"]: i for i, a in enumerate(self.assets)}
        for i, a in enumerate(self.assets):                       # who depends on whom, inside this pak
            a["internal"] = 1
            for g in dict.fromkeys(g for _p, g in a["uses"]):
                if g in index:
                    self.assets[index[g]]["dependents"].append(i)
                    a["internal"] += 1
        uses, deps = [], []
        for a in self.assets:
            a["uses"].sort()
            a["uses_index"], a["deps_index"] = (len(uses) if a["uses"] else 0), (len(deps) if a["dependents"] else 0)
            uses += [p for p, _g in a["uses"]]
            deps += a["dependents"]
        # a slab is as big as its pages laid end to end, each starting on its own alignment
        size = [0] * len(self.slabs)
        for p in self.pages:
            s = p["slab"]
            size[s] = -(-size[s] // p["alignment"]) * p["alignment"] + p["size"]
        out = bytearray(HEADER.size)
        for s, n in zip(self.slabs, size):
            out += SLAB.pack(s["flags"], s["alignment"], n)
        for p in self.pages:
            out += PAGE.pack(p["slab"], p["alignment"], p["size"])
        for p in sorted(self.pointers):
            out += PTR.pack(*p)
        for a in self.assets:
            out += ASSET.pack(a["guid"], b"\0" * 8, *a["head"], *a["cpu"], -1, a["page_end"], a["internal"],
                              a["deps_index"], a["uses_index"], len(a["dependents"]), len(a["uses"]), a["head_size"],
                              a["version"], a["type"].encode("ascii").ljust(4, b"\0"))
        for p in uses:
            out += PTR.pack(*p)
        out += struct.pack(f"<{len(deps)}I", *deps)
        for p in self.pages:
            body = bytearray(p["size"])
            for lump in p["lumps"]:
                body[lump.offset:lump.offset + lump.size] = lump.data
            out += body
        if file_time is None:
            # the header's build time (Windows FILETIME, 100 ns ticks since 1601).  A fixed value - midnight
            # UTC on 2024-01-01 - so that the same inputs always give the same file.
            file_time = (1704067200 + 11644473600) * 10_000_000
        HEADER.pack_into(out, 0, MAGIC, 7, 0, file_time, b"\0" * 8, len(out), b"\0" * 8, len(out), b"\0" * 8,
                         0, len(self.slabs), len(self.pages), 0, len(self.pointers), len(self.assets), len(uses),
                         len(deps), 0, 0)
        with open(path, "wb") as f:
            f.write(out)
        return len(out)


def add_texture(b, path, width, height, fmt, mips, template=None):
    """path: 'texture/<name>.rpak'.  mips: the block-compressed mip levels, largest first, all kept in the pak."""
    if len(mips) > 13:
        raise ValueError(f"{path}: {len(mips)} mip levels, the engine takes at most 13")
    g = guid(path)
    a = b.begin(g, path)
    head = b.lump(TXTR_HEAD.size, SF_HEAD, 8)
    stem = path[len("texture/"):-len(".rpak")] if path.startswith("texture/") and path.endswith(".rpak") else path
    name_lump = b.lump(len(stem) + 1, SF_CPU | SF_DEV, 1)         # debug name, as RePak's dev builds keep it
    name_lump.data[:len(stem)] = stem.encode("ascii")
    b.pointer(head, 8, name_lump)
    name_ptr = name_lump.ptr()
    sizes = [-(-len(m) // 16) * 16 for m in mips]
    data = b.lump(sum(sizes), SF_CPU | SF_TEMP, 16)
    o = 0
    for m, n in zip(reversed(mips), reversed(sizes)):             # smallest mip first
        data.data[o:o + len(m)] = m
        o += n
    t = template or {}
    TXTR_HEAD.pack_into(head.data, 0, g, *name_ptr, width, height, t.get("depth", 0), fmt, sum(sizes),
                        t.get("compression", 0), 0, 1, t.get("resource_flags", 0), t.get("usage_flags", 0),
                        len(mips), 0, bytes(12), 0)
    b.finish(a, head, data, 8, "txtr")
    return g


def add_material(b, path, name, template, textures, width, height, uber=None, surface=None):
    """path: 'material/<name>_skn.rpak'.  template: a read_material() result whose render settings, shader set,
    depth materials and shader constants are reused.  textures: {slot: guid}."""
    g = guid(path)
    a = b.begin(g, path)
    head = b.lump(MATL_HEAD_SIZE, SF_HEAD, 16)
    head.data[M_DX:M_DX + 0x40] = template["dx_states"]
    struct.pack_into("<I", head.data, M_SAMPLERS, template["samplers"])
    struct.pack_into("<I", head.data, M_FEATURES, template["features"])
    struct.pack_into("<II", head.data, M_FLAGS, template["flags"], template["flags2"])
    struct.pack_into("<Q", head.data, M_GUID, g)
    name_lump = b.lump(len(name) + 1, SF_CPU | SF_DEV, 1)
    name_lump.data[:len(name)] = name.encode("ascii")
    b.pointer(head, M_NAME, name_lump)

    n = max(template.get("texture_slots") or len(template["textures"]), max(textures) + 1)
    surface = template["surface"] if surface is None else surface
    surf = (surface.encode("ascii") + b"\0") if surface else b""
    surf2 = (template["surface2"].encode("ascii") + b"\0") if template["surface2"] else b""
    data = b.lump(n * 16 + len(surf) + len(surf2), SF_CPU, 8)
    for slot, tg in sorted(textures.items()):
        b.use(a, data, slot * 8, tg)
    data.data[n * 16:n * 16 + len(surf)] = surf
    data.data[n * 16 + len(surf):] = surf2
    b.pointer(head, M_TEX, data, 0)
    b.pointer(head, M_STREAMTEX, data, n * 8)
    if surf:
        b.pointer(head, M_SURF, data, n * 16)
    if surf2:
        b.pointer(head, M_SURF2, data, n * 16 + len(surf))
    for i, pg in enumerate(template["pass_materials"]):
        b.use(a, head, M_PASS + 8 * i, pg)
    b.use(a, head, M_SHADERSET, template["shader_set"])
    struct.pack_into("<h", head.data, M_NUMSTREAM, 0)
    struct.pack_into("<hh", head.data, M_WIDTH, width, height)

    uber = template["uber"] if uber is None else uber
    cpu = b.lump(16 + len(uber), SF_CPU | SF_TEMP, 8)
    struct.pack_into("<II", cpu.data, 8, len(uber), template.get("uber_version", 3))
    cpu.data[16:] = uber
    b.pointer(cpu, 0, cpu, 16)
    b.finish(a, head, cpu, 12, "matl")
    return g
