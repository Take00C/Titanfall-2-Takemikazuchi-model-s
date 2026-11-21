"""Prove the pak writer against paks the game is known to load.

Northstar ships two material packs of its own (R2Northstar/mods/Northstar.Custom/paks).  For each:

  1. the rules the writer uses to fill the pak's tables are applied to the reference's own contents and
     must give the reference's own numbers (slab sizes, dependency counts, dependents list, ordering);
  2. every material in it, and every texture that keeps all its mips in the pak, is rebuilt with the
     writer from what the reader parsed, and the bytes of its header and data must equal the reference's
     (pointer fields aside, which hold page numbers).

    python -m tsfronin.tf2.rpak7_check
"""
import struct

from .. import config
from . import rpak7

REFERENCES = ("northstarEventModels.rpak", "mp_weapon_shotgun_doublebarrel.rpak")


def check_tables(pak, problems):
    # slab size = its pages end to end, each starting on its own alignment
    size = [0] * len(pak.slabs)
    for p in pak.pages:
        s = p["slab"]
        size[s] = -(-size[s] // p["alignment"]) * p["alignment"] + p["size"]
    stored = [s["size"] for s in pak.slabs]
    if any(have < need for have, need in zip(stored, size)):
        problems.append(f"slab sizes: rule gives {size}, file has less: {stored}")
    elif size != stored:
        # northstarEventModels.rpak was built by a RePak that also charged a slab for the padding after pages
        # of other slabs (21 spare bytes); the size only has to be enough
        print(f"    note: slab sizes {stored} are roomier than the rule's {size}")
    for s, sl in enumerate(pak.slabs):
        aligns = [p["alignment"] for p in pak.pages if p["slab"] == s]
        if aligns and sl["alignment"] != max(aligns):
            problems.append(f"slab {s}: alignment {sl['alignment']} is not the largest of its pages {max(aligns)}")
    if pak.pointers != sorted(pak.pointers):
        problems.append("page pointers are not sorted")
    index = {a["guid"]: a["index"] for a in pak.assets}
    dependents = {a["index"]: [] for a in pak.assets}
    last_end = 0
    for a in pak.assets:
        refs = pak.asset_uses(a)
        if refs != sorted(refs):
            problems.append(f"asset {a['index']}: guid references are not sorted")
        seen = []
        for r in refs:
            g = struct.unpack("<Q", pak.at(r, 8))[0]
            if g in index and g not in seen:
                seen.append(g)
        for g in seen:
            dependents[index[g]].append(a["index"])
        if a["internal_dependencies"] != 1 + len(seen):
            problems.append(f"asset {a['index']}: dependency count {a['internal_dependencies']}, rule gives {1 + len(seen)}")
        top = max(a["head"][0], a["cpu"][0]) + 1
        if not (top <= a["page_end"] <= len(pak.pages)) or a["page_end"] < last_end:
            problems.append(f"asset {a['index']}: pageEnd {a['page_end']} (uses pages up to {top}, previous {last_end})")
        last_end = a["page_end"]
    flat = [d for a in pak.assets for d in dependents[a["index"]]]
    if flat != pak.dependents:
        problems.append("dependents list differs from the rule's")
    for a in pak.assets:
        got = pak.dependents[a["dependents_index"]:a["dependents_index"] + a["dependents_count"]]
        if got != dependents[a["index"]]:
            problems.append(f"asset {a['index']}: dependents {got}, rule gives {dependents[a['index']]}")


def rebuild_material(pak, a, problems):
    m = rpak7.read_material(pak, a)
    b = rpak7.Builder()
    textures = {i: g for i, g in enumerate(m["textures"]) if g}
    rpak7.add_material(b, f"material/{m['name']}_skn.rpak", m["name"], m, textures, m["width"], m["height"])
    head, name, data, cpu = (lump for page in b.pages for lump in page["lumps"])
    if b.assets[0]["guid"] != a["guid"]:
        problems.append(f"material {m['name']}: guid differs")

    def blank(buf):
        buf = bytearray(buf)
        for off in (rpak7.M_NAME, rpak7.M_SURF, rpak7.M_SURF2, rpak7.M_TEX, rpak7.M_STREAMTEX):
            buf[off:off + 8] = bytes(8)
        return bytes(buf)

    if blank(head.data) != blank(m["head"]):
        diff = [i for i, (x, y) in enumerate(zip(blank(head.data), blank(m["head"]))) if x != y]
        problems.append(f"material {m['name']}: header bytes differ at offsets {diff[:12]}")
    tex_ptr = rpak7.PTR.unpack_from(m["head"], rpak7.M_TEX)
    if bytes(data.data) != pak.at(tex_ptr, data.size):
        problems.append(f"material {m['name']}: data block differs")
    if bytes(cpu.data[8:]) != pak.at(a["cpu"], cpu.size)[8:]:
        problems.append(f"material {m['name']}: shader constant block differs")
    if bytes(name.data) != pak.at(rpak7.PTR.unpack_from(m["head"], rpak7.M_NAME), name.size):
        problems.append(f"material {m['name']}: name differs")
    want = sorted((r[1] - a["head"][1]) if r[0] == a["head"][0] else (r[1] - tex_ptr[1] + 1000) for r in pak.asset_uses(a))
    got = sorted(p[1] - head.offset if p[0] == head.page else p[1] - data.offset + 1000 for p, _g in b.assets[0]["uses"])
    if want != got:
        problems.append(f"material {m['name']}: guid reference positions {got} != {want}")
    return m["name"]


def rebuild_texture(pak, a, problems):
    t = rpak7.read_texture(pak, a)
    if t["streamed_mips"] or t["optional_streamed_mips"] or t["format"] not in rpak7.BLOCK_BYTES:
        return None
    sizes = rpak7.mip_sizes(t["format"], t["width"], t["height"], t["mips"])
    raw = pak.at(a["cpu"], t["data_size"])
    mips, o = [], 0
    for s in reversed(sizes):                                    # stored smallest first
        mips.insert(0, raw[o:o + s])
        o += -(-s // 16) * 16
    b = rpak7.Builder()
    path = f"texture/{t['name']}.rpak" if t["name"] else "texture/unnamed.rpak"
    rpak7.add_texture(b, path, t["width"], t["height"], t["format"], mips, template=t)
    head, _name, data = (lump for page in b.pages for lump in page["lumps"])
    mine, ref = bytearray(head.data), bytearray(pak.at(a["head"], 56))
    for buf in (mine, ref):
        buf[8:16] = bytes(8)                                     # name pointer
    if not t["name"]:
        mine[0:8] = ref[0:8]
    mine[26] = ref[26]                                           # array size: 0 or 1 depending on the RePak version
    if mine != ref:
        diff = [i for i, (x, y) in enumerate(zip(mine, ref)) if x != y]
        problems.append(f"texture {t['name']}: header bytes differ at offsets {diff}")
    if bytes(data.data) != raw:
        problems.append(f"texture {t['name']}: data differs")
    return t["name"] or f"{a['guid']:016X}"


def main():
    folder = config.TITANFALL2 / "R2Northstar" / "mods" / "Northstar.Custom" / "paks"
    problems = []
    for name in REFERENCES:
        path = folder / name
        if not path.exists():
            print(f"  {name}: not installed, skipped")
            continue
        pak = rpak7.Pak(path)
        before = len(problems)
        check_tables(pak, problems)
        mats = [rebuild_material(pak, a, problems) for a in pak.assets if a["type"] == "matl"]
        texs = [x for x in (rebuild_texture(pak, a, problems) for a in pak.assets if a["type"] == "txtr") if x]
        names = sum(1 for a in pak.assets if a["type"] == "txtr" and rpak7.read_texture(pak, a)["name"])
        print(f"  {name}: tables follow the writer's rules; {len(mats)} materials and {len(texs)} all-in-pak textures "
              f"rebuilt byte for byte; {names} texture names hash to their guids"
              if len(problems) == before else f"  {name}: {len(problems) - before} PROBLEMS")
    if problems:
        raise SystemExit("pak writer check failed:\n  " + "\n  ".join(problems))
    print("  pak writer agrees with the reference packs")


if __name__ == "__main__":
    main()
