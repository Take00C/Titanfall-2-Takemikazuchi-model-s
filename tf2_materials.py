"""Stage 55 - the Takemikazuchi's own materials, as a Titanfall 2 material pack.

    build/clean/<unit>/textures/*            Project MIKHAIL's maps (base colour, normal, metal, roughness, AO)
    data/material_skn.json                   render settings of Titanfall's standard skinned-model material
 -> build/tf2/materials/<unit>/tsf_takemikazuchi.rpak
    build/tf2/materials/<unit>/materials.json   which material each texture set got (read by tf2_build)

Titanfall 2 only reads materials from rpak files, and looks one up by the hash of
"material/<name in the .mdl>_skn.rpak".  The pack holds one material per Takemikazuchi texture set
plus one for its lit panels, and every texture they use, with all mip levels inside the pack.

Project MIKHAIL shades with base colour + metalness + roughness; Titanfall 2 with albedo + specular
colour + gloss.  The standard conversion:

    albedo   = base colour x (1 - metal)
    specular = 0.04 where not metal, the base colour where metal
    gloss    = 1 - roughness
    normal   = the same map with green negated (measured: the two games' maps slope opposite ways in
               green - see NORMAL_GREEN_FLIP)

The lit panels are separate triangles in Project MIKHAIL, drawn by a colour-only material; they get
a material of their own here whose only contribution is that colour in the glow slot.

    python -m tsfronin.tf2_materials [unit]
    python -m tsfronin.tf2_materials --capture-template     (re-derive data/material_skn.json)
"""
import hashlib
import json
import struct
import time

import numpy as np
from PIL import Image

from . import config
from .tf2 import bcn, rpak7

Image.MAX_IMAGE_PIXELS = None
TEMPLATE = config.REPO / "data" / "material_skn.json"
OUT_DIR = config.BUILD / "tf2" / "materials"
MATERIAL_DIR = "models/titans/tsf_takemikazuchi"
PAK_NAME = "tsf_takemikazuchi.rpak"
GUN_PAK_NAME = "tsf_takemikazuchi_gun.rpak"

# Measured on 2026-11-15 by testing which sign of green makes each map a consistent height-field slope:
# every Project MIKHAIL normal map came out +1, Northstar's own Titanfall 2 weapon normal maps -1.
NORMAL_GREEN_FLIP = True
DIELECTRIC_SPECULAR = 0.04
# How bright the lit panels are relative to Project MIKHAIL's emissive value.  1.0 = the value as stored.
GLOW_INTENSITY = config.GLOW_INTENSITY


# ------------------------------------------------------------------ the material template
def capture_template():
    """Read the render settings of a standard skinned-model material out of Northstar's own pack
    (R2Northstar/mods/Northstar.Custom/paks/mp_weapon_shotgun_doublebarrel.rpak) into data/material_skn.json.
    The build does not need that pack; this is how the data file was made and how to check it."""
    src = config.TITANFALL2 / "R2Northstar" / "mods" / "Northstar.Custom" / "paks" / "mp_weapon_shotgun_doublebarrel.rpak"
    pak = rpak7.Pak(src)
    want = rpak7.guid("material/models/weapons/twinbshotgun/twinbshotgun_skn.rpak")
    m = rpak7.read_material(pak, pak.by_guid[want])
    dx = m["dx_states"]
    blend = struct.unpack_from("<4I", dx, 0)
    mask, depth_stencil, raster, _pad = struct.unpack_from("<IHHQ", dx, 16)
    if dx[:0x20] != dx[0x20:]:
        raise SystemExit("the template's two render states differ; the data file assumes they are equal")
    formats = {}
    for slot, g in enumerate(m["textures"]):
        if g:
            formats[str(slot)] = rpak7.read_texture(pak, pak.by_guid[g])["format_name"]
    data = {
        "_about": [
            "Render settings of Titanfall 2's standard skinned-model material (shader set 0xC3ACAF7F1DC7F389:",
            "colour, normal, gloss, specular, glow, AO, cavity).  Taken from the material",
            "'models/weapons/twinbshotgun/twinbshotgun' in Northstar.Custom's mp_weapon_shotgun_doublebarrel.rpak",
            "by `python -m tsfronin.tf2_materials --capture-template`.  The field names are RePak's.",
            "shaderSet and the depth materials are hashes of assets in the base game's common.rpak.",
            "uber: the material's shader constants, as stored (56 little-endian floats; float 24-26 albedo tint,",
            "40-42 glow tint, 52 gloss scale, 53-55 specular floor; 49-51 are unused -1 integers)."
        ],
        "shaderType": "skn",
        "shaderSet": f"0x{m['shader_set']:016X}",
        "$depthShadowMaterial": f"0x{m['pass_materials'][0]:016X}",
        "$depthPrepassMaterial": f"0x{m['pass_materials'][1]:016X}",
        "$depthVSMMaterial": f"0x{m['pass_materials'][2]:016X}",
        "$colpassMaterial": f"0x{m['pass_materials'][3]:X}",
        "glueFlags": f"0x{m['flags']:08X}", "glueFlags2": f"0x{m['flags2']:08X}",
        "features": f"0x{m['features']:08X}", "samplers": f"0x{m['samplers']:08X}",
        "blendStates": [f"0x{x:08X}" for x in blend], "blendStateMask": f"0x{mask:X}",
        "depthStencilFlags": f"0x{depth_stencil:X}", "rasterizerFlags": f"0x{raster:X}",
        "textureSlotCount": len(m["textures"]),
        "slots": {"col": 0, "nml": 1, "gls": 2, "spc": 3, "ilm": 4, "ao": 11, "cav": 12},
        "slotFormatsInSource": formats,
        "uberVersion": m["uber_version"],
        "uber": m["uber"].hex(),
    }
    TEMPLATE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TEMPLATE} from {src.name}: shader set {data['shaderSet']}, {len(m['textures'])} texture slots, "
          f"formats {formats}, uber {len(m['uber'])} bytes")
    return data


def load_template():
    t = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    n = lambda s: int(s, 16)
    state = struct.pack("<4IIHHQ", *(n(x) for x in t["blendStates"]), n(t["blendStateMask"]),
                        n(t["depthStencilFlags"]), n(t["rasterizerFlags"]), 0)
    return {"shader_set": n(t["shaderSet"]),
            "pass_materials": [n(t["$depthShadowMaterial"]), n(t["$depthPrepassMaterial"]), n(t["$depthVSMMaterial"]),
                               n(t["$colpassMaterial"])],
            "dx_states": state + state, "flags": n(t["glueFlags"]), "flags2": n(t["glueFlags2"]),
            "features": n(t["features"]), "samplers": n(t["samplers"]), "texture_slots": t["textureSlotCount"],
            "surface": "metal_titan", "surface2": "", "uber": bytes.fromhex(t["uber"]),
            "uber_version": t["uberVersion"], "slots": t["slots"]}


def uber_with(uber, **floats):
    """the shader constants with some floats replaced: albedo_tint / glow_tint take (r, g, b)"""
    at = {"albedo_tint": 24, "glow_tint": 40}
    b = bytearray(uber)
    for name, rgb in floats.items():
        struct.pack_into("<3f", b, at[name] * 4, *rgb)
    return bytes(b)


# ------------------------------------------------------------------ converting one texture set
def _load(path, mode):
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def _resized(a, size):
    """float (H, W[, C]) -> (size[1], size[0][, C]) with smooth resampling"""
    if (a.shape[1], a.shape[0]) == tuple(size):
        return a
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a, mode="F").resize(size, Image.BILINEAR), dtype=np.float32)
    return np.stack([_resized(a[..., c], size) for c in range(a.shape[2])], axis=2)


def convert_set(clean, maps):
    """-> {kind: (float image, mip kind, encoder)} for one Project MIKHAIL texture set"""
    if "bc" not in maps or "nm" not in maps:
        raise SystemExit(f"texture set lacks a base colour or normal map: {sorted(maps)}")
    base = bcn.srgb_to_linear(_load(clean / maps["bc"]["file"], "RGB"))
    size = (base.shape[1], base.shape[0])
    metal = _load(clean / maps["mt"]["file"], "L") if "mt" in maps else None
    out = {}
    if metal is None:
        out["col"] = (base, "srgb", "bc1")
        out["spc"] = (np.full((4, 4, 3), DIELECTRIC_SPECULAR, np.float32), "srgb", "bc1")
    else:
        out["col"] = (base * (1.0 - _resized(metal, size))[..., None], "srgb", "bc1")
        small = (metal.shape[1], metal.shape[0])              # specular at the metal map's own size
        m3 = metal[..., None]
        out["spc"] = (DIELECTRIC_SPECULAR * (1.0 - m3) + _resized(base, small) * m3, "srgb", "bc1")
    n = _load(clean / maps["nm"]["file"], "RGB") * (255.0 / 127.5) - 1.0
    if NORMAL_GREEN_FLIP:
        n[..., 1] = -n[..., 1]
    out["nml"] = (n, "normal", "bc5")
    if "rh" in maps:
        out["gls"] = ((1.0 - _load(clean / maps["rh"]["file"], "L"))[..., None], "linear", "bc4")
    else:
        out["gls"] = (np.full((4, 4, 1), 0.5, np.float32), "linear", "bc4")
    if "ao" in maps:
        # the slot is an sRGB texture; storing the value sRGB-encoded makes the shader read back the AO itself
        out["ao"] = (np.repeat(_load(clean / maps["ao"]["file"], "L")[..., None], 3, axis=2), "srgb", "bc1")
    return out


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _check(level, raw, enc):
    """decode what was just encoded and compare (texture2ddecoder returns BGRA)"""
    import texture2ddecoder as t2d
    h, w = level.shape[:2]
    pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
    dec = {"bc1": t2d.decode_bc1, "bc4": t2d.decode_bc4, "bc5": t2d.decode_bc5}[enc](raw, pw, ph)
    img = np.frombuffer(dec, np.uint8).reshape(ph, pw, 4)[:h, :w, [2, 1, 0]]
    return _psnr(level, img[..., :level.shape[2]])


class Packer:
    def __init__(self):
        self.b = rpak7.Builder()
        self.cache = {}                                       # source fingerprint -> texture guid
        self.textures = []
        self.psnr = {}

    def texture(self, name, image, kind, enc, fingerprint=None):
        if fingerprint and fingerprint in self.cache:
            return self.cache[fingerprint]
        encode, fmt = bcn.ENCODERS[enc]
        levels = bcn.mip_chain(image, kind)
        mips = [encode(lv) for lv in levels]
        self.psnr.setdefault(name.rsplit("_", 1)[-1], []).append(_check(levels[0], mips[0], enc))
        path = f"texture/{MATERIAL_DIR}/{name}.rpak"
        g = rpak7.add_texture(self.b, path, levels[0].shape[1], levels[0].shape[0], fmt, mips)
        self.textures.append({"path": path, "guid": f"0x{g:016X}", "size": [levels[0].shape[1], levels[0].shape[0]],
                              "format": rpak7.FORMATS[fmt], "mips": len(mips), "bytes": sum(len(m) for m in mips)})
        if fingerprint:
            self.cache[fingerprint] = g
        return g

    def flat(self, name, value, kind, enc):
        """a 4x4 texture of one value"""
        c = {"bc1": 3, "bc4": 1, "bc5": 3}[enc]
        return self.texture(name, np.broadcast_to(np.asarray(value, np.float32), (4, 4, c)).copy(), kind, enc,
                            fingerprint=("flat", name))


def _fingerprint(clean, maps, kinds):
    h = hashlib.sha1()
    for k in kinds:
        if k in maps:
            h.update(k.encode() + hashlib.sha1((clean / maps[k]["file"]).read_bytes()).digest())
    return h.hexdigest()


SOURCES = {"col": ("bc", "mt"), "spc": ("bc", "mt"), "nml": ("nm",), "gls": ("rh",), "ao": ("ao",)}


def packs(unit=None):
    """[(pack file name, [units])].  The body and the sword share the main pack; the optional gun has a pack of
    its own, so that leaving the gun out leaves the main pack exactly as it is without it."""
    unit = unit or config.UNIT
    out = [(PAK_NAME, [unit] + ([config.SWORD_UNIT] if config.SWORD_UNIT else []))]
    if config.GUN_UNIT:
        out.append((GUN_PAK_NAME, [config.GUN_UNIT]))
    return out


def build(unit=None, extra_units=None):
    """The packs for `unit`, the sword and the gun (see packs()), written to build/tf2/materials/<unit>/.
    extra_units: put exactly these beside the body in one main pack instead (for experiments)."""
    unit = unit or config.UNIT
    plan = packs(unit) if extra_units is None else [(PAK_NAME, [unit, *extra_units])]
    dst = OUT_DIR / unit
    dst.mkdir(parents=True, exist_ok=True)
    result = {"pak": str(dst / PAK_NAME), "paks": {}, "units": {}, "textures": []}
    for i, (pak_name, units) in enumerate(plan):
        have = [u for u in units if (config.CLEAN_DIR / u / "manifest.json").exists()]
        for u in units:
            if u not in have:
                print(f"  note: {u} has not been extracted and cleaned; {pak_name} will not hold its materials")
        if not have:
            continue
        # the packs must not share asset names: the small flat textures of every pack but the main one carry its name
        r = _build_pack(dst / pak_name, have, "" if i == 0 else pak_name.rsplit(".", 1)[0] + "_")
        result["units"].update(r["units"])
        result["textures"] += r["textures"]
        result["paks"][pak_name] = {"path": str(dst / pak_name), "units": have}
    (dst / "materials.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _build_pack(pak_path, units, flat_prefix):
    t0 = time.time()
    tpl = load_template()
    slots = tpl["slots"]
    p = Packer()
    black = p.flat(f"{flat_prefix}shared_black", (0.0, 0.0, 0.0), "srgb", "bc1")
    white = p.flat(f"{flat_prefix}shared_white", (1.0, 1.0, 1.0), "srgb", "bc1")
    flat_nml = p.flat(f"{flat_prefix}shared_flat_nml", (0.0, 0.0, 1.0), "normal", "bc5")
    no_gls = p.flat(f"{flat_prefix}shared_zero_gls", (0.0,), "linear", "bc4")
    result = {"units": {}}
    for u in units:
        clean = config.CLEAN_DIR / u
        man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
        names = {}
        for set_name, maps in man["texture_sets"].items():
            conv = None
            guids = {}
            for kind in ("col", "nml", "gls", "spc", "ao"):
                fp = (kind, _fingerprint(clean, maps, SOURCES[kind]))
                if fp not in p.cache:
                    conv = conv or convert_set(clean, maps)
                    if kind not in conv:
                        guids[kind] = white
                        continue
                    image, mip_kind, enc = conv[kind]
                    guids[kind] = p.texture(f"{u}_{set_name}_{kind}", image, mip_kind, enc, fingerprint=fp)
                else:
                    guids[kind] = p.cache[fp]
            w, h = Image.open(clean / maps["bc"]["file"]).size
            name = f"{MATERIAL_DIR}/{u}_{set_name}"
            rpak7.add_material(p.b, f"material/{name}_skn.rpak", name, tpl,
                               {slots["col"]: guids["col"], slots["nml"]: guids["nml"], slots["gls"]: guids["gls"],
                                slots["spc"]: guids["spc"], slots["ilm"]: black, slots["ao"]: guids["ao"],
                                slots["cav"]: white}, w, h)
            names[set_name] = name
            print(f"  {u} / {set_name}: {name}  ({time.time() - t0:.0f}s)")
        glow = None
        if man.get("glow"):
            e = [float(x) for x in man["glow"]["emissive_linear"]]
            if config.GLOW_COLOR:
                e = [float(x) for x in config.GLOW_COLOR.split(",")]
            ilm = p.texture(f"{u}_glow_ilm", np.broadcast_to(np.asarray(e, np.float32), (4, 4, 3)).copy(), "srgb", "bc1")
            glow = f"{MATERIAL_DIR}/{u}_glow"
            rpak7.add_material(p.b, f"material/{glow}_skn.rpak", glow, tpl,
                               {slots["col"]: black, slots["nml"]: flat_nml, slots["gls"]: no_gls, slots["spc"]: black,
                                slots["ilm"]: ilm, slots["ao"]: white, slots["cav"]: white}, 4, 4,
                               uber=uber_with(tpl["uber"], glow_tint=(GLOW_INTENSITY,) * 3))
            print(f"  {u} / lit panels: {glow}  glow colour (linear) {e} x {GLOW_INTENSITY}")
        result["units"][u] = {"materials": names, "glow": glow}

    size = p.b.write(pak_path)
    result["textures"] = p.textures
    print(f"  compression check (decoded top level against the source, PSNR in dB, higher is closer): "
          + ", ".join(f"{k} {min(v):.1f}-{max(v):.1f}" for k, v in sorted(p.psnr.items())))
    print(f"  wrote {pak_path}: {size / 1e6:.1f} MB, {len(p.b.assets)} assets "
          f"({len(p.textures)} textures, {len(p.b.assets) - len(p.textures)} materials), {len(p.b.pages)} pages, "
          f"{len(p.b.slabs)} slabs  ({time.time() - t0:.0f}s)")
    verify(pak_path, result)
    return result


def verify(pak_path, result):
    """Read the pack back and check what the game will rely on."""
    pak = rpak7.Pak(pak_path)
    tpl = load_template()
    problems = []
    if pak.end_of_pages != len(pak.data):
        problems.append(f"pages end at {pak.end_of_pages}, file is {len(pak.data)} bytes")
    if pak.pointers != sorted(pak.pointers):
        problems.append("page pointers are not in order")
    external = {tpl["shader_set"], *[g for g in tpl["pass_materials"] if g]}
    for a in pak.assets:
        refs = pak.asset_uses(a)
        if refs != sorted(refs):
            problems.append(f"asset {a['index']}: guid references not in order")
        internal = set()
        for r in refs:
            g = struct.unpack("<Q", pak.at(r, 8))[0]
            if g in pak.by_guid:
                internal.add(g)
            elif g not in external:
                problems.append(f"asset {a['index']}: reference to unknown asset {g:016X}")
        if a["internal_dependencies"] != 1 + len(internal):
            problems.append(f"asset {a['index']}: dependency count {a['internal_dependencies']} != {1 + len(internal)}")
        if a["type"] == "txtr":
            t = rpak7.read_texture(pak, a)
            want = sum(-(-s // 16) * 16 for s in rpak7.mip_sizes(t["format"], t["width"], t["height"], t["mips"]))
            if t["data_size"] != want or a["guid"] != rpak7.guid(f"texture/{t['name']}.rpak"):
                problems.append(f"texture {t['name']}: size {t['data_size']} (expected {want}) or guid mismatch")
        elif a["type"] == "matl":
            m = rpak7.read_material(pak, a)
            if a["guid"] != rpak7.guid(f"material/{m['name']}_skn.rpak"):
                problems.append(f"material {m['name']}: guid does not match its name")
            missing = [i for i in tpl["slots"].values() if not m["textures"][i]]
            if missing:
                problems.append(f"material {m['name']}: empty texture slots {missing}")
    wanted = [n for u in result["units"].values() for n in [*u["materials"].values(), u["glow"]] if n]
    for n in wanted:
        if rpak7.guid(f"material/{n}_skn.rpak") not in pak.by_guid:
            problems.append(f"material {n} is not in the pack")
    if problems:
        raise SystemExit("material pack failed its read-back check:\n  " + "\n  ".join(problems))
    print(f"  read back: {len(pak.assets)} assets, {len(pak.pointers)} pointers, {len(pak.uses)} guid references, "
          f"slabs {[(hex(s['flags']), s['alignment'], s['size']) for s in pak.slabs]}; all {len(wanted)} materials "
          f"resolve by name")


if __name__ == "__main__":
    import sys
    if "--capture-template" in sys.argv:
        capture_template()
    else:
        build(next((a for a in sys.argv[1:] if not a.startswith("-")), None))
