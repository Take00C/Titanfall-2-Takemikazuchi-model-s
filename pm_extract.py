"""Stage 10 - Project MIKHAIL pack.bin  ->  build/pm/<unit>/

    <unit>.json            rig + every LOD's meshes + skin weights
    <unit>.indices.npy     triangle list
    textures/*.png         decoded maps (base colour, normal, metal, ao, masks)

Drives the mikhail-toolkit as a library so the output lands in this project's
build/ folder instead of inside the toolkit checkout.  The game install is only
ever read.

    python -m tsfronin.pm_extract [unit ...] [--no-tex]
"""
import argparse
import json
import os
import re
import sys

from . import config, pm_materials


def _toolkit():
    """Import the toolkit's exporter with its paths pointed at our config."""
    sys.dont_write_bytecode = True                            # leave no __pycache__ in the toolkit checkout
    os.environ["MIKHAIL_GAME"] = str(config.MIKHAIL_GAME)     # read at import time
    os.environ["MIKHAIL_KEY"] = str(config.MIKHAIL_KEY)
    sys.path.insert(0, str(config.MIKHAIL_TOOLKIT / "mikhail"))
    import export_rigged as er
    er.GAME = str(config.MIKHAIL_GAME)
    er.PACK = str(config.MIKHAIL_GAME / "obb" / "pack.bin")
    er.OUT_ROOT = str(config.PM_DIR)
    return er


_NM_MIDDLE = re.compile(rb"([\x20-\x7e]{1,200})_nm_middle\.png")


def add_normal_maps(er, pack, info, built, unit_dir):
    """Decode the normal maps the toolkit skips.

    A Type-00 scene asks for "<part>_nm_middle.png" (the game's mid-quality tier)
    on six of its eight texture sets.  The toolkit's classifier only knows the
    plain "_nm." suffix, so those parts came out with no normal map at all.
    Resolve each reference against the archive and take the full-resolution
    "<part>_nm.png" that sits beside it when there is one.
    """
    import txb_fast
    from PIL import Image
    names = {e.name for e in pack.entries}
    model_dir = os.path.dirname(info["model"])
    have = {m["part"] for m in built["materials"] if m["role"] == "normal"}
    pref = er.uv_preferred_shape(built["meshes"])
    scn = pack.payload(info["model"])
    for ref in sorted(set(_NM_MIDDLE.findall(scn))):
        rel = ref.decode("ascii")
        part = os.path.basename(rel)
        if part in have:
            continue
        base = os.path.normpath(os.path.join(model_dir, rel)).replace(os.sep, "/")
        entry = next((c for c in (base + "_nm.png.txb.desktop.zstd",
                                  base + "_nm_middle.png.txb.desktop.zstd") if c in names), None)
        if entry is None:
            print(f"    ! normal map for {part}: {rel}_nm_middle.png is not in the archive")
            continue
        w, h, fmt, arr = txb_fast.decode(pack.payload(entry), prefer=pref)
        png = f"textures/{part}_nm.png"
        Image.fromarray(arr[:, :, :3], "RGB").save(os.path.join(unit_dir, png))
        built["materials"].append({"part": part, "role": "normal", "png": png})
        info.setdefault("textures", []).append(("normal", entry))
        have.add(part)
        print(f"    tex {part}/normal: {w}x{h} {fmt}  (from {entry.split('/chr/tsf/')[-1]})")


def extract(units, want_tex=True, kind="tsf"):
    """kind: "tsf" for a mech (chr/tsf/<unit>), "weapon" for equipment (equip/weapon/<unit>)."""
    config.check_inputs()
    er = _toolkit()
    config.PM_DIR.mkdir(parents=True, exist_ok=True)
    pack = er.FpdArchive(er.PACK)
    ps = er.parse_scene_module()
    found = {u["unit"]: u for u in er.find_units(pack, kind)}
    missing = [u for u in units if u not in found]
    if missing:
        raise SystemExit(f"not in pack.bin: {missing}")
    tex_cache = {}
    results = {}
    for unit in units:
        info = found[unit]
        built = er.build_unit(pack, info, ps, want_tex=want_tex, tex_cache=tex_cache)
        unit_dir = config.PM_DIR / built["name"]
        unit_dir.mkdir(parents=True, exist_ok=True)
        if want_tex:
            add_normal_maps(er, pack, info, built, str(unit_dir))
        # material parameters (glow colour, user-colour palette, ...) - the toolkit does not decode these
        mats = pm_materials.parse_materials(pack.payload(info["model"]))
        toolkit_glow = [m.get("drawblockGlow") for m in built["meshes"]]
        pm_materials.resolve_drawblocks(built["meshes"], mats)
        changed = sum(1 for a, m in zip(toolkit_glow, built["meshes"]) if a != m.get("drawblockGlow"))
        if changed:
            print(f"    glow drawblocks: {changed} mesh(es) corrected from the material records")
        (unit_dir / "materials.json").write_text(json.dumps({"unit": unit, "kind": kind, "materials": mats}, indent=1),
                                                 encoding="utf-8")
        built["kind"] = kind
        glow = pm_materials.glow_of(mats)
        print(f"    materials: {len(mats)} records; glow emissive (linear) = "
              f"{glow['emissiveColor_linear'] if glow else 'none'}")
        # which archive entries the unit actually used, including maps borrowed from siblings
        built["textureSources"] = [{"role": r, "entry": t} for r, t in info.get("textures") or []]
        dst = unit_dir / f"{built['name']}.json"
        with open(dst, "w") as f:
            json.dump(built, f)
        results[unit] = dst
        print(f"  => {unit}: {len(built['meshes'])} meshes, {built['vertices']:,} verts, "
              f"{len(built['nodes'])} nodes, {len(built['materials'])} maps -> {unit_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("units", nargs="*", default=None)
    ap.add_argument("--kind", choices=["tsf", "weapon"], default="tsf")
    ap.add_argument("--no-tex", action="store_true")
    a = ap.parse_args()
    units = a.units or ([config.UNIT] if a.kind == "tsf" else [config.SWORD_UNIT])
    extract(units, want_tex=not a.no_tex, kind=a.kind)


if __name__ == "__main__":
    main()
