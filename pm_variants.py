"""Check: do the eight Type-00 colourways share one mesh?

Parses every variant's scene (no textures), fingerprints the LOD0 geometry,
UVs, topology and skin, and reports which archive folder each variant takes
each texture role from.  If the fingerprints match, a second colourway is a
texture swap and not a second conversion.

    python -m tsfronin.pm_variants            -> build/pm/variants.json
"""
import hashlib
import json
import os

import numpy as np

from . import config
from .pm_extract import _toolkit


def _fp(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def fingerprint(er, built, unit_dir):
    idx = np.load(os.path.join(unit_dir, built["name"] + ".indices.npy"))
    parts = {}
    for m in built["meshes"]:
        if m.get("lod") not in (None, 0):
            continue
        ntri = sum((b.get("indexCount") or 0) // 3 for b in m["drawblocks"])
        i0 = m.get("indexStart", 0) if m.get("localIndices") else m["vertexStart"]
        parts[m["part"]] = {
            "verts": m["vertexCount"], "tris": ntri,
            "pos": _fp(np.round(np.asarray(m["positions"], np.float64), 5)),
            "uv": _fp(np.round(np.asarray(m["uv0"], np.float64), 5)),
            "topo": _fp(idx[i0:i0 + ntri * 3]),
            "skin": _fp(np.round(np.asarray(m["weights"], np.float64), 4),
                        np.asarray(m["joints"], np.int64),
                        "|".join(m["jointNames"]).encode()),
        }
    return parts


def main():
    config.check_inputs()
    er = _toolkit()
    er.OUT_ROOT = str(config.PM_DIR / "_variants")       # scratch output, geometry only
    os.makedirs(er.OUT_ROOT, exist_ok=True)
    pack = er.FpdArchive(er.PACK)
    ps = er.parse_scene_module()
    found = {u["unit"]: u for u in er.find_units(pack, "tsf")}
    report = {}
    for unit in config.TYPE00_VARIANTS:
        info = found[unit]
        built = er.build_unit(pack, info, ps, want_tex=False)
        unit_dir = os.path.join(er.OUT_ROOT, built["name"])
        srcs = {}
        for role, entry in info.get("textures") or []:
            folder = entry.split("/chr/tsf/")[1].split("/")[0] if "/chr/tsf/" in entry else "common"
            srcs.setdefault(folder, set()).add(role)
        report[unit] = {
            "nodes": len(built["nodes"]),
            "node_names": _fp("|".join(n["name"] for n in built["nodes"]).encode()),
            "parts": fingerprint(er, built, unit_dir),
            "texture_folders": {k: sorted(v) for k, v in sorted(srcs.items())},
        }

    ref_name = config.UNIT
    ref = report[ref_name]
    print(f"\nreference: {ref_name}   ({len(ref['parts'])} LOD0 parts, {ref['nodes']} nodes)")
    print(f"{'variant':16s} {'geometry':9s} {'uv':9s} {'topology':9s} {'skin':9s} {'skeleton':9s}  textures from")
    for unit, r in report.items():
        def same(key):
            diff = [p for p in ref["parts"] if r["parts"].get(p, {}).get(key) != ref["parts"][p][key]]
            return "same" if not diff and set(r["parts"]) == set(ref["parts"]) else "DIFF:" + ",".join(diff)[:40]
        skel = "same" if r["node_names"] == ref["node_names"] else "DIFF"
        r["matches_reference"] = {k: same(k) for k in ("pos", "uv", "topo", "skin")} | {"skeleton": skel}
        tex = "; ".join(f"{k}[{','.join(x[:2] for x in v)}]" for k, v in r["texture_folders"].items())
        print(f"{unit:16s} {same('pos'):9s} {same('uv'):9s} {same('topo'):9s} {same('skin'):9s} {skel:9s}  {tex}")
    out = config.PM_DIR / "variants.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
