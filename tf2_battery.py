"""Stage 66 - the battery canister, made invisible.

Titanfall 2 hangs a separate little model, the battery canister, on a point at the top of a light titan's
rear hull.  The Takemikazuchi has no hull there, so the canister stands in the open behind its head.  No
part of the Takemikazuchi is thick enough to hide it in (every armour block was tested from all round), so
the canister's visible mesh is taken out instead: the file is rebuilt from its own recipe with a single
speck of a triangle (a hundredth of a unit) where the canister was.  Its bones, its named points and all 15
of its animations stay stock, so everything the game does with the canister still works - it just is not seen.

Side effect, stated plainly: the game uses this one file for every light titan, so Northstar's canister is
not drawn either (on this player's screen only).  HIDE_BATTERY=0 leaves the file out of the mod.

    build/tf2/stock/models/titans/light/titan_light_rodeo_battery.mdl
 -> build/tf2/tsf_battery/titan_light_rodeo_battery.mdl

    python -m tsfronin.tf2_battery
"""
import json
import re

import numpy as np

from . import config, tf2_sword
from .tf2 import mdl53
from .tf2_build import compile_qc, smd_header
from .tf2_extract import STOCK_DIR

BATTERY = "models/titans/light/titan_light_rodeo_battery.mdl"
OUT = config.BUILD / "tf2" / "tsf_battery"
SPECK = 0.01                                                 # game units


def build():
    src = STOCK_DIR / BATTERY.replace("/", "\\")
    if not src.exists():
        raise SystemExit(f"{src} missing - run the tf2-extract stage first")
    name = src.stem
    qc_src = tf2_sword.decompile_file(src, tf2_sword.DECOMPILED / name)
    info, h, d = mdl53.parse(str(src))
    bones = mdl53.read_bones(d, h)
    verts, _ = mdl53.vvd_vertices(d, h)
    used = np.unique(verts["bone"][verts["weight"] > 0])
    if len(used) != 1 or len(info["textures"]) != 1:
        raise SystemExit(f"{name}: expected one mesh bone and one material, found bones {used.tolist()}, "
                         f"materials {info['textures']}")
    bone, material = int(used[0]), info["textures"][0]
    c = verts["pos"].astype(np.float64).mean(0)
    corner = lambda dx, dz: (f"{bone} {c[0] + dx:.6f} {c[1]:.6f} {c[2] + dz:.6f} 0.000000 1.000000 0.000000 "
                             f"0.500000 0.500000 1 {bone} 1.000000")
    lines = list(smd_header(bones)) + ["triangles", material, corner(0, 0), corner(0, SPECK), corner(SPECK, 0), "end"]
    smd = tf2_sword.DECOMPILED / name / "tsf" / "hidden.smd"
    smd.parent.mkdir(parents=True, exist_ok=True)
    smd.write_text("\n".join(lines) + "\n", encoding="ascii")
    text, n = re.subn(r'(?m)^\tstudio "[^"]+"$', '\tstudio "tsf/hidden.smd"', qc_src.read_text(encoding="utf-8"))
    if n != 1 or re.search(r"(?m)^\$lod ", text):
        raise SystemExit(f"{name}.qc: expected one mesh and no lower levels of detail, found {n} meshes")
    qc = qc_src.with_name(f"{name}_tsf.qc")
    qc.write_text(text, encoding="utf-8")
    out = compile_qc(qc, OUT / f"{name}.mdl")

    a, _ha, _da = mdl53.parse(str(src))
    b, hb, db = mdl53.parse(str(out))
    differ = [k for k in ("bones", "attachments", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams",
                          "sequences", "nodes", "srcbonetransforms", "mass", "surfaceprop", "textures")
              if a.get(k) != b.get(k)]
    left, _ = mdl53.vvd_vertices(db, hb)
    ok, rep = tf2_sword.validate_swapped(out, src)
    (OUT / f"{name}.validate.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    anim = rep["animations"]
    print(f"  {name}: {len(verts)} vertices -> {len(left)} (a speck {SPECK} units across); "
          f"identical to stock: bones, named points, sequences and the rest" + (f"; DIFFERENT: {differ}" if differ else ""))
    print(f"    against stock, frame by frame: {anim['frames']} animation frames, largest differences "
          f"{ {k: round(float(x), 6) for k, x in anim['maximum'].items()} }, issues {anim['issues'] or 'none'}; "
          f"physics {'equal' if rep['physics']['ok'] else 'DIFFERENT'}; bone flags {'equal' if rep['bone_flags_equal'] else 'DIFFERENT'}")
    if differ or not ok or len(left) > 3:
        raise SystemExit(f"{name}: the rebuilt canister differs from stock in more than its mesh - not staged")
    return out


if __name__ == "__main__":
    build()
