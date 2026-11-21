"""Stage 60 - compile the retargeted mesh into a Ronin body .mdl.

Takes the decompiled stock Ronin (its QC is the recipe for everything that is not mesh:
bones, attachments, hitboxes, IK chains, physics, the animation includes) and swaps only
the render meshes:

    build/retarget/<unit>/fit.npz                       the mesh on Ronin's skeleton
    build/tf2/decompiled/titan_light_locust/*.qc        stock recipe (from decompile53.py)
 -> build/tf2/decompiled/titan_light_locust/tsf/*.smd   one SMD per Ronin bodygroup
    build/tf2/decompiled/titan_light_locust/titan_light_locust_tsf.qc
    build/tf2/tsf_body/titan_light_locust.mdl           compiled with r2studiomdl

What changes in the QC: the six body `$bodygroup` blocks point at the new SMDs (the damaged
variant reuses the intact mesh for now), the `$lod` blocks are dropped (one level of detail
for now), and `$boneusage` / `$bbox`, which the decompiler retained from the stock mesh, are
removed so the compiler recomputes them.  Everything else is the stock text, untouched.

    python -m tsfronin.tf2_build [unit]
"""
import json
import math
import re
import subprocess
import time

import numpy as np

from . import config, tools
from .retarget import OUT_DIR, RONIN_SKELETON
from .tf2 import mdl53
from .tf2_extract import RONIN_BODY, STOCK_DIR

DECOMPILED = config.BUILD / "tf2" / "decompiled" / "titan_light_locust"
BODY_OUT = config.BUILD / "tf2" / "tsf_body"
HAS_BLANK = {"torso", "left_arm", "right_arm"}          # stock bodygroups that end with a `blank` state


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def smd_euler(R):
    """(rx, ry, rz) radians such that R = Rz(rz) @ Ry(ry) @ Rx(rx) - the SMD / studiomdl convention."""
    if abs(R[2, 0]) > 1 - 1e-6:                          # pitch of +-90 degrees: yaw and roll are one turn
        # (the threshold is loose on purpose: bone rotations are stored as 32-bit floats, and just
        # short of the pole the general formula divides noise by noise)
        ry = -math.copysign(math.pi / 2, R[2, 0])
        return math.atan2(-R[1, 2], R[1, 1]), ry, 0.0
    return math.atan2(R[2, 1], R[2, 2]), -math.asin(R[2, 0]), math.atan2(R[1, 0], R[0, 0])


def _euler_matrix(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def smd_header(bones):
    lines = ["version 1", "nodes"]
    lines += [f'{b["index"]} "{b["name"]}" {b["parent"]}' for b in bones]
    lines += ["end", "skeleton", "time 0"]
    worst = 0.0
    for b in bones:
        R = quat_matrix(b["local_quat"])
        e = smd_euler(R)
        worst = max(worst, float(np.abs(_euler_matrix(*e) - R).max()))
        p = b["local_pos"]
        lines.append(f'{b["index"]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}')
    if worst > 1e-5:
        raise SystemExit(f"bone rotation -> SMD Euler conversion is off by {worst:.2e}")
    lines.append("end")
    return lines


def texture_direction(ua, ub, uc):
    """A triangle with no area in the texture has no tangent direction, and the compiler stores a zero one for
    its corners.  -> None if the triangle is fine, else its three texture coordinates moved just enough apart:
    painted from a single texel, the corners are spread by a thousandth of the texture; painted from a line of
    texels, the corner opposite the longer edge is moved a thousandth off the line, so what it samples stays."""
    ua, ub, uc = (np.asarray(x, dtype=np.float64) for x in (ua, ub, uc))
    e1, e2 = ub - ua, uc - ua
    if abs(e1[0] * e2[1] - e1[1] * e2[0]) >= 1e-12:
        return None
    if max(np.abs(e1).max(), np.abs(e2).max()) < 1e-6:
        return ua, ua + (1e-3, 0.0), ua + (0.0, 1e-3)
    d = e1 if np.abs(e1).max() >= np.abs(e2).max() else e2
    off = np.array([-d[1], d[0]]) / np.hypot(d[0], d[1]) * 1e-3
    return (ua, ub, uc + off) if d is e1 else (ua, ub + off, uc)


def write_smds(unit, materials, lod=0):
    """materials: {"materials": {texture set: material name}, "glow": material name for the lit panels}
    lod: which level of detail of the fit; level 0 writes tsf_<group>.smd, level k tsf_<group>_lod<k>.smd"""
    level = "" if lod == 0 else f"_lod{lod}"
    if not (OUT_DIR / unit / f"fit{level}.npz").exists():
        raise SystemExit(f"the fit for level of detail {lod} is missing - run the retarget stage (python -m tsfronin.retarget)")
    fit = np.load(OUT_DIR / unit / f"fit{level}.npz")
    rep = json.loads((OUT_DIR / unit / f"report{level}.json").read_text(encoding="utf-8"))
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    M = np.array(sk["model_to_frame"])                    # frame = M @ model, so model = M.T @ frame
    _info, h, d = mdl53.parse(str(STOCK_DIR / RONIN_BODY.replace("/", "\\")))
    bones = mdl53.read_bones(d, h)
    header = smd_header(bones)
    out_dir = DECOMPILED / "tsf"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    spread = 0
    for gi, group in enumerate(rep["groups"]):
        lines = list(header) + ["triangles"]
        ntri = 0
        for part in rep["parts"]:
            pos = fit[f"{part}.pos"].astype(np.float64) @ M   # row vectors: p_model = M.T @ p_frame
            nrm = fit[f"{part}.nrm"].astype(np.float64) @ M
            uv, tri = fit[f"{part}.uv"], fit[f"{part}.tri"]
            bone, bw = fit[f"{part}.bone"], fit[f"{part}.bw"]
            sel = np.nonzero(fit[f"{part}.group"] == gi)[0]
            if not len(sel):
                continue
            texset = rep["parts"][part]["texset"]
            if texset not in materials["materials"]:
                raise SystemExit(f"part {part} uses texture set {texset!r}, which has no material in the pack")
            painted, lit = materials["materials"][texset], materials.get("glow")
            lit_tri = fit[f"{part}.mat"] == 1                    # Project MIKHAIL's colour-only (glow) triangles
            if lit_tri.any() and not lit:
                raise SystemExit(f"part {part} has lit panels but the pack has no glow material")
            # one text line per vertex, built once and reused by every triangle corner
            vtx = []
            for i in np.unique(tri[sel]):
                links = [(int(bone[i, k]), float(bw[i, k])) for k in range(2) if bw[i, k] > 1e-4]
                tot = sum(w for _b, w in links)
                links = [(b, w / tot) for b, w in links]
                parent = max(links, key=lambda t: t[1])[0]
                p, n = pos[i], nrm[i]
                vtx.append((int(i), (f"{parent} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}",
                                     f"{len(links)} " + " ".join(f"{b} {w:.6f}" for b, w in links))))
            halves = dict(vtx)
            line = lambda i, t: f"{halves[i][0]} {t[0]:.6f} {t[1]:.6f} {halves[i][1]}"
            text = {i: line(i, uv[i]) for i in halves}
            for t in sel:
                a, b, c = (int(x) for x in tri[t])
                # level 0 is the approved body and is written exactly as fitted (the compiler stores proper
                # tangents for it as it is); only the lower levels, where it does not, get the adjustment
                moved = texture_direction(uv[a], uv[b], uv[c]) if lod else None
                if moved is None:
                    lines += [lit if lit_tri[t] else painted, text[a], text[b], text[c]]
                else:                                            # see texture_direction()
                    lines += [lit if lit_tri[t] else painted, line(a, moved[0]), line(b, moved[1]), line(c, moved[2])]
                    spread += 1
            ntri += len(sel)
        lines.append("end")
        if not ntri:
            raise SystemExit(f"level of detail {lod} has no triangles for Ronin's bodygroup {group!r}")
        body = "\n".join(lines) + "\n"
        for suffix in ("", "_d"):                           # the damaged state reuses the intact mesh for now
            (out_dir / f"tsf_{group}{suffix}{level}.smd").write_text(body, encoding="ascii")
        stats[group] = ntri
    if spread:
        print(f"  level of detail {lod}: {spread} triangles with no area in the texture given a texture direction")
    return stats


def patch_qc(groups):
    src = (DECOMPILED / "titan_light_locust.qc").read_text(encoding="utf-8")
    text = src
    removed = {}
    for name, pattern in (("$boneusage lines", r"(?m)^\$boneusage .*\n"),
                          ("$bbox lines", r"(?m)^\$bbox .*\n")):
        text, n = re.subn(pattern, "", text)
        removed[name] = n
    # Levels of detail.  The stock recipe swaps each bodygroup mesh (intact and damaged) for a simpler one at three
    # distances; the same swaps are kept, pointing at the fit's own lower levels.  Lines for anything that is
    # not one of the six body bodygroups (the hit-point markers) stay as they are.
    stock_ref = {}
    for g in groups:
        m = re.search(r'(?ms)^\$bodygroup "%s"\n\{(.*?)^\}' % re.escape(g), src)
        refs = re.findall(r'studio "([^"]+)"', m.group(1)) if m else []
        if len(refs) != 2:
            raise SystemExit(f'expected an intact and a damaged mesh in the stock $bodygroup "{g}", found {refs}')
        stock_ref[refs[0]], stock_ref[refs[1]] = (g, ""), (g, "_d")
    level = [0]

    def lod_block(m):
        level[0] += 1
        out, repointed = [], 0
        for ln in m.group(2).splitlines():
            r = re.match(r'\treplacemodel "([^"]+)" "[^"]+"$', ln)
            if r and r.group(1) in stock_ref:
                g, d = stock_ref[r.group(1)]
                if not (DECOMPILED / "tsf" / f"tsf_{g}{d}_lod{level[0]}.smd").exists():
                    raise SystemExit(f"tsf_{g}{d}_lod{level[0]}.smd was not written - the fit has no level of detail {level[0]}")
                ln = f'\treplacemodel "tsf/tsf_{g}{d}.smd" "tsf/tsf_{g}{d}_lod{level[0]}.smd"'
                repointed += 1
            out.append(ln)
        if repointed != len(stock_ref):
            raise SystemExit(f"{m.group(1)}: repointed {repointed} meshes, expected {len(stock_ref)} "
                             f"(an intact and a damaged one for each of {len(groups)} bodygroups)")
        return m.group(1) + "\n{\n" + "\n".join(out) + "\n}"

    if config.BODY_LODS:
        text, n_lod = re.subn(r'(?ms)^(\$lod [^\n]*)\n\{\n(.*?)\n\}', lod_block, text)
        removed["$lod blocks kept and repointed"] = n_lod
    else:                                                     # full detail at every distance
        text, n_lod = re.subn(r"(?ms)^\$lod [^\n]*\n\{.*?^\}\n", "", text)
        removed["$lod blocks"] = n_lod
    for g in groups:
        block = [f'$bodygroup "{g}"', "{", f'\tstudio "tsf/tsf_{g}.smd"', f'\tstudio "tsf/tsf_{g}_d.smd"']
        if g in HAS_BLANK:
            block.append("\tblank")
        block.append("}")
        text, n = re.subn(r'(?ms)^\$bodygroup "%s"\n\{.*?^\}' % re.escape(g), lambda _m: "\n".join(block), text)
        if n != 1:
            raise SystemExit(f'expected exactly one $bodygroup "{g}" block in the stock QC, found {n}')
    text, restored = restore_hit_data_groups(text)
    # the named points this skin moves (where the stowed sword and the battery are drawn) - data/attach_map.json
    from . import tf2_attach
    points, _extra = tf2_attach.plan()
    text = tf2_attach.patch_qc(text, points)
    dst = DECOMPILED / "titan_light_locust_tsf.qc"
    dst.write_text(text, encoding="utf-8")
    kept = len(text.splitlines())
    print(f"  QC: {len(src.splitlines())} stock lines -> {kept}; removed {removed}; 6 bodygroups repointed; "
          f"hit-data group restored on {restored} hitboxes; {len(points)} named points moved")
    return dst, points


def restore_hit_data_groups(text, stock_file=None):
    """decompile53.py (r2-model-compiler 0.5.0) writes each $hbox with its name and crit flag but
    drops the hit-data group string ("hip", "torso", "left_arm" ...), which game scripts use to
    turn a hit into a bodygroup damage state.  Put it back from the stock file.  The compiler
    reads the optional tail of a $hbox as: "name" crit "hit data group"."""
    _info, h, d = mdl53.parse(str(stock_file or STOCK_DIR / RONIN_BODY.replace("/", "\\")))
    stock = mdl53.read_hitboxes(d, h)
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("$hbox ")]
    if len(idx) != len(stock):
        raise SystemExit(f"QC has {len(idx)} $hbox lines but the stock model has {len(stock)} hitboxes")
    n = 0
    for i, hb in zip(idx, stock):
        if not hb["hitdata"]:
            continue
        head = lines[i].split()[:9]                         # $hbox group "bone" min(3) max(3)
        lines[i] = " ".join(head) + f' "{hb["name"]}" {hb["crit"]} "{hb["hitdata"]}"'
        n += 1
    return "\n".join(lines), n


def compile_qc(qc, out):
    exe = tools.find("r2-model-compiler", "r2studiomdl.exe")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    t0 = time.time()
    # The compiler cannot open a path longer than 259 characters (Windows' old limit).  When the recipe's
    # folder is deep enough for one of its files to pass that, hand the compiler a short alias of the folder
    # (a directory junction in the temp folder, removed again straight after; the files themselves stay put).
    import os
    import tempfile
    from pathlib import Path
    work, alias = qc.parent, None
    if os.name == "nt" and max((len(str(p)) for p in work.rglob("*")), default=0) > 250:
        import _winapi
        alias = Path(tempfile.gettempdir()) / f"tsfqc_{os.getpid()}"
        if alias.exists():
            os.rmdir(alias)                                  # a leftover alias: removes the link, never its target
        _winapi.CreateJunction(str(work), str(alias))
        work = alias
    try:
        r = subprocess.run([str(exe), "-o", str(out), str(work / qc.name)], cwd=str(work), capture_output=True,
                           text=True, errors="replace")
    finally:
        if alias is not None:
            os.rmdir(alias)
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-14:]
    for ln in tail:
        print("    " + ln[-170:])
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"r2studiomdl failed (exit code {r.returncode})")
    warn = re.search(r"\((\d+) warnings?\)", r.stdout)
    if warn and int(warn.group(1)):
        hits = [ln for ln in r.stdout.splitlines() if "warning" in ln.lower()][:12]
        print(f"  {warn.group(1)} warnings:")
        for ln in hits:
            print("    " + ln[-170:])
    print(f"  compiled in {time.time() - t0:.1f}s -> {out}  ({out.stat().st_size:,} bytes)")
    return out


def compare_with_stock(out, points=None):
    """Every table that is not mesh must equal stock - except the named points moved on purpose (points),
    which must have compiled exactly as planned."""
    a, _ha, _da = mdl53.parse(str(STOCK_DIR / RONIN_BODY.replace("/", "\\")))
    b, _hb, _db = mdl53.parse(str(out))
    same, differ = [], []
    if points:
        from . import tf2_attach
        tf2_attach.check_built(b, points)
    for key in ("bones", "attachments", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams",
                "sequences", "nodes", "srcbonetransforms", "mass", "surfaceprop"):
        if key == "attachments" and points:
            continue                                         # checked point by point above
        x, y = a[key], b[key]
        if key == "bones":                                   # usage flags legitimately follow the new mesh
            strip = lambda L: [{k: v for k, v in e.items() if k != "flags"} for e in L]
            x, y = strip(x), strip(y)
        if key == "includemodels":
            x, y = [e["name"] for e in x], [e["name"] for e in y]
        (same if x == y else differ).append(key)
    # the summary above does not see every hitbox field; compare them all
    hx, hy = mdl53.read_hitboxes(_da, _ha), mdl53.read_hitboxes(_db, _hb)
    if hx == hy:
        same.append(f"all {len(hx)} hitboxes (bone, box, crit, hit-data group)")
        differ = [k for k in differ if k != "hitboxsets"]
    elif "hitboxsets" not in differ:
        differ.append("hitboxsets")
    print(f"  identical to stock: {', '.join(same)}")
    if differ:
        print(f"  DIFFERENT from stock: {', '.join(differ)}")
    print("  bodygroups:", [(p["name"], len(p["models"]), [m["numvertices"] for m in p["models"]]) for p in b["bodyparts"]])
    print("  materials:", b["textures"])
    return differ


MOD = "TSFRonin.Takemikazuchi"


def validate(out=None):
    """Run the compiler project's own validator against stock, in this process."""
    import contextlib
    import io
    import runpy
    import sys
    out = out or BODY_OUT / "titan_light_locust.mdl"
    script = tools.find("r2-model-compiler", "validate53.py")
    report = BODY_OUT / "validate.json"
    old = sys.argv
    sys.argv = [str(script), str(out), "--against", str(STOCK_DIR / RONIN_BODY.replace("/", "\\")),
                "--report", str(report)]
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = old
    rep = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    print(f"  validate53: exit code {code}, ok = {rep.get('ok')}")
    for key in ("structure", "metadata", "physics", "render_geometry", "map_collision"):
        v = rep.get(key)
        if isinstance(v, dict):
            short = {k: v[k] for k in list(v)[:6] if not isinstance(v[k], (dict, list)) or k == "issues"}
            print(f"    {key}: {json.dumps(short)[:260]}")
    return rep.get("ok")


def stage_mod():
    """build/mods/TSFRonin.Takemikazuchi/ - the same body file at Ronin's and Ronin Prime's paths
    (they share one skeleton, and the game shows whichever the player has equipped)."""
    import shutil
    from . import ingame
    from .tf2_extract import RONIN_PRIME
    from . import tf2_materials
    src = BODY_OUT / "titan_light_locust.mdl"
    pak = tf2_materials.OUT_DIR / config.UNIT / tf2_materials.PAK_NAME
    if not pak.exists():
        raise SystemExit(f"{pak} missing - run the materials stage first")
    from . import tf2_sword
    swords = {n: tf2_sword.OUT / f"{n}.mdl" for n in tf2_sword.SWORDS}
    missing = [n for n, p in swords.items() if not p.exists()]
    if missing:
        raise SystemExit(f"sword models not built yet: {missing} - run the sword stage first")
    # optional pieces, each its own file(s) in the mod so it can be left out without touching anything else
    extras = {}
    if config.HIDE_BATTERY:
        from . import tf2_battery
        extras[tf2_battery.BATTERY] = tf2_battery.OUT / tf2_battery.BATTERY.rsplit("/", 1)[1]
    if config.GUN_UNIT:
        from . import tf2_gun
        extras.update({f"{tf2_gun.GUN_DIR}/{n}.mdl": tf2_gun.OUT / f"{n}.mdl" for n in tf2_gun.GUNS})
    if config.POV_ARMS:
        from . import tf2_arms
        extras.update({f"{tf2_arms.ARMS_DIR}/{n}.mdl": tf2_arms.OUT / f"{n}.mdl" for n in tf2_arms.ARMS})
    missing = [g for g, p in extras.items() if not p.exists()]
    if missing:
        raise SystemExit(f"not built yet: {missing} - run the battery / gun / arms stage first (or switch them off)")
    # The pilot while boarding (config.HIDE_PILOT_ON_EMBARK): the pilots' shared animation file with the 28
    # Stryder-class boarding animations rebuilt (tsfronin/tf2_pilot.py).  A client script cannot do it (the game
    # refuses to hide anything the server owns; tried in 0.8.0 / 0.8.1), so the mod is model and texture files only.
    if config.HIDE_PILOT_ON_EMBARK:
        from . import tf2_pilot
        extras[tf2_pilot.FILE] = tf2_pilot.OUT / f"{tf2_pilot.NAME}.mdl"
        if not extras[tf2_pilot.FILE].exists():
            raise SystemExit(f"{extras[tf2_pilot.FILE]} missing - run the pilot stage first (or set HIDE_PILOT_ON_EMBARK=0)")
    mod = ingame.stage_mod(MOD, "Type-00R Takemikazuchi over Ronin and Ronin Prime, with its Type-74 blade as "
                                "Ronin's sword (work in progress)", version="0.10.1", load_priority=3)
    for game_path in (RONIN_BODY, RONIN_PRIME):
        dst = mod / "mod" / game_path.replace("/", "\\")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for n, p in swords.items():                              # world and first-person sword, regular and Prime
        dst = mod / "mod" / tf2_sword.SWORD_DIR.replace("/", "\\") / f"{n}.mdl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    for game_path, p in extras.items():
        dst = mod / "mod" / game_path.replace("/", "\\")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    # the material packs, loaded once the base game's common pack (shaders, depth materials) is in:
    # the main one (body and sword) and, when the gun is in the mod, the gun's own
    (mod / "paks").mkdir(exist_ok=True)
    wanted = [pak] + ([pak.with_name(tf2_materials.GUN_PAK_NAME)] if config.GUN_UNIT else [])
    for p in wanted:
        if not p.exists():
            raise SystemExit(f"{p} missing - run the materials stage first")
        shutil.copyfile(p, mod / "paks" / p.name)
    (mod / "paks" / "rpak.json").write_text(json.dumps({"Postload": {p.name: "common.rpak" for p in wanted}}, indent="\t")
                                            + "\n", encoding="utf-8")
    files = [p for p in mod.rglob("*") if p.is_file()]
    print(f"  staged {MOD}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB -> {mod}")
    return mod


def build(unit=None):
    unit = unit or config.UNIT
    from . import tf2_materials
    rep = json.loads((OUT_DIR / unit / "report.json").read_text(encoding="utf-8"))
    listing = tf2_materials.OUT_DIR / unit / "materials.json"
    if not listing.exists():
        raise SystemExit(f"{listing} missing - run the materials stage first (python -m tsfronin.tf2_materials)")
    materials = json.loads(listing.read_text(encoding="utf-8"))["units"][unit]
    t0 = time.time()
    stock_qc = (DECOMPILED / "titan_light_locust.qc").read_text(encoding="utf-8")
    lods = list(range(len(re.findall(r"(?m)^\$lod ", stock_qc)) + 1))    # the stock body's levels of detail
    if not config.BODY_LODS:
        lods = [0]
    for lod in lods:
        stats = write_smds(unit, materials, lod)
        print(f"  level of detail {lod}: {sum(stats.values()):,} triangles; per bodygroup {stats}")
    print(f"  SMDs written in {time.time() - t0:.1f}s")
    qc, points = patch_qc(rep["groups"])
    out = compile_qc(qc, BODY_OUT / "titan_light_locust.mdl")
    differ = compare_with_stock(out, points)
    if differ:
        raise SystemExit(f"the rebuilt body differs from stock outside its mesh and the planned points: {differ}")
    return {"mdl": str(out), "differences_from_stock": differ}


if __name__ == "__main__":
    import sys
    if "--stage" in sys.argv:                                    # put the built files together as a Northstar mod
        stage_mod()
    else:
        build(next((a for a in sys.argv[1:] if not a.startswith("-")), None))
