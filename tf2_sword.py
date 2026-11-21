"""Stage 65 - Ronin's sword replaced by the Takemikazuchi's Type-74 blade.

Ronin's sword is four model files: the one other players see (in the hand and holstered on the back),
the one the pilot sees from the cockpit (which also carries the 17 first-person sword animations), and
the Prime versions of both.  Each is treated like the body: decompiled into its QC recipe, the mesh
swapped, recompiled; bones, attachments and every animation frame stay stock.

    build/clean/<sword unit>/                      the Type-74 blade as cleaned
    data/sword_map.json                            how it sits in Ronin's hand
    build/tf2/decompiled/<file>/                   stock recipes (decompile53.py)
 -> build/tf2/tsf_sword/<file>.mdl

    python -m tsfronin.tf2_sword              build all four
    python -m tsfronin.tf2_sword --roundtrip  prove the tools on the stock files first (rebuild unchanged, compare)
"""
import contextlib
import io
import json
import re
import runpy
import subprocess
import sys
import time

import numpy as np

from . import config, tools
from .tf2 import mdl53
from .tf2_build import compile_qc, smd_header
from .tf2_extract import STOCK_DIR

SWORD_DIR = "models/weapons/titan_sword"
SWORDS = ["w_titan_sword", "w_titan_sword_prime", "atpov_titan_sword", "atpov_titan_sword_prime"]
DECOMPILED = config.BUILD / "tf2" / "decompiled"
ROUNDTRIP = config.BUILD / "tf2" / "roundtrip"
OUT = config.BUILD / "tf2" / "tsf_sword"
SWORD_MAP = config.REPO / "data" / "sword_map.json"


def stock(name):
    return STOCK_DIR / SWORD_DIR.replace("/", "\\") / f"{name}.mdl"


def decompile(name):
    qc = DECOMPILED / name / f"{name}.qc"
    if qc.exists() and qc.stat().st_mtime >= stock(name).stat().st_mtime:
        return qc                                            # already taken apart from this very file: one process fewer
    return decompile_file(stock(name), DECOMPILED / name)


def decompile_file(src, dst):
    """any stock .mdl -> its QC recipe and meshes/animations in dst"""
    script = tools.find("r2-model-compiler", "decompile53.py")
    name = src.stem
    if not src.exists():
        raise SystemExit(f"{src} missing - run the tf2-extract stage first")
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, errors="replace")
    qc = dst / f"{name}.qc"
    if r.returncode != 0 or not qc.exists():
        print((r.stdout + r.stderr)[-1500:])
        raise SystemExit(f"decompile53.py failed on {name} (exit code {r.returncode})")
    print(f"  decompiled {name} in {time.time() - t0:.1f}s: {len(qc.read_text(encoding='utf-8').splitlines())} QC lines, "
          f"{len(list(dst.rglob('*.smd')))} SMD files")
    return qc


def validate(built, against, report):
    """The compiler project's own validator: structure, bone transforms, every animation frame, physics.
    -> (ok, report dict)"""
    script = tools.find("r2-model-compiler", "validate53.py")
    old = sys.argv
    sys.argv = [str(script), str(built), "--against", str(against), "--report", str(report)]
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = old
    rep = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    return code == 0 and bool(rep.get("ok")), rep, buf.getvalue()


def validate_swapped(built, against):
    """For a file whose mesh was replaced on purpose.  The validator's whole-file comparison stops at the first
    difference in mesh layout (the new blade has two materials where Ronin's has one) and then reports nothing
    else, so its checks are called one by one here: every animation frame, the physics, the collision used by
    maps, the bone flags, and the structure of the rebuilt file on its own.  -> (ok, report)"""
    script = tools.find("r2-model-compiler", "validate53.py")
    with contextlib.redirect_stdout(io.StringIO()):
        v = runpy.run_path(str(script), run_name="validate53_as_library")
    rep = {"structure": v["validate"](str(built), False),
           "animations": v["compare_animations"](str(against), str(built)),
           "physics": v["compare_physics"](str(against), str(built)),
           "map_collision": v["compare_map_collision"](str(against), str(built))}
    meta = v["compare_metadata"](str(against), str(built))
    # the boxes and the mesh file names legitimately follow the new mesh; the bone flags must not change
    rep["bone_flags_equal"] = "bone_flags differs" not in meta["issues"]
    rep["metadata_differences"] = meta["issues"]
    ok = (not rep["animations"]["issues"] and rep["physics"]["ok"] and rep["bone_flags_equal"]
          and not rep["map_collision"].get("issues"))
    return ok, rep


def _summary(rep):
    out = []
    for key, v in rep.items():
        if isinstance(v, dict):
            small = {k: x for k, x in v.items() if not isinstance(x, (dict, list))}
            issues = v.get("issues") or v.get("errors")
            out.append(f"    {key}: {json.dumps(small)[:230]}" + (f"  ISSUES {json.dumps(issues)[:300]}" if issues else ""))
        elif key != "ok":
            out.append(f"    {key}: {json.dumps(v)[:230]}")
    return "\n".join(out)


def roundtrip(names=SWORDS):
    """Rebuild each stock sword from its own decompiled recipe, untouched, and compare with the original."""
    ok_all = True
    for name in names:
        qc = decompile(name)
        out = compile_qc(qc, ROUNDTRIP / f"{name}.mdl")
        ok, rep, text = validate(out, stock(name), ROUNDTRIP / f"{name}.validate.json")
        print(f"  {name}: round trip {'matches stock' if ok else 'DIFFERS'}")
        print(_summary(rep))
        if not ok:
            print("    " + "\n    ".join(text.strip().splitlines()[-12:]))
        ok_all &= ok
    return ok_all


# ------------------------------------------------------------------ the fit
def body_scale():
    from .retarget import OUT_DIR
    rep = OUT_DIR / config.UNIT / "report.json"
    if not rep.exists():
        raise SystemExit(f"{rep} missing - run the retarget stage first (the sword uses the body's scale)")
    return float(json.loads(rep.read_text(encoding="utf-8"))["scale_units_per_metre"])


def fitted(lod=0):
    """The Type-74 blade in Ronin's sword frame (game units): positions, normals, uv, triangles, per-triangle
    glow flag.  One rigid copy: scale, the half turn from data/sword_map.json, grip marker on Ronin's hand point."""
    sm = json.loads(SWORD_MAP.read_text(encoding="utf-8"))
    clean = config.CLEAN_DIR / config.SWORD_UNIT
    markers = {m["name"]: np.array(m["pos"], dtype=np.float64)
               for m in json.loads((clean / "skeleton.json").read_text(encoding="utf-8"))["markers"]}
    z = np.load(clean / f"lod{lod}.npz")
    R = np.array(sm["axes"]["ronin_from_tsf"], dtype=np.float64)
    if abs(np.linalg.det(R) - 1.0) > 1e-9 or not np.allclose(R @ R.T, np.eye(3)):
        raise SystemExit("sword_map.json: axes.ronin_from_tsf is not a rotation")
    s = body_scale()
    grip = markers[sm["grip"]["tsf_marker"]]
    at = np.array(sm["grip"]["ronin"], dtype=np.float64)
    place = lambda p: at + s * ((np.asarray(p, dtype=np.float64) - grip) @ R.T)
    pos = place(z["body.pos"])
    nrm = z["body.nrm"].astype(np.float64) @ R.T
    return {"pos": pos, "nrm": nrm, "uv": z["body.uv"], "tri": z["body.tri"].astype(np.int64),
            "glow": z["body.mat"] == 1, "scale": s, "markers": {k: place(v) for k, v in markers.items()}}


def check_fit(mesh, name="w_titan_sword"):
    """The fitted blade must point where Ronin's sword points and end near Ronin's tip, edge on Ronin's edge side."""
    sm = json.loads(SWORD_MAP.read_text(encoding="utf-8"))["checks"]
    _info, h, d = mdl53.parse(str(stock(name)))
    bones = {b["name"]: np.array(b["bind_pos"]) for b in mdl53.read_bones(d, h)}
    tip, want = mesh["markers"][sm["tip_marker"]], bones[sm["ronin_tip_bone"]]
    axis = int(np.argmax(np.abs(want)))
    short = float(want[axis] - tip[axis]) * float(np.sign(want[axis]))
    if tip[axis] * want[axis] <= 0 or abs(short) > float(sm["max_tip_shortfall_units"]):
        raise SystemExit(f"sword fit: the blade's tip lands at {tip.round(1).tolist()}, Ronin's tip bone is at "
                         f"{want.round(1).tolist()} - the mapping in sword_map.json is wrong")
    edge = bones["sword_edge"]
    e_axis = int(np.argmax(np.abs(edge - [0, edge[1], 0])))                 # the axis the edge is offset along
    along = mesh["pos"][:, axis]
    mid = (along > 0.35 * tip[axis]) & (along < 0.65 * tip[axis]) if tip[axis] > 0 else (along < 0.35 * tip[axis]) & (along > 0.65 * tip[axis])
    belly = float(mesh["pos"][mid, e_axis].mean())
    if belly * edge[e_axis] <= 0:
        raise SystemExit(f"sword fit: the blade's belly is on the {'-' if belly < 0 else '+'} side, Ronin's edge bone on the other")
    lo, hi = mesh["pos"].min(0), mesh["pos"].max(0)
    print(f"  fit at {mesh['scale']:.4f} units/m: blade tip {tip.round(1).tolist()} (Ronin's tip bone {want.round(1).tolist()}, "
          f"{short:+.1f} units short), belly {belly:+.1f} on Ronin's edge side ({edge[e_axis]:+.1f}); "
          f"bounds x {lo[0]:+.1f}..{hi[0]:+.1f}  y {lo[1]:+.1f}..{hi[1]:+.1f}  z {lo[2]:+.1f}..{hi[2]:+.1f}")


def mesh_bone(name):
    """the one bone that carries the stock sword's mesh"""
    _info, h, d = mdl53.parse(str(stock(name)))
    bones = mdl53.read_bones(d, h)
    verts, _ = mdl53.vvd_vertices(d, h)
    used = np.unique(verts["bone"][:, 0][verts["weight"][:, 0] > 0])
    want = json.loads(SWORD_MAP.read_text(encoding="utf-8"))["mesh_bone"][name]
    if len(used) != 1 or bones[int(used[0])]["name"] != want:
        raise SystemExit(f"{name}: expected the whole mesh on {want!r}, found bones {[bones[int(b)]['name'] for b in used]}")
    return bones, int(used[0])


def write_smd(name, lod, materials):
    bones, bi = mesh_bone(name)
    m = fitted(lod)
    lines = list(smd_header(bones)) + ["triangles"]
    pos, nrm, uv = m["pos"], m["nrm"], m["uv"]
    text = [f"{bi} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {t[0]:.6f} {t[1]:.6f} 1 {bi} 1.000000"
            for p, n, t in zip(pos, nrm, uv)]
    painted, lit = materials["materials"]["common"], materials.get("glow")
    if m["glow"].any() and not lit:
        raise SystemExit("the blade has lit panels but the pack has no glow material for it")
    for t, (a, b, c) in enumerate(m["tri"]):
        lines += [lit if m["glow"][t] else painted, text[a], text[b], text[c]]
    lines.append("end")
    out = DECOMPILED / name / "tsf" / f"blade_lod{lod}.smd"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(m["tri"])


def world_box(meshes):
    """The fitted blade's own box (over every level of detail) as a QC wants it: the game's axes
    (forward, left, up) are the file's (z, x, y)."""
    lo = np.min([m["pos"].min(0) for m in meshes], axis=0)
    hi = np.max([m["pos"].max(0) for m in meshes], axis=0)
    return (lo[2], lo[0], lo[1], hi[2], hi[0], hi[1])


def patch_qc(name, lods, box=None):
    """box: (min forward, min left, min up, max forward, max left, max up) of the new mesh for a file in which the
    sword does not move (the world model: one `ref` pose).  None = drop the stored boxes and let the compiler work
    them out per animation (the first-person files, where it replays every sequence and gets them right)."""
    src = (DECOMPILED / name / f"{name}.qc").read_text(encoding="utf-8")
    body = re.search(r'(?m)^\$body "([^"]+)" "([^"]+)"$', src)
    if not body or len(re.findall(r'(?m)^\$body ', src)) != 1:
        raise SystemExit(f"{name}.qc: expected exactly one $body line")
    ref = body.group(2)
    text = src.replace(body.group(0), f'$body "{body.group(1)}" "tsf/blade_lod0.smd"')
    blocks = re.findall(r'(?m)^\treplacemodel "%s" "([^"]+)"$' % re.escape(ref), src)
    if len(blocks) != len(lods) - 1:
        raise SystemExit(f"{name}.qc has {len(blocks)} lower levels of detail, the fit supplies {len(lods) - 1}")
    for i, old in enumerate(blocks, start=1):
        text = text.replace(f'\treplacemodel "{ref}" "{old}"', f'\treplacemodel "tsf/blade_lod0.smd" "tsf/blade_lod{lods[i]}.smd"')
    # Boxes stored for the stock mesh.  With the lines simply dropped, r2studiomdl 0.5.0 writes a box for a
    # one-pose world model that mixes the new mesh with left-over stock numbers on the wrong axis (the blade came
    # out 178 units "wide" instead of 8).  The sword on Ronin's back in the loadout screen is a prop of this file
    # and the game places a prop by its stored box, so the world files get the new mesh's own box written out;
    # the first-person files are replayed per sequence by the compiler, which gets those right.
    # (Whether the wrong box cost anything is not established: a slowdown in that screen went away on the day
    # this was corrected, but the owner's video settings changed the same morning and the old file was smooth too.)
    if box is not None:
        nums = " ".join(f"{v:.6f}" for v in box)
        n_hull = len(re.findall(r"(?m)^\$bbox .*$", text))
        n_seq = len(re.findall(r"(?m)^\tbbox .*$", text))
        if n_hull != 1 or n_seq != len(re.findall(r"(?m)^\$sequence ", text)):
            raise SystemExit(f"{name}.qc: expected one $bbox and one bbox per sequence, found {n_hull} and {n_seq}")
        text = re.sub(r"(?m)^\$bbox .*$", f"$bbox {nums}", text)
        text = re.sub(r"(?m)^\tbbox .*$", f"\tbbox {nums}", text)
        boxes = f"{n_hull + n_seq} stored boxes rewritten for the new mesh ({nums})"
    else:
        text, n_hull = re.subn(r"(?m)^\$bbox .*\n", "", text)
        text, n_seq = re.subn(r"(?m)^\tbbox .*\n", "", text)
        boxes = f"{n_hull + n_seq} stored boxes dropped"
    dst = DECOMPILED / name / f"{name}_tsf.qc"
    dst.write_text(text, encoding="utf-8")
    print(f"  {name}.qc: mesh -> tsf/blade_lod0.smd, {len(blocks)} lower levels repointed, {boxes}; "
          f"bones, attachments, animations, physics untouched")
    return dst


def check_boxes(name, out, box, tol=0.05):
    """A world sword's stored boxes (the model's and its one sequence's) must be the mesh's own box."""
    _info, h, d = mdl53.parse(str(out))
    want = np.array(box, dtype=np.float64)
    got = [("model", np.array(list(h["hull_min"]) + list(h["hull_max"]), dtype=np.float64))]
    ssz = mdl53.size(mdl53.SEQDESC_FIELDS)
    for i in range(h["numlocalseq"]):
        o = h["localseqindex"] + i * ssz
        s = mdl53.unpack(mdl53.SEQDESC_FIELDS, d, o)
        got.append((mdl53.cstr(d, o + s["szlabelindex"]), np.array(list(s["bbmin"]) + list(s["bbmax"]), dtype=np.float64)))
    bad = [(n, g.round(2).tolist()) for n, g in got if np.abs(g - want).max() > tol]
    if bad:
        raise SystemExit(f"{name}: stored boxes differ from the mesh's box {want.round(2).tolist()}: {bad}")
    print(f"    stored boxes equal the mesh's own box (forward, left, up) {want[:3].round(2).tolist()} .. {want[3:].round(2).tolist()}")


def compare(name, out):
    """Everything that is not mesh must equal stock."""
    a, _ha, _da = mdl53.parse(str(stock(name)))
    b, _hb, _db = mdl53.parse(str(out))
    same, differ = [], []
    for key in ("bones", "attachments", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams",
                "sequences", "nodes", "srcbonetransforms", "mass", "surfaceprop"):
        x, y = a.get(key), b.get(key)
        if key == "sequences":                               # the stored boxes follow the new mesh
            strip = lambda L: [{k: v for k, v in e.items() if "bb" not in k.lower()} if isinstance(e, dict) else e for e in (L or [])]
            x, y = strip(x), strip(y)
        (same if x == y else differ).append(key)
    print(f"  {name}: identical to stock: {', '.join(same)}" + (f";  DIFFERENT: {', '.join(differ)}" if differ else ""))
    print(f"    mesh: {[(p['name'], [m['numvertices'] for m in p['models']]) for p in b['bodyparts']]}  materials: {b['textures']}")
    return differ


def build(names=SWORDS):
    from . import tf2_materials
    listing = tf2_materials.OUT_DIR / config.UNIT / "materials.json"
    units = json.loads(listing.read_text(encoding="utf-8"))["units"] if listing.exists() else {}
    if config.SWORD_UNIT not in units:
        raise SystemExit(f"the material pack has no materials for {config.SWORD_UNIT} - run the materials stage")
    materials = units[config.SWORD_UNIT]
    check_fit(fitted(0))
    results = {}
    for name in names:
        decompile(name)
        n_lower = len(re.findall(r"(?m)^\$lod ", (DECOMPILED / name / f"{name}.qc").read_text(encoding="utf-8")))
        lods = list(range(n_lower + 1))                      # Project MIKHAIL's own levels 0..n stand in for Ronin's
        tris = [write_smd(name, lod, materials) for lod in lods]
        box = world_box([fitted(lod) for lod in lods]) if name.startswith("w_") else None
        qc = patch_qc(name, lods, box)
        out = compile_qc(qc, OUT / f"{name}.mdl")
        if box is not None:
            check_boxes(name, out, box)
        differ = compare(name, out)
        ok, rep = validate_swapped(out, stock(name))
        (OUT / f"{name}.validate.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        anim = rep["animations"]
        worst = {k: round(float(x), 6) for k, x in anim["maximum"].items()}
        print(f"    triangles per level of detail {tris}")
        print(f"    against stock, frame by frame: {anim['frames']} animation frames ({anim['bone_frames']} bone-frames), "
              f"largest differences {worst}, issues {anim['issues'] or 'none'}")
        print(f"    physics {'equal' if rep['physics']['ok'] else 'DIFFERENT ' + str(rep['physics']['issues'])}; "
              f"bone flags {'equal' if rep['bone_flags_equal'] else 'DIFFERENT'}; "
              f"follows the new mesh as expected: {rep['metadata_differences']}")
        if differ or not ok:
            raise SystemExit(f"{name}: the rebuilt sword differs from stock outside its mesh - not staged")
        results[name] = {"mdl": str(out), "animation_frames_checked": anim["frames"], "ok": ok}
    return results


# ------------------------------------------------------------------ pictures for checking the fit
def preview(px=4.0):
    """Two sheets, drawn without a renderer:
    sword_in_grip.png   the fitted blade over Ronin's stock sword, in the sword files' own frame
    sword_on_back.png   the blade where Ronin holsters its sword, on the fitted body (arms hanging)"""
    from PIL import Image, ImageDraw, ImageFont
    from . import preview as pv
    from .retarget import OUT_DIR, RONIN_SKELETON
    clean = config.CLEAN_DIR / config.SWORD_UNIT
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    m = fitted(0)
    z = np.load(clean / "lod0.npz")
    as_fit = {"body.pos": m["pos"].astype(np.float32), "body.nrm": m["nrm"].astype(np.float32), "body.uv": z["body.uv"],
              "body.tri": z["body.tri"], "body.mat": z["body.mat"]}
    SP, SN, SC = pv._samples(as_fit, clean, man)
    SP = SP.astype(np.float64)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    dst = config.RENDER_DIR / "retarget"
    dst.mkdir(parents=True, exist_ok=True)

    # ---- in the grip: sword frame
    _info, h, d = mdl53.parse(str(stock("w_titan_sword")))
    verts, _ = mdl53.vvd_vertices(d, h)
    Q = verts["pos"].astype(np.float64)
    bones = {b["name"]: np.array(b["bind_pos"]) for b in mdl53.read_bones(d, h)}
    marks = [("Ronin tip bone", bones["sword_tip"]), ("Ronin edge bone", bones["sword_edge"]),
             ("blade tip marker", m["markers"]["blade2"])]
    tiles = []
    for title, ah, av, depth in (("side on: along the blade (Y) across, edge side (Z) up", 1, 2, 0),
                                 ("from above: along the blade (Y) across, thickness (X) up", 1, 0, 2)):
        allp = np.vstack([SP, Q])
        lo_h, hi_h = allp[:, ah].min() - 12, allp[:, ah].max() + 12
        lo_v, hi_v = allp[:, av].min() - 14, allp[:, av].max() + 14
        W, H = int((hi_h - lo_h) * px), int((hi_v - lo_v) * px) + 34
        img = np.full((H, W, 3), (38, 40, 46), np.uint8)
        for pts, cols, size in ((Q, np.full((len(Q), 3), 235.0), 2), (SP, np.clip(255.0 * (SC / 255.0) ** 0.6, 0, 255), 3)):
            order = np.argsort(pts[:, depth])
            x = ((pts[order, ah] - lo_h) * px).astype(int)
            y = (H - 1 - (pts[order, av] - lo_v) * px).astype(int)
            c = cols[order].astype(np.uint8)
            for dx in range(size):
                for dy in range(size):
                    img[np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)] = c
        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        ox, oy = (0 - lo_h) * px, H - 1 - (0 - lo_v) * px
        dr.line([(ox - 16, oy), (ox + 16, oy)], fill=(255, 80, 80), width=2)
        dr.line([(ox, oy - 16), (ox, oy + 16)], fill=(255, 80, 80), width=2)
        dr.text((ox + 6, oy + 6), "hand point", fill=(255, 120, 120), font=font)
        for label, p in marks:
            qx, qy = (p[ah] - lo_h) * px, H - 1 - (p[av] - lo_v) * px
            dr.ellipse([qx - 5, qy - 5, qx + 5, qy + 5], outline=(255, 220, 60), width=2)
            dr.text((qx - 60, qy - 24), label, fill=(255, 220, 60), font=font)
        for t in range(int(np.ceil(lo_h / 20) * 20), int(hi_h), 20):
            tx = (t - lo_h) * px
            dr.line([(tx, H - 12), (tx, H - 1)], fill=(150, 150, 160))
            dr.text((tx + 2, H - 32), str(t), fill=(150, 150, 160), font=font)
        dr.text((8, 4), f"Type-74 blade (colour) over Ronin's stock sword (white dots) - {title}", fill=(240, 240, 240), font=font)
        tiles.append(pil)
    sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles) + 8), (20, 20, 24))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y)); y += t.height + 8
    grip = dst / "sword_in_grip.png"
    sheet.save(grip)

    # ---- on the back: the holster bone carries the sword with its axes turned (sword x, y, z -> holster z, x, y;
    #      Ronin's own sword_tip_side / sword_edge_side attachments on that bone give the same mapping)
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    hb = next(b for b in sk["bones"] if b["name"] == "jx_sidearm_holster")
    Rb, tb = np.array(hb["rot"]), np.array(hb["pos"])
    to_world = lambda p: (p[:, [2, 0, 1]] @ Rb.T) + tb
    to_dir = lambda n: n[:, [2, 0, 1]] @ Rb.T
    fit = np.load(OUT_DIR / config.UNIT / "fit.npz")
    bman = json.loads((config.CLEAN_DIR / config.UNIT / "manifest.json").read_text(encoding="utf-8"))
    BP, BN, BC = pv._samples(fit, config.CLEAN_DIR / config.UNIT, bman, pv.arms_down(fit, sk["bones"], 90.0))
    stock_w = to_world(Q)
    clouds = [("Ronin's stock sword in the same holster (white dots)", stock_w, None, np.full((len(Q), 3), 235.0)),
              ("Type-74 blade in Ronin's holster", to_world(SP), to_dir(SN), SC)]
    up = np.array([0.0, 0.0, 1.0])
    tiles = []
    for label, P2, N2, C2 in clouds:
        for view in ("back", "left side"):
            right, toward = pv.VIEWS[view]
            P = np.vstack([BP.astype(np.float64), P2])
            N = np.vstack([BN, N2 if N2 is not None else np.tile(toward, (len(P2), 1))])
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
            C = np.vstack([BC, C2])
            is_sword = np.arange(len(P)) >= len(BP)
            half = np.abs(P @ right).max() + 8
            top = P[:, 2].max() + 6
            W, H = int(2 * half * px), int(top * px) + 60
            img = np.full((H, W, 3), (38, 40, 46), np.uint8)
            light = toward * 0.75 + up * 0.5 + right * 0.3
            light /= np.linalg.norm(light)
            shade = np.clip(N @ light, 0, 1) * 0.75 + 0.3
            col = np.clip(255.0 * (C / 255.0) ** 0.55 * shade[:, None], 0, 255)
            if N2 is None:
                col[is_sword] = 235.0
            col[is_sword & (N2 is not None)] = np.clip(col[is_sword & (N2 is not None)] * 1.5 + 30, 0, 255)   # lift the dark blade
            keep = ((N @ toward) > -0.2) | is_sword
            order = np.argsort((P @ toward)[keep])
            xs = ((P[keep] @ right + half) * px).astype(int)[order]
            ys = (H - 30 - P[keep][:, 2] * px).astype(int)[order]
            cs = col[keep][order].astype(np.uint8)
            for dx in (0, 1, 2):
                for dy in (0, 1, 2):
                    img[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = cs
            pil = Image.fromarray(img)
            dr = ImageDraw.Draw(pil)
            dr.text((8, 6), f"{label} - {view}", fill=(240, 240, 240), font=font)
            tiles.append(pil)
    sheet = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), (20, 20, 24))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, sheet.height - t.height)); x += t.width
    back = dst / "sword_on_back.png"
    sheet.save(back)
    tip_w = to_world(m["markers"]["blade2"][None, :])[0]
    grip_w = to_world(np.zeros((1, 3)))[0]
    print(f"wrote {grip} and {back}")
    print(f"  holstered: hand point at (forward, left, up) {grip_w.round(1).tolist()}, blade tip at {tip_w.round(1).tolist()}")
    return grip, back


if __name__ == "__main__":
    if "--roundtrip" in sys.argv:
        sys.exit(0 if roundtrip() else 1)
    build()
