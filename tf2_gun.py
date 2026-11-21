"""Stage 67 (optional) - Ronin's Leadwall replaced by the Takemikazuchi's Type-87 assault cannon.

The Leadwall is two model files (the game calls the model titan_triple_threat): the one other players see in
the titan's hands, and the one the pilot sees from the cockpit, which also carries the 19 first-person
animations.  Each is treated like the sword: decompiled into its recipe, the mesh swapped, recompiled; bones
and every animation frame stay stock.  Two named points that say where effects appear (muzzle flash, ejected
cases) are moved onto the new gun, and the Leadwall's separate magazine and optic pieces become specks.

Nothing about how the weapon fires is in these files; it stays a Leadwall in everything but looks.
GUN_UNIT empty switches the whole stage off and leaves the two files out of the mod.

    build/clean/<gun unit>/                 the Type-87 as cleaned
    data/gun_map.json                       how it sits in Ronin's hand
 -> build/tf2/tsf_gun/<file>.mdl

    python -m tsfronin.tf2_gun              build both
    python -m tsfronin.tf2_gun --preview    draw the fit over the stock Leadwall
"""
import json
import re
import sys

import numpy as np

from . import config, tf2_attach, tf2_sword
from .tf2 import mdl53
from .tf2_build import compile_qc, smd_header
from .tf2_extract import STOCK_DIR

GUN_DIR = "models/weapons/titan_triple_threat"
GUNS = ["w_titan_triple_threat", "atpov_titan_triple_threat"]
OUT = config.BUILD / "tf2" / "tsf_gun"
GUN_MAP = config.REPO / "data" / "gun_map.json"
SPECK = 0.01                                                 # game units


def stock(name):
    return STOCK_DIR / GUN_DIR.replace("/", "\\") / f"{name}.mdl"


def gun_scale(gm):
    """Game units per metre for the gun, from the rule in gun_map.json."""
    from .retarget import OUT_DIR
    body = tf2_sword.body_scale()
    rule = gm["size"]["rule"]
    if rule == "body_scale":
        return body
    if rule == "times_body_scale":
        return body * float(gm["size"]["value"])
    if rule == "in_proportion_to_the_gun_hand":
        seg = json.loads((OUT_DIR / config.UNIT / "report.json").read_text(encoding="utf-8"))["segments"]
        hand = float(seg["hand_R"]["axial_stretch_vs_plain_scale"])
        if not 1.0 <= hand <= 3.0:
            raise SystemExit(f"the fit enlarges the gun hand {hand}x - not a size to give the gun")
        return body * hand
    raise SystemExit(f"gun_map.json: unknown size rule {rule!r}")


def fitted(lod=0):
    """The Type-87 in the Leadwall's frame (game units): one rigid copy - scale, the turn from data/gun_map.json,
    grip marker on Ronin's hand point."""
    gm = json.loads(GUN_MAP.read_text(encoding="utf-8"))
    clean = config.CLEAN_DIR / config.GUN_UNIT
    if not (clean / "manifest.json").exists():
        raise SystemExit(f"{clean} missing - run the gun-extract and gun-clean stages first")
    markers = {m["name"]: np.array(m["pos"], dtype=np.float64)
               for m in json.loads((clean / "skeleton.json").read_text(encoding="utf-8"))["markers"]}
    z = np.load(clean / f"lod{lod}.npz")
    R = tf2_attach._rotation(gm["axes"]["ronin_from_tsf"], "gun_map.json axes")
    s = gun_scale(gm)
    grip = markers[gm["grip"]["tsf_marker"]]
    at = np.array(gm["grip"]["ronin"], dtype=np.float64)
    place = lambda p: at + s * ((np.asarray(p, dtype=np.float64) - grip) @ R.T)
    return {"pos": place(z["body.pos"]), "nrm": z["body.nrm"].astype(np.float64) @ R.T, "uv": z["body.uv"],
            "tri": z["body.tri"].astype(np.int64), "glow": z["body.mat"] == 1, "scale": s, "place": place,
            "markers": {k: place(v) for k, v in markers.items()}}


def pieces(mesh):
    """-> the separate piece each triangle belongs to (triangles joined through shared corner positions),
    numbered by size, largest first"""
    pos, tri = np.asarray(mesh["pos"], dtype=np.float64), np.asarray(mesh["tri"], dtype=np.int64)
    _, inv = np.unique(np.round(pos / 0.02).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    parent = np.arange(inv.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in inv[tri]:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    root = np.array([find(a) for a in inv[tri[:, 0]]])
    ids, counts = np.unique(root, return_counts=True)
    order = {r: k for k, r in enumerate(ids[np.argsort(-counts, kind="stable")])}
    return np.array([order[r] for r in root])


def magazine_mask(mesh):
    """-> one flag per triangle: part of the box magazine (gun_map.json first_person_magazine), or None when the
    first-person reload is switched off (GUN_RELOAD=0: the gun then stays one piece)"""
    if not config.GUN_RELOAD:
        return None
    gm = json.loads(GUN_MAP.read_text(encoding="utf-8"))
    spec = gm["first_person_magazine"]
    R = tf2_attach._rotation(gm["axes"]["ronin_from_tsf"], "gun_map.json axes")
    clean = np.asarray(mesh["pos"], dtype=np.float64) @ R / mesh["scale"]   # back to the cleaned gun's metres
    box = spec["pieces_inside_m"]
    inside = np.all([(clean[:, k] >= box[a][0]) & (clean[:, k] <= box[a][1]) for k, a in enumerate("xyz")], axis=0)
    piece = pieces(mesh)
    tri = np.asarray(mesh["tri"], dtype=np.int64)
    chosen = [p for p in np.unique(piece) if inside[np.unique(tri[piece == p])].all()]
    if len(chosen) != int(spec["expect_pieces"]):
        raise SystemExit(f"gun_map.json first_person_magazine: {len(chosen)} pieces lie inside the box, "
                         f"{spec['expect_pieces']} expected (pieces {chosen}) - look at build/renders/retarget/gun_pieces.png")
    return np.isin(piece, chosen)


def check_fit(mesh):
    c = json.loads(GUN_MAP.read_text(encoding="utf-8"))["checks"]
    f, u = "xyz".index(c["forward_axis"]), "xyz".index(c["up_axis"])
    muzzle, upper = mesh["markers"][c["muzzle_marker"]], mesh["markers"][c["upper_barrel_marker"]]
    bulk = float(mesh["pos"][:, u].mean())
    if muzzle[f] <= 0 or upper[u] <= muzzle[u] or bulk <= 0:
        raise SystemExit(f"gun fit: the muzzle lands at {muzzle.round(1).tolist()}, the short barrel at "
                         f"{upper.round(1).tolist()}, the middle of the gun {bulk:+.1f} above the hand - the gun points "
                         f"backward or is upside down (gun_map.json axes)")
    lo, hi = mesh["pos"].min(0), mesh["pos"].max(0)
    print(f"  fit at {mesh['scale']:.4f} units/m: muzzle {muzzle.round(1).tolist()}, short barrel {upper.round(1).tolist()}; "
          f"bounds x {lo[0]:+.1f}..{hi[0]:+.1f}  y {lo[1]:+.1f}..{hi[1]:+.1f}  z {lo[2]:+.1f}..{hi[2]:+.1f}")


def moved_points(name, mesh):
    """-> {point: (bone, position on that bone, stock orientation)} for the effect points of one file"""
    gm = json.loads(GUN_MAP.read_text(encoding="utf-8"))["points"]
    info, h, d = mdl53.parse(str(stock(name)))
    bones = {b["name"]: b for b in mdl53.read_bones(d, h)}
    att = {a["name"]: a for a in info["attachments"]}
    out, where = {}, {}
    for point, spec in gm.items():
        if point.startswith("_"):
            continue
        if point not in att:
            raise SystemExit(f"{name} has no point named {point!r}")
        target = mesh["markers"][spec["tsf_marker"]] if "tsf_marker" in spec else mesh["place"](spec["tsf_point_m"])
        b = bones[att[point]["bone"]]
        out[point] = (b["name"], np.array(b["bind_rot"]).T @ (target - np.array(b["bind_pos"])), np.array(att[point]["rot"]))
        where[point] = target
    return out, att, where


def write_smds(name, lods, materials):
    gm = json.loads(GUN_MAP.read_text(encoding="utf-8"))
    _info, h, d = mdl53.parse(str(stock(name)))
    bones = mdl53.read_bones(d, h)
    index = {b["name"]: b["index"] for b in bones}
    want = gm["mesh_bone"][name]
    if want not in index:
        raise SystemExit(f"{name} has no bone {want!r}")
    bi = index[want]
    painted, lit = materials["materials"]["common"], materials.get("glow")
    dst = tf2_sword.DECOMPILED / name / "tsf"
    dst.mkdir(parents=True, exist_ok=True)
    tris = []
    magazine = gm["first_person_magazine"]

    def triangles(m, sel, bone):
        """SMD triangle lines for the selected triangles of mesh m, every corner on one bone"""
        corner = lambda i, t: (f"{bone} {m['pos'][i][0]:.6f} {m['pos'][i][1]:.6f} {m['pos'][i][2]:.6f} "
                               f"{m['nrm'][i][0]:.6f} {m['nrm'][i][1]:.6f} {m['nrm'][i][2]:.6f} {t[0]:.6f} {t[1]:.6f} 1 {bone} 1.000000")
        uv = m["uv"].astype(np.float64)
        e1, e2 = uv[m["tri"][:, 1]] - uv[m["tri"][:, 0]], uv[m["tri"][:, 2]] - uv[m["tri"][:, 0]]
        flat = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 1e-12
        point = flat & (np.abs(e1).max(1) < 1e-6) & (np.abs(e2).max(1) < 1e-6)
        spread = [0, 0]
        lines = []
        for t in np.nonzero(sel)[0]:
            a, b, c = m["tri"][t]
            mat = lit if m["glow"][t] else painted
            # A triangle with no area in the texture has no tangent direction, and the compiler stores a zero
            # one for its corners.  Two kinds occur.  Painted from a single texel: its corners are spread by a
            # thousandth of the texture (two texels).  Painted from a line of texels: the corner opposite the
            # longer edge is moved a thousandth off the line, so the line it samples stays what it was.
            if point[t]:
                spread[0] += 1
                lines += [mat, corner(a, uv[a]), corner(b, uv[a] + (1e-3, 0.0)), corner(c, uv[a] + (0.0, 1e-3))]
            elif flat[t]:
                spread[1] += 1
                along_ab = np.abs(e1[t]).max() >= np.abs(e2[t]).max()       # the longer edge leaves corner a for b
                d = e1[t] if along_ab else e2[t]
                off = np.array([-d[1], d[0]]) / np.hypot(d[0], d[1]) * 1e-3
                if along_ab:
                    lines += [mat, corner(a, uv[a]), corner(b, uv[b]), corner(c, uv[c] + off)]
                else:
                    lines += [mat, corner(a, uv[a]), corner(b, uv[b] + off), corner(c, uv[c])]
            else:
                lines += [mat, corner(a, uv[a]), corner(b, uv[b]), corner(c, uv[c])]
        return lines, spread

    split = name == magazine["file"] and config.GUN_RELOAD
    if split:
        from . import tf2_reload
        mag_bone = index[tf2_reload.MAGAZINE_BONE()]
    for lod in lods:
        m = fitted(lod)
        if m["glow"].any() and not lit:
            raise SystemExit("the gun has lit panels but the pack has no glow material for it")
        keep = np.ones(len(m["tri"]), dtype=bool)
        if split:
            mag = magazine_mask(m)
            keep = ~mag
            if lod == lods[0]:
                body, _s = triangles(m, mag, mag_bone)
                (dst / "magazine.smd").write_text("\n".join(list(smd_header(bones)) + ["triangles"] + body + ["end"]) + "\n",
                                                  encoding="ascii")
                print(f"    {name}: the box magazine ({int(mag.sum())} triangles) is the file's magazine piece, on "
                      f"{bones[mag_bone]['name']}")
        lines, spread = triangles(m, keep, bi)
        if lod == lods[0]:
            print(f"    {name} level {lod}: {spread[0]} single-texel and {spread[1]} texel-line triangles given a "
                  f"texture direction, of {int(keep.sum())}")
        (dst / f"gun_lod{lod}.smd").write_text("\n".join(list(smd_header(bones)) + ["triangles"] + lines + ["end"]) + "\n",
                                               encoding="ascii")
        tris.append(int(keep.sum()))
    for part, bone in gm["other_parts"].items():
        if part.startswith("_") or bone not in index:
            continue
        b = bones[index[bone]]
        c = np.array(b["bind_pos"], dtype=np.float64)
        corner = lambda dx, dz, u, v: (f"{b['index']} {c[0] + dx:.6f} {c[1]:.6f} {c[2] + dz:.6f} 0.000000 1.000000 0.000000 "
                                       f"{u:.6f} {v:.6f} 1 {b['index']} 1.000000")
        lines = list(smd_header(bones)) + ["triangles", painted, corner(0, 0, 0.5, 0.5), corner(0, SPECK, 0.5, 0.501),
                                           corner(SPECK, 0, 0.501, 0.5), "end"]
        (dst / f"speck_{part}.smd").write_text("\n".join(lines) + "\n", encoding="ascii")
    return tris


def patch_qc(name, lods, points, box=None):
    gm = json.loads(GUN_MAP.read_text(encoding="utf-8"))
    src = (tf2_sword.DECOMPILED / name / f"{name}.qc").read_text(encoding="utf-8")
    body = re.search(r'(?m)^\$body "([^"]+)" "([^"]+)"$', src)
    if not body or len(re.findall(r'(?m)^\$body ', src)) != 1:
        raise SystemExit(f"{name}.qc: expected exactly one $body line")
    ref = body.group(2)
    text = src.replace(body.group(0), f'$body "{body.group(1)}" "tsf/gun_lod0.smd"')
    # the Leadwall's separate pieces -> specks; in the first-person file, with its own reload, the magazine piece
    # is the Type-87's box magazine
    others = {}
    mag = gm["first_person_magazine"]
    split = name == mag["file"] and config.GUN_RELOAD

    def group(m):
        part = m.group(1)
        if part not in gm["other_parts"]:
            raise SystemExit(f"{name}.qc has a piece {part!r} that gun_map.json does not cover")
        mesh = "tsf/magazine.smd" if split and part == mag["part"] else f"tsf/speck_{part}.smd"
        inner, n = re.subn(r'studio "([^"]+)"', lambda s: others.setdefault(s.group(1), part) and f'studio "{mesh}"',
                           m.group(2))
        if n != 1:
            raise SystemExit(f"{name}.qc: piece {part!r} has {n} meshes, expected one")
        return f'$bodygroup "{part}"\n{{{inner}}}'

    text = re.sub(r'(?ms)^\$bodygroup "([^"]+)"\n\{(.*?)^\}', group, text)
    # lower levels of detail: the gun's own, in order; the specks need none
    level = [0]

    def lod(m):
        level[0] += 1
        if level[0] >= len(lods):
            raise SystemExit(f"{name}.qc has more levels of detail than the gun supplies")
        keep = []
        for ln in m.group(2).splitlines():
            r = re.match(r'\treplacemodel "([^"]+)" "[^"]+"$', ln)
            if r and r.group(1) == ref:
                keep.append(f'\treplacemodel "tsf/gun_lod0.smd" "tsf/gun_lod{lods[level[0]]}.smd"')
            elif r and r.group(1) in others:
                continue
            elif ln.strip():
                keep.append(ln)
        return m.group(1) + "\n{\n" + "\n".join(keep) + "\n}"

    text = re.sub(r'(?ms)^(\$lod [^\n]*)\n\{\n(.*?)^\}', lod, text)
    if level[0] != len(lods) - 1:
        raise SystemExit(f"{name}.qc has {level[0]} lower levels of detail, the fit supplies {len(lods) - 1}")
    # The Leadwall lists a second 'skin' (its own paint in a camouflage-capable version).  The Type-87 has its own
    # two materials and no alternative, so the list goes: the file then names only materials it draws with, as the
    # rebuilt sword does.  (A skin number the file does not have falls back to the first, as it does for the
    # battery canister, which the game gives the titan's skin number although it has one skin.)
    text, n_skin = re.subn(r'(?ms)^\$texturegroup "skinfamilies"\n\{.*?^\}\n', "", text)
    # The stored boxes (the model's and each sequence's) belong to the Leadwall's mesh.  For the world model, where
    # the gun does not move within its own file, they are rewritten for the new mesh; the first-person file's boxes
    # depend on its animations, so they are dropped and the compiler works them out, as for the first-person sword.
    if box is not None:
        nums = " ".join(f"{v:.6f}" for v in box)
        rest = re.search(r"(?m)^\$bbox (.*)$", text)
        if not rest:
            raise SystemExit(f"{name}.qc has no $bbox line")
        old = np.array([float(x) for x in rest.group(1).split()])
        # how far the new gun reaches beyond the Leadwall's own box, on its worst side
        pad = float(max(0.0, *(old[:3] - np.array(box[:3])), *(np.array(box[3:]) - old[3:])))

        def seq_box(m):
            b = np.array([float(x) for x in m.group(1).split()])
            if np.allclose(b, old, atol=1e-4):               # a sequence in which the gun sits as at rest
                return f"\tbbox {nums}"
            # a sequence that carries the gun somewhere else (the firing sequence is authored at the titan's
            # hand): its box cannot be redone without replaying it, so the stock box is grown by that reach
            g = np.concatenate([b[:3] - pad, b[3:] + pad])
            return "\tbbox " + " ".join(f"{v:.6f}" for v in g)

        text, n_hull = re.subn(r"(?m)^\$bbox .*$", f"$bbox {nums}", text)
        text, n_seq = re.subn(r"(?m)^\tbbox (.*)$", seq_box, text)
        boxes = f"{n_hull + n_seq} stored boxes rewritten for the new mesh (moving sequences: stock box grown by {pad:.1f})"
    else:
        text, n_hull = re.subn(r"(?m)^\$bbox .*\n", "", text)
        text, n_seq = re.subn(r"(?m)^\tbbox .*\n", "", text)
        boxes = f"{n_hull + n_seq} stored boxes dropped"
    text = tf2_attach.patch_qc(text, points)
    if split:
        from . import tf2_reload
        text = tf2_reload.patch_qc(text)
    dst = tf2_sword.DECOMPILED / name / f"{name}_tsf.qc"
    dst.write_text(text, encoding="utf-8")
    print(f"  {name}.qc: mesh -> tsf/gun_lod0.smd, {level[0]} lower levels repointed, pieces made specks: {sorted(set(others.values()))}, "
          f"{boxes}, points moved: {', '.join(points)}; bones, animations, physics untouched")
    return dst


def compare(name, out, points, stock_points):
    a, _ha, _da = mdl53.parse(str(stock(name)))
    b, _hb, _db = mdl53.parse(str(out))
    tf2_attach.check_built(b, points, stock_points)
    same, differ = [], []
    for key in ("bones", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams", "sequences", "nodes",
                "srcbonetransforms", "mass", "surfaceprop"):
        x, y = a.get(key), b.get(key)
        if key == "sequences":                               # the stored boxes follow the new mesh
            if config.GUN_RELOAD and name == json.loads(GUN_MAP.read_text(encoding="utf-8"))["first_person_magazine"]["file"]:
                from . import tf2_reload
                x = tf2_reload.expected_sequences(x)         # the events the new reload moves, moved
            strip = lambda L: [{k: v for k, v in e.items() if "bb" not in k.lower()} if isinstance(e, dict) else e for e in (L or [])]
            x, y = strip(x), strip(y)
        (same if x == y else differ).append(key)
    print(f"  {name}: identical to stock: {', '.join(same)}" + (f";  DIFFERENT: {', '.join(differ)}" if differ else ""))
    print(f"    mesh: {[(p['name'], [m['numvertices'] for m in p['models']]) for p in b['bodyparts']]}  materials: {b['textures']}")
    return differ


def build(names=GUNS):
    if not config.GUN_UNIT:
        print("  GUN_UNIT is empty: Ronin's Leadwall is left as it is")
        return {}
    from . import tf2_materials
    listing = tf2_materials.OUT_DIR / config.UNIT / "materials.json"
    units = json.loads(listing.read_text(encoding="utf-8"))["units"] if listing.exists() else {}
    if config.GUN_UNIT not in units:
        raise SystemExit(f"the material pack has no materials for {config.GUN_UNIT} - run the materials stage")
    mesh = fitted(0)
    check_fit(mesh)
    results = {}
    for name in names:
        tf2_sword.decompile_file(stock(name), tf2_sword.DECOMPILED / name)
        n_lower = len(re.findall(r"(?m)^\$lod ", (tf2_sword.DECOMPILED / name / f"{name}.qc").read_text(encoding="utf-8")))
        lods = list(range(n_lower + 1))
        tris = write_smds(name, lods, units[config.GUN_UNIT])
        points, stock_points, where = moved_points(name, mesh)
        print("  effect points on the new gun (left, up, forward): "
              + ", ".join(f"{k} {v.round(1).tolist()}" for k, v in where.items()))
        box = None
        if name.startswith("w_"):
            # the recipe's boxes are in the game's own axes: forward, left, up = the file's z, x, y
            lo, hi = mesh["pos"].min(0), mesh["pos"].max(0)
            box = (lo[2], lo[0], lo[1], hi[2], hi[0], hi[1])
        reload = name == json.loads(GUN_MAP.read_text(encoding="utf-8"))["first_person_magazine"]["file"] and config.GUN_RELOAD
        if reload:
            from . import tf2_reload
            tf2_reload.author()
        qc = patch_qc(name, lods, points, box)
        out = compile_qc(qc, OUT / f"{name}.mdl")
        differ = compare(name, out, points, stock_points)
        ok, rep = tf2_sword.validate_swapped(out, stock(name))
        if reload:
            # the six reloads are meant to differ from stock: every animation is compared on its own instead
            chk = tf2_reload.check(out)
            print("\n".join(chk["lines"]))
            ok = (rep["physics"]["ok"] and rep["bone_flags_equal"] and not rep["map_collision"].get("issues") and chk["ok"])
            rep["animations"]["issues"] = [] if chk["ok"] else ["see the per-animation comparison"]
        (OUT / f"{name}.validate.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        anim = rep["animations"]
        print(f"    triangles per level of detail {tris}")
        print(f"    against stock, frame by frame: {anim['frames']} animation frames ({anim['bone_frames']} bone-frames), "
              f"largest differences { {k: round(float(x), 6) for k, x in anim['maximum'].items()} }, issues {anim['issues'] or 'none'}")
        print(f"    physics {'equal' if rep['physics']['ok'] else 'DIFFERENT ' + str(rep['physics']['issues'])}; "
              f"bone flags {'equal' if rep['bone_flags_equal'] else 'DIFFERENT'}; "
              f"follows the new mesh as expected: {rep['metadata_differences']}")
        if differ or not ok:
            raise SystemExit(f"{name}: the rebuilt gun differs from stock outside its mesh and the planned points - not staged")
        results[name] = {"mdl": str(out), "animation_frames_checked": anim["frames"], "ok": ok}
    return results


def preview(px=6.0):
    """build/renders/retarget/gun_in_grip.png - the fitted Type-87 (colour) over the stock Leadwall (white dots),
    in the gun files' own frame, with the hand point and the effect points before and after."""
    from PIL import Image, ImageDraw, ImageFont
    from . import preview as pv
    clean = config.CLEAN_DIR / config.GUN_UNIT
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    m = fitted(0)
    z = np.load(clean / "lod0.npz")
    as_fit = {"body.pos": m["pos"].astype(np.float32), "body.nrm": m["nrm"].astype(np.float32), "body.uv": z["body.uv"],
              "body.tri": z["body.tri"], "body.mat": z["body.mat"]}
    SP, _SN, SC = pv._samples(as_fit, clean, man)
    SP = SP.astype(np.float64)
    name = GUNS[0]
    _info, h, d = mdl53.parse(str(stock(name)))
    Q = mdl53.vvd_vertices(d, h)[0]["pos"].astype(np.float64)
    bones = {b["name"]: np.array(b["bind_pos"]) for b in mdl53.read_bones(d, h)}
    points, _att, where = moved_points(name, m)
    marks = [(f"{k} (Leadwall)", bones[k], (150, 200, 255)) for k in points if k in bones]
    marks += [(f"{k} (moved)", p, (255, 220, 60)) for k, p in where.items()]
    marks += [("left hand", bones["l_hand_ik"], (150, 200, 255))] if "l_hand_ik" in bones else []
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for title, ah, av, depth in (("side on: along the barrel (Z) across, up (Y) up", 2, 1, 0),
                                 ("from above: along the barrel (Z) across, the titan's left (X) up", 2, 0, 1)):
        allp = np.vstack([SP, Q])
        lo_h, hi_h = allp[:, ah].min() - 10, allp[:, ah].max() + 10
        lo_v, hi_v = allp[:, av].min() - 12, allp[:, av].max() + 12
        W, H = int((hi_h - lo_h) * px), int((hi_v - lo_v) * px) + 30
        img = np.full((H, W, 3), (38, 40, 46), np.uint8)
        for pts, cols, size in ((Q, np.full((len(Q), 3), 225.0), 2), (SP, np.clip(255.0 * (SC / 255.0) ** 0.6, 0, 255), 3)):
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
        dr.line([(ox - 14, oy), (ox + 14, oy)], fill=(255, 80, 80), width=2)
        dr.line([(ox, oy - 14), (ox, oy + 14)], fill=(255, 80, 80), width=2)
        dr.text((ox + 6, oy + 6), "gun hand", fill=(255, 120, 120), font=font)
        for label, p, col in marks:
            qx, qy = (p[ah] - lo_h) * px, H - 1 - (p[av] - lo_v) * px
            dr.ellipse([qx - 5, qy - 5, qx + 5, qy + 5], outline=col, width=2)
            dr.text((qx - 50, qy - 22), label, fill=col, font=font)
        dr.text((8, 4), f"Type-87 (colour) over Ronin's Leadwall (white dots) - {title}", fill=(240, 240, 240), font=font)
        tiles.append(pil)
    sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles) + 8), (20, 20, 24))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y)); y += t.height + 8
    dst = config.RENDER_DIR / "retarget"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "gun_in_grip.png"
    sheet.save(out)
    print(f"wrote {out}  {sheet.size}")
    return out


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()
    else:
        build()
