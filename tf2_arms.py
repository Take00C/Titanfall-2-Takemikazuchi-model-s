"""Stage 68 - the Takemikazuchi's arms on Ronin's first-person arm files (what the pilot sees from the cockpit).

Ronin's cockpit view draws two more files than the body: the arms (one file for Ronin, one for Ronin Prime).
They are treated like the swords: decompiled into their QC recipe, the mesh swapped, recompiled; the 68 bones,
71 named points and 33 animations stay stock.

    build/retarget/<unit>/fit.npz              the approved body fit (its arm triangles are reused)
    data/pov_arms_map.json                     how they are laid on the first-person joints
    build/tf2/decompiled/<file>/               stock recipes (decompile53.py)
 -> build/tf2/tsf_arms/<file>.mdl

    python -m tsfronin.tf2_arms              build both
    python -m tsfronin.tf2_arms --roundtrip  prove the tools on the stock files first (rebuild unchanged, compare)
"""
import json
import re
import sys

import numpy as np

from . import config
from .retarget import OUT_DIR, RONIN_SKELETON
from .tf2 import mdl53
from .tf2_build import compile_qc, smd_header
from .tf2_extract import RONIN_BODY, STOCK_DIR
from . import tf2_sword

ARMS_DIR = "models/weapons/arms"
ARMS = ["pov_titan_light_locust", "pov_titan_light_ronin_prime"]
DECOMPILED = config.BUILD / "tf2" / "decompiled"
ROUNDTRIP = config.BUILD / "tf2" / "roundtrip"
OUT = config.BUILD / "tf2" / "tsf_arms"
ARMS_MAP = config.REPO / "data" / "pov_arms_map.json"


def stock(name):
    return STOCK_DIR / ARMS_DIR.replace("/", "\\") / f"{name}.mdl"


def decompile(name):
    qc = DECOMPILED / name / f"{name}.qc"
    if qc.exists() and qc.stat().st_mtime >= stock(name).stat().st_mtime:
        return qc
    return tf2_sword.decompile_file(stock(name), DECOMPILED / name)


def roundtrip(names=ARMS):
    """Rebuild each stock arm file from its own decompiled recipe, untouched, and compare with the original."""
    ok_all = True
    for name in names:
        qc = decompile(name)
        out = compile_qc(qc, ROUNDTRIP / f"{name}.mdl")
        ok, rep, text = tf2_sword.validate(out, stock(name), ROUNDTRIP / f"{name}.validate.json")
        print(f"  {name}: round trip {'matches stock' if ok else 'DIFFERS'}")
        print(tf2_sword._summary(rep))
        if not ok:
            print("    " + "\n    ".join(text.strip().splitlines()[-14:]))
        a, _h, _d = mdl53.parse(str(stock(name)))
        b, _h2, _d2 = mdl53.parse(str(out))
        print(f"    skin families stock {len(a['skinfamilies'])} x {len(a['skinfamilies'][0])}, rebuilt "
              f"{len(b['skinfamilies'])} x {len(b['skinfamilies'][0])}: {'equal' if a['skinfamilies'] == b['skinfamilies'] else 'DIFFERENT ' + str(b['skinfamilies'])}")
        ok_all &= ok
    return ok_all


# ------------------------------------------------------------------ the arms
def body_arms():
    """The approved body fit's arm triangles (level 0), in the body file's own axes (x left, y up, z forward):
    positions, normals, uv, triangles, the body bone each vertex rides (by name), a material name per triangle."""
    from . import tf2_materials
    amap = json.loads(ARMS_MAP.read_text(encoding="utf-8"))
    fit = np.load(OUT_DIR / config.UNIT / "fit.npz")
    rep = json.loads((OUT_DIR / config.UNIT / "report.json").read_text(encoding="utf-8"))
    listing = json.loads((tf2_materials.OUT_DIR / config.UNIT / "materials.json").read_text(encoding="utf-8"))
    materials = listing["units"][config.UNIT]
    M = np.array(json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))["model_to_frame"])
    _info, h, d = mdl53.parse(str(STOCK_DIR / RONIN_BODY.replace("/", "\\")))
    body_bones = mdl53.read_bones(d, h)
    want = [rep["groups"].index(g) for g in amap["take_groups"]]
    P, N, UV, T, B, MAT, GROUP = [], [], [], [], [], [], []
    base = 0
    for part in rep["parts"]:
        sel = np.nonzero(np.isin(fit[f"{part}.group"], want))[0]
        if not len(sel):
            continue
        tri = fit[f"{part}.tri"][sel].astype(np.int64)
        used = np.unique(tri)
        bw, bone = fit[f"{part}.bw"][used], fit[f"{part}.bone"][used]
        k = np.argmax(bw, axis=1)
        if (bw[np.arange(len(used)), k] < 0.9999).any():
            raise SystemExit(f"part {part}: an arm vertex is shared between two bones; the first-person arms are built rigid per bone")
        remap = np.full(int(tri.max()) + 1, -1, dtype=np.int64)
        remap[used] = base + np.arange(len(used))
        P.append(fit[f"{part}.pos"][used].astype(np.float64) @ M)
        N.append(fit[f"{part}.nrm"][used].astype(np.float64) @ M)
        UV.append(fit[f"{part}.uv"][used])
        B += [body_bones[int(b)]["name"] for b in bone[np.arange(len(used)), k]]
        T.append(remap[tri])
        texset = rep["parts"][part]["texset"]
        painted, lit = materials["materials"][texset], materials.get("glow")
        MAT += [lit if m == 1 else painted for m in fit[f"{part}.mat"][sel]]
        GROUP += [rep["groups"][int(g)] for g in fit[f"{part}.group"][sel]]
        base += len(used)
    return {"pos": np.vstack(P), "nrm": np.vstack(N), "uv": np.vstack(UV), "tri": np.vstack(T), "bone": np.array(B),
            "mat": MAT, "group": GROUP, "body_bones": body_bones, "scale": float(rep["scale_units_per_metre"]),
            "segments": rep["segments"]}


def onto_first_person(arms, pov_bones, quiet=False):
    """Lay the body's arm vertices on the first-person rig.  -> (positions, normals, report lines)"""
    amap = json.loads(ARMS_MAP.read_text(encoding="utf-8"))
    sections = amap["between_the_joints"]
    bb = {b["name"]: b for b in arms["body_bones"]}
    pb = {b["name"]: b for b in pov_bones}
    pos, nrm = arms["pos"].copy(), arms["nrm"].copy()
    notes, worst_hand = [], 0.0
    for name in sorted(set(arms["bone"])):
        if name not in pb:
            raise SystemExit(f"the body's arm uses bone {name!r}, which the first-person rig does not have")
        sel = arms["bone"] == name
        Rb, tb = np.array(bb[name]["bind_rot"]), np.array(bb[name]["bind_pos"])
        Rp, tp = np.array(pb[name]["bind_rot"]), np.array(pb[name]["bind_pos"])
        q = (arms["pos"][sel] - tb) @ Rb                  # the vertex in its bone's own axes
        nl = arms["nrm"][sel] @ Rb
        if name in sections:
            child = sections[name]
            ab = Rb.T @ (np.array(bb[child]["bind_pos"]) - tb)
            ap = Rp.T @ (np.array(pb[child]["bind_pos"]) - tp)
            Lb, Lp = float(np.linalg.norm(ab)), float(np.linalg.norm(ap))
            ab, ap = ab / Lb, ap / Lp
            ratio = Lp / Lb
            s = q @ ab
            q = q - np.outer(s, ab) + np.outer(s * ratio, ap)
            sn = nl @ ab
            nl = nl - np.outer(sn, ab) + np.outer(sn / ratio, ap)
            nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-12)
            across = np.linalg.norm(q - np.outer(q @ ap, ap), axis=1)
            before = np.linalg.norm((arms["pos"][sel] - tb) @ Rb - np.outer(s, ab), axis=1)
            if abs(float(across.max()) - float(before.max())) > 0.05:
                raise SystemExit(f"{name}: thickness changed ({float(before.max()):.2f} -> {float(across.max()):.2f} from the bone's axis)")
            # against the model's own proportions: the body fit's stretch along this section times this one
            seg = ("upper_arm_" if name.endswith("shoulder") else "forearm_") + name[4].upper()
            net = float(arms["segments"][seg]["axial_stretch_vs_plain_scale"]) * ratio
            notes.append(f"    {name:16s} {int(sel.sum()):5d} vertices: between the joints, {Lb:.2f} -> {Lp:.2f} units long (x{ratio:.4f}; "
                         f"x{net:.3f} of the model's own length at body scale, the body has x{net / ratio:.3f}); "
                         f"runs {float((s * ratio).min()):+.1f}..{float((s * ratio).max()):+.1f} along the bone, "
                         f"up to {float(across.max()):.1f} from its axis, as on the body ({float(before.max()):.1f})")
        pos[sel] = q @ Rp.T + tp
        nrm[sel] = nl @ Rp.T
        if name not in sections:
            moved = float(np.linalg.norm(pos[sel] - arms["pos"][sel], axis=1).max())
            worst_hand = max(worst_hand, moved)
    limit = float(amap["checks"]["hand_may_move_units"])
    if worst_hand > limit:
        raise SystemExit(f"a hand vertex would move {worst_hand:.3f} units against the approved body hand (limit {limit})")
    notes.append(f"    hands and fingers ({int((~np.isin(arms['bone'], list(sections))).sum())} vertices): same place on their bones; "
                 f"largest move against the body fit {worst_hand:.4f} units")
    if not quiet:
        print("\n".join(notes))
    return pos, nrm, notes


def finger_joints(bone, pos, pov_bones, quiet=False):
    """Which of Ronin's finger joints turns each finger piece.

    The body fit hangs the Takemikazuchi's three pieces of a finger on Ronin's three finger joints in order.  The
    Takemikazuchi's hand is smaller than Ronin's, so a piece's own knuckle (its inner end) is not where that
    joint is, and when Ronin curls the finger the piece swings about a point near its far end.  Here every
    piece is given to the joint of the same finger that sits at its inner end (within the distance in
    data/pov_arms_map.json); a piece with no joint there stays rigid with the piece it grows from (the palm, for
    the first).  Nothing is moved or resized: at rest the hand is unchanged.  -> (bone per vertex, table lines)"""
    amap = json.loads(ARMS_MAP.read_text(encoding="utf-8"))["finger_joints"]
    tol = float(amap["joint_within_units_of_the_inner_end"])
    by = {b["name"]: np.array(b["bind_pos"]) for b in pov_bones}
    out = bone.copy()
    lines = []
    for side in "lr":
        wrist = f"def_{side}_wrist"
        for finger in ("Thumb", "Index", "Mid", "Ring"):
            chain = [f"def_{side}_fin{finger}{k}" for k in "ABC"]
            prev = wrist
            for k, n in enumerate(chain):
                sel = bone == n
                if not sel.any():
                    continue
                P = pos[sel]
                d = np.linalg.norm(P - by[wrist], axis=1)
                inner = P[d <= np.percentile(d, 15)].mean(0)          # the end of the piece nearest the wrist
                gaps = [float(np.linalg.norm(by[c] - inner)) for c in chain[:k + 1]]
                j = int(np.argmin(gaps))
                driver = chain[j] if gaps[j] <= tol else prev
                out[sel] = driver
                lines.append(f"      {n:18s} {int(sel.sum()):5d} vertices: its inner end is "
                             + ", ".join(f"{g:.1f} from {c[-1]}" for g, c in zip(gaps, chain[:k + 1]))
                             + f" -> turned by {driver[6:]}" + ("" if driver != n else " (unchanged)"))
                prev = driver
    if not quiet:
        print("    finger pieces (joint of the same finger at each piece's inner end):")
        print("\n".join(lines))
    return out, lines


def write_smd(name, arms, pos, nrm):
    _info, h, d = mdl53.parse(str(stock(name)))
    bones = mdl53.read_bones(d, h)
    index = {b["name"]: b["index"] for b in bones}
    lines = list(smd_header(bones)) + ["triangles"]
    text = []
    for p, n, t, b in zip(pos, nrm, arms["uv"], arms["bone"]):
        bi = index[b]
        text.append(f"{bi} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {t[0]:.6f} {t[1]:.6f} 1 {bi} 1.000000")
    order = []
    for (a, b, c), m in zip(arms["tri"], arms["mat"]):
        if m not in order:
            order.append(m)
        lines += [m, text[a], text[b], text[c]]
    lines.append("end")
    out = DECOMPILED / name / "tsf" / "arms_lod0.smd"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return order


def stock_box(name):
    """The stock arm mesh's own box in the recipe's axes (forward, left, up = the file's z, x, y)."""
    _info, h, d = mdl53.parse(str(stock(name)))
    verts, _ = mdl53.vvd_vertices(d, h)
    p = verts["pos"].astype(np.float64)
    return np.concatenate([p.min(0)[[2, 0, 1]], p.max(0)[[2, 0, 1]]])


def stored_boxes(path):
    """-> (the model's box, [each sequence's box]) as stored in a compiled file, each (min x3, max x3)"""
    _info, h, d = mdl53.parse(str(path))
    hull = np.array(list(h["hull_min"]) + list(h["hull_max"]), dtype=np.float64)
    ssz = mdl53.size(mdl53.SEQDESC_FIELDS)
    seqs = []
    for i in range(h["numlocalseq"]):
        s = mdl53.unpack(mdl53.SEQDESC_FIELDS, d, h["localseqindex"] + i * ssz)
        seqs.append(np.array(list(s["bbmin"]) + list(s["bbmax"]), dtype=np.float64))
    return hull, seqs


def patch_qc(name, material_order, pos, piece_overhang=0.0):
    """piece_overhang: how far any one piece of the new arms reaches beyond the stock piece on the same bone
    (a piece is rigid on its bone, so this bounds the overhang in every pose, not only at rest)."""
    src = (DECOMPILED / name / f"{name}.qc").read_text(encoding="utf-8")
    body = re.findall(r'(?m)^\$body "([^"]+)" "([^"]+)"$', src)
    if len(body) != 1 or re.search(r"(?m)^\$(lod|bodygroup) ", src):
        raise SystemExit(f"{name}.qc: expected exactly one $body line and no $lod / $bodygroup")
    text = src.replace(f'$body "{body[0][0]}" "{body[0][1]}"', f'$body "{body[0][0]}" "tsf/arms_lod0.smd"')
    # Skin families: the game asks this file for a skin number (the warpaint).  Every number the stock file
    # answers must still exist, so the block keeps its number of rows, each row naming the new materials.
    block = re.search(r'(?ms)^\$texturegroup "skinfamilies"\n\{\n(.*?)^\}\n', text)
    if not block:
        raise SystemExit(f"{name}.qc: no skin family block")
    rows = [ln for ln in block.group(1).splitlines() if ln.strip()]
    row = "\t{ " + " ".join(f'"{m}"' for m in material_order) + " }"
    text = text.replace(block.group(0), '$texturegroup "skinfamilies"\n{\n' + "\n".join([row] * len(rows)) + "\n}\n")
    # Stored boxes: kept, and grown where the new arms reach beyond the stock arms (same rule as the world gun).
    lo, hi = pos.min(0), pos.max(0)
    new = np.array([lo[2], lo[0], lo[1], hi[2], hi[0], hi[1]])
    old = stock_box(name)
    pad = float(max(0.0, piece_overhang, *(old[:3] - new[:3]), *(new[3:] - old[3:])))

    def grow(m):
        v = np.array([float(x) for x in m.group(2).split()])
        g = np.concatenate([v[:3] - pad, v[3:] + pad])
        return m.group(1) + " ".join(f"{x:.6f}" for x in g)

    text, n_box = re.subn(r"(?m)^(\$bbox |\tbbox )(.*)$", grow, text)
    n_seq = len(re.findall(r"(?m)^\$sequence ", text))
    if n_box != 1 + n_seq:
        raise SystemExit(f"{name}.qc: {n_box} stored boxes found, expected the model's and one per sequence ({1 + n_seq})")
    dst = DECOMPILED / name / f"{name}_tsf.qc"
    dst.write_text(text, encoding="utf-8")
    print(f"  {name}.qc: mesh -> tsf/arms_lod0.smd; {len(rows)} skin rows kept, each naming {len(material_order)} new materials; "
          f"{n_box} stored boxes kept and grown by {pad:.2f} units (the furthest any new piece reaches beyond the stock piece "
          f"on its bone, or the new arms beyond the stock arms at rest, whichever is more); bones, points, animations untouched")
    return dst, len(rows), pad


def compare(name, out, rows, arms, pad, pos):
    """Everything that is not mesh must equal stock; and the mesh must be what was asked for."""
    a, ha, da = mdl53.parse(str(stock(name)))
    b, hb, db = mdl53.parse(str(out))
    same, differ = [], []
    for key in ("bones", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams",
                "sequences", "nodes", "srcbonetransforms", "mass", "surfaceprop"):
        (same if a.get(key) == b.get(key) else differ).append(key)
    for key in ("illumposition", "eyeposition"):            # passed through the recipe as text with six decimals
        gap = float(np.abs(np.array(a[key], dtype=np.float64) - np.array(b[key], dtype=np.float64)).max())
        (same if gap <= 1e-4 else differ).append(key)
    # stored boxes, read back from the built file: each must be the stock box grown by the stated amount
    (h0, s0), (h1, s1) = stored_boxes(stock(name)), stored_boxes(out)
    grown = lambda v: np.concatenate([v[:3] - pad, v[3:] + pad])
    off = [float(np.abs(grown(h0) - h1).max())] + [float(np.abs(grown(x) - y).max()) for x, y in zip(s0, s1)]
    if len(s0) == len(s1) and max(off) <= 0.01:
        same.append(f"{1 + len(s1)} stored boxes = stock grown by {pad:.2f}")
    else:
        differ.append(f"stored boxes (largest departure from stock + {pad:.2f}: {max(off):.3f})")
    # skin rows: every row must name the same materials in the same order, so any skin number looks the same
    if any(row != list(range(len(row))) for row in b["skinfamilies"]):
        differ.append(f"skin rows are not all the same materials: {b['skinfamilies']}")
    # the compiled mesh against what was asked for: on every bone, the piece's extent must be the intended one
    cv, _ = mdl53.vvd_vertices(db, hb)
    cb = np.array([b["bones"][int(x)]["name"] for x in cv["bone"][:, 0]])
    cp = cv["pos"].astype(np.float64)
    gap = 0.0
    for n in sorted(set(arms["bone"])):
        want, got = pos[arms["bone"] == n], cp[cb == n]
        if not len(got):
            differ.append(f"no compiled vertices on {n}")
            continue
        gap = max(gap, float(np.abs(want.min(0) - got.min(0)).max()), float(np.abs(want.max(0) - got.max(0)).max()))
    if gap > 0.002:
        differ.append(f"compiled mesh departs from the intended positions by {gap:.4f}")
    else:
        same.append(f"compiled pieces where intended (bone by bone, within {gap:.4f})")
    # named points: same names, bones and flags; numbers equal to the compiler's last digit (it re-derives them)
    pa, pb = a["attachments"], b["attachments"]
    labels = lambda L: [(p["name"], p["bone"], p["flags"]) for p in L]
    worst_p = max((float(np.abs(np.array(x["pos"]) - np.array(y["pos"])).max()) for x, y in zip(pa, pb)), default=0.0)
    worst_r = max((float(np.abs(np.array(x["rot"]) - np.array(y["rot"])).max()) for x, y in zip(pa, pb)), default=0.0)
    if labels(pa) == labels(pb) and worst_p <= 0.011 and worst_r <= 5e-5:
        same.append(f"{len(pa)} named points (largest difference {worst_p:.3f} units, {worst_r:.1e} in the turn)")
    else:
        differ.append(f"named points (position {worst_p:.3f}, turn {worst_r:.1e})")
    print(f"  {name}: identical to stock: {', '.join(same)}" + (f";  DIFFERENT: {', '.join(differ)}" if differ else ""))
    verts, _ = mdl53.vvd_vertices(db, hb)
    bones = mdl53.read_bones(db, hb)
    used = sorted({bones[int(x)]["name"] for x in np.unique(verts["bone"][:, 0])})
    sverts, _ = mdl53.vvd_vertices(da, ha)
    sused = {bones[int(x)]["name"] for x in np.unique(sverts["bone"][:, 0])}
    problems = list(differ)
    if (verts["numbones"] != 1).any():
        problems.append("a vertex with more than one bone")
    if not set(used) <= sused:
        problems.append(f"vertices on bones the stock arms do not use: {sorted(set(used) - sused)}")
    if len(b["skinfamilies"]) != rows:
        problems.append(f"{len(b['skinfamilies'])} skin families, stock has {rows}")
    if any("ronin" in t.lower() for t in b["textures"]):
        problems.append(f"a Ronin material is still referenced: {b['textures']}")
    tang = np.frombuffer(db, dtype="<f4", count=len(verts) * 4,
                         offset=mdl53.vvd_header(db, hb)["offset"] + mdl53.vvd_header(db, hb)["tangentDataStart"]).reshape(-1, 4)
    zero_t = int((np.linalg.norm(tang[:, :3], axis=1) < 0.5).sum())
    print(f"    mesh: {len(verts):,} vertices in the file ({len(sverts):,} stock), {len(arms['tri']):,} triangles; on {len(used)} bones "
          f"(stock {len(sused)}); materials {b['textures']}; skin families {len(b['skinfamilies'])} x {len(b['skinfamilies'][0])}; "
          f"vertices without a texture direction {zero_t}")
    return problems


def thickness(name, arms, pos):
    """How thick and how long each piece is on its bone, stock arms against new: what decides whether the new
    arms block more of the cockpit view.  A piece is rigid on its bone, so these hold in every pose."""
    sections = json.loads(ARMS_MAP.read_text(encoding="utf-8"))["between_the_joints"]
    _info, h, d = mdl53.parse(str(stock(name)))
    bones = mdl53.read_bones(d, h)
    by = {b["name"]: b for b in bones}
    verts, _ = mdl53.vvd_vertices(d, h)
    sp, sb = verts["pos"].astype(np.float64), np.array([bones[int(x)]["name"] for x in verts["bone"][:, 0]])
    kids = {}
    for b in bones:
        if b["parent"] >= 0:
            kids.setdefault(bones[b["parent"]]["name"], []).append(b)
    rows = []
    for n in sorted(set(sb) | set(arms["bone"])):
        R, t = np.array(by[n]["bind_rot"]), np.array(by[n]["bind_pos"])
        # the bone's axis: towards the next joint of the arm where the map names one, else its furthest child joint
        if n in sections:
            axis = np.array(by[sections[n]]["bind_pos"]) - t
        else:
            ch = sorted((c for c in kids.get(n, []) if c["name"].startswith("def_")),
                        key=lambda c: -float(np.linalg.norm(np.array(c["bind_pos"]) - t)))
            axis = (np.array(ch[0]["bind_pos"]) - t) if ch else R[:, 0]
        if np.linalg.norm(axis) < 1e-6:
            axis = R[:, 0]
        axis = axis / np.linalg.norm(axis)

        def span(P):
            if not len(P):
                return None
            q = P - t
            s = q @ axis
            r = np.linalg.norm(q - np.outer(s, axis), axis=1)
            return float(s.min()), float(s.max()), float(r.max())

        rows.append((n, span(sp[sb == n]), span(pos[arms["bone"] == n])))
    print("    piece by piece (along its bone from..to, furthest from the bone's axis), stock arms | new arms:")
    for n, a, b in rows:
        fa = f"{a[0]:+6.1f}..{a[1]:+6.1f}, {a[2]:5.1f}" if a else "        (none)       "
        fb = f"{b[0]:+6.1f}..{b[1]:+6.1f}, {b[2]:5.1f}" if b else "(none)"
        print(f"      {n:18s} {fa}  |  {fb}")
    return rows


def build(names=ARMS):
    arms = body_arms()
    results = {}
    for name in names:
        decompile(name)
        _info, h, d = mdl53.parse(str(stock(name)))
        pov_bones = mdl53.read_bones(d, h)
        per_group = ", ".join(g + " " + str(arms["group"].count(g)) for g in sorted(set(arms["group"])))
        print(f"  {name}: {len(arms['tri']):,} arm triangles of the approved body fit ({per_group})")
        pos, nrm, _notes = onto_first_person(arms, pov_bones)
        body_fit_arms = arms
        if json.loads(ARMS_MAP.read_text(encoding="utf-8"))["finger_joints"]["rule"] == "joint_at_the_inner_end":
            arms = dict(arms, bone=finger_joints(arms["bone"], pos, pov_bones)[0])
        order = write_smd(name, arms, pos, nrm)
        pieces = thickness(name, arms, pos)
        over = max([0.0] + [max(new[2] - old[2], new[1] - old[1], old[0] - new[0]) for _n, old, new in pieces if old and new])
        qc, rows, pad = patch_qc(name, order, pos, over)
        out = compile_qc(qc, OUT / f"{name}.mdl")
        problems = compare(name, out, rows, arms, pad, pos)
        ok, rep = tf2_sword.validate_swapped(out, stock(name))
        unexpected = sorted(set(rep["metadata_differences"]) - {"hull_min differs", "hull_max differs", "model_names differs"})
        if unexpected or not ok:
            problems.append(f"the validator reports {unexpected or 'a failed check'}")
        (OUT / f"{name}.validate.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        anim = rep["animations"]
        worst = {k: round(float(x), 6) for k, x in anim["maximum"].items()}
        print(f"    against stock, frame by frame: {anim['frames']} animation frames ({anim['bone_frames']} bone-frames), "
              f"largest differences {worst}, issues {anim['issues'] or 'none'}")
        print(f"    bone flags {'equal' if rep['bone_flags_equal'] else 'DIFFERENT'}; physics {rep['physics']}; "
              f"follows the new mesh as expected: {rep['metadata_differences']}")
        if anim["issues"] or not rep["bone_flags_equal"]:
            problems.append("animations or bone flags differ from stock")
        if problems:
            out.unlink()                                   # a rejected file must not be left where the mod stage would take it
            raise SystemExit(f"{name}: the rebuilt arms differ from stock outside their mesh - removed, not staged: {problems}")
        results[name] = {"mdl": str(out), "animation_frames_checked": anim["frames"], "triangles": int(len(arms["tri"]))}
        arms = body_fit_arms                                 # the next file starts from the body fit's own joints again
    return results


# ------------------------------------------------------------------ a picture of the cockpit view
def _smd_frames(path, wanted):
    """-> (names, parents, {frame number: (pos, euler)}) of an animation SMD; wanted: frame numbers, -1 = last"""
    names, parents, frames, cur = {}, {}, {}, None
    section = None
    for ln in open(path, encoding="ascii", errors="replace"):
        s = ln.strip()
        if s in ("nodes", "skeleton", "triangles"):
            section = s
            continue
        if s == "end":
            section = None
            continue
        if section == "nodes":
            i, rest = s.split(" ", 1)
            nm = rest[rest.index('"') + 1:rest.rindex('"')]
            names[int(i)], parents[int(i)] = nm, int(rest[rest.rindex('"') + 1:])
        elif section == "skeleton":
            if s.startswith("time"):
                cur = int(s.split()[1])
                frames[cur] = {}
            else:
                v = s.split()
                frames[cur][int(v[0])] = [float(x) for x in v[1:7]]
    last = max(frames)
    return names, parents, {w: frames[last if w == -1 else w] for w in wanted}, last


def _posed(names, parents, frame):
    """world matrix of every bone of one SMD frame, by name"""
    from .tf2_build import _euler_matrix
    W = {}
    for i in sorted(names):
        p = frame[i]
        L = np.eye(4)
        L[:3, :3], L[:3, 3] = _euler_matrix(p[3], p[4], p[5]), p[:3]
        W[i] = L if parents[i] < 0 else W[parents[i]] @ L
    return {names[i]: W[i] for i in W}


def preview():
    """build/renders/retarget/cockpit_arms.png - see tf2_arms_preview"""
    from . import tf2_arms_preview
    return tf2_arms_preview.draw()


if __name__ == "__main__":
    if "--roundtrip" in sys.argv:
        sys.exit(0 if roundtrip() else 1)
    build()
