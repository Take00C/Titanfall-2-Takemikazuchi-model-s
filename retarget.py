"""Stage 50 - hang the cleaned Takemikazuchi on the stock Ronin skeleton.

Reads   build/clean/<unit>/          the cleaned mesh with its original skin
        build/tf2/ronin/skeleton.json  Ronin's bind pose (pipeline frame)
        data/bone_map.json           the hand-authored mapping
Writes  build/retarget/<unit>/fit.npz    every vertex placed in Ronin's bind pose, with Ronin bone weights
        build/retarget/<unit>/report.json

Nothing of Ronin's is changed: the skeleton is the stock one, and each Takemikazuchi vertex is
moved to where its Ronin bone sits in the bind pose and weighted to that bone (at most two).

    python -m tsfronin.retarget [unit]
"""
import json
import re

import numpy as np

from . import config

OUT_DIR = config.BUILD / "retarget"
RONIN_SKELETON = config.BUILD / "tf2" / "ronin" / "skeleton.json"
BONE_MAP = config.REPO / "data" / "bone_map.json"


def shortest_arc(u, d):
    """Rotation taking unit vector u onto unit vector d by the smallest turn."""
    v = np.cross(u, d)
    c = float(np.dot(u, d))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-9:                                   # opposite: turn half way round any perpendicular
        a = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(u, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        return 2 * np.outer(a, a) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Block:
    """Rigid piece: uniform scale about an anchor, no rotation."""

    def __init__(self, tsf_anchor, ronin_anchor, scale):
        self.a, self.b, self.s = tsf_anchor, ronin_anchor, scale

    def place(self, p, n):
        return self.b + self.s * (p - self.a), n


class Segment:
    """A straight Takemikazuchi limb section laid along a chain of Ronin bones."""

    def __init__(self, name, tsf_from, tsf_to, pts, bones, scale, stretch="axial", align=True,
                 max_scale=None, blend=5.0):
        self.name, self.bones, self.blend = name, bones, blend
        self.o = tsf_from
        self.L = float(np.linalg.norm(tsf_to - tsf_from))
        self.u = (tsf_to - tsf_from) / self.L
        self.P = [np.asarray(x, dtype=np.float64) for x in pts]
        d = [self.P[i + 1] - self.P[i] for i in range(len(self.P) - 1)]
        self.len = [float(np.linalg.norm(x)) for x in d]
        self.dir = [x / l for x, l in zip(d, self.len)]
        self.cum = np.concatenate([[0.0], np.cumsum(self.len)])
        chain = float(self.cum[-1])
        if stretch == "axial":
            self.axial, self.radial = chain / self.L, scale
        elif stretch == "uniform":
            g = chain / self.L
            if max_scale is not None:
                g = min(g, scale * max_scale)
            self.axial = self.radial = g
        else:
            self.axial = self.radial = scale
        self.align = align
        if align:
            self.Q = [shortest_arc(self.u, self.dir[0])]
            for i in range(1, len(self.dir)):
                self.Q.append(shortest_arc(self.dir[i - 1], self.dir[i]) @ self.Q[-1])
        else:
            self.Q = [np.eye(3)] * len(self.dir)
        self.info = {"tsf_length_m": round(self.L, 3), "ronin_chain_units": round(chain, 2),
                     "axial_units_per_m": round(self.axial, 3), "radial_units_per_m": round(self.radial, 3),
                     "axial_stretch_vs_plain_scale": round(self.axial / scale, 3)}

    def _rigid(self, i, a2, r, n_r, n_a):
        axis = self.dir[i] if self.align else self.u
        pos = self.P[i] + (a2 - self.cum[i])[:, None] * axis + self.radial * (r @ self.Q[i].T)
        nrm = (n_r / self.radial) @ self.Q[i].T + (n_a / self.axial)[:, None] * axis
        return pos, nrm

    def place(self, p, n):
        """-> positions, normals, (V,2) chain-bone indices, (V,2) weights"""
        rel = p - self.o
        a = rel @ self.u
        r = rel - np.outer(a, self.u)
        n_a = n @ self.u
        n_r = n - np.outer(n_a, self.u)
        a2 = a * self.axial
        nseg = len(self.dir)
        idx = np.clip(np.searchsorted(self.cum[1:-1], a2, side="right"), 0, nseg - 1)
        pos = np.zeros_like(p)
        nrm = np.zeros_like(n)
        bi = np.stack([idx, idx], axis=1)
        bw = np.stack([np.ones(len(p)), np.zeros(len(p))], axis=1)
        for i in range(nseg):
            m = idx == i
            if m.any():
                pos[m], nrm[m] = self._rigid(i, a2[m], r[m], n_r[m], n_a[m])
        for j in range(1, nseg):                        # smooth the bend at each interior joint
            w = self.blend
            m = np.abs(a2 - self.cum[j]) < w
            if not m.any():
                continue
            t = (a2[m] - (self.cum[j] - w)) / (2 * w)
            p0, n0 = self._rigid(j - 1, a2[m], r[m], n_r[m], n_a[m])
            p1, n1 = self._rigid(j, a2[m], r[m], n_r[m], n_a[m])
            pos[m] = (1 - t)[:, None] * p0 + t[:, None] * p1
            nrm[m] = (1 - t)[:, None] * n0 + t[:, None] * n1
            bi[m] = [j - 1, j]
            bw[m] = np.stack([1 - t, t], axis=1)
        return pos, nrm, bi, bw


def expand(text, side):
    return text.replace("{S}", side.upper()).replace("{s}", side.lower())


def connected_pieces(pos, tri):
    """-> for every vertex, the id of the connected piece of mesh it belongs to.  Vertices at the same place
    count as one (texture seams split them), so a piece is what a modeller would call one shell."""
    _u, weld = np.unique(np.round(pos, 4), axis=0, return_inverse=True)
    weld = weld.ravel()
    parent = list(range(int(weld.max()) + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in weld[tri].tolist():
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    return np.array([find(a) for a in weld.tolist()], dtype=np.int64)


def body_scale(bm, man, sk):
    """Game units per metre, from the rule in bone_map.json."""
    rule = bm["size"]["rule"]
    if rule == "same_height_as_ronin":                   # overall height of the stock Ronin mesh / of the model
        up = "XYZ".index(sk["model_space"]["up"][1])
        return float(sk["mesh_bounds_model_space"]["size"][up]) / float(man["bounds_m"]["height_z"])
    if rule == "units_per_metre":
        return float(bm["size"]["value"])
    raise SystemExit(f"bone_map.json: unknown size rule {rule!r}")


def _block(bm, tpos, rpos, s):
    for kind, table in (("tsf", tpos), ("ronin", rpos)):
        missing = [n for n in bm["anchor"][kind] if n not in table]
        if missing:
            raise SystemExit(f"bone_map.json anchor names {missing}, which do not exist")
    a_tsf = np.mean([tpos[n] for n in bm["anchor"]["tsf"]], axis=0)
    a_ron = np.mean([rpos[n] for n in bm["anchor"]["ronin"]], axis=0)
    a_tsf[2] = a_ron[2] = 0.0
    return Block(a_tsf, a_ron, s)


def body_block(unit=None, bone_map=None):
    """Where the rigid body stands on Ronin: the same placement the fit uses, for anything else that has to
    be put on the body in the model's own terms (a stowed weapon on its holder, for instance).
    block.place(point in the cleaned model, metres) -> point on the fitted body, game units."""
    unit = unit or config.UNIT
    clean = config.CLEAN_DIR / unit
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    tsf = json.loads((clean / "skeleton.json").read_text(encoding="utf-8"))["joints"]
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    bm = json.loads((bone_map or BONE_MAP).read_text(encoding="utf-8"))
    return _block(bm, {j["name"]: np.array(j["pos"], dtype=np.float64) for j in tsf},
                  {b["name"]: np.array(b["pos"], dtype=np.float64) for b in sk["bones"]}, body_scale(bm, man, sk))


LODS = [0, 1, 2, 3]      # Ronin's body has four levels of detail; Project MIKHAIL's levels 0-3 stand in for them


def retarget_all(unit=None):
    """The fit for every level of detail (the same bone map; level 0 is the model seen up close)."""
    return [retarget(unit, lod=lod) for lod in LODS]


def retarget(unit=None, bone_map=None, tag=None, lod=0):
    """bone_map / tag: try a different mapping file without touching the real fit
    (the result goes to build/retarget/<unit>__<tag>/)."""
    unit = unit or config.UNIT
    clean = config.CLEAN_DIR / unit
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    tsf = json.loads((clean / "skeleton.json").read_text(encoding="utf-8"))["joints"]
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    ron = sk["bones"]
    bm = json.loads((bone_map or BONE_MAP).read_text(encoding="utf-8"))
    z = np.load(clean / f"lod{lod}.npz")
    s = body_scale(bm, man, sk)

    tpos = {j["name"]: np.array(j["pos"], dtype=np.float64) for j in tsf}
    tname = [j["name"] for j in tsf]
    rpos = {b["name"]: np.array(b["pos"], dtype=np.float64) for b in ron}
    rindex = {b["name"]: b["index"] for b in ron}
    rparent = {b["name"]: b["parent_name"] for b in ron}

    def need(table, name, what):
        if name not in table:
            raise SystemExit(f"bone_map.json names {what} {name!r}, which does not exist")
        return table[name]

    # ---- the plan: for every Takemikazuchi joint, a placer and a weighting rule
    # the rigid body: lined up with Ronin front-to-back and side-to-side at the anchor joints, and standing
    # on Ronin's ground (the cleaned model's soles are at Z = 0, Ronin's origin is on its ground)
    block = _block(bm, tpos, rpos, s)
    plan = {}                                            # joint -> (placer, pinned Ronin bone or None)
    segments = []
    for spec in bm["segments"]:
        for side in "LR":
            a, b = (expand(x, side) for x in spec["tsf"])
            chain = [expand(x, side) for x in spec["ronin"]]
            seg = Segment(f"{spec['name']}_{side}", need(tpos, a, "Takemikazuchi joint"),
                          need(tpos, b, "Takemikazuchi joint"), [need(rpos, c, "Ronin bone") for c in chain],
                          chain, s, spec.get("stretch", "axial"), spec.get("align", True),
                          spec.get("max_scale"), float(bm.get("blend_half_width", 5.0)))
            seg.ground = bool(spec.get("ground"))
            seg.fittings = None
            if spec.get("fittings_stay_with_body"):
                f = spec["fittings_stay_with_body"]
                seg.fittings = (float(f["end_before"]) * seg.L, need(rindex, expand(f["bone"], side), "Ronin bone"))
            segments.append(seg)
            for j, pinned in spec["joints"].items():
                j = expand(j, side)
                need(tpos, j, "Takemikazuchi joint")
                if pinned is not None:
                    pinned = expand(pinned, side)
                    need(rindex, pinned, "Ronin bone")
                plan[j] = (seg, pinned)
    # joints that stay with the rigid block but are weighted to a column of Ronin bones by height
    bands = {}
    for spec in bm.get("banded", []):
        for side in "LR":
            chain = [expand(x, side) for x in spec["bones"]]
            cuts = [float(need(rpos, expand(x, side), "Ronin bone")[2]) for x in spec["change_at_height_of"]]
            if len(cuts) != len(chain) - 1 or any(a <= b for a, b in zip(cuts, cuts[1:])):
                raise SystemExit(f"banded rule {spec['name']}: need {len(chain) - 1} descending change-over heights, got {cuts}")
            band = {"bones": [need(rindex, c, "Ronin bone") for c in chain], "cuts": cuts,
                    "blend": float(spec.get("blend_half_width", 8.0)), "name": f"{spec['name']}_{side}"}
            for j in spec["joints"]:
                j = expand(j, side)
                need(tpos, j, "Takemikazuchi joint")
                plan[j] = (block, None)
                bands[j] = band
    unmapped = []
    for j in tname:
        if j in plan:
            continue
        for pattern, bone in bm["block"]:
            if re.search(pattern, j):
                need(rindex, bone, "Ronin bone")
                plan[j] = (block, bone)
                break
        else:
            unmapped.append(j)
    if unmapped:
        raise SystemExit(f"bone_map.json does not cover these Takemikazuchi joints: {unmapped}")

    # ---- Ronin bone -> bodygroup
    def under(bone, top):
        while bone is not None:
            if bone == top:
                return True
            bone = rparent[bone]
        return False

    def group_of(bone):
        for g in ("left_arm", "right_arm", "left_leg", "right_leg"):
            if under(bone, bm["bodygroups"][g]):
                return g
        return "hip" if bone in bm["bodygroups"]["hip"] else "torso"

    groups = ["hip", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    bone_group = np.array([groups.index(group_of(b["name"])) for b in ron])

    # ---- place every vertex
    left_out = []
    for j in bm.get("left_out", {}).get("joints", []):
        need(tpos, j, "Takemikazuchi joint")
        left_out.append(tname.index(j))
    out = {}
    report = {"unit": unit, "scale_units_per_metre": round(s, 5), "size_rule": bm["size"]["rule"],
              "height_units": round(float(man["bounds_m"]["height_z"]) * s, 2), "joint_offsets_units": {}, "parts": {},
              "segments": {x.name: x.info for x in segments}}
    # how far the body's own joints are from the Ronin joints that move them
    for tj, rb in (("Shoulder_L", "def_l_shoulder"), ("Hip_L", "def_l_thigh"), ("Knee_L", "def_l_thighLow"),
                   ("Ankle_L", "def_l_ankle")):
        if tj in tpos and rb in rpos:
            report["joint_offsets_units"][f"{tj} vs {rb}"] = (block.place(tpos[tj], None)[0] - rpos[rb]).round(1).tolist()
    for part, info in man["lods"][str(lod)]["parts"].items():
        pos, nrm, uv = (z[f"{part}.{k}"].astype(np.float64) for k in ("pos", "nrm", "uv"))
        tri, mat = z[f"{part}.tri"].astype(np.int64), z[f"{part}.mat"]
        jnt, wgt = z[f"{part}.jnt"].astype(np.int64), z[f"{part}.wgt"].astype(np.float64)
        dom = jnt[np.arange(len(jnt)), np.argmax(wgt, axis=1)]     # each vertex follows its strongest joint
        if left_out:                                              # pieces the owner asked to leave off the body
            gone = np.isin(dom, left_out)[tri].any(axis=1)
            if gone.any():
                report.setdefault("left_out_triangles", {})[part] = int(gone.sum())
                tri, mat = tri[~gone], mat[~gone]
        new_p = np.zeros_like(pos)
        new_n = np.zeros_like(nrm)
        bone = np.zeros((len(pos), 2), dtype=np.int64)
        bw = np.zeros((len(pos), 2))
        piece = None
        for ji in np.unique(dom):
            placer, pinned = plan[tname[ji]]
            m = dom == ji
            if isinstance(placer, Segment) and placer.fittings and pinned is None:
                # fittings: connected pieces on this joint that end well short of the far joint stay with the
                # rigid body; only the limb itself is laid along Ronin's bones
                reach, fit_bone = placer.fittings
                if piece is None:
                    piece = connected_pieces(pos, tri)
                along = (pos - placer.o) @ placer.u
                stay = np.zeros(len(pos), dtype=bool)
                for pid in np.unique(piece[m]):
                    sel = m & (piece == pid)
                    if along[sel].max() <= reach:
                        stay |= sel
                if stay.any():
                    new_p[stay], new_n[stay] = block.place(pos[stay], nrm[stay])
                    bone[stay] = fit_bone
                    bw[stay] = [1.0, 0.0]
                    key = f"{placer.name} fittings kept with the body (vertices)"
                    report.setdefault("fittings", {})[key] = report.get("fittings", {}).get(key, 0) + int(stay.sum())
                    m = m & ~stay
                    if not m.any():
                        continue
            if isinstance(placer, Block) and tname[ji] in bands:
                band = bands[tname[ji]]
                p2, n2 = placer.place(pos[m], nrm[m])
                new_p[m], new_n[m] = p2, n2
                zc = p2[:, 2]
                # which band each vertex is in: 0 above the first change-over height, 1 below it, ...
                k = np.zeros(len(zc), dtype=np.int64)
                for c in band["cuts"]:
                    k += zc < c
                b2 = np.stack([k, k], axis=1)
                w2 = np.stack([np.ones(len(zc)), np.zeros(len(zc))], axis=1)
                for ci, c in enumerate(band["cuts"]):            # blend across each change-over
                    near = np.abs(zc - c) < band["blend"]
                    t = (c + band["blend"] - zc[near]) / (2 * band["blend"])    # 0 just above, 1 just below
                    b2[near] = [ci, ci + 1]
                    w2[near] = np.stack([1 - t, t], axis=1)
                bone[m] = np.array(band["bones"])[b2]
                bw[m] = w2
            elif isinstance(placer, Block):
                new_p[m], new_n[m] = placer.place(pos[m], nrm[m])
                bone[m] = rindex[pinned]
                bw[m] = [1.0, 0.0]
            else:
                p2, n2, ci, cw = placer.place(pos[m], nrm[m])
                new_p[m], new_n[m] = p2, n2
                if pinned is not None:
                    bone[m] = rindex[pinned]
                    bw[m] = [1.0, 0.0]
                else:
                    chain_idx = np.array([rindex[c] for c in placer.bones])
                    bone[m] = chain_idx[np.minimum(ci, len(placer.bones) - 2)]
                    bw[m] = cw
        new_n /= np.maximum(np.linalg.norm(new_n, axis=1, keepdims=True), 1e-12)
        out[part] = {"pos": new_p, "nrm": new_n, "uv": uv, "tri": tri, "mat": mat, "bone": bone, "bw": bw,
                     "dom": dom, "src": pos}
        report["parts"][part] = {"verts": int(len(pos)), "tris": int(len(tri))}

    # feet: stand them on the ground (Z = 0) after placement
    for seg in segments:
        if not getattr(seg, "ground", False):
            continue
        joints = {j for j, (pl, _b) in plan.items() if pl is seg}
        ids = [tname.index(j) for j in joints]
        low = min((o["pos"][np.isin(o["dom"], ids), 2].min() for o in out.values()
                   if np.isin(o["dom"], ids).any()), default=None)
        if low is not None:
            for o in out.values():
                o["pos"][np.isin(o["dom"], ids), 2] -= low
            seg.info["lifted_to_ground_units"] = round(float(-low), 2)

    # ---- the parts that must keep their place: compare each with a plain scaled copy of the model
    keep = bm.get("stay_in_place")
    if keep:
        limit = float(keep["max_shift_units"])
        shifts, moved = {}, []
        for ji, name in enumerate(tname):
            if not re.search(keep["joints"], name):
                continue
            new, old = [], []
            for o in out.values():
                m = o["dom"] == ji
                if m.any():
                    new.append(o["pos"][m])
                    old.append(block.place(o["src"][m], None)[0])
            if not new:
                continue
            d = np.vstack(new).mean(0) - np.vstack(old).mean(0)
            shifts[name] = d.round(2).tolist()
            if float(np.linalg.norm(d)) > limit:
                moved.append(f"{name} by {d.round(1).tolist()}")
        report["stay_in_place_shift_units"] = {k: v for k, v in shifts.items() if any(abs(x) > 0.05 for x in v)}
        if moved:
            raise SystemExit("the fit moves parts that must keep their sculpted place (forward, left, up; limit "
                             f"{limit:g} units):\n  " + "\n  ".join(moved))

    # ---- per-triangle bodygroup (majority of its corners' strongest bone) and a flat save
    arrays = {}
    counts = {g: 0 for g in groups}
    for part, o in out.items():
        strongest = o["bone"][np.arange(len(o["bone"])), np.argmax(o["bw"], axis=1)]
        g3 = bone_group[strongest][o["tri"]]
        tg = np.where(g3[:, 1] == g3[:, 2], g3[:, 1], g3[:, 0])
        for k, v in (("pos", o["pos"].astype(np.float32)), ("nrm", o["nrm"].astype(np.float32)),
                     ("uv", o["uv"].astype(np.float32)), ("tri", o["tri"].astype(np.int32)),
                     ("mat", o["mat"].astype(np.uint8)), ("bone", o["bone"].astype(np.int16)),
                     ("bw", o["bw"].astype(np.float32)), ("group", tg.astype(np.uint8))):
            arrays[f"{part}.{k}"] = v
        for gi, g in enumerate(groups):
            counts[g] += int((tg == gi).sum())
        report["parts"][part]["texset"] = man["lods"][str(lod)]["parts"][part]["texset"]
    allp = np.vstack([o["pos"] for o in out.values()])
    report["bodygroup_triangles"] = counts
    report["bounds_units"] = {"min": allp.min(0).round(2).tolist(), "max": allp.max(0).round(2).tolist()}
    report["groups"] = groups
    report["ronin_bones_used"] = sorted({ron[i]["name"] for o in out.values() for i in np.unique(o["bone"][o["bw"] > 0])})
    dst = OUT_DIR / (f"{unit}__{tag}" if tag else unit)
    dst.mkdir(parents=True, exist_ok=True)
    suffix = "" if lod == 0 else f"_lod{lod}"
    report["lod"] = lod
    np.savez_compressed(dst / f"fit{suffix}.npz", **arrays)
    (dst / f"report{suffix}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if lod:                                                   # the lower levels: one line each
        print(f"{unit} level of detail {lod}: {sum(len(o['tri']) for o in out.values()):,} triangles, "
              f"{sum(len(o['pos']) for o in out.values()):,} vertices; per bodygroup {counts}"
              + (f"; fittings kept with the body {report['fittings']}" if report.get("fittings") else ""))
        return report

    print(f"{unit} on Ronin's skeleton: {s:.4f} units/m ({bm['size']['rule']}), "
          f"{sum(len(o['pos']) for o in out.values()):,} vertices")
    print("  its own joints relative to the Ronin joints that move them (forward, left, up):",
          "; ".join(f"{k} {v}" for k, v in report["joint_offsets_units"].items()))
    for k, v in report.get("fittings", {}).items():
        print(f"  {k}: {v}")
    if keep:
        sh = report["stay_in_place_shift_units"]
        print(f"  parts that must keep their sculpted place: all within {keep['max_shift_units']:g} units"
              + (f"; moved at all: {sh}" if sh else "; none moved"))
    b = report["bounds_units"]
    print(f"  bounds (forward, left, up): {b['min']} .. {b['max']}   height {b['max'][2] - b['min'][2]:.1f} units "
          f"(stock Ronin mesh 234.2)")
    print("  limb sections (length in metres -> Ronin chain in units, stretch along the axis vs plain scale):")
    for seg in segments:
        i = seg.info
        print(f"    {seg.name:12s} {i['tsf_length_m']:5.2f} m -> {i['ronin_chain_units']:6.1f} u   x{i['axial_stretch_vs_plain_scale']:.2f}"
              + (f"   lifted {i['lifted_to_ground_units']} u to the ground" if "lifted_to_ground_units" in i else ""))
    for b in {id(v): v for v in bands.values()}.values():
        print(f"    {b['name']:12s} kept straight with the body; weighted by height to "
              f"{[ron[i]['name'] for i in b['bones']]}, changing over at z = {[round(c, 1) for c in b['cuts']]} "
              f"(+-{b['blend']:g})")
    print("  triangles per Ronin bodygroup:", counts)
    print(f"  Ronin bones carrying vertices: {len(report['ronin_bones_used'])}")
    print(f"  -> {dst}")
    return report


if __name__ == "__main__":
    import sys
    retarget_all(sys.argv[1] if len(sys.argv) > 1 else None)
