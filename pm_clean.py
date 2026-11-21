"""Stage 20 - build/pm/<unit>/  ->  build/clean/<unit>/

Turns the raw toolkit export into a game-neutral, cleaned model that every
later stage reads.  Nothing in here is specific to Titanfall.

    lod<N>.npz        per part:  <part>.pos  (V,3) float32   metres
                                 <part>.nrm  (V,3) float32   unit length, never zero
                                 <part>.uv   (V,2) float32   V up (Blender / SMD convention)
                                 <part>.tri  (T,3) int32
                                 <part>.mat  (T,)  uint8     0 = the part's texture set, 1 = glow
                                 <part>.jnt  (V,4) int16     indices into skeleton.json "joints"
                                 <part>.wgt  (V,4) float32   rows sum to 1
    skeleton.json     the joints the skin uses: name, parent, bind position + 3x3 rotation
    textures/         <set>_bc.png RGB, <set>_nm.png RGB (Z rebuilt), <set>_mt/_rh/_ao.png L,
                      <set>_cm.png colour-region mask, <set>_em.png baked emissive
    manifest.json     parts, texture sets, bounds, counts, everything that was repaired, and
                      "glow": the unit's own lit-panel colour (triangles with mat == 1)

Frame: +X forward, +Y left, +Z up (Source engine and Blender agree on this), feet at
Z = 0, centred on X/Y.  Size is the game's own (metres); scaling to a titan is a
later, explicit step.

    python -m tsfronin.pm_clean [unit]
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image

from . import config

DEGENERATE_AREA = 1e-10          # m^2; PM ships zero-area triangles carrying zero-length normals
ROLE_FILE = {"baseColor": "bc", "normal": "nm", "metalness": "mt", "roughness": "rh",
             "occlusion": "ao", "userColorMask": "cm", "emissive": "em"}


def yup_to_zup(a):
    """PM is Y-up: (x, y, z) -> (x, -z, y)"""
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)


def node_world(n):
    """(position, 3x3 rotation) of a node, PM frame.  The toolkit stores the matrix row-major;
    accept either translation layout rather than assume one."""
    m = np.asarray(n["worldMatrix"], dtype=np.float64).reshape(4, 4)
    if abs(m[3, 3] - 1) < 1e-6 and np.allclose(m[:3, 3], 0):
        return m[3, :3].copy(), m[:3, :3].T.copy()          # row-vector convention
    return m[:3, 3].copy(), m[:3, :3].copy()                # column-vector convention


def mesh_triangles(data_idx, mesh, index_end=None):
    """Triangle list (indices local to the mesh) + per-triangle glow flag.
    Same slicing rule as the toolkit's blender_rig.py, with one repair: the toolkit leaves some draw blocks
    without a count (the leg's lit panels at levels of detail 1 and 2 of the Type-00).  Such a block runs from
    its own offset to the next block's, or to the end of the mesh's indices (index_end: where the next mesh's
    begin); it is taken only if that span is whole triangles that all stay inside this mesh's vertices."""
    v0, vn = mesh["vertexStart"], mesh["vertexCount"]
    local = mesh.get("localIndices", False)
    i0 = mesh.get("indexStart", 0) if local else v0
    shift = 0 if local else v0
    blocks = mesh["drawblocks"]
    counts = []
    mesh["_recovered_blocks"] = 0
    for i, b in enumerate(blocks):
        n = b.get("indexCount") or 0
        if not n and local and index_end is not None and b.get("indexOffset") is not None:
            nxt = next((x["indexOffset"] for x in blocks[i + 1:] if x.get("indexOffset") is not None), index_end)
            gap = nxt - b["indexOffset"]
            if gap > 0 and gap % 3 == 0:
                span = data_idx[b["indexOffset"]:nxt].astype(np.int64)
                if len(span) == gap and span.min() >= 0 and span.max() < vn:
                    n = gap
                    mesh["_recovered_blocks"] += 1
        counts.append(n // 3)
    ntri = sum(counts)
    tri = data_idx[i0:i0 + ntri * 3].astype(np.int64).reshape(-1, 3) - shift
    flags = mesh.get("drawblockGlow") or []
    glow = np.concatenate([np.full(c, bool(flags[i]) if i < len(flags) else False)
                           for i, c in enumerate(counts)]) if counts else np.zeros(0, bool)
    ok = ((tri >= 0) & (tri < vn)).all(axis=1) & (tri[:, 0] != tri[:, 1]) \
        & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return tri[ok], glow[:len(tri)][ok], int((~ok).sum())


def clean_mesh(pos, nrm, uv, jnt, wgt, tri, glow):
    """Drop degenerate and duplicate triangles, re-index, and repair zero-length normals."""
    rep = {}
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    fn = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    keep = area > DEGENERATE_AREA
    rep["degenerate_tris"] = int((~keep).sum())
    tri, glow, fn = tri[keep], glow[keep], fn[keep]
    _, first = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
    rep["duplicate_tris"] = int(len(tri) - len(first))
    first.sort()
    tri, glow, fn = tri[first], glow[first], fn[first]
    used = np.unique(tri)
    rep["unreferenced_verts"] = int(len(pos) - len(used))
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pos, nrm, uv, jnt, wgt = pos[used], nrm[used], uv[used], jnt[used], wgt[used]
    tri = remap[tri]
    # zero-length normals -> area-weighted face normal of the triangles that use the vertex
    ln = np.linalg.norm(nrm, axis=1)
    bad = ln < 1e-6
    rep["zero_normals"] = int(bad.sum())
    if bad.any():
        acc = np.zeros_like(pos)
        for k in range(3):
            np.add.at(acc, tri[:, k], fn)
        al = np.linalg.norm(acc, axis=1)
        acc[al < 1e-12] = (0.0, 0.0, 1.0)
        nrm = np.where(bad[:, None], acc, nrm)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    # winding must agree with the shading normals (the toolkit's own oracle)
    vnf = nrm[tri].sum(axis=1)
    rep["winding_agreement"] = round(float((np.einsum("ij,ij->i", fn, vnf) > 0).mean()), 4)
    return pos, nrm, uv, jnt, wgt, tri, glow, rep


def facing_angle(nodes):
    """Angle (about Z, in the Z-up frame) of the model's forward direction, from toes vs ankles."""
    by = {n["name"]: n for n in nodes}
    for toe, ankle in (("Toes_L", "Ankle_L"), ("Toes_R", "Ankle_R")):
        if toe in by and ankle in by:
            d = yup_to_zup(node_world(by[toe])[0]) - yup_to_zup(node_world(by[ankle])[0])
            d[2] = 0
            if np.linalg.norm(d) > 1e-6:
                return math.atan2(d[1], d[0])
    return None            # not a mech (a weapon, a prop): no facing to establish


def normalise_textures(data, src_dir, out_dir, used_sets, report):
    """Copy each used texture set with fixed names and channel layouts."""
    os.makedirs(out_dir, exist_ok=True)
    sets = {}
    for t in data["materials"]:
        short = ROLE_FILE.get(t["role"])
        if short is None or t["part"] not in used_sets:
            continue
        src = os.path.join(src_dir, t["png"])
        if not os.path.exists(src):
            continue
        im = Image.open(src)
        if short == "nm":
            # BC5 carries X and Y only; rebuild Z or the surface shades black
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            x, y = a[..., 0] * 2 - 1, a[..., 1] * 2 - 1
            a[..., 2] = np.sqrt(np.clip(1 - x * x - y * y, 0, 1)) * 0.5 + 0.5
            im = Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
        elif short in ("mt", "rh", "ao"):
            im = im.convert("L")
        else:
            im = im.convert("RGB")
        fn = f"{t['part']}_{short}.png"
        im.save(os.path.join(out_dir, fn))
        sets.setdefault(t["part"], {})[short] = {"file": f"textures/{fn}", "size": list(im.size)}
    for s in sorted(used_sets):
        missing = [r for r in ("bc", "nm", "mt", "rh", "ao") if r not in sets.get(s, {})]
        if missing:
            report.setdefault("texture_sets_missing_maps", {})[s] = missing
    return sets


def clean(unit):
    src_dir = config.PM_DIR / unit
    out_dir = config.CLEAN_DIR / unit
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(src_dir / f"{unit}.json"))
    idx = np.load(src_dir / f"{unit}.indices.npy")
    report = {}

    # ---- frame: Y-up -> Z-up, then turn to face +X
    # A mech is turned to face +X and stood on Z=0.  Anything else (a weapon) keeps the game's own
    # origin and orientation after the Y-up -> Z-up change, because its origin is its grip point.
    ang = facing_angle(data["nodes"])
    is_mech = ang is not None
    ang = ang or 0.0
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])        # rotation about Z by -ang

    def to_frame(v):                                         # points / directions, (N,3)
        return yup_to_zup(v) @ R.T

    A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # the Y-up -> Z-up matrix

    # ---- skeleton: only the joints some mesh is skinned to, parented to the nearest such ancestor
    nodes = data["nodes"]
    by_name = {n["name"]: n for n in nodes}
    joint_names = []
    for m in data["meshes"]:
        for nm in m["jointNames"]:
            if nm and nm not in joint_names and nm in by_name:
                joint_names.append(nm)
    jset = set(joint_names)

    def joint_parent(n):
        p, hops = n.get("parentIndex"), 0
        while p is not None and 0 <= p < len(nodes) and hops < 128:
            if nodes[p]["name"] in jset:
                return nodes[p]["name"]
            p, hops = nodes[p].get("parentIndex"), hops + 1
        return None

    # parents before children
    order, seen = [], set()

    def visit(nm):
        if nm in seen:
            return
        seen.add(nm)
        p = joint_parent(by_name[nm])
        if p:
            visit(p)
        order.append(nm)

    for nm in joint_names:
        visit(nm)
    jindex = {nm: i for i, nm in enumerate(order)}

    lods = sorted({(m.get("lod") or 0) for m in data["meshes"]})
    lod0 = [m for m in data["meshes"] if (m.get("lod") or 0) == 0]
    P0 = np.vstack([to_frame(np.asarray(m["positions"], dtype=np.float64)) for m in lod0])
    lo, hi = P0.min(0), P0.max(0)
    offset = (np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])  # feet on Z=0, centred
              if is_mech else np.zeros(3))

    joints = []
    for nm in order:
        p, rot = node_world(by_name[nm])
        p = to_frame(p[None, :] * data.get("scale", 1.0))[0] + offset
        rot = R @ A @ rot @ A.T                               # same change of basis as the points
        par = joint_parent(by_name[nm])
        joints.append({"name": nm, "parent": jindex[par] if par else -1,
                       "pos": [round(float(x), 6) for x in p],
                       "rot": [[round(float(x), 6) for x in row] for row in rot]})
    left = next((j for j in joints if j["name"] == "Shoulder_L"), None)
    if left is not None and left["pos"][1] <= 0:
        raise SystemExit("frame check failed: Shoulder_L is not on +Y (model is mirrored)")
    # named nodes that are not skin joints (a weapon's grip and blade markers, for instance)
    markers = []
    for n in nodes:
        if n["name"] in jset or n["name"] in ("RootNode",) or n["name"].startswith("lod_"):
            continue
        if any(x["name"] == n["name"] for x in markers) or n["name"] in ("body", "body_shadow"):
            continue
        p, rot = node_world(n)
        rot = R @ A @ rot @ A.T
        markers.append({"name": n["name"],
                        "pos": [round(float(x), 6) for x in to_frame(p[None, :] * data.get("scale", 1.0))[0] + offset],
                        "rot": [[round(float(x), 6) for x in row] for row in rot]})
    # a mech's own attach points for stowed weapons (blade and gun holders on the back): position, orientation,
    # and the skin joint they ride.  The game lays a weapon's origin onto one of these.
    mounts = []
    if is_mech:
        for n in nodes:
            if n["name"].endswith("_mount_L") or n["name"].endswith("_mount_R"):
                p, rot = node_world(n)
                rot = R @ A @ rot @ A.T
                mounts.append({"name": n["name"], "rides": joint_parent(n),
                               "pos": [round(float(x), 6) for x in to_frame(p[None, :] * data.get("scale", 1.0))[0] + offset],
                               "rot": [[round(float(x), 6) for x in row] for row in rot]})
    (out_dir / "skeleton.json").write_text(json.dumps({"joints": joints, "markers": markers if not is_mech else [],
                                                       "mounts": mounts}, indent=1), encoding="utf-8")

    # ---- meshes
    texture_sets = sorted({t["part"] for t in data["materials"] if t["role"] == "baseColor"})

    def texset_of(m, part):
        ts = m.get("texset") or part
        if ts in texture_sets:
            return ts
        if len(texture_sets) == 1:            # a weapon has one atlas whatever the mesh is called
            return texture_sets[0]
        raise SystemExit(f"{unit}: mesh {m['index']} ({part}) names texture set {ts!r}, "
                         f"but the unit has {texture_sets}")

    manifest_lods = {}
    used_sets = set()
    skipped = []
    index_starts = sorted({x.get("indexStart", 0) for x in data["meshes"] if x.get("localIndices")})
    for lod in lods:
        arrays, parts = {}, {}
        lod_meshes = [x for x in data["meshes"] if (x.get("lod") or 0) == lod]
        if not is_mech:
            # the toolkit labels every weapon mesh "chest"; a weapon LOD is one "body" mesh, plus
            # (on the low LODs) a simplified shadow proxy that is not part of the visible model
            lod_meshes.sort(key=lambda x: -x["vertexCount"])
            skipped += [{"lod": lod, "mesh": x["index"], "verts": x["vertexCount"], "why": "shadow proxy"}
                        for x in lod_meshes[1:]]
            lod_meshes = lod_meshes[:1]
        for m in lod_meshes:
            part = m["part"] if is_mech else "body"
            pos = to_frame(np.asarray(m["positions"], dtype=np.float64)) + offset
            nrm = to_frame(np.asarray(m["normals"], dtype=np.float64))
            uv = np.asarray(m["uv0"], dtype=np.float64).copy()
            uv[:, 1] = 1.0 - uv[:, 1]                          # PM is V-down (glTF); we store V-up
            w = np.asarray(m["weights"], dtype=np.float64)
            j = np.asarray(m["joints"], dtype=np.int64)
            names = m["jointNames"]
            lut = np.array([jindex.get(nm, -1) for nm in names] + [-1], dtype=np.int64)
            j = np.where((j >= 0) & (j < len(names)), j, len(names))
            gj = lut[j]
            w = np.where(gj >= 0, w, 0.0)
            tot = w.sum(axis=1, keepdims=True)
            unbound = int((tot[:, 0] < 1e-6).sum())
            w = w / np.maximum(tot, 1e-9)
            gj = np.where(w > 0, gj, 0)
            later = [s for s in index_starts if s > m.get("indexStart", 0)]
            tri, glow, dropped = mesh_triangles(idx, m, later[0] if later else len(idx))
            pos, nrm, uv, gj, w, tri, glow, rep = clean_mesh(pos, nrm, uv, gj, w, tri, glow)
            rep["out_of_range_tris"] = dropped
            if m.get("_recovered_blocks"):
                rep["draw_blocks_recovered"] = m["_recovered_blocks"]
                report.setdefault("draw_blocks_without_a_count_recovered", []).append(
                    {"lod": lod, "part": part, "blocks": m["_recovered_blocks"]})
            rep["unbound_verts"] = unbound
            arrays[f"{part}.pos"] = pos.astype(np.float32)
            arrays[f"{part}.nrm"] = nrm.astype(np.float32)
            arrays[f"{part}.uv"] = uv.astype(np.float32)
            arrays[f"{part}.tri"] = tri.astype(np.int32)
            arrays[f"{part}.mat"] = glow.astype(np.uint8)
            arrays[f"{part}.jnt"] = gj.astype(np.int16)
            arrays[f"{part}.wgt"] = w.astype(np.float32)
            texset = texset_of(m, part)
            used_sets.add(texset)
            parts[part] = {"texset": texset, "verts": int(len(pos)), "tris": int(len(tri)),
                           "glow_tris": int(glow.sum()),
                           "joints": sorted({order[i] for i in np.unique(gj[w > 0])}),
                           "repairs": rep}
        np.savez_compressed(out_dir / f"lod{lod}.npz", **arrays)
        manifest_lods[str(lod)] = {"file": f"lod{lod}.npz", "parts": parts,
                                   "verts": sum(p["verts"] for p in parts.values()),
                                   "tris": sum(p["tris"] for p in parts.values())}

    sets = normalise_textures(data, str(src_dir), str(out_dir / "textures"), used_sets, report)

    # ---- lit panels: the unit's own glow material, never a default colour
    glow = None
    mats_file = src_dir / "materials.json"
    materials = json.load(open(mats_file))["materials"] if mats_file.exists() else []
    g = next((m for m in materials if m.get("shaderType") == "ShaderType_COLOR_ONLY"), None)
    if g is not None:
        glow = {"emissive_linear": g["emissiveColor_linear"],
                "albedo_srgb8": (g.get("albedoColor_rgba8") or [0, 0, 0])[:3],
                "bloomScale": g.get("bloomScale"), "shaderType": g.get("shaderType"),
                "source": f"{g['name']} in the unit's own scene file"}
    if skipped:
        report["skipped_meshes"] = skipped
    n_glow = sum(p["glow_tris"] for p in manifest_lods["0"]["parts"].values())
    if n_glow and glow is None:
        raise SystemExit(f"{unit}: {n_glow} triangles use the glow material but no glow_mat was extracted "
                         f"(re-run the extract stage)")
    size = hi - lo
    manifest = {
        "unit": unit, "source": data.get("source"),
        "frame": ("+X forward, +Y left, +Z up; metres; feet at Z=0" if is_mech else
                  "the game's own origin and orientation, Y-up converted to Z-up (x, y, z) -> (x, -z, y); metres"),
        "bounds_m": {"length_x": round(float(size[0]), 4), "width_y": round(float(size[1]), 4),
                     "height_z": round(float(size[2]), 4)},
        "facing_rotation_deg": round(math.degrees(-ang), 3),
        "kind": "mech" if is_mech else "weapon",
        "joints": len(joints), "lods": manifest_lods, "texture_sets": sets, "glow": glow,
        "materials": [m for m in materials if m.get("shaderType") != "ShaderType_COLOR_ONLY"],
        "texture_sources": data.get("textureSources", []), "report": report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    l0 = manifest_lods["0"]
    print(f"{unit}: {len(lods)} LODs, {len(joints)} joints, "
          f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m (L x W x H)")
    for lod in lods:
        L = manifest_lods[str(lod)]
        print(f"  LOD{lod}: {len(L['parts'])} parts  {L['verts']:>7,} verts  {L['tris']:>7,} tris")
    print("  LOD0 parts:")
    for part, p in l0["parts"].items():
        r = p["repairs"]
        print(f"    {part:10s} set={p['texset']:7s} {p['verts']:>6,} v {p['tris']:>6,} t  glow {p['glow_tris']:>4}  "
              f"degenerate {r['degenerate_tris']:>3}  dup {r['duplicate_tris']:>3}  zero-nrm {r['zero_normals']:>3}  "
              f"unbound {r['unbound_verts']}  winding {r['winding_agreement']:.1%}")
    if glow:
        e = glow["emissive_linear"]
        srgb = [round(255 * (12.92 * x if x <= 0.0031308 else 1.055 * min(x, 1.0) ** (1 / 2.4) - 0.055)) for x in e]
        print(f"  glow panels: {n_glow} triangles, emissive {e} linear = sRGB {srgb}")
    if report:
        print("  report:", json.dumps(report))
    print(f"  -> {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("unit", nargs="?", default=config.UNIT)
    clean(ap.parse_args().unit)


if __name__ == "__main__":
    main()
