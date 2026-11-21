"""Titanfall 2's AI navigation meshes (Northstar.CustomServers/mod/maps/navmesh/<map>_<hull>.nm), read only, and a
survey of what they mean for the BETA Tank (TSFRonin.BETA) on every stock map (owner 2026-11-17: "not all the maps
are the same ... the Tank AI might be all messed up when on different maps.  Review all the main maps").

Format (research 2026-11-17, build/research/beta/... r_navigation2.txt; checked there on all four mp_rise hulls:
every tile's dataSize matched the layout below, 0 mismatches over 1664 tiles):
  file header 52 bytes: 'MSET', u32 version 5, numTiles, f32 orig[3], f32 tileW, tileH, u32 maxTiles, maxPolys,
    polyGroupCount, traverseTableSize, traverseTableCount; then per tile {u32 tileRef, u32 dataSize} + data
  tile data: 'DNAV' header of 104 bytes (polyCount @24, vertCount @32, maxLinkCount @36, offMeshConCount @56,
    walkableHeight/Radius/Climb f32 @64/68/72), then verts (3 x f32), polys (48 bytes), links (16 bytes), ...
  poly: u32 firstLink, u16 verts[6], u16 neis[6], u16 flags, u8 vertCount, u8 areaAndType, u16 groupId,
    u16 surfaceArea, f32 center[3]
  link: u32 ref, u32 next, u8 edge, side, bmin, bmax, u8 traverseType (0xFF: plain walk), u8 traverseDist, u16 rev
  poly ref = salt << 22 | tile << log2(maxPolys) | poly.  z is up (game axes).
The Tank walks the Prowler's mesh, med_short (HULL_PROWLER); small is the pilot-sized one (where people stand).
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

from . import config

NAVMESH = config.TITANFALL2 / "R2Northstar" / "mods" / "Northstar.CustomServers" / "mod" / "maps" / "navmesh"
OUT = config.BUILD / "research" / "beta" / "maps"
STOCK = ["mp_angel_city", "mp_black_water_canal", "mp_coliseum", "mp_colony02", "mp_complex3", "mp_crashsite3",
         "mp_drydock", "mp_eden", "mp_forwardbase_kodai", "mp_glitch", "mp_grave", "mp_homestead", "mp_lf_deck",
         "mp_lf_meadow", "mp_lf_stacks", "mp_lf_township", "mp_lf_traffic", "mp_lf_uma", "mp_relic02", "mp_rise",
         "mp_thaw", "mp_wargames"]
NULL_LINK = 0xFFFFFFFF


class NavMesh:
    """polygons (centre, outline, flags, group, area) and their links (neighbour, traverse type) of one .nm"""

    def __init__(self, path):
        d = Path(path).read_bytes()
        if d[:4] not in (b"TESM", b"MSET"):                           # 'MSET' stored as a little-endian u32
            raise SystemExit(f"{path}: not an MSET navmesh")
        (self.version, n_tiles) = struct.unpack_from("<II", d, 4)
        self.max_polys = struct.unpack_from("<I", d, 36)[0]
        self.group_count = struct.unpack_from("<I", d, 40)[0]
        self.poly_bits = max(1, (self.max_polys - 1).bit_length())  # Detour: ilog2(nextPow2(maxPolys))
        o = 52
        centers, outlines, flags, groups, areas, links = [], [], [], [], [], []
        self.tile_index = {}                                          # tile number -> (first poly index, count)
        self.walk = None
        for _t in range(n_tiles):
            tile_ref, size = struct.unpack_from("<II", d, o)
            base = o + 8
            o = base + size
            if size < 104 or d[base:base + 4] not in (b"VAND", b"DNAV"):
                continue
            n_poly = struct.unpack_from("<I", d, base + 24)[0]
            n_vert = struct.unpack_from("<I", d, base + 32)[0]
            n_link = struct.unpack_from("<I", d, base + 36)[0]
            if self.walk is None:
                self.walk = struct.unpack_from("<fff", d, base + 64)
            tile_no = (tile_ref >> self.poly_bits) & ((1 << (22 - self.poly_bits)) - 1)
            vo = base + 104
            verts = np.frombuffer(d, dtype="<f4", count=3 * n_vert, offset=vo).reshape(-1, 3)
            po = vo + 12 * n_vert
            lo = po + 48 * n_poly
            first = len(centers)
            self.tile_index[tile_no] = (first, n_poly)
            tile_links = [struct.unpack_from("<IIBBBBBBH", d, lo + 16 * i) for i in range(n_link)]
            for p in range(n_poly):
                q = po + 48 * p
                first_link = struct.unpack_from("<I", d, q)[0]
                vidx = struct.unpack_from("<6H", d, q + 4)
                fl, vc, area_type, group, surf = struct.unpack_from("<HBBHH", d, q + 28)
                cx, cy, cz = struct.unpack_from("<fff", d, q + 36)
                vc = min(vc, 6)
                outlines.append(verts[list(vidx[:vc])].copy() if vc else np.zeros((0, 3), dtype="<f4"))
                centers.append((cx, cy, cz))
                flags.append(fl)
                groups.append(group)
                areas.append(surf)
                out = []
                li, guard = first_link, 0
                while li != NULL_LINK and li < len(tile_links) and guard < 64:
                    ref, nxt, edge, side, bmin, bmax, ttype, tdist, rev = tile_links[li]
                    out.append((ref, ttype))
                    li, guard = nxt, guard + 1
                links.append(out)
        self.center = np.array(centers, dtype=np.float64).reshape(-1, 3)
        self.outline = outlines
        self.flags = np.array(flags, dtype=np.int64)
        self.group = np.array(groups, dtype=np.int64)
        self.links = links
        self.walkable = self.flags != 0                               # research: disabled polys have flags 0 (group 1)
        self.area = np.array([_area(p) for p in outlines])
        self.slope = np.array([_slope(p) for p in outlines])         # degrees from level

    def resolve(self, ref):
        """a link's neighbour as a poly index of this mesh, or None"""
        poly = ref & ((1 << self.poly_bits) - 1)
        tile = (ref >> self.poly_bits) & ((1 << (22 - self.poly_bits)) - 1)
        t = self.tile_index.get(tile)
        if t is None or poly >= t[1]:
            return None
        return t[0] + poly


def _area(p):
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _slope(p):
    """the polygon's tilt from level, degrees (Newell's normal of its outline)"""
    if len(p) < 3:
        return 0.0
    q = p.astype(np.float64)
    r = np.roll(q, -1, axis=0)
    n = np.array([np.sum((q[:, 1] - r[:, 1]) * (q[:, 2] + r[:, 2])),
                  np.sum((q[:, 2] - r[:, 2]) * (q[:, 0] + r[:, 0])),
                  np.sum((q[:, 0] - r[:, 0]) * (q[:, 1] + r[:, 1]))])
    ln = float(np.linalg.norm(n))
    if ln < 1e-6:
        return 0.0
    return math.degrees(math.acos(min(1.0, abs(n[2]) / ln)))


def _inside(px, py, p):
    """point in the polygon's outline (x, y)"""
    n = len(p)
    if n < 3:
        return False
    x, y = p[:, 0], p[:, 1]
    c = False
    j = n - 1
    for i in range(n):
        if (y[i] > py) != (y[j] > py):
            xi = x[j] + (py - y[j]) * (x[i] - x[j]) / (y[i] - y[j])
            if px < xi:
                c = not c
        j = i
    return c


class Grid:
    """polygons by 256-unit cells of their outline's box, for 'what Prowler polygon is under this point'"""

    def __init__(self, mesh, mask, cell=256.0):
        self.mesh, self.cell, self.cells = mesh, cell, {}
        for i in np.nonzero(mask)[0]:
            p = mesh.outline[i]
            if len(p) < 3:
                continue
            x0, y0 = np.floor(p[:, :2].min(0) / cell).astype(int)
            x1, y1 = np.floor(p[:, :2].max(0) / cell).astype(int)
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    self.cells.setdefault((gx, gy), []).append(i)

    def floors(self, x, y):
        """(poly, z at the polygon's centre) of every polygon whose outline contains (x, y)"""
        out = []
        for i in self.cells.get((int(math.floor(x / self.cell)), int(math.floor(y / self.cell))), ()):
            p = self.mesh.outline[i]
            if _inside(x, y, p):
                out.append((i, float(p[:, 2].mean())))
        return out


def survey(name):
    """one map: the Tank's mesh (med_short) against where pilots can stand (small)"""
    tank = NavMesh(NAVMESH / f"{name}_med_short.nm")
    human = NavMesh(NAVMESH / f"{name}_small.nm")
    r = {"map": name}
    w = tank.walkable
    r["tank_polys"] = int(len(tank.center))
    r["tank_walkable_polys"] = int(w.sum())
    r["tank_walkable_area_m2"] = round(float(tank.area[w].sum()) / 39.37 ** 2)
    r["human_walkable_area_m2"] = round(float(human.area[human.walkable].sum()) / 39.37 ** 2)
    r["coverage_of_human_area"] = round(float(tank.area[w].sum()) / max(1.0, float(human.area[human.walkable].sum())), 3)
    # the ground the Tank walks on: how much of it is tilted, and how far (its walk is made on level ground)
    wa = tank.area[w]
    sl = tank.slope[w]
    tot = float(wa.sum()) or 1.0
    r["tank_ground_slope_share"] = {
        band: round(float(wa[(sl >= lo) & (sl < hi)].sum()) / tot, 3)
        for band, lo, hi in (("0-5", 0, 5), ("5-15", 5, 15), ("15-25", 15, 25), ("25-35", 25, 35), ("35+", 35, 91))}
    r["tank_ground_steepest_deg"] = round(float(sl.max()), 1) if len(sl) else 0.0
    r["disabled_polys_group1"] = int(((tank.flags == 0) & (tank.group == 1)).sum())
    r["disabled_polys_other"] = int(((tank.flags == 0) & (tank.group != 1)).sum())
    zs = tank.center[w, 2]
    r["tank_z_range"] = [round(float(zs.min())), round(float(zs.max()))] if len(zs) else None
    # connected pieces of the Tank's mesh (groups), by walkable area
    areas = {}
    for g, a in zip(tank.group[w], tank.area[w]):
        areas[int(g)] = areas.get(int(g), 0.0) + float(a)
    total = sum(areas.values()) or 1.0
    ordered = sorted(areas.values(), reverse=True)
    r["tank_groups"] = len(areas)
    r["largest_group_share"] = round(ordered[0] / total, 3) if ordered else 0.0
    r["islands_over_5pct"] = sum(1 for a in ordered[1:] if a / total >= 0.05)
    r["islands_small"] = sum(1 for a in ordered[1:] if a / total < 0.05)
    # climbs: traverse links by type and by how far up they go (centre to centre)
    types, rises = {}, []
    for i, out in enumerate(tank.links):
        if not w[i]:
            continue
        for ref, ttype in out:
            if ttype == 0xFF:
                continue
            j = tank.resolve(ref)
            if j is None:
                continue
            types[int(ttype)] = types.get(int(ttype), 0) + 1
            dz = float(tank.center[j, 2] - tank.center[i, 2])
            if dz > 0:
                rises.append(dz)
    r["traverse_links_by_type"] = dict(sorted(types.items()))
    rises = np.array(rises)
    r["climbs_up"] = {"64-191": int(((rises >= 64) & (rises < 192)).sum()), "192-447": int(((rises >= 192) & (rises < 448)).sum()),
                      "448+": int((rises >= 448).sum())}
    # perches: places a pilot can stand (small mesh polygons) with no Tank polygon within 48 units of height under
    # them; how high each is above the Tank's floor below it (the nearest Tank polygon under it within 600 below)
    tg = Grid(tank, w)
    perch_area, heights = 0.0, []
    no_floor_area = 0.0
    hw = np.nonzero(human.walkable)[0]
    for i in hw:
        x, y, z = human.center[i]
        floors = tg.floors(x, y)
        if any(abs(fz - z) <= 48.0 for _j, fz in floors):
            continue
        a = float(human.area[i])
        perch_area += a
        below = [z - fz for _j, fz in floors if 48.0 < z - fz <= 2000.0]
        if below:
            heights.append((min(below), a))
        else:
            no_floor_area += a
    human_total = float(human.area[hw].sum()) or 1.0
    r["perch_share_of_human_area"] = round(perch_area / human_total, 3)
    bands = {"48-159 (its jump-up bite)": 0.0, "160-699 (script climbs)": 0.0, "700+ (too high)": 0.0,
             "no Tank floor under it": no_floor_area}
    for h, a in heights:
        if h < 160:
            bands["48-159 (its jump-up bite)"] += a
        elif h < 700:
            bands["160-699 (script climbs)"] += a
        else:
            bands["700+ (too high)"] += a
    r["perch_area_by_height_share"] = {k: round(v / human_total, 3) for k, v in bands.items()}
    r["walkable_height_radius_climb"] = [round(float(x), 1) for x in (tank.walk or (0, 0, 0))]
    return r


COLOURS = {"jump": (60, 190, 60), "climb": (235, 200, 40), "too_high": (220, 40, 40), "leap": (240, 130, 30),
           "isolated": (150, 60, 200), "interior": (40, 120, 230)}


def places(name, min_area_m2=3.0, picture=True):
    """Concrete places on one map where pilots can stand and the Tank has no floor (0.4.2; owner 2026-11-18: "look at
    all the maps and see which places would cause the Tank class to be unable to reach ... where its high the tank
    jump might not be able to reach it so they need to climb").  Pilot-mesh polygons with no Tank polygon within 48
    units under them, joined into connected patches (walk links); for each patch the Tank floor nearest it at its own
    level (a room or passage too narrow for the Tank) and below it (a ledge, roof, rock), how far across the ground and
    how high.  Kinds: interior (Tank floor at its level beside it, but none on it), jump (a ledge up to 160 over Tank
    floor within 250: the AI's own jump-up bite), climb (160-700 over Tank floor within 250: the script's wall climb),
    too_high (over 700: beyond BETA_CLIMB_MAX_RISE), leap (the nearest lower Tank floor 250-900 away: only a leap
    across), isolated (none within 900).  -> dict; optional top-down picture build/research/beta/maps/<map>_places.png"""
    tank = NavMesh(NAVMESH / f"{name}_med_short.nm")
    human = NavMesh(NAVMESH / f"{name}_small.nm")
    w = tank.walkable
    tg = Grid(tank, w)
    hw = np.nonzero(human.walkable)[0]
    perch = set()
    for i in hw:
        x, y, z = human.center[i]
        if not any(abs(fz - z) <= 48.0 for _j, fz in tg.floors(x, y)):
            perch.add(int(i))
    # connected patches (union-find over walk links between perch polygons)
    parent = {i: i for i in perch}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in perch:
        for ref, ttype in human.links[i]:
            if ttype != 0xFF:
                continue
            j = human.resolve(ref)
            if j is not None and j in perch:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in perch:
        groups.setdefault(find(i), []).append(i)
    # Tank floor points: every walkable Tank polygon's corners and centre, with its group (the largest = main)
    areas = {}
    for g, a in zip(tank.group[w], tank.area[w]):
        areas[int(g)] = areas.get(int(g), 0.0) + float(a)
    main_group = max(areas, key=areas.get) if areas else -1
    pts, pgroup = [], []
    for j in np.nonzero(w)[0]:
        for v in list(tank.outline[j]) + [tank.center[j]]:
            pts.append(v)
            pgroup.append(int(tank.group[j]))
    pts = np.array(pts, dtype=np.float64).reshape(-1, 3)
    pgroup = np.array(pgroup)
    out = []
    for _root, polys in groups.items():
        a_units = float(sum(human.area[p] for p in polys))
        a_m2 = a_units / 39.37 ** 2
        if a_m2 < min_area_m2:
            continue
        verts = np.concatenate([np.asarray(human.outline[p], dtype=np.float64).reshape(-1, 3) for p in polys]
                               + [human.center[polys]])
        zc = float(np.median(human.center[polys, 2]))
        lo, hi = verts[:, :2].min(0) - 900.0, verts[:, :2].max(0) + 900.0
        near = np.nonzero((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))[0]
        best_level = best_low = None
        if len(near):
            P = pts[near]
            # horizontal distance from each Tank point to the nearest patch vertex (chunked)
            d2 = np.full(len(P), np.inf)
            vz = np.full(len(P), np.nan)
            for k in range(0, len(verts), 256):
                V = verts[k:k + 256]
                dd = ((P[:, None, :2] - V[None, :, :2]) ** 2).sum(-1)
                m = dd.argmin(1)
                dmin = dd[np.arange(len(P)), m]
                better = dmin < d2
                d2[better] = dmin[better]
                vz[better] = V[m[better], 2]
            dist = np.sqrt(d2)
            rise = vz - P[:, 2]
            level = np.abs(rise) <= 48.0
            low = rise > 48.0
            if level.any():
                k = int(np.argmin(np.where(level, dist, np.inf)))
                best_level = (float(dist[k]), float(rise[k]), bool(pgroup[near][k] == main_group))
            if low.any():
                # the lowest-cost way up: nearest across the ground (ties: lower rise)
                cost = np.where(low, dist + 0.25 * rise, np.inf)
                k = int(np.argmin(cost))
                best_low = (float(dist[k]), float(rise[k]), bool(pgroup[near][k] == main_group), P[k].tolist())
        if best_level is not None and best_level[0] <= 250.0 and (best_low is None or best_low[0] > 250.0 or best_low[1] < 64):
            kind = "interior"
        elif best_low is not None and best_low[0] <= 250.0:
            kind = "jump" if best_low[1] <= 160.0 else ("climb" if best_low[1] <= 700.0 else "too_high")
        elif best_low is not None and best_low[0] <= 900.0:
            kind = "leap"
        else:
            kind = "isolated"
        c = (human.center[polys] * human.area[polys][:, None]).sum(0) / max(1e-6, a_units)
        out.append({"kind": kind, "area_m2": round(a_m2, 1), "centre": [round(float(v)) for v in c], "z": round(zc),
                    "rise": round(best_low[1]) if best_low else None, "across": round(best_low[0]) if best_low else None,
                    "base": [round(v) for v in best_low[3]] if best_low else None,
                    "base_on_main": best_low[2] if best_low else None,
                    "level_floor_across": round(best_level[0]) if best_level else None, "polys": len(polys)})
    out.sort(key=lambda r: -r["area_m2"])
    for k, r in enumerate(out):
        r["id"] = k + 1
    summary = {}
    for r in out:
        s = summary.setdefault(r["kind"], {"places": 0, "area_m2": 0.0})
        s["places"] += 1
        s["area_m2"] = round(s["area_m2"] + r["area_m2"], 1)
    rises = sorted(r["rise"] for r in out if r["kind"] in ("climb", "too_high"))
    res = {"map": name, "summary": summary, "climb_rises": rises, "places": out}
    if picture:
        _picture(name, tank, human, out)
    return res


def _picture(name, tank, human, places_):
    """top-down: Tank floor grey, pilot-only floor pale, the places coloured by kind and numbered (largest first)"""
    from PIL import Image, ImageDraw, ImageFont
    w = tank.walkable
    allc = np.concatenate([tank.center[w, :2], human.center[human.walkable, :2]])
    lo, hi = allc.min(0) - 200, allc.max(0) + 200
    size = 1500
    s = size / float(max(hi - lo))
    W, H = int((hi[0] - lo[0]) * s) + 1, int((hi[1] - lo[1]) * s) + 1
    img = Image.new("RGB", (W, H + 40), (24, 24, 28))
    dr = ImageDraw.Draw(img)
    to = lambda p: ((p[0] - lo[0]) * s, H - (p[1] - lo[1]) * s)
    for i in np.nonzero(human.walkable)[0]:
        p = human.outline[i]
        if len(p) >= 3:
            dr.polygon([to(v) for v in p], fill=(58, 58, 66))
    for i in np.nonzero(w)[0]:
        p = tank.outline[i]
        if len(p) >= 3:
            dr.polygon([to(v) for v in p], fill=(120, 120, 128))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for r in places_:
        x, y = to(r["centre"])
        rad = max(4.0, math.sqrt(r["area_m2"]) * 39.37 * s * 0.5)
        dr.ellipse([x - rad, y - rad, x + rad, y + rad], outline=COLOURS[r["kind"]], width=3)
        if r["id"] <= 40:
            dr.text((x + rad + 2, y - 7), f'{r["id"]}', fill=COLOURS[r["kind"]], font=font)
    xk = 8
    for k, col in COLOURS.items():
        dr.rectangle([xk, H + 12, xk + 14, H + 26], fill=col)
        dr.text((xk + 18, H + 11), k, fill=(230, 230, 230), font=font)
        xk += 130
    dr.text((W - 330, H + 11), f"{name}: grey Tank floor, dark pilot-only", fill=(230, 230, 230), font=font)
    img.save(OUT / f"{name}_places.png")


def run_places(names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names or STOCK:
        try:
            r = places(name)
        except Exception as e:
            r = {"map": name, "error": repr(e)}
        rows.append(r)
        print(name, json.dumps(r.get("summary", r)), "climb rises:", r.get("climb_rises", [])[:40])
    (OUT / "stock_places.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", OUT / "stock_places.json")
    return rows


def run(names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names or STOCK:
        try:
            r = survey(name)
        except Exception as e:                                        # report and go on
            r = {"map": name, "error": repr(e)}
        rows.append(r)
        print(json.dumps(r))
    (OUT / "stock_maps.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", OUT / "stock_maps.json")
    return rows


if __name__ == "__main__":
    run()
