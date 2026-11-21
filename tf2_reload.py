"""The Type-87's reload as the pilot sees it (part of the gun stage; data/reload_map.json, GUN_RELOAD).

Ronin's first-person gun file carries six reload animations (the full reload, the two 'late' ones the game resumes
from after an interruption, and the same three for an empty magazine).  They were made for the Leadwall's drum,
fed in at the gun's left side.  The Type-87's box magazine stands on top of the gun.  The new animations keep the
stock timing, sounds and events frame for frame and re-author only:

  the gun        a roll about its own barrel and a small shift, added to the stock motion, then back onto it
  the magazine   on def_detail_3 (see reload_map.json): out of the well, hidden, back in the left hand, pushed in
  the left hand  eased onto the new magazine, holding it by its top, then eased back onto the stock motion
  the right hand the stock hold of the grip, solved again for the moved gun

Camera, fingers, collarbones and everything else are the stock rows.  The arms are posed by solving, frame by
frame, for the shoulder and elbow that put the wrist where it has to be (the elbow kept near where the stock
animation has it).  Every authored value is checked against the range the file format can store for that bone,
and after compiling, every frame of the built file is compared: the 13 other animations against stock, the six
reloads against what was authored here.
"""
import json
import math

import numpy as np

from . import config, tf2_sword
from .smd_anim import Anim, align, blend, rot_axis, slerp, smooth
from .tf2 import mdl53
from .tf2_build import _euler_matrix, smd_euler
from .tf2_extract import STOCK_DIR

NAME = "atpov_titan_triple_threat"
DEC = tf2_sword.DECOMPILED / NAME
RELOAD_MAP = config.REPO / "data" / "reload_map.json"
ROT_M90 = _euler_matrix(0.0, 0.0, -math.pi / 2)        # studiomdl's turn of root bones, as decompile53 undoes it


def spec():
    return json.loads(RELOAD_MAP.read_text(encoding="utf-8"))


def MAGAZINE_BONE():
    return spec()["magazine_bone"]["bone"]


def stock_mdl():
    from .tf2_gun import stock
    return stock(NAME)


def T(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def R4(R):
    m = np.eye(4)
    m[:3, :3] = R
    return m


def deg_turn(t):
    return _euler_matrix(*np.radians(t))


def keyed(keys, f):
    """-> (key a, key b, eased fraction) around frame f"""
    if f <= keys[0]["frame"]:
        return keys[0], keys[0], 0.0
    if f >= keys[-1]["frame"]:
        return keys[-1], keys[-1], 0.0
    for a, b in zip(keys, keys[1:]):
        if a["frame"] <= f <= b["frame"]:
            return a, b, smooth((f - a["frame"]) / (b["frame"] - a["frame"]))


# ------------------------------------------------------------------ bones, ranges
def bone_records():
    """-> {name: {"rot": bind Euler, "range": stored range per axis, "root": bool}} from the stock file, with the
    magazine bone's range as reload_map.json widens it"""
    _info, h, d = mdl53.parse(str(stock_mdl()))
    out = {}
    for i in range(h["numbones"]):
        o = h["boneindex"] + i * mdl53.size(mdl53.BONE_FIELDS)
        b = mdl53.unpack(mdl53.BONE_FIELDS, d, o)
        out[mdl53.cstr(d, o + b["sznameindex"])] = {"rot": np.array(b["rot"]), "range": np.array(b["rotscale"]) * 32767,
                                                     "root": b["parent"] < 0}
    mb = spec()["magazine_bone"]
    out[mb["bone"]]["range"] = np.array(mb["rotscale"]) * 32767
    return out


def stored_excess(rec, R):
    """how far a local rotation lies outside what the file can store for this bone (radians, summed over axes)"""
    e = np.array(smd_euler(R))
    if rec["root"]:
        e[2] += math.pi / 2                                   # root bones are stored turned a quarter about z
    d = (e - rec["rot"] + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(np.abs(d) - rec["range"] + 1e-4, 0, None).sum())


# ------------------------------------------------------------------ the arms
def hinge(R):
    return math.atan2(R[0, 2], R[0, 0])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def solve_arm(a, rows, side, target, pole, recs, bend_sign, previous=None):
    """Shoulder, elbow and wrist rotations that put the wrist of one arm onto `target` (4x4, world), the elbow as
    near `pole` as the arm allows, every value inside its stored range.  The collarbone is not moved.
    -> (rotations {bone index: 3x3}, report)"""
    ix = a.index
    n = lambda s: ix[f"def_{side}_{s}"]
    clav, sh, mid, el, wr = n("clav"), n("shoulder"), n("shoulderMid"), n("elbow"), n("wrist")
    Wall = _world(a, rows)
    Wc = Wall[clav]
    C = Wc[:3, :3]
    R_m = _euler_matrix(*rows[mid, 3:6])
    p_m, p_e, p_w = rows[mid, :3], rows[el, :3], rows[wr, :3]
    P = target[:3, 3]
    reach = lambda h: float(np.linalg.norm(p_m + R_m @ (p_e + Ry(h) @ p_w)))
    p_s = rows[sh, :3]
    s0 = C @ p_s + Wc[:3, 3]
    extra = {}
    if float(np.linalg.norm(P - s0)) > reach(0.0) - 0.5:
        # out of reach: swing the collarbone (within what the file can store) towards the target, no further than needed
        Rp = Wall[a.parents[clav]][:3, :3]
        R_align = align(C @ p_s, P - Wc[:3, 3])
        best_t = 0.0
        for t in np.arange(0.05, 1.0001, 0.05):
            Ct = slerp(np.eye(3), R_align, float(t)) @ C
            if stored_excess(recs[f"def_{side}_clav"], Rp.T @ Ct) > 1e-3:
                break
            best_t = float(t)
            if float(np.linalg.norm(P - (Ct @ p_s + Wc[:3, 3]))) <= reach(0.0) - 0.5:
                break
        if best_t > 0:
            C = slerp(np.eye(3), R_align, best_t) @ C
            s0 = C @ p_s + Wc[:3, 3]
            extra[clav] = Rp.T @ C
    D = float(np.linalg.norm(P - s0))
    far = bend_sign * 3.0
    short = 0.0
    if D >= reach(0.0):
        h, short = 0.0, D - reach(0.0)
    elif D <= reach(far):
        h = far
    else:
        lo, hi = 0.0, far
        for _ in range(60):
            mid_h = (lo + hi) / 2
            if reach(mid_h) > D:
                lo = mid_h
            else:
                hi = mid_h
        h = (lo + hi) / 2
    q = p_m + R_m @ (p_e + Ry(h) @ p_w)
    e_loc = p_m + R_m @ p_e
    axis = (P - s0) / max(D, 1e-9)
    R0 = align(q, P - s0)

    def evaluate(phi):
        Rws = rot_axis(axis, phi) @ R0
        elbow = s0 + Rws @ e_loc
        Rsl = C.T @ Rws
        Rwl = (Rws @ R_m @ Ry(h)).T @ target[:3, :3]
        over = stored_excess(recs[f"def_{side}_shoulder"], Rsl) + stored_excess(recs[f"def_{side}_wrist"], Rwl)
        cost = float(np.sum((elbow - pole) ** 2)) + 1e6 * over
        if previous is not None:
            cost += 4.0 * float(np.sum((elbow - previous) ** 2))
        return cost, Rsl, Rwl, elbow, over

    coarse = np.radians(np.arange(0.0, 360.0, 2.0))
    phi0 = coarse[int(np.argmin([evaluate(p)[0] for p in coarse]))]
    fine = phi0 + np.radians(np.arange(-2.0, 2.0, 0.05))
    cost, Rsl, Rwl, elbow, over = min((evaluate(p) for p in fine), key=lambda r: r[0])
    extra.update({sh: Rsl, el: Ry(h), wr: Rwl})
    return extra, {"short": short, "over": over, "elbow": elbow, "bend": h}


def _world(a, rows):
    W = {}
    for i in a.order:
        r = rows[i]
        m = np.eye(4)
        m[:3, :3] = _euler_matrix(r[3], r[4], r[5])
        m[:3, 3] = r[:3]
        p = a.parents[i]
        W[i] = m if p < 0 else W[p] @ m
    return W


# ------------------------------------------------------------------ the hold
def hand_axes(a):
    """the new left hand's finger direction and palm side in the wrist's own axes, and the point in the middle
    of the palm's surface (wrist-local), measured on the cockpit arms"""
    from . import tf2_arms
    s = spec()["left_hand"]
    _i, ha, da = mdl53.parse(str(tf2_arms.stock(tf2_arms.ARMS[0])))
    pov = mdl53.read_bones(da, ha)
    by = {b["name"]: b for b in pov}
    Rw, tw = np.array(by["def_l_wrist"]["bind_rot"]), np.array(by["def_l_wrist"]["bind_pos"])
    loc = lambda p: (np.asarray(p, dtype=np.float64) - tw) @ Rw
    f = loc(by["def_l_finMidA"]["bind_pos"])
    f /= np.linalg.norm(f)
    W = a.world(s["closed_hand_frame"])
    tip = (np.linalg.inv(W[a.index["def_l_wrist"]]) @ np.append(W[a.index["def_l_finMidC"]][:3, 3], 1))[:3]
    curl = tip - loc(by["def_l_finMidC"]["bind_pos"])
    n = curl - f * (curl @ f)
    n /= np.linalg.norm(n)
    arms = tf2_arms.body_arms()
    pos, _nrm, _notes = tf2_arms.onto_first_person(arms, pov, quiet=True)
    P = loc(pos[arms["bone"] == "def_l_wrist"])
    palm = f * float(((P @ f).min() + (P @ f).max()) / 2) + n * float((P @ n).max())
    return f, n, palm


def hold_matrix(a, magazine_top):
    """the left wrist's pose in the gun's frame while it holds the seated magazine by its top"""
    s = spec()["left_hand"]
    f_w, n_w, palm = hand_axes(a)
    f_m = np.array(s["fingers"], dtype=np.float64)
    n_m = np.array(s["palm_faces"], dtype=np.float64)
    n_m /= np.linalg.norm(n_m)
    f_m = f_m - n_m * (f_m @ n_m)
    f_m /= np.linalg.norm(f_m)
    Wm = np.column_stack([f_m, n_m, np.cross(f_m, n_m)])
    Ww = np.column_stack([f_w, n_w, np.cross(f_w, n_w)])
    G = np.eye(4)
    G[:3, :3] = Wm @ Ww.T
    G[:3, 3] = magazine_top - G[:3, :3] @ palm
    return G


# ------------------------------------------------------------------ the grip
def _apply(M, P):
    return P @ M[:3, :3].T + M[:3, 3]


class MagazineSolid:
    """The seated magazine as a solid in the gun's frame: its side outline (up y, forward z) on a fine grid, and its
    thickness (x).  Its own two box pieces, nothing else of the gun."""

    def __init__(self, step=0.1):
        from . import tf2_gun
        m = tf2_gun.fitted(0)
        mask = tf2_gun.magazine_mask(m)
        pos = np.asarray(m["pos"], dtype=np.float64)
        tri = np.asarray(m["tri"], dtype=np.int64)[mask]
        P = pos[np.unique(tri)]
        self.half = float(np.abs(P[:, 0]).max())
        self.step = step
        self.lo = P[:, 1:].min(0) - 1.0
        shape = tuple((np.ceil((P[:, 1:].max(0) + 1.0 - self.lo) / step)).astype(int) + 1)
        grid = np.zeros(shape, bool)
        for t in tri:
            a, b, c = pos[t[0], 1:], pos[t[1], 1:], pos[t[2], 1:]
            v0, v1 = b - a, c - a
            d = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(d) < 1e-9:
                continue                                      # a face seen edge-on from the side
            mn = np.clip(np.floor((np.minimum(np.minimum(a, b), c) - self.lo) / step).astype(int), 0, np.array(shape) - 1)
            mx = np.clip(np.ceil((np.maximum(np.maximum(a, b), c) - self.lo) / step).astype(int), 0, np.array(shape) - 1)
            Y, Z = np.meshgrid(self.lo[0] + step * np.arange(mn[0], mx[0] + 1),
                               self.lo[1] + step * np.arange(mn[1], mx[1] + 1), indexing="ij")
            py, pz = Y - a[0], Z - a[1]
            u = (py * v1[1] - pz * v1[0]) / d
            v = (v0[0] * pz - v0[1] * py) / d
            grid[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1] |= (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        self.grid = grid
        self._grown = {0: grid}

    def _grow(self, margin):
        k = int(round(margin / self.step))
        if k not in self._grown:
            g = self.grid
            out = g.copy()
            for dy in range(-k, k + 1):
                for dz in range(-k, k + 1):
                    if dy * dy + dz * dz <= k * k and (dy or dz):
                        out[max(dy, 0):g.shape[0] + min(dy, 0), max(dz, 0):g.shape[1] + min(dz, 0)] |= \
                            g[max(-dy, 0):g.shape[0] + min(-dy, 0), max(-dz, 0):g.shape[1] + min(-dz, 0)]
            self._grown[k] = out
        return self._grown[k]

    def inside(self, P, margin=0.0):
        """which points (gun frame, magazine seated) lie inside the magazine grown by `margin`"""
        g = self._grow(margin)
        i = np.round((P[:, 1:] - self.lo) / self.step).astype(int)
        ok = (i >= 0).all(1) & (i[:, 0] < g.shape[0]) & (i[:, 1] < g.shape[1])
        out = np.zeros(len(P), bool)
        out[ok] = g[i[ok, 0], i[ok, 1]] & (np.abs(P[ok, 0]) <= self.half + margin)
        return out

    def top_edge(self):
        """the straight part of the top edge, as seen from the side: (direction rear -> front, its two ends) in the
        gun's frame (x = 0)"""
        cols = np.where(self.grid.any(0))[0]
        top = np.array([np.where(self.grid[:, j])[0].max() for j in cols])
        z = self.lo[1] + self.step * cols
        y = self.lo[0] + self.step * top
        z_peak = z[int(np.argmax(y))]
        sel = (z > z_peak + 1.0) & (z < z.max() - 1.0)
        m, c = np.polyfit(z[sel], y[sel], 1)
        z0, z1 = z[sel].min(), z[sel].max()
        p0, p1 = np.array([0.0, m * z0 + c, z0]), np.array([0.0, m * z1 + c, z1])
        return (p1 - p0) / np.linalg.norm(p1 - p0), p0, p1


def left_hand_mesh():
    """the cockpit arms' left hand piece by piece: {bone name: vertices in that bone's own frame}, and the arms
    file's bones by name"""
    from . import tf2_arms
    _i, ha, da = mdl53.parse(str(tf2_arms.stock(tf2_arms.ARMS[0])))
    pov = mdl53.read_bones(da, ha)
    by = {b["name"]: b for b in pov}
    arms = tf2_arms.body_arms()
    pos, _n, _x = tf2_arms.onto_first_person(arms, pov, quiet=True)
    bones, _l = tf2_arms.finger_joints(arms["bone"], pos, pov, quiet=True)
    out = {}
    for b in sorted(set(bones)):
        if b == "def_l_wrist" or b.startswith("def_l_fin"):
            R, t = np.array(by[b]["bind_rot"]), np.array(by[b]["bind_pos"])
            out[b] = (pos[bones == b] - t) @ R
    return out, by


def hand_chain(a, rows, rot=None):
    """{bone index: 4x4 relative to the left wrist} for the bones below it, from one frame's rows; `rot` replaces
    local rotations {bone index: 3x3}"""
    wr = a.index["def_l_wrist"]
    out = {wr: np.eye(4)}
    for i in a.order:
        p = a.parents[i]
        if p in out and i not in out:
            R = rot[i] if rot is not None and i in rot else _euler_matrix(*rows[i, 3:6])
            out[i] = out[p] @ T(rows[i, :3]) @ R4(R)
    return out


def hand_points(a, chain, mesh, names=None):
    """the hand's vertices relative to the wrist (only the pieces in `names`, if given)"""
    return np.concatenate([_apply(chain[a.index[b]], P) for b, P in mesh.items() if names is None or b in names])


def _axis_angle(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / max(np.linalg.norm(w), 1e-12), math.atan2(np.linalg.norm(w) / 2, (np.trace(R) - 1) / 2)


def grip(a, quiet=False):
    """The left hand's hold of the seated magazine (reload_map.json left_hand.grip): the palm flat on one side of it,
    the knuckles along the straight part of its top edge, every finger bent at its joints until it touches the
    magazine, the thumb closed onto it the way the stock animation closes it.  Nothing of the hand is moved or
    resized; only joints turn.  -> (G: the wrist's pose in the gun's frame, {bone index: local rotation}, report)"""
    g = spec()["left_hand"]["grip"]
    solid = MagazineSolid()
    mesh, by = left_hand_mesh()
    ix = a.index
    recs = bone_records()
    rows = a.rows[0]
    fingers = [n for n in ix if n.startswith("def_l_fin")]
    open_rot = {ix[n]: _euler_matrix(*recs[n]["rot"]) for n in fingers}
    chain = hand_chain(a, rows, open_rot)
    Rw, tw = np.array(by["def_l_wrist"]["bind_rot"]), np.array(by["def_l_wrist"]["bind_pos"])
    worst = max(float(np.linalg.norm(chain[ix[n]][:3, 3] - (np.array(by[n]["bind_pos"]) - tw) @ Rw)) for n in fingers if n in by)
    if worst > 0.05:
        raise SystemExit(f"the gun file's finger joints are not where the arms file has them ({worst:.3f} apart)")
    # the hand's orientation: palm onto the side named, knuckles along the top edge, fingers pointing up past it
    f_w, n_w, _palm = hand_axes(a)
    e, p0, p1 = solid.top_edge()
    n_m = np.array(g["palm_faces"], dtype=np.float64)
    n_m /= np.linalg.norm(n_m)
    f_m = np.cross(e, n_m)
    f_m = (f_m if f_m[1] > 0 else -f_m) / np.linalg.norm(f_m)
    Wm = np.column_stack([f_m, n_m, np.cross(f_m, n_m)])
    Ww = np.column_stack([f_w, n_w, np.cross(f_w, n_w)])
    G = np.eye(4)
    G[:3, :3] = Wm @ Ww.T
    knuckles = np.mean([chain[ix[f"def_l_fin{f}A"]][:3, 3] for f in ("Index", "Mid", "Ring")], axis=0)
    on_edge = p0 + g["knuckles_along_edge"] * (p1 - p0)
    clear = float(g["clearance"])
    # slide the open hand (thumb aside) onto the magazine's side until the palm touches
    body = [b for b in mesh if "Thumb" not in b]
    start = on_edge + g["knuckles_above_edge"] * f_m - 20.0 * n_m
    gap = None
    for s in np.arange(0.0, 40.0, 0.05):
        G[:3, 3] = start + s * n_m - G[:3, :3] @ knuckles
        if solid.inside(_apply(G, hand_points(a, chain, mesh, body)), clear).any():
            gap = s - 0.05
            break
    if gap is None or gap < 0:
        raise SystemExit("the open hand cannot be laid onto the magazine's side")
    G[:3, 3] = start + gap * n_m - G[:3, :3] @ knuckles
    rot = dict(open_rot)
    below = lambda i: {j for j in ix.values() if j == i or i in _ancestors(a, j)}
    driven = {ix[b] for b in mesh}
    report = {"palm": "on the magazine's side", "bent": {}}

    def touches(i):
        names = [b for b in mesh if ix[b] in below(i)]
        ch = hand_chain(a, rows, rot)
        return bool(solid.inside(_apply(G, hand_points(a, ch, mesh, names)), clear).any())

    curl_axis = np.array(g["curl_axis"], dtype=np.float64)
    for f in ("Index", "Mid", "Ring"):
        for k in "ABC":
            i = ix[f"def_l_fin{f}{k}"]
            if not (below(i) & driven):
                continue                                      # turns nothing of this hand
            base = rot[i]
            step = math.radians(1.0)
            th = 0.0
            if touches(i):
                report["bent"][a.names[i]] = "touches before bending"
                continue
            while th < math.radians(g["most_bend_degrees"]):
                rot[i] = base @ rot_axis(curl_axis, th + step)
                if touches(i):
                    break
                th += step
            rot[i] = base @ rot_axis(curl_axis, th)
            report["bent"][a.names[i]] = round(math.degrees(th), 1)
    # the thumb: about the axis the stock animation closes it about, from open until it touches
    t = ix["def_l_finThumbA"]
    closed = _euler_matrix(*a.rows[spec()["left_hand"]["closed_hand_frame"], t, 3:6])
    t_axis, _ang = _axis_angle(open_rot[t].T @ closed)
    # first swung out in the palm's plane ('thumb_swing', about an axis in the wrist's own frame), so it lies
    # along the magazine's side instead of standing up past its top
    sw = g.get("thumb_swing", {"axis_in_wrist": [0, 0, 1], "degrees": 0})
    Rt = chain[t][:3, :3]                                     # the thumb joint's turn against the wrist, at rest
    swung = open_rot[t] @ rot_axis(Rt.T @ np.array(sw["axis_in_wrist"], dtype=np.float64), math.radians(sw["degrees"]))
    th_list = np.radians(np.arange(-g["thumb_open_degrees"], g["most_bend_degrees"] + 0.5, 1.0))
    free_seen, chosen, first_free = False, None, None
    for th in th_list:
        rot[t] = swung @ rot_axis(t_axis, th)
        hit = touches(t)
        if not hit:
            free_seen, chosen = True, th
            first_free = th if first_free is None else first_free
        elif free_seen:
            break
    if chosen is None:
        raise SystemExit("the thumb finds no place beside the magazine")
    rot[t] = swung @ rot_axis(t_axis, chosen)
    report["bent"]["def_l_finThumbA"] = round(math.degrees(chosen), 1)
    keep = {i: R for i, R in rot.items() if i in {ix[n] for n in fingers} and (below(i) & driven)}
    # the open hand, as it arrives and lets go: fingers straight, the thumb swung out and opened a little further
    # than its grip (never further than where it first clears the magazine)
    opened = {i: open_rot[i] for i in keep}
    opened[t] = swung @ rot_axis(t_axis, max(first_free, chosen - math.radians(g["thumb_opens_by_degrees"])))
    final = hand_chain(a, rows, rot)
    pts = _apply(G, hand_points(a, final, mesh))
    report["inside_the_magazine"] = int(solid.inside(pts, 0.0).sum())
    report["touching"] = int(solid.inside(pts, clear + 0.3).sum())
    if not quiet:
        print("  grip: palm on the magazine's side, joints bent (degrees): "
              + ", ".join(f"{n[6:]} {v}" for n, v in report["bent"].items())
              + f"; hand vertices inside the magazine {report['inside_the_magazine']}, within {clear + 0.3:.2f} of it {report['touching']}")
    return G, keep, report, opened


def _ancestors(a, j):
    out = set()
    while a.parents[j] >= 0:
        j = a.parents[j]
        out.add(j)
    return out


def finger_weight(f):
    """how far the left fingers are in their grip at full-reload frame f (0 = the stock fingers)"""
    g = spec()["left_hand"].get("grip")
    if not g:
        return 0.0
    c0, c1 = g["close_frames"]
    o0, o1 = g["open_frames"]
    if f <= c0 or f >= o1:
        return 0.0
    if f < c1:
        return smooth((f - c0) / (c1 - c0))
    if f <= o0:
        return 1.0
    return 1.0 - smooth((f - o0) / (o1 - o0))


def widened_ranges():
    """{bone: rotscale} for the left finger bones whose authored values lie outside their stock stored range (the
    grip bends them further than Ronin's reload does); the stored range is widened to fit, nothing else changes"""
    sp = spec()
    recs = bone_records()
    need = {}
    for fn in sp["files"]:
        a = Anim(str(DEC / "tsf" / fn))
        for name, i in a.index.items():
            if not name.startswith("def_l_fin"):
                continue
            for k in range(a.nframes):
                e = np.array(smd_euler(_euler_matrix(*a.rows[k, i, 3:6])))
                d = np.abs((e - recs[name]["rot"] + math.pi) % (2 * math.pi) - math.pi)
                need[name] = np.maximum(need.get(name, np.zeros(3)), d)
    out = {}
    for name, d in need.items():
        rng = recs[name]["range"]
        if (d > rng - 1e-4).any():
            out[name] = np.maximum(rng, d * 1.02 + math.radians(0.5)) / 32767
    return out


# ------------------------------------------------------------------ authoring
def magazine_geometry():
    """(centre, middle of the top cap, bind position of the magazine bone) in the gun's frame"""
    from . import tf2_gun
    m = tf2_gun.fitted(0)
    mask = tf2_gun.magazine_mask(m)
    if mask is None:
        raise SystemExit("GUN_RELOAD is off: the gun has no separate magazine")
    pos = np.asarray(m["pos"], dtype=np.float64)[np.unique(np.asarray(m["tri"])[mask])]
    lo, hi = pos.min(0), pos.max(0)
    centre = (lo + hi) / 2
    top = pos[pos[:, 1] > hi[1] - 2.0]
    top_mid = np.array([top[:, 0].mean(), hi[1], top[:, 2].mean()])
    qc = (DEC / f"{NAME}.qc").read_text(encoding="utf-8")
    import re
    mline = re.search(r'(?m)^\$definebone "%s" "([^"]*)" (\S+) (\S+) (\S+)' % MAGAZINE_BONE(), qc)
    return centre, top_mid, np.array([float(mline.group(k)) for k in (2, 3, 4)]), mline.group(1)


def magazine_motion(f, centre):
    """the magazine's rigid motion in the gun's frame at full-reload frame f"""
    a, b, s = keyed(spec()["magazine"]["keys"], f)
    move = (1 - s) * np.array(a["move"], dtype=np.float64) + s * np.array(b["move"], dtype=np.float64)
    R = slerp(deg_turn(a["turn"]), deg_turn(b["turn"]), s)
    return T(centre + move) @ R4(R) @ T(-centre)


def gun_change(f):
    """(roll in radians about the gun's barrel, shift in the SMD root frame) at full-reload frame f"""
    a, b, s = keyed(spec()["gun"]["keys"], f)
    roll = math.radians((1 - s) * a["roll"] + s * b["roll"])
    left, fwd, up = ((1 - s) * np.array(a["move"], dtype=np.float64) + s * np.array(b["move"], dtype=np.float64))
    return roll, np.array([left, -fwd, up])                 # view (left, forward, up) -> SMD (+x left, -y forward, +z up)


def author(quiet=False, strict=True):
    """write the six new reload animations to build/tf2/decompiled/<file>/tsf/ -> report"""
    sp = spec()
    recs = bone_records()
    centre, top_mid, b3, b3_parent = magazine_geometry()
    if b3_parent != "def_c_base":
        raise SystemExit(f"{MAGAZINE_BONE()} hangs from {b3_parent}, expected def_c_base (the gun)")
    lh = sp["left_hand"]
    out_dir = DEC / "tsf"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    full = [fn for fn, off in sp["files"].items() if off == 0][0]
    gp = lh.get("grip")
    if gp:
        G, finger_rot, report["grip"], open_rot = grip(Anim(str(DEC / full)), quiet=quiet)
        let_go = gp["open_frames"][1]                         # the fingers open before the hand moves off
        approach = np.array(gp["approach_from"], dtype=np.float64)
    else:
        G, finger_rot, open_rot = hold_matrix(Anim(str(DEC / full)), top_mid), {}, {}
        let_go, approach = lh["hold_until"], np.zeros(3)
    for fn, offset in sp["files"].items():
        a = Anim(str(DEC / fn))
        ix = a.index
        gun, base, mag = ix["ja_c_propGun"], ix["def_c_base"], ix[MAGAZINE_BONE()]
        bend = {s: math.copysign(1.0, hinge(_euler_matrix(*a.rows[0, ix[f"def_{s}_elbow"], 3:6])) or -1.0) for s in "lr"}
        rows = a.rows.copy()
        # where the stock left hand is when the new motion hands back to it (out of view); for a 'late' file that
        # starts after that frame there is nothing to hand back
        k_park = lh["stock_again_from"] - offset
        park = _world(a, a.rows[k_park])[ix["def_l_wrist"]] if 0 <= k_park < a.nframes else None
        worst = {"short": 0.0, "over": 0.0, "wrist_error": 0.0, "right_short": 0.0}
        prev = {"l": None, "r": None}
        for k in range(a.nframes):
            f = offset + k
            stock = _world(a, a.rows[k])
            r = rows[k]
            # the gun
            roll, shift = gun_change(f)
            Gs = stock[gun]
            Gn = T(shift) @ Gs @ R4(_euler_matrix(0.0, 0.0, roll))
            r[gun, :3], r[gun, 3:6] = Gn[:3, 3], smd_euler(Gn[:3, :3])
            # the magazine: its motion in the gun's frame, on a bone whose rest place is b3 on the gun
            D = magazine_motion(f, centre)
            L3 = D @ T(b3)
            r[mag, :3], r[mag, 3:6] = L3[:3, 3], smd_euler(L3[:3, :3])
            world = _world(a, r)
            Wbase = world[base]
            moved = abs(roll) > 1e-9 or np.abs(shift).max() > 1e-9
            # the right hand: its stock hold of the grip, on the moved gun
            if moved:
                target = Gn @ np.linalg.inv(Gs) @ stock[ix["def_r_wrist"]]
                rot, info = solve_arm(a, r, "r", target, stock[ix["def_r_elbow"]][:3, 3], recs, bend["r"], prev["r"])
                for i, R in rot.items():
                    r[i, 3:6] = smd_euler(R)
                prev["r"] = info["elbow"]
                worst["right_short"], worst["over"] = max(worst["right_short"], info["short"]), max(worst["over"], info["over"])
            # the left hand
            hold = lambda ff: Wbase @ magazine_motion(min(max(ff, lh["hold_from"]), lh["hold_until"]), centre) @ G
            hw = 0.0                                          # how near the hand is to its hold (for the fingers)
            if f <= lh["stock_until"] or f >= lh["stock_again_from"]:
                target = None
            elif f < lh["hold_from"]:
                # in from the side named by 'approach_from' (gun frame), so the open hand meets the magazine flat
                s = smooth((f - lh["stock_until"]) / (lh["hold_from"] - lh["stock_until"]))
                near = Wbase @ magazine_motion(lh["hold_from"], centre) @ T(approach * (1 - s)) @ G
                target, hw = blend(stock[ix["def_l_wrist"]], near, s), s
            elif f <= let_go:
                target, hw = hold(f), 1.0
            else:
                s = smooth((f - let_go) / (lh["released_by"] - let_go))
                off = Wbase @ T(np.array(lh["lift"], dtype=np.float64) * smooth(min(1.0, 2 * s))) @ G
                target, hw = blend(off, park, s), 1.0 - s
            if target is not None:
                sh = np.array(lh.get("elbow_shift", [0, 0, 0]), dtype=np.float64)
                pole = stock[ix["def_l_elbow"]][:3, 3] + np.array([sh[0], -sh[1], sh[2]])
                rot, info = solve_arm(a, r, "l", target, pole, recs, bend["l"], prev["l"])
                for i, R in rot.items():
                    r[i, 3:6] = smd_euler(R)
                prev["l"] = info["elbow"]
                worst["short"], worst["over"] = max(worst["short"], info["short"]), max(worst["over"], info["over"])
                if info["short"] > 0.25:
                    worst.setdefault("short_at", []).append((f, round(info["short"], 2)))
                got = _world(a, r)[ix["def_l_wrist"]]
                worst["wrist_error"] = max(worst["wrist_error"], float(np.linalg.norm(got[:3, 3] - target[:3, 3])))
            else:
                prev["l"] = None
            # the left fingers: open (straight, as at rest) while the hand is near the magazine and not holding it,
            # closed into the grip while it holds it, the stock fingers away from it
            w = finger_weight(f)
            if hw > 0 or w > 0:
                for i, Rg in finger_rot.items():
                    at_rest = slerp(_euler_matrix(*a.rows[k, i, 3:6]), open_rot[i], hw)
                    r[i, 3:6] = smd_euler(slerp(at_rest, Rg, w))
            if not moved:
                prev["r"] = None
        # every authored value must be storable
        authored = [gun, mag] + [ix[f"def_{s}_{b}"] for s in "lr" for b in ("clav", "shoulder", "elbow", "wrist")]
        bad = []
        for k in range(a.nframes):
            for i in authored:
                nm = a.names[i]
                if nm.endswith("_elbow"):
                    continue                                    # a pure hinge; its stored form is always in range
                x = stored_excess(recs[nm], _euler_matrix(*rows[k, i, 3:6]))
                if x > 1e-3:
                    e = np.array(smd_euler(_euler_matrix(*rows[k, i, 3:6])))
                    if recs[nm]["root"]:
                        e[2] += math.pi / 2
                    d = (e - recs[nm]["rot"] + math.pi) % (2 * math.pi) - math.pi
                    axes = "".join("xyz"[c] for c in range(3) if abs(d[c]) > recs[nm]["range"][c])
                    bad.append((offset + k, nm, axes, round(x, 3)))
        a.rows = rows
        a.write(str(out_dir / fn))
        report[fn] = dict(worst, out_of_range=bad, out_of_range_count=len(bad), frames=a.nframes)
        if not quiet:
            print(f"  {fn}: {a.nframes} frames written; left arm short of its target by up to {worst['short']:.2f}, "
                  f"right hand off the grip by up to {worst['right_short']:.2f}, "
                  f"left wrist off its target by up to {worst['wrist_error']:.3f}; values out of range: {len(bad)}"
                  + (f" {bad}" if offset == 0 else "")
                  + (f"; arm short at (frame, units) {worst['short_at']}" if offset == 0 and worst.get("short_at") else ""))
    if strict and lh.get("grip") and report["grip"]["inside_the_magazine"]:
        raise SystemExit("the grip puts part of the hand inside the magazine - see above")
    if strict and any(v.get("out_of_range_count") for v in report.values()):
        raise SystemExit("some authored rotations lie outside what the file can store - see above")
    return report


def patch_qc(text):
    """the six reload animations -> the authored ones; the magazine bone's stored range widened for them"""
    import re
    sp = spec()
    mb = sp["magazine_bone"]
    n = 0
    for fn in sp["files"]:
        pat = re.compile(r'(?m)^(\$animation "[^"]+" )"%s"(.*)$' % re.escape(fn))
        m = pat.search(text)
        if not m:
            raise SystemExit(f"the recipe has no animation reading {fn}")
        rest = re.sub(r' posscale "%s" \S+' % mb["bone"], "", m.group(2)) + f' posscale "{mb["bone"]}" {mb["posscale"]}'
        text = text[:m.start()] + f'{m.group(1)}"tsf/{fn}"{rest}' + text[m.end():]
        n += 1
    line = re.compile(r'(?m)^\$bonerotscale "%s" .*$' % mb["bone"])
    if not line.search(text):
        raise SystemExit(f"the recipe has no $bonerotscale line for {mb['bone']}")
    text = line.sub('$bonerotscale "%s" %s' % (mb["bone"], " ".join("%.9g" % v for v in mb["rotscale"])), text)
    # the left fingers' stored ranges, where the grip bends them further than Ronin's reload does
    wide = widened_ranges()
    for bone, rs in sorted(wide.items()):
        fl = re.compile(r'(?m)^\$bonerotscale "%s" .*$' % bone)
        if not fl.search(text):
            raise SystemExit(f"the recipe has no $bonerotscale line for {bone}")
        text = fl.sub('$bonerotscale "%s" %s' % (bone, " ".join("%.9g" % v for v in rs)), text)
    if wide:
        print("  reload: stored turning range widened for " + ", ".join(
            f"{b[6:]} (to {', '.join(f'{math.degrees(v * 32767):.0f}' for v in rs)} degrees)" for b, rs in sorted(wide.items())))
    # events moved (reload_map.json "events"): each must be found exactly once in its sequence
    moved = []
    for r in sp.get("events", {}).get("retime", []):
        block = re.search(r'(?ms)^\$sequence "%s"\n\{.*?^\}' % re.escape(r["sequence"]), text)
        if not block:
            raise SystemExit(f"the recipe has no sequence {r['sequence']}")
        old = '{ event "%s" %d ' % (r["event"], r["from"])
        if block.group(0).count(old) != 1:
            raise SystemExit(f"{r['sequence']}: expected one {r['event']} at frame {r['from']}")
        new_block = block.group(0).replace(old, '{ event "%s" %d ' % (r["event"], r["to"]))
        text = text[:block.start()] + new_block + text[block.end():]
        moved.append(f"{r['sequence']} {r['event']} {r['from']}->{r['to']}")
    print(f"  reload: {n} animations repointed to the authored ones; {mb['bone']} may turn and move in them"
          + (f"; events moved: {', '.join(moved)}" if moved else ""))
    return text


def expected_sequences(sequences):
    """the stock sequence records (mdl53.parse) as they must read after the events reload_map.json moves"""
    import copy
    out = copy.deepcopy(sequences)
    for r in spec().get("events", {}).get("retime", []):
        frames = Anim(str(DEC / r["file"])).nframes
        for s in out:
            if s.get("label") != r["sequence"]:
                continue
            hits = [e for e in s["events"] if e["name"] == r["event"] and abs(e["cycle"] - r["from"] / (frames - 1)) < 2e-3]
            if len(hits) != 1:
                raise SystemExit(f"stock {r['sequence']}: expected one {r['event']} at frame {r['from']}")
            hits[0]["cycle"] = round(r["to"] / (frames - 1), 3)
    return out


def hidden_frames():
    """full-reload frames during which the game hides the magazine piece"""
    sp = spec()
    hide = next((r["to"] for r in sp.get("events", {}).get("retime", []) if r["event"] == "AE_WPN_CLIPBODYGROUP_HIDE"), 10)
    return hide, 31


# ------------------------------------------------------------------ checking the built file
def check(built, tolerance_pos=0.05, tolerance_deg=0.05):
    """Every frame of every animation in the built file: the 13 others against stock, the six reloads against the
    authored SMDs.  -> {"ok": bool, "lines": [...]}"""
    import contextlib
    import io
    import runpy
    from . import tools
    with contextlib.redirect_stdout(io.StringIO()):
        dc = runpy.run_path(str(tools.find("r2-model-compiler", "decompile53.py")), run_name="decompile53_as_library")
    m53 = dc["mdl53"]
    sp = spec()
    authored = {}
    for fn, _off in sp["files"].items():
        authored[fn] = Anim(str(DEC / "tsf" / fn))
    qc = (DEC / f"{NAME}_tsf.qc").read_text(encoding="utf-8")
    import re
    files = dict(re.findall(r'(?m)^\$animation "([^"]+)" "([^"]+)"', qc))
    da, db = open(stock_mdl(), "rb").read(), open(built, "rb").read()
    ma, mb = m53.Model(da), m53.Model(db)
    ba, bb = ma.bones(), mb.bones()
    lines, ok = [], True
    for ai in range(ma.hdr["numlocalanim"]):
        aa, name, fa = dc["decode_anim"](da, ma, ba, ai)
        ab, other, fb = dc["decode_anim"](db, mb, bb, ai)
        if name != other or aa["numframes"] != ab["numframes"]:
            return {"ok": False, "lines": [f"animation {ai}: {name} / {other} differ in name or length"]}
        src = files.get(name, "")
        mine = src.startswith("tsf/") and src[4:] in authored
        worst_p = worst_r = 0.0
        for k in range(aa["numframes"]):
            if mine:
                an = authored[src[4:]]
                ref = []
                for i in range(len(bb)):
                    rr = an.rows[k, i]
                    p, R = np.array(rr[:3]), _euler_matrix(*rr[3:6])
                    if bb[i]["parent"] < 0:                 # authored in the decompiler's frame for root bones
                        p, R = ROT_M90.T @ p, ROT_M90.T @ R
                    ref.append((p, R))
            else:
                ref = [(np.array(x[0]), np.array(x[1])[:3, :3]) for x in fa[k]]
            for i, (p, R) in enumerate(ref):
                q = fb[k][i]
                worst_p = max(worst_p, float(np.linalg.norm(np.array(q[0]) - p)))
                Rb = np.array(q[1])[:3, :3]
                c = (np.trace(R.T @ Rb) - 1) / 2
                worst_r = max(worst_r, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        good = worst_p <= tolerance_pos and worst_r <= tolerance_deg
        ok &= good
        lines.append(f"    {name:26s} {aa['numframes']:3d} frames against {'the authored reload' if mine else 'stock'}: "
                     f"largest difference {worst_p:.4f} units, {worst_r:.4f} degrees{'' if good else '   <-- TOO LARGE'}")
    return {"ok": ok, "lines": lines}
