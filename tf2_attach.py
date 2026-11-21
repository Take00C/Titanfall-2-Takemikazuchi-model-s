"""The named points of Ronin's body that this skin moves (part of the compile stage).

Ronin's body file carries about a hundred named points ("attachments") that tell the game where to draw
things that hang on the titan.  data/attach_map.json lists the ones this skin moves.  They are display-only:
no hitbox, bone, animation or gameplay value is involved, and every other point stays exactly as stock.

The stowed sword.  Project MIKHAIL's own model has a point on each back holder where a stowed weapon is
laid (pm_clean exports them as "mounts").  The sword's grip marker is laid on the chosen holder's point, the
way that game does it, and Ronin's SIDEARM_HOLSTER point - plus the two points Ronin keeps on the stowed
sword's tip and edge - is moved so that Titanfall 2 draws the stowed sword exactly there.

How Titanfall 2 lays a holstered weapon on that point is read off Ronin's own file: its sword_tip_side and
sword_edge_side points are the in-hand sword_tip and sword_edge points carried onto the holster, which fixes
the turn between the sword file's axes and the point's axes ((x, y, z) -> (z, x, y)).

    python -m tsfronin.tf2_attach            print the plan
    python -m tsfronin.tf2_attach --preview  draw it
"""
import json
import math
import re
import sys

import numpy as np

from . import config
from .retarget import OUT_DIR, RONIN_SKELETON, body_block
from .tf2 import mdl53
from .tf2_extract import RONIN_BODY, STOCK_DIR

ATTACH_MAP = config.REPO / "data" / "attach_map.json"


def _rotation(R, what):
    R = np.asarray(R, dtype=np.float64)
    if abs(np.linalg.det(R) - 1.0) > 1e-4 or not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        raise SystemExit(f"{what} is not a rotation")
    return R


def stock_points():
    info, _h, _d = mdl53.parse(str(STOCK_DIR / RONIN_BODY.replace("/", "\\")))
    return {a["name"]: a for a in info["attachments"]}


def ronin_bones():
    return {b["name"]: b for b in json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))["bones"]}


def stowed_sword(spec, stock, bones):
    """-> ({point: (bone, pos, rot)}, (R, t)) where R, t take the sword file's coordinates onto the fitted body"""
    from . import tf2_sword
    clean = config.CLEAN_DIR / config.UNIT
    mounts = {m["name"]: m for m in json.loads((clean / "skeleton.json").read_text(encoding="utf-8")).get("mounts", [])}
    if spec["holder"] not in mounts:
        raise SystemExit(f"attach_map.json names holder point {spec['holder']!r}; the model has {sorted(mounts)} "
                         f"(re-run the clean stage if that list is empty)")
    holder = mounts[spec["holder"]]
    sm = json.loads(tf2_sword.SWORD_MAP.read_text(encoding="utf-8"))
    markers = {m["name"]: m for m in json.loads((config.CLEAN_DIR / config.SWORD_UNIT / "skeleton.json")
                                                .read_text(encoding="utf-8"))["markers"]}
    grip = markers[sm["grip"]["tsf_marker"]]
    if "rot" not in grip:
        raise SystemExit("the cleaned sword has no orientation on its grip marker - re-run the sword-clean stage")
    R_fit = _rotation(sm["axes"]["ronin_from_tsf"], "sword_map.json axes")
    hand = np.array(sm["grip"]["ronin"], dtype=np.float64)
    # sword file coordinates -> fitted body: the grip marker's frame laid on the holder point's frame
    R_T = _rotation(np.array(holder["rot"]) @ np.array(grip["rot"]).T @ R_fit.T, "the holder placement")
    t_T = body_block().place(np.array(holder["pos"], dtype=np.float64), None)[0] - R_T @ hand

    a0 = stock[spec["point"]]
    A0r, A0p = np.array(a0["rot"]), np.array(a0["pos"])
    turns = []
    for side, in_hand in spec["carried_along"].items():
        s, h = stock[side], stock[in_hand]
        if s["bone"] != a0["bone"]:
            raise SystemExit(f"{side} is on {s['bone']}, {spec['point']} on {a0['bone']}")
        C = A0r.T @ np.array(s["rot"]) @ np.array(h["rot"]).T
        off = float(np.abs(A0r.T @ (np.array(s["pos"]) - A0p) - C @ np.array(h["pos"])).max())
        if off > 0.01:
            raise SystemExit(f"{side} is not {in_hand} carried onto the holster (off by {off:.3f} units)")
        turns.append(C)
    C = _rotation(turns[0], "the holstered sword's turn")
    if any(float(np.abs(c - C).max()) > 1e-4 for c in turns):
        raise SystemExit("Ronin's stowed-sword points disagree about how the sword is turned in the holster")
    b = bones[a0["bone"]]
    Rb, tb = np.array(b["rot"]), np.array(b["pos"])
    A_rot = _rotation(Rb.T @ R_T @ C.T, "the new holster point")
    A_pos = Rb.T @ (t_T - tb)
    out = {spec["point"]: (a0["bone"], A_pos, A_rot)}
    move = A_rot @ A0r.T                                       # the same move, applied to the points that ride along
    for side in spec["carried_along"]:
        s = stock[side]
        out[side] = (s["bone"], move @ (np.array(s["pos"]) - A0p) + A_pos, move @ np.array(s["rot"]))
    return out, (R_T, t_T)


def plan(quiet=False):
    am = json.loads(ATTACH_MAP.read_text(encoding="utf-8"))
    stock, bones = stock_points(), ronin_bones()
    points, placed = stowed_sword(am["stowed_sword"], stock, bones)
    if not quiet:
        on_body = lambda p, bn: (np.array(bones[bn]["rot"]) @ np.array(p) + np.array(bones[bn]["pos"])).round(1).tolist()
        for name, (bone, pos, _rot) in points.items():
            print(f"  point {name:16s} on {bone}: {on_body(stock[name]['pos'], bone)} -> {on_body(pos, bone)} (forward, left, up)")
    return points, placed


def qc_line(name, bone, pos, rot):
    from .tf2_build import smd_euler
    rx, ry, rz = smd_euler(np.asarray(rot))                    # rot = Rz(rz) Ry(ry) Rx(rx); the QC wants ry rz rx, degrees
    return (f'$attachment "{name}" "{bone}" {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} '
            f'rotate {math.degrees(ry):.6f} {math.degrees(rz):.6f} {math.degrees(rx):.6f}')


def patch_qc(text, points):
    for name, (bone, pos, rot) in points.items():
        text, n = re.subn(r'(?m)^\$attachment "%s" .*$' % re.escape(name), lambda _m: qc_line(name, bone, pos, rot), text)
        if n != 1:
            raise SystemExit(f'expected exactly one $attachment "{name}" line in the stock QC, found {n}')
    return text


def check_built(built_info, points, stock=None):
    """The compiled file must carry the moved points as planned and every other point exactly as stock
    (stock: the stock file's points by name; default Ronin's body)."""
    stock = stock or stock_points()
    built = {a["name"]: a for a in built_info["attachments"]}
    if list(built) != list(stock):
        raise SystemExit("the compiled body's list of named points differs from stock")
    worst, drift = 0.0, 0.0
    for name, a in built.items():
        if name in points:
            bone, pos, rot = points[name]
            err = max(float(np.abs(np.array(a["pos"]) - pos).max()), float(np.abs(np.array(a["rot"]) - rot).max()) * 50)
            worst = max(worst, err)
            if a["bone"] != bone or err > 0.02:
                raise SystemExit(f"point {name} did not compile as planned (off by {err:.4f})")
        else:
            s = stock[name]
            # the compiler rebuilds each point's matrix from its angles; the last stored digit can differ
            d = max(float(np.abs(np.array(a["pos"]) - np.array(s["pos"])).max()),
                    float(np.abs(np.array(a["rot"]) - np.array(s["rot"])).max()))
            drift = max(drift, d)
            if a["bone"] != s["bone"] or a["flags"] != s["flags"] or d > 5e-5:
                raise SystemExit(f"point {name} changed (by {d:.6f}), and it is not one of the planned moves")
    print(f"  named points: {len(points)} moved as planned ({', '.join(points)}; largest deviation {worst:.4f} units), "
          f"the other {len(built) - len(points)} as stock (largest difference {drift:.6f})")


def surface(pos, tri, nrm, spacing=0.4, cap=80):
    """Points spread evenly over every triangle, at most `spacing` apart, with the triangle's outward normal."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    e = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)])
    n = np.clip(np.ceil(e / spacing).astype(int), 1, cap)
    gn = np.cross(b - a, c - a)
    sgn = np.sign((gn * nrm[tri].sum(1)).sum(1))               # the shading normals say which side is out
    sgn[sgn == 0] = 1
    gn = gn * sgn[:, None]
    gn /= np.maximum(np.linalg.norm(gn, axis=1, keepdims=True), 1e-12)
    P, N = [], []
    for k in np.unique(n):
        m = np.nonzero(n == k)[0]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        ok = i + j <= k
        u, v = i[ok] / k, j[ok] / k
        w = 1 - u - v
        pts = a[m][:, None, :] * w[None, :, None] + b[m][:, None, :] * u[None, :, None] + c[m][:, None, :] * v[None, :, None]
        P.append(pts.reshape(-1, 3))
        N.append(np.repeat(gn[m], len(u), axis=0))
    return np.vstack(P), np.vstack(N)


def preview(px=5.0, extra=None, tag="stowed"):
    """build/renders/retarget/<unit>_stowed.png - the body with its arms hanging and the sword where the game
    will draw it when it is stowed, the holders in a lighter blue.  Solid surfaces, nearest in front.
    extra: more [(points, normals, rgb)] to draw (the carried gun, for instance)."""
    from PIL import Image, ImageDraw, ImageFont
    from . import preview as pv, tf2_sword
    _points, (R_T, t_T) = plan(quiet=True)
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    fit = np.load(OUT_DIR / config.UNIT / "fit.npz")
    clouds = []
    for part, (pos, nrm) in pv.arms_down(fit, sk["bones"], 90.0).items():
        P, N = surface(pos.astype(np.float64), fit[f"{part}.tri"], nrm.astype(np.float64))
        clouds.append((P, N, (150, 150, 235) if "mount" in part else (70, 80, 190)))
    sw = tf2_sword.fitted(0)
    clouds.append(surface(sw["pos"] @ R_T.T + t_T, sw["tri"], sw["nrm"] @ R_T.T) + ((235, 235, 235),))
    clouds += list(extra or [])
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    up = np.array([0.0, 0.0, 1.0])
    P = np.vstack([c[0] for c in clouds]); N = np.vstack([c[1] for c in clouds])
    C = np.vstack([np.tile(np.array(c[2], dtype=np.float64), (len(c[0]), 1)) for c in clouds])

    def tile(toward, label, lowest=70.0):
        toward = np.asarray(toward, dtype=np.float64)
        toward /= np.linalg.norm(toward)
        right = np.cross(up, toward) if abs(toward[2]) < 0.99 else np.array([0.0, -1.0, 0.0])
        right /= np.linalg.norm(right)
        vup = np.cross(toward, right)
        keep = ((N @ toward) > 0.0) & (P[:, 2] > lowest)
        p, n, c = P[keep], N[keep], C[keep]
        light = toward * 0.7 + up * 0.5 + right * 0.35
        light /= np.linalg.norm(light)
        col = np.clip(c * (np.clip(n @ light, 0, 1) * 0.7 + 0.3)[:, None], 0, 255).astype(np.uint8)
        xs, ys = p @ right, p @ vup
        x0, y0 = xs.min() - 3, ys.min() - 3
        W, H = int((xs.max() + 3 - x0) * px), int((ys.max() + 3 - y0) * px) + 26
        X, Y = ((xs - x0) * px).astype(int), (H - 1 - (ys - y0) * px).astype(int)
        order = np.argsort(p @ toward)
        img = np.full((H, W, 3), (38, 40, 46), np.uint8)
        for dx in range(3):
            for dy in range(3):
                img[np.clip(Y[order] + dy, 0, H - 1), np.clip(X[order] + dx, 0, W - 1)] = col[order]
        pil = Image.fromarray(img)
        ImageDraw.Draw(pil).text((6, 3), label, fill=(240, 240, 240), font=font)
        return pil

    rows = [[tile((-1.0, 0, 0), "back"), tile((0.0, -1, 0), "right side"), tile((-0.75, -0.55, 0.37), "from behind-right, above")],
            [tile((0.6, 0.75, 0.25), "from front-left"), tile((1.0, 0.0, 0.0), "front"), tile((-0.6, 0.75, 0.25), "from behind-left")]]
    W = max(sum(t.width for t in r) + 8 * len(r) for r in rows)
    sheet = Image.new("RGB", (W, sum(max(t.height for t in r) for r in rows) + 8 * len(rows)), (20, 20, 24))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y)); x += t.width + 8
        y += max(t.height for t in r) + 8
    dst = config.RENDER_DIR / "retarget"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{config.UNIT}_{tag}.png"
    sheet.save(out)
    print(f"wrote {out}  {sheet.size}")
    return out


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()
    else:
        plan()
