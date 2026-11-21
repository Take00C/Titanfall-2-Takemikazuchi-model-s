"""Quick look at the retargeted mesh without starting a renderer.

A point-splat picture drawn with numpy inside the current process: every vertex, triangle
centre and edge midpoint is projected orthographically, coloured from the part's base-colour
texture, shaded by its normal, and painted far-to-near.  Ronin's bones are drawn on top.
Good enough to see whether limbs sit on the right joints; not a beauty render.

    python -m tsfronin.preview [unit]     -> build/renders/retarget/<unit>_fit.png
"""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .retarget import OUT_DIR, RONIN_SKELETON

VIEWS = {            # name: (screen-right axis, depth axis pointing at the viewer)
    "front": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    "left side": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "back": (np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
}


def arms_down(fit, ron, degrees=75.0):
    """Pose check: swing both arms down about Ronin's shoulder joints, skinning the way the game does
    (each vertex follows its bones by weight).  -> {part: (positions, normals)}"""
    parent = {b["name"]: b["parent_name"] for b in ron}
    at = {b["name"]: np.array(b["pos"], dtype=np.float64) for b in ron}

    def under(bone, top):
        while bone is not None:
            if bone == top:
                return True
            bone = parent[bone]
        return False

    moves = []
    for b in ron:
        R, pivot = np.eye(3), np.zeros(3)
        for side, sign in (("l", -1.0), ("r", 1.0)):
            if under(b["name"], f"def_{side}_shoulder"):
                a = np.radians(sign * degrees)
                c, s = np.cos(a), np.sin(a)
                R, pivot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), at[f"def_{side}_shoulder"]
        moves.append((R, pivot))
    out = {}
    for key in fit.files:
        part, kind = key.rsplit(".", 1)
        if kind != "pos":
            continue
        p, n = fit[key].astype(np.float64), fit[f"{part}.nrm"].astype(np.float64)
        bone, bw = fit[f"{part}.bone"], fit[f"{part}.bw"]
        p2, n2 = np.zeros_like(p), np.zeros_like(n)
        for k in range(bone.shape[1]):
            for bi in np.unique(bone[:, k]):
                m = bone[:, k] == bi
                R, pivot = moves[bi]
                w = bw[m, k][:, None]
                p2[m] += w * ((p[m] - pivot) @ R.T + pivot)
                n2[m] += w * (n[m] @ R.T)
        out[part] = (p2.astype(np.float32), n2.astype(np.float32))
    return out


def _samples(fit, clean, man, posed=None):
    """points, normals, colours for every part"""
    P, N, C = [], [], []
    glow = man.get("glow") or {}
    e = glow.get("emissive_linear") or [1.0, 0.0, 1.0]
    glow_rgb = np.array([12.92 * x if x <= 0.0031308 else 1.055 * min(x, 1.0) ** (1 / 2.4) - 0.055 for x in e]) * 255
    for part, info in man["lods"]["0"]["parts"].items():
        pos, nrm, uv = fit[f"{part}.pos"], fit[f"{part}.nrm"], fit[f"{part}.uv"]
        if posed:
            pos, nrm = posed[part]
        tri, mat = fit[f"{part}.tri"], fit[f"{part}.mat"]
        tex = man["texture_sets"].get(info["texset"], {}).get("bc")
        if tex:
            im = np.asarray(Image.open(clean / tex["file"]).convert("RGB"), dtype=np.float32)
        else:
            im = np.full((2, 2, 3), 128, np.float32)
        h, w = im.shape[:2]

        def colour(u):
            x = np.clip((u[:, 0] % 1.0) * w, 0, w - 1).astype(int)
            y = np.clip((1.0 - (u[:, 1] % 1.0)) * h, 0, h - 1).astype(int)
            return im[y, x]

        # triangle corners, centres and edge midpoints
        a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
        ua, ub, uc = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
        tn = nrm[tri].mean(axis=1)
        for q, uq in ((a, ua), (b, ub), (c, uc), ((a + b + c) / 3, (ua + ub + uc) / 3), ((a + b) / 2, (ua + ub) / 2),
                      ((b + c) / 2, (ub + uc) / 2), ((c + a) / 2, (uc + ua) / 2)):
            col = colour(uq)
            col[mat == 1] = glow_rgb
            P.append(q); N.append(tn); C.append(col)
    return np.vstack(P), np.vstack(N), np.vstack(C)


def original_as_fit(unit, scale):
    """The cleaned Project MIKHAIL model itself, untouched except for the scale, in the same layout as a fit
    (its arms hang as it was built).  The yardstick every fit is compared with."""
    clean = config.CLEAN_DIR / unit
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(clean / "lod0.npz")
    out = {}
    for part in man["lods"]["0"]["parts"]:
        out[f"{part}.pos"] = (z[f"{part}.pos"].astype(np.float64) * scale).astype(np.float32)
        for k in ("nrm", "uv", "tri", "mat"):
            out[f"{part}.{k}"] = z[f"{part}.{k}"]
    return out


def compare(unit, columns, tag, px_per_unit=4.0, degrees=90.0):
    """Side-by-side sheets at one scale.  columns: [(label, fit tag or "original" or None for the real fit)].
    Fits are drawn with the arms swung down about Ronin's shoulder joints, the way the game holds them.
    -> build/renders/retarget/<unit>_<tag>_front.png and _side.png"""
    clean = config.CLEAN_DIR / unit
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    sk = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))
    ron = sk["bones"]
    scale = json.loads((OUT_DIR / unit / "report.json").read_text(encoding="utf-8"))["scale_units_per_metre"]
    clouds = []
    for label, which in columns:
        if which == "ronin":                                  # the stock Ronin mesh, plain grey, for size
            from .tf2 import mdl53
            from .tf2_extract import RONIN_BODY, STOCK_DIR
            _info, h, d = mdl53.parse(str(STOCK_DIR / RONIN_BODY.replace("/", "\\")))
            verts, _ = mdl53.vvd_vertices(d, h)
            P = (np.array(sk["model_to_frame"]) @ verts["pos"].astype(np.float64).T).T
            P = P[np.abs(P[:, 1]) < 95.0]                     # its bind pose has the arms straight out; cut them short
            N, C = None, np.full((len(P), 3), 165.0, np.float32)
        elif which == "original":
            P, N, C = _samples(original_as_fit(unit, scale), clean, man)
        else:
            fit = np.load(OUT_DIR / (f"{unit}__{which}" if which else unit) / "fit.npz")
            P, N, C = _samples(fit, clean, man, arms_down(fit, ron, degrees))
        P = P.astype(np.float64).copy()
        P[:, 0] -= np.median(P[:, 0])                         # centre each figure the same way
        P[:, 1] -= (P[:, 1].min() + P[:, 1].max()) / 2
        if which != "ronin":
            P[:, 2] -= P[:, 2].min()
        if N is not None:
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        clouds.append((label, P, N, C))
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    up = np.array([0.0, 0.0, 1.0])
    top = max(c[1][:, 2].max() for c in clouds)
    H = int(top * px_per_unit) + 90
    outs = []
    for view in ("front", "left side"):
        right, toward = VIEWS[view]
        half = max(np.abs(c[1] @ right).max() for c in clouds) + 8
        W = int(2 * half * px_per_unit)
        tiles = []
        for label, P, N, C in clouds:
            img = np.full((H, W, 3), (38, 40, 46), np.uint8)
            x = ((P @ right + half) * px_per_unit).astype(int)
            y = (H - 30 - P[:, 2] * px_per_unit).astype(int)
            if N is None:
                col, keep = C.astype(np.uint8), np.ones(len(P), bool)
            else:
                light = toward * 0.75 + up * 0.5 + right * 0.3
                light /= np.linalg.norm(light)
                shade = np.clip(N @ light, 0, 1) * 0.75 + 0.3
                col = np.clip(255.0 * (C / 255.0) ** 0.55 * shade[:, None], 0, 255).astype(np.uint8)
                keep = (N @ toward) > -0.2
            order = np.argsort((P @ toward)[keep])
            xs, ys, cs = x[keep][order], y[keep][order], col[keep][order]
            pil = Image.fromarray(img)
            d = ImageDraw.Draw(pil)
            for zz in range(0, int(top) + 1, 25):               # the same height lines behind every figure
                yy = H - 30 - zz * px_per_unit
                d.line([(0, yy), (W, yy)], fill=(58, 60, 68) if zz % 50 else (92, 95, 106))
                if zz % 50 == 0:
                    d.text((4, yy - 24), str(zz), fill=(150, 150, 160), font=font)
            img = np.asarray(pil).copy()
            for dx in (0, 1, 2):
                for dy in (0, 1, 2):
                    img[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = cs
            pil = Image.fromarray(img)
            d = ImageDraw.Draw(pil)
            tip = H - 30 - P[:, 2].max() * px_per_unit           # a tick at this figure's own top
            d.line([(W - 70, tip), (W - 6, tip)], fill=(255, 200, 60), width=2)
            d.text((W - 150, tip - 26), f"top {P[:, 2].max():.0f}", fill=(255, 200, 60), font=font)
            d.text((60, 8), f"{label} - {view}", fill=(240, 240, 240), font=font)
            tiles.append(pil)
        sheet = Image.new("RGB", (sum(t.width for t in tiles), H), (38, 40, 46))
        xo = 0
        for t in tiles:
            sheet.paste(t, (xo, 0)); xo += t.width
        dst = config.RENDER_DIR / "retarget"
        dst.mkdir(parents=True, exist_ok=True)
        out = dst / f"{unit}_{tag}_{view.split()[-1]}.png"
        sheet.save(out)
        outs.append(out)
        print(f"wrote {out}  {sheet.size}")
    return outs


def render(unit=None, height_px=1000, pose=None, tag=None):
    """pose=None draws the bind pose (arms out); pose="arms_down" swings the arms down first."""
    unit = unit or config.UNIT
    clean = config.CLEAN_DIR / unit
    man = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    fit = np.load(OUT_DIR / unit / "fit.npz")
    ron = json.loads(RONIN_SKELETON.read_text(encoding="utf-8"))["bones"]
    if pose not in (None, "arms_down"):
        raise SystemExit(f"unknown pose {pose!r}")
    P, N, C = _samples(fit, clean, man, arms_down(fit, ron) if pose == "arms_down" else None)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    lo, hi = P.min(0), P.max(0)
    scale = (height_px - 60) / (hi[2] - min(lo[2], 0.0))
    up = np.array([0.0, 0.0, 1.0])
    tiles = []
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for name, (right, toward) in VIEWS.items():
        xs = P @ right
        width_px = int((xs.max() - xs.min()) * scale) + 60
        img = np.full((height_px, width_px, 3), (38, 40, 46), np.uint8)
        sx = ((xs - xs.min()) * scale + 30).astype(int)
        sy = (height_px - 30 - (P[:, 2] - min(lo[2], 0.0)) * scale).astype(int)
        light = toward * 0.75 + up * 0.5 + right * 0.3
        light /= np.linalg.norm(light)
        facing = N @ toward
        shade = np.clip(N @ light, 0, 1) * 0.75 + 0.3
        # the paint is dark; lift it so shapes read in a small picture (this is a fit check, not a colour check)
        col = np.clip(255.0 * (C / 255.0) ** 0.55 * shade[:, None], 0, 255).astype(np.uint8)
        keep = facing > -0.2                                   # drop surfaces that face away
        order = np.argsort((P @ toward)[keep])                 # far first, near last
        x, y, c = sx[keep][order], sy[keep][order], col[keep][order]
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                xx, yy = np.clip(x + dx, 0, width_px - 1), np.clip(y + dy, 0, height_px - 1)
                img[yy, xx] = c
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        # Ronin's bones that carry stock vertices, as a wire skeleton
        pts = {b["name"]: np.array(b["pos"]) for b in ron}
        def to_px(p):
            return (float((p @ right - xs.min()) * scale + 30), float(height_px - 30 - (p[2] - min(lo[2], 0.0)) * scale))
        for b in ron:
            if pose:                                           # the bones are only drawn for the bind pose
                break
            if b["parent_name"] and (b["stock_vertices"] or b["name"].startswith("def_")) and b["parent_name"] in pts:
                if pts[b["parent_name"]].any() or b["parent_name"] == "def_c_hip":
                    d.line([to_px(pts[b["parent_name"]]), to_px(pts[b["name"]])], fill=(255, 60, 60), width=2)
        ground = height_px - 30 + min(lo[2], 0.0) * scale
        d.line([(0, ground), (width_px, ground)], fill=(110, 110, 120), width=1)
        d.text((10, 6), name, fill=(235, 235, 235), font=font)
        tiles.append(pil)
    sheet = Image.new("RGB", (sum(t.width for t in tiles), height_px), (38, 40, 46))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0)); x += t.width
    dst = config.RENDER_DIR / "retarget"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{unit}_fit{'_' + pose if pose else ''}{'_' + tag if tag else ''}.png"
    sheet.save(out)
    print(f"wrote {out}  {sheet.size}  ({len(P):,} samples"
          + ("; arms swung down about Ronin's shoulder joints)" if pose else "; red lines = Ronin's bones in the bind pose)"))
    return out


if __name__ == "__main__":
    import sys
    render(sys.argv[1] if len(sys.argv) > 1 else None)
