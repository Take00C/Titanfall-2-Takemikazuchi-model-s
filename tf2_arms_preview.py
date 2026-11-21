"""Check pictures for the first-person arms, drawn without a renderer (a small depth-buffered triangle filler).

    build/renders/retarget/cockpit_arms.png

Two held poses taken from the game's own first-person animations (the gun at the hip; the sword at rest), each
seen from the pilot's eye (the file's CAMERA point) and again magnified on one hand.  Three columns: Ronin's
stock arms as the mod shows them today, the new arms with every finger piece on the joint the body fit uses,
and the new arms with each finger piece on the joint at its own knuckle (data/pov_arms_map.json, finger_joints).
Neighbouring pieces are tinted a little differently so the joints can be seen.

    python -m tsfronin.tf2_arms_preview
"""
import json
import re
import struct

import numpy as np

from . import config, tf2_arms, tf2_gun, tf2_sword
from .tf2 import mdl53
from .tf2_build import _euler_matrix
from .tf2_extract import STOCK_DIR


def stock_triangles(d, h):
    """triangles of a one-mesh, one-level file as indices into its vertex table (from the embedded draw lists)"""
    o = h["vtxOffset"]
    nbp, bpo = struct.unpack_from("<ii", d, o + 28)
    out = []
    for bp in range(nbp):
        b0 = o + bpo + bp * 8
        nmodels, mo = struct.unpack_from("<ii", d, b0)
        for m in range(nmodels):
            m0 = b0 + mo + m * 8
            nl, lo = struct.unpack_from("<ii", d, m0)
            if not nl:
                continue
            l0 = m0 + lo                                       # level 0
            nmesh, meo, _sw = struct.unpack_from("<iif", d, l0)
            for me in range(nmesh):
                e0 = l0 + meo + me * 9
                nsg, sgo, _f = struct.unpack_from("<iiB", d, e0)
                for g in range(nsg):
                    g0 = e0 + sgo + g * 33
                    nv, vo, ni, io = struct.unpack_from("<4i", d, g0)
                    vid = np.frombuffer(d, dtype=np.dtype([("w", "u1", 3), ("n", "u1"), ("id", "<u2"), ("b", "i1", 3)]),
                                        count=nv, offset=g0 + vo)["id"].astype(np.int64)
                    idx = np.frombuffer(d, dtype="<u2", count=ni, offset=g0 + io).astype(np.int64)
                    out.append(vid[idx].reshape(-1, 3))
    return np.vstack(out)


def _place(P, B, bind, W):
    """rigid pieces from their bind pose onto the posed bones"""
    out = np.zeros_like(P)
    for b in set(B):
        sel = B == b
        Rb, tb = np.array(bind[b]["bind_rot"]), np.array(bind[b]["bind_pos"])
        out[sel] = ((P[sel] - tb) @ Rb) @ W[b][:3, :3].T + W[b][:3, 3]
    return out


def _camera(qc_path, W):
    """the eye and its axes from the recipe's CAMERA point (`rotate a b c` = Rz(b) Ry(a) Rx(c))"""
    m = re.search(r'(?m)^\$attachment "CAMERA" "([^"]+)" (\S+) (\S+) (\S+) rotate (\S+) (\S+) (\S+)',
                  qc_path.read_text(encoding="utf-8"))
    off = np.array([float(m.group(k)) for k in (2, 3, 4)])
    a, b, c = (np.radians(float(m.group(k))) for k in (5, 6, 7))
    Wc = W[m.group(1)]
    C = Wc[:3, :3] @ _euler_matrix(c, a, b)
    return Wc[:3, 3] + Wc[:3, :3] @ off, C[:, 0], C[:, 1], C[:, 2]


def _fill(meshes, eye, fwd, left, up, fov, width, height):
    """meshes: [(posed vertices, triangles, colour per triangle)] -> image, flat shaded, nearest surface wins"""
    f = (width / 2) / np.tan(np.radians(fov) / 2)
    img = np.full((height, width, 3), (30, 32, 38), np.float32)
    zbuf = np.full((height, width), np.inf, np.float32)
    for P, tri, cols in meshes:
        v = P - eye
        depth = v @ fwd
        sx = width / 2 - f * (v @ left) / np.maximum(depth, 1e-6)
        sy = height / 2 - f * (v @ up) / np.maximum(depth, 1e-6)
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = np.abs(n @ fwd) * 0.65 + 0.35
        for t in np.nonzero((depth[tri] > 2.0).all(1))[0]:
            i0, i1, i2 = tri[t]
            x0, x1, x2, y0, y1, y2 = sx[i0], sx[i1], sx[i2], sy[i0], sy[i1], sy[i2]
            xa, xb = int(max(0, np.floor(min(x0, x1, x2)))), int(min(width - 1, np.ceil(max(x0, x1, x2))))
            ya, yb = int(max(0, np.floor(min(y0, y1, y2)))), int(min(height - 1, np.ceil(max(y0, y1, y2))))
            if xa > xb or ya > yb:
                continue
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
            w1 = ((gx - x0) * (y2 - y0) - (x2 - x0) * (gy - y0)) / area
            w2 = ((x1 - x0) * (gy - y0) - (gx - x0) * (y1 - y0)) / area
            w0 = 1 - w1 - w2
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * depth[i0] + w1 * depth[i1] + w2 * depth[i2]
            zb = zbuf[ya:yb + 1, xa:xb + 1]
            win = inside & (z < zb)
            zb[win] = z[win]
            img[ya:yb + 1, xa:xb + 1][win] = cols[t] * shade[t]
    return np.clip(img, 0, 255).astype(np.uint8)


def _tints(bones, base):
    """one colour per triangle: the base colour, lighter or darker from piece to piece"""
    names = sorted(set(bones))
    k = np.array([names.index(b) for b in bones])
    return np.array(base, dtype=np.float32)[None, :] * (0.72 + 0.14 * (k % 3))[:, None]


def draw(width=760):
    from PIL import Image, ImageDraw, ImageFont
    name = tf2_arms.ARMS[0]
    _i, ha, da = mdl53.parse(str(tf2_arms.stock(name)))
    arm_bones = mdl53.read_bones(da, ha)
    arm_bind = {b["name"]: b for b in arm_bones}
    sv, _ = mdl53.vvd_vertices(da, ha)
    s_pos, s_bone = sv["pos"].astype(np.float64), np.array([arm_bones[int(x)]["name"] for x in sv["bone"][:, 0]])
    s_tri = stock_triangles(da, ha)
    arms = tf2_arms.body_arms()
    pos, _nrm, _notes = tf2_arms.onto_first_person(arms, arm_bones, quiet=True)
    at_knuckle, _lines = tf2_arms.finger_joints(arms["bone"], pos, arm_bones, quiet=True)
    tri = arms["tri"]
    variants = [("Ronin's arms (in the game now)", s_pos, s_bone, s_tri, _tints(s_bone[s_tri[:, 0]], (215, 215, 220))),
                ("new arms, finger pieces on the joints the body uses", pos, arms["bone"], tri,
                 _tints(arms["bone"][tri[:, 0]], (135, 125, 255))),
                ("new arms, each finger piece on the joint at its knuckle", pos, at_knuckle, tri,
                 _tints(arms["bone"][tri[:, 0]], (110, 225, 170)))]

    scenes = []
    for label, folder, smd, stock_rel, mesh, bone, zoom in (
            ("gun held at the hip", "atpov_titan_triple_threat", "anim_at_ads_in_seq.smd",
             tf2_gun.GUN_DIR + "/atpov_titan_triple_threat.mdl", tf2_gun.fitted(0),
             json.loads(tf2_gun.GUN_MAP.read_text(encoding="utf-8"))["mesh_bone"]["atpov_titan_triple_threat"], "def_l_wrist"),
            ("sword held", "atpov_titan_sword", "anim_at_idle_seq.smd",
             tf2_sword.SWORD_DIR + "/atpov_titan_sword.mdl", tf2_sword.fitted(0),
             json.loads(tf2_sword.SWORD_MAP.read_text(encoding="utf-8"))["mesh_bone"]["atpov_titan_sword"], "def_r_wrist")):
        d = tf2_arms.DECOMPILED / folder
        names, parents, frames, _last = tf2_arms._smd_frames(d / smd, [0])
        W = tf2_arms._posed(names, parents, frames[0])
        _i, hw, dw = mdl53.parse(str(STOCK_DIR / stock_rel.replace("/", "\\")))
        bind = {b["name"]: b for b in mdl53.read_bones(dw, hw)}
        wp = np.asarray(mesh["pos"], dtype=np.float64)
        wt = np.asarray(mesh["tri"], dtype=np.int64)
        weapon = (_place(wp, np.full(len(wp), bone), bind, W), wt, np.tile(np.array([[105, 108, 116]], np.float32), (len(wt), 1)))
        scenes.append((label, W, weapon, _camera(d / f"{folder}.qc", W), zoom))

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    height = int(width * 9 / 16)
    rows = []
    for label, W, weapon, (eye, fwd, left, up), zoom in scenes:
        target = W[zoom][:3, 3]
        zf = (target - eye) / np.linalg.norm(target - eye)
        zl = np.cross(up, zf)
        zl /= np.linalg.norm(zl)
        zu = np.cross(zf, zl)
        side = "left" if "_l_" in zoom else "right"
        for view, (f_, l_, u_), fov in ((f"{label}: the pilot's view", (fwd, left, up), 80.0),
                                        (f"{label}: the {side} hand, magnified", (zf, zl, zu), 30.0)):
            tiles = []
            for title, P, B, T, cols in variants:
                pil = Image.fromarray(_fill([weapon, (_place(P, B, arm_bind, W), T, cols)], eye, f_, l_, u_, fov, width, height))
                dr = ImageDraw.Draw(pil)
                dr.text((8, 6), title, fill=(240, 240, 240), font=font)
                dr.text((8, height - 24), view, fill=(200, 200, 205), font=font)
                tiles.append(pil)
            rows.append(tiles)
            print(f"  drawn: {view}")
    sheet = Image.new("RGB", (3 * width + 16, len(rows) * (height + 8)), (15, 15, 18))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * (width + 8), r * (height + 8)))
    dst = config.RENDER_DIR / "retarget" / "cockpit_arms.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)
    print("wrote", dst, sheet.size)
    return dst


if __name__ == "__main__":
    draw()
