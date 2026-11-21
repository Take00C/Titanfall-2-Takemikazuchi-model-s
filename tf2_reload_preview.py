"""Check pictures for the Type-87's first-person reload, drawn without a renderer.

    build/renders/retarget/reload_pilot_view.png   frames of the full reload from the pilot's eye
    build/renders/retarget/reload_outside.png      the magazine change seen from above and behind the left hand

The gun (grey; the magazine lighter), the cockpit arms (green) posed by the authored animation; the magazine is
left out while the game hides it (frames 10-30).

    python -m tsfronin.tf2_reload_preview
"""
import numpy as np

from . import config, tf2_arms, tf2_gun, tf2_reload
from .smd_anim import Anim
from .tf2 import mdl53
from .tf2_arms_preview import _camera, _fill, _place, _tints

PILOT_FRAMES = [0, 3, 6, 9, 12, 15, 18, 21, 31, 34, 37, 40, 42, 45, 54, 70]
OUTSIDE_FRAMES = [6, 9, 12, 16, 19, 34, 40, 44]
GRIP_FRAMES = [6, 8, 10, 14, 34, 38, 42, 44, 45, 46, 48, 50]


def scene():
    m = tf2_gun.fitted(0)
    mask = tf2_gun.magazine_mask(m)
    pos, tri = np.asarray(m["pos"], dtype=np.float64), np.asarray(m["tri"], dtype=np.int64)
    _c, _t, b3, _p = tf2_reload.magazine_geometry()
    ident = np.eye(3).tolist()
    gun_bind = {"def_c_base": {"bind_rot": ident, "bind_pos": [0, 0, 0]},
                tf2_reload.MAGAZINE_BONE(): {"bind_rot": ident, "bind_pos": b3.tolist()}}
    body = (pos, np.full(len(pos), "def_c_base"), tri[~mask], np.tile([[118, 120, 128]], (int((~mask).sum()), 1)).astype(np.float32))
    mag = (pos, np.full(len(pos), tf2_reload.MAGAZINE_BONE()), tri[mask], np.tile([[190, 175, 120]], (int(mask.sum()), 1)).astype(np.float32))
    _i, ha, da = mdl53.parse(str(tf2_arms.stock(tf2_arms.ARMS[0])))
    pov = mdl53.read_bones(da, ha)
    arm_bind = {b["name"]: b for b in pov}
    arms = tf2_arms.body_arms()
    apos, _n, _x = tf2_arms.onto_first_person(arms, pov, quiet=True)
    bones, _l = tf2_arms.finger_joints(arms["bone"], apos, pov, quiet=True)
    arm = (apos, bones, arms["tri"], _tints(arms["bone"][arms["tri"][:, 0]], (110, 225, 170)))
    return gun_bind, arm_bind, body, mag, arm


def draw(width=560):
    from PIL import Image, ImageDraw, ImageFont
    a = Anim(str(tf2_reload.DEC / "tsf" / "anim_at_reload_seq.smd"))
    gun_bind, arm_bind, body, mag, arm = scene()
    qc = tf2_reload.DEC / f"{tf2_reload.NAME}.qc"
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    height = int(width * 9 / 16)
    hide, show = tf2_reload.hidden_frames()
    out = []
    centre, top_mid, b3, _p = tf2_reload.magazine_geometry()
    for sheet_name, frames, view in (("reload_pilot_view.png", PILOT_FRAMES, "pilot"),
                                     ("reload_outside.png", OUTSIDE_FRAMES, "outside"),
                                     ("reload_grip_pilot.png", GRIP_FRAMES, "grip_pilot"),
                                     ("reload_grip_far_side.png", GRIP_FRAMES, "grip_far")):
        tiles = []
        for f in frames:
            Wi = a.world(f)
            W = {a.names[i]: M for i, M in Wi.items()}
            eye, fwd, left, up = _camera(qc, W)
            fov = 70.0
            if view == "outside":
                target = W["def_l_wrist"][:3, 3]
                eye = target - 120.0 * fwd + 80.0 * up + 60.0 * left
                fov = 55.0
            elif view.startswith("grip"):
                # close on the hand and the top of the magazine: from the pilot's eye, or from the gun's far side
                Wm = W[tf2_reload.MAGAZINE_BONE()] @ tf2_reload.T(-b3)
                target = 0.5 * (W["def_l_wrist"][:3, 3] + (Wm @ np.append(top_mid, 1))[:3])
                if view == "grip_far":
                    gx, gy, gz = (W["def_c_base"][:3, k] for k in range(3))
                    eye = target - 70.0 * gx + 30.0 * gy - 20.0 * gz
                    up = gy
                fov = 32.0 if view == "grip_pilot" else 40.0
            if view != "pilot":
                fwd = (target - eye) / np.linalg.norm(target - eye)
                left = np.cross(up, fwd)
                left /= np.linalg.norm(left)
                up = np.cross(fwd, left)
            meshes = []
            for P, B, Tr, C, bind in ((body[0], body[1], body[2], body[3], gun_bind),
                                      (mag[0], mag[1], mag[2], mag[3], gun_bind),
                                      (arm[0], arm[1], arm[2], arm[3], arm_bind)):
                if P is mag[0] and B is mag[1] and hide <= f < show:
                    continue                                  # the game hides the magazine
                meshes.append((_place(P, B, bind, W), Tr, C))
            img = Image.fromarray(_fill(meshes, eye, fwd, left, up, fov, width, height))
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), f"frame {f}", fill=(240, 240, 240), font=font)
            tiles.append(img)
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (width + 6), rows * (height + 6)), (15, 15, 18))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * (width + 6), (k // cols) * (height + 6)))
        dst = config.RENDER_DIR / "retarget" / sheet_name
        sheet.save(dst)
        print("wrote", dst, sheet.size)
        out.append(dst)
    return out


def where(frames=None, stock=False):
    """Where things are in the pilot's view, frame by frame: (degrees to the right, degrees up, distance) from the
    eye along the camera's axes, for the gun's grip, the top of the magazine, the left wrist and elbow."""
    from .smd_anim import Anim as A
    fn = "anim_at_reload_seq.smd"
    a = A(str(tf2_reload.DEC / fn)) if stock else A(str(tf2_reload.DEC / "tsf" / fn))
    centre, top_mid, b3, _p = tf2_reload.magazine_geometry()
    qc = tf2_reload.DEC / f"{tf2_reload.NAME}.qc"
    print(f"  {'stock' if stock else 'authored'} reload: (right, up, distance) from the eye")
    for f in frames or range(0, 76, 3):
        Wi = a.world(f)
        W = {a.names[i]: M for i, M in Wi.items()}
        eye, fwd, left, up = _camera(qc, W)

        def view(p):
            v = np.asarray(p) - eye
            d = float(v @ fwd)
            return (round(float(np.degrees(np.arctan2(-(v @ left), d))), 1), round(float(np.degrees(np.arctan2(v @ up, d))), 1),
                    round(float(np.linalg.norm(v)), 1))
        Wm = W[tf2_reload.MAGAZINE_BONE()] @ tf2_reload.T(-b3)
        top = (Wm @ np.append(top_mid, 1))[:3]
        print(f"    {f:3d}: grip {view(W['def_c_base'][:3, 3])}  magazine top {view(top)}  "
              f"left wrist {view(W['def_l_wrist'][:3, 3])}  left elbow {view(W['def_l_elbow'][:3, 3])}  "
              f"right wrist {view(W['def_r_wrist'][:3, 3])}")


def grip_check(frames=None):
    """frame by frame, how many vertices of the left hand lie inside the magazine (while the game shows it), and
    how many lie within half a unit of it (the hand holding it)"""
    a = Anim(str(tf2_reload.DEC / "tsf" / "anim_at_reload_seq.smd"))
    solid = tf2_reload.MagazineSolid()
    mesh, _by = tf2_reload.left_hand_mesh()
    centre, _t, b3, _p = tf2_reload.magazine_geometry()
    hide, show = tf2_reload.hidden_frames()
    ix = a.index
    out = []
    for f in frames or range(0, 60):
        W = a.world(f)
        seat = np.linalg.inv(W[ix["def_c_base"]] @ tf2_reload.magazine_motion(f, centre))
        P = np.concatenate([tf2_reload._apply(seat @ W[ix[b]], V) for b, V in mesh.items()])
        shown = not (hide <= f < show)
        n_in, n_near = int(solid.inside(P, 0.0).sum()), int(solid.inside(P, 0.5).sum())
        out.append((f, shown, n_in, n_near))
    print("  left hand against the magazine (frame: inside / within 0.5; * = the game hides the magazine):")
    print("   " + "  ".join(f"{f}{'' if s else '*'}:{i}/{n}" for f, s, i, n in out))
    bad = [f for f, s, i, _n in out if s and i]
    print(f"  frames with the hand inside the shown magazine: {bad or 'none'}")
    return bad


if __name__ == "__main__":
    draw()
