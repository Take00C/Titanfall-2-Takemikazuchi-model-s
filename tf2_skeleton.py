"""Stage 41 - what the stock Ronin body is made of  ->  build/tf2/ronin/

    structure.json   bones, attachments, hitbox sets, bodygroups, materials, IK chains, includes
    skeleton.json    the 106 bones with their bind pose, converted to the pipeline frame
                     (+X forward, +Y left, +Z up) and still in the game's units (inches)
    vertex_bones.json  how many stock vertices each bone carries, per body part

This is the reference the skin has to fit: it keeps every bone, attachment, hitbox and
animation include of this file and replaces only the mesh.

    python -m tsfronin.tf2_skeleton
"""
import json

import numpy as np

from . import config
from .tf2 import mdl53
from .tf2_extract import RONIN_BODY, STOCK_DIR

OUT_DIR = config.BUILD / "tf2" / "ronin"


def model_axes(pos):
    """Which model-space axis is up / forward, from the vertex cloud itself."""
    ext = pos.max(0) - pos.min(0)
    lo = pos.min(0)
    return ext, lo


def main():
    path = STOCK_DIR / RONIN_BODY.replace("/", "\\")
    if not path.exists():
        raise SystemExit(f"{path} missing - run the tf2-extract stage first")
    info, h, d = mdl53.parse(str(path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "structure.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    bones = mdl53.read_bones(d, h)
    verts, _ = mdl53.vvd_vertices(d, h)
    pos = verts["pos"].astype(np.float64)
    ext, lo = model_axes(pos)
    P = np.array([b["bind_pos"] for b in bones])
    by = {b["name"]: b for b in bones}

    # Model space -> pipeline frame.  Decide the axes from data, not from convention: up is the
    # axis along which the hatch sits above the ankles, forward is where the toes point, and
    # left is the side the def_l_* bones are on.
    up = np.array(by["def_embark_hatch"]["bind_pos"]) - np.array(by["def_l_ankle"]["bind_pos"])
    up_axis = int(np.argmax(np.abs(up)))
    fwd = np.array(by["def_l_ball"]["bind_pos"]) - np.array(by["def_l_ankle"]["bind_pos"])
    fwd[up_axis] = 0
    fwd_axis = int(np.argmax(np.abs(fwd)))
    left_axis = 3 - up_axis - fwd_axis
    M = np.zeros((3, 3))
    M[0, fwd_axis] = np.sign(fwd[fwd_axis])
    M[2, up_axis] = np.sign(up[up_axis])
    M[1, left_axis] = np.sign(by["def_l_clav"]["bind_pos"][left_axis] - by["def_r_clav"]["bind_pos"][left_axis])
    if abs(np.linalg.det(M) - 1) > 1e-6:
        raise SystemExit(f"model axes are not a rotation (det {np.linalg.det(M):+.1f}); check the bind pose")

    out_bones = []
    for b in bones:
        out_bones.append({"index": b["index"], "name": b["name"],
                          "parent": b["parent"], "parent_name": bones[b["parent"]]["name"] if b["parent"] >= 0 else None,
                          "pos": [round(float(x), 4) for x in M @ np.array(b["bind_pos"])],
                          "rot": [[round(float(x), 6) for x in row] for row in M @ np.array(b["bind_rot"])],
                          "model_pos": b["bind_pos"]})

    # which bones the stock mesh is actually skinned to
    nb = verts["numbones"]
    counts = np.zeros(len(bones), dtype=np.int64)
    for k in range(3):
        sel = nb > k
        w = verts["weight"][:, k][sel]
        np.add.at(counts, verts["bone"][:, k][sel][w > 0], 1)
    for b, c in zip(out_bones, counts.tolist()):
        b["stock_vertices"] = c

    axes = "XYZ"
    skeleton = {
        "source": RONIN_BODY, "units": "inches (1 unit = 2.54 cm)",
        "frame": "+X forward, +Y left, +Z up",
        "model_space": {"forward": ("+" if M[0, fwd_axis] > 0 else "-") + axes[fwd_axis],
                        "left": ("+" if M[1, left_axis] > 0 else "-") + axes[left_axis],
                        "up": ("+" if M[2, up_axis] > 0 else "-") + axes[up_axis]},
        "model_to_frame": M.tolist(),
        "mesh_bounds_model_space": {"min": lo.round(3).tolist(), "size": ext.round(3).tolist()},
        "bones": out_bones,
    }
    (OUT_DIR / "skeleton.json").write_text(json.dumps(skeleton, indent=1), encoding="utf-8")

    fp = (M @ pos.T).T
    print(f"{RONIN_BODY}")
    c = info["counts"]
    print(f"  {c['numbones']} bones, {c['numlocalattachments']} attachments, {c['numbodyparts']} bodygroups, "
          f"{c['numtextures']} materials, {c['numikchains']} IK chains, {c['numlocalseq']} local sequence, "
          f"{c['numincludemodels']} animation includes")
    print(f"  model space: forward {skeleton['model_space']['forward']}, left {skeleton['model_space']['left']}, "
          f"up {skeleton['model_space']['up']}")
    print(f"  stock mesh: {len(verts):,} vertices; size {np.ptp(fp[:, 0]):.1f} long x {np.ptp(fp[:, 1]):.1f} wide x "
          f"{np.ptp(fp[:, 2]):.1f} tall units ({np.ptp(fp[:, 2]) * 0.0254:.2f} m tall), feet at z={fp[:, 2].min():.1f}")
    skinned = [b for b in out_bones if b["stock_vertices"]]
    print(f"  {len(skinned)} of {len(out_bones)} bones carry stock vertices")
    print("  bodygroups:", ", ".join(f"{bp['name']}({len(bp['models'])})" for bp in info["bodyparts"]))
    print("  IK chains:", "; ".join(f"{k['name']}: {' > '.join(k['links'])}" for k in info["ikchains"]))
    print("  key joints (forward, left, up):")
    for n in ("def_c_hip", "def_c_spineC", "def_embark_hatch", "def_c_head", "def_l_clav", "def_l_shoulder",
              "def_l_elbow", "def_l_wrist", "def_l_thigh", "def_l_thighLow", "def_l_knee", "def_l_ankle", "def_l_ball"):
        b = next(x for x in out_bones if x["name"] == n)
        print(f"    {n:18s} ({b['pos'][0]:+7.1f}, {b['pos'][1]:+7.1f}, {b['pos'][2]:+7.1f})  verts {b['stock_vertices']:>6,}")
    print(f"  -> {OUT_DIR}")


if __name__ == "__main__":
    main()
