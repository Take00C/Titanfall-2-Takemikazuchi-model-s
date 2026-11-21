"""Stage 69 (optional) - the pilot is not drawn while boarding a Stryder-class titan (Ronin, Northstar).

The game shows boarding from an outside camera on the player's own pilot, who climbs into a hatch.  On the
Takemikazuchi nothing opens, so the pilot sinks into the chest plate.  A client script cannot hide the pilot (the
game refuses Hide and DisableDraw on anything the server owns; tried in 0.8.0 and 0.8.1), but the client animates
the pilot from its own model files.  The 28 boarding animations for Stryder-class titans live in one animation
file every pilot model includes, models/humans/pete/pete_scripted_r1.mdl (547 animations).  It is rebuilt with,
in those 28 only, every part of the pilot except the camera's own chain of bones (jx_c_delta > jx_c_pov >
jx_c_camera, which carries the 'VDU' point the boarding camera rides) moved far below the ground and shrunk to a
hundredth.  The view follows exactly the stock path; the body, the weapon in its hands and anything attached to
it are simply not there.  Every other animation must come out of the rebuild unchanged, frame for frame.

Side effects (told to the owner): on this player's screen every pilot boarding a Ronin or Northstar is hidden, not
only their own; in a match this PC hosts, its server reads the same file, so the pilot's hit boxes go with the
body for those two seconds (the game makes a boarding pilot invulnerable).

    python -m tsfronin.tf2_pilot            extract, decompile, author, compile, check  (about five minutes)
"""
import contextlib
import io
import json
import math
import re
import runpy
import sys
import time

import numpy as np

from . import config, tf2_sword, tools
from .smd_anim import Anim
from .tf2 import mdl53, vpk
from .tf2_build import compile_qc
from .tf2_extract import STOCK_DIR, VPK_INDEX

FILE = "models/humans/pete/pete_scripted_r1.mdl"
NAME = "pete_scripted_r1"
DEC = tf2_sword.DECOMPILED / NAME
OUT = config.BUILD / "tf2" / "tsf_pilot"
SEQUENCE_PREFIX = "pt_mount_stryder"
CAMERA_CHAIN = ("jx_c_delta", "jx_c_pov", "jx_c_camera")
DROP = 2000.0           # game units below
SHRINK = 0.01
POSSCALE = 0.08         # stored position step for the moved bones: range 32767 x 0.08 = 2621 units


def stock():
    return STOCK_DIR / FILE.replace("/", "\\")


def extract():
    if not stock().exists():
        vpk.extract(str(config.TITANFALL2 / "vpk" / VPK_INDEX), [FILE], str(STOCK_DIR), workers=1)
    return stock()


def decompile():
    qc = DEC / f"{NAME}.qc"
    if qc.exists() and qc.stat().st_mtime >= stock().stat().st_mtime:
        return qc
    return tf2_sword.decompile_file(stock(), DEC)


def boarding_animations(text):
    """-> {animation name: SMD file} used by the Stryder-class boarding sequences"""
    seqs = re.findall(r'(?ms)^\$sequence "([^"]+)"\n\{(.*?)^\}', text)
    names = set()
    for label, body in seqs:
        if label.lower().startswith(SEQUENCE_PREFIX):
            names.update(re.findall(r'(?m)^\t"([^"]+)"', body))
    files = dict(re.findall(r'(?m)^\$animation "([^"]+)" "([^"]+)"', text))
    missing = [n for n in names if n not in files]
    if missing:
        raise SystemExit(f"{NAME}.qc: no $animation for {missing}")
    return {n: files[n] for n in sorted(names)}


def moved_bones(a):
    """the topmost bones that are not the camera's chain: moving these moves the whole pilot and its props"""
    return [i for i in sorted(a.names) if a.names[i] not in CAMERA_CHAIN
            and (a.parents[i] < 0 or a.names[a.parents[i]] in CAMERA_CHAIN)]


def author():
    text = decompile().read_text(encoding="utf-8")
    anims = boarding_animations(text)
    out_dir = DEC / "tsf"
    out_dir.mkdir(parents=True, exist_ok=True)
    tops_by_file = {}
    for name, fn in anims.items():
        a = Anim(str(DEC / fn))
        if any(n not in a.index for n in CAMERA_CHAIN):
            raise SystemExit(f"{fn}: the camera's chain of bones is not in this animation")
        tops = moved_bones(a)
        down = np.array([0.0, 0.0, -DROP])
        for f in range(a.nframes):
            W = a.world(f)
            for i in tops:
                p = a.parents[i]
                Rp = np.eye(3) if p < 0 else W[p][:3, :3]
                a.rows[f, i, 0:3] = a.rows[f, i, 0:3] + Rp.T @ down
                if a.rows.shape[2] >= 9:
                    a.rows[f, i, 6:9] = SHRINK
        if a.rows.shape[2] < 9:
            raise SystemExit(f"{fn}: no scale columns to shrink the pilot with")
        a.write(str(out_dir / fn))
        tops_by_file[fn] = [a.names[i] for i in tops]
    first = next(iter(tops_by_file.values()))
    print(f"  {len(anims)} boarding animations written; in each, moved {DROP:.0f} down and shrunk to {SHRINK}: {first}")
    return anims, tops_by_file


def patch_qc(anims, tops_by_file):
    text = (DEC / f"{NAME}.qc").read_text(encoding="utf-8")
    for name, fn in anims.items():
        pat = re.compile(r'(?m)^(\$animation "%s" )"%s"(.*)$' % (re.escape(name), re.escape(fn)))
        m = pat.search(text)
        if not m:
            raise SystemExit(f"{NAME}.qc: cannot find the $animation line for {name}")
        rest = m.group(2)
        for b in tops_by_file[fn]:
            rest = re.sub(r' posscale "%s" \S+' % re.escape(b), "", rest)
            rest += f' posscale "{b}" {POSSCALE}'
        text = text[:m.start()] + f'{m.group(1)}"tsf/{fn}"{rest}' + text[m.end():]
    from .tf2_build import restore_hit_data_groups
    text, n_hit = restore_hit_data_groups(text, stock())          # the decompiler drops these; put them back
    print(f"  {NAME}.qc: hit-data groups of {n_hit} hit boxes restored from the stock file")
    dst = DEC / f"{NAME}_tsf.qc"
    dst.write_text(text, encoding="utf-8")
    print(f"  {NAME}.qc: {len(anims)} animations repointed to the authored ones; everything else stock text")
    return dst


def check(built):
    """Every table but the animations equal to stock; the 519 other animations equal frame for frame; in the 28
    boarding ones the camera's chain exactly as stock and the rest of the pilot far below and shrunk."""
    with contextlib.redirect_stdout(io.StringIO()):
        v = runpy.run_path(str(tools.find("r2-model-compiler", "validate53.py")), run_name="validate53_as_library")
    a, _ha, _da = mdl53.parse(str(stock()))
    b, _hb, _db = mdl53.parse(str(built))
    same, differ = [], []
    for key in ("bones", "attachments", "hitboxsets", "ikchains", "includemodels", "keyvalues", "poseparams",
                "sequences", "nodes", "srcbonetransforms", "mass", "surfaceprop"):
        (same if a.get(key) == b.get(key) else differ).append(key)
    print(f"  {NAME}: identical to stock: {', '.join(same)}" + (f";  DIFFERENT: {', '.join(differ)}" if differ else ""))
    text = (DEC / f"{NAME}.qc").read_text(encoding="utf-8")
    authored = set(boarding_animations(text))
    seq_labels = {lbl for lbl, body in re.findall(r'(?ms)^\$sequence "([^"]+)"\n\{(.*?)^\}', text)
                  if lbl.lower().startswith(SEQUENCE_PREFIX)}
    for key in list(differ):
        x, y = a.get(key) or [], b.get(key) or []
        if len(x) != len(y):
            print(f"    {key}: {len(x)} entries in stock, {len(y)} rebuilt")
            continue
        rows = []
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                fields = {k: (p.get(k), q.get(k)) for k in p if p.get(k) != q.get(k)} if isinstance(p, dict) else (p, q)
                rows.append((i, p.get("label", p.get("name")) if isinstance(p, dict) else i, fields))
        expected = key == "sequences" and all(lbl in seq_labels and set(f) <= {"flags"} for _i, lbl, f in rows)
        print(f"    {key}: {len(rows)} entries differ" + (" - only the boarding sequences' flags (bone scale)" if expected else "")
              + ("" if expected else f": {rows[:4]}"))
        if expected:
            differ.remove(key)
    dc = v["dc"]
    da, db = stock().read_bytes(), built.read_bytes()
    ma, mb = dc.mdl53.Model(da), dc.mdl53.Model(db)
    ba, bb = ma.bones(), mb.bones()
    chain = [i for i, x in enumerate(bb) if x["name"] in CAMERA_CHAIN]
    worst = {"position": 0.0, "rotation_degrees": 0.0, "scale": 0.0}
    bad, mine, frames = [], 0, 0
    t0 = time.time()
    for ai in range(ma.hdr["numlocalanim"]):
        aa, name, fa = dc.decode_anim(da, ma, ba, ai, with_scale=True)
        ab, other, fb = dc.decode_anim(db, mb, bb, ai, with_scale=True)
        if name != other or aa["numframes"] != ab["numframes"] or aa["flags"] & ~0x20000 != ab["flags"] & ~0x20000:
            bad.append(f"{name}: name, length or flags differ")
            continue
        frames += aa["numframes"]
        delta = bool(aa["flags"] & 4)
        fma = dc.decode_frame_movement(da, ma, ai, range(aa["numframes"]))
        fmb = dc.decode_frame_movement(db, mb, ai, range(ab["numframes"]))
        if (fma is None) != (fmb is None) or (fma is not None and max(abs(x - y) for p, q in zip(fma, fmb) for x, y in zip(p, q)) > 1e-3):
            bad.append(f"{name}: root motion differs")
        if name in authored:
            mine += 1
            for k in range(aa["numframes"]):
                wa, wb = v["world_pose"](ba, fa[k]), v["world_pose"](bb, fb[k])
                for i in chain:
                    d = math.dist([r[3] for r in wa[i]], [r[3] for r in wb[i]])
                    if d > 0.05:
                        bad.append(f"{name} frame {k}: camera bone {bb[i]['name']} moved {d:.3f}")
                        break
                hip = [i for i, x in enumerate(bb) if x["name"] == "def_c_hip"][0]
                drop = wa[hip][2][3] - wb[hip][2][3]
                size = math.sqrt(sum(wb[hip][r][0] ** 2 for r in range(3)))
                if drop < 0.9 * DROP or size > 2 * SHRINK:
                    bad.append(f"{name} frame {k}: the pilot is only {drop:.0f} below / {size:.3f} of its size")
                    break
            continue
        m = v["accelerated_pose_metrics"](ba, bb, fa, fb, fma, fmb, delta)
        for key in worst:
            val = m[key][0] if key in m else 0.0
            worst[key] = max(worst[key], val)
        if m.get("position", (0,))[0] > v["POSITION_TOLERANCE"] or m.get("rotation_degrees", (0,))[0] > v["ROTATION_TOLERANCE"] \
                or m.get("scale", (0,))[0] > v["SCALE_TOLERANCE"]:
            bad.append(f"{name}: differs from stock {({k: round(m[k][0], 4) for k in m})}")
    ok = not differ and not bad
    print(f"  animations: {ma.hdr['numlocalanim']} ({frames:,} frames) in {time.time() - t0:.0f}s; the {ma.hdr['numlocalanim'] - mine} "
          f"not touched equal stock within {({k: round(x, 5) for k, x in worst.items()})}; the {mine} boarding ones: camera "
          f"chain as stock, pilot moved and shrunk" + ("" if not bad else f"; PROBLEMS: {bad[:6]} ({len(bad)})"))
    return ok


def build():
    if not config.HIDE_PILOT_ON_EMBARK:
        print("  HIDE_PILOT_ON_EMBARK is off: the pilot animation file is left as it is")
        return None
    extract()
    anims, tops = author()
    qc = patch_qc(anims, tops)
    out = compile_qc(qc, OUT / f"{NAME}.mdl")
    if not check(out):
        out.unlink()
        raise SystemExit(f"{NAME}: the rebuilt file differs from stock outside the boarding animations - removed, not staged")
    return out


if __name__ == "__main__":
    build()
