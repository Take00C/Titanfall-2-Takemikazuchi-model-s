"""Experiment E0 - does a loose .mdl at Ronin's stock path override the game's copy?

Takes the stock Ronin body, moves only the vertices that ride the hatch bone straight up by
a fixed distance (an in-place edit of the embedded vertex buffer - no compiler involved, every
other byte identical to stock), and stages it as a throwaway Northstar mod:

    build/mods/TSFRonin.OverrideProbe/
        mod.json
        mod/models/titans/light/titan_light_locust.mdl

If Ronin shows up in game with its hatch floating above its shoulders, three things are proven
on this exact build: the loose-file override reaches the model cache, the stock skeleton and
animations drive edited vertices, and the engine does not hash vertex content.

    python -m tsfronin.probe_override            stage only
    python -m tsfronin.probe_override --install  stage and copy into R2Northstar/mods
    python -m tsfronin.probe_override --uninstall
"""
import argparse

import numpy as np

from . import ingame
from .tf2 import mdl53
from .tf2_extract import RONIN_BODY, RONIN_PRIME, STOCK_DIR

MOD = "TSFRonin.OverrideProbe"
BONE = "def_embark_hatch"
LIFT = 45.0            # game units (inches); about a fifth of Ronin's height, unmistakable in a menu


def build():
    """Both Ronin bodies get the same edit: the regular one and Ronin Prime are separate files
    (same skeleton), and the game shows whichever the player has chosen."""
    mod = ingame.stage_mod(MOD, "Test only: stock Ronin and Ronin Prime with the hatch raised, "
                                "to prove loose-file model override")
    for game_path in (RONIN_BODY, RONIN_PRIME):
        src = STOCK_DIR / game_path.replace("/", "\\")
        if not src.exists():
            raise SystemExit(f"{src} missing - run `python -m tsfronin.tf2_extract` first")
        data = bytearray(src.read_bytes())
        info, h, _ = mdl53.parse(str(src))
        bones = [b["name"] for b in info["bones"]]
        bi = bones.index(BONE)
        verts, _start = mdl53.vvd_vertices(data, h)
        # dominant bone of each vertex = the influence with the largest weight
        k = np.argmax(np.where(np.arange(3)[None, :] < verts["numbones"][:, None], verts["weight"], -1.0), axis=1)
        dominant = verts["bone"][np.arange(len(verts)), k]
        sel = dominant == bi
        # model space is Y-up (see build/tf2/ronin/skeleton.json: forward +Z, left +X, up +Y)
        verts["pos"][sel, 1] += LIFT
        dst = mod / "mod" / game_path.replace("/", "\\")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        changed = int(np.count_nonzero(np.frombuffer(src.read_bytes(), np.uint8) != np.frombuffer(data, np.uint8)))
        print(f"  {game_path}: {int(sel.sum()):,} of {len(verts):,} vertices on {BONE} raised {LIFT:g} units; "
              f"{changed:,} of {len(data):,} bytes differ from stock, size unchanged")
    return mod


REFERENCE = "models/tsfronin_test/stock_ronin.mdl"      # an untouched copy of stock Ronin under a new path
IDLE = "CQB_Idle_MP"                                    # a standing idle from titan_light_mp_core


def run():
    """Launch the game once: untouched stock Ronin on the left (mounted under a new path),
    Ronin's real path on the right.  If the override works, only the right one has a raised hatch."""
    build()
    stock = STOCK_DIR / RONIN_BODY.replace("/", "\\")
    r = ingame.run("e0_override_probe", [MOD],
                   spawns=[{"model": REFERENCE, "sequence": IDLE, "dist": 560, "side": -190},
                           {"model": RONIN_BODY, "sequence": IDLE, "dist": 560, "side": 190}],
                   extra_files={REFERENCE: stock})
    print(f"\nrun finished in {r.get('seconds')}s; failures: {r['failures'] or 'none'}")
    for m in r.get("markers", []):
        print("   ", m)
    for s in r["screenshots"]:
        print("  screenshot:", s)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--run", action="store_true", help="install, launch the game once, screenshot, clean up")
    a = ap.parse_args()
    if a.uninstall:
        ingame.uninstall(MOD)
        return
    if a.run:
        run()
        return
    build()
    if a.install:
        ingame.install(MOD)


if __name__ == "__main__":
    main()
