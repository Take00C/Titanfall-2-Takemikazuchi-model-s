"""Stage 40 - stock Ronin models out of the installed Titanfall 2  ->  build/tf2/stock/

Read-only on the game: the files are copied out of the VPK archives (LZHAM-decompressed,
each checked against the CRC32 in the VPK index) and nothing is written to the install.
The copies are the base the skin is built on - Ronin's skeleton, attachments, hitboxes and
animation includes are kept, only the visible mesh and materials are replaced.

    python -m tsfronin.tf2_extract
"""
import json
import os
import time

from . import config
from .tf2 import vpk

VPK_INDEX = "englishclient_mp_common.bsp.pak000_dir.vpk"

# Everything that makes up the multiplayer Ronin.  The body is the file the skin replaces;
# the rest is reference (the animation sets define the joints the body must follow) or a
# later, optional target (sword, first-person arms, Prime).
RONIN_BODY = "models/titans/light/titan_light_locust.mdl"
RONIN_PRIME = "models/titans/light/titan_light_ronin_prime.mdl"     # same skeleton, separate file
SWORD_WORLD = "models/weapons/titan_sword/w_titan_sword.mdl"
SWORD_WORLD_PRIME = "models/weapons/titan_sword/w_titan_sword_prime.mdl"
RONIN_SET = [
    RONIN_BODY,
    RONIN_PRIME,
    SWORD_WORLD_PRIME,
    "models/weapons/titan_sword/atpov_titan_sword_prime.mdl",
    "models/weapons/arms/pov_titan_light_ronin_prime.mdl",
    # animation sets the body pulls in with $includemodel (never modified)
    "models/titans/light/titan_light_mp_core.mdl",
    "models/titans/light/titan_light_sp_core.mdl",
    "models/titans/light/titan_light_mp_reactions.mdl",
    "models/titans/light/titan_light_mp_scripted.mdl",
    "models/titans/light/titan_light_mp_synced.mdl",
    "models/titans/light/titan_light_mp_embark.mdl",
    "models/titans/light/titan_light_mp_melee.mdl",
    "models/titans/light/titan_light_anims_workspace.mdl",
    # sword (world model) and what the pilot sees from the cockpit
    "models/weapons/titan_sword/w_titan_sword.mdl",
    "models/weapons/titan_sword/atpov_titan_sword.mdl",
    "models/weapons/arms/pov_titan_light_locust.mdl",
    "models/weapons/arms/pov_titan_light_cockpit.mdl",
    "models/titans/light/titan_light_rodeo_battery.mdl",
    # Ronin's gun, the Leadwall (world model and the pilot's view of it)
    "models/weapons/titan_triple_threat/w_titan_triple_threat.mdl",
    "models/weapons/titan_triple_threat/atpov_titan_triple_threat.mdl",
]

STOCK_DIR = config.BUILD / "tf2" / "stock"


def main():
    index = config.TITANFALL2 / "vpk" / VPK_INDEX
    if not index.exists():
        raise SystemExit(f"Titanfall 2 not found: {index}  (set TITANFALL2)")
    t0 = time.time()
    out = vpk.extract(str(index), RONIN_SET, str(STOCK_DIR), workers=config.WORKERS)
    manifest = {"source": VPK_INDEX,
                "files": {n: {"crc": v["crc"], "size": v["size"]} for n, v in out.items()}}
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    (STOCK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(v["size"] for v in out.values())
    print(f"{len(out)} files, {total / 1e6:.1f} MB, all CRC-checked, in {time.time() - t0:.0f}s -> {STOCK_DIR}")
    return out


if __name__ == "__main__":
    main()
