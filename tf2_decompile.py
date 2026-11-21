"""Stage 45 - decompile the stock Ronin body into a QC and SMDs  ->  build/tf2/decompiled/titan_light_locust/

The QC is the recipe for everything in the model that is not mesh (bones, attachments, hitboxes,
IK chains, physics, animation includes); tf2_build keeps it and swaps only the meshes.  The work
is done by decompile53.py from the pinned r2-model-compiler release (see tools.py).

    python -m tsfronin.tf2_decompile
"""
import subprocess
import sys
import time

from . import tools
from .tf2_build import DECOMPILED
from .tf2_extract import RONIN_BODY, STOCK_DIR


def main():
    script = tools.find("r2-model-compiler", "decompile53.py")
    src = STOCK_DIR / RONIN_BODY.replace("/", "\\")
    if not src.exists():
        raise SystemExit(f"{src} missing - run the tf2-extract stage first")
    DECOMPILED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script), str(src), str(DECOMPILED)], capture_output=True, text=True,
                       errors="replace")
    for ln in [x for x in (r.stdout + r.stderr).splitlines() if x.strip()][-6:]:
        print("    " + ln[-170:])
    qc = DECOMPILED / "titan_light_locust.qc"
    if r.returncode != 0 or not qc.exists():
        raise SystemExit(f"decompile53.py failed (exit code {r.returncode})")
    smds = list(DECOMPILED.glob("*.smd"))
    print(f"  decompiled {RONIN_BODY} in {time.time() - t0:.1f}s: {len(qc.read_text(encoding='utf-8').splitlines())} QC lines, "
          f"{len(smds)} SMD files -> {DECOMPILED}")


if __name__ == "__main__":
    main()
