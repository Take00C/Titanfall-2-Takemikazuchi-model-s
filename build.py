#!/usr/bin/env python3
"""One command from the installed games to the finished model.

    python build.py                 run every stage, in order
    python build.py extract clean   run only the named stages
    python build.py --list          show the stages
    python build.py --unit type00r_00   build a different Type-00 colourway

Everything it writes goes under build/, which holds nothing that cannot be
regenerated: delete the folder and run this again.  The two game installs are
only ever read.  Paths come from tsfronin/config.py (environment variables or
config.local.json override the defaults).

Needs: Python 3.11+ to run this file (it creates .venv from requirements.txt
by itself) and Blender for the render stage.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENV_PY = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv():
    if VENV_PY.exists():
        return
    if sys.version_info < (3, 11):
        raise SystemExit("run build.py with Python 3.11 or newer")
    print("== creating .venv")
    subprocess.check_call([sys.executable, "-m", "venv", str(REPO / ".venv")])
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check",
                           "-q", "-r", str(REPO / "requirements.txt")])


def py(module, *a):
    return [str(VENV_PY), "-m", module, *a]


def blender(script, *a):
    sys.path.insert(0, str(REPO))
    from tsfronin import config
    if not config.BLENDER.exists():
        raise SystemExit(f"Blender not found at {config.BLENDER} (set BLENDER)")
    return [str(config.BLENDER), "-b", "--factory-startup", "--python",
            str(REPO / "tsfronin" / "blender" / script), "--", *a]


def stages(unit):
    b = REPO / "build"
    from tsfronin import config
    glow = ["--glow", config.GLOW_COLOR] if config.GLOW_COLOR else []
    glow += ["--device", config.RENDER_DEVICE, "--threads", str(config.RENDER_THREADS)]
    return [
        ("extract", "Project MIKHAIL pack.bin -> raw mesh JSON + PNG maps",
         lambda: py("tsfronin.pm_extract", unit)),
        ("clean", "raw export -> cleaned, Z-up, re-indexed model + normalised maps",
         lambda: py("tsfronin.pm_clean", unit)),
        ("sword-extract", "the Type-74 blade out of Project MIKHAIL's pack.bin",
         lambda: py("tsfronin.pm_extract", "--kind", "weapon", config.SWORD_UNIT)),
        ("sword-clean", "the blade cleaned the same way (its own origin kept: that is where the hand holds it)",
         lambda: py("tsfronin.pm_clean", config.SWORD_UNIT)),
        ("gun-extract", "the Type-87 assault cannon out of Project MIKHAIL's pack.bin (skipped when GUN_UNIT is empty)",
         lambda: py("tsfronin.pm_extract", "--kind", "weapon", config.GUN_UNIT) if config.GUN_UNIT else py("tsfronin.tf2_gun")),
        ("gun-clean", "the gun cleaned the same way",
         lambda: py("tsfronin.pm_clean", config.GUN_UNIT) if config.GUN_UNIT else py("tsfronin.tf2_gun")),
        ("render", "review renders (Blender)",
         lambda: blender("render_preview.py", str(b / "clean" / unit), str(b / "renders" / unit), *glow)),
        ("sheet", "tile the renders into one contact sheet",
         lambda: py("tsfronin.contact_sheet", unit)),
        ("tools", "fetch the pinned model compiler (checked against its release hash)",
         lambda: py("tsfronin.tools")),
        ("tf2-extract", "stock Ronin files out of Titanfall 2's VPKs (read only)",
         lambda: py("tsfronin.tf2_extract")),
        ("tf2-skeleton", "Ronin's skeleton and structure, as the reference the skin must fit",
         lambda: py("tsfronin.tf2_skeleton")),
        ("tf2-decompile", "stock Ronin body -> QC recipe + SMDs",
         lambda: py("tsfronin.tf2_decompile")),
        ("retarget", "hang the Takemikazuchi on Ronin's skeleton (data/bone_map.json)",
         lambda: py("tsfronin.retarget", unit)),
        ("materials", "Takemikazuchi maps -> Titanfall 2 material pack (.rpak)",
         lambda: py("tsfronin.tf2_materials", unit)),
        ("compile", "new meshes + stock recipe -> Ronin body .mdl, the stowed sword's seat moved (data/attach_map.json)",
         lambda: py("tsfronin.tf2_build", unit)),
        ("sword", "Ronin's four sword files rebuilt around the Type-74 blade (data/sword_map.json)",
         lambda: py("tsfronin.tf2_sword")),
        ("battery", "the battery canister of light titans rebuilt without its visible mesh (HIDE_BATTERY)",
         lambda: py("tsfronin.tf2_battery")),
        ("gun", "Ronin's two Leadwall files rebuilt around the Type-87 assault cannon (data/gun_map.json; GUN_UNIT)",
         lambda: py("tsfronin.tf2_gun")),
        ("arms", "Ronin's two first-person arm files (the cockpit view) rebuilt with the body fit's arms (data/pov_arms_map.json; POV_ARMS)",
         lambda: py("tsfronin.tf2_arms")),
        ("pilot", "the pilots' shared animation file rebuilt so the pilot is not drawn while boarding a Ronin (HIDE_PILOT_ON_EMBARK)",
         lambda: py("tsfronin.tf2_pilot")),
        ("mod", "assemble the Northstar mod folder under build/mods (nothing is installed)",
         lambda: py("tsfronin.tf2_build", "--stage")),
        # still to come: levels of detail for the body, the installer
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="*", help="stages to run (default: all)")
    ap.add_argument("--unit", default=None, help="Project MIKHAIL unit code (default: config UNIT)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from tsfronin import config
    unit = a.unit or config.UNIT
    all_stages = stages(unit)
    if a.list:
        for name, what, _ in all_stages:
            print(f"  {name:14s} {what}")
        return
    unknown = [s for s in a.stage if s not in {n for n, _, _ in all_stages}]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; see --list")
    ensure_venv()
    env = dict(os.environ, UNIT=unit)
    t_all = time.time()
    for name, what, cmd in all_stages:
        if a.stage and name not in a.stage:
            continue
        print(f"\n== {name}: {what}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd(), cwd=str(REPO), env=env)
        if r.returncode != 0:
            raise SystemExit(f"stage '{name}' failed with exit code {r.returncode}")
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nall done in {time.time() - t_all:.0f}s -> {REPO / 'build'}")


if __name__ == "__main__":
    main()
