"""Where everything lives.

Nothing here is game data.  Every path can be overridden with an environment
variable of the same name, or with a `config.local.json` next to build.py
(`{"MIKHAIL_GAME": "E:/Games/Project MIKHAIL", ...}`), so the same scripts run
on a machine whose games are installed somewhere else.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"          # every regenerated file goes here, nothing else does

_DEFAULTS = {
    # the two installed games (read-only inputs)
    "MIKHAIL_GAME": r"D:\SteamLibrary\steamapps\common\Project MIKHAIL",
    "TITANFALL2": r"D:\SteamLibrary\steamapps\common\Titanfall2",
    # mikhail-toolkit checkout and the archive keystream the user derives from their own game
    "MIKHAIL_TOOLKIT": r"C:\Program Files (x86)\Steam\steamapps\common\_MuvLuvWarno\tools\mikhail-toolkit",
    "MIKHAIL_KEY": "",          # default: <MIKHAIL_TOOLKIT>/decryptKey.bin
    "BLENDER": r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    # which Type-00 to build (type00r_00, the purple one, shares this exact mesh)
    "UNIT": "type00r_bl_00",
    # How hard the build may push the machine.  The defaults are deliberately gentle: one worker
    # process, CPU rendering on a few threads.  Raise them on a machine that is known to be stable
    # (WORKERS=8 makes the Titanfall 2 extraction ~10x faster; RENDER_DEVICE=gpu the renders ~5x).
    "WORKERS": "1",
    "RENDER_DEVICE": "cpu",
    "RENDER_THREADS": "4",
    # the sword that replaces Ronin's: bl74_00 is the Type-74 PB Blade the Takemikazuchi carries
    "SWORD_UNIT": "bl74_00",
    # 1 = the body has Ronin's four levels of detail (the game swaps to Project MIKHAIL's simpler meshes of the same
    # mech when the titan is far away, as it does for the stock Ronin).  0 = full detail at every distance.
    # Up close the model is the same either way.
    "BODY_LODS": "1",
    # The gun that replaces Ronin's Leadwall: gun87_00 is the Type-87 assault cannon the Takemikazuchi carries.
    # Empty = leave the Leadwall alone (the gun files are then simply not put in the mod).
    "GUN_UNIT": "gun87_00",
    # 1 = ship the battery canister of light titans without its visible mesh (it stands in the open behind the
    # Takemikazuchi's head otherwise).  The game uses that one file for Northstar too.  0 = leave the canister alone.
    "HIDE_BATTERY": "1",
    # 1 = the arms the pilot sees from the cockpit are the Takemikazuchi's (Ronin's two first-person arm files are
    # rebuilt with the body fit's arms).  0 = leave Ronin's own arms in the cockpit view.
    "POV_ARMS": "1",
    # 1 = the pilot is not drawn while boarding a Ronin or Northstar (the pilots' shared animation file
    # models/humans/pete/pete_scripted_r1.mdl with its 28 Stryder-class boarding animations rebuilt; tsfronin/tf2_pilot.py).
    # On this player's screen that holds for every pilot boarding those titans.  0 = leave the pilots' file alone.
    "HIDE_PILOT_ON_EMBARK": "1",
    # 1 = the gun seen from the cockpit gets its own reload (data/reload_map.json): the box magazine on top of the
    # Type-87 is changed by hand.  0 = Ronin's stock reload motion (made for the Leadwall's drum) is kept.
    "GUN_RELOAD": "1",
    # Lit-panel colour as linear "r,g,b".  Empty = use the colour extracted from the unit's own
    # glow material (0.73, 0.024, 0.0 for type00r_bl_00).  Set it only to deliberately depart
    # from the game, e.g. "0.913,0.223,0.010" for the lighter orange of the model-kit artwork.
    "GLOW_COLOR": "",
    # Brightness of the lit panels in Titanfall 2 relative to the extracted emissive value (1 = as stored).
    "GLOW_INTENSITY": "1.0",
}

_local = {}
_local_file = REPO / "config.local.json"
if _local_file.exists():
    _local = json.loads(_local_file.read_text(encoding="utf-8"))


def get(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    v = _local.get(name)
    if isinstance(v, bool):                         # a switch written as true / false in config.local.json
        return "1" if v else "0"
    if v is None or v == "":
        return _DEFAULTS[name]
    return str(v)


MIKHAIL_GAME = Path(get("MIKHAIL_GAME"))
TITANFALL2 = Path(get("TITANFALL2"))
MIKHAIL_TOOLKIT = Path(get("MIKHAIL_TOOLKIT"))
MIKHAIL_KEY = Path(get("MIKHAIL_KEY") or (MIKHAIL_TOOLKIT / "decryptKey.bin"))
BLENDER = Path(get("BLENDER"))
UNIT = get("UNIT")
SWORD_UNIT = get("SWORD_UNIT")
WORKERS = max(1, int(get("WORKERS")))
RENDER_DEVICE = get("RENDER_DEVICE").lower()
RENDER_THREADS = max(1, int(get("RENDER_THREADS")))
GUN_UNIT = os.environ.get("GUN_UNIT", _local.get("GUN_UNIT", _DEFAULTS["GUN_UNIT"])).strip()   # "" is a valid choice
BODY_LODS = get("BODY_LODS").strip() not in ("", "0", "false", "no")
HIDE_BATTERY = get("HIDE_BATTERY").strip() not in ("", "0", "false", "no")
POV_ARMS = get("POV_ARMS").strip() not in ("", "0", "false", "no")
HIDE_PILOT_ON_EMBARK = get("HIDE_PILOT_ON_EMBARK").strip() not in ("", "0", "false", "no")
GUN_RELOAD = get("GUN_RELOAD").strip() not in ("", "0", "false", "no")
GLOW_COLOR = get("GLOW_COLOR")
GLOW_INTENSITY = float(get("GLOW_INTENSITY"))

# The eight Type-00 Takemikazuchi colourways in the pack
TYPE00_VARIANTS = [
    "type00a_00", "type00c_00", "type00c_bl_00", "type00f_bl_00",
    "type00f_rd_00", "type00f_yl_00", "type00r_00", "type00r_bl_00",
]

PM_DIR = BUILD / "pm"           # stage 10: raw toolkit export  (json + png)
CLEAN_DIR = BUILD / "clean"     # stage 20: cleaned, Z-up, game-neutral mesh + maps
RENDER_DIR = BUILD / "renders"  # stage 30: review renders


def check_inputs():
    """Fail early, and say exactly what is missing."""
    problems = []
    if not (MIKHAIL_GAME / "obb" / "pack.bin").exists():
        problems.append(f"Project MIKHAIL not found: {MIKHAIL_GAME}\\obb\\pack.bin  (set MIKHAIL_GAME)")
    if not (MIKHAIL_TOOLKIT / "mikhail" / "export_rigged.py").exists():
        problems.append(f"mikhail-toolkit not found: {MIKHAIL_TOOLKIT}  (set MIKHAIL_TOOLKIT)")
    if not MIKHAIL_KEY.exists():
        problems.append(f"archive keystream not found: {MIKHAIL_KEY}  "
                        f"(run the toolkit's tools/make_key.py against your own game, or set MIKHAIL_KEY)")
    if problems:
        raise SystemExit("cannot build:\n  " + "\n  ".join(problems))
