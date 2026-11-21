"""In-game tests: stage Northstar mods, run the game once unattended, collect screenshots and the log.

How a run works (the launch recipe and the spawn-screenshot-quit idea follow the test harness
of taskinoz/r2-model-compiler, MIT):

  * a throwaway script mod, TSFRonin.TestHarness, is installed next to the mods under test;
  * NorthstarLauncher.exe is started straight into a private LAN match that is never reported
    to the master server (no menus, no clicks, no online play);
  * the mod's SERVER script spawns the requested models as animated props in front of the
    player; its CLIENT script hides the HUD, takes screenshots and quits the game;
  * the screenshots (<game>/r2/screenshots/*.tga) and the Northstar log are copied to
    build/ingame/<run>/, every mod this project installed is removed again, and the three
    user files a launch can touch (enabledmods.json, videoconfig.txt, settings.cfg) are put
    back exactly as they were.

The only place this project writes inside the Titanfall 2 folder is
<Titanfall2>/R2Northstar/mods/TSFRonin.*/ - folders it creates and deletes again.
Nothing in vpk/, r2/ or bin/ is modified.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import config

MODS_STAGING = config.BUILD / "mods"                       # mods are assembled here first
RUNS_DIR = config.BUILD / "ingame"
GAME_MODS = config.TITANFALL2 / "R2Northstar" / "mods"
GAME_LOGS = config.TITANFALL2 / "R2Northstar" / "logs"
GAME_SHOTS = config.TITANFALL2 / "r2" / "screenshots"
OWNED_PREFIX = "TSFRonin."                                 # only folders with this prefix are ever removed
HARNESS = "TSFRonin.TestHarness"

# files a test launch may rewrite; saved before and restored after every run
USER_FILES = [config.TITANFALL2 / "R2Northstar" / "enabledmods.json",
              Path(os.path.expanduser("~")) / "Documents" / "Respawn" / "Titanfall2" / "local" / "videoconfig.txt",
              Path(os.path.expanduser("~")) / "Documents" / "Respawn" / "Titanfall2" / "local" / "settings.cfg"]


# ----------------------------------------------------------------------------- mods
def stage_mod(name, description, version="0.0.1", load_priority=3, extra=None):
    """Create build/mods/<name>/ with a mod.json; returns the folder."""
    if not name.startswith(OWNED_PREFIX):
        raise ValueError(f"mod names must start with {OWNED_PREFIX!r}")
    d = MODS_STAGING / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    meta = {"Name": name, "Description": description, "Version": version, "LoadPriority": load_priority}
    meta.update(extra or {})
    (d / "mod.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")
    return d


def install(name):
    """Copy a staged mod into the game's mods folder (replacing our own previous copy)."""
    src, dst = MODS_STAGING / name, GAME_MODS / name
    if not (src / "mod.json").exists():
        raise SystemExit(f"{src} is not a staged mod")
    if not GAME_MODS.exists():
        raise SystemExit(f"Northstar mods folder not found: {GAME_MODS}")
    uninstall(name)
    shutil.copytree(src, dst)
    files = [p for p in dst.rglob("*") if p.is_file()]
    print(f"  installed {name}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    return dst


def uninstall(name):
    if not name.startswith(OWNED_PREFIX):
        raise ValueError(f"refusing to remove a mod this project does not own: {name}")
    dst = GAME_MODS / name
    if dst.exists():
        shutil.rmtree(dst)
        print(f"  removed {name} from the game")


def installed():
    return sorted(p.name for p in GAME_MODS.iterdir() if p.is_dir() and p.name.startswith(OWNED_PREFIX))


# ----------------------------------------------------------------------------- harness scripts
_SERVER = r'''
global function TSFTest_Init

struct { bool spawned } file

void function TSFTest_Init()
{
%PRECACHE%
    AddCallback_OnPlayerRespawned( TSFTest_OnSpawn )
    printl( "[TSF] harness server init" )
}

void function TSFTest_OnSpawn( entity player )
{
    if ( file.spawned )
        return
    file.spawned = true
    thread TSFTest_SpawnThread( player )
}

void function TSFTest_Spawn( entity player, asset m, string seq, float dist, float side, float yawOffset )
{
    vector fwd = AnglesToForward( < 0, player.EyeAngles().y, 0 > )
    vector right = AnglesToRight( < 0, player.EyeAngles().y, 0 > )
    vector pos = player.GetOrigin() + fwd * dist + right * side
    entity prop = CreatePropDynamic( m, pos, < 0, player.EyeAngles().y + 180 + yawOffset, 0 >, SOLID_VPHYSICS )
    if ( !IsValid( prop ) )
    {
        printl( "[TSF] FAILED to spawn " + m )
        return
    }
    printl( "[TSF] spawned " + m + " at " + pos + " mins " + prop.GetBoundingMins() + " maxs " + prop.GetBoundingMaxs() )
    if ( seq != "" )
    {
        if ( prop.LookupSequence( seq ) >= 0 )
        {
            prop.Anim_Play( seq )
            printl( "[TSF] animation " + m + " sequence=" + seq )
        }
        else
            printl( "[TSF] MISSING sequence " + seq + " on " + m )
    }
}

void function TSFTest_SpawnThread( entity player )
{
    wait 1.0
    if ( !IsValid( player ) )
        return
    player.SetInvulnerable()
%SPAWNS%
    printl( "[TSF] spawn done" )
}
'''

_CLIENT = r'''
global function TSFTest_ClientInit

void function TSFTest_ClientInit()
{
    // Never act in the lobby: if the direct launch falls back to the menu and someone walks
    // into the lobby by hand, the harness must not screenshot and quit the game under them.
    if ( GetMapName() == "mp_lobby" )
    {
        printl( "[TSF] lobby - harness idle" )
        return
    }
    thread TSFTest_Capture()
}

void function TSFTest_Capture()
{
    wait %FIRST%
    entity player = GetLocalClientPlayer()
    if ( !IsValid( player ) )
    {
        printl( "[TSF] no local player for screenshot" )
        return
    }
    player.ClientCommand( "cl_drawhud 0" )
    player.ClientCommand( "r_drawviewmodel 0" )
    wait 1.0
%SHOTS%
    wait 2.0
    printl( "[TSF] capture done" )
    if ( %QUIT% )
        player.ClientCommand( "quit" )
}
'''


_UI = r'''
global function TSFTest_UIInit

void function TSFTest_UIInit()
{
    thread TSFTest_UILaunch()
}

void function TSFTest_UIWait( float seconds )
{
    float end = Time() + seconds
    while ( Time() < end )
        WaitFrame()
}

// Start the match from the menu, the way the "Launch Northstar" button does: only after the
// master-server login has completed.  (Passing +map on the command line makes this launcher
// try to join its own server before its login token is valid, which fails with
// "Failed to authenticate with own server".)
void function TSFTest_UILaunch()
{
    float deadline = Time() + 90.0
    while ( !NSIsMasterServerAuthenticated() && Time() < deadline )
        WaitFrame()
    printl( "[TSF] ui: master server authenticated = " + NSIsMasterServerAuthenticated() )
    TSFTest_UIWait( %SETTLE% )
    for ( int attempt = 1; attempt <= 3; attempt++ )
    {
        if ( uiGlobal.loadedLevel != "" )
            return
        printl( "[TSF] ui: starting %MAP% (attempt " + attempt + ")" )
        ClientCommand( "setplaylist private_match" )
        ClientCommand( "mp_gamemode %MODE%" )
        ClientCommand( "map %MAP%" )
        TSFTest_UIWait( 40.0 )
    }
    if ( uiGlobal.loadedLevel == "" )
        printl( "[TSF] FAILED to start %MAP%" )
}
'''


def stage_harness(spawns, shots=2, first_shot=14.0, shot_gap=3.0, extra_files=None, quit_after=True,
                  map_name="mp_lf_stacks", mode="tdm", settle=10.0):
    """spawns: [{"model": "models/...mdl", "sequence": "", "dist": 520, "side": 0, "yaw": 0}]
    extra_files: {"models/tsfronin_test/x.mdl": Path} - files mounted by the harness mod itself
    (for example an untouched copy of a stock model under a new path, as a side-by-side reference)."""
    mod = stage_mod(HARNESS, "Test only: spawns models in front of the player, screenshots, quits",
                    version="0.1.0", load_priority=100,
                    extra={"Scripts": [
                        {"Path": "tsfronin_test.nut", "RunOn": "SERVER && MP",
                         "ServerCallback": {"After": "TSFTest_Init"}},
                        {"Path": "tsfronin_test_cl.nut", "RunOn": "CLIENT && MP",
                         "ClientCallback": {"After": "TSFTest_ClientInit"}},
                        {"Path": "ui/tsfronin_test_ui.nut", "RunOn": "UI",
                         "UICallback": {"After": "TSFTest_UIInit"}}]})
    scripts = mod / "mod" / "scripts" / "vscripts"
    (scripts / "ui").mkdir(parents=True)
    (scripts / "ui" / "tsfronin_test_ui.nut").write_text(
        _UI.replace("%MAP%", map_name).replace("%MODE%", mode).replace("%SETTLE%", str(float(settle))),
        encoding="ascii")
    models = sorted({s["model"] for s in spawns})
    precache = "\n".join(f'    PrecacheModel( $"{m}" )' for m in models)
    lines = []
    for s in spawns:
        lines.append('    TSFTest_Spawn( player, $"%s", "%s", %s, %s, %s )' % (
            s["model"], s.get("sequence", ""), float(s.get("dist", 520)), float(s.get("side", 0)),
            float(s.get("yaw", 0))))
    sv = _SERVER.replace("%PRECACHE%", precache).replace("%SPAWNS%", "\n".join(lines))
    shot_lines = []
    for i in range(shots):
        shot_lines.append('    player.ClientCommand( "screenshot" )\n    printl( "[TSF] screenshot %d requested" )' % (i + 1))
        if i + 1 < shots:
            shot_lines.append("    wait %s" % float(shot_gap))
    cl = _CLIENT.replace("%FIRST%", str(float(first_shot))).replace("%SHOTS%", "\n".join(shot_lines)) \
        .replace("%QUIT%", "true" if quit_after else "false")
    (scripts / "tsfronin_test.nut").write_text(sv, encoding="ascii")
    (scripts / "tsfronin_test_cl.nut").write_text(cl, encoding="ascii")
    for game_path, local in (extra_files or {}).items():
        dst = mod / "mod" / game_path.replace("/", os.sep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)
    return mod


# ----------------------------------------------------------------------------- one run
def game_running():
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    return [n for n in ("NorthstarLauncher.exe", "Titanfall2.exe") if n in out]


def run(name, mods, spawns, map_name="mp_lf_stacks", mode="tdm", timeout=300, width=1280, height=720, **harness):
    """Install `mods` + the harness, run the game once, collect results into build/ingame/<name>/.
    Returns a dict with the screenshots (PNG paths), the log path, marker lines and failures."""
    from PIL import Image
    busy = game_running()
    if busy:
        raise SystemExit(f"{', '.join(busy)} is already running - close the game before a test run")
    launcher = config.TITANFALL2 / "NorthstarLauncher.exe"
    if not launcher.exists():
        raise SystemExit(f"{launcher} not found")
    out = RUNS_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    backup = out / "user_files"
    backup.mkdir()
    saved = []
    for f in USER_FILES:
        if f.exists():
            shutil.copy2(f, backup / f.name)
            saved.append(f)
    stage_harness(spawns, map_name=map_name, mode=mode, **harness)
    result = {"name": name, "mods": list(mods), "map": map_name, "screenshots": [], "failures": []}
    try:
        for m in list(mods) + [HARNESS]:
            install(m)
        shots_before = set(GAME_SHOTS.glob("*")) if GAME_SHOTS.exists() else set()
        logs_before = set(GAME_LOGS.glob("*.txt"))
        args = [str(launcher), "-multiple", "-novid", "-nosound", "-windowed", "-w", str(width), "-h", str(height),
                "+ip", "127.0.0.1", "+hostport", "37042", "+sv_lan", "1",
                "+ns_report_server_to_masterserver", "0", "+ns_report_sp_server_to_masterserver", "0",
                "+sv_cheats", "1"]      # the map is started by the harness's menu script, not by +map
        print("  launching:", " ".join(args[1:]))
        t0 = time.time()
        p = subprocess.Popen(args, cwd=str(config.TITANFALL2))
        timed_out = False
        while p.poll() is None:
            if time.time() - t0 > timeout:
                timed_out = True
                print("  timeout; stopping the test instance")
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                break
            time.sleep(2)
        result["seconds"] = round(time.time() - t0)
        if timed_out:
            result["failures"].append("timed out")
        time.sleep(1)
        new_logs = sorted(set(GAME_LOGS.glob("*.txt")) - logs_before, key=lambda q: q.stat().st_mtime)
        text = ""
        for lg in new_logs:
            shutil.copy2(lg, out / lg.name)
            text += lg.read_text(encoding="utf-8", errors="replace")
        result["log"] = str(out / new_logs[-1].name) if new_logs else None
        result["markers"] = [ln.split("] ", 1)[-1].strip() for ln in text.splitlines() if "[TSF]" in ln]
        for marker in ("SCRIPT ERROR", "COMPILE ERROR", "[TSF] FAILED", "[TSF] MISSING"):
            if marker in text:
                result["failures"].append(marker)
        if text.count("[TSF] spawned ") != len(spawns):
            result["failures"].append(f"spawned {text.count('[TSF] spawned ')} of {len(spawns)} models")
        new_shots = sorted(set(GAME_SHOTS.glob("*")) - shots_before, key=lambda q: q.stat().st_mtime) \
            if GAME_SHOTS.exists() else []
        for i, s in enumerate(new_shots):
            dst = out / f"shot{i + 1}.png"
            Image.open(s).convert("RGB").save(dst)
            s.unlink()                                   # our own screenshot; leave the folder as we found it
            result["screenshots"].append(str(dst))
        if not new_shots:
            result["failures"].append("no screenshot captured")
    finally:
        for m in installed():
            uninstall(m)
        for f in saved:
            shutil.copy2(backup / f.name, f)
        print(f"  restored {', '.join(f.name for f in saved)}")
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
