"""Third-party tools the build needs, fetched on demand and pinned by SHA-256.

Nothing here is bundled with the project.  Each tool is downloaded from its own release
page into build/tools/<name>-<version>/ and refused if the archive's hash differs from
the pinned one, so a rebuild on another machine gets exactly the same binaries.

    python -m tsfronin.tools            fetch everything that is missing
    python -m tsfronin.tools --list
"""
import argparse
import hashlib
import urllib.request
import zipfile

from . import config

TOOLS_DIR = config.BUILD / "tools"

TOOLS = {
    # Native Titanfall 2 (studio v53) model compiler + Python decompiler / validator.  MIT.
    "r2-model-compiler": {
        "version": "0.5.0",
        "url": "https://github.com/taskinoz/r2-model-compiler/releases/download/v0.5.0/r2-model-compiler-win64.zip",
        "sha256": "348bafa38e645d20d240fc1f4478340781206d6bc5b214f8548a14cba43375ab",
        "size": 711414,
        "check": "r2studiomdl.exe",
    },
}


def tool_dir(name):
    return TOOLS_DIR / f"{name}-{TOOLS[name]['version']}"


def find(name, filename):
    """Path of a file inside an installed tool (searches the unpacked tree)."""
    hits = sorted(tool_dir(name).rglob(filename))
    if not hits:
        raise SystemExit(f"{filename} not found under {tool_dir(name)} - run `python -m tsfronin.tools`")
    return hits[0]


def fetch(name):
    t = TOOLS[name]
    dst = tool_dir(name)
    if dst.exists() and list(dst.rglob(t["check"])):
        print(f"  {name} {t['version']}: present")
        return dst
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive = TOOLS_DIR / t["url"].rsplit("/", 1)[1]
    print(f"  {name} {t['version']}: downloading {t['url']}")
    req = urllib.request.Request(t["url"], headers={"User-Agent": "build-script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != t["size"] or digest != t["sha256"]:
        raise SystemExit(f"{name}: download does not match the pinned release\n"
                         f"  expected {t['size']} bytes sha256 {t['sha256']}\n"
                         f"  got      {len(blob)} bytes sha256 {digest}\nnothing was unpacked")
    archive.write_bytes(blob)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():                       # refuse paths that would leave the tool folder
            p = (dst / m.filename).resolve()
            if dst.resolve() not in p.parents and p != dst.resolve():
                raise SystemExit(f"{name}: archive entry escapes the tool folder: {m.filename}")
        z.extractall(dst)
    print(f"  {name}: sha256 verified, unpacked {len(z.namelist())} entries -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for name, t in TOOLS.items():
        if a.list:
            print(f"{name} {t['version']}  {t['url']}  sha256 {t['sha256']}")
        else:
            fetch(name)


if __name__ == "__main__":
    main()
