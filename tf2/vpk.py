"""Respawn VPK (Titanfall 2) reader: directory index + chunk extraction.  Read-only.

    englishclient_<name>.bsp.pak000_dir.vpk     the index
    client_<name>.bsp.pak000_NNN.vpk            the data archives it points into

Index layout (version 2.3):
    u32 0x55AA1234, u16 2, u16 3, u32 treeSize, u32 dataSize
    tree: ext\\0 { path\\0 { file\\0 entry }* \\0 }* \\0 ... \\0
    entry: u32 crc32, u16 preloadBytes, u16 archiveIndex,
           then chunks { u32 loadFlags, u16 textureFlags, u64 offset, u64 compressed, u64 uncompressed }
           each followed by u16: 0xFFFF ends the entry, anything else means another chunk follows
A chunk holds at most 1 MiB of file data and is LZHAM-compressed when compressed != uncompressed.
"""
import os
import re
import struct
import zlib
from concurrent.futures import ProcessPoolExecutor

from . import lzham

MAGIC = 0x55AA1234


class VpkError(Exception):
    pass


def _cstr(f):
    out = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            return out.decode("latin-1")
        out += c


def read_dir(dir_path):
    """{'models/x/y.mdl': {'crc': int, 'archive': int, 'chunks': [(offset, csize, usize), ...]}}"""
    entries = {}
    with open(dir_path, "rb") as f:
        sig, vmaj, vmin, tree_size, _data_size = struct.unpack("<IHHII", f.read(16))
        if sig != MAGIC or (vmaj, vmin) != (2, 3):
            raise VpkError(f"{dir_path}: not a Respawn VPK 2.3 index (magic {sig:#x}, version {vmaj}.{vmin})")
        while True:
            ext = _cstr(f)
            if not ext:
                break
            while True:
                path = _cstr(f)
                if not path:
                    break
                while True:
                    name = _cstr(f)
                    if not name:
                        break
                    crc, _preload, archive = struct.unpack("<IHH", f.read(8))
                    chunks = []
                    while True:
                        _lf, _tf, off, csize, usize = struct.unpack("<IHQQQ", f.read(30))
                        chunks.append((off, csize, usize))
                        (term,) = struct.unpack("<H", f.read(2))
                        if term == 0xFFFF:
                            break
                    full = (path + "/" if path.strip() else "") + name + "." + ext
                    entries[full] = {"crc": crc, "archive": archive, "chunks": chunks}
        if f.tell() != 16 + tree_size:
            raise VpkError(f"{dir_path}: index parse ended at {f.tell()}, expected {16 + tree_size}")
    return entries


def archive_path(dir_path, index):
    base = os.path.basename(dir_path)
    m = re.match(r"^[a-z]+(client_.*pak000)_dir\.vpk$", base)
    if not m:
        raise VpkError(f"unexpected index file name: {base}")
    return os.path.join(os.path.dirname(dir_path), f"{m.group(1)}_{index:03d}.vpk")


def _read_chunk(job):
    path, off, csize, usize = job
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(csize)
    return raw if csize == usize else bytes(lzham.decompress(raw, usize))


def extract(dir_path, names, out_root, workers=1, log=print):
    """Extract the named entries under out_root.  Every file must match the CRC32 the index
    stores for it; a mismatch raises and nothing is written for that file."""
    entries = read_dir(dir_path)
    missing = [n for n in names if n not in entries]
    if missing:
        raise VpkError(f"not in {os.path.basename(dir_path)}: {missing}")
    todo, done = [], {}
    for n in names:
        dst = os.path.join(out_root, n.replace("/", os.sep))
        if os.path.exists(dst) and (zlib.crc32(open(dst, "rb").read()) & 0xFFFFFFFF) == entries[n]["crc"]:
            done[n] = dst
            log(f"  up to date  {n}")
        else:
            todo.append(n)
    jobs = [(archive_path(dir_path, entries[n]["archive"]), off, cs, us)
            for n in todo for off, cs, us in entries[n]["chunks"]]
    if jobs:
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(_read_chunk, jobs, chunksize=1))
        else:                                   # one chunk at a time, in this process
            parts = [_read_chunk(j) for j in jobs]
        i = 0
        for n in todo:
            k = len(entries[n]["chunks"])
            blob = b"".join(parts[i:i + k])
            i += k
            crc = zlib.crc32(blob) & 0xFFFFFFFF
            if crc != entries[n]["crc"]:
                raise VpkError(f"{n}: CRC {crc:08x} does not match the index ({entries[n]['crc']:08x})")
            dst = os.path.join(out_root, n.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(blob)
            done[n] = dst
            log(f"  extracted   {n}  {len(blob):,} bytes  crc {crc:08x}")
    return {n: {"path": done[n], "crc": "%08x" % entries[n]["crc"],
                "size": sum(us for _o, _c, us in entries[n]["chunks"])} for n in names}
