# Read-only extractor: decompress selected entries of a Titanfall 2 VPK into the scratch dir.
import struct, sys, os, re, zlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf2lzham_py

def read_cstr(f):
    out = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b'\x00':
            break
        out += c
    return out.decode('latin-1')

def list_dir(path):
    entries = {}
    with open(path, 'rb') as f:
        sig, vmaj, vmin, tree_size, data_size = struct.unpack('<IHHII', f.read(16))
        assert sig == 0x55AA1234
        while True:
            ext = read_cstr(f)
            if ext == '':
                break
            while True:
                p = read_cstr(f)
                if p == '':
                    break
                while True:
                    name = read_cstr(f)
                    if name == '':
                        break
                    crc, preload = struct.unpack('<IH', f.read(6))
                    parts = []
                    while True:
                        archive_index, = struct.unpack('<H', f.read(2))
                        if archive_index == 0xFFFF:
                            break
                        load_flags, tex_flags, offset, csize, usize = struct.unpack('<HIQQQ', f.read(30))
                        parts.append((archive_index, offset, csize, usize))
                    full = (p + '/' if p.strip() else '') + name + '.' + ext
                    entries[full] = (crc, parts)
    return entries

def extract(dirpath, names, outroot, verbose=True):
    ents = list_dir(dirpath)
    base = os.path.basename(dirpath)
    m = re.match(r'^[a-z]+(client_.*pak000)_dir\.vpk$', base)
    stem = m.group(1)
    vdir = os.path.dirname(dirpath)
    res = {}
    for name in names:
        if name not in ents:
            print('MISSING', name)
            continue
        outp = os.path.join(outroot, name.replace('/', os.sep))
        crc, parts = ents[name]
        if os.path.exists(outp):
            d = open(outp, 'rb').read()
            if (zlib.crc32(d) & 0xFFFFFFFF) == crc:
                res[name] = outp
                if verbose: print('cached', name)
                continue
        t0 = time.time()
        blob = bytearray()
        for (ai, off, cs, us) in parts:
            ap = os.path.join(vdir, '%s_%03d.vpk' % (stem, ai))
            with open(ap, 'rb') as f:
                f.seek(off)
                raw = f.read(cs)
            if cs == us:
                blob += raw
            else:
                blob += tf2lzham_py.decompress(raw, us)
        ok = (zlib.crc32(bytes(blob)) & 0xFFFFFFFF) == crc
        if verbose: print('%s %s size=%d parts=%d %.1fs' % ('OK ' if ok else 'CRC-MISMATCH', name, len(blob), len(parts), time.time() - t0))
        if ok:
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            with open(outp, 'wb') as f:
                f.write(blob)
            res[name] = outp
    return res

if __name__ == '__main__':
    dirpath = sys.argv[1]
    outroot = sys.argv[2]
    pats = sys.argv[3:]
    ents = list_dir(dirpath)
    names = []
    for p in pats:
        rx = re.compile(p, re.I)
        names += [n for n in ents if rx.search(n) and n not in names]
    extract(dirpath, names, outroot)
