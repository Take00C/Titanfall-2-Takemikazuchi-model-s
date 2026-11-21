# Read-only lister for UNCOMPRESSED Titanfall 2 (version 7) rpak files.
# Layout taken from r-ex/rsx src/game/rtech/cpakfile.h (PakHdr_v7_t, PakAsset_v6_t).
import struct, sys, collections, re

BSL = bytes([92])  # backslash


def parse(path, show_names=True):
    with open(path, 'rb') as f:
        data = f.read()
    magic, version, flags = struct.unpack_from('<4sHH', data, 0)
    createdTime, crc, cmpSize = struct.unpack_from('<QQq', data, 8)
    dcmpSize, = struct.unpack_from('<q', data, 0x28)
    (streamBuf, numSeg, numPages, patchCount, numPtrs, numAssets, numGuidRefs,
     numDeps, numExt, extSize) = struct.unpack_from('<hHHhiiiiii', data, 0x38)
    print('== ' + path)
    print('   magic=%r version=%d flags=0x%04x fileSize=%d cmpSize=%d dcmpSize=%d' % (magic, version, flags, len(data), cmpSize, dcmpSize))
    print('   streamBuf=%d segs=%d pages=%d patchCount=%d ptrs=%d assets=%d guidRefs=%d deps=%d ext=%d/%d' % (
        streamBuf, numSeg, numPages, patchCount, numPtrs, numAssets, numGuidRefs, numDeps, numExt, extSize))
    compressed = (flags >> 8) & 0xFF
    if (compressed & 1) or cmpSize != dcmpSize:
        print('   COMPRESSED - asset table not readable without rtech decompression')
        return None
    off = 0x58
    if patchCount > 0:
        off += 8 + 16 * patchCount + 2 * patchCount
    stream = data[off:off + streamBuf]
    print('   streaming files:', [s.decode('latin1') for s in stream.split(b'\0') if s])
    off += streamBuf
    off += numSeg * 0x10
    pages = []
    for i in range(numPages):
        pages.append(struct.unpack_from('<iiI', data, off + i * 0xC))
    off += numPages * 0xC
    off += numPtrs * 8
    assets = []
    for i in range(numAssets):
        e = data[off + i * 0x48: off + (i + 1) * 0x48]
        guid, = struct.unpack_from('<Q', e, 0)
        hdrSize, ver = struct.unpack_from('<Ii', e, 0x3C)
        typ = e[0x44:0x48]
        assets.append((guid, typ, ver, hdrSize))
    off += numAssets * 0x48
    off += numGuidRefs * 8
    off += numDeps * 4
    off += numExt * 4 + extSize
    hist = collections.Counter((a[1].rstrip(b'\0').decode('latin1'), a[2]) for a in assets)
    print('   asset type histogram (type, assetVersion): count')
    for k, v in sorted(hist.items()):
        print('      %s: %d' % (k, v))
    if show_names:
        blob = data[off:]
        pat = re.compile(rb'[A-Za-z0-9_\-./ ' + re.escape(BSL) + rb']{6,}')
        names = sorted(set(m.group().decode('latin1') for m in pat.finditer(blob)
                           if (b'/' in m.group() or BSL in m.group())))
        print('   path-like strings in page data (%d):' % len(names))
        for n in names[:200]:
            print('      ', n)
    return hist


if __name__ == '__main__':
    for p in sys.argv[1:]:
        parse(p)
