# Read-only: list asset GUIDs/types of an UNCOMPRESSED v7 rpak and test which
# "material/<name>_<type>.rpak" string hashes to each matl GUID.
# Hash ported from r-ex/RePak src/logic/rtech.cpp (StringToGuidAligned).
import struct, sys, re

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def string_to_guid(s):
    b = s.encode('latin1') + b'\0' * 8
    v1 = 0
    i = 0
    pos = 0
    while True:
        w = struct.unpack_from('<I', b, pos)[0]
        v4 = (~w) & (w - 0x1010101) & 0x80808080 & M32
        v5 = (v4 ^ ((v4 - 1) & M32)) & M32
        v6 = ((v5 & w) ^ 0x5C5C5C5C) & M32
        v7 = (~v6) & ((v6 - 0x1010101) & M32) & 0x80808080 & M32
        v8 = v7 & ((-v7) & M32)
        if v7 != v8:
            v9 = 0xFF000000
            while True:
                v10 = v9
                if (v9 & v6) == 0:
                    v8 |= v9 & 0x80808080
                v9 >>= 8
                if not (v10 >= 0x100):
                    break
        v11 = (0x633D5F1 * v1) & M64
        inner = (((v5 & w) - 45 * (v8 >> 7)) & M32) & 0xDFDFDFDF
        v12 = ((0xFB8C4D96501 * inner) & M64) >> 24
        if v4:
            break
        pos += 4
        t = (v11 + v12) & M64
        v1 = ((t >> 61) ^ t) & M64
        i += 4
    v13 = v5.bit_length() - 1 if v5 else -1
    return (v12 + v11 - 0xAE502812AA7333 * ((i + v13 // 8) & M32)) & M64


def parse(path):
    with open(path, 'rb') as f:
        data = f.read()
    (streamBuf, numSeg, numPages, patchCount, numPtrs, numAssets, numGuidRefs,
     numDeps, numExt, extSize) = struct.unpack_from('<hHHhiiiiii', data, 0x38)
    off = 0x58 + streamBuf + numSeg * 0x10 + numPages * 0xC + numPtrs * 8
    assets = []
    for i in range(numAssets):
        e = data[off + i * 0x48: off + (i + 1) * 0x48]
        guid, = struct.unpack_from('<Q', e, 0)
        assets.append((guid, e[0x44:0x48].rstrip(b'\0').decode('latin1')))
    return data, assets


if __name__ == '__main__':
    path = sys.argv[1]
    names = sys.argv[2:]
    data, assets = parse(path)
    byguid = dict(assets)
    print('== ' + path)
    for n in names:
        hits = []
        for pre, suffs in (('material/', ('_skn', '_fix', '_wld', '_gen', '')), ('texture/', ('',))):
            for sfx in suffs:
                s = pre + n + sfx + '.rpak'
                g = string_to_guid(s)
                if g in byguid:
                    hits.append('%s -> 0x%016X [%s]' % (s, g, byguid[g]))
        print('  %-55s %s' % (n, hits if hits else 'NO MATCH'))
