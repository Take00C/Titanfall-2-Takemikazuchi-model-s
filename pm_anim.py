"""Project MIKHAIL animation takes ("<name>.editMA.take.zstd") -> joint poses per frame.

A take is an EPK tag stream (the same serialisation as the mesh scenes, see the mikhail-toolkit's
docs/SCENE_FORMAT_FINDINGS.md) holding one object with, among others:

    fps            key frames are counted at this rate (120 on tank_00)
    start, end     the take's span at 30 per second; start2, end2 the same at `fps`
    curves[]       one Maya-style animation curve per joint attribute:
        value_type          1 = number, 6 = visibility (on/off)
        values[]            the key values (f32; rotations in radians, translations in the scene's units)
        target              "<node path>:joint.<attr>.<axis>", e.g. ".../Root_M/Spine1_M:joint.rotate.x"
        interpolation_type[]  one per key
        keyFrames[]         the frame of each key (u8 / u16)

Measured on tank_00 walk_00 (2026-11-16): 650 curves = 65 joints x (translate, rotate, scale) x 3 + visibility;
keys on (nearly) every frame 0..100; constant attributes are one key at frame 0.

Tag stream as decoded here (opcode = tag & 0x7f for tags >= 0x80; a tag < 0x80 is that small integer):
    0x00-0x02 small flag values (0x81, 0x82 seen in the take settings)
    0x03 class prefix (03 03 <crc32>)      0x04 no-op marker        0x05 end of object / array
    0x06, 0x26 NUL-terminated string        0x07 field key (87 <crc32 of the field name>)
    0x08 bool (u8: visibility keys)  0x09 u8  0x0a u16  0x0c u32  0x0f f8  0x10 f16  0x11 f32  0x12 f64  0x15 u32
    0x13 array <u8 n> <element tag> data    0x14 array <u16 n> <element tag> data
    0x16+n array of n fixed elements, <element tag> follows
    0x27 array <u8 n> of values             0x28 array <u16 n> of values (closed by 0x05)
    0x40+n array of n values (closed by 0x05 when n > 0)   0x50+n object of n (key, value) pairs, closed by 0x05
    0x0d one-byte marker                    0xff null
"""
import math
import struct
import zlib

import numpy as np

ELEM = {0x08: ("B", 1), 0x09: ("B", 1), 0x0a: ("H", 2), 0x0b: ("H", 2), 0x0c: ("I", 4), 0x0f: ("B", 1), 0x10: ("e", 2),
        0x11: ("f", 4), 0x12: ("d", 8), 0x15: ("I", 4)}
# 0x12 = f64: found 2026-11-19 in four fort_00 death takes (deadB/F/L/R_00), in the event data beside an effect name
# (EF_FORT_DIRT_00) and voice cues (fort_voice_*): the bytes read as 1.8 / 0.8 as a double, and with it all four
# parse to their last byte (build/worker/scripts/fort_tag92_probe.py)
FIELDS = ["ActionCancel", "ExtraData", "NotifyClip", "curves", "duration", "end", "end2", "extradata", "fps", "frame",
          "head", "interpolation_type", "isRepeat", "keyFrames", "layerId", "param", "start", "start2", "target",
          "value_type", "values", "version", "name", "type", "data", "count", "length", "loop", "time"]


def f8(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x07
    v = (m / 8.0) * 2 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


class Reader:
    def __init__(self, d, names=()):
        self.d = d
        self.crc = {zlib.crc32(s.encode()) & 0xffffffff: s for s in list(FIELDS) + list(names)}
        p = 0x10                                              # the sorted name pool after the 16-byte EPK header
        while p < len(d):
            e = d.find(b"\x00", p)
            if e <= p or not all(32 <= c < 127 for c in d[p:e]):
                break
            s = d[p:e].decode()
            self.crc.setdefault(zlib.crc32(s.encode()) & 0xffffffff, s)
            p = e + 1
        self.start = p

    def name(self, k):
        return self.crc.get(k, "crc:%08x" % k)

    def fixed(self, i, n, et):
        eid = et & 0x7f
        fmt, sz = ELEM[eid]
        raw = self.d[i:i + n * sz]
        if eid == 0x0f:
            vals = [f8(x) for x in raw]
        elif fmt == "B":
            vals = list(raw)
        else:
            vals = list(struct.unpack("<%d%s" % (n, fmt), raw))
        return vals, i + n * sz

    def value(self, i):
        """-> (python value, next index)"""
        d = self.d
        t = d[i]
        if t < 0x80:
            return t, i + 1
        if t == 0xff:
            return None, i + 1
        op = t & 0x7f
        if op in (0x00, 0x01, 0x02):                           # small flag / enum values (seen: 0x81, 0x82)
            return op, i + 1
        if op == 0x04 or op == 0x0d:                           # markers in front of a value
            return self.value(i + 1)
        if op == 0x03:
            sub = d[i + 1]
            if sub == 3:
                cls = self.name(int.from_bytes(d[i + 2:i + 6], "little"))
                v, j = self.value(i + 6)
                if isinstance(v, dict):
                    v = dict(v, __class__=cls)
                return v, j
            if sub == 2:
                e = d.find(b"\x00", i + 2)
                v, j = self.value(e + 1)
                return v, j
            return self.value(i + 2)
        if op in (0x06, 0x26):
            e = d.find(b"\x00", i + 1)
            return d[i + 1:e].decode("latin-1"), e + 1
        if op in ELEM:
            fmt, sz = ELEM[op]
            if op == 0x0f:
                return f8(d[i + 1]), i + 2
            return struct.unpack_from("<" + fmt, d, i + 1)[0], i + 1 + sz
        if op == 0x13:
            return self.fixed(i + 3, d[i + 1], d[i + 2])
        if op == 0x14:
            n = struct.unpack_from("<H", d, i + 1)[0]
            return self.fixed(i + 4, n, d[i + 3])
        if 0x16 <= op < 0x26 and (d[i + 1] & 0x80) and (d[i + 1] & 0x7f) in ELEM:
            return self.fixed(i + 2, op - 0x16, d[i + 1])
        if op in (0x27, 0x28):
            if op == 0x27:
                n, j = d[i + 1], i + 2
            else:                                               # a8 <u16 count>  (650 curves = a8 8a 02)
                n, j = struct.unpack_from("<H", d, i + 1)[0], i + 3
            out = []
            for _ in range(n):
                v, j = self.value(j)
                out.append(v)
            if j < len(d) and d[j] == 0x85:
                j += 1
            return out, j
        if 0x40 <= op < 0x50:
            out, j = [], i + 1
            for _ in range(op - 0x40):
                v, j = self.value(j)
                out.append(v)
            if out and j < len(d) and d[j] == 0x85:              # a non-empty array is closed like an object
                j += 1
            return out, j
        if 0x50 <= op < 0x60:
            obj, j = {}, i + 1
            for _ in range(op - 0x50):
                while d[j] == 0x84:
                    j += 1
                if d[j] != 0x87:
                    raise ValueError(f"expected a field key at {j:#x}, found {d[j]:#04x}")
                key = self.name(int.from_bytes(d[j + 1:j + 5], "little"))
                obj[key], j = self.value(j + 5)
            if j < len(d) and d[j] == 0x85:
                j += 1
            return obj, j
        raise ValueError(f"unknown tag {t:#04x} at {i:#x}")


def parse_take(d):
    """take bytes -> {"fields": root fields without curves, "curves": [...]}"""
    if d[:4] != b"EPK\x00":
        raise ValueError("not an EPK payload")
    r = Reader(d)
    i = r.start
    while d[i] < 0x80 or d[i] == 0x80:                         # a small header of plain bytes before the root object
        i += 1
    root, end = r.value(i)
    curves = []
    for c in root.get("curves", []):
        path, attr = c["target"].rsplit(":", 1)
        node = path.split("/")[-1]
        vals = c["values"] if isinstance(c["values"], list) else [c["values"]]
        keys = c["keyFrames"] if isinstance(c["keyFrames"], list) else [c["keyFrames"]]
        if len(vals) != len(keys):
            raise ValueError(f"{c['target']}: {len(vals)} values for {len(keys)} keys")
        curves.append({"node": node, "path": path, "attr": attr.replace("joint.", ""), "type": c.get("value_type"),
                       "frames": np.array(keys, dtype=np.float64), "values": np.array(vals, dtype=np.float64),
                       "interp": c.get("interpolation_type")})
    fields = {k: v for k, v in root.items() if k != "curves"}
    return {"fields": fields, "curves": curves, "parsed_to": end, "size": len(d)}


def sample(curve, f):
    """the curve's value at frame f (linear between keys, held outside them)"""
    fr, v = curve["frames"], curve["values"]
    if len(fr) == 1:
        return float(v[0])
    return float(np.interp(f, fr, v))


def last_frame(take):
    return int(max(c["frames"].max() for c in take["curves"]))


def euler_xyz(r):
    cx, sx = math.cos(r[0]), math.sin(r[0])
    cy, sy = math.cos(r[1]), math.sin(r[1])
    cz, sz = math.cos(r[2]), math.sin(r[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Rig:
    """The unit's scene nodes (name, parent, bind translate / rotate / jointOrient) in the scene's own units."""

    def __init__(self, scene):
        self.names = [n["name"] for n in scene.nodes]
        self.parent = list(scene.parent)
        self.T = [np.array(n["T"], dtype=np.float64) for n in scene.nodes]
        self.R = [np.array(n["R"], dtype=np.float64) for n in scene.nodes]
        self.JO = [np.array(n["JO"], dtype=np.float64) for n in scene.nodes]
        self.index = {}
        for k, nm in enumerate(self.names):
            self.index.setdefault(nm, k)
        order, seen = [], set()

        def visit(k):
            if k in seen:
                return
            if self.parent[k] is not None:
                visit(self.parent[k])
            seen.add(k)
            order.append(k)
        for k in range(len(self.names)):
            visit(k)
        self.order = order

    def world(self, T=None, R=None):
        """world matrix per node; T / R: {node index: local translate / rotate} replacing the bind values"""
        W = [None] * len(self.names)
        for k in self.order:
            t = T.get(k, self.T[k]) if T else self.T[k]
            r = R.get(k, self.R[k]) if R else self.R[k]
            L = np.eye(4)
            L[:3, :3] = euler_xyz(self.JO[k]) @ euler_xyz(r)
            L[:3, 3] = t
            p = self.parent[k]
            W[k] = L if p is None else W[p] @ L
        return W

    def pose(self, take, f):
        """local translate / rotate per node index at frame f of a take (un-animated attributes keep the bind values);
        -> (T, R, unknown node names)"""
        T, R, unknown = {}, {}, set()
        for c in take["curves"]:
            k = self.index.get(c["node"])
            if k is None:
                unknown.add(c["node"])
                continue
            a = c["attr"]
            if a.startswith("translate.") or a.startswith("rotate."):
                axis = "xyz".index(a[-1])
                tgt = T if a[0] == "t" else R
                if k not in tgt:
                    tgt[k] = (self.T[k] if a[0] == "t" else self.R[k]).copy()
                tgt[k][axis] = sample(c, f)
        return T, R, unknown
