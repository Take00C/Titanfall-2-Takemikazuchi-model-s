"""Reading, posing and writing SMD animations (the text form the model compiler reads).

An SMD animation lists the skeleton ("nodes": index, name, parent) and then, per frame ("time N"), one row per
bone: position x y z, rotation rx ry rz in radians (R = Rz(rz) Ry(ry) Rx(rx)), and - in the files
r2-model-compiler's decompiler writes - three scale columns.  Every value is relative to the bone's parent.
"""
import math

import numpy as np

from .tf2_build import _euler_matrix, smd_euler


class Anim:
    def __init__(self, path):
        self.path = path
        self.names, self.parents, frames, cur, section = {}, {}, {}, None, None
        for ln in open(path, encoding="ascii", errors="replace"):
            s = ln.strip()
            if s in ("nodes", "skeleton", "triangles"):
                section = s
                continue
            if s == "end":
                section = None
                continue
            if section == "nodes":
                i, rest = s.split(" ", 1)
                self.names[int(i)] = rest[rest.index('"') + 1:rest.rindex('"')]
                self.parents[int(i)] = int(rest[rest.rindex('"') + 1:])
            elif section == "skeleton":
                if s.startswith("time"):
                    cur = int(s.split()[1])
                    frames[cur] = {}
                else:
                    v = s.split()
                    frames[cur][int(v[0])] = [float(x) for x in v[1:]]
        n = len(self.names)
        self.index = {v: k for k, v in self.names.items()}
        self.nframes = max(frames) + 1
        width = max(len(r) for f in frames.values() for r in f.values())
        self.rows = np.zeros((self.nframes, n, width))
        for f, rows in frames.items():
            for i, r in rows.items():
                self.rows[f, i, :len(r)] = r
        if width >= 9:
            self.rows[:, :, 6:9][self.rows[:, :, 6:9] == 0] = 1.0     # an absent scale column means 1
        self.order = sorted(self.names, key=lambda i: (self._depth(i), i))

    def _depth(self, i):
        d = 0
        while self.parents[i] >= 0:
            i = self.parents[i]
            d += 1
        return d

    def local(self, f, i):
        r = self.rows[f, i]
        m = np.eye(4)
        m[:3, :3] = _euler_matrix(r[3], r[4], r[5])
        m[:3, 3] = r[:3]
        return m

    def world(self, f, rows=None):
        """-> {bone index: 4x4 world matrix} for frame f (rows: an optional replacement rows array for that frame)"""
        W = {}
        for i in self.order:
            r = self.rows[f, i] if rows is None else rows[i]
            m = np.eye(4)
            m[:3, :3] = _euler_matrix(r[3], r[4], r[5])
            m[:3, 3] = r[:3]
            p = self.parents[i]
            W[i] = m if p < 0 else W[p] @ m
        return W

    def set_local(self, f, i, M):
        """write a 4x4 local matrix into frame f, bone i (rotation as the compiler's own Euler form)"""
        e = smd_euler(M[:3, :3])
        self.rows[f, i, 0:3] = M[:3, 3]
        self.rows[f, i, 3:6] = e

    def set_rotation(self, f, i, R):
        self.rows[f, i, 3:6] = smd_euler(R)

    def write(self, path):
        lines = ["version 1", "nodes"]
        lines += [f'{i} "{self.names[i]}" {self.parents[i]}' for i in sorted(self.names)]
        lines += ["end", "skeleton"]
        w = self.rows.shape[2]
        for f in range(self.nframes):
            lines.append(f"time {f}")
            for i in sorted(self.names):
                lines.append(f"{i} " + " ".join("%.9g" % v for v in self.rows[f, i, :w]))
        lines.append("end")
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")


def rot_axis(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def align(u, v):
    """the smallest rotation taking direction u onto direction v"""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(np.dot(u, v))
    ax = np.cross(u, v)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [0, 1.0, 0])
        return rot_axis(perp, math.pi)
    return rot_axis(ax, math.atan2(s, c))


def quat(R):
    """rotation matrix -> unit quaternion (w, x, y, z)"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(Ra, Rb, t):
    qa, qb = quat(Ra), quat(Rb)
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = float(np.clip(np.dot(qa, qb), -1, 1))
    if d > 0.9995:
        q = qa + t * (qb - qa)
    else:
        th = math.acos(d)
        q = (math.sin((1 - t) * th) * qa + math.sin(t * th) * qb) / math.sin(th)
    return quat_matrix(q / np.linalg.norm(q))


def blend(Ma, Mb, t):
    """4x4 rigid transforms: position lerp, rotation slerp"""
    M = np.eye(4)
    M[:3, :3] = slerp(Ma[:3, :3], Mb[:3, :3], t)
    M[:3, 3] = (1 - t) * Ma[:3, 3] + t * Mb[:3, 3]
    return M


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def angle_between(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
