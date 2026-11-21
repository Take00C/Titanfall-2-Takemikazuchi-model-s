"""Posing a Project MIKHAIL BETA unit's meshes from its own takes, for previews and facts (numpy only).

    src = tf2_beta.Source("fort_00")
    poser = Poser(src, [0])                      # mesh indices (build/pm/<unit>/<unit>.json "meshes")
    P, tri = poser.at(take, frame)               # vertices in the scene's units (cm), +Y up; triangles

`at(..., travel=False)` drops the travel the take keys on Main_JT (x / z translate), so a strip of frames stays in
place; turning is kept.  `root(take, frame)` gives that travel and the turn for labels.
"""
import math

import numpy as np


class Poser:
    def __init__(self, src, mesh_indices):
        self.src = src
        self.rig = src.rig
        self.bind_inv = [np.linalg.inv(W) for W in self.rig.world()]
        self.meshes = []
        base = 0
        tris = []
        for k in mesh_indices:
            m = src.mesh(k)
            jn = [self.rig.index[x] for x in m["joints"]]
            self.meshes.append((m["pos"], m["w"], m["j"], jn))
            tris.append(np.asarray(m["tri"], dtype=np.int64) + base)
            base += len(m["pos"])
        self.tri = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
        self.main = self.rig.index.get("Main_JT")

    def world(self, take, f, travel=True):
        T, R, _unknown = self.rig.pose(take, f)
        if not travel and self.main is not None and self.main in T:
            t = T[self.main].copy()
            t[0] = self.rig.T[self.main][0]
            t[2] = self.rig.T[self.main][2]
            T[self.main] = t
        return self.rig.world(T, R)

    def skin(self, W):
        out = []
        for P, wgt, jidx, jn in self.meshes:
            M = np.stack([W[j] @ self.bind_inv[j] for j in jn])
            Q = np.zeros_like(P)
            for s in range(wgt.shape[1]):
                A = M[jidx[:, s]]
                Q += wgt[:, s:s + 1] * (np.einsum("nij,nj->ni", A[:, :3, :3], P) + A[:, :3, 3])
            out.append(Q)
        return np.concatenate(out) if out else np.zeros((0, 3))

    def at(self, take, f, travel=True):
        return self.skin(self.world(take, f, travel)), self.tri

    def root(self, take, f):
        """(travel x, z in the scene's units; turn about the up axis in degrees) keyed on Main_JT / Main2_JT at f"""
        T, R, _unknown = self.rig.pose(take, f)
        x = z = turn = 0.0
        if self.main is not None:
            if self.main in T:
                x = float(T[self.main][0] - self.rig.T[self.main][0])
                z = float(T[self.main][2] - self.rig.T[self.main][2])
            if self.main in R:
                turn += float(R[self.main][1])
        m2 = self.rig.index.get("Main2_JT")
        if m2 is not None and m2 in R:
            turn += float(R[m2][1])
        return x, z, math.degrees(turn)
