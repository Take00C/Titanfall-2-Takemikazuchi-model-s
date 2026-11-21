"""Block compression for the texture formats Titanfall 2 materials use: BC1 (colour), BC4 (one channel)
and BC5 (two channels, normal maps).  Plain numpy, every block of an image at once.

Each 4x4 block is stored as two end values and a small index per pixel choosing a blend of them:
BC1 keeps two RGB565 colours and 2-bit indices (4 colours on the line between them); BC4 keeps two
8-bit values and 3-bit indices (8 values); BC5 is two BC4 blocks, red then green.
"""
import numpy as np

CHUNK = 32768                                   # blocks handled per pass, to keep the working arrays small


def _blocks(img):
    """(H, W, C) -> (blocks, 16, C), H and W padded up to a multiple of 4 by repeating the edge."""
    h, w = img.shape[:2]
    ph, pw = -h % 4, -w % 4
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    h, w, c = img.shape
    return img.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)


def encode_bc1(rgb):
    """uint8 (H, W, 3) -> bytes, 8 per block, opaque four-colour mode."""
    blocks = _blocks(np.asarray(rgb, dtype=np.uint8)[..., :3])
    out = np.zeros((len(blocks), 8), dtype=np.uint8)
    shifts = (2 * np.arange(16, dtype=np.uint32))
    for s in range(0, len(blocks), CHUNK):
        b = blocks[s:s + CHUNK].astype(np.float32)
        mean = b.mean(axis=1, keepdims=True)
        d = b - mean
        cov = np.einsum("bpi,bpj->bij", d, d)
        axis = np.ones((len(b), 3), dtype=np.float32)
        for _ in range(6):                                       # power iteration: the line the colours lie along
            axis = np.einsum("bij,bj->bi", cov, axis)
            axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
        flat = np.linalg.norm(axis, axis=1) < 0.5                # single-colour block: no direction
        axis[flat] = (0.57735, 0.57735, 0.57735)
        t = np.einsum("bpi,bi->bp", d, axis)
        ends = []
        for tt in (t.max(axis=1), t.min(axis=1)):
            c = np.clip(mean[:, 0] + tt[:, None] * axis, 0, 255)
            ends.append((np.rint(c[:, 0] * (31 / 255)).astype(np.uint32), np.rint(c[:, 1] * (63 / 255)).astype(np.uint32),
                         np.rint(c[:, 2] * (31 / 255)).astype(np.uint32)))
        c0 = (ends[0][0] << 11) | (ends[0][1] << 5) | ends[0][2]
        c1 = (ends[1][0] << 11) | (ends[1][1] << 5) | ends[1][2]
        swap = c0 < c1                                           # colour 0 must be the larger number
        c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)

        def expand(c):
            r, g, bl = (c >> 11) & 31, (c >> 5) & 63, c & 31
            return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (bl << 3) | (bl >> 2)], axis=1).astype(np.float32)

        e0, e1 = expand(c0), expand(c1)
        pal = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], axis=1)        # (B, 4, 3)
        dist = ((b[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(axis=3)             # (B, 16, 4)
        idx = dist.argmin(axis=2).astype(np.uint32)
        bits = (idx << shifts).sum(axis=1, dtype=np.uint32)
        o = out[s:s + CHUNK]
        o[:, 0], o[:, 1] = c0 & 255, c0 >> 8
        o[:, 2], o[:, 3] = c1 & 255, c1 >> 8
        for k in range(4):
            o[:, 4 + k] = (bits >> (8 * k)) & 255
    return out.tobytes()


def _bc4_blocks(values):
    """(B, 16) uint8 -> (B, 8) uint8"""
    out = np.zeros((len(values), 8), dtype=np.uint8)
    shifts = (3 * np.arange(16, dtype=np.uint64))
    w = np.arange(8, dtype=np.float32)
    for s in range(0, len(values), CHUNK):
        v = values[s:s + CHUNK].astype(np.float32)
        a0, a1 = v.max(axis=1), v.min(axis=1)                    # a0 > a1 selects the eight-value mode
        # palette order in the file: a0, a1, then six blends from a0 towards a1
        pal = np.empty((len(v), 8), dtype=np.float32)
        pal[:, 0], pal[:, 1] = a0, a1
        for k in range(1, 7):
            pal[:, k + 1] = ((7 - k) * a0 + k * a1) / 7
        idx = np.abs(v[:, :, None] - pal[:, None, :]).argmin(axis=2).astype(np.uint64)
        bits = (idx << shifts).sum(axis=1, dtype=np.uint64)
        o = out[s:s + CHUNK]
        o[:, 0], o[:, 1] = a0.astype(np.uint8), a1.astype(np.uint8)
        for k in range(6):
            o[:, 2 + k] = (bits >> np.uint64(8 * k)) & np.uint64(255)
    return out


def encode_bc4(gray):
    """uint8 (H, W) -> bytes, 8 per block."""
    g = np.asarray(gray, dtype=np.uint8)
    return _bc4_blocks(_blocks(g[..., None])[:, :, 0]).tobytes()


def encode_bc5(rg):
    """uint8 (H, W, 2) -> bytes, 16 per block: the red block, then the green block."""
    b = _blocks(np.asarray(rg, dtype=np.uint8)[..., :2])
    return np.concatenate([_bc4_blocks(b[:, :, 0]), _bc4_blocks(b[:, :, 1])], axis=1).tobytes()


def halve(img):
    """float (H, W, C) -> next mip level by averaging 2x2 (a 1-pixel side stays 1)."""
    h, w = img.shape[:2]
    if h > 1:
        img = (img[0:h - h % 2:2] + img[1:h:2]) * 0.5
    if w > 1:
        img = (img[:, 0:w - w % 2:2] + img[:, 1:w:2]) * 0.5
    return img


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def mip_chain(img, kind):
    """img: float32 (H, W, C) in 0..1.  kind: 'srgb' (linear-light colour, stored sRGB), 'linear' (stored as is)
    or 'normal' (C = 3 unit vectors; x and y are stored).  -> list of uint8 arrays, largest first, down to 1x1."""
    levels = []
    cur = np.asarray(img, dtype=np.float32)
    while True:
        if kind == "srgb":
            levels.append(np.rint(linear_to_srgb(cur) * 255).astype(np.uint8))
        elif kind == "normal":
            n = cur / np.maximum(np.linalg.norm(cur, axis=2, keepdims=True), 1e-6)
            levels.append(np.rint(np.clip(n[..., :2] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
        else:
            levels.append(np.rint(np.clip(cur, 0, 1) * 255).astype(np.uint8))
        if cur.shape[0] == 1 and cur.shape[1] == 1:
            return levels
        cur = halve(cur)


ENCODERS = {"bc1": (encode_bc1, 1), "bc4": (lambda a: encode_bc4(a[..., 0]), 6), "bc5": (encode_bc5, 8)}
