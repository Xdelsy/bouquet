def _startup_fail(message):
    """Show a readable startup error (messagebox if tkinter works, else print) and exit."""
    try:
        import tkinter
        from tkinter import messagebox

        _r = tkinter.Tk()
        _r.withdraw()
        messagebox.showerror("Bouquet", message)
        _r.destroy()
    except Exception:
        print(message)
    raise SystemExit(1)


try:
    import tkinter  # noqa: F401
except ImportError:
    _startup_fail("Не найден модуль tkinter.\n"
                  "Установите Python с поддержкой Tk (на python.org он входит в установщик).")

try:
    import numpy  # noqa: F401
except ImportError:
    _startup_fail("Не найден модуль numpy.\nУстановите его командой:\n\n    pip install numpy")

import math
import sys
import time

import numpy as np

COLS, ROWS = 128, 72
CW, CH = 7, 11
FPS = 24
TEXT_T0 = 6.5
TEXT_DUR = 1.8

if sys.platform == "darwin":
    FONT_NAME = "Menlo"
elif sys.platform.startswith("win"):
    FONT_NAME = "Consolas"
else:
    FONT_NAME = "DejaVu Sans Mono"

W_PX, H_PX = COLS * CW, ROWS * CH
NCELL = COLS * ROWS
SCALE = H_PX * 0.38
CAM_D = 5.0

rng = np.random.default_rng(11)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def mix(a, b, k):
    return a + (b - a) * k


def ease_out_back(x):
    x = min(max(x, 0.0), 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Flower:
    LAYERS = [(1.00, 1.00, 0.00, 1.00), (0.80, 0.70, 0.50, 0.85), (0.60, 0.42, 0.25, 0.75)]

    def __init__(self, head, axis, R, n, col, delay):
        self.head, self.R, self.delay = np.array(head, float), R, delay
        self.axis = norm(np.array(axis, float))
        ref = np.array([0, 0, 1.0]) if abs(self.axis[2]) < 0.9 else np.array([1.0, 0, 0])
        self.e1 = norm(np.cross(self.axis, ref))
        self.e2 = np.cross(self.axis, self.e1)
        self.col = np.array(col, float)
        self.spin0 = rng.random() * 6.28
        nu, nv = 12, 7
        U, V, PH, LF, OF, WM, LY = [], [], [], [], [], [], []
        for li, (lf, of, ph, wf) in enumerate(self.LAYERS):
            m = max(5, n - li)
            for k in range(m):
                u = ((np.arange(nu)[:, None] + rng.random((nu, nv))) / nu).ravel()
                v = np.clip(np.linspace(-1, 1, nv)[None, :] +
                            (rng.random((nu, nv)) - 0.5) * (2 / (nv - 1)), -1, 1).ravel()
                U.append(u); V.append(v)
                size = nu * nv
                PH.append(np.full(size, (k + ph) / m * 6.2832 + li * 0.5 + rng.normal(0, 0.05)))
                LF.append(np.full(size, lf)); OF.append(np.full(size, of))
                WM.append(np.full(size, 0.34 * R * (8 / m) * wf)); LY.append(np.full(size, li))
        self.U, self.V, self.PHI = map(np.concatenate, (U, V, PH))
        self.LF, self.OF, self.WM, self.LY = map(np.concatenate, (LF, OF, WM, LY))
        dark, light = self.col * 0.42, mix(self.col, np.array([255.0, 255, 255]), 0.5)
        a = np.clip(self.U / 0.5, 0, 1)[:, None]
        c = mix(dark, self.col, a)
        c = mix(c, light, np.clip((self.U[:, None] - 0.55) / 0.45, 0, 1))
        c = c * (0.90 + 0.10 * np.cos(self.V * 9.4))[:, None]
        c = c * (1 - 0.28 * np.abs(self.V) ** 3)[:, None]
        c = c * (1 - 0.10 * self.LY)[:, None]
        self.color = c
        k = np.arange(260)
        rr = R * 0.24 * np.sqrt(k / 260.0)
        th = k * 2.39996
        self.cx, self.cy = rr * np.cos(th), rr * np.sin(th)
        self.cz = R * 0.13 * (1 - (rr / (R * 0.24)) ** 2)
        gold = np.array([235.0, 175, 45]); brown = np.array([110.0, 62, 18])
        self.ccol = np.where((k % 3 == 0)[:, None], gold, brown)
        self.cn = norm(np.stack([self.cx * 3, self.cy * 3, np.full(260, R * 0.24)], 1))

    def points(self, b, t, tip):
        R = self.R
        L = R * (0.40 + 0.62 * b) * self.LF
        a0 = (0.10 + 0.80 * b) * self.OF
        A = 0.30 + 0.35 * b
        ang = a0 + A * self.U
        rad = L * (np.cos(a0) - np.cos(ang)) / A + R * 0.05
        z = L * (np.sin(ang) - np.sin(a0)) / A
        w = self.WM * np.sin(np.pi * self.U ** 0.75) ** 0.7 * (0.35 + 0.65 * b)
        phi = self.PHI + t * 0.12 + self.spin0
        cph, sph = np.cos(phi), np.sin(phi)
        off = self.V * w
        lx = rad * cph - off * sph
        ly = rad * sph + off * cph
        lz = z + 1.5 * off * off / (L + 1e-3)
        n = np.stack([-np.cos(ang) * cph, -np.cos(ang) * sph, np.sin(ang)], 1)
        n = n + np.stack([-sph, cph, np.zeros_like(cph)], 1) * (-self.V * 0.7)[:, None]
        P = tip + np.outer(lx, self.e1) + np.outer(ly, self.e2) + np.outer(lz, self.axis)
        N = norm(np.outer(n[:, 0], self.e1) + np.outer(n[:, 1], self.e2) + np.outer(n[:, 2], self.axis))
        C = self.color
        G = np.full(len(P), 0.45)
        if b > 0.45:
            s = smooth((b - 0.45) / 0.4)
            Pc = tip + np.outer(self.cx * s, self.e1) + np.outer(self.cy * s, self.e2) + \
                np.outer(self.cz * s + R * 0.02, self.axis)
            Nc = norm(np.outer(self.cn[:, 0], self.e1) + np.outer(self.cn[:, 1], self.e2) +
                      np.outer(self.cn[:, 2], self.axis))
            P = np.vstack([P, Pc]); N = np.vstack([N, Nc])
            C = np.vstack([C, self.ccol]); G = np.concatenate([G, np.full(len(Pc), 0.2)])
        return P, N, C, G


def build_flowers():
    spec = [
        (0.00, 0.72, 0.10, 0.24, 10, (255, 205, 45), 0.10),
        (-0.42, 0.58, 0.05, 0.21, 8, (255, 120, 175), 0.16),
        (0.42, 0.58, 0.05, 0.21, 8, (170, 95, 245), 0.20),
        (-0.80, 0.30, -0.05, 0.19, 7, (110, 160, 255), 0.26),
        (0.80, 0.30, -0.05, 0.19, 7, (255, 140, 55), 0.30),
        (-0.22, 0.30, 0.18, 0.23, 8, (255, 60, 95), 0.06),
        (0.24, 0.32, 0.16, 0.22, 8, (240, 245, 255), 0.04),
        (0.00, 0.47, 0.28, 0.18, 7, (255, 90, 160), 0.12),
        (-0.58, 0.12, 0.00, 0.17, 7, (255, 200, 60), 0.24),
        (0.60, 0.10, 0.00, 0.17, 7, (200, 110, 255), 0.28),
        (-0.28, 0.02, 0.10, 0.15, 6, (150, 190, 255), 0.02),
        (0.18, 0.00, 0.14, 0.15, 6, (255, 110, 90), 0.00),
    ]
    out = []
    for x, y, z, R, n, col, delay in spec:
        axis = np.array([x * 0.6, y * 0.25, 1.0])
        out.append(Flower((x, y, z), axis, R, n, col, delay))
    return out


FLOWERS = build_flowers()
P0 = np.array([0.0, -0.62, 0.0])


def bez(P2, u):
    P1 = np.array([P2[0] * 0.2, (P0[1] + P2[1]) * 0.5, P2[2] * 0.2])
    u = np.asarray(u)[:, None] if np.ndim(u) else u
    return (1 - u) ** 2 * P0 + 2 * (1 - u) * u * P1 + u ** 2 * P2


def bez_tan(P2, u):
    P1 = np.array([P2[0] * 0.2, (P0[1] + P2[1]) * 0.5, P2[2] * 0.2])
    u = np.asarray(u)[:, None] if np.ndim(u) else u
    return norm(2 * (1 - u) * (P1 - P0) + 2 * u * (P2 - P1))


def stem_points(f, p):
    tipfull = f.head - f.axis * 0.03
    n = int(90 * p) + 3
    u = np.linspace(0, p, n)
    pts = bez(tipfull, u)
    tg = bez_tan(tipfull, u)
    e1 = norm(np.cross(tg, np.array([0, 0, 1.0])))
    e2 = np.cross(tg, e1)
    Ps, Ns = [], []
    for a in (0, 2.1, 4.2):
        d = np.cos(a) * e1 + np.sin(a) * e2
        Ps.append(pts + d * 0.014); Ns.append(d)
    P, N = np.vstack(Ps), np.vstack(Ns)
    sh = 0.8 + 0.2 * np.sin(np.tile(u, 3) * 40)
    C = np.stack([34 * sh, 150 * sh, 62 * sh], 1)
    tip = bez(tipfull, np.array([p]))[0]
    return P, N, C, np.full(len(P), 0.2), tip


ENV_W, ENV_TOP, ENV_BOT, ENV_NOTCH = 0.72, -0.2, -0.85, 0.34
ENV_C = np.array([0.0, -0.525, 0.0])
CREAM = np.array([243.0, 231, 208])
PINK = np.array([226.0, 118, 150])


def env_ytop(x):
    return ENV_TOP - ENV_NOTCH * (1 - np.abs(x) / ENV_W)


def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    k = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - (ax + k * vx), py - (ay + k * vy))


def pink_pattern(x, y):
    return PINK * (0.88 + 0.12 * np.sin((x + y) * 95))[:, None]


def make_envelope():
    n = 12000
    x = rng.uniform(-ENV_W, ENV_W, n)
    y = rng.uniform(ENV_BOT, ENV_TOP, n)
    keep = y <= env_ytop(x)
    x, y = x[keep], y[keep]
    Pf = np.stack([x, y, np.full(len(x), 0.03)], 1)
    Cf = np.tile(CREAM, (len(x), 1)) * (0.93 + 0.07 * rng.random(len(x)))[:, None]
    apex = (0.0, ENV_TOP - ENV_NOTCH)
    d = np.minimum(seg_dist(x, y, -ENV_W, ENV_BOT, apex[0], apex[1]),
                   seg_dist(x, y, ENV_W, ENV_BOT, apex[0], apex[1]))
    Cf[d < 0.007] = np.array([196.0, 170, 140])
    edge = (env_ytop(x) - y) < 0.012
    Cf[edge] = np.array([255.0, 248, 232])
    side = (np.abs(x) > ENV_W - 0.01) | (y < ENV_BOT + 0.01)
    Cf[side] = np.array([205.0, 185, 155])
    nb = 12000
    xb = rng.uniform(-ENV_W, ENV_W, nb)
    yb = rng.uniform(ENV_BOT, ENV_TOP, nb)
    Pb = np.stack([xb, yb, np.full(nb, -0.04)], 1)
    Cb = pink_pattern(xb, yb)
    shade = 0.70 + 0.30 * smooth((ENV_TOP - yb) / 0.5)
    Cb = Cb * shade[:, None]
    P = np.vstack([Pf, Pb])
    C = np.vstack([Cf, Cb])
    N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    return P, N, C, np.full(len(P), 0.12)


def make_flap():
    n = 9000
    x = rng.uniform(-ENV_W, ENV_W, n)
    depth = ENV_NOTCH * (1 - np.abs(x) / ENV_W)
    yl = rng.random(n) * depth
    cream = np.tile(CREAM, (n, 1)) * (0.96 + 0.04 * rng.random(n))[:, None]
    edge = (depth - yl) < 0.012
    cream[edge] = np.array([205.0, 185, 155])
    sx, sy = x, yl - 0.20
    seal = np.hypot(sx, sy) < 0.055
    cream[seal] = np.array([200.0, 30, 55]) * (1.0 - 0.35 * np.hypot(sx, sy)[seal] / 0.055)[:, None]
    gloss = np.where(seal, 0.7, 0.08)
    pink = pink_pattern(x, yl)
    return x, yl, cream, pink, gloss


FLAP_X, FLAP_YL, FLAP_OUT, FLAP_IN, FLAP_G = make_flap()
ENV_P, ENV_N, ENV_C_COL, ENV_G = make_envelope()


def flap_points(a):
    ca, sa = math.cos(a), math.sin(a)
    yrel = -FLAP_YL
    out_z, in_z = 0.035, 0.02
    y_o = ENV_TOP + yrel * ca - out_z * sa
    z_o = yrel * sa + out_z * ca
    y_i = ENV_TOP + yrel * ca - in_z * sa
    z_i = yrel * sa + in_z * ca
    Po = np.stack([FLAP_X, y_o, z_o], 1)
    Pi = np.stack([FLAP_X, y_i, z_i], 1)
    no = np.tile([0.0, -sa, ca], (len(Po), 1))
    ni = -no
    return Po, no, Pi, ni


def make_leaf(root, d, length, wd, droop):
    nu, nv = 16, 7
    u = ((np.arange(nu)[:, None] + rng.random((nu, nv))) / nu).ravel()
    v = np.clip(np.linspace(-1, 1, nv)[None, :] + (rng.random((nu, nv)) - 0.5) * 0.3, -1, 1).ravel()
    d = norm(np.array(d, float))
    up = np.array([0, -1.0, 0])
    side = norm(np.cross(d, np.array([0, 1.0, 0])))
    C0 = root + np.outer(u, d * length) + np.outer(u * u, up * droop * length)
    tg = norm(np.outer(np.ones_like(u), d * length) + np.outer(u, up * 2 * droop * length))
    w = wd * np.sin(np.pi * u) ** 0.7
    P = C0 + np.outer(v * w, side)
    N = norm(np.cross(side, tg))
    rib = np.abs(v) < 0.10
    base = mix(np.array([28.0, 115, 48]), np.array([70.0, 175, 85]), np.abs(v)[:, None] * 0.8)
    base[rib] = np.array([130.0, 205, 120])
    base = base * (0.75 + 0.25 * u)[:, None]
    return P, N, base, np.full(len(P), 0.35), np.tile(root, (len(P), 1))


def make_leaves():
    out = []
    for k in range(9):
        th = math.pi * (0.1 + 0.8 * k / 8)
        root = np.array([0.32 * math.cos(th), -0.28, 0.04 * math.sin(th)])
        d = (math.cos(th) * 0.95, 0.75, 0.35)
        out.append(make_leaf(root, d, 0.42, 0.07, 0.45))
    P, N, C, G, R = (np.vstack(x) if x[0].ndim > 1 else np.concatenate(x) for x in zip(*out))
    return P, N, C, G, R


LEAVES = make_leaves()

GLYPHS = {
    "П": ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#"],
    "О": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "Ш": ["#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#####"],
    "Л": ["..###", ".#..#", ".#..#", ".#..#", ".#..#", "#...#", "#...#"],
    "И": ["#...#", "#..##", "#.#.#", "#.#.#", "##..#", "#...#", "#...#"],
    "Г": ["#####", "#....", "#....", "#....", "#....", "#....", "#...."],
    "У": ["#...#", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "Я": [".####", "#...#", "#...#", ".####", "..#.#", ".#..#", "#...#"],
    "Т": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "Ь": ["#....", "#....", "#....", "####.", "#...#", "#...#", "####."],
    "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
}
LINES = ["ПОШЛИ", "ГУЛЯТЬ?"]


def build_text():
    tx, ty = [], []
    r0 = ROWS - 16
    for li, line in enumerate(LINES):
        width = len(line) * 12 - 2
        c0 = (COLS - width) // 2
        for gi, ch in enumerate(line):
            for ry, row in enumerate(GLYPHS[ch]):
                for rx, v in enumerate(row):
                    if v == "#":
                        for dx in (0, 1):
                            tx.append(c0 + gi * 12 + rx * 2 + dx)
                            ty.append(r0 + li * 8 + ry)
    return np.array(tx, float), np.array(ty, float)


TX, TY = build_text()
NT = len(TX)
SX = TX + (rng.random(NT) - 0.5) * 60
SY = ROWS + 4 + rng.random(NT) * 26
TDEL = rng.random(NT) * 2.6
TPH = rng.random(NT) * 6.28

yy, xx = np.mgrid[0:ROWS, 0:COLS]
CXS = ((xx + 0.5) * CW).ravel().astype(float)
CYS = ((yy + 0.5) * CH).ravel().astype(float)
vig = 1 - 0.75 * (((xx / COLS - 0.5) * 1.3) ** 2 + ((yy / ROWS - 0.5) * 1.1) ** 2)
vig = np.clip(vig, 0.1, 1).ravel()
BG = np.stack([3 + 4 * vig, 8 + 9 * vig, 7 + 8 * vig], 1)
STAR_IDX = rng.choice(NCELL, 160, replace=False)
STAR_PH = rng.random(160) * 6.28
FIRE = rng.random((110, 4))
GOLD_P = np.array([255.0, 226, 150])
ICE_P = np.array([170.0, 205, 255])
FCOL = np.where((rng.random(110) > 0.5)[:, None], GOLD_P, ICE_P)
PET = rng.random((18, 5))
PET_COL = np.array([f.col for f in FLOWERS])
RND = rng.integers(0, 256, NCELL)
IDX8 = np.arange(NCELL) & 7

LIGHT = norm(np.array([-0.45, 0.65, 0.62]))
FILL = norm(np.array([0.6, -0.2, 0.5]))
HALF = norm(LIGHT + np.array([0, 0, 1.0]))


def render(t, frame, yaw_extra=0.0):
    e_in = ease_out_back(t / 1.5)
    env_s = 0.35 + 0.65 * e_in
    env_dy = -(1 - e_in) * 0.5
    env_fade = float(smooth(t / 0.7))
    flap_a = float(smooth((t - 1.7) / 1.4)) * 2.9
    L = float(np.clip((t - 2.9) / 6.0, 0, 1))

    pts_P, pts_N, pts_C, pts_G, pts_K = [], [], [], [], []

    def add(P, N, C, G, K=None):
        pts_P.append(P); pts_N.append(N); pts_C.append(C); pts_G.append(G)
        pts_K.append(np.zeros(len(P), bool) if K is None else K)

    def env_tf(P):
        return ENV_C + (P - ENV_C) * env_s + np.array([0.0, env_dy, 0.0])

    add(env_tf(ENV_P), ENV_N, ENV_C_COL * env_fade, ENV_G)
    Po, no, Pi, ni = flap_points(flap_a)
    add(env_tf(Po), no, FLAP_OUT * env_fade, FLAP_G, np.ones(len(Po), bool))
    add(env_tf(Pi), ni, FLAP_IN * env_fade, np.full(len(Pi), 0.08), np.ones(len(Pi), bool))

    heads = []
    if L > 0:
        for f in FLOWERS:
            p = float(smooth((L - f.delay * 0.4) / 0.42))
            if p <= 0.001:
                continue
            P, N, C, G, tip = stem_points(f, p)
            add(P, N, C, G)
            b = float(smooth((L - 0.22 - f.delay) / 0.5)) * (1 + 0.025 * math.sin(t * 1.3 + f.spin0))
            b = min(1.0, b)
            P, N, C, G = f.points(b, t, tip)
            add(P, N, C, G)
            heads.append((tip, f.R * (0.3 + 0.7 * b), f.col, b))
        k = float(smooth((L - 0.05) / 0.4))
        if k > 0:
            P, N, C, G, R = LEAVES
            add(R + (P - R) * k, N, C, G)

    fade = float(smooth((t - 9.0) / 2.0))
    if fade > 0.02:
        x = (PET[:, 0] - 0.5) * 2.4 + 0.15 * np.sin(t * 0.9 + PET[:, 1] * 6)
        y = 1.1 - ((1 - PET[:, 1]) * 2.1 + t * (0.10 + 0.12 * PET[:, 2])) % 2.1
        z = (PET[:, 3] - 0.5) * 1.2 + 0.15 * np.cos(t * 0.8 + PET[:, 0] * 6)
        pc = PET_COL[(PET[:, 4] * len(PET_COL)).astype(int)]
        for dx in (0.0, 0.012, 0.024):
            add(np.stack([x + dx, y - dx * 0.5, z], 1), np.tile([0.3, 0.8, 0.5], (len(x), 1)),
                pc * fade, np.full(len(x), 0.3))

    P = np.vstack(pts_P); N = np.vstack(pts_N); C = np.vstack(pts_C)
    G = np.concatenate(pts_G); K = np.concatenate(pts_K)

    yaw = 0.45 * math.sin(t * 0.45) + yaw_extra
    pitch = 0.28 + 0.05 * math.sin(t * 0.3)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    def rot(A):
        x = A[:, 0] * cy_ + A[:, 2] * sy_
        z = -A[:, 0] * sy_ + A[:, 2] * cy_
        y = A[:, 1] * cp - z * sp
        z2 = A[:, 1] * sp + z * cp
        return x, y, z2

    def screen(x, y, z):
        f = CAM_D / (CAM_D - z)
        return W_PX / 2 + x * f * SCALE, H_PX * 0.40 - (y - 0.03) * f * SCALE, f

    x, y, z = rot(P)
    nx, ny, nz = rot(N)
    keep = ~(K & (nz < 0))
    x, y, z, nx, ny, nz, C, G = x[keep], y[keep], z[keep], nx[keep], ny[keep], nz[keep], C[keep], G[keep]
    sgn = np.where(nz < 0, -1.0, 1.0)
    nx, ny, nz = nx * sgn, ny * sgn, nz * sgn
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    fil = np.clip(nx * FILL[0] + ny * FILL[1] + nz * FILL[2], 0, 1)
    spec = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1) ** 40 * G
    rim = (1 - nz) ** 2.5 * 0.10
    shade = (0.36 + 0.80 * lam + 0.22 * fil + rim)[:, None]
    depth = (0.82 + 0.18 * np.clip((z + 1.0) / 2.0, 0, 1))[:, None]
    col = C * shade * depth + (255 * 0.28 * spec)[:, None]

    px, py, _ = screen(x, y, z)
    cc = (px // CW).astype(int)
    rr = (py // CH).astype(int)
    ok = (cc >= 0) & (cc < COLS) & (rr >= 0) & (rr < ROWS)
    idx = (rr * COLS + cc)[ok]
    col, zz = col[ok], z[ok]
    order = np.argsort(zz)
    fg = np.zeros((NCELL, 3)); zb = np.full(NCELL, -9.0); mask = np.zeros(NCELL, bool)
    fg[idx[order]] = col[order]
    zb[idx[order]] = zz[order]
    mask[idx] = True

    bg = BG.copy()
    tw = 0.5 + 0.5 * np.sin(t * 1.6 + STAR_PH)
    bg[STAR_IDX] += (np.stack([60 + 140 * tw, 70 + 140 * tw, 90 + 150 * tw], 1)) * 0.75
    for tip, rad, hc, b in heads:
        if b < 0.05:
            continue
        hx_, hy_, hz_ = rot(tip[None, :])
        hpx, hpy, ff = screen(hx_, hy_, hz_)
        sig = rad * ff[0] * SCALE * 0.9
        d2 = (CXS - hpx[0]) ** 2 + (CYS - hpy[0]) ** 2
        g = np.exp(-d2 / (2 * sig * sig)) * 0.10 * b
        bg += g[:, None] * hc

    out = np.where(mask[:, None], fg, bg)

    ffade = float(smooth(t / 1.2))
    if ffade > 0.02:
        fx = (FIRE[:, 0] - 0.5) * 2.6 + 0.12 * np.sin(t * 0.5 + FIRE[:, 3] * 6)
        fy = 1.1 - ((1 - FIRE[:, 1]) * 2.2 + t * (0.04 + 0.08 * FIRE[:, 2])) % 2.2
        fz = (FIRE[:, 2] - 0.5) * 1.2 + 0.12 * np.cos(t * 0.6 + FIRE[:, 0] * 6)
        ax, ay, az = rot(np.stack([fx, fy, fz], 1))
        fpx, fpy, _ = screen(ax, ay, az)
        fc = (fpx // CW).astype(int)
        fr = (fpy // CH).astype(int)
        tw2 = 0.5 + 0.5 * np.sin(t * 2.2 + FIRE[:, 3] * 19)
        for dr, dc, kk in ((0, 0, 1.0), (1, 0, .18), (-1, 0, .18), (0, 1, .12), (0, -1, .12)):
            c2, r2 = fc + dc, fr + dr
            v = (c2 >= 0) & (c2 < COLS) & (r2 >= 0) & (r2 < ROWS)
            ii = (r2 * COLS + c2)[v]
            vis = (az[v] > zb[ii]) | ~mask[ii]
            ii = ii[vis]
            a = (kk * ffade * tw2[v][vis] * 0.95)[:, None]
            np.add.at(out, ii, a * FCOL[v][vis])

    tau = (t - TEXT_T0 - TDEL) / TEXT_DUR
    e = 1 - (1 - np.clip(tau, 0, 1)) ** 3
    tx = SX + (TX - SX) * e + np.sin(t * 3 + TPH) * (1 - e) * 2.5
    ty = SY + (TY - SY) * e
    live = tau > 0
    shimmer = 0.88 + 0.12 * np.sin(t * 3 + TPH)
    tcol = mix(np.array([140.0, 200, 255]), np.array([255.0, 226, 160]), e[:, None])
    tcol = tcol * (0.55 + 0.45 * e)[:, None] * np.where(e >= 1, shimmer, 1.0)[:, None]
    text_mask = np.zeros(NCELL, bool)
    for dy, kk in ((1.6, 0.35), (0.0, 1.0)):
        r2 = np.rint(ty + dy * (1 - e)).astype(int)
        c2 = np.rint(tx).astype(int)
        v = live & (r2 >= 0) & (r2 < ROWS) & (c2 >= 0) & (c2 < COLS)
        ii = r2[v] * COLS + c2[v]
        out[ii] = tcol[v] * kk
        mask[ii] = True
        text_mask[ii] = True

    rgb = np.clip(out, 0, 255).astype(np.uint8)
    phase = (frame + IDX8) // 8
    bit_dyn = (RND >> (phase % 8)) & 1
    bit = np.where(mask, bit_dyn, RND & 1)
    lum = out.max(axis=1)
    level = np.where(mask & ~text_mask, (lum >= 45).astype(int) + (lum >= 238).astype(int), 1)
    code = (level * 2 + bit).astype(np.uint8)
    return rgb, code


CHARS = ".:018#"


def main():
    import tkinter as tk

    root = tk.Tk()
    root.title("Пошли гулять?")
    root.configure(bg="black")
    root.resizable(False, False)
    cv = tk.Canvas(root, width=W_PX, height=H_PX, bg="black", highlightthickness=0, bd=0)
    cv.pack()

    font = (FONT_NAME, 8, "bold")
    items = [cv.create_text((i % COLS) * CW + CW / 2, (i // COLS) * CH + CH / 2,
                            text="0", fill="#000000", font=font) for i in range(NCELL)]
    H = ["%02x" % v for v in range(256)]
    shown = np.full((NCELL, 3), -100, dtype=np.int16)
    shown_code = np.full(NCELL, 99, dtype=np.int16)
    root.update()
    st = {"paused": False, "t": 0.0, "frame": 0, "t0": time.perf_counter(), "yaw": 0.0}

    def tick():
        t_start = time.perf_counter()
        if not st["paused"]:
            st["t"] = t_start - st["t0"]
            st["frame"] += 1
            rgb, code = render(st["t"], st["frame"], st["yaw"])
            diff = np.abs(rgb.astype(np.int16) - shown).max(axis=1)
            changed = np.nonzero((diff >= 9) | (code != shown_code))[0]
            shown[changed] = rgb[changed]
            shown_code[changed] = code[changed]
            for i in changed.tolist():
                r, g, b = rgb[i]
                cv.itemconfig(items[i], text=CHARS[code[i]], fill="#" + H[r] + H[g] + H[b])
        spent = int((time.perf_counter() - t_start) * 1000)
        root.after(max(1, int(1000 / FPS) - spent), tick)

    def toggle(_e=None):
        if st["paused"]:
            st["t0"] = time.perf_counter() - st["t"]
        st["paused"] = not st["paused"]

    def replay(_e=None):
        st["t0"] = time.perf_counter()
        st["paused"] = False

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<space>", toggle)
    root.bind("r", replay)
    root.bind("<Left>", lambda e: st.update(yaw=st["yaw"] - 0.15))
    root.bind("<Right>", lambda e: st.update(yaw=st["yaw"] + 0.15))
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()