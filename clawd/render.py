# Clawd 3D raymarch renderer (numpy SDF) -> frames PNG
import numpy as np, sys, os
from PIL import Image

SS = int(os.environ.get("SS", 1))
W0, H0 = int(os.environ.get("W", 480)), int(os.environ.get("H", 360))
W, H = W0 * SS, H0 * SS
N = int(os.environ.get("N", 48))
OUT = os.environ.get("OUT", "frames")
BG = np.array([13, 17, 23]) / 255.0
os.makedirs(OUT, exist_ok=True)

ORANGE = np.array([0.86, 0.43, 0.27])  # #D97757 claude
DARK = np.array([0.06, 0.05, 0.05])

def sd_rbox(p, c, b, r):
    q = np.abs(p - c) - (np.array(b) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r

def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], -1)

def scene(p, t):
    ph = 2 * np.pi * t
    yaw = 0.3 * np.sin(ph)
    bob = 0.06 * np.abs(np.sin(2 * ph))
    q = rot_y(p, -yaw)
    q = q - np.array([0, bob, 0])
    body = sd_rbox(q, np.array([0, 0.95, 0]), (1.0, 0.62, 0.55), 0.06)
    # arms (wave)
    wave = 0.12 * np.sin(4 * ph)
    armL = sd_rbox(q, np.array([-1.18, 0.95 + wave, 0]), (0.24, 0.17, 0.2), 0.04)
    armR = sd_rbox(q, np.array([1.18, 0.95 - wave, 0]), (0.24, 0.17, 0.2), 0.04)
    d = np.minimum(body, np.minimum(armL, armR))
    # legs (4, alternating step)
    for i, x in enumerate((-0.72, -0.36, 0.36, 0.72)):
        lift = 0.07 * max(0.0, np.sin(4 * ph + i * np.pi))
        ly = 0.2 + lift - bob
        leg = sd_rbox(q, np.array([x, ly + 0.0, 0]), (0.11, 0.2, 0.15), 0.03)
        d = np.minimum(d, leg)
    # eyes (inset black) - blink
    blink = 1.0
    tt = (t * 2) % 1.0
    if 0.46 < tt < 0.54:
        blink = 0.15
    eh = 0.2 * blink
    eyes = np.minimum(
        sd_rbox(q, np.array([-0.4, 1.1, 0.55]), (0.085, eh, 0.06), 0.02),
        sd_rbox(q, np.array([0.4, 1.1, 0.55]), (0.085, eh, 0.06), 0.02))
    mat = np.where(eyes < 0.004, 1, 0)
    ground = p[..., 1] + 0.0
    m = np.where(ground < d, 2, mat)
    return np.minimum(d, ground), m

def normal(p, t):
    e = 0.002
    ex = np.array([e, 0, 0]); ey = np.array([0, e, 0]); ez = np.array([0, 0, e])
    n = np.stack([scene(p + ex, t)[0] - scene(p - ex, t)[0],
                  scene(p + ey, t)[0] - scene(p - ey, t)[0],
                  scene(p + ez, t)[0] - scene(p - ez, t)[0]], -1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

def march(ro, rd, t, steps=90, tmax=20):
    dist = np.zeros(ro.shape[0]); hit = np.zeros(ro.shape[0], bool)
    alive = np.ones(ro.shape[0], bool)
    for _ in range(steps):
        idx = np.where(alive)[0]
        if idx.size == 0: break
        d, _ = scene(ro[idx] + rd[idx] * dist[idx, None], t)
        dist[idx] += d
        h = d < 0.001; hit[idx[h]] = True
        alive[idx[h | (dist[idx] > tmax)]] = False
    return dist, hit

def shadow(ro, rd, t, k=10):
    res = np.ones(ro.shape[0]); s = np.full(ro.shape[0], 0.03)
    for _ in range(40):
        d, _ = scene(ro + rd * s[:, None], t)
        res = np.minimum(res, k * np.maximum(d, 0) / s)
        s += np.clip(d, 0.02, 0.3)
    return np.clip(res, 0, 1)

def ao(p, n, t):
    o = np.zeros(p.shape[0])
    for i in range(1, 5):
        h = 0.06 * i
        d, _ = scene(p + n * h, t)
        o += (h - d) / (2 ** i)
    return np.clip(1 - 1.6 * o, 0.35, 1)

def render(fi):
    t = fi / N
    cam = np.array([0.0, 1.95, 3.5]); look = np.array([0, 0.66, 0])
    f = look - cam; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    ys, xs = np.mgrid[0:H, 0:W]
    px = (2 * (xs + 0.5) - W) / H; py = -(2 * (ys + 0.5) - H) / H
    rd = (px[..., None] * r + py[..., None] * u + 1.9 * f).reshape(-1, 3)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(cam, rd.shape).copy()
    # background: dark github gradient
    # flat GitHub dark bg (#0d1117) so the GIF blends into the page
    col = np.broadcast_to(BG, (H * W, 3)).copy()
    dist, hit = march(ro, rd, t)
    idx = np.where(hit)[0]
    p = ro[idx] + rd[idx] * dist[idx, None]
    _, m = scene(p, t)
    n = normal(p, t)
    L = np.array([-0.5, 0.85, 0.6]); L /= np.linalg.norm(L)
    dif = np.clip((n * L).sum(-1), 0, 1)
    sh = shadow(p + n * 0.01, np.broadcast_to(L, p.shape), t)
    a = ao(p, n, t)
    hv = L - rd[idx]; hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
    spec = np.clip((n * hv).sum(-1), 0, 1) ** 32
    rim = np.clip(1 + (n * rd[idx]).sum(-1), 0, 1) ** 3
    base = np.where(m[:, None] == 1, DARK, ORANGE)
    sky = 0.5 + 0.5 * n[:, 1]
    fill = np.clip((n * np.array([0.8, 0.25, 0.5])).sum(-1), 0, 1)
    c = base * (1.05 * dif * sh + 0.38 * sky * a + 0.45 * fill * a)[:, None] \
        + (0.35 * spec * sh)[:, None] * np.where(m[:, None] == 1, 1.0, 0.6) \
        + (0.5 * rim * a)[:, None] * np.array([0.35, 0.55, 1.0])
    # ground: floor with shadow, fades into bg
    gi = m == 2
    fl = np.exp(-0.12 * (p[gi, 0] ** 2 + p[gi, 2] ** 2) * 1.0)
    # lit floor == BG exactly; only the shadow darkens it
    floor = BG * (0.05 + 0.95 * sh[gi] * a[gi])[:, None]
    bgc = col[idx[gi]]
    c[gi] = bgc * (1 - fl[:, None]) + floor * fl[:, None]
    # glow ring under clawd
    c = np.clip(c, 0, 1)
    col[idx] = c
    
    img = (np.clip(col, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)
    Image.fromarray(img).resize((W0, H0), Image.LANCZOS).save(f"{OUT}/f{fi:03d}.png")

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for i in range(a, b):
        render(i); print("frame", i, flush=True)
