# Clawd as real 3D mesh (ASCII STL). Units ~ cm. Eyes are recessed sockets.
boxes = []
def box(cx, cy, cz, sx, sy, sz):
    boxes.append((cx - sx, cy - sy, cz - sz, cx + sx, cy + sy, cz + sz))

box(0, 0.95, 0, 1.0, 0.62, 0.55)            # body
box(-1.18, 0.95, 0, 0.24, 0.17, 0.2)        # arm L
box(1.18, 0.95, 0, 0.24, 0.17, 0.2)         # arm R
for x in (-0.72, -0.36, 0.36, 0.72):
    box(x, 0.2, 0, 0.11, 0.2, 0.15)         # legs

def quads(b):
    x0, y0, z0, x1, y1, z1 = b
    v = lambda i, j, k: ((x0, x1)[i], (y0, y1)[j], (z0, z1)[k])
    return [((0, 0, -1), [v(0,0,0), v(0,1,0), v(1,1,0), v(1,0,0)]),
            ((0, 0, 1),  [v(0,0,1), v(1,0,1), v(1,1,1), v(0,1,1)]),
            ((-1, 0, 0), [v(0,0,0), v(0,0,1), v(0,1,1), v(0,1,0)]),
            ((1, 0, 0),  [v(1,0,0), v(1,1,0), v(1,1,1), v(1,0,1)]),
            ((0, -1, 0), [v(0,0,0), v(1,0,0), v(1,0,1), v(0,0,1)]),
            ((0, 1, 0),  [v(0,1,0), v(0,1,1), v(1,1,1), v(1,1,0)])]

out = ["solid clawd"]
def tri(n, a, b, c):
    # STL: z-up -> swap y/z so it stands upright in GitHub viewer
    sw = lambda p: (p[0], -p[2], p[1])
    n2 = sw(n)
    out.append(f" facet normal {n2[0]} {n2[1]} {n2[2]}\n  outer loop")
    for p in (a, b, c):
        q = sw(p); out.append(f"   vertex {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")
    out.append("  endloop\n endfacet")

for b in boxes:
    for n, (a, bb, c, d) in quads(b):
        tri(n, a, bb, c); tri(n, a, c, d)
# eyes: dark "pupils" as slightly protruding plates (single-color viewer -> relief)
for ex in (-0.4, 0.4):
    for n, (a, bb, c, d) in quads((ex - 0.085, 0.9, 0.55, ex + 0.085, 1.3, 0.62)):
        tri(n, a, bb, c); tri(n, a, c, d)
out.append("endsolid clawd")
open("../assets/clawd.stl", "w").write("\n".join(out) + "\n")
print("facets", sum(1 for l in out if "facet normal" in l))
