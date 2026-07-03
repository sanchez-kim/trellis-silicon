"""Dependency-light comparison renderer (trimesh + numpy + PIL only).

Vertex-splat z-buffer rasterizer: projects each mesh under a FIXED orthographic
camera (identical across variants), 3 views (front / 3-4 / side). Each vertex is
splatted as a small depth-tested disc, shaded by normal.light and tinted by the
baked base color sampled per-vertex. Good enough to judge shape fidelity (holes,
silhouette, surface detail) and texture. Composites a labeled grid via PIL.
"""
import sys, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont

RES = 512
SPLAT = 2                      # disc radius in px (fills gaps between verts)
FRAME = 0.62                   # world half-extent mapped to image (fixed camera)
LIGHT = np.array([0.4, 0.7, 0.6]); LIGHT /= np.linalg.norm(LIGHT)
BG = 18

def load(path):
    s = trimesh.load(path, process=False)
    g = list(s.geometry.values())[0] if isinstance(s, trimesh.Scene) else s
    v = np.asarray(g.vertices, np.float32)
    n = np.asarray(g.vertex_normals, np.float32)
    try:
        col = np.asarray(g.visual.to_color().vertex_colors, np.float32)[:, :3] / 255.0
        if col.shape[0] != v.shape[0]:
            col = np.full((v.shape[0], 3), 0.8, np.float32)
    except Exception:
        col = np.full((v.shape[0], 3), 0.8, np.float32)
    return v, n, col

def roty(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)

def render(v, n, col, R):
    vv = v @ R.T; nn = n @ R.T
    # orthographic: x right, y up, z toward camera (+z closer)
    xs = ((vv[:, 0] / FRAME) * 0.5 + 0.5) * (RES - 1)
    ys = ((-vv[:, 1] / FRAME) * 0.5 + 0.5) * (RES - 1)
    z = vv[:, 2]
    shade = np.clip(nn @ LIGHT, 0, 1) * 0.8 + 0.2
    rgb = np.clip(col * shade[:, None], 0, 1)
    img = np.full((RES, RES, 3), BG / 255.0, np.float32)
    zbuf = np.full((RES, RES), -1e9, np.float32)
    order = np.argsort(z)              # far -> near, painter within splat via zbuf
    xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
    for dx in range(-SPLAT, SPLAT + 1):
        for dy in range(-SPLAT, SPLAT + 1):
            if dx * dx + dy * dy > SPLAT * SPLAT: continue
            px = xi[order] + dx; py = yi[order] + dy
            m = (px >= 0) & (px < RES) & (py >= 0) & (py < RES)
            px, py = px[m], py[m]; zo = z[order][m]; co = rgb[order][m]
            closer = zo > zbuf[py, px]
            px, py, zo, co = px[closer], py[closer], zo[closer], co[closer]
            # resolve duplicate pixel writes: keep nearest via lexsort
            idx = py * RES + px
            srt = np.argsort(zo)        # ascending; later writes (nearer) win
            img[py[srt], px[srt]] = co[srt]
            zbuf[py[srt], px[srt]] = zo[srt]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255))
    return im

def main():
    variants = [("test_steps12.glb", "steps=12 (baseline)"),
                ("test_steps8.glb", "steps=8"),
                ("test_steps6.glb", "steps=6")]
    views = [("front", 0), ("3/4", 40), ("side", 90)]
    rows = []
    for path, name in variants:
        v, n, col = load(path)
        tiles = []
        for vn, deg in views:
            im = Image.fromarray(render(v, n, col, roty(deg)))
            tiles.append(label(im, f"{name}  [{vn}]  {len(v):,}v"))
        row = Image.new("RGB", (RES * 3, RES), (0, 0, 0))
        for i, t in enumerate(tiles): row.paste(t, (i * RES, 0))
        rows.append(row)
        row.save(f"steps_comparison_{name.split()[0].replace('=','')}.png")
    grid = Image.new("RGB", (RES * 3, RES * 3), (0, 0, 0))
    for i, r in enumerate(rows): grid.paste(r, (0, i * RES))
    grid.save("steps_comparison_grid.png")
    print("wrote steps_comparison_grid.png and per-variant rows")

if __name__ == "__main__":
    main()
