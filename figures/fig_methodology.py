"""Methodology overview for Paper A, drawn in code (no generative AI).

Every image inside the panels is study data: Landsat-8 date-2 scenes, the preprocessed reference with the
tile split, and a real 512 x 512 RO tile before and after scene-level FDA toward PA (from the FDA cache).
All settings in the text are the ones in the code (dataset_cd.py, train_cd.py, run_sweep_cd.py).
Output: paper/figures/fig_methodology.{png,pdf}, 180 mm wide, 600 dpi.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
sys.path.insert(0, BASE)
from dataset_cd import DOMAIN_DATA, _load_band_stack, build_domain_arrays, tile_split

OUT = os.path.join(BASE, "paper", "figures", "fig_methodology")
MM = 1 / 25.4
W, H = 180.0, 122.0
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED = "#1a1a1a", "#4a4a4a"
COL = {1: "#5f8fbf", 2: "#c9a227", 3: "#6250d6", 4: "#1baf7a", 5: "#eb6834"}
NEW_DEF = (0.847, 0.106, 0.376)

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")


def tint(c, f):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def t(x, y, s, size=6.6, ha="left", va="center", weight="normal", color=INK, **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight=weight, color=color, zorder=10, **kw)


def panel(x, y, w, h, n, title):
    c = COL[n]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2", fc=tint(c, 0.9), ec=c,
                                lw=1.0, zorder=1))
    ax.add_patch(Circle((x + 4.0, y + h - 4.0), 2.6, fc="white", ec=c, lw=1.0, zorder=3))
    t(x + 4.0, y + h - 4.05, str(n), size=8, ha="center", weight="bold", color=c)
    t(x + 8.2, y + h - 4.0, title, size=8, weight="bold")


def card(x, y, w, h, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=ec, lw=0.6,
                                zorder=2))


def img(arr, x, y, w, h, interp="lanczos", ec=INK):
    ax.imshow(arr, extent=(x, x + w, y, y + h), zorder=4, interpolation=interp, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.5, zorder=5))


def flow(x0, y0, x1, y1, color="#6b6b6b", width=2.2):
    d = np.array([x1 - x0, y1 - y0], float); L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    head = 2.4; p0 = np.array([x0, y0]); pb = p0 + u * (L - head)
    pts = [p0 + n * width / 2, pb + n * width / 2, pb + n * width, p0 + u * L, pb - n * width, pb - n * width / 2,
           p0 - n * width / 2]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=2))


def arr(x0, y0, x1, y1, color=INK, lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=1.4,head_width=0.8",
                                 mutation_scale=1.6, color=color, lw=lw, ls=ls, zorder=6, shrinkA=0, shrinkB=0))


def rgb_scene(dk, step):
    a, nd = _load_band_stack(os.path.join(BASE, DOMAIN_DATA[dk]["img_t2"]), return_nodata=True)
    a = a[:, ::step, ::step]; ok = ~nd[::step, ::step]
    o = np.zeros(a.shape[1:] + (3,), np.float32)
    for k, b in enumerate((3, 2, 1)):           # OLI B4, B3, B2
        lo, hi = np.percentile(a[b][ok], [2, 98]); o[..., k] = np.clip((a[b] - lo) / (hi - lo), 0, 1) ** 0.85
    return o


def rgb_norm(X, sl, ref=None):
    """True colour of date 2 from a normalized 14-band array (channels 7..13 = date 2 B1..B7)."""
    bands = [7 + 3, 7 + 2, 7 + 1]
    o = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, 3), np.float32)
    lims = []
    for k, b in enumerate(bands):
        v = np.asarray(X[b, sl[0], sl[1]], np.float32)
        lo, hi = (np.percentile(v, [2, 98]) if ref is None else ref[k])
        lims.append((lo, hi))
        o[..., k] = np.clip((v - lo) / (hi - lo), 0, 1) ** 0.85
    return o, lims


# ---------------------------------------------------------------- layout
R1y, R1h = 62.0, 58.0
R2y, R2h = 2.0, 55.0
P1 = (2.0, R1y, 54.0, R1h); P2 = (61.0, R1y, 56.0, R1h); P3 = (122.0, R1y, 56.0, R1h)
P4 = (2.0, R2y, 86.0, R2h); P5 = (93.0, R2y, 85.0, R2h)
flow(P1[0] + P1[2] + 0.3, R1y + 30, P2[0] - 0.3, R1y + 30)
flow(P2[0] + P2[2] + 0.3, R1y + 30, P3[0] - 0.3, R1y + 30)
flow(P4[0] + P4[2] + 0.3, R2y + 28, P5[0] - 0.3, R2y + 28)
xs, xe = P3[0] + P3[2] / 2, P4[0] + P4[2] / 2
ym = (R1y + R2y + R2h) / 2
ax.add_line(Line2D([xs, xs, xe], [R1y - 0.2, ym, ym], color="#6b6b6b", lw=1.6, zorder=2))
flow(xe, ym + 0.8, xe, R2y + R2h + 0.3, width=1.6)

# ---------------------------------------------------------------- 1 data
x, y, w, h = P1
panel(x, y, w, h, 1, "Data")
thumbs = {"PA": rgb_scene("PA", 4), "RO": rgb_scene("RO", 8), "MA": rgb_scene("MA", 4)}
ty = y + 27.0
img(thumbs["PA"], x + 3.0, ty + 12.5, 26.0, 26.0 * thumbs["PA"].shape[0] / thumbs["PA"].shape[1])
img(thumbs["RO"], x + 3.0, ty, 24.0, 24.0 * thumbs["RO"].shape[0] / thumbs["RO"].shape[1])
img(thumbs["MA"], x + 31.0, ty, 11.0 * thumbs["MA"].shape[1] / thumbs["MA"].shape[0], 11.0)
for lab, (lx, ly) in {"PA": (x + 4.0, ty + 23.0), "RO": (x + 4.0, ty + 10.5), "MA": (x + 32.0, ty + 9.5)}.items():
    t(lx, ly, lab, size=6.4, weight="bold", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
lines = ["3 sites: PA, RO (Amazon), MA (Cerrado)", "Landsat-8 OLI, bands B1–B7, 30 m",
         "Two dates per site, 1 year apart", "PRODES new and past deforestation",
         "6 directed source→target pairs"]
for i, s in enumerate(lines):
    t(x + 3.0, y + 22.0 - i * 3.8, "• " + s, size=6.2)

# ---------------------------------------------------------------- 2 preprocessing
x, y, w, h = P2
panel(x, y, w, h, 2, "Preprocessing")
_, yp, vp = build_domain_arrays("PA")
yy = np.load(yp)[::4, ::4].astype(bool); vv = np.load(vp)[::4, ::4].astype(bool)
ref = np.ones(yy.shape + (3,), np.float32) * 0.92
ref[~vv] = 0.25; ref[vv & yy] = NEW_DEF
rw = 50.0; rh = rw * ref.shape[0] / ref.shape[1]
ry = y + h - 10.5 - rh
img(ref, x + 3.0, ry, rw, rh, interp="nearest")
gx, gy = DOMAIN_DATA["PA"]["grid"]; sp = tile_split("PA")
for r in range(gx):
    for c in range(gy):
        tid = r * gy + c
        kind = "train" if tid in sp["train"] else "val" if tid in sp["val"] else None
        if kind:
            ax.add_patch(Rectangle((x + 3.0 + c * rw / gy, ry + rh - (r + 1) * rh / gx), rw / gy, rh / gx, fc="none",
                                   ec="#1a1a1a" if kind == "train" else "#2a8fc4", lw=0.8,
                                   ls="-" if kind == "train" else (0, (2.5, 1.2)), zorder=6))
t(x + 3.0, ry - 1.8, "PA reference; training (solid) and validation (dashed) tiles", size=5.6, va="top",
  color=MUTED)
lines = ["Input check: 16-bit DN, ≤ 1 % no data, labels 0/1", "z-score per band, scene and date; 14 bands",
         "Exclude: past deforestation, 2-px buffer,", "   polygons < MMU (69 / 11 px), no data",
         "Fixed tile split; 64-px patches, stride 8,", "   ≥ 2 % new deforestation"]
for i, s in enumerate(lines):
    t(x + 3.0, ry - 6.0 - i * 3.5, ("• " if not s.startswith("   ") else "") + s, size=6.0)

# ---------------------------------------------------------------- 3 training
x, y, w, h = P3
panel(x, y, w, h, 3, "Training (source labels only)")
Xro = np.load(os.path.join(BASE, ".cache_cd", "RO", "X.npy"), mmap_mode="r")
Xst = np.load(os.path.join(BASE, ".cache_cd", "RO", "X_styled_PA_b0.01_0.npy"), mmap_mode="r")
Xpa = np.load(os.path.join(BASE, ".cache_cd", "PA", "X.npy"), mmap_mode="r")
sl = (slice(1024, 1536), slice(1536, 2048))     # one full 512 x 512 FDA tile
o_ro, lims = rgb_norm(Xro, sl)
o_st, _ = rgb_norm(Xst, sl, ref=lims)
o_pa, _ = rgb_norm(Xpa, (slice(512, 1024), slice(1024, 1536)))
s_ = 14.0; iy = y + 25.0
for k, (im_, lab) in enumerate([(o_ro, "RO tile"), (o_pa, "PA tile\n(target style)"), (o_st, "RO tile\nstyled to PA")]):
    ix = x + 3.0 + k * (s_ + 4.0)
    img(im_, ix, iy, s_, s_)
    t(ix + s_ / 2, iy - 1.2, lab, size=5.6, ha="center", va="top", color=MUTED)
t(x + 3.0 + s_ + 2.0, iy + s_ / 2, "+", size=9, ha="center", weight="bold", color=COL[3])
arr(x + 3.0 + 2 * s_ + 4.6, iy + s_ / 2, x + 3.0 + 2 * (s_ + 4.0) - 0.6, iy + s_ / 2, color=COL[3], lw=0.9)
t(x + 3.0, iy + s_ + 2.4, "Scene-level FDA (512-px tiles, β = 0.01):", size=6.0, weight="bold", va="bottom")
lines = ["FADE-Net, evidential loss, Adam 1e-4, batch 32", "Early stop on source validation loss",
         "Source-only, or FDA with 50 % (mix50) or", "   100 % (scale 1.0) styled patches"]
for i, s in enumerate(lines):
    t(x + 3.0, y + 13.5 - i * 3.5, ("• " if not s.startswith("   ") else "") + s, size=6.0)

# ---------------------------------------------------------------- 4 inference / adaptation
x, y, w, h = P4
panel(x, y, w, h, 4, "Inference on the target scene")
card(x + 3.0, y + 22.0, 38.0, 22.0, COL[4])
t(x + 5.0, y + 40.5, "Without adaptation", size=6.6, weight="bold")
for i, s in enumerate(["BN statistics from the", "source training data"]):
    t(x + 5.0, y + 35.8 - i * 3.4, s, size=6.0)
card(x + 45.0, y + 22.0, 38.0, 22.0, COL[4])
t(x + 47.0, y + 40.5, "BN-TTA", size=6.6, weight="bold")
for i, s in enumerate(["Reset BN statistics, re-estimate", "from 200 × 32 random 64-px", "target windows (no labels)"]):
    t(x + 47.0, y + 35.8 - i * 3.4, s, size=6.0)
t(x + 3.0, y + 16.5, "Stitched prediction: 64-px windows, stride 16 (75 % overlap),", size=6.0)
t(x + 3.0, y + 13.0, "keep the central 32 × 32 px of each window", size=6.0)
t(x + 3.0, y + 8.0, r"p(deforestation) = $\alpha_1 / S$;  label 1 if p ≥ 0.5", size=6.0)
# small stitching sketch
gx0, gy0 = x + 66.0, y + 3.0
for k in range(3):
    ax.add_patch(Rectangle((gx0 + k * 3.0, gy0 + k * 1.2), 12.0, 12.0, fc="none", ec=COL[4], lw=0.6, zorder=5))
ax.add_patch(Rectangle((gx0 + 6.0 + 3.0, gy0 + 2.4 + 3.0), 6.0, 6.0, fc=tint(COL[4], 0.5), ec=COL[4], lw=0.6, zorder=6))

# ---------------------------------------------------------------- 5 evaluation
x, y, w, h = P5
panel(x, y, w, h, 5, "Evaluation and analysis")
# 6-pair graph
cx, cy, R = x + 17.0, y + 27.0, 10.0
nodes = {"PA": (cx - R, cy + R * 0.55), "RO": (cx + R, cy + R * 0.55), "MA": (cx, cy - R * 0.85)}
for a in nodes:
    for b in nodes:
        if a != b:
            (x0, y0), (x1, y1) = nodes[a], nodes[b]
            d = np.array([x1 - x0, y1 - y0]); n = np.array([-d[1], d[0]]) / np.hypot(*d) * 0.9
            arr(x0 + n[0] + d[0] * 0.18, y0 + n[1] + d[1] * 0.18, x1 + n[0] - d[0] * 0.18, y1 + n[1] - d[1] * 0.18,
                color=COL[5], lw=0.6)
for k, (px, py) in nodes.items():
    ax.add_patch(Circle((px, py), 3.3, fc="white", ec=COL[5], lw=0.9, zorder=7))
    t(px, py, k, size=6.2, ha="center", weight="bold")
t(cx, y + 9.5, "6 pairs × 3 seeds ×\n6 configurations = 108 runs", size=6.0, ha="center", va="center")
lines = ["Configurations: source-only, FDA mix50, FDA", "   scale 1.0, each with and without BN-TTA",
         "Metrics on target test tiles: F1, precision,", "   recall at p ≥ 0.5, and AP",
         "Per-seed paired comparisons (wins / 3)", "Spectral distance between sites",
         "All reported numbers generated from logs"]
for i, s in enumerate(lines):
    t(x + 36.0, y + 42.0 - i * 4.6, ("• " if not s.startswith("   ") else "") + s, size=6.0)

fig.savefig(OUT + ".png", dpi=600, facecolor="white")
fig.savefig(OUT + ".pdf", facecolor="white")
print("saved", OUT)
