"""FADE-Net architecture diagram, drawn in code from models.py (as run: Mamba fallback block).

Every layer, channel count and tensor size below is the one in models.py with base_dim=128 and a
14 x 64 x 64 input; parameter counts come from paper/facts.json (measured by paper_facts.py).
Output: paper/figures/fig_architecture.{png,pdf}, 180 mm wide, 600 dpi. No generative AI.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "paper", "figures", "fig_architecture")
facts = json.load(open(os.path.join(BASE, "paper", "facts.json")))["model"]
MM = 1 / 25.4
W, H = 180.0, 97.0
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
INK = "#1a1a1a"
C_IN, C_ENC, C_BOT, C_DEC, C_HEAD = "#bfbfbf", "#2a78d6", "#6250d6", "#1baf7a", "#eb6834"


def shade(c, f):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f) if f > 0 else (r * (1 + f), g * (1 + f), b * (1 + f))


def cuboid(ax, x, y, w, h, d, color, z=3):
    """Front face (x, y, w, h) plus a depth offset d (drawn up-right)."""
    dx, dy = d * 0.7, d * 0.5
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(color, 0.45), ec=INK, lw=0.4, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(color, -0.25), ec=INK, lw=0.4, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=shade(color, 0.15), ec=INK, lw=0.4, zorder=z + 0.1))
    return x + w + dx, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.8, ls="-", rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=1.6,head_width=0.9",
                                 mutation_scale=1.6, color=color, lw=lw, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def t(ax, x, y, s, size=7, ha="center", va="center", weight="normal", color=INK, **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight=weight, color=color, zorder=10, **kw)


fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

# ---- main path: block heights encode spatial size (64 / 32 / 16), widths encode channels
side = {64: 30.0, 32: 20.0, 16: 12.0}
thick = {14: 1.2, 128: 2.4, 256: 3.6, 512: 5.2, 2: 0.9}
base_y = 51.0
blocks = [  # (key, label lines, channels, spatial, colour)
    ("in", ["Input", "date 1 ‖ date 2"], 14, 64, C_IN),
    ("s1", ["Conv 3×3", "BN, ReLU"], 128, 64, C_ENC),
    ("s2", ["Conv 3×3, stride 2", "BN, ReLU"], 256, 32, C_ENC),
    ("s3", ["Conv 3×3, stride 2", "BN, ReLU"], 512, 16, C_BOT),
    ("b1", ["Bottleneck", "block 1"], 512, 16, C_BOT),
    ("b2", ["Bottleneck", "block 2"], 512, 16, C_BOT),
    ("u1", ["TConv 4×4, stride 2"], 256, 32, C_DEC),
    ("cat", ["Concat"], 512, 32, C_DEC),
    ("f", ["Conv 3×3", "BN, ReLU"], 256, 32, C_DEC),
    ("u2", ["TConv 4×4, stride 2"], 128, 64, C_DEC),
    ("h", ["Conv 1×1", "Softplus"], 2, 64, C_HEAD),
]
x = 4.0
pos = {}
gap = 6.6
for key, lab, ch, sp, col in blocks:
    h = side[sp]; w = thick[ch]
    y = base_y - h / 2
    xr, ym = cuboid(ax, x, y, w, h, d=h * 0.28, color=col)
    pos[key] = dict(x0=x, x1=xr, y0=y, y1=y + h, ym=ym, top=y + h + h * 0.28 * 0.5)
    t(ax, x + w / 2 + h * 0.1, y - 1.6, f"{ch}×{sp}×{sp}", size=5.8, va="top", color="#4a4a4a")
    for i, s in enumerate(lab):
        t(ax, x + w / 2 + h * 0.1, pos[key]["top"] + 1.2 + (len(lab) - 1 - i) * 2.6, s, size=5.9, va="bottom")
    x = xr + gap
keys = [b[0] for b in blocks]
for a, b in zip(keys[:-1], keys[1:]):
    arrow(ax, pos[a]["x1"] + 0.4, base_y, pos[b]["x0"] - 0.4, base_y)

# skip connection: stem output (32x32x256) -> concat
xa = (pos["s2"]["x0"] + pos["s2"]["x1"]) / 2
xb = (pos["cat"]["x0"] + pos["cat"]["x1"]) / 2
ylow = pos["in"]["y0"] - 6.5
ax.add_line(Line2D([xa, xa, xb], [pos["s2"]["y0"] - 4.0, ylow, ylow], color=C_ENC, lw=0.9, zorder=2))
arrow(ax, xb, ylow, xb, pos["cat"]["y0"] - 4.0, color=C_ENC, lw=0.9)
t(ax, (xa + xb) / 2, ylow - 1.6, "skip connection: 256 × 32 × 32 encoder features", size=6, color=C_ENC, va="top")

# group labels
def bracket(x0, x1, y, s, col):
    ax.add_line(Line2D([x0, x0, x1, x1], [y - 1.2, y, y, y - 1.2], color=col, lw=0.8, zorder=2))
    t(ax, (x0 + x1) / 2, y + 1.0, s, size=6.6, weight="bold", color=col, va="bottom")
ytop = 85.5
bracket(pos["s1"]["x0"], pos["s3"]["x1"], ytop, "Encoder (stem + stage 2)", C_ENC)
bracket(pos["b1"]["x0"], pos["b2"]["x1"], ytop, f"Bottleneck ({facts['bottleneck_params'] / 1e6:.2f} M parameters)", C_BOT)
bracket(pos["u1"]["x0"], pos["u2"]["x1"], ytop, f"Decoder ({facts['decoder_params'] / 1e6:.2f} M)", C_DEC)
bracket(pos["h"]["x0"] - 1.0, pos["h"]["x1"] + 1.0, ytop, "Head", C_HEAD)

# ---- lower row: bottleneck block detail and evidential output
def box(x, y, w, h, ec, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=ec, lw=0.8,
                                zorder=1))
    t(ax, x + 2.0, y + h - 2.2, title, size=6.8, weight="bold", ha="left", color=ec)

bx, by, bw, bh = 4.0, 2.0, 104.0, 21.0
box(bx, by, bw, bh, C_BOT, "Bottleneck block (as run; the planned Mamba block was replaced by this fallback)")
steps = ["DW Conv 7×7", "BN", "Conv 1×1\n512→2048", "GELU", "Conv 1×1\n2048→512"]
sx = bx + 14.0
cy = by + 8.0
prev = None
ws = [14.5, 7.0, 15.0, 8.5, 15.0]
t(ax, bx + 6.0, cy, "x", size=7.5, style="italic")
arrow(ax, bx + 7.6, cy, sx - 0.4, cy)
for s, w_ in zip(steps, ws):
    ax.add_patch(FancyBboxPatch((sx, cy - 3.6), w_, 7.2, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=shade(C_BOT, 0.82), ec=C_BOT, lw=0.6, zorder=3))
    t(ax, sx + w_ / 2, cy, s, size=5.8)
    if prev is not None:
        arrow(ax, prev, cy, sx - 0.3, cy)
    prev = sx + w_ + 0.3
    sx += w_ + 3.2
sum_x = sx + 1.5
ax.add_patch(plt.Circle((sum_x, cy), 1.8, fc="white", ec=INK, lw=0.6, zorder=4))
t(ax, sum_x, cy + 0.1, "+", size=8)
arrow(ax, prev, cy, sum_x - 1.9, cy)
ax.add_line(Line2D([bx + 9.5, bx + 9.5, sum_x], [cy, cy + 5.6, cy + 5.6], color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=2))
arrow(ax, sum_x, cy + 5.6, sum_x, cy + 1.9, lw=0.6)
t(ax, sum_x + 3.0, cy, "2x + f(x)", size=6.2, ha="left")
t(ax, bx + 2.0, by + 2.0, "Two residual paths (inner block and wrapper), so each block outputs 2x + f(x).",
  size=5.8, ha="left", color="#4a4a4a")

ex, ey, ew, eh = 112.0, 2.0, 64.0, 21.0
box(ex, ey, ew, eh, C_HEAD, "Evidential output (per pixel)")
lines = [r"evidence $e_k = \mathrm{softplus}(z_k) \geq 0$,  $k \in \{0, 1\}$",
         r"$\alpha_k = e_k + 1$,   $S = \alpha_0 + \alpha_1$",
         r"$p(\mathrm{deforestation}) = \alpha_1 / S$;  label 1 if $\geq 0.5$",
         r"uncertainty $u = 2 / S$"]
for i, s in enumerate(lines):
    t(ax, ex + 3.0, ey + eh - 6.6 - i * 3.6, s, size=6.2, ha="left")
arrow(ax, (pos["h"]["x0"] + pos["h"]["x1"]) / 2, pos["h"]["y0"] - 4.0, ex + ew * 0.78, ey + eh + 0.3, color=C_HEAD,
      lw=0.8)

t(ax, W - 2.0, H - 2.0, f"FADE-Net, {facts['params'] / 1e6:.2f} M parameters", size=7, ha="right", va="top",
  weight="bold")
t(ax, 2.0, H - 2.0, "Block height: spatial size (64, 32, 16 px); block width: channels", size=6, ha="left", va="top",
  color="#4a4a4a")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")
fig.savefig(OUT + ".pdf", facecolor="white")
print("saved", OUT)
