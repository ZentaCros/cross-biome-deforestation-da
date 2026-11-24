"""
Spectral domain-gap analysis and FDA swap-size figure for Paper A.

Measured on BOTH dates, on two versions of each scene:
  raw        - Landsat-8 DNs as stored (NaN no-data median-filled, as the loader does)
  normalized - per-scene z-scored bands, i.e. exactly what the network receives
Spectra: radially averaged log(1+|FFT|) of 20 random 512x512 tiles per scene (seed 42).
Distance between two domains: mean |difference| of their mean profiles, averaged over
7 bands and 2 dates. All numbers are written to figures/spectral_distances.json.

Outputs: fig_spectral_profiles.{pdf,png}, fig_spectral_distance.{pdf,png},
         fig_fda_granularity.{pdf,png}, spectral_distances.json
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(OUT, '..')
sys.path.insert(0, BASE)
from dataset_cd import DOMAIN_DATA, _load_band_stack, _normalize_scene

DPI = 300
DOMAINS = ["PA", "RO", "MA"]
LABEL = {"PA": "PA (Amazon)", "RO": "RO (Amazon)", "MA": "MA (Cerrado)"}
COLOR = {"PA": "#2a78d6", "RO": "#eb6834", "MA": "#1baf7a"}   # validated categorical slots 1-3
STYLE = {"PA": "-", "RO": "--", "MA": ":"}                    # secondary encoding (not color alone)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# Landsat-8 OLI B1-B7, as stored (the base code computes NDVI with index 4 = NIR, 3 = red).
BANDS = ["Coastal (B1)", "Blue (B2)", "Green (B3)", "Red (B4)", "NIR (B5)", "SWIR1 (B6)", "SWIR2 (B7)"]
TILE, N_TILES = 512, 20

plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "font.family": "DejaVu Sans"})


def radial_profile(a):
    H, W = a.shape
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((Y - H // 2) ** 2 + (X - W // 2) ** 2).astype(int)
    n = min(H, W) // 2
    sums = np.bincount(r.ravel(), a.ravel(), minlength=n)[:n]
    cnt = np.bincount(r.ravel(), minlength=n)[:n]
    return sums / np.maximum(cnt, 1)


def scene_profiles(img):
    """[7, n_tiles, 256] log-amplitude radial profiles."""
    C, H, W = img.shape
    rng = np.random.RandomState(42)
    out = np.zeros((C, N_TILES, TILE // 2))
    for t in range(N_TILES):
        i = rng.randint(0, H - TILE + 1)
        j = rng.randint(0, W - TILE + 1)
        tile = img[:, i:i + TILE, j:j + TILE]
        for b in range(C):
            amp = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(tile[b]))))
            out[b, t] = radial_profile(amp)
    return out


def compute():
    prof = {"raw": {}, "normalized": {}}
    for dk in DOMAINS:
        for date in ["img_t1", "img_t2"]:
            print(f"  {dk} {date}")
            img = _load_band_stack(os.path.join(BASE, DOMAIN_DATA[dk][date]))
            prof["raw"][(dk, date)] = scene_profiles(img)
            prof["normalized"][(dk, date)] = scene_profiles(_normalize_scene(img))
            del img
    dist = {}
    for kind in prof:
        D = np.zeros((3, 3))
        for a, da in enumerate(DOMAINS):
            for b, db in enumerate(DOMAINS):
                vals = [np.abs(prof[kind][(da, d)].mean(1)[bd] - prof[kind][(db, d)].mean(1)[bd]).mean()
                        for d in ["img_t1", "img_t2"] for bd in range(7)]
                D[a, b] = float(np.mean(vals))
        dist[kind] = D
    return prof, dist


def fig_profiles(prof):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), dpi=DPI)
    f = np.arange(TILE // 2)
    for b in range(7):
        ax = axes[b // 4, b % 4]
        for dk in DOMAINS:
            p = np.concatenate([prof["normalized"][(dk, d)][b] for d in ["img_t1", "img_t2"]])
            m, s = p.mean(0), p.std(0)
            ax.fill_between(f[1:], (m - s)[1:], (m + s)[1:], color=COLOR[dk], alpha=0.12, lw=0)
            ax.plot(f[1:], m[1:], STYLE[dk], color=COLOR[dk], lw=1.2, label=LABEL[dk])
        ax.axvline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_title(BANDS[b], fontsize=7.5, color=INK, pad=3)
        ax.grid(True, color=GRID, lw=0.4)
        ax.tick_params(labelsize=6, length=2)
        if b % 4 == 0:
            ax.set_ylabel("log(1+|F|)", fontsize=6.5)
        if b >= 3:
            ax.set_xlabel("Radial frequency (bins, 512 px tile)", fontsize=6)
    leg = axes[1, 3]
    leg.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    leg.legend(h, l, loc="center", frameon=False, fontsize=7,
               title="Per-scene z-scored input\n(both dates, 40 tiles)", title_fontsize=6.5)
    leg.text(0.5, 0.12, "grey vertical line: FDA swap\nhalf-width at 512 px, β = 0.01",
             ha="center", fontsize=5.8, color=MUTED, transform=leg.transAxes)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    save(fig, "fig_spectral_profiles")


def fig_distance(dist):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), dpi=DPI)
    titles = {"raw": "(a) Raw DNs", "normalized": "(b) z-scored input (model view)"}
    for ax, kind in zip(axes, ["raw", "normalized"]):
        D = dist[kind]
        ax.imshow(D, cmap="Blues", vmin=0, vmax=D.max() * 1.15)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{D[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if D[i, j] > D.max() * 0.65 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(DOMAINS); ax.set_yticklabels(DOMAINS)
        ax.set_title(titles[kind], fontsize=7.5, color=INK)
        ax.tick_params(length=0, labelsize=7, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_spectral_distance")


def fig_fda():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), dpi=DPI,
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    betas = np.linspace(0.001, 0.05, 500)
    for N, c, ls in zip([64, 128, 256, 512], ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"],
                        ["-", "--", "-.", ":"]):
        ax.step(betas, [int(N * b) for b in betas], where="post", color=c, ls=ls, lw=1.3, label=f"N = {N}")
    ax.axvline(0.01, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(0.0105, 22, "β = 0.01", fontsize=6, color=MUTED)
    ax.set_xlabel("β"); ax.set_ylabel("Swap half-width ⌊βN⌋ (bins)")
    ax.set_title("(a) Swap size vs. β", fontsize=7.5, color=INK)
    ax.grid(True, color=GRID, lw=0.4); ax.legend(fontsize=6, frameon=False)
    for ax, N, ttl in [(axes[1], 64, "(b) N = 64, β = 0.01"), (axes[2], 512, "(c) N = 512, β = 0.01")]:
        b = int(N * 0.01)
        m = np.zeros((N, N))
        c = N // 2
        if b > 0:
            m[c - b:c + b, c - b:c + b] = 1          # same slice as fda_source_to_target
        z = 16                                       # show the central 32 x 32 frequency bins
        ax.imshow(m[c - z:c + z, c - z:c + z], cmap="Blues", vmin=0, vmax=1.3,
                  extent=[-z - 0.5, z - 0.5, z - 0.5, -z - 0.5], interpolation="nearest")
        ax.set_title(ttl, fontsize=7.5, color=INK)
        ax.set_xlabel("Frequency bin (from DC)")
        ax.text(0, z - 3, f"{(2 * b) ** 2} of {N * N:,} bins swapped", ha="center", fontsize=6, color=INK)
        ax.tick_params(labelsize=6, length=2)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_fda_granularity")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


if __name__ == "__main__":
    prof, dist = compute()
    with open(os.path.join(OUT, "spectral_distances.json"), "w") as f:
        json.dump({"domains": DOMAINS, "tile": TILE, "n_tiles_per_scene": N_TILES, "seed": 42,
                   "dates": ["img_t1", "img_t2"],
                   **{k: v.round(6).tolist() for k, v in dist.items()}}, f, indent=1)
    for k, D in dist.items():
        print(k, "PA-RO %.4f PA-MA %.4f RO-MA %.4f" % (D[0, 1], D[0, 2], D[1, 2]))
    fig_profiles(prof)
    fig_distance(dist)
    fig_fda()
