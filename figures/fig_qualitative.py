"""Qualitative results on PA: where adaptation changes the predictions.

Row 1, MA->PA: source-only vs source-only + BN-TTA (the operating-point shift).
Row 2, RO->PA: source-only vs FDA scale 1.0.
For each row the seed is chosen by rule, not by eye: the seed whose F1 change (adapted minus source-only)
is the median of the three seeds. Predictions are recomputed from the saved checkpoints with the same code
as the experiments (train_cd.stitched_scene_probs / bn_adapt) and checked against the logged test F1.
The window is the 9 km x 9 km area inside PA test tiles with the most new deforestation.
Output: paper/figures/fig_qualitative.{png,pdf}, 180 mm wide, 600 dpi.
"""
import os
import sys
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
sys.path.insert(0, BASE)
os.chdir(BASE)
from dataset_cd import DOMAIN_DATA, _load_band_stack, build_domain_arrays, tile_mask
from models import FADENet
from train_cd import stitched_scene_probs, masked_metrics, bn_adapt, device

OUT = os.path.join(BASE, "paper", "figures", "fig_qualitative")
CACHE = os.path.join(BASE, "results", "qualitative")
SEEDS = [42, 123, 456]
WIN = 300
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
TP, FP, FN = "#1baf7a", "#eb6834", "#6250d6"           # validated palette (all pairs pass CVD checks)
TN, VOID = "#eeeeea", "#3a3a3a"
NEW_DEF = (0.847, 0.106, 0.376)

ROWS = [  # (source, method key in summary, checkpoint tag pattern of adapted model, uses BN-TTA, label)
    ("MA", "bntta", "cd_source_only_{s}_to_PA_seed{sd}", True, "+ BN-TTA"),
    ("RO", "fda_scale1", "cd_fda_{s}_to_PA_seed{sd}", False, "FDA scale 1.0"),
]


def summary():
    rows = list(csv.DictReader(open(os.path.join(BASE, "results", "paperA_summary.csv"))))
    return {(r["method"], r["source"], r["target"], int(r["seed"])): r for r in rows}


def median_seed(S, src, method):
    d = {sd: float(S[(method, src, "PA", sd)]["tgt_F1"]) - float(S[("source_only", src, "PA", sd)]["tgt_F1"])
         for sd in SEEDS}
    order = sorted(SEEDS, key=lambda k: d[k])
    return order[1], d


@torch.no_grad()
def probs_for(tag, src, bntta):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{tag}{'_bntta' if bntta else ''}_PA.npy")
    if os.path.exists(f):
        return np.load(f).astype(np.float32)
    model = FADENet(in_channels=14, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", tag, "best_model.pt"), map_location=device))
    if bntta:
        bn_adapt(model, build_domain_arrays("PA")[0])
    p = stitched_scene_probs(model, build_domain_arrays("PA")[0])
    np.save(f, p.astype(np.float16))
    return p


def error_map(p, y, v):
    pred = p >= 0.5
    out = np.empty(y.shape + (3,), np.float32)
    out[:] = matplotlib.colors.to_rgb(VOID)
    out[v & ~y & ~pred] = matplotlib.colors.to_rgb(TN)
    out[v & y & pred] = matplotlib.colors.to_rgb(TP)
    out[v & ~y & pred] = matplotlib.colors.to_rgb(FP)
    out[v & y & ~pred] = matplotlib.colors.to_rgb(FN)
    return out


def main():
    S = summary()
    xp, yp, vp = build_domain_arrays("PA")
    y = np.load(yp).astype(bool); v = np.load(vp).astype(bool)
    H, W = y.shape
    test = tile_mask("PA", H, W, "test")
    # window: most new deforestation, fully inside test tiles
    Sy = np.pad(np.cumsum(np.cumsum((v & y).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    St = np.pad(np.cumsum(np.cumsum((~test).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    box = lambda A, i, j: A[i + WIN, j + WIN] - A[i, j + WIN] - A[i + WIN, j] + A[i, j]
    cand = [(box(Sy, i, j), i, j) for i in range(0, H - WIN, 25) for j in range(0, W - WIN, 25) if box(St, i, j) == 0]
    _, wi, wj = max(cand)
    t2, nd = _load_band_stack(os.path.join(BASE, DOMAIN_DATA["PA"]["img_t2"]), return_nodata=True)
    rgb = np.zeros((H, W, 3), np.float32)
    ok = ~nd[:H, :W]
    for k, b in enumerate((3, 2, 1)):                  # OLI B4, B3, B2
        lo, hi = np.percentile(t2[b][:H, :W][ok], [2, 98])
        rgb[..., k] = np.clip((t2[b][:H, :W] - lo) / (hi - lo), 0, 1) ** 0.85
    ref = np.empty((H, W, 3), np.float32); ref[:] = matplotlib.colors.to_rgb(VOID)
    ref[v & ~y] = matplotlib.colors.to_rgb(TN); ref[v & y] = NEW_DEF

    panels = []
    for src, method, adapted_tpl, bntta, lab in ROWS:
        sd, deltas = median_seed(S, src, method)
        so_tag = f"cd_source_only_{src}_to_PA_seed{sd}"
        p_so = probs_for(so_tag, src, False)
        p_ad = probs_for(adapted_tpl.format(s=src, sd=sd), src, bntta)
        f_so = masked_metrics(p_so, yp, vp, "PA", "test")["F1"]
        f_ad = masked_metrics(p_ad, yp, vp, "PA", "test")["F1"]
        log_so = float(S[("source_only", src, "PA", sd)]["tgt_F1"])
        log_ad = float(S[(method, src, "PA", sd)]["tgt_F1"])
        assert abs(f_so - log_so) < 1e-3 and abs(f_ad - log_ad) < 1e-3, (f_so, log_so, f_ad, log_ad)
        win = lambda m: m[wi:wi + WIN, wj:wj + WIN]
        cnt = lambda p: dict(zip(["TP", "FP", "FN"], [int((win(v & y & (p >= .5))).sum()),
                                                       int((win(v & ~y & (p >= .5))).sum()),
                                                       int((win(v & y & (p < .5))).sum())]))
        panels.append(dict(src=src, sd=sd, lab=lab, deltas=deltas, so=error_map(p_so, y, v), ad=error_map(p_ad, y, v),
                           f_so=log_so, f_ad=log_ad, c_so=cnt(p_so), c_ad=cnt(p_ad)))
        print(f"{src}->PA seed {sd} (deltas {deltas}): F1 source-only {log_so:.3f} -> {lab} {log_ad:.3f} "
              f"(recomputed {f_so:.3f} / {f_ad:.3f}); window counts {cnt(p_so)} -> {cnt(p_ad)}")

    # ---- draw: 2 rows x 4 columns, square tiles
    Wmm = 180.0
    s = 40.0; gap = 2.5; left = 10.0
    Hmm = 6 + 2 * (s + 9.5) + 3
    fig = plt.figure(figsize=(Wmm * MM, Hmm * MM))
    def axmm(x, yy, w, h):
        return fig.add_axes([x / Wmm, 1 - (yy + h) / Hmm, w / Wmm, h / Hmm])
    for r, P in enumerate(panels):
        y0 = 6 + r * (s + 9.5)
        cols = [(win_rgb := rgb[wi:wi + WIN, wj:wj + WIN], "Date 2 image", "lanczos"),
                (ref[wi:wi + WIN, wj:wj + WIN], "Reference", "nearest"),
                (P["so"][wi:wi + WIN, wj:wj + WIN], f"Source-only (scene F$_1$ {P['f_so']:.3f})", "nearest"),
                (P["ad"][wi:wi + WIN, wj:wj + WIN], f"{P['lab']} (scene F$_1$ {P['f_ad']:.3f})", "nearest")]
        for c, (arr, title, interp) in enumerate(cols):
            ax = axmm(left + c * (s + gap), y0, s, s)
            ax.imshow(arr, interpolation=interp)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            fig.text((left + c * (s + gap) + s / 2) / Wmm, 1 - (y0 - 0.8) / Hmm, title, ha="center", va="bottom",
                     fontsize=7)
            if c >= 2:
                cc = P["c_so"] if c == 2 else P["c_ad"]
                ax.text(0.02, 0.02, f"TP {cc['TP']:,}  FP {cc['FP']:,}  FN {cc['FN']:,}", transform=ax.transAxes,
                        fontsize=5.8, va="bottom", ha="left",
                        bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        fig.text(4.0 / Wmm, 1 - (y0 + s / 2) / Hmm, f"({'ab'[r]}) {P['src']}$\\rightarrow$PA, seed {P['sd']}",
                 rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
    handles = [Patch(fc=NEW_DEF, label="New deforestation (reference)"), Patch(fc=TP, label="Correct detection (TP)"),
               Patch(fc=FP, label="False alarm (FP)"), Patch(fc=FN, label="Missed (FN)"),
               Patch(fc=TN, ec="#9a9a9a", lw=0.4, label="No deforestation, correct"),
               Patch(fc=VOID, label="Excluded")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.3, frameon=False, handlelength=1.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(OUT + ".png", dpi=600, facecolor="white")
    fig.savefig(OUT + ".pdf", facecolor="white")
    print("window (row, col):", (wi, wj), "| saved", OUT)


if __name__ == "__main__":
    main()
