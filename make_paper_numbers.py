"""Turn measured results into the numbers, tables and result figures of Paper A.

Inputs : results/paperA_summary.csv (aggregate_results.py), paper/facts.json (paper_facts.py),
         figures/spectral_distances.json (figures/spectral_analysis.py)
Outputs: paper/numbers.tex      - one LaTeX macro per number quoted in the text
         paper/tab_results.tex  - main results table (mean +- std over seeds, wins/n)
         paper/tab_data.tex     - dataset table
         paper/figures/fig_transfer_matrix.{pdf,png}, fig_method_seeds.{pdf,png}
Safe to re-run at any time; methods with no finished runs print as '--'.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(BASE, "paper")
FIG = os.path.join(PAPER, "figures")
PAIRS = [("PA", "RO"), ("RO", "PA"), ("MA", "RO"), ("RO", "MA"), ("PA", "MA"), ("MA", "PA")]
SEEDS = [42, 123, 456]
METHODS = [("source_only", "SO", "Source-only"), ("bntta", "TTA", "+ BN-TTA"),
           ("fda_mix50", "MIX", "FDA mix50"), ("fda_mix50_bntta", "MIXTTA", "FDA mix50 + BN-TTA"),
           ("fda_scale1", "ONE", "FDA scale 1.0"), ("fda_scale1_bntta", "ONETTA", "FDA scale 1.0 + BN-TTA")]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
MCOLOR = {"source_only": "#2a78d6", "bntta": "#eb6834", "fda_mix50": "#1baf7a",
          "fda_mix50_bntta": "#eda100", "fda_scale1": "#6250d6", "fda_scale1_bntta": "#e87ba4"}
MMARK = {"source_only": "o", "bntta": "^", "fda_mix50": "s", "fda_mix50_bntta": "D",
         "fda_scale1": "v", "fda_scale1_bntta": "P"}

rows = list(csv.DictReader(open(os.path.join(BASE, "results", "paperA_summary.csv"))))
for r in rows:
    r["seed"] = int(r["seed"])
    for k in ("tgt_F1", "tgt_P", "tgt_R", "tgt_IoU", "tgt_AP", "src_F1"):
        r[k] = float(r[k])
facts = json.load(open(os.path.join(PAPER, "facts.json")))
spec = json.load(open(os.path.join(BASE, "figures", "spectral_distances.json")))


def get(m, s, t, seed, key="tgt_F1"):
    v = [r[key] for r in rows if r["method"] == m and r["source"] == s and r["target"] == t and r["seed"] == seed]
    return v[0] if v else None


def vals(m, s, t, key="tgt_F1"):
    return [x for x in (get(m, s, t, sd, key) for sd in SEEDS) if x is not None]


# ---- determinism check: one source model per (source, seed), shared by both targets ----
for src in ["PA", "RO", "MA"]:
    tg = [t for s, t in PAIRS if s == src]
    for sd in SEEDS:
        a, b = get("source_only", src, tg[0], sd, "src_F1"), get("source_only", src, tg[1], sd, "src_F1")
        assert a is not None and abs(a - b) < 1e-12, f"source model {src} seed {sd} differs across targets"

macros = {}


def mac(name, value):
    assert name.isalpha(), name
    macros[name] = value


def sgn(x):
    """Signed 3-decimal number with a typeset minus sign."""
    s = f"{x:+.3f}"
    return s.replace("-", "$-$")


def f3(x):
    return f"{x:.3f}"


pairname = {p: p[0] + p[1] for p in PAIRS}
# F1 macros: \SOPAROM ...; AP macros carry an "AP" infix: \SOAPPAROM ...; P/R: \SOPPAROM, \SORPAROM
for key, infix in [("tgt_F1", ""), ("tgt_AP", "AP"), ("tgt_P", "P"), ("tgt_R", "R")]:
    for m, short, _ in METHODS:
        for p in PAIRS:
            v = vals(m, *p, key)
            if not v:
                continue
            mac(f"{short}{infix}{pairname[p]}M", f3(np.mean(v)))
            mac(f"{short}{infix}{pairname[p]}S", f3(np.std(v, ddof=1)) if len(v) > 1 else "--")
            mac(f"{short}{infix}{pairname[p]}N", str(len(v)))
            if m != "source_only":
                d = [get(m, *p, sd, key) - get("source_only", *p, sd, key) for sd in SEEDS
                     if get(m, *p, sd, key) is not None]
                mac(f"{short}{infix}{pairname[p]}W", str(sum(x > 0 for x in d)))
                mac(f"{short}{infix}{pairname[p]}D", sgn(np.mean(d)))
        if m != "source_only":
            d = [get(m, *p, sd, key) - get("source_only", *p, sd, key) for p in PAIRS for sd in SEEDS
                 if get(m, *p, sd, key) is not None]
            if d:
                mac(f"{short}{infix}AllW", str(sum(x > 0 for x in d)))
                mac(f"{short}{infix}AllN", str(len(d)))
                mac(f"{short}{infix}AllD", sgn(np.mean(d)))

# Head-to-head comparisons (same pair, same seed) and wins by source domain.
SHORTOF = {m: s for m, s, _ in METHODS}
def compare(a, b, key, pairs):
    d = [get(a, *p, sd, key) - get(b, *p, sd, key) for p in pairs for sd in SEEDS
         if get(a, *p, sd, key) is not None and get(b, *p, sd, key) is not None]
    assert all(x != 0 for x in d), "exact tie: wins + losses would not add up"
    return sum(x > 0 for x in d), len(d), (np.mean(d) if d else 0.0)

for key, infix in [("tgt_F1", ""), ("tgt_AP", "AP")]:
    for a, b in [("fda_mix50_bntta", "fda_mix50"), ("fda_scale1_bntta", "fda_scale1"),
                 ("fda_scale1", "fda_mix50")]:
        w, n, dm = compare(a, b, key, PAIRS)
        if n:
            tag = f"{SHORTOF[a]}vs{SHORTOF[b]}{infix}"
            mac(f"{tag}W", str(w)); mac(f"{tag}N", str(n)); mac(f"{tag}D", sgn(dm))
    for m in ["bntta", "fda_mix50", "fda_scale1"]:
        for src in ["PA", "RO", "MA"]:
            w, n, dm = compare(m, "source_only", key, [p for p in PAIRS if p[0] == src])
            if n:
                mac(f"{SHORTOF[m]}{infix}Src{src}W", str(w)); mac(f"{SHORTOF[m]}{infix}Src{src}N", str(n))
                mac(f"{SHORTOF[m]}{infix}Src{src}L", str(n - w))
                mac(f"{SHORTOF[m]}{infix}Src{src}D", sgn(dm))
# Best mean per pair over all six configurations
for p in PAIRS:
    best = max(METHODS, key=lambda m: np.mean(vals(m[0], *p)) if vals(m[0], *p) else -1)
    mac(f"Best{pairname[p]}", best[2].replace("+", "+\\,"))
# PA->MA FDA scale 1.0: near-identical F1 over seeds although the models differ (AP, P, R differ)
v = vals("fda_scale1", "PA", "MA")
if len(v) == 3:
    mac("ONEPAMAFRange", f"{max(v) - min(v):.4f}")
    a_ = vals("fda_scale1", "PA", "MA", "tgt_AP")
    mac("ONEPAMAAPLo", f3(min(a_))); mac("ONEPAMAAPHi", f3(max(a_)))

# in-domain (source-test) F1 of the source-only models: one model per (source, seed)
num = {1: "One", 2: "Two", 3: "Three"}
for src in ["PA", "RO", "MA"]:
    t0 = [t for s, t in PAIRS if s == src][0]
    v = vals("source_only", src, t0, "src_F1")
    mac(f"InDom{src}M", f3(np.mean(v)))
    mac(f"InDom{src}S", f3(np.std(v, ddof=1)))
so = [np.mean(vals("source_only", *p)) for p in PAIRS]
mac("SOMin", f3(min(so)))
mac("SOMax", f3(max(so)))
# seed-to-seed spread of source-only target F1 (std over 3 seeds), min / max over pairs
sds = [np.std(vals("source_only", *p), ddof=1) for p in PAIRS]
mac("SOStdMin", f3(min(sds)))
mac("SOStdMax", f3(max(sds)))
# range of one pair's three seeds (largest)
rng_ = {pairname[p]: max(vals("source_only", *p)) - min(vals("source_only", *p)) for p in PAIRS}
k = max(rng_, key=rng_.get)
mac("SOWidestPair", f"{k[:2]}$\\to${k[2:]}")
mac("SOWidestLo", f3(min(vals("source_only", k[:2], k[2:]))))
mac("SOWidestHi", f3(max(vals("source_only", k[:2], k[2:]))))
# BN-TTA effect on the SOURCE test set (stats re-estimated on target)
d = [get("bntta", *p, sd, "src_F1") - get("source_only", *p, sd, "src_F1") for p in PAIRS for sd in SEEDS]
assert np.mean(d) < 0
mac("TTASrcDropMean", f3(-np.mean(d)))   # positive size of the drop; the text says "lowers by"

# ---- dataset / model / protocol facts ----
for dk, d in facts["domains"].items():
    mac(f"H{dk}", str(d["H"]))
    mac(f"W{dk}", str(d["W"]))
    mac(f"Valid{dk}", f"{100 * d['valid_frac']:.1f}")
    mac(f"Def{dk}", f"{100 * d['def_frac_of_valid']:.2f}")
    mac(f"TrainPatches{dk}", f"{d['train_patches']:,}".replace(",", "{,}"))
    mac(f"ValPatches{dk}", f"{d['val_patches']:,}".replace(",", "{,}"))
    mac(f"NoData{dk}", f"{100 * max(d['img_t1_nodata_frac'], d['img_t2_nodata_frac']):.3f}")
    mac(f"TestPx{dk}", f"{d['test_valid_px']:,}".replace(",", "{,}"))
    mac(f"TestDefPx{dk}", f"{d['test_def_px']:,}".replace(",", "{,}"))
md = facts["model"]
mac("ParamsM", f"{md['params'] / 1e6:.2f}")
mac("ParamsEncM", f"{md['encoder_params'] / 1e6:.2f}")
mac("ParamsDecM", f"{md['decoder_params'] / 1e6:.2f}")
mac("ParamsBotM", f"{md['bottleneck_params'] / 1e6:.2f}")
for dk in ["PA", "RO", "MA"]:
    t = facts["fda"][f"{dk}_tiles"]
    mac(f"FDATiles{dk}", str(t["n_tiles"]))
    mac(f"FDAUnstyled{dk}", str(t["n_unstyled"]))
    mac(f"FDAUnstyledPct{dk}", f"{100 * t['unstyled_px_frac']:.1f}")
mac("FDABinsFiveTwelve", f"{facts['fda']['N512']['swapped_bins']}")
mac("FDATotalFiveTwelve", f"{facts['fda']['N512']['total_bins']:,}".replace(",", "{,}"))
mac("FDASwapPct", f"{100 * facts['fda']['N512']['swapped_bins'] / facts['fda']['N512']['total_bins']:.3f}")


def epoch_stats(prefix, exclude=None):
    v = [x for k, x in facts["runs"].items() if k.startswith(prefix) and not (exclude and k.startswith(exclude))]
    b = [x["best_epoch"] for x in v]
    s = [x["stop_epoch"] for x in v]
    return b, s


# One distinct source-only model per (source, seed): take the first target of each source.
uniq = [facts["runs"][f"cd_source_only_{s}_to_{[t for ss, t in PAIRS if ss == s][0]}_seed{sd}"]
        for s in ["PA", "RO", "MA"] for sd in SEEDS]
b = [x["best_epoch"] for x in uniq]; s = [x["stop_epoch"] for x in uniq]
mac("BestEpMin", str(min(b))); mac("BestEpMax", str(max(b))); mac("BestEpMed", str(int(np.median(b))))
mac("StopEpMin", str(min(s))); mac("StopEpMax", str(max(s)))
mac("BestEpOne", str(sum(x == 1 for x in b))); mac("BestEpN", str(len(b)))
# How far the selected (min val loss) checkpoint is from the best source-val F1 of the run
gaps = []
for s_ in ["PA", "RO", "MA"]:
    t0 = [t for ss, t in PAIRS if ss == s_][0]
    for sd in SEEDS:
        R = list(csv.DictReader(open(os.path.join(BASE, "results", f"cd_source_only_{s_}_to_{t0}_seed{sd}", "metrics.csv"))))
        L = [float(r["SrcVal_Loss"]) for r in R]; F = [float(r["SrcVal_F1"]) for r in R]
        best = None; bl = float("inf")
        for i, v in enumerate(L):          # same rule as train_cd.py
            if v < bl - 1e-5:
                bl, best = v, i
        gaps.append(max(F) - F[best])
mac("ValFGapMin", f3(min(gaps))); mac("ValFGapMax", f3(max(gaps)))
# BN-TTA raises F1 exactly on the pairs with the lowest source-only F1 (claim in the text)
order = sorted(PAIRS, key=lambda p: np.mean(vals("source_only", *p)))
signs = [np.mean(vals("bntta", *p)) > np.mean(vals("source_only", *p)) for p in order]
assert signs == [True] * 4 + [False] * 2, signs
assert {order[4][0], order[5][0]} == {"MA"}
# duplicate runs (same source + seed, other target) must have identical epoch logs
for s_ in ["PA", "RO", "MA"]:
    tg = [t for ss, t in PAIRS if ss == s_]
    for sd in SEEDS:
        a_, b_ = (facts["runs"][f"cd_source_only_{s_}_to_{t}_seed{sd}"] for t in tg)
        assert a_["best_epoch"] == b_["best_epoch"] and a_["stop_epoch"] == b_["stop_epoch"]

# spectral distances
names = spec["domains"]
for kind, tag in [("raw", "Raw"), ("normalized", "Norm")]:
    D = np.array(spec[kind])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mac(f"Spec{tag}{names[i]}{names[j]}", f3(D[i, j]))

with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
    f.write("% Generated by make_paper_numbers.py from measured results. Do not edit by hand.\n")
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")

# ---- main results table ----
def cell(m, p, key):
    v = vals(m, *p, key)
    if not v:
        return "--"
    best = max(np.mean(vals(mm, *p, key)) for mm, _, _ in METHODS if vals(mm, *p, key))
    txt = f"{np.mean(v):.3f}" + (f"$\\pm${np.std(v, ddof=1):.3f}" if len(v) > 1 else "")
    if abs(np.mean(v) - best) < 1e-12:
        txt = f"\\textbf{{{txt}}}"
    if m != "source_only":
        w = sum(get(m, *p, sd, key) > get("source_only", *p, sd, key) for sd in SEEDS
                if get(m, *p, sd, key) is not None)
        txt += f" ({w}/{len(v)})"
    return txt


for key, fname in [("tgt_F1", "tab_results.tex"), ("tgt_AP", "tab_results_ap.tex")]:
    lines = ["% Generated by make_paper_numbers.py. Do not edit by hand.",
             "\\begin{tabular}{@{}l" + "c" * len(PAIRS) + "@{}}", "\\toprule",
             "Method & " + " & ".join(f"{s}$\\to${t}" for s, t in PAIRS) + " \\\\", "\\midrule"]
    for m, _, label in METHODS:
        lines.append(label + " & " + " & ".join(cell(m, p, key) for p in PAIRS) + " \\\\")
        if m == "bntta":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, fname), "w").write("\n".join(lines) + "\n")

# ---- data table ----
dl = ["% Generated by make_paper_numbers.py. Do not edit by hand.",
      "\\begin{tabular}{@{}lccc@{}}", "\\toprule", " & PA & RO & MA \\\\", "\\midrule"]
dd = facts["domains"]
def row(label, fn):
    return label + " & " + " & ".join(fn(dd[k]) for k in ["PA", "RO", "MA"]) + " \\\\"
dl.append(row("Scene size (px)", lambda d: f"{d['H']}$\\times${d['W']}"))
dl.append(row("Tile grid", lambda d: f"{d['grid'][0]}$\\times${d['grid'][1]}"))
dl.append(row("Tiles train/val/test", lambda d: f"{d['tiles']['train']}/{d['tiles']['val']}/{d['tiles']['test']}"))
dl.append(row("MMU (px)", lambda d: str(d["mmu_px"])))
dl.append(row("Valid pixels (\\%)", lambda d: f"{100 * d['valid_frac']:.1f}"))
dl.append(row("Deforestation (\\% of valid)", lambda d: f"{100 * d['def_frac_of_valid']:.2f}"))
dl.append(row("Train / val patches", lambda d: f"{d['train_patches']:,} / {d['val_patches']:,}".replace(",", "{,}")))
dl.append(row("Test pixels (valid)", lambda d: f"{d['test_valid_px']:,}".replace(",", "{,}")))
dl += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PAPER, "tab_data.tex"), "w").write("\n".join(dl) + "\n")

# ---- figures ----
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


# transfer matrix: rows = training (source) domain, cols = test domain; diagonal = in-domain
doms = ["PA", "RO", "MA"]
M = np.zeros((3, 3)); Sd = np.zeros((3, 3))
for i, s in enumerate(doms):
    for j, t in enumerate(doms):
        if s == t:
            t0 = [tt for ss, tt in PAIRS if ss == s][0]
            v = vals("source_only", s, t0, "src_F1")
        else:
            v = vals("source_only", s, t)
        M[i, j], Sd[i, j] = np.mean(v), np.std(v, ddof=1)
fig, ax = plt.subplots(figsize=(3.2, 2.7), dpi=300)
ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.3f}\n± {Sd[i, j]:.3f}", ha="center", va="center", fontsize=7,
                color="#ffffff" if M[i, j] > 0.55 else INK)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(doms); ax.set_yticklabels(doms)
ax.set_xlabel("Test domain"); ax.set_ylabel("Training (source) domain")
ax.tick_params(length=0, colors=INK)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout(pad=0.3)
save(fig, "fig_transfer_matrix")

# per-seed results: one panel per pair, every method, each seed a dot, mean a bar tick
avail = [(m, lab) for m, _, lab in METHODS if any(vals(m, *p) for p in PAIRS)]
fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.5), dpi=300, sharey=True)
for ax, p in zip(axes, PAIRS):
    for k, (m, lab) in enumerate(avail):
        v = vals(m, *p)
        if not v:
            continue
        x = np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v))
        ax.scatter(x, v, s=12, marker=MMARK[m], color=MCOLOR[m], edgecolors="white", linewidths=0.4,
                   zorder=3, label=lab)
        ax.hlines(np.mean(v), k - 0.3, k + 0.3, color=INK, lw=1.0, zorder=4)
    base = np.mean(vals("source_only", *p))
    ax.axhline(base, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_title(f"{p[0]}→{p[1]}", fontsize=7.5, color=INK)
    ax.set_xticks([]); ax.set_xlim(-0.6, len(avail) - 0.4)
    ax.grid(True, axis="y", color=GRID, lw=0.4)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Target-test F1")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(avail), frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.07), markerscale=1.4)
fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
save(fig, "fig_method_seeds")

# The spectral / FDA figures are made by figures/spectral_analysis.py; the paper reads paper/figures.
import shutil
for n in ["fig_spectral_profiles", "fig_spectral_distance", "fig_fda_granularity"]:
    for ext in ("png", "pdf"):
        src = os.path.join(BASE, "figures", f"{n}.{ext}")
        if os.path.exists(src):  # only after figures/spectral_analysis.py has been run (needs the data)
            shutil.copy2(src, os.path.join(FIG, f"{n}.{ext}"))

print(f"{len(macros)} macros -> paper/numbers.tex; tables and figures written")
print("methods with data:", [lab for _, lab in avail])
