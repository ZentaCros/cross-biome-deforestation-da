"""Collect Paper A sweep results into results/paperA_summary.csv and print tables."""
import os
import csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
PAIRS = [("PA", "RO"), ("RO", "PA"), ("MA", "RO"),
         ("RO", "MA"), ("PA", "MA"), ("MA", "PA")]
SEEDS = [42, 123, 456]
METHODS = {
    "source_only": "cd_source_only_{s}_to_{t}_seed{sd}",
    "bntta": "cd_source_only_bntta_{s}_to_{t}_seed{sd}",
    "fda_mix50": "cd_fda_mix50_{s}_to_{t}_seed{sd}",
    "fda_mix50_bntta": "cd_fda_mix50_bntta_{s}_to_{t}_seed{sd}",
    "fda_scale1": "cd_fda_{s}_to_{t}_seed{sd}",
    "fda_scale1_bntta": "cd_fda_bntta_{s}_to_{t}_seed{sd}",
}
SHORT = {"source_only": "src-only", "bntta": "+BN-TTA", "fda_mix50": "mix50",
         "fda_mix50_bntta": "mix50+TTA", "fda_scale1": "s=1.0", "fda_scale1_bntta": "s=1.0+TTA"}


def read(tag):
    path = os.path.join(RESULTS, tag, "final_summary.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {(r["domain"], r["region"]): r for r in csv.DictReader(f)}


def main():
    rows = []
    for method, pat in METHODS.items():
        for s, t in PAIRS:
            for sd in SEEDS:
                res = read(pat.format(s=s, t=t, sd=sd))
                if res is None:
                    continue
                tgt, src = res[(t, "test")], res[(s, "test")]
                rows.append({"method": method, "source": s, "target": t, "seed": sd,
                             "tgt_F1": float(tgt["F1"]), "tgt_P": float(tgt["Precision"]),
                             "tgt_R": float(tgt["Recall"]), "tgt_IoU": float(tgt["IoU"]),
                             "tgt_AP": float(tgt["AP"]), "src_F1": float(src["F1"])})
    out = os.path.join(RESULTS, "paperA_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out}\n")

    def pick(m, s, t, key="tgt_F1", seed=None):
        return [r[key] for r in rows if r["method"] == m and r["source"] == s
                and r["target"] == t and (seed is None or r["seed"] == seed)]

    def ms(v):
        return f"{np.mean(v):.4f}+-{np.std(v, ddof=1):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "-")

    print("Target-test F1, mean+-std over seeds (n shown)")
    print(f"{'pair':8s} {'src-only':>16s} {'+BN-TTA':>16s} {'srcF1 src-only':>16s} {'srcF1 +BN-TTA':>16s}")
    for s, t in PAIRS:
        a, b = pick("source_only", s, t), pick("bntta", s, t)
        print(f"{s}->{t:5s} {ms(a):>16s} {ms(b):>16s} "
              f"{ms(pick('source_only', s, t, 'src_F1')):>16s} {ms(pick('bntta', s, t, 'src_F1')):>16s}  n={len(a)},{len(b)}")

    print("\nSeed 42 only, all methods (target-test F1 | source-test F1)")
    print(f"{'pair':8s} {'src-only':>15s} {'+BN-TTA':>15s} {'FDA mix50':>15s} {'FDA+BN-TTA':>15s} {'FDA s=1.0':>15s}")
    for s, t in PAIRS:
        cells = []
        for m in ["source_only", "bntta", "fda_mix50", "fda_mix50_bntta", "fda_scale1"]:
            v, sv = pick(m, s, t, seed=42), pick(m, s, t, "src_F1", seed=42)
            cells.append(f"{v[0]:.4f}|{sv[0]:.3f}" if v else "-")
        print(f"{s}->{t:5s} " + " ".join(f"{c:>15s}" for c in cells))

    print("\nAll methods, target-test F1 mean+-std over seeds (n in brackets)")
    print(f"{'pair':8s} " + " ".join(f"{SHORT[m]:>20s}" for m in METHODS))
    for s, t in PAIRS:
        print(f"{s}->{t:5s} " + " ".join(
            f"{ms(pick(m, s, t)) + f' [{len(pick(m, s, t))}]':>20s}" for m in METHODS))

    # Per-seed change vs source-only, matched by seed (same source model / same seed).
    for m in [m for m in METHODS if m != "source_only"]:
        print(f"\nPer-seed effect of {SHORT[m]} vs source-only (seeds {SEEDS}; target F1 diff)")
        for s, t in PAIRS:
            d = []
            for sd in SEEDS:
                a, b = pick("source_only", s, t, seed=sd), pick(m, s, t, seed=sd)
                d.append(f"{b[0] - a[0]:+.4f}" if a and b else "   -   ")
            print(f"{s}->{t:5s} " + "  ".join(d))


if __name__ == "__main__":
    main()
