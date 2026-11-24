"""Write README.md. The results table and headline numbers are read from results/paperA_summary.csv and
paper/numbers.tex, so nothing in the README is typed by hand. Run after make_paper_numbers.py."""
import csv
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "results" / "paperA_summary.csv")))
N = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\n", (HERE / "paper" / "numbers.tex").read_text()))
N = {k: v.replace("$-$", "−").replace("{,}", ",") for k, v in N.items()}
PAIRS = [("PA", "RO"), ("RO", "PA"), ("MA", "RO"), ("RO", "MA"), ("PA", "MA"), ("MA", "PA")]
METHODS = [("source_only", "Source-only"), ("bntta", "+ BN-TTA"), ("fda_mix50", "FDA mix50"),
           ("fda_mix50_bntta", "FDA mix50 + BN-TTA"), ("fda_scale1", "FDA scale 1.0"),
           ("fda_scale1_bntta", "FDA scale 1.0 + BN-TTA")]


def table(key):
    def vals(m, s, t):
        return [float(r[key]) for r in rows if r["method"] == m and r["source"] == s and r["target"] == t]
    out = ["| Method | " + " | ".join(f"{s}→{t}" for s, t in PAIRS) + " |",
           "|---|" + "---:|" * len(PAIRS)]
    best = {p: max(np.mean(vals(m, *p)) for m, _ in METHODS) for p in PAIRS}
    for m, lab in METHODS:
        cells = []
        for p in PAIRS:
            v = vals(m, *p)
            c = f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
            cells.append(f"**{c}**" if abs(np.mean(v) - best[p]) < 1e-12 else c)
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(out)


assert len(rows) == 108
README = f"""# How much do simple domain adaptation methods help cross-biome deforestation detection?

**A multi-seed evaluation on Landsat-8.** Code, per-run results and paper source.

![Study sites](paper/figures/ms/fig_study_area_code.jpg)
<sub>The three Landsat-8 sites: Pará (PA) and Rondônia (RO) in the Amazon, Maranhão (MA) in the
Amazon–Cerrado transition, with the reference new deforestation (magenta), the training and validation tiles,
and 6 × 6 km detail windows. Every figure in this repository is drawn in code from the study data.</sub>

## What this study asks

Deep change-detection models trained on one region lose accuracy on another. Two adaptation methods are
cheap enough to try first: re-estimating batch-normalization statistics on the target scene at test time
(**BN-TTA**), and restyling the source images with the low-frequency spectrum of the target
(**Fourier Domain Adaptation, FDA**). How much do they actually help, and how reliably?

We evaluate all **6 directed source → target pairs × 3 seeds × 6 configurations = 108 runs**, reporting
F1, precision and recall at a fixed threshold, and threshold-free average precision (AP).

## Main findings

- **Transfer is strongly directional, and seeds matter.** Without adaptation, target F1 ranges from
  {N['SOMin']} (PA→RO) to {N['SOMax']} (MA→PA). For RO→PA, three seeds of an otherwise identical run give
  {N['SOWidestLo']} to {N['SOWidestHi']}, as wide as most method effects.
- **BN-TTA improves ranking more reliably than thresholded accuracy.** It raises AP in {N['TTAAPAllW']} of
  {N['TTAAPAllN']} paired runs (mean {N['TTAAPAllD']}) but F1 in only {N['TTAAllW']} of {N['TTAAllN']} (mean
  {N['TTAAllD']}). On the pairs trained on MA it raises recall, lowers precision, and F1 falls.
- **FDA at β = 0.01 does nothing on 64 × 64 patches**: the swap window is empty. Applied to 512 × 512 scene
  tiles, it raises F1 in {N['MIXSrcROW']} of {N['MIXSrcRON']} paired runs when training on RO (mean
  {N['MIXSrcROD']}) and lowers it in {N['MIXSrcMAL']} of {N['MIXSrcMAN']} when training on MA.
- **No configuration is best on every pair**; for both MA-source pairs the unadapted model is best.

<p align="center"><img src="paper/figures/fig_method_seeds.png" width="100%"></p>
<sub>Target F1 per seed (markers) and mean (black tick) for every method and pair; the dashed line is the
source-only mean.</sub>

### Target-domain F1 (mean ± std over seeds 42, 123, 456; bold = best per pair)

{table("tgt_F1")}

### Target-domain average precision (AP)

{table("tgt_AP")}

### What the methods change on the map

![Qualitative results](paper/figures/ms/fig_qualitative.jpg)
<sub>A 9 × 9 km PA test window, for the seed with the median effect of each method. (a) BN-TTA turns most
missed deforestation into detections but adds false alarms. (b) FDA removes most of the source-only false
alarms, while the misses remain.</sub>

### The FDA patch-size pitfall

![FDA swap size](paper/figures/fig_fda_granularity.png)
<sub>FDA swaps a square of half-width ⌊βN⌋ around zero frequency. At N = 64 and β = 0.01 that square is
empty; at N = 512 it is 10 × 10 bins.</sub>

<table><tr>
<td width="45%"><img src="paper/figures/fig_transfer_matrix.png"><br><sub>Source-only F1: rows are the
training site, columns the test site; the diagonal is in-domain accuracy.</sub></td>
<td width="55%"><img src="paper/figures/fig_spectral_distance.png"><br><sub>Spectral distance between sites on
raw and normalized inputs. The closest pair after normalization (PA–RO) is the hardest transfer.</sub></td>
</tr></table>

## Method

![Workflow](paper/figures/ms/fig_methodology.jpg)

![Architecture](paper/figures/ms/fig_architecture.png)
<sub>FADE-Net as run ({N['ParamsM']} M parameters): a small encoder–decoder with an evidential (Dirichlet)
output. The bottleneck uses a ConvNeXt-style block because the planned Mamba block was not available.</sub>

## Reproducing the paper

**Without data or GPU:** regenerate every number, table and result figure from the committed run outputs and
compile the paper.

```bash
pip install -r requirements.txt
python aggregate_results.py
python make_paper_numbers.py
python paper/audit_numbers.py          # expects: 0 unlisted numbers
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**With the data and a CUDA GPU:** rerun the experiments, then the steps above.

```bash
python run_sweep_cd.py --worker A      # optionally in parallel: --worker B
python aggregate_results.py
python paper_facts.py
python figures/spectral_analysis.py
python figures/fig_study_area_code.py
python figures/fig_architecture.py
python figures/fig_methodology.py
python figures/fig_qualitative.py      # needs the trained checkpoints
python figures/make_ms_copies.py
python make_paper_numbers.py
```

Training is deterministic on our hardware: the same source site and seed reproduce the same checkpoint.
Model weights (about 2.8 GB) are not stored in git and are available from the author on request.

## Data

The Landsat-8 images and PRODES-derived reference maps are the public benchmark data of Noa et al. (2021) and
de Moura et al. (2025); download links are given in the Data Availability Statement of de Moura et al.
(2025), *Forests* 16(5):742, https://doi.org/10.3390/f16050742. Place the `.npy` files at the paths listed in
`DOMAIN_DATA` in `dataset_cd.py` (under `data/`) and verify them:

```bash
sha256sum -c data_aux/SHA256SUMS.txt
```

The loader refuses damaged inputs: image values must be 16-bit digital numbers (NaN no-data on at most 1 % of
pixels) and reference rasters must contain only 0 and 1. While preparing this study, one copy of the RO files
had been damaged during a file copy without any error; the checksums are those of the intact files used here.

## Repository layout

| Path | Contents |
|---|---|
| `dataset_cd.py` | Input checks, masks (past deforestation, 2-px buffer, minimum mapping unit), tile split, patches |
| `fda.py` | Fourier Domain Adaptation and its scene-level (512 px tile) application |
| `models.py`, `loss.py` | FADE-Net (as run) and the weighted evidential loss |
| `train_cd.py`, `eval_bn_tta.py` | Training with early stopping on source validation loss; evaluation with optional BN-TTA |
| `run_sweep_cd.py`, `watch_progress.py` | The 108-run sweep (safe to run as parallel workers) and a progress monitor |
| `aggregate_results.py` | Collects all runs into `results/paperA_summary.csv` |
| `paper_facts.py` | Measures data sizes, patch counts, parameter counts and epochs into `paper/facts.json` |
| `make_paper_numbers.py` | Writes every number, table and result figure of the paper |
| `make_readme.py` | Writes this README from the same results |
| `paper/audit_numbers.py` | Fails if any number in the paper text is typed by hand instead of generated |
| `figures/` | Scripts for all figures |
| `results/` | Test metrics (`final_summary.csv`) and per-epoch logs (`metrics.csv`) for each of the 108 runs |
| `paper/` | LaTeX source (`main.tex`) and `pfg/build_pfg.py` (Springer Nature layout) |
| `data_aux/` | SHA-256 checksums of the input files; Natural Earth boundaries for the study-area map |

## Related repository

This repository replaces the results of the author's earlier project
[`convnext-edl-deforestation-amazon`](https://github.com/ZentaCros/convnext-edl-deforestation-amazon), whose
reported numbers were obtained with damaged RO input files and a different protocol and should not be used.

## Licence

Code: MIT (see `LICENSE`). The input data remain under the terms of their original providers.
"""
(HERE / "README.md").write_text(README, encoding="utf-8", newline="\n")
print("README.md written,", len(README.splitlines()), "lines")
