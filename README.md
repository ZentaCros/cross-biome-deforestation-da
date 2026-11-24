# How much do simple domain adaptation methods help cross-biome deforestation detection?

Code, per-run results and paper source for a multi-seed evaluation of two inexpensive domain adaptation
methods for Landsat-8 deforestation change detection on three Brazilian sites: Pará (PA) and Rondônia (RO)
in the Amazon, and Maranhão (MA) in the Amazon–Cerrado transition.

- **Methods compared:** source-only training, test-time re-estimation of batch-normalization statistics
  (BN-TTA), and scene-level Fourier Domain Adaptation (FDA) with 50 % or 100 % styled patches, each with and
  without BN-TTA.
- **Design:** all 6 directed source→target pairs × 3 seeds × 6 configurations = 108 runs.
- **Metrics:** F1, precision and recall of the deforestation class at threshold 0.5, and average precision
  (AP), on the test tiles of the target scene.

Main findings (all numbers are generated from `results/`):

- Without adaptation, target F1 ranges from 0.131 (PA→RO) to 0.711 (MA→PA); for one pair the three seeds
  span 0.130–0.487.
- BN-TTA raises AP in 14 of 18 paired runs but F1 in only 12 of 18; on the pairs trained on MA it raises
  recall, lowers precision, and F1 falls.
- FDA with β = 0.01 does nothing on 64 × 64 patches (the swap window is empty). Applied to 512 × 512 scene
  tiles, it helps when training on RO and hurts when training on MA, and it does not improve AP overall.

## Repository layout

| Path | Contents |
|---|---|
| `dataset_cd.py` | Input checks, masks (past deforestation, 2-px buffer, minimum mapping unit), tile split, patches |
| `fda.py` | Fourier Domain Adaptation and its scene-level (512 px tile) application |
| `models.py`, `loss.py` | FADE-Net (as run: convolutional fallback in the bottleneck) and the weighted evidential loss |
| `train_cd.py`, `eval_bn_tta.py` | Training with early stopping on source validation loss; evaluation with optional BN-TTA |
| `run_sweep_cd.py`, `watch_progress.py` | The 108-run sweep (safe to run as parallel workers) and a progress monitor |
| `aggregate_results.py` | Collects all runs into `results/paperA_summary.csv` |
| `paper_facts.py` | Measures data sizes, patch counts, parameter counts and epochs into `paper/facts.json` |
| `make_paper_numbers.py` | Writes every number, table and result figure of the paper (`paper/numbers.tex`, `paper/tab_*.tex`) |
| `paper/audit_numbers.py` | Fails if any number in the text is typed by hand instead of generated |
| `figures/` | Scripts for all figures, drawn in code from the study data |
| `results/` | `final_summary.csv` (test metrics) and `metrics.csv` (per-epoch log) for each of the 108 runs |
| `paper/` | LaTeX source (`main.tex`, preprint layout) and `pfg/build_pfg.py` (Springer Nature layout) |
| `data_aux/` | SHA-256 checksums of the input files; Natural Earth boundaries for the study-area map |

## Data

The Landsat-8 images and PRODES-derived reference maps are the public benchmark data of Noa et al. (2021) and
de Moura et al. (2025); download links are given in the Data Availability Statement of de Moura et al. (2025),
*Forests* 16(5):742, https://doi.org/10.3390/f16050742. Place the `.npy` files at the paths listed in
`DOMAIN_DATA` in `dataset_cd.py` (under `data/`) and check them:

```bash
sha256sum -c data_aux/SHA256SUMS.txt
```

The loader refuses damaged inputs: image values must be 16-bit digital numbers (NaN no-data allowed on at
most 1 % of pixels) and reference rasters must contain only 0 and 1. While preparing this study, one copy of
the RO files had been damaged during a file copy without any error; the checksums above are those of the
intact files used for every result here.

## Reproducing the paper

Without data or GPU, regenerate all numbers, tables and result figures from the committed run outputs and
compile the paper:

```bash
pip install -r requirements.txt
python aggregate_results.py
python make_paper_numbers.py
python paper/audit_numbers.py          # expects: 0 unlisted numbers
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

With the data and a CUDA GPU, rerun the experiments:

```bash
python run_sweep_cd.py --worker A      # optionally a second worker: --worker B
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

Training is deterministic on our hardware: the same source domain and seed reproduce the same checkpoint.
Model weights (`checkpoints/`, about 2.8 GB) are not stored in git; they are available from the author on
request.

## Use of AI tools

Claude (Anthropic), used through Claude Code, helped write and debug this code and draft the paper text, as
described in Section 4.5 of the paper. Every reported number is generated from the logged run outputs and
checked by `paper/audit_numbers.py`.

## Related repository

This repository replaces the results of the author's earlier project
[`convnext-edl-deforestation-amazon`](https://github.com/ZentaCros/convnext-edl-deforestation-amazon), whose
reported numbers were obtained with damaged RO input files and a different protocol and should not be used.

## Licence

Code: MIT (see `LICENSE`). The input data remain under the terms of their original providers.
