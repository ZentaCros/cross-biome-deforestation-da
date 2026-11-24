"""Measure every dataset / model / protocol fact Paper A states; write paper/facts.json.

Nothing here is typed in by hand: sizes come from the cached arrays, patch counts
from the dataset class, parameters from the model, epochs from each run's metrics.csv.
CPU only (safe to run while the GPU sweep is going).
"""
import os
import csv
import json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
from dataset_cd import (DOMAIN_DATA, MMU_PIXELS, BUFFER_PX, MIN_DEF_FRACTION,
                        ChangeDetectionDataset, build_domain_arrays, tile_mask, tile_split)
import models

PAIRS = [("PA", "RO"), ("RO", "PA"), ("MA", "RO"), ("RO", "MA"), ("PA", "MA"), ("MA", "PA")]
SEEDS = [42, 123, 456]
facts = {"domains": {}, "model": {}, "fda": {}, "runs": {}}

# ---- domains ----
for dk in ["PA", "RO", "MA"]:
    xp, yp, vp = build_domain_arrays(dk)
    C, H, W = np.load(xp, mmap_mode="r").shape
    y = np.load(yp).astype(bool)
    v = np.load(vp).astype(bool)
    d = {"bands": int(C), "H": int(H), "W": int(W),
         "valid_frac": float(v.mean()),
         "def_frac_of_valid": float(y[v].mean()),
         "grid": list(DOMAIN_DATA[dk]["grid"]),
         "tiles": {k: len(s) for k, s in tile_split(dk).items()},
         "mmu_px": MMU_PIXELS[dk]}
    for split in ["train", "val", "test"]:
        sel = v & tile_mask(dk, H, W, split)
        d[f"{split}_valid_px"] = int(sel.sum())
        d[f"{split}_def_px"] = int((y & sel).sum())
    for split in ["train", "val"]:
        d[f"{split}_patches"] = len(ChangeDetectionDataset(dk, split, stride=8).patch_coords)
    # raw image no-data (NaN) per date, as the loader reports it
    for date in ["img_t1", "img_t2"]:
        raw = np.load(os.path.join(BASE, DOMAIN_DATA[dk][date]), mmap_mode="r")
        arr = np.asarray(raw)
        if arr.shape[-1] < 20:
            arr = np.moveaxis(arr, -1, 0)
        d[f"{date}_raw_shape"] = list(arr.shape)
        d[f"{date}_nodata_frac"] = float(np.isnan(arr).any(axis=0).mean())
        d[f"{date}_file"] = os.path.basename(DOMAIN_DATA[dk][date])
        del arr
    facts["domains"][dk] = d
    del y, v

facts["protocol"] = {"buffer_px": BUFFER_PX, "min_def_fraction": MIN_DEF_FRACTION,
                     "patch": 64, "train_stride": 8}

# ---- model ----
net = models.FADENet(in_channels=14, num_classes=2, base_dim=128)
facts["model"] = {"mamba_available": bool(models.MAMBA_AVAILABLE),
                  "params": int(sum(p.numel() for p in net.parameters())),
                  "trainable_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
                  "base_dim": 128}
for name, mod in [("encoder", net.encoder), ("decoder", net.decoder),
                  ("bottleneck", net.encoder.mamba_bottleneck)]:
    facts["model"][f"{name}_params"] = int(sum(p.numel() for p in mod.parameters()))

# ---- FDA swap size: square of half-width int(N*beta) around DC ----
for N in [64, 128, 256, 512]:
    b = int(N * 0.01)
    facts["fda"][f"N{N}"] = {"half_width": b, "swapped_bins": (2 * b) ** 2, "total_bins": N * N}
# Edge tiles left unstyled by style_scene (int(size*beta) < 1)
for dk in ["PA", "RO", "MA"]:
    H, W = facts["domains"][dk]["H"], facts["domains"][dk]["W"]
    # style_scene runs on the raw scene; use raw size of date 1
    _, rH, rW = facts["domains"][dk]["img_t1_raw_shape"]
    tiles = [(min(512, rH - i), min(512, rW - j)) for i in range(0, rH, 512) for j in range(0, rW, 512)]
    unstyled = [(h, w) for h, w in tiles if int(h * 0.01) < 1 or int(w * 0.01) < 1]
    px_unstyled = sum(h * w for h, w in unstyled)
    facts["fda"][f"{dk}_tiles"] = {"n_tiles": len(tiles), "n_unstyled": len(unstyled),
                                   "unstyled_px_frac": px_unstyled / (rH * rW)}

# ---- runs: epochs to best checkpoint / stop ----
def epochs(tag):
    p = os.path.join(BASE, "results", tag, "metrics.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    best, best_ep = float("inf"), None
    for r in rows:
        v = float(r["SrcVal_Loss"])
        if v < best - 1e-5:
            best, best_ep = v, int(r["Epoch"])
    return {"best_epoch": best_ep, "stop_epoch": int(rows[-1]["Epoch"]),
            "mean_epoch_s": float(np.mean([float(r["Epoch_Time_s"]) for r in rows]))}

for exp in ["cd_source_only", "cd_fda_mix50", "cd_fda"]:
    for s, t in PAIRS:
        for sd in SEEDS:
            tag = f"{exp}_{s}_to_{t}_seed{sd}"
            e = epochs(tag)
            if e:
                facts["runs"][tag] = e

out = os.path.join(BASE, "paper", "facts.json")
with open(out, "w") as f:
    json.dump(facts, f, indent=1)
print(json.dumps({k: v for k, v in facts.items() if k != "runs"}, indent=1))
for exp in ["cd_source_only", "cd_fda_mix50", "cd_fda"]:
    b = [v["best_epoch"] for k, v in facts["runs"].items() if k.startswith(exp + "_") and
         (exp != "cd_fda" or not k.startswith("cd_fda_mix50"))]
    st = [v["stop_epoch"] for k, v in facts["runs"].items() if k.startswith(exp + "_") and
          (exp != "cd_fda" or not k.startswith("cd_fda_mix50"))]
    if b:
        print(f"{exp}: n={len(b)} best epoch min/median/max = {min(b)}/{int(np.median(b))}/{max(b)}; "
              f"stop epoch {min(st)}/{int(np.median(st))}/{max(st)}")
