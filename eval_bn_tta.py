"""Evaluate an existing checkpoint with optional BN-TTA adaptation."""
import os
import csv
import argparse
import numpy as np
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             jaccard_score, average_precision_score)

from utils import log_change
from dataset_cd import build_domain_arrays, tile_mask
from models import FADENet
from train_cd import stitched_scene_probs, masked_metrics, bn_adapt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    model = FADENet(in_channels=14, num_classes=2).to(device)
    ckpt = args.checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log_change(f"Loaded {ckpt}", "EVAL")

    if args.bn_tta:
        tgt_xp, _, _ = build_domain_arrays(args.target_domain)
        bn_adapt(model, tgt_xp, n_steps=args.bn_steps)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for dom, split in [(args.target_domain, "test"),
                       (args.target_domain, "all"),
                       (args.source_domain, "test")]:
        xp, yp, vp = build_domain_arrays(dom)
        probs = stitched_scene_probs(model, xp)
        if split == "all":
            y = np.array(np.load(yp, mmap_mode='r'))
            v = np.array(np.load(vp, mmap_mode='r')).astype(bool)
            yt, pt = y[v], probs[:y.shape[0], :y.shape[1]][v]
            pred = (pt >= 0.5).astype(np.int64)
            m = {"F1": f1_score(yt, pred, pos_label=1, zero_division=0),
                 "Precision": precision_score(yt, pred, pos_label=1, zero_division=0),
                 "Recall": recall_score(yt, pred, pos_label=1, zero_division=0),
                 "IoU": jaccard_score(yt, pred, pos_label=1, zero_division=0),
                 "AP": average_precision_score(yt, pt) if yt.max() > 0 else 0.0,
                 "n_pixels": int(v.sum())}
        else:
            m = masked_metrics(probs, yp, vp, dom, split)
        m.update({"domain": dom, "region": split})
        rows.append(m)
        log_change(f"[{dom}/{split}] F1={m['F1']:.4f} P={m['Precision']:.4f} "
                   f"R={m['Recall']:.4f} IoU={m['IoU']:.4f} AP={m['AP']:.4f} "
                   f"({m['n_pixels']} px)", "RESULT")

    csv_path = os.path.join(out_dir, "final_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["domain", "region", "F1", "Precision",
                                          "Recall", "IoU", "AP", "n_pixels"])
        w.writeheader()
        w.writerows(rows)
    log_change(f"Done. Target test F1: {rows[0]['F1']:.4f}", "DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--source_domain", type=str, default="PA")
    p.add_argument("--target_domain", type=str, default="RO")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--bn_tta", action="store_true")
    p.add_argument("--bn_steps", type=int, default=200)
    args = p.parse_args()
    main(args)
