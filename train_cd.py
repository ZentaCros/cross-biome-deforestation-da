"""
Benchmark-faithful training for bi-temporal deforestation change detection.

Task and protocol per Vega et al. 2023 (J-STARS): 14-band bi-temporal input,
void masking (nodata, past deforestation, 2 px buffer, sub-MMU polygons),
spatially disjoint tile splits, >=2% balanced training patches with
rotation/flip augmentation, early stopping on SOURCE validation loss, and
final metrics from stitched sliding-window inference (75% overlap, center
crops) over the TARGET domain's test tiles.

Example:
  python train_cd.py --source_domain PA --target_domain RO --seed 42 \
      --experiment_name cd_source_only
"""
import os
import gc
import csv
import time
import random
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             jaccard_score, average_precision_score)

from utils import log_change
from dataset_cd import (ChangeDetectionDataset, build_domain_arrays,
                        tile_mask)
from models import FADENet
from loss import weighted_edl_loss

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def bn_adapt(model, x_path, patch=64, n_steps=200, batch=32):
    """BN-TTA: update batch norm running stats with target domain patches."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average
    model.train()
    X = np.load(x_path, mmap_mode='r')
    C, H, W = X.shape
    rng = np.random.RandomState(0)
    for _ in range(n_steps):
        buf = []
        for _ in range(batch):
            i = rng.randint(0, max(H - patch, 1))
            j = rng.randint(0, max(W - patch, 1))
            buf.append(np.array(X[:, i:i + patch, j:j + patch]))
        xb = torch.from_numpy(np.stack(buf)).float().to(device)
        model(xb)
    model.eval()
    log_change(f"BN-TTA: {n_steps} steps x {batch} patches from {os.path.basename(x_path)}", "ADAPT")


@torch.no_grad()
def stitched_scene_probs(model, x_path, patch=64, overlap=0.75, batch=64):
    """Slide over the whole scene, keep center crops, return P(class=1) [H,W]."""
    model.eval()
    X = np.load(x_path, mmap_mode='r')
    C, H, W = X.shape
    stride = max(1, int(patch * (1 - overlap)))
    margin = patch // 4

    prob_sum = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    fb_sum = np.zeros((H, W), dtype=np.float32)   # full-window fallback
    fb_cnt = np.zeros((H, W), dtype=np.float32)

    rows = list(range(0, max(H - patch, 0) + 1, stride))
    cols = list(range(0, max(W - patch, 0) + 1, stride))
    if rows[-1] != H - patch:
        rows.append(H - patch)
    if cols[-1] != W - patch:
        cols.append(W - patch)

    buf, pos = [], []
    def flush():
        if not buf:
            return
        xb = torch.from_numpy(np.stack(buf)).float().to(device)
        evidence = model(xb)
        alpha = evidence + 1.0
        p1 = (alpha[:, 1] / alpha.sum(dim=1)).cpu().numpy()
        for (i, j), pr in zip(pos, p1):
            ci0, ci1 = i + margin, i + patch - margin
            cj0, cj1 = j + margin, j + patch - margin
            prob_sum[ci0:ci1, cj0:cj1] += pr[margin:-margin, margin:-margin]
            cnt[ci0:ci1, cj0:cj1] += 1
            fb_sum[i:i + patch, j:j + patch] += pr
            fb_cnt[i:i + patch, j:j + patch] += 1
        buf.clear(); pos.clear()

    for i in rows:
        for j in cols:
            buf.append(np.array(X[:, i:i + patch, j:j + patch]))
            pos.append((i, j))
            if len(buf) >= batch:
                flush()
    flush()

    probs = np.where(cnt > 0, prob_sum / np.maximum(cnt, 1),
                     fb_sum / np.maximum(fb_cnt, 1))
    return probs


def masked_metrics(probs, y_path, v_path, domain_key, split="test", thr=0.5):
    y = np.array(np.load(y_path, mmap_mode='r'))
    v = np.array(np.load(v_path, mmap_mode='r')).astype(bool)
    H, W = y.shape
    sel = v & tile_mask(domain_key, H, W, split)
    yt = y[sel]
    pt = probs[:H, :W][sel]
    pred = (pt >= thr).astype(np.int64)
    return {
        "F1": f1_score(yt, pred, pos_label=1, average='binary', zero_division=0),
        "Precision": precision_score(yt, pred, pos_label=1, average='binary', zero_division=0),
        "Recall": recall_score(yt, pred, pos_label=1, average='binary', zero_division=0),
        "IoU": jaccard_score(yt, pred, pos_label=1, average='binary', zero_division=0),
        "AP": average_precision_score(yt, pt) if yt.max() > 0 else 0.0,
        "n_pixels": int(sel.sum()),
    }


def main(args):
    set_seed(args.seed)
    log_change(f"Device: {device} | CD protocol run", "SETUP")
    class_w = [args.class_weight_bg, args.class_weight_def]

    exp_tag = f"{args.experiment_name}_{args.source_domain}_to_{args.target_domain}_seed{args.seed}"
    results_dir = os.path.join("results", exp_tag)
    ckpt_dir = os.path.join("checkpoints", exp_tag)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")
    log_change(f"Experiment: {exp_tag} | weights={class_w}", "TRAIN")

    styled_paths = None
    if args.use_fda:
        from dataset_cd import build_styled_arrays
        styled_paths = build_styled_arrays(args.source_domain, args.target_domain,
                                           beta=args.fda_beta, n_styled=args.n_styled)
        log_change(f"Scene-level FDA: {len(styled_paths)} styled variants "
                   f"(beta={args.fda_beta})", "AFDA")

    train_ds = ChangeDetectionDataset(args.source_domain, "train",
                                      stride=args.train_stride,
                                      styled_x_paths=styled_paths)
    val_ds = ChangeDetectionDataset(args.source_domain, "val",
                                    stride=args.train_stride, augment=False,
                                    styled_x_paths=styled_paths)
    if styled_paths:
        train_ds.transfer_scale = args.transfer_scale
        val_ds.transfer_scale = args.transfer_scale
        log_change(f"transfer_scale={args.transfer_scale} (mix ratio)", "AFDA")
    # Windows spawns workers from scratch; re-spawning every epoch left the GPU idle ~60%.
    keep = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=keep)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=keep)

    model = FADENet(in_channels=14, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train_Loss", "SrcVal_Loss",
                                "SrcVal_F1", "Epoch_Time_s"])

    best_val = float('inf')
    patience_counter = 0
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                evidence = model(xb)
                loss = weighted_edl_loss(evidence, yb, epoch, 2, 10,
                                         class_weights=class_w, device=device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / max(len(train_loader), 1)

        model.eval()
        vtotal, preds, gts = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                evidence = model(xb)
                vtotal += weighted_edl_loss(evidence, yb, epoch, 2, 10,
                                            class_weights=class_w,
                                            device=device).item()
                alpha = evidence + 1.0
                p = alpha.argmax(dim=1)
                m = yb != -1
                preds.append(p[m].cpu().numpy())
                gts.append(yb[m].cpu().numpy())
        val_loss = vtotal / max(len(val_loader), 1)
        preds = np.concatenate(preds) if preds else np.array([0])
        gts = np.concatenate(gts) if gts else np.array([0])
        val_f1 = f1_score(gts, preds, pos_label=1, average='binary', zero_division=0)

        dt = time.time() - t0
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{val_f1:.6f}", f"{dt:.1f}"])
        log_change(f"Epoch [{epoch}/{args.epochs}] TrainLoss: {train_loss:.4f} "
                   f"SrcValLoss: {val_loss:.4f} SrcValF1: {val_f1:.4f}", "EVAL")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            log_change(f"New best source-val loss {val_loss:.4f}; checkpoint saved", "CHECKPOINT")
        else:
            patience_counter += 1
            if args.patience > 0 and epoch >= args.min_epochs \
                    and patience_counter >= args.patience:
                log_change(f"Early stopping at epoch {epoch}", "TRAIN")
                break

    # ===== Final stitched evaluation with the best checkpoint =====
    # Shut down the persistent loader workers (~2.4 GB) before loading full scenes.
    del train_loader, val_loader
    gc.collect()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if args.bn_tta:
        tgt_xp, _, _ = build_domain_arrays(args.target_domain)
        bn_adapt(model, tgt_xp)
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

    with open(os.path.join(results_dir, "final_summary.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["domain", "region", "F1", "Precision",
                                          "Recall", "IoU", "AP", "n_pixels"])
        w.writeheader()
        w.writerows(rows)
    log_change(f"Training complete. Target test F1: {rows[0]['F1']:.4f}", "DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source_domain", type=str, default="PA", choices=["MA", "RO", "PA"])
    p.add_argument("--target_domain", type=str, default="RO", choices=["MA", "RO", "PA"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--experiment_name", type=str, default="cd")
    p.add_argument("--class_weight_bg", type=float, default=0.4)
    p.add_argument("--class_weight_def", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_epochs", type=int, default=5)
    p.add_argument("--train_stride", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_fda", action="store_true",
                   help="Scene-level FDA: train on target-styled source variants")
    p.add_argument("--fda_beta", type=float, default=0.01)
    p.add_argument("--n_styled", type=int, default=3)
    p.add_argument("--transfer_scale", type=float, default=0.5,
                   help="Probability of drawing a styled patch (0=clean only, 1=styled only)")
    p.add_argument("--bn_tta", action="store_true",
                   help="BN-TTA: adapt batch norm stats to target domain before evaluation")
    args = p.parse_args()
    main(args)
