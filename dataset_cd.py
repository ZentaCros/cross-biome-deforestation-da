"""
Bi-temporal change-detection dataset following the benchmark protocol of
Vega et al. 2023 (J-STARS, DOI 10.1109/JSTARS.2023.3327573) on the
PA / RO / MA Landsat-8 deforestation sites.

Protocol implemented here:
- Input: date-1 and date-2 scenes concatenated -> 14 bands, each band
  z-score normalized per scene.
- Label: REFERENCE raster (new deforestation between the two dates).
- Inputs are checked on load: image values must be 16-bit DNs or NaN no-data
  (at most 1% of pixels) and references strictly 0/1, otherwise loading fails.
- Mask (excluded from loss and metrics): image no-data on either date, pixels
  deforested before date 1 (PAST_REFERENCE == 1), a 2-pixel buffer around
  new-deforestation polygons, and polygons below the PRODES minimum mapping
  unit (69 px for Amazon sites, 11 px for the Cerrado/MA site).
- Spatially disjoint tile splits per domain (seeded, fixed): RO on a 10x10
  grid with 20%/5%/75% train/val/test tiles; PA and MA on a 5x3 grid with
  26%/13%/60% (4/2/9 tiles).
- Training patches: 64x64, sliding window inside train tiles, kept only if
  they hold >= 2% new-deforestation pixels among valid ones; augmented with
  90-degree rotations and flips at __getitem__ time.
"""
import os
import gc
import random
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

from utils import log_change

MMU_PIXELS = {"MA": 11, "RO": 69, "PA": 69}  # PRODES minimum mapping unit
BUFFER_PX = 2
MIN_DEF_FRACTION = 0.02

DOMAIN_DATA = {
    "MA": {
        "name": "Cerrado_MA",
        "img_t1": "data/Cerrado_Images/18_08_2017_image_R220_63_MA.npy",
        "img_t2": "data/Cerrado_Images/21_08_2018_image_R220_63_MA.npy",
        "past_ref": "data/Cerrado_References/PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_MA.npy",
        "ref": "data/Cerrado_References/REFERENCE_2018_EPSG4674_R220_63_MA.npy",
        "grid": (5, 3), "tiles": (4, 2),   # train, val tiles out of 15; rest test
    },
    "RO": {
        "name": "Amazon_RO",
        "img_t1": "data/Amazon_Images/18_07_2016_image_R232_67_RO.npy",
        "img_t2": "data/Amazon_Images/21_07_2017_image_R232_67_RO.npy",
        "past_ref": "data/Amazon_References/PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67.npy",
        "ref": "data/Amazon_References/REFERENCE_2017_EPSG32620_R232_67_RO.npy",
        "grid": (10, 10), "tiles": (20, 5),  # out of 100; rest test
    },
    "PA": {
        "name": "Amazon_PA",
        "img_t1": "data/Amazon_PA_Images/02_08_2016_image_R225_62_PA.npy",
        "img_t2": "data/Amazon_PA_Images/20_07_2017_image_R225_62_PA.npy",
        "past_ref": "data/Amazon_References/PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62.npy",
        "ref": "data/Amazon_References/REFERENCE_2017_EPSG4674_R225_62_PA.npy",
        "grid": (5, 3), "tiles": (4, 2),
    },
}

SPLIT_SEED = 42  # tile assignment is fixed across every experiment


MAX_NODATA_FRACTION = 0.01


def _load_band_stack(path, return_nodata=False):
    """[7, H, W] float32. NaN pixels are no-data: filled with the band median
    so they do not skew normalization, and reported via the no-data mask."""
    img = np.load(path)
    if img.shape[-1] < 20:  # channels-last layouts
        img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    nan = np.isnan(img)
    # Landsat-8 Collection-1 DNs are 16-bit. Anything else means a damaged file
    # (a broken copy of the RO scenes once passed silently as zeros and +-1e10).
    bad = ~nan & ~(np.isfinite(img) & (img > 0) & (img < 65536))
    nodata = nan.any(axis=0)
    if bad.any() or nodata.mean() > MAX_NODATA_FRACTION:
        raise ValueError(f"{path}: {100 * bad.mean():.3f}% out-of-range values and "
                         f"{100 * nodata.mean():.2f}% no-data pixels; the file is likely damaged")
    for b in range(img.shape[0]):
        if nan[b].any():
            img[b][nan[b]] = np.median(img[b][~nan[b]])
    return (img, nodata) if return_nodata else img


def _normalize_scene(img):
    C, H, W = img.shape
    flat = img.reshape(C, -1).T
    flat = StandardScaler().fit_transform(flat)
    return flat.T.reshape(C, H, W).astype(np.float32)


def _clean_ref(path):
    """Reference raster -> (binary mask of 1s, validity mask)."""
    a = np.load(path).astype(np.float64)
    odd = ~((a == 0) | (a == 1))
    if odd.any():
        raise ValueError(f"{path}: {100 * odd.mean():.2f}% of values are not 0/1; "
                         f"the reference raster is likely damaged")
    valid = np.ones(a.shape, dtype=bool)
    ones = a == 1
    return ones.astype(np.uint8), valid


def build_domain_arrays(domain_key, cache_dir=None):
    """Preprocess one domain once; cache X (14,H,W), y (H,W), valid (H,W)."""
    spec = DOMAIN_DATA[domain_key]
    base = os.path.dirname(os.path.abspath(__file__))
    if cache_dir is None:
        cache_dir = os.path.join(base, ".cache_cd", domain_key)
    os.makedirs(cache_dir, exist_ok=True)
    px, py, pv = (os.path.join(cache_dir, n + ".npy") for n in ("X", "y", "valid"))

    if all(os.path.exists(p) for p in (px, py, pv)):
        return px, py, pv

    log_change(f"[{domain_key}] Building bi-temporal change-detection arrays...", "DATA")
    t1, nd1 = _load_band_stack(os.path.join(base, spec["img_t1"]), return_nodata=True)
    t2, nd2 = _load_band_stack(os.path.join(base, spec["img_t2"]), return_nodata=True)
    t1, t2 = _normalize_scene(t1), _normalize_scene(t2)
    H = min(t1.shape[1], t2.shape[1]); W = min(t1.shape[2], t2.shape[2])

    ref, ref_valid = _clean_ref(os.path.join(base, spec["ref"]))
    past, past_valid = _clean_ref(os.path.join(base, spec["past_ref"]))
    H = min(H, ref.shape[0], past.shape[0]); W = min(W, ref.shape[1], past.shape[1])
    t1, t2 = t1[:, :H, :W], t2[:, :H, :W]
    ref, ref_valid = ref[:H, :W], ref_valid[:H, :W]
    past, past_valid = past[:H, :W], past_valid[:H, :W]

    X = np.concatenate([t1, t2], axis=0)  # [14, H, W]
    del t1, t2

    # Valid pixels: image data on both dates, never previously deforested
    img_ok = ~(nd1[:H, :W] | nd2[:H, :W])
    valid = ref_valid & past_valid & (past == 0) & img_ok

    # Remove polygons under the minimum mapping unit (counted as void)
    labeled, n = ndimage.label(ref)
    if n:
        sizes = ndimage.sum_labels(np.ones_like(ref), labeled, index=np.arange(1, n + 1))
        small_ids = np.flatnonzero(sizes < MMU_PIXELS[domain_key]) + 1
        if small_ids.size:
            small = np.isin(labeled, small_ids)
            valid &= ~small
            ref = ref.copy(); ref[small] = 0
    # 2-pixel buffer ring around remaining polygons is void
    dil = ndimage.binary_dilation(ref.astype(bool), iterations=BUFFER_PX)
    valid &= ~(dil & (ref == 0))

    y = ref.astype(np.int64)
    # Write then rename, so a parallel run never loads a half-written cache file.
    for path, arr in ((py, y), (pv, valid.astype(np.uint8)), (px, X.astype(np.float32))):
        tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
    stats = 100 * y[valid.astype(bool)].mean() if valid.any() else 0.0
    log_change(f"[{domain_key}] X{X.shape} valid={100*valid.mean():.1f}% "
               f"deforestation={stats:.2f}% of valid", "DATA")
    del X, y, valid
    gc.collect()
    return px, py, pv


def build_styled_arrays(src_key, tgt_key, beta=0.01, n_styled=3):
    """Scene-level FDA for the bi-temporal task: style each raw source date
    with crops from the matching raw target date, then normalize and concat.
    Returns list of styled X cache paths (same shape as the source X)."""
    from fda import style_scene
    base = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base, ".cache_cd", src_key)
    os.makedirs(cache_dir, exist_ok=True)
    xp, _, _ = build_domain_arrays(src_key)
    ref_shape = np.load(xp, mmap_mode='r').shape  # (14, H, W)

    src_spec, tgt_spec = DOMAIN_DATA[src_key], DOMAIN_DATA[tgt_key]
    paths = []
    raw_tgt = None
    for k in range(n_styled):
        out = os.path.join(cache_dir, f"X_styled_{tgt_key}_b{beta:g}_{k}.npy")
        if not os.path.exists(out):
            if raw_tgt is None:
                raw_tgt = {d: _load_band_stack(os.path.join(base, tgt_spec[d]))
                           for d in ("img_t1", "img_t2")}
            dates = []
            for d in ("img_t1", "img_t2"):
                src_raw = _load_band_stack(os.path.join(base, src_spec[d]))
                # Python's hash() of a str changes every interpreter run; crc32 does not.
                # One seed per (pair, beta, variant, date) so a variant never depends
                # on which other variants were already cached.
                seed = zlib.crc32(f"{src_key}->{tgt_key}-b{beta:g}-v{k}-{d}".encode())
                styled = style_scene(src_raw, [raw_tgt[d]], beta,
                                     rng=np.random.default_rng(seed))
                del src_raw
                dates.append(_normalize_scene(styled))
                del styled
            H = min(dates[0].shape[1], dates[1].shape[1], ref_shape[1])
            W = min(dates[0].shape[2], dates[1].shape[2], ref_shape[2])
            X = np.concatenate([dates[0][:, :H, :W], dates[1][:, :H, :W]], axis=0)
            del dates
            # Write then rename, so a parallel run never loads a half-written file.
            tmp = f"{out[:-4]}.{os.getpid()}.tmp.npy"
            np.save(tmp, X.astype(np.float32))
            os.replace(tmp, out)
            del X
            gc.collect()
            log_change(f"[{src_key}->{tgt_key}] styled X variant {k} saved (beta={beta})", "DATA")
        paths.append(out)
    del raw_tgt
    gc.collect()
    return paths


def tile_split(domain_key):
    """Deterministic tile assignment -> dict with 'train'/'val'/'test' tile ids."""
    spec = DOMAIN_DATA[domain_key]
    gx, gy = spec["grid"]
    n_tiles = gx * gy
    n_train, n_val = spec["tiles"]
    rng = random.Random(f"{SPLIT_SEED}-{domain_key}")
    ids = list(range(n_tiles))
    rng.shuffle(ids)
    return {"train": set(ids[:n_train]),
            "val": set(ids[n_train:n_train + n_val]),
            "test": set(ids[n_train + n_val:])}


def tile_of(i, j, H, W, grid):
    gx, gy = grid
    th, tw = H / gx, W / gy
    return int(min(i // th, gx - 1)) * gy + int(min(j // tw, gy - 1))


def tile_mask(domain_key, H, W, split):
    """Boolean [H, W] mask of pixels whose tile belongs to `split`."""
    spec = DOMAIN_DATA[domain_key]
    gx, gy = spec["grid"]
    wanted = tile_split(domain_key)[split]
    rows = (np.minimum((np.arange(H) * gx // H), gx - 1))[:, None]
    cols = (np.minimum((np.arange(W) * gy // W), gy - 1))[None, :]
    return np.isin(rows * gy + cols, list(wanted))


class ChangeDetectionDataset(Dataset):
    """Patch dataset over one domain and one split, benchmark-faithful."""

    def __init__(self, domain_key, split, patch_size=64, stride=8,
                 min_def_fraction=MIN_DEF_FRACTION, augment=None,
                 styled_x_paths=None):
        self.domain_key = domain_key
        self.split = split
        self.ps = int(patch_size)
        self.augment = augment if augment is not None else (split == "train")
        # Scene-level FDA: styled variants of X (same shape); transfer_scale
        # is the probability of drawing a styled variant (layer B anneals it).
        self.styled_x_paths = styled_x_paths or []
        self.transfer_scale = 1.0

        self.x_path, self.y_path, self.v_path = build_domain_arrays(domain_key)
        y = np.load(self.y_path, mmap_mode='r')
        H, W = y.shape
        self.H, self.W = H, W
        in_split = tile_mask(domain_key, H, W, split)
        valid = np.array(np.load(self.v_path, mmap_mode='r'), dtype=bool)

        y_arr = np.array(y)
        keep_min = float(min_def_fraction)
        coords = []
        ps = self.ps
        for i in range(0, H - ps + 1, stride):
            for j in range(0, W - ps + 1, stride):
                if not in_split[i:i + ps, j:j + ps].all():
                    continue
                v = valid[i:i + ps, j:j + ps]
                if not v.any():
                    continue
                if split == "train" or split == "val":
                    d = (y_arr[i:i + ps, j:j + ps][v] == 1).mean() if v.any() else 0.0
                    if d < keep_min:
                        continue
                coords.append((i, j))
        self.patch_coords = coords
        del y_arr, valid
        gc.collect()
        log_change(f"[{domain_key}/{split}] {len(coords)} patches "
                   f"(ps={ps}, stride={stride}, min_def={keep_min if split != 'test' else 0})", "DATA")

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        i, j = self.patch_coords[idx]
        ps = self.ps
        use_styled = (self.styled_x_paths
                      and random.random() < self.transfer_scale)
        xp = random.choice(self.styled_x_paths) if use_styled else self.x_path
        X = np.array(np.load(xp, mmap_mode='r')[:, i:i + ps, j:j + ps])
        y = np.array(np.load(self.y_path, mmap_mode='r')[i:i + ps, j:j + ps])
        v = np.array(np.load(self.v_path, mmap_mode='r')[i:i + ps, j:j + ps])
        y = y.copy()
        y[v == 0] = -1  # ignore index for loss and metrics

        if self.augment:
            k = random.randint(0, 3)
            if k:
                X = np.rot90(X, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()
            if random.random() < 0.5:
                X = X[:, ::-1, :].copy(); y = y[::-1, :].copy()
            if random.random() < 0.5:
                X = X[:, :, ::-1].copy(); y = y[:, ::-1].copy()

        return torch.from_numpy(X).float(), torch.from_numpy(y).long()
