"""Fourier Domain Adaptation (Yang & Soatto 2020) as used in this study.

fda_source_to_target: swap the centred low-frequency square (half-width int(N*beta)) of the amplitude spectrum.
style_scene: apply it tile by tile (512 px) to a whole raw scene. Copied unchanged from the original project.
"""
import numpy as np


def fda_source_to_target(src_img, trg_img, beta=0.01):
    """
    Swaps low-frequency amplitude from trg_img into src_img.
    Expects arrays of shape [C, H, W]
    """
    fft_src = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg_img, axes=(-2, -1))

    fft_src_shift = np.fft.fftshift(fft_src, axes=(-2, -1))
    fft_trg_shift = np.fft.fftshift(fft_trg, axes=(-2, -1))

    amp_src, pha_src = np.abs(fft_src_shift), np.angle(fft_src_shift)
    amp_trg, _ = np.abs(fft_trg_shift), np.angle(fft_trg_shift)

    C, H, W = src_img.shape
    b_h, b_w = int(H * beta), int(W * beta)
    c_h, c_w = H // 2, W // 2

    amp_src_styled = np.copy(amp_src)

    if b_h > 0 and b_w > 0:
        amp_src_styled[:, c_h - b_h:c_h + b_h, c_w - b_w:c_w + b_w] = amp_trg[:, c_h - b_h:c_h + b_h, c_w - b_w:c_w + b_w]

    fft_src_styled_shift = amp_src_styled * np.exp(1j * pha_src)
    fft_src_styled = np.fft.ifftshift(fft_src_styled_shift, axes=(-2, -1))
    src_styled_img = np.fft.ifft2(fft_src_styled, axes=(-2, -1))

    return np.real(src_styled_img).astype(np.float32)



def style_scene(src_scene, trg_scene_mmaps, beta, tile=512, rng=None):
    """Scene-level FDA: style a full source scene tile-by-tile with random
    same-size crops from the target scenes.

    Patch-level FDA at 64 px is a measured no-op for beta <= 0.01
    (int(64*0.01) = 0 frequency bins), so transfer must happen at scene/tile
    granularity, as in Yang & Soatto 2020. At tile=512 and beta=0.01 the swap
    covers a 10x10 low-frequency block - a real but gentle transfer.

    src_scene: [C, H, W] float32 (preprocessed). trg_scene_mmaps: list of
    memory-mapped [C, H, W] target scenes. Tiles too small for the swap
    (int(size*beta) < 1) are left unchanged. Returns a new float32 array.
    """
    if rng is None:
        rng = np.random.default_rng()
    styled = src_scene.copy()
    C, H, W = src_scene.shape
    for i in range(0, H, tile):
        h = min(tile, H - i)
        for j in range(0, W, tile):
            w = min(tile, W - j)
            if int(h * beta) < 1 or int(w * beta) < 1:
                continue
            trg = trg_scene_mmaps[int(rng.integers(len(trg_scene_mmaps)))]
            tH, tW = trg.shape[1], trg.shape[2]
            if tH < h or tW < w:
                continue
            ti = int(rng.integers(tH - h + 1))
            tj = int(rng.integers(tW - w + 1))
            trg_crop = np.array(trg[:, ti:ti + h, tj:tj + w], dtype=np.float32)
            styled[:, i:i + h, j:j + w] = fda_source_to_target(
                styled[:, i:i + h, j:j + w], trg_crop, beta)
    return styled

