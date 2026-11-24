"""300 dpi copies of the 600 dpi figure masters for the manuscript PDF (paper/figures/ms/).

Photographic figures become high-quality JPEG, line diagrams PNG. The 600 dpi masters stay in paper/figures
and are the files to upload as separate figure files. Run after regenerating any figure.
"""
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
JOBS = {"fig_study_area_code.png": "jpg", "fig_qualitative.png": "jpg", "fig_methodology.png": "jpg",
        "fig_architecture.png": "png"}
os.makedirs(os.path.join(FIG, "ms"), exist_ok=True)
for f, fmt in JOBS.items():
    im = Image.open(os.path.join(FIG, f)).convert("RGB")
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    out = os.path.join(FIG, "ms", f.replace(".png", "." + fmt))
    if fmt == "jpg":
        im.save(out, quality=93, subsampling=0, dpi=(300, 300))
    else:
        im.save(out, optimize=True, dpi=(300, 300))
    print(out, im.size, f"{os.path.getsize(out) / 1e6:.2f} MB")
