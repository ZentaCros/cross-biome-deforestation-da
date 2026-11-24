"""Number audit for main.tex.

Results and data numbers must come from macros (numbers.tex, generated from logs).
Every number typed directly into the text must be a configuration constant listed
here with the place in the code (or source) it was checked against. Exit code 1 on
any unlisted number or any undefined macro.
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED = {
    # value: where it was checked
    "0": "class index 0 / b = 0 at N = 64 / reference values 0 and 1 / e_k >= 0",
    "50": "method name 'mix50' (transfer_scale 0.5)",
    "30": "Landsat-8 OLI 30 m pixel size",
    "5.5": "model name Claude Opus 5.5 (AI-use statement)",
    "6": "6 x 6 km detail windows = 200 px x 30 m (fig_study_area_code.WIN)",
    "2550": "benchmark RO size, de Moura 2025 Table 2", "5120": "benchmark RO size, de Moura 2025 Table 2",
    "1098": "benchmark PA size, de Moura 2025 Table 2", "2600": "benchmark PA size, de Moura 2025 Table 2",
    "1700": "benchmark MA size, de Moura 2025 Table 2", "1440": "benchmark MA size, de Moura 2025 Table 2",
    "15": "15 tiles for PA and MA (dataset_cd grid 5x3; de Moura 2025 Table 2)",
    "54": "DeepLabv3+ has more than 54M parameters (de Moura 2025 Fig. 4 caption)",
    "1": "epoch 1 / Dirichlet +1 / KL target Dir(1) / min(1, t/10) (loss.py, train_cd.py)",
    "2": "2-pixel buffer (dataset_cd.BUFFER_PX); two classes; 2 dates; 2b x 2b window",
    "3": "three seeds / three sites / 3x3 conv / three styled versions (n_styled=3)",
    "4": "4 train tiles PA/MA (dataset_cd tiles); 4x expansion (FallbackSS2D)",
    "5": "min_epochs=5 (train_cd); 5 val tiles RO; b=5 at N=512",
    "6.25": "PRODES Amazon MMU (ha) -> 69 px",
    "7": "seven Landsat-8 bands; 7x7 depthwise conv (FallbackSS2D)",
    "8": "patch stride 8 (train_stride)",
    "9": "9 test tiles PA/MA; 9 distinct source models",
    "10": "patience 10; lambda_t = min(1, t/10); 10x10 grid RO",
    "11": "MMU 11 px for MA (MMU_PIXELS)",
    "14": "14-band input",
    "16": "stride 16 = 64*(1-0.75) (stitched_scene_probs); 16x16 bottleneck",
    "18": "MA first date 18 August 2017 / RO first date 18 July 2016 (file names)",
    "20": "20 train tiles RO; 20 tiles per scene/date (spectral_analysis); 20 July 2017 (PA file name)",
    "21": "RO 21 July 2017, MA 21 August 2018 (file names)",
    "32": "batch 32; central 32x32 crop; 32x32 resolution; central 32x32 bins in Fig.",
    "40": "40 tiles = 20 per date x 2 dates (spectral_analysis)",
    "42": "seed 42 / SPLIT_SEED",
    "62": "PA path/row 225/62",
    "63": "MA path/row 220/63",
    "64": "patch size 64",
    "67": "RO path/row 232/67",
    "69": "MMU 69 px for PA/RO",
    "75": "75 test tiles RO; 75% overlap",
    "90": "90-degree rotations (ChangeDetectionDataset)",
    "100": "max 100 epochs (train_cd --epochs)",
    "123": "seed 123", "456": "seed 456",
    "128": "128 channels (base_dim)",
    "200": "200 BN-TTA steps (--bn_steps)",
    "220": "MA path 220", "225": "PA path 225", "232": "RO path 232",
    "256": "256 channels (base_dim*2)",
    "512": "512 channels / 512 px FDA tiles",
    "2016": "dates", "2017": "dates", "2018": "dates",
    "4050": "GPU model name",
    "0.01": "beta = 0.01 (fda_beta); 1% no-data limit (MAX_NODATA_FRACTION)",
    "0.64": "64 * 0.01",
    "0.4": "class weight no-change (class_weight_bg)",
    "2.0": "class weight deforestation (class_weight_def)",
    "0.5": "threshold 0.5 (masked_metrics); s = 0.5 (transfer_scale)",
    "1.0": "s = 1.0 (transfer_scale)",
    "0.75": "overlap",
    "10^{-4}": "lr 1e-4",
}

tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(os.path.join(HERE, "numbers.tex")).read()))
defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))

bad = []
body_start = tex.index(r"\begin{document}")
pre_lines = tex[:body_start].count("\n")   # preamble = layout settings only, no reported numbers
for ln, line in enumerate(tex.splitlines(), 1):
    if ln <= pre_lines:
        continue
    line = re.sub(r"(?<!\\)%.*", "", line)                       # comments
    line = re.sub(r"\\(input|includegraphics|ref|label|cite[a-z]*|usepackage|documentclass|"
                  r"bibliography(style)?|url|href)(\[[^\]]*\])*\{[^}]*\}", "", line)
    line = re.sub(r"\[[0-9.]+(pt|cm|em)\]", "", line)
    line = re.sub(r"width=[0-9.]*\\(column|text)width", "", line)
    line = re.sub(r"margin=[0-9.]+cm|[0-9]+pt|blue![0-9]+|green![0-9]+|itemsep=[0-9]+pt|topsep=[0-9]+pt", "", line)
    line = re.sub(r"\\textsuperscript\{1\}|F\$_1\$|\\texttt\{[^}]*@[^}]*\}", "", line)
    for m in re.findall(r"\\([A-Za-z]+)", line):
        pass
    for num in re.findall(r"10\^\{-4\}|\d+(?:\.\d+)?", line):
        if num not in ALLOWED:
            bad.append((ln, num, line.strip()[:110]))

used = set(re.findall(r"\\([A-Za-z]+)", tex))
macro_like = {u for u in used if re.match(r"^(SO|TTA|MIX|ONE|InDom|Spec|Best|Stop|FDA|Params|NoData|Val|H|W|Def|Train|Test)", u)}
undefined = sorted(u for u in macro_like - defined if u not in {"Huge", "TODO"})

for ln, num, ctx in bad:
    print(f"UNLISTED number {num!r} at line {ln}: {ctx}")
for u in undefined:
    print(f"UNDEFINED macro \\{u}")
print(f"\n{len(bad)} unlisted numbers, {len(undefined)} undefined result macros")
sys.exit(1 if bad or undefined else 0)
