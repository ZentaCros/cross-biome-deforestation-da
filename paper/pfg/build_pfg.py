"""Build the PFG (Springer Nature, sn-jnl, author-year) manuscript from the master paper/main.tex.

The body (Introduction to Conclusion), floats, numbers and bibliography come unchanged from the master,
so the arXiv and PFG versions cannot drift apart. Only the front matter (abstract <= 250 words, 4-6
keywords, Springer author block) and the back matter (Statements and Declarations) are PFG-specific.
Run:  python build_pfg.py   then compile pfg/main.tex (pdflatex, bibtex, pdflatex x2).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "main.tex").read_text(encoding="utf-8")


def cut(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


body = cut(SRC, r"\section{Introduction}", r"\section*{Acknowledgments}").rstrip()
body = body.replace(r"\input{tab_", r"\input{../tab_")
# sn-jnl wraps tables in threeparttable: a tabular may not sit inside \resizebox, and [H] is not supported.
for a, b in [(r"\begin{table}[H]", r"\begin{table}[htbp]"),
             (r"\small\input{../tab_data}", r"\input{../tab_data}"),
             (r"\resizebox{\textwidth}{!}{\input{../tab_results}}",
              r"\scriptsize\setlength{\tabcolsep}{2.5pt}\input{../tab_results}"),
             (r"\resizebox{\textwidth}{!}{\input{../tab_results_ap}}",
              r"\scriptsize\setlength{\tabcolsep}{2.5pt}\input{../tab_results_ap}")]:
    assert body.count(a) == 1, a
    body = body.replace(a, b)
# The two results tables (~600 pt) do not fit the 372 pt text width: set them landscape.
def sideways(m):
    blk = m.group(0)
    if "tab_results" not in blk:
        return blk
    blk = blk.replace(r"\begin{table}[htbp]", r"\begin{sidewaystable}").replace(r"\end{table}", r"\end{sidewaystable}")
    return blk.replace(r"\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\footnotesize\setlength{\tabcolsep}{4pt}")
body = re.sub(r"\\begin\{table\}\[htbp\].*?\\end\{table\}", sideways, body, flags=re.S)
assert body.count("sidewaystable}") == 4
macros = cut(SRC, r"\newcommand{\pending}", r"\title{")

ABSTRACT = r"""Deep learning models for deforestation change detection lose accuracy when applied to a region other than the one they were trained on. We measure how much two inexpensive domain adaptation methods help on three Landsat-8 sites in Brazil: Par\'a (PA) and Rond\^onia (RO) in the Amazon, and Maranh\~ao (MA) in the Amazon--Cerrado transition. The methods are test-time re-estimation of batch-normalization statistics (BN-TTA) and Fourier domain adaptation (FDA). We evaluate all six source--target pairs with three training seeds each and report the F1 score at a fixed threshold and the threshold-free average precision (AP). Without adaptation, target F1 ranges from \SOMin{} (\pr{PA}{RO}) to \SOMax{} (\pr{MA}{PA}), and the spread across seeds of one pair (\SOWidestLo{}--\SOWidestHi{} for \SOWidestPair{}) is as large as most method effects. BN-TTA raises AP in \TTAAPAllW{} of \TTAAPAllN{} paired runs but F1 in only \TTAAllW{} of \TTAAllN{}; on the pairs trained on MA it raises recall, lowers precision, and F1 falls. FDA with the common setting {\unboldmath$\beta = 0.01$} changes nothing on {\unboldmath$64\times64$}-pixel patches; applied to scene tiles, it raises F1 in \MIXSrcROW{} of \MIXSrcRON{} paired runs when training on RO and lowers it in \MIXSrcMAL{} of \MIXSrcMAN{} when training on MA, and it does not improve AP overall. Spectral distances between the sites do not explain these differences. No configuration is best on every pair, so results on this benchmark should be reported over several seeds and with a threshold-free metric."""

KEYWORDS = "Deforestation detection, Domain adaptation, Test-time adaptation, Fourier domain adaptation, Landsat-8, Amazon"

BACK = r"""
\backmatter

\bmhead{Acknowledgements}
This work builds on the author's Final Year Project at GIFT University, Gujranwala, Pakistan.

\section*{Statements and Declarations}

\bmhead{Funding} No funding was received for conducting this study.

\bmhead{Competing interests} The author has no relevant financial or non-financial interests to disclose.

\bmhead{Ethics approval and consent to participate} Not applicable.

\bmhead{Consent for publication} Not applicable.

\bmhead{Data availability} The Landsat-8 images and PRODES-derived reference maps of the three sites are the public benchmark data of \citet{noa2021adversarial} and \citet{demoura2025enhancing}, available from the links given by \citet{demoura2025enhancing}. The per-run results, logs and the tile split used here are available in the code repository below.

\bmhead{Code availability} The code for preprocessing, training, evaluation, the number-generation and audit scripts, and all figures is available at \url{https://github.com/ZentaCros/cross-biome-deforestation-da}.

\bmhead{Use of AI tools} The use of AI tools is described in Section~\ref{sec:impl}.

\bmhead{Author contribution} Muhammad Hamza Azeem: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing -- review and editing, Visualization.
"""

out = rf"""%% PFG submission (Springer Nature LaTeX template, sn-basic author-year). Generated by build_pfg.py from ../main.tex.
\documentclass[pdflatex,sn-basic]{{sn-jnl}}

\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{enumitem}}
\usepackage{{float}}
\usepackage[section]{{placeins}}
\input{{glyphtounicode}}
\pdfgentounicode=1
\graphicspath{{{{../}}}}
\raggedbottom
\renewcommand{{\topfraction}}{{0.95}}\renewcommand{{\bottomfraction}}{{0.9}}\renewcommand{{\textfraction}}{{0.05}}\renewcommand{{\floatpagefraction}}{{0.85}}

\input{{../numbers}}
{macros}
\begin{{document}}

\title[Simple Domain Adaptation for Cross-Biome Deforestation Detection]{{How Much Do Simple Domain Adaptation Methods Help Cross-Biome Deforestation Detection? A Multi-Seed Evaluation on Landsat-8}}

\author*[1]{{\fnm{{Muhammad Hamza}} \sur{{Azeem}}}}\email{{221980023@gift.edu.pk}}

\affil*[1]{{\orgdiv{{Department of Computer Science, School of Engineering and Applied Sciences}}, \orgname{{GIFT University}}, \orgaddress{{\city{{Gujranwala}}, \postcode{{52250}}, \country{{Pakistan}}}}}}

\abstract{{{ABSTRACT}}}

\keywords{{{KEYWORDS}}}

\maketitle

{body}

{BACK}
\bibliography{{../references}}

\end{{document}}
"""
(HERE / "main.tex").write_text(out, encoding="utf-8")

# Abstract word count with macros expanded from numbers.tex
nums = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\n", (HERE.parent / "numbers.tex").read_text()))
txt = re.sub(r"\\pr\{(\w+)\}\{(\w+)\}", r"\1-\2", ABSTRACT)
txt = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: nums.get(m.group(1), m.group(0)), txt)
txt = re.sub(r"\$[^$]*\$", "X", txt)
n = len(re.findall(r"[A-Za-z0-9][\w.\-'\\]*", txt))
k = len(KEYWORDS.split(","))
assert 150 <= n <= 250, f"abstract has {n} words (PFG: 150-250)"
assert 4 <= k <= 6, k
print(f"wrote {HERE / 'main.tex'} | abstract {n} words | {k} keywords")
