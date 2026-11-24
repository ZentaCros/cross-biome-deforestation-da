"""Study-area figure for Paper A, drawn entirely in code from the study data.

No generative AI. Sources:
  State and country outlines: Natural Earth 1:10m admin-1 (public domain), read from Paper 2's local copy.
  Site extents: Noa et al. (2021), Table 1 (latitude/longitude ranges of the three scenes).
  Scenes: this study's Landsat-8 OLI/TIRS arrays (true colour = bands Red, Green, Blue).
  Labels: PRODES-derived reference and past-deforestation rasters as preprocessed by dataset_cd.py.
  Tiles: the fixed tile split used in every experiment (dataset_cd.tile_split).
Output: paper/figures/fig_study_area_code.{png,pdf}, 180 mm wide, 600 dpi.
"""
import os
import sys
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
sys.path.insert(0, BASE)
from dataset_cd import DOMAIN_DATA, _load_band_stack, _clean_ref, build_domain_arrays, tile_split

NE = os.path.join(BASE, "data_aux", "ne_10m_admin1_south_america.geojson")  # Natural Earth 1:10m admin-1 subset
OUT = os.path.join(BASE, "paper", "figures", "fig_study_area_code")
MM = 1 / 25.4
W_MM, H_MM = 180.0, 166.0

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.5})
INK, MUTED = "#1a1a1a", "#5f5f5f"
NEW_DEF = (0.847, 0.106, 0.376)      # #d81b60 new deforestation (between the two dates)
PAST_DEF = (0.56, 0.56, 0.56)        # past deforestation (before date 1)
NO_CHANGE = (0.91, 0.94, 0.90)       # valid, no new deforestation
VOID = (0.20, 0.20, 0.20)            # excluded: buffer, sub-MMU polygons, no-data
WIN_COL = "#ffd400"                  # detail-window frames
SITE_FILL = {"PA": "#cfe3c4", "RO": "#cfe3c4", "MA": "#f3e3b5"}

# Scene extents from Noa et al. (2021), Table 1 (degrees, minutes, seconds; S and W negative)
def dms(d, m, s):
    return -(d + m / 60 + s / 3600)
EXTENT = {  # lon_min, lon_max, lat_min, lat_max
    "PA": (dms(51, 16, 12), dms(50, 34, 4), dms(3, 26, 16), dms(3, 8, 21)),
    "RO": (dms(64, 20, 51), dms(62, 56, 41), dms(10, 18, 35), dms(9, 36, 51)),
    "MA": (dms(44, 1, 23), dms(43, 37, 55), dms(5, 12, 48), dms(4, 44, 52)),
}
STATE = {"PA": "Pará", "RO": "Rondônia", "MA": "Maranhão"}
DATES = {"PA": ("2 Aug 2016", "20 Jul 2017"), "RO": ("18 Jul 2016", "21 Jul 2017"),
         "MA": ("18 Aug 2017", "21 Aug 2018")}
LETTER = {"RO": "b", "PA": "c", "MA": "d"}
WIN = 200  # detail window size in pixels (6 km at 30 m)


def ax_mm(fig, x, y, w, h):
    """Axes placed in millimetres from the top-left corner of the figure."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def stretch(img, valid, gamma=0.85):
    """Per-band 2-98 % linear stretch over valid pixels, then gamma."""
    out = np.zeros(img.shape[1:] + (3,), np.float32)
    # Stored bands are Landsat-8 OLI B1-B7 (coastal, blue, green, red, NIR, SWIR1, SWIR2);
    # the base code's NDVI uses index 4 as NIR and 3 as red. True colour = indices 3, 2, 1.
    for k, b in enumerate((3, 2, 1)):                 # Red, Green, Blue
        lo, hi = np.percentile(img[b][valid], [2, 98])
        out[..., k] = np.clip((img[b] - lo) / (hi - lo), 0, 1) ** gamma
    out[~valid] = 1.0
    return out


def tile_rects(dk, H, W):
    gx, gy = DOMAIN_DATA[dk]["grid"]
    rows = np.minimum(np.arange(H) * gx // H, gx - 1)
    cols = np.minimum(np.arange(W) * gy // W, gy - 1)
    r0 = [int(np.argmax(rows == r)) for r in range(gx)] + [H]
    c0 = [int(np.argmax(cols == c)) for c in range(gy)] + [W]
    sp = tile_split(dk)
    rects = []
    for r in range(gx):
        for c in range(gy):
            t = r * gy + c
            kind = "train" if t in sp["train"] else "val" if t in sp["val"] else "test"
            rects.append((c0[c], r0[r], c0[c + 1] - c0[c], r0[r + 1] - r0[r], kind))
    return rects


def load_site(dk):
    t1, nd1 = _load_band_stack(os.path.join(BASE, DOMAIN_DATA[dk]["img_t1"]), return_nodata=True)
    t2, nd2 = _load_band_stack(os.path.join(BASE, DOMAIN_DATA[dk]["img_t2"]), return_nodata=True)
    _, yp, vp = build_domain_arrays(dk)
    y = np.load(yp).astype(bool)
    v = np.load(vp).astype(bool)
    H, W = y.shape
    past, _ = _clean_ref(os.path.join(BASE, DOMAIN_DATA[dk]["past_ref"]))
    past = past[:H, :W].astype(bool)
    ok = ~(nd1[:H, :W] | nd2[:H, :W])
    rgb1 = stretch(t1[:, :H, :W], ok)
    rgb2 = stretch(t2[:, :H, :W], ok)
    ref = np.empty((H, W, 3), np.float32)
    ref[:] = VOID
    ref[v & ~y] = NO_CHANGE
    ref[past & ~(v & y)] = PAST_DEF
    ref[v & y] = NEW_DEF
    # Detail window: most new deforestation inside test tiles only.
    test = np.zeros((H, W), bool)
    for x0, y0, w, h, kind in tile_rects(dk, H, W):
        if kind == "test":
            test[y0:y0 + h, x0:x0 + w] = True
    S = np.pad(np.cumsum(np.cumsum((v & y).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    T = np.pad(np.cumsum(np.cumsum((~test).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    box = lambda A, i, j: A[i + WIN, j + WIN] - A[i, j + WIN] - A[i + WIN, j] + A[i, j]
    best, pos = -1, (0, 0)
    for i in range(0, H - WIN, 25):
        for j in range(0, W - WIN, 25):
            if box(T, i, j) == 0:
                s = box(S, i, j)
                if s > best:
                    best, pos = s, (i, j)
    return dict(rgb1=rgb1, rgb2=rgb2, ref=ref, H=H, W=W, win=pos, rects=tile_rects(dk, H, W))


def north_arrow(ax, x=0.955, y=0.80, color="white", boxed=True):
    if boxed:
        ax.add_patch(Rectangle((x - 0.028, y - 0.075), 0.056, 0.19, transform=ax.transAxes, fc="white",
                               ec="none", alpha=0.8, zorder=6))
        color = "black"
    ax.annotate("", xy=(x, y + 0.1), xycoords="axes fraction", xytext=(x, y), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=color, lw=1.0), zorder=7)
    ax.text(x, y - 0.012, "N", transform=ax.transAxes, ha="center", va="top", fontsize=7, fontweight="bold",
            color=color, zorder=7)


def scene_panel(fig, site, dk, x, y, w, h):
    ax = ax_mm(fig, x, y, w, h)
    ax.imshow(site["rgb2"], interpolation="lanczos")
    over = np.zeros((site["H"], site["W"], 4), np.float32)
    newdef = np.all(site["ref"] == np.array(NEW_DEF, np.float32), -1)
    over[newdef] = (*NEW_DEF, 0.95)
    ax.imshow(over, interpolation="nearest")
    for x0, y0, tw, th, kind in site["rects"]:
        if kind == "train":
            ax.add_patch(Rectangle((x0, y0), tw, th, fc="none", ec="white", lw=0.9, zorder=3))
        elif kind == "val":
            ax.add_patch(Rectangle((x0, y0), tw, th, fc="none", ec="#7fd3ff", lw=0.9, ls=(0, (3, 1.5)), zorder=3))
    i, j = site["win"]
    ax.add_patch(Rectangle((j, i), WIN, WIN, fc="none", ec=WIN_COL, lw=1.1, zorder=4))
    ax.text(j + WIN + 25, i + WIN / 2, f"({LETTER[dk]}1)", color=WIN_COL, fontsize=7, fontweight="bold",
            va="center", ha="left", zorder=5)
    ax.set_xlim(0, site["W"]); ax.set_ylim(site["H"], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_artist(ScaleBar(30, "m", location="lower left", length_fraction=0.22, box_alpha=0.8, pad=0.4,
                           border_pad=0.5, font_properties={"size": 6.5}, width_fraction=0.012))
    north_arrow(ax)
    d1, d2 = DATES[dk]
    fig.text(x / W_MM, 1 - (y - 0.8) / H_MM, f"({LETTER[dk]}) {STATE[dk]} ({dk}), {d2}",
             fontsize=7, fontweight="bold", color=INK, ha="left", va="bottom")
    return ax


def locator(fig, x, y, w, h):
    ax = ax_mm(fig, x, y, w, h)
    gdf = gpd.read_file(NE, columns=["admin", "name", "geometry"])
    sa = gdf[gdf["admin"].isin(["Brazil", "Peru", "Bolivia", "Colombia", "Venezuela", "Guyana", "Suriname",
                                "France", "Paraguay", "Argentina", "Uruguay", "Ecuador", "Chile"])]
    sa = sa.cx[-82:-30, -36:14]
    countries = sa.dissolve(by="admin")
    countries.plot(ax=ax, facecolor="#eeeeea", edgecolor="#9a9a9a", linewidth=0.35, zorder=1)
    br = gdf[gdf["admin"] == "Brazil"]
    br.plot(ax=ax, facecolor="#fbfbf8", edgecolor="#b5b5b0", linewidth=0.25, zorder=2)
    for dk in ("PA", "RO", "MA"):
        br[br["name"] == STATE[dk]].plot(ax=ax, facecolor=SITE_FILL[dk], edgecolor="#6b6b66", linewidth=0.45,
                                          zorder=3)
    gdf[gdf["admin"] == "Brazil"].dissolve().boundary.plot(ax=ax, color="#3d3d3d", linewidth=0.6, zorder=4)
    lab = {"PA": (-55.2, 1.6), "RO": (-69.5, -14.6), "MA": (-41.0, 0.4)}
    for dk, (l0, l1, b0, b1) in EXTENT.items():
        cx, cy = (l0 + l1) / 2, (b0 + b1) / 2
        ax.add_patch(Rectangle((l0 - 0.35, b0 - 0.35), (l1 - l0) + 0.7, (b1 - b0) + 0.7, fc=NEW_DEF, ec="black",
                               lw=0.5, zorder=6))
        tx, ty = lab[dk]
        ax.add_line(Line2D([cx, tx], [cy, ty], color="black", lw=0.5, zorder=5))
        ax.text(tx, ty, f"{dk}", fontsize=7.5, fontweight="bold", ha="center",
                va="bottom" if ty > cy else "top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(-75, -33); ax.set_ylim(-34.5, 6.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(-14)))
    ax.set_xticks([-70, -60, -50, -40]); ax.set_yticks([-30, -20, -10, 0])
    ax.set_xticklabels([f"{abs(t)}°W" for t in [-70, -60, -50, -40]], fontsize=6)
    ax.set_yticklabels(["30°S", "20°S", "10°S", "0°"], fontsize=6)
    ax.tick_params(length=2, width=0.4, pad=1)
    ax.grid(True, color="#c8c8c8", lw=0.3, ls=(0, (2, 2)), zorder=0)
    ax.set_facecolor("#e6eef4")
    # Scale bar: 500 km at 14 deg S
    km_per_deg = 111.32 * np.cos(np.deg2rad(14))
    x0b, y0b = -73.5, -32.5
    L = 500 / km_per_deg
    ax.add_patch(Rectangle((x0b, y0b), L / 2, 0.55, fc="black", ec="black", lw=0.3, zorder=8))
    ax.add_patch(Rectangle((x0b + L / 2, y0b), L / 2, 0.55, fc="white", ec="black", lw=0.3, zorder=8))
    ax.text(x0b, y0b + 0.9, "0", fontsize=5.8, ha="center", va="bottom", zorder=8)
    ax.text(x0b + L, y0b + 0.9, "500 km", fontsize=5.8, ha="center", va="bottom", zorder=8)
    ax.set_xlabel(""); ax.set_ylabel("")
    north_arrow(ax, x=0.92, y=0.83, color="black", boxed=False)
    fig.text(x / W_MM, 1 - (y - 0.8) / H_MM, "(a) Location of the three sites in Brazil",
             fontsize=7, fontweight="bold", color=INK, ha="left", va="bottom")
    return ax


def detail_row(fig, sites, y, s, gap_in=1.0, gap_grp=4.0):
    x = 2.0
    for dk in ("RO", "PA", "MA"):
        site = sites[dk]
        i, j = site["win"]
        d1, d2 = DATES[dk]
        crops = [(site["rgb1"], f"Date 1\n{d1}", "lanczos"), (site["rgb2"], f"Date 2\n{d2}", "lanczos"),
                 (site["ref"], "Reference\n", "nearest")]
        for k, (arr, lab, interp) in enumerate(crops):
            ax = ax_mm(fig, x + k * (s + gap_in), y, s, s)
            ax.imshow(arr[i:i + WIN, j:j + WIN], interpolation=interp)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(WIN_COL); sp.set_linewidth(1.0)
            fig.text((x + k * (s + gap_in) + s / 2) / W_MM, 1 - (y - 0.6) / H_MM, lab, fontsize=5.9, ha="center",
                     va="bottom", color=INK)
        gw = 3 * s + 2 * gap_in
        fig.text((x + gw / 2) / W_MM, 1 - (y + s + 0.8) / H_MM,
                 f"({LETTER[dk]}1) {dk}, 6 km × 6 km window in a test tile", fontsize=6.3, fontweight="bold",
                 ha="center", va="top", color=INK)
        x += gw + gap_grp


def main():
    sites = {dk: load_site(dk) for dk in ("RO", "PA", "MA")}
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    top = 4.0
    # Row 1: locator (a) + RO (b), same height
    h1 = 116.0 * sites["RO"]["H"] / sites["RO"]["W"]
    locator(fig, 6.5, top, 52.0, h1)
    scene_panel(fig, sites["RO"], "RO", 62.0, top, 116.0, h1)
    # Row 2: PA (c) + MA (d), same height, filling the width
    y2 = top + h1 + 5.0
    rPA = sites["PA"]["W"] / sites["PA"]["H"]
    rMA = sites["MA"]["W"] / sites["MA"]["H"]
    h2 = (176.0 - 4.0) / (rPA + rMA)
    scene_panel(fig, sites["PA"], "PA", 2.0, y2, h2 * rPA, h2)
    scene_panel(fig, sites["MA"], "MA", 2.0 + h2 * rPA + 4.0, y2, h2 * rMA, h2)
    # Row 3: detail windows
    y3 = y2 + h2 + 7.5
    s = (176.0 - 6 * 1.0 - 2 * 4.0) / 9
    detail_row(fig, sites, y3, s)
    # Legend
    yl = y3 + s + 5.2
    handles = [Patch(fc=NEW_DEF, ec="none", label="New deforestation (between the dates)"),
               Patch(fc=PAST_DEF, ec="none", label="Deforested before date 1"),
               Patch(fc=NO_CHANGE, ec="#9a9a9a", lw=0.4, label="No new deforestation"),
               Patch(fc=VOID, ec="none", label="Excluded (buffer, < MMU, no data)"),
               Line2D([], [], color="black", lw=0.9, label="Training tiles (white on maps)"),
               Line2D([], [], color="#2a8fc4", lw=0.9, ls=(0, (3, 1.5)), label="Validation tiles"),
               Patch(fc="none", ec="#c9a800", lw=1.0, label="Detail window")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - yl / H_MM), ncol=4, fontsize=6.2,
               frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    fig.savefig(OUT + ".png", dpi=600, facecolor="white")
    fig.savefig(OUT + ".pdf", facecolor="white")
    print("saved", OUT, "| windows:", {k: v["win"] for k, v in sites.items()}, "| fig height used to", round(yl + 8, 1), "mm")


if __name__ == "__main__":
    main()
