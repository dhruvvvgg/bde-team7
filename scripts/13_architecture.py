"""Part J: draw docs/architecture_preprocessing.png (300 dpi) with matplotlib.
The Mermaid source docs/architecture_preprocessing.mmd describes the same graph.
Shapes in labels are read from the committed outputs, not typed by hand.

Run: python scripts/13_architecture.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
FILL = {"src": "#e7f0fb", "proc": "#ffffff", "store": "#e5f5ee", "future": "#ffffff"}
EDGE = {"src": "#2a78d6", "proc": "#52514e", "store": "#1baf7a", "future": "#898781"}


def shape(p):
    d = pd.read_parquet(p)
    return f"{d.shape[0]:,} x {d.shape[1]}"


def main() -> None:
    S = {"main": shape(config.FINAL_DIR), "ext": shape(config.EXT_DIR / "extension_features.parquet"),
         "spl": shape(config.PROCESSED_DIR / "splits.parquet"), "ch": shape(config.CHIRPS_TABLE),
         "ch3": shape(chirps.CHIRPS_3X3_TABLE), "che": shape(config.CHIRPS_EXT_TABLE)}
    cw = pd.read_csv(config.CROSSWALK)
    n_wells = len(cw)
    rows = n_wells * 92

    boxes = {  # id: (x, y, w, h, kind, title, detail)
        "gwsrc": (0.4, 11.0, 5.4, 1.5, "src", "Figshare groundwater dataset (CGWB)",
                  "doi:10.6084/m9.figshare.29293877.v3 · CC BY 4.0\nrefined table: 2,759 wells x 92 rounds (wide)"),
        "chsrc": (8.2, 11.0, 5.4, 1.5, "src", "CHIRPS v3.0 (Climate Hazards Center)",
                  "monthly global rasters, 0.05° · HTTPS\n282 files (1999-07..2022-12) + 24 (2023-24)"),
        "ing": (0.4, 8.9, 5.4, 1.5, "proc", "1. Acquisition & ingestion",
                f"gw_io.load_wide · surrogate well_id (SHA-1)\nout: well_crosswalk.csv ({n_wells:,} x {cw.shape[1]})"),
        "chx": (8.2, 8.9, 5.4, 1.5, "proc", "2. Raster streaming & extraction",
                f"one file at a time → sample → delete\nout: chirps_monthly_by_well ({S['ch']})"),
        "resh": (0.4, 6.8, 5.4, 1.5, "proc", "3. Schema: wide → long",
                 f"reshape.to_long · 1 row per well x round\nout: {rows:,} x 16 (Mon-YY → date, season)"),
        "clean": (0.4, 4.7, 5.4, 1.5, "proc", "4. Cleaning & flagging",
                  "nothing removed · placeholders, duplicates, names\nflags: zero, extreme, deeper-than-well"),
        "rainw": (8.2, 6.8, 5.4, 1.5, "proc", "5. Rainfall integration (windows)",
                  "1/3/6/12-month, monsoon, Oct-Dec totals\nmonths ≤ M-1 only · LeakageError assertion"),
        "feat": (8.2, 4.7, 5.4, 1.5, "proc", "6. Time-aware feature engineering",
                 "past-only baselines, lags, trend, deviations\nperturb-the-future tests at 4 cutoffs"),
        "stress": (4.3, 2.6, 5.4, 1.5, "proc", "7. Stress classification & validation",
                   "expanding (primary) · rolling-10 · t-adjusted\nsame classify(); suspect_reading flags"),
        "main": (0.0, 0.3, 4.4, 1.5, "store", "Main feature table",
                 f"processed/gw_features_by_state/\nParquet by state_slug · {S['main']}"),
        "spl": (4.8, 0.3, 4.4, 1.5, "store", "Splits file",
                f"processed/splits.parquet · {S['spl']}\ncutoffs: config/splits.json"),
        "ext": (9.6, 0.3, 4.4, 1.5, "store", "Extension holdout 2023-24",
                f"processed/extension_2023_2024/\n{S['ext']} · kept separate"),
        "mod": (2.2, -2.2, 4.4, 1.2, "future", "Modeling stage (teammate)", "classifiers, evaluation"),
        "dash": (7.4, -2.2, 4.4, 1.2, "future", "Dashboard stage (teammate)", "maps, time series, alerts"),
    }
    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for k, (x, y, w, h, kind, title, detail) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=FILL[kind], ec=EDGE[kind], lw=2 if kind != "future" else 1.6,
                                    ls="--" if kind == "future" else "-"))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, weight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.22, detail, ha="center", va="center", fontsize=8.6, color=INK2,
                linespacing=1.35)

    def mid(k, side):
        x, y, w, h = boxes[k][:4]
        return {"top": (x + w / 2, y + h), "bottom": (x + w / 2, y), "left": (x, y + h / 2),
                "right": (x + w, y + h / 2)}[side]

    def arrow(a, sa, b, sb, dashed=False, label=None, rad=0.0):
        p = FancyArrowPatch(mid(a, sa), mid(b, sb), arrowstyle="-|>", mutation_scale=16, lw=1.6,
                            color=INK2 if not dashed else "#898781", ls="--" if dashed else "-",
                            connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4)
        ax.add_patch(p)
        if label:
            (x1, y1), (x2, y2) = mid(a, sa), mid(b, sb)
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=8, color=INK2, va="center")

    arrow("gwsrc", "bottom", "ing", "top", label="CSV (wide)")
    arrow("chsrc", "bottom", "chx", "top", label="GeoTIFF, streamed")
    arrow("ing", "bottom", "resh", "top")
    arrow("resh", "bottom", "clean", "top")
    arrow("chx", "bottom", "rainw", "top", label="monthly x well")
    arrow("clean", "right", "rainw", "left", rad=0.15)
    ax.text(6.0, 5.9, "clean long table", fontsize=8, color=INK2, rotation=38)
    arrow("rainw", "bottom", "feat", "top")
    arrow("feat", "bottom", "stress", "right", rad=-0.2)
    for k in ("main", "spl", "ext"):
        arrow("stress", "bottom", k, "top")
    arrow("main", "bottom", "mod", "top", dashed=True)
    arrow("spl", "bottom", "mod", "top", dashed=True)
    arrow("ext", "bottom", "mod", "top", dashed=True)
    arrow("main", "bottom", "dash", "top", dashed=True)
    arrow("mod", "right", "dash", "left", dashed=True)

    ax.text(0.4, 12.95, "Groundwater stream", fontsize=10, color="#2a78d6", weight="bold")
    ax.text(8.2, 12.95, "Rainfall stream", fontsize=10, color="#2a78d6", weight="bold")
    ax.text(7.0, -2.75, "dashed = later stages, owned by teammates (extend here)", ha="center", fontsize=9,
            color="#898781", style="italic")
    ax.text(7.0, 13.6, "Preprocessing architecture: Seasonal Groundwater Stress Assessment across India",
            ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(14.1, 6.0, f"Side resources\n3x3 rainfall check:\n{S['ch3']}\nCHIRPS 2023-24:\n{S['che']}",
            fontsize=8.3, color=INK2, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f3ef", ec="#c3c2b7"))
    ax.set_xlim(-0.3, 16.6)
    ax.set_ylim(-3.0, 14.0)
    ax.axis("off")
    out = config.REPO_ROOT / "docs" / "architecture_preprocessing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
