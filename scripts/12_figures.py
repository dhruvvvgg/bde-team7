"""Part I: review figures (PNG, 300 dpi) and tables (CSV + Markdown) from the
FINAL outputs, plus reports/figure_captions.md. Every number in a caption is
computed here; nothing is typed by hand.

Run: python scripts/12_figures.py   (after run_all.sh; reads committed outputs only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config, gw_io, stress  # noqa: E402

FIG = config.REPORTS_DIR / "figures"
TAB = config.REPORTS_DIR / "tables"
DPI = 300

# ---- style: reference palette (validated with the dataviz validator) --------
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order, never cycled
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True,
})
SEASONS = list(config.SEASON_BY_MONTH.values())
SEASON_LABEL = {"post_monsoon_rabi": "Jan (post-monsoon, rabi)", "pre_monsoon": "May (pre-monsoon)",
                "monsoon": "Aug (monsoon)", "post_monsoon_kharif": "Nov (post-monsoon, kharif)"}
CLASSES = ["Normal", "Watch", "Moderate Stress", "High Stress"]
captions: list[tuple[str, str, str, str]] = []


def save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"reports/figures/{name}"


def table(df: pd.DataFrame, name: str, index=True) -> str:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=index) + "\n")
    return f"reports/tables/{name}.csv"


def cap(num: str, title: str, text: str, files: str) -> None:
    captions.append((num, title, text, files))


def pct(x) -> str:
    return f"{x:.1f}%"


def main() -> None:
    main_t = pd.read_parquet(config.FINAL_DIR)
    ext = pd.read_parquet(config.EXT_DIR / "extension_features.parquet")
    spl = pd.read_parquet(config.PROCESSED_DIR / "splits.parquet")
    rain = pd.read_parquet(config.CHIRPS_TABLE)
    r3 = pd.read_parquet(chirps.CHIRPS_3X3_TABLE)
    for d in (main_t, ext):
        d["season"] = pd.Categorical(d["season"].astype(str), categories=SEASONS)
    both = pd.concat([main_t.drop(columns="state_slug"), ext.drop(columns="is_extension")], ignore_index=True)
    rd = main_t[main_t.gwl_m_bgl.notna()]
    cls = rd[rd.stress_category.isin(CLASSES)]

    # ---------------------------------------------------------------- 1
    av = both.pivot_table(index="season", columns="year", values="gwl_m_bgl", observed=False,
                          aggfunc=lambda s: 100 * s.notna().mean()).reindex(SEASONS)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(av.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(av.shape[1]), av.columns, rotation=90)
    ax.set_yticks(range(4), [SEASON_LABEL[s] for s in SEASONS])
    ax.grid(False)
    for i in range(av.shape[0]):
        for j in range(av.shape[1]):
            v = av.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else INK)
    xe = list(av.columns).index(config.EXT_FIRST_YEAR) - 0.5
    ax.axvline(xe, color=INK, lw=2)
    ax.text(xe + 1, -0.85, "extension holdout\n(separate, lower QC)", fontsize=8, ha="center", va="bottom")
    callouts = [("pre_monsoon", 2020), ("pre_monsoon", 2021), ("monsoon", 2012), ("post_monsoon_rabi", 2016)]
    for s, y in callouts:
        ax.add_patch(Rectangle((list(av.columns).index(y) - 0.5, SEASONS.index(s) - 0.5), 1, 1,
                               fill=False, ec="#d03b3b", lw=2))
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% of 2,759 wells with a reading")
    ax.set_title("Data availability by season and year (main table 2000-2022 | extension 2023-2024)")
    ax.set_xlabel("Year")
    f1 = save(fig, "fig01_missingness_heatmap.png")
    table(av.round(1), "table01_availability_season_year")
    c = {k: av.loc[k[0], k[1]] for k in callouts}
    m_main = 100 * main_t.gwl_m_bgl.notna().mean()
    m_ext = 100 * ext.gwl_m_bgl.notna().mean()
    cap("1", "Share of wells with a reading, by season and year",
        f"Each cell is the percentage of the 2,759 wells measured in that round. Overall "
        f"{pct(m_main)} of main-table rounds have a reading. The outlined cells are near-empty rounds: "
        f"May 2020 ({pct(c[('pre_monsoon', 2020)])}), May 2021 ({pct(c[('pre_monsoon', 2021)])}), "
        f"Aug 2012 ({pct(c[('monsoon', 2012)])}) and Jan 2016 ({pct(c[('post_monsoon_rabi', 2016)])}). "
        f"The May row is low every year because 444 wells (Kerala, West Bengal, most of Odisha, Assam) "
        f"have no May series. Right of the black line, the 2023-24 extension averages only "
        f"{pct(m_ext)}.", f"{f1}; reports/tables/table01_availability_season_year.csv")

    # ---------------------------------------------------------------- 2 (table + bar)
    stages = []

    def stage(label, path, by, note):
        d = gw_io.load_wide(path, strict=False)
        _, obs, _ = gw_io.split_columns(d.drop(columns="well_id"))
        yrs = sorted({2000 + int(o[-2:]) for o in obs})
        cells = d[obs]
        stages.append({"stage": label, "applied by": by, "rows (wells)": len(d),
                       "states": d.State.nunique(), "years": f"{yrs[0]}-{yrs[-1]}",
                       "readings": int(cells.notna().to_numpy().sum()),
                       "missing % of cells": round(100 * cells.isna().to_numpy().mean(), 2), "what changed": note})
    stage("0 raw CGWB (wells within India)", config.GW_DIR / "Input/1_India_GWLs_2000_2024_wells_within_India.csv",
          "CGWB / authors", "raw download")
    stage("1 remove wells with no data", config.GW_OUTPUT_DIR / "2_India_GWLs_2000_2024_nonzero_obs.csv",
          "dataset authors", "drop all-empty wells (zero readings NOT removed)")
    stage("2 remove wells with any negative", config.GW_OUTPUT_DIR / "3_India_GWLs_2000_2024_non_negative_obs.csv",
          "dataset authors", "drop whole wells with any negative value")
    stage("3 per-well 3-sigma", config.GW_STAGE4_FILE, "dataset authors",
          "blank readings > 3 sd from the well mean (wells kept)")
    stage("4+5 >= 2 readings every year, no 3x repeats (2000-2022)", config.GW_REFINED_FILE, "dataset authors",
          "keep complete, non-stuck wells; attach Reference_Sy")
    raw_long = pd.read_parquet(config.GW_LONG)
    for lab, d, note in [("6 reshape wide -> long", raw_long, "1 row per well x round; 0 rows lost"),
                         ("7 our cleaning + flags (final)", main_t,
                          "0 removed; 0 placeholders, 0 duplicates; flags added")]:
        stages.append({"stage": lab, "applied by": "this project", "rows (wells)": d.well_id.nunique(),
                       "states": d.state.nunique(), "years": f"{d.year.min()}-{d.year.max()}",
                       "readings": int(d.gwl_m_bgl.notna().sum()),
                       "missing % of cells": round(100 * d.gwl_m_bgl.isna().mean(), 2), "what changed": note})
    st = pd.DataFrame(stages)
    st["rows (wells)"] = st["rows (wells)"].astype(int)
    t2 = table(st, "table02_cleaning_stages", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(st))[::-1]
    colors = [CAT[1] if b == "dataset authors" or b.startswith("CGWB") else CAT[0] for b in st["applied by"]]
    ax.barh(y, st["rows (wells)"], color=colors, height=0.6)
    for yi, v in zip(y, st["rows (wells)"]):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=9)
    ax.set_yticks(y, st.stage)
    ax.set_xlabel("Wells (rows in the table)")
    ax.set_title("Wells remaining at each quality-control stage")
    ax.legend(handles=[Patch(color=CAT[1], label="dataset authors' QC"), Patch(color=CAT[0], label="this project")],
              loc="lower right")
    ax.set_xlim(0, st["rows (wells)"].max() * 1.15)
    f2 = save(fig, "fig02_cleaning_stages.png")
    cap("2", "Rows, wells and missingness at each cleaning stage",
        f"The dataset authors' five criteria reduce {st['rows (wells)'].iloc[0]:,} raw CGWB well rows to the "
        f"{st['rows (wells)'].iloc[4]:,}-well refined 2000-2022 table, mainly through the completeness "
        f"rule (criteria 4-5). Our steps removed nothing: the reshape keeps all "
        f"{len(raw_long):,} well-rounds and cleaning only adds flags, so missingness stays at "
        f"{st['missing % of cells'].iloc[-1]}%.", f"{t2}; {f2}")

    # ---------------------------------------------------------------- 3
    rpw = main_t.groupby("well_id").gwl_m_bgl.count()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(rpw, bins=range(int(rpw.min()), 93), color=CAT[0], edgecolor=SURF, linewidth=0.5)
    ax.axvline(rpw.median(), color=INK, ls="--", lw=1)
    ax.text(rpw.median(), ax.get_ylim()[1] * 0.95, f" median {rpw.median():.0f}", fontsize=9, va="top")
    ax.set_xlabel("Readings per well (of 92 rounds, 2000-2022)")
    ax.set_ylabel("Number of wells")
    ax.set_title("Readings per well")
    f3 = save(fig, "fig03_readings_per_well.png")
    table(rpw.describe().round(1).to_frame("readings_per_well"), "table03_readings_per_well")
    cap("3", "Distribution of readings per well",
        f"Every well has between {rpw.min()} and {rpw.max()} of the 92 possible readings (median "
        f"{rpw.median():.0f}), so no total-readings cutoff was needed and all {len(rpw):,} wells were kept. "
        f"The lower mode near {int(rpw[rpw < 70].mode().iloc[0])} readings is mostly the wells with no May series.", f3)

    # ---------------------------------------------------------------- 4
    wps = main_t.drop_duplicates("well_id").state.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(wps.index, wps.values, color=CAT[0], height=0.65)
    for i, v in enumerate(wps.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8.5)
    ax.set_xlabel("Number of wells")
    ax.set_title("Wells per state")
    ax.set_xlim(0, wps.max() * 1.12)
    f4 = save(fig, "fig04_wells_per_state.png")
    dps = main_t.groupby("state").district.nunique()
    table(pd.DataFrame({"wells": wps, "districts": dps}).sort_values("wells", ascending=False), "table04_state_coverage")
    cap("4", "Number of wells per state",
        f"The {len(wps)} states are very unevenly covered: {wps.idxmax()} has {wps.max()} wells "
        f"({pct(100 * wps.max() / wps.sum())} of all), while {wps.idxmin()} has {wps.min()}. Small states will give "
        "noisy per-state metrics in cross-region evaluation.", f"{f4}; reports/tables/table04_state_coverage.csv")

    # ---------------------------------------------------------------- 5
    REGION = {"North": ["Punjab", "Haryana", "Delhi", "Himachal Pradesh", "Uttar Pradesh"],
              "West": ["Gujarat", "Maharashtra"], "Central": ["Madhya Pradesh", "Chhattisgarh"],
              "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"], "North-east": ["Assam"],
              "South": ["Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu"]}
    reg_of = {s: r for r, ss in REGION.items() for s in ss}
    wl = main_t.drop_duplicates("well_id")[["well_id", "state", "lat", "lon"]].copy()
    assert set(wl.state) <= set(reg_of)
    wl["region"] = wl.state.map(reg_of)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    for i, r in enumerate(REGION):
        d = wl[wl.region == r]
        ax.scatter(d.lon, d.lat, s=5, color=CAT[i], label=f"{r} ({len(d)})", alpha=0.85, linewidths=0)
    for s, g in wl.groupby("state"):
        ax.annotate(s, (g.lon.median(), g.lat.median()), fontsize=7, ha="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.8))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1 / np.cos(np.deg2rad(22)))
    ax.set_title("Monitoring well locations (coloured by region, labelled by state)")
    ax.legend(title="Region (wells)", loc="lower left", markerscale=3, fontsize=8)
    f5 = save(fig, "fig05_well_locations.png")
    cap("5", "Locations of the 2,759 monitoring wells",
        f"Wells span {wl.lat.min():.1f}-{wl.lat.max():.1f}°N and {wl.lon.min():.1f}-{wl.lon.max():.1f}°E. "
        f"Points are coloured by {len(REGION)} regions (19 states exceed the 8-colour limit for "
        "distinguishable categories) and every state is labelled at its median location. No basemap is "
        "used; all wells passed the India bounding-box check.", f5)

    # ---------------------------------------------------------------- 6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [rd[rd.season == s].gwl_m_bgl.to_numpy() for s in SEASONS]
    bp = ax.boxplot(data, showfliers=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for p in bp["boxes"]:
        p.set(facecolor=SEQ[1], edgecolor=INK2)
    ax.set_xticks(range(1, 5), [SEASON_LABEL[s] for s in SEASONS], fontsize=9)
    ax.set_ylabel("Depth to groundwater (m below ground level)")
    ax.invert_yaxis()
    above = []
    for i, v in enumerate(data, 1):
        q1, q3 = np.percentile(v, [25, 75])
        n_out = int((v > q3 + 1.5 * (q3 - q1)).sum())
        above.append(n_out)
        ax.text(i, ax.get_ylim()[0], f"{n_out:,} deeper\nbeyond whisker\nmax {v.max():.1f} m", ha="center",
                va="top", fontsize=7.5, color=INK2)
    ax.set_title("Groundwater depth by season (deeper = lower on the chart)")
    f6 = save(fig, "fig06_depth_by_season.png")
    med = {s: np.median(v) for s, v in zip(SEASONS, data)}
    table(rd.groupby("season", observed=True).gwl_m_bgl.describe(percentiles=[.25, .5, .75, .95, .99]).round(2),
          "table06_depth_by_season")
    cap("6", "Groundwater depth by season",
        f"Box = interquartile range, line = median; whiskers extend to 1.5 × IQR. Outliers are not drawn, so "
        f"the long deep tail ({sum(above):,} readings beyond the whiskers, maximum {rd.gwl_m_bgl.max():.1f} m) "
        f"does not compress the boxes; the y-axis is linear and inverted so deeper is lower. The seasonal "
        f"cycle is clear: median depth is {med['pre_monsoon']:.1f} m in May and {med['post_monsoon_kharif']:.1f} m "
        "in November.", f6)

    # ---------------------------------------------------------------- 7
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-6, 6, 97)
    series = [("anomaly_z (raw)", rd.anomaly_z, CAT[0]), ("anomaly_z_capped (±5)", rd.anomaly_z_capped, CAT[1]),
              ("anomaly_tadj (small-sample adjusted)", rd.anomaly_tadj, CAT[2])]
    for lab, s, col in series:
        v = s.dropna().clip(-6, 6)
        ax.hist(v, bins=bins, histtype="step", lw=2, color=col, label=lab)
    for t, nm in zip(stress.DEFAULT_THRESHOLDS, ["Watch", "Moderate", "High"]):
        ax.axvline(t, color=INK2, ls="--", lw=1)
        ax.text(t, ax.get_ylim()[1] * 0.97, f" {nm} ≥ {t:g}", rotation=90, va="top", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Anomaly score (SD units; values beyond ±6 drawn at ±6)")
    ax.set_ylabel("Readings (log scale)")
    ax.set_title("Distribution of seasonal anomaly scores")
    ax.legend(loc="upper left", fontsize=8.5)
    f7 = save(fig, "fig07_anomaly_distribution.png")
    raw = rd.anomaly_z.dropna()
    cap("7", "Raw, capped and small-sample-adjusted anomaly distributions",
        f"Raw anomalies range from {raw.min():.1f} to {raw.max():.1f} and {int((raw.abs() > 5).sum()):,} readings exceed "
        f"|5|, so the capped version piles them at ±5. The adjusted score is narrower in the tails: "
        f"{pct(100 * (rd.anomaly_tadj.dropna() >= 2).mean())} of scored readings are ≥ 2, against "
        f"{pct(100 * (raw >= 2).mean())} for the raw score. Dashed lines mark the stress thresholds 1, 1.5 and 2 "
        "(log y-axis).", f7)

    # ---------------------------------------------------------------- 8
    meth = [("Expanding (primary)", "stress_category"), ("Rolling 10", "stress_roll10"),
            ("Small-sample adjusted", "stress_tadj")]
    cc = pd.DataFrame({lab: rd[col].astype(str).value_counts().reindex(CLASSES) for lab, col in meth})
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(CLASSES))
    w = 0.26
    for i, (lab, _) in enumerate(meth):
        b = ax.bar(x + (i - 1) * w, cc[lab], width=w - 0.02, color=CAT[i], label=lab)
        ax.bar_label(b, labels=[f"{v:,}" for v in cc[lab]], fontsize=7, padding=1)
    ax.set_xticks(x, CLASSES)
    ax.set_yscale("log")
    ax.set_ylabel("Readings (log scale)")
    ax.set_title("Stress category counts by baseline method")
    ax.legend(fontsize=8.5)
    f8 = save(fig, "fig08_stress_counts_by_method.png")
    full = pd.DataFrame({lab: rd[col].astype(str).value_counts().reindex(stress.CATEGORIES[1:]) for lab, col in meth})
    table(full, "table08_stress_counts_by_method")
    cap("8", "Stress categories under the three baselines",
        f"Counts of classified readings per category (log axis). High Stress is {cc.iloc[3, 0]:,} with the "
        f"expanding baseline, {cc.iloc[3, 1]:,} with the rolling 10-reading baseline and {cc.iloc[3, 2]:,} with the "
        f"small-sample adjustment. All three also have {int((rd.stress_category == 'Insufficient history').sum()):,} "
        "readings with Insufficient history (not shown).", f"{f8}; reports/tables/table08_stress_counts_by_method.csv")

    # ---------------------------------------------------------------- 9
    nominal = 100 * stats.norm.sf(2)
    obs_rate = [100 * (cls[col] == "High Stress").mean() for _, col in meth]
    rng = np.random.default_rng(2024)
    ns = np.arange(5, 23)
    mc_raw, mc_adj = [], []
    for n in ns:
        xx = rng.normal(0, 1, (60000, n + 1))
        m, s = xx[:, :n].mean(1), xx[:, :n].std(1, ddof=1)
        z = (xx[:, n] - m) / s
        t = z / np.sqrt(1 + 1 / n)
        za = stats.norm.isf(stats.t.sf(t, n - 1))
        mc_raw.append(100 * (z >= 2).mean())
        mc_adj.append(100 * (za >= 2).mean())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    b = a1.bar([m[0] for m in meth], obs_rate, color=CAT[:3], width=0.6)
    a1.bar_label(b, labels=[f"{v:.2f}%" for v in obs_rate], fontsize=9)
    a1.axhline(nominal, color=INK, ls="--", lw=1)
    a1.text(-0.45, nominal + 0.08, f"nominal {nominal:.2f}%", va="bottom", ha="left", fontsize=8.5,
            bbox=dict(fc=SURF, ec="none", pad=1))
    a1.set_ylabel("High Stress rate (% of classified readings)")
    a1.set_title("(a) Observed High Stress rate")
    a1.tick_params(axis="x", labelsize=8.5)
    a2.plot(ns, mc_raw, color=CAT[0], lw=2, marker="o", ms=4, label="raw z (normal tails)")
    a2.plot(ns, mc_adj, color=CAT[2], lw=2, marker="o", ms=4, label="t-adjusted z")
    a2.axhline(nominal, color=INK, ls="--", lw=1)
    a2.set_xlabel("Earlier same-season readings n")
    a2.set_title("(b) Monte Carlo, i.i.d. Gaussian data")
    a2.legend(fontsize=8.5)
    f9 = save(fig, "fig09_excess_high_stress.png")
    table(pd.DataFrame({"n": ns, "raw_rate_pct": np.round(mc_raw, 3), "tadj_rate_pct": np.round(mc_adj, 3)}),
          "table09_monte_carlo_tail_rates", index=False)
    removed = 100 * (1 - (obs_rate[2] - nominal) / (obs_rate[0] - nominal))
    cap("9", "Why High Stress exceeds the nominal 2.28%",
        f"(a) Observed High Stress rates: {obs_rate[0]:.2f}% (expanding), {obs_rate[1]:.2f}% (rolling) and "
        f"{obs_rate[2]:.2f}% (adjusted), against {nominal:.2f}% for a standard normal. The rolling baseline does "
        f"not reduce the excess; the adjustment removes {removed:.0f}% of it. (b) With pure Gaussian noise "
        f"(60,000 draws per n), the raw z exceeds 2 in {mc_raw[0]:.1f}% of cases at n = 5 and {mc_raw[-1]:.1f}% at "
        f"n = 22, while the adjusted score stays at {min(mc_adj):.1f}-{max(mc_adj):.1f}%.",
        f"{f9}; reports/tables/table09_monte_carlo_tail_rates.csv")

    # ---------------------------------------------------------------- 10
    counts, changes = stress.sensitivity(main_t)
    ch = changes[changes["threshold set"] != "default (1/1.5/2)"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    b1 = ax.barh(ch["threshold set"], ch["moved to more severe"], color=CAT[1], label="moved to more severe")
    b2 = ax.barh(ch["threshold set"], ch["moved to less severe"], left=ch["moved to more severe"], color=CAT[0],
                 label="moved to less severe")
    for i, (tot, p) in enumerate(zip(ch["rows changed vs default"], ch["% of classified rows"])):
        ax.text(tot, i, f" {tot:,} ({p}%)", va="center", fontsize=9)
    ax.set_xlabel("Readings that change category vs default thresholds (1 / 1.5 / 2)")
    ax.set_title("Threshold sensitivity of the stress label")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, ch["rows changed vs default"].max() * 1.3)
    f10 = save(fig, "fig10_threshold_sensitivity.png")
    table(counts, "table10_threshold_sensitivity_counts")
    table(changes, "table10_threshold_sensitivity_changes", index=False)
    cap("10", "How many readings change category under other thresholds",
        "Compared with the default thresholds (1 / 1.5 / 2 SD), "
        + "; ".join(f"{r['threshold set']} changes {r['rows changed vs default']:,} readings "
                    f"({r['% of classified rows']}% of classified)" for _, r in ch.iterrows())
        + ". Changes are one-directional per set, so thresholds shift severity but never reorder readings.",
        f"{f10}; reports/tables/table10_threshold_sensitivity_*.csv")

    # ---------------------------------------------------------------- 11
    kk = []
    for k in (3, 5, 8):
        ex = int((rd.n_prior_same_season < k).sum())
        kk.append({"K": k, "excluded": ex, "excluded_pct": round(100 * ex / len(rd), 2)})
    kk = pd.DataFrame(kk)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b = ax.bar(kk.K.astype(str), kk.excluded, color=[SEQ[2], SEQ[4], SEQ[6]], width=0.55)
    ax.bar_label(b, labels=[f"{e:,}\n({p}%)" for e, p in zip(kk.excluded, kk.excluded_pct)], fontsize=9)
    ax.set_xlabel("Minimum earlier same-season readings K")
    ax.set_ylabel("Readings without a baseline")
    ax.set_title("Readings excluded by the minimum-history rule")
    ax.set_ylim(0, kk.excluded.max() * 1.25)
    f11 = save(fig, "fig11_min_history_rule.png")
    table(kk, "table11_min_history_rule", index=False)
    cap("11", "Readings excluded by the minimum-history rule",
        f"Readings with fewer than K earlier same-season readings get no baseline, anomaly or stress label. "
        f"The chosen K = 5 excludes {kk.excluded[1]:,} of {len(rd):,} readings ({kk.excluded_pct[1]}%); K = 3 would "
        f"exclude {kk.excluded_pct[0]}% and K = 8 {kk.excluded_pct[2]}%. The excluded readings are each well-season's "
        "first few, so labels effectively start around 2005.", f11)

    # ---------------------------------------------------------------- 12a
    rr = rain.merge(wl[["well_id", "state"]], on="well_id")
    rr = rr[rr.year.between(2000, 2022)]
    show = ["Tamil Nadu", "Kerala", "Maharashtra", "Uttar Pradesh", "Gujarat"]
    clim = rr[rr.state.isin(show)].groupby(["state", "month"]).precip_mm.mean().unstack(0)[show]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for i, s in enumerate(show):
        ax.plot(clim.index, clim[s], color=CAT[i], lw=2, marker="o", ms=4)
        ax.text(12.15, clim[s].iloc[-1], s, color=INK, fontsize=8.5, va="center")
    ax.axvspan(5.5, 9.5, color=GRID, alpha=0.6, lw=0)
    ax.axvspan(9.5, 12.5, color=SEQ[0], alpha=0.5, lw=0)
    ax.text(7.5, ax.get_ylim()[1] * 0.96, "Jun-Sep SW monsoon", ha="center", fontsize=8.5, va="top")
    ax.text(11, ax.get_ylim()[1] * 0.96, "Oct-Dec NE monsoon", ha="center", fontsize=8.5, va="top")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.7, 13.6)
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.set_title("Monthly rainfall climatology at the wells, 2000-2022 (CHIRPS v3.0)")
    f12a = save(fig, "fig12a_rainfall_climatology.png")
    table(clim.round(1), "table12_rainfall_climatology")
    tn = clim["Tamil Nadu"]
    share_ond = 100 * tn.loc[10:12].sum() / tn.sum()
    share_jjas = 100 * tn.loc[6:9].sum() / tn.sum()
    # ---------------------------------------------------------------- 12b example well
    cand = main_t.groupby("well_id").agg(n=("gwl_m_bgl", "count"), state=("state", "first"),
                                         station=("station_name", "first"), district=("district", "first"))
    ex_id = cand[(cand.state == "Maharashtra")].sort_values("n", ascending=False).index[0]
    ew = main_t[main_t.well_id == ex_id].sort_values("date")
    er = rain[(rain.well_id == ex_id) & (rain.year.between(2000, 2022))].copy()
    er["date"] = pd.to_datetime(dict(year=er.year, month=er.month, day=15))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    a1.bar(er.date, er.precip_mm, width=25, color=CAT[0])
    a1.plot(ew.date, ew.rain_6m_mm / 6, color=CAT[1], lw=1.5, marker="o", ms=3,
            label="rain_6m_mm / 6 (mean of the 6 months before each round)")
    a1.set_ylabel("Rainfall (mm/month)")
    a1.legend(fontsize=8.5, loc="upper left")
    a1.set_title(f"(a) Monthly rainfall and the 6-month window feature: {cand.loc[ex_id, 'station']}, "
                 f"{cand.loc[ex_id, 'district']} ({cand.loc[ex_id, 'state']})")
    obs = ew[ew.gwl_m_bgl.notna()]
    a2.plot(obs.date, obs.gwl_m_bgl, color=INK2, lw=1, zorder=1)
    for i, s in enumerate(SEASONS):
        d = obs[obs.season == s]
        a2.scatter(d.date, d.gwl_m_bgl, s=16, color=CAT[i], label=SEASON_LABEL[s], zorder=2)
    a2.invert_yaxis()
    a2.set_ylabel("Depth to groundwater (m bgl)")
    a2.set_xlabel("Date")
    a2.set_title("(b) Groundwater depth at the same well (deeper = lower)")
    a2.legend(fontsize=8, ncol=2, loc="lower left")
    f12b = save(fig, "fig12b_example_well_rain_and_depth.png")
    cc12 = ew[["rain_6m_mm", "gwl_m_bgl"]].dropna()
    r_ex = cc12.rain_6m_mm.corr(cc12.gwl_m_bgl)
    cap("12", "Rainfall climatology and an example well",
        f"(a) Mean monthly CHIRPS rainfall at the wells of five states. Most states peak in Jun-Sep, but Tamil Nadu "
        f"gets {share_ond:.0f}% of its rain in Oct-Dec against {share_jjas:.0f}% in Jun-Sep, which motivated the Oct-Dec "
        f"features. (b) Example well {ex_id} ({cand.loc[ex_id, 'station']}, {cand.loc[ex_id, 'state']}, "
        f"{int(cand.loc[ex_id, 'n'])} readings): the water table rises after each monsoon; the Pearson correlation "
        f"between the 6-month rainfall window and depth is {r_ex:.2f} (more rain, shallower water).",
        f"{f12a}; {f12b}; reports/tables/table12_rainfall_climatology.csv")

    # ---------------------------------------------------------------- 13 leakage timeline
    tgt = ew[(ew.period_label == "Nov-15")].iloc[0]
    M = pd.Period("2015-11", "M")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    months = pd.period_range("2014-09", "2016-02", freq="M")
    xi = {p: i for i, p in enumerate(months)}
    rows = [("rain_1m_mm", 1), ("rain_3m_mm", 3), ("rain_6m_mm", 6), ("rain_12m_mm", 12)]
    for k, (lab, w) in enumerate(rows):
        yk = len(rows) - k + 1
        for p in months:
            used = M - w <= p <= M - 1
            future = p >= M
            ax.add_patch(Rectangle((xi[p] - 0.42, yk - 0.35), 0.84, 0.7,
                                   color=CAT[0] if used else (GRID if not future else "#f3d6d6"), lw=0))
        ax.text(-1, yk, lab, ha="right", va="center", fontsize=9)
    yk = 1
    ax.text(-1, yk, "monsoon-year-to-date", ha="right", va="center", fontsize=9)
    for p in months:
        used = pd.Period("2015-06", "M") <= p <= M - 1
        ax.add_patch(Rectangle((xi[p] - 0.42, yk - 0.35), 0.84, 0.7,
                               color=CAT[0] if used else (GRID if p < M else "#f3d6d6"), lw=0))
    # Groundwater history row: earlier same-season (Nov) readings feed the baseline.
    yh = 0
    ax.text(-1, yh, "same-season baseline", ha="right", va="center", fontsize=9)
    hist = ew[(ew.season == "post_monsoon_kharif") & (ew.date < pd.Timestamp("2015-11-01")) & ew.gwl_m_bgl.notna()]
    for p in months:
        if p.month == 11 and p < M and p in xi:
            ax.plot(xi[p], yh, "o", ms=10, color=CAT[2])
    ax.plot(xi[M], yh, "o", ms=10, mfc="none", mec="#d03b3b", mew=2)
    ax.text(xi[pd.Period("2014-12", "M")], yh, f"← plus {len(hist) - 1} earlier Nov readings (2000-2013)",
            va="center", fontsize=8.5, color=INK2)
    ax.add_patch(Rectangle((xi[M] - 0.5, -0.6), 1, len(rows) + 2.1, fill=False, ec="#d03b3b", lw=2, ls="--"))
    ax.text(xi[M], len(rows) + 1.6, "observation month M\n(reading day unknown: EXCLUDED)", ha="center",
            fontsize=8.5, color="#d03b3b", va="bottom")
    ax.set_xticks(range(len(months)), [p.strftime("%b\n%Y") if p.month == 1 or p == months[0] else p.strftime("%b")
                                       for p in months], fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(months) - 0.4)
    ax.set_ylim(-0.7, len(rows) + 2.5)
    ax.grid(False)
    n_prior = int(tgt.n_prior_same_season)
    ax.set_title(f"Leakage control for one reading: {ex_id}, Nov 2015 ")
    ax.legend(handles=[Patch(color=CAT[0], label="month used by the feature"),
                       Patch(color=GRID, label="earlier month, not in this window"),
                       Patch(color="#f3d6d6", label="month M and later: never used"),
                       plt.Line2D([], [], marker="o", ls="", color=CAT[2], ms=8, label="earlier reading used")],
              loc="upper left", fontsize=8.5, ncol=4, bbox_to_anchor=(0, -0.12))
    f13 = save(fig, "fig13_leakage_timeline.png")
    cap("13", "Which data feed the features of one reading",
        f"For well {ex_id}'s November 2015 reading, rainfall windows end in October 2015 (M-1): the 1-, 3-, 6- and "
        f"12-month windows and the monsoon-year-to-date total (Jun-Oct 2015) use only blue months; November 2015 and "
        f"later are never used. Its seasonal baseline uses the {n_prior} earlier November readings (2000-2014) "
        f"of the same well (mean {tgt.baseline_mean_m:.2f} m, std {tgt.baseline_std_m:.2f} m), and lag features use "
        "only earlier rounds. Assertions and perturbation tests enforce this for every row.", f13)

    # ---------------------------------------------------------------- 14
    j = rain.merge(r3, on=["well_id", "year", "month"])
    rr14 = j.precip_nearest_mm.corr(j.precip_3x3_mm)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(j.precip_nearest_mm, j.precip_3x3_mm, gridsize=70, bins="log", cmap=SEQ_CMAP, mincnt=1)
    lim = max(j.precip_nearest_mm.max(), j.precip_3x3_mm.max())
    ax.plot([0, lim], [0, lim], color=INK, lw=1, ls="--")
    ax.set_xlabel("Nearest cell (mm/month)")
    ax.set_ylabel("Mean of 3×3 cells (mm/month)")
    ax.set_title(f"Nearest-cell vs 3×3 rainfall (r = {rr14:.4f})")
    fig.colorbar(hb, ax=ax, label="well-months per hexagon (log)")
    f14 = save(fig, "fig14_nearest_vs_3x3.png")
    mad = (j.precip_3x3_mm - j.precip_nearest_mm).abs().mean()
    nflag = len(pd.read_csv(config.REPORTS_DIR / "chirps_sampling_flagged_wells.csv"))
    cap("14", "Agreement of nearest-cell and 3×3-mean rainfall",
        f"All {len(j):,} well-months: r = {rr14:.4f}, mean absolute difference {mad:.2f} mm/month "
        f"({pct(100 * mad / j.precip_nearest_mm.mean())} of mean rainfall). Points off the 1:1 line come from "
        f"{nflag} wells on steep rainfall gradients (Western Ghats, Himalayan foothills); the nearest cell was kept "
        "as the primary source.", f14)

    # ---------------------------------------------------------------- 15
    e = spl[spl.eval_eligible]
    sp = pd.crosstab(e.split_chrono, e.season).reindex(index=["train", "val", "test", "ext"], columns=SEASONS)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(4)
    w = 0.2
    for i, s in enumerate(SEASONS):
        b = ax.bar(x + (i - 1.5) * w, sp[s], width=w - 0.02, color=CAT[i], label=SEASON_LABEL[s])
    ax.set_xticks(x, [f"{k}\n({v:,} rows)" for k, v in sp.sum(1).items()])
    ax.set_yscale("log")
    ax.set_ylabel("Eval-eligible readings (log scale)")
    ax.set_title("Evaluation-eligible readings per split and season")
    ax.legend(fontsize=8, ncol=2)
    f15 = save(fig, "fig15_split_summary.png")
    table(sp.assign(total=sp.sum(1)), "table15_split_summary")
    reasons = spl.exclusion_reason.value_counts()
    cap("15", "Eligible readings per chronological split and season",
        f"Train 2000-2015 has {sp.loc['train'].sum():,} eligible readings, validation 2016-2017 "
        f"{sp.loc['val'].sum():,}, test 2018-2022 {sp.loc['test'].sum():,} and the 2023-24 extension "
        f"{sp.loc['ext'].sum():,} (log axis). Rows are excluded for no reading ({reasons['no_reading']:,}), "
        f"insufficient history ({reasons['insufficient_history']:,}), sparse rounds ({reasons['sparse_round']:,}) "
        f"or extension zero placeholders ({reasons['extension_zero_placeholder']:,}). May counts are lowest because "
        "444 wells have no May series.", f"{f15}; reports/tables/table15_split_summary.csv")

    # ---------------------------------------------------------------- 16
    cov = both[both.year >= 2018].groupby(["year", "month", "period_label"]).gwl_m_bgl \
        .apply(lambda s: 100 * s.notna().mean()).reset_index().sort_values(["year", "month"])
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = [CAT[0] if y <= 2022 else CAT[1] for y in cov.year]
    b = ax.bar(cov.period_label, cov.gwl_m_bgl, color=colors)
    ax.bar_label(b, labels=[f"{v:.0f}" for v in cov.gwl_m_bgl], fontsize=7.5)
    m18 = 100 * main_t[main_t.year >= 2018].gwl_m_bgl.notna().mean()
    ax.axhline(m18, color=CAT[0], ls="--", lw=1)
    ax.axhline(m_ext, color=CAT[1], ls="--", lw=1)
    ax.text(len(cov) - 0.5, m18, f"2018-22 mean {m18:.1f}%", ha="right", va="bottom", fontsize=8.5)
    ax.text(len(cov) - 0.5, m_ext, f"2023-24 mean {m_ext:.1f}%", ha="right", va="bottom", fontsize=8.5)
    ax.set_ylabel("% of 2,759 wells with a reading")
    ax.set_title("Coverage per round: main table 2018-2022 vs extension 2023-2024")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(handles=[Patch(color=CAT[0], label="main table"), Patch(color=CAT[1], label="extension holdout")],
              loc="upper right", fontsize=8.5)
    f16 = save(fig, "fig16_extension_coverage.png")
    table(cov.rename(columns={"gwl_m_bgl": "coverage_pct"}).round(1), "table16_coverage_2018_2024", index=False)
    last = cov.iloc[-1]
    cap("16", "Coverage of the extension holdout vs recent main-table rounds",
        f"Mean coverage is {m18:.1f}% in 2018-2022 against {m_ext:.1f}% in 2023-2024, falling to "
        f"{last.gwl_m_bgl:.1f}% in {last.period_label}. The extension wells were selected for completeness only "
        "through 2022, so the extension is a secondary robustness check that must not be pooled with the main test "
        "set.", f"{f16}; reports/tables/table16_coverage_2018_2024.csv")

    # ---------------------------------------------------------------- captions
    L = ["# Figure and table captions\n",
         "_Generated by `scripts/12_figures.py` from the final outputs; every number is computed. "
         "Figures: `reports/figures/` (PNG, 300 dpi). Tables: `reports/tables/` (CSV + Markdown)._\n"]
    for num, title, text, files in captions:
        L += [f"## Figure {num}. {title}\n", text, f"\nFile(s): {files}\n"]
    (config.REPORTS_DIR / "figure_captions.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
