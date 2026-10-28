"""Generate notebooks/error_metrics_before.ipynb and notebooks/error_metrics_after.ipynb.

Both notebooks run the same analysis (same cells, same metrics module,
gwstress.error_metrics) on different data:
  before: the RAW CGWB table (and, like-for-like, the raw values of the same
          2,759 wells that survive cleaning)
  after : the CLEANED table the pipeline uses, followed by a before/after
          comparison section.

The notebooks are then executed with nbconvert so the committed files carry
their tables and graphs:
    jupyter nbconvert --to notebook --execute --inplace notebooks/error_metrics_before.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/error_metrics_after.ipynb
(run "before" first: "after" cross-checks its numbers against the CSV the
"before" notebook saves).

Run: python scripts/16_make_error_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwstress import error_metrics as em  # noqa: E402

REPO_URL = "https://github.com/dhruvvvgg/bde-team7.git"
EXAMPLE_WELL = "W42fd920d92"   # same example well as figures 12b/13 (Pimpri (kh) malvasti, Pune)

FORMULAS = {
    "RMSE": "√( mean(e²) )", "MSE": "mean(e²)", "MAE": "mean(|e|)",
    "MAPE": "100 · mean(|e| / |y|), only where |y| ≥ 1 m", "R2": "1 − Σe² / Σ(y − ȳ)²",
    "NRMSE": "RMSE / (max y − min y)", "MedAE": "median(|e|)", "Bias": "mean(e)",
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def metric_definitions() -> str:
    rows = ["| Metric | Full form | Formula (e = predicted − actual, y = actual) | Better when | Purpose |",
            "|---|---|---|---|---|"]
    for _, r in em.METRIC_INFO.iterrows():
        rows.append(f"| **{r.Metric}** | {r['Full form']} | `{FORMULAS[r.Metric]}` | {r['Better when']} | {r.Purpose} |")
    return "\n".join(rows)


INTRO = """
## How these scores are produced

Data preprocessing does not make predictions, so there is nothing for RMSE, MAE or R² to score directly.
To get *before* and *after* error scores anyway, the **same fixed forecasting rules** are applied to the data
before and after cleaning, and scored on the **2018–2022 test period** (the readings the rules try to predict).
If cleaning makes the data more reliable and regular, the same rule scores better.

Each rule predicts a well's depth to groundwater (metres below ground) for a monitoring round using **only earlier rounds**:

| Rule | Prediction for a round |
|---|---|
| **same season last year** | the reading of the same season one year earlier (e.g. May 2019 → May 2018) |
| **seasonal mean of earlier years** | the mean of all earlier readings of that season, needing at least 5 (the pipeline's seasonal baseline) |
| **previous round** | the immediately previous round (a weak rule that ignores the seasonal cycle) |

These scores measure **data quality, not model skill**; the classifier results belong to the modeling stage.
"""

NOTES = """
## Reading notes

- **Units:** RMSE, MAE, MedAE and Bias are in metres of depth to groundwater; MSE is in m²; MAPE is a percentage; R² and NRMSE have no unit.
- **MAPE** divides by the actual value, which explodes for shallow water (0–1 m is common). It is therefore computed only
  where |actual| ≥ 1 m, and the number of readings used is reported (`MAPE_n`). **sMAPE** is a bounded alternative.
- **NRMSE** is RMSE divided by the range of the actual values (max − min); `NRMSE_mean` divides by the mean instead. Because the raw table
  contains impossible values (−97 m to 300 m), its range is huge, which makes its NRMSE look small without the forecasts being better.
- **R²** is measured against each dataset's own spread (variance). Datasets with extreme values have a large spread, so R² can look high
  there. Compare R² only between datasets with the same wells (the like-for-like comparison).
- **Bias** is the mean of (predicted − actual). Positive means the rule predicts water deeper than what was measured. The rules learn from earlier
  years, so a bias appears when the test period differs from them; section 1 reports the measured difference in mean depth.
- The number of scored readings `n` differs between rules (a rule needs its own earlier readings to exist) and between datasets.
"""

SETUP = f"""
import os, subprocess, sys, warnings
from pathlib import Path

REPO_URL = "{REPO_URL}"      # if the repository is private, put an access token in the URL (never save it in the notebook)

# Use the current folder if it is the repository; otherwise clone it (e.g. in Colab).
if Path("src/gwstress").exists():
    ROOT = Path.cwd()
elif Path("../src/gwstress").exists():          # opened from notebooks/
    ROOT = Path("..").resolve()
else:
    if not Path("bde-team7").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "bde-team7"], check=True)
    ROOT = Path("bde-team7").resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown
from gwstress import error_metrics as em

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
BEFORE, AFTER = "#eb6834", "#2a78d6"            # entity colours: orange = before cleaning, blue = after
INK2, GRID = "#52514e", "#e1e0d9"
plt.rcParams.update({{"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False}})
SHORT = {{"same season last year": "same season\\nlast year", "seasonal mean of earlier years": "seasonal mean\\nof earlier years",
         "previous round": "previous\\nround"}}
METRICS = [("RMSE", "m"), ("MSE", "m²"), ("MAE", "m"), ("MAPE", "%"), ("R2", ""), ("NRMSE", "fraction of range"),
           ("MedAE", "m"), ("Bias", "m")]
ROUND_LABELS = [f"{{m}}-{{y % 100:02d}}" for y in range(2000, 2023) for m in em.SEASONS]
OUT = ROOT / "reports" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
print("repository:", ROOT)
"""

# --- analysis cells shared by both notebooks --------------------------------
LOAD = """
ds = em.__LOADER__()
COLOR = __COLOR__
display(ds.describe().rename(columns={"value": ds.name}).round(3))
"""

DISTRIBUTION = """
v = ds.values[np.isfinite(ds.values)]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].hist(np.clip(v, -100, 300), bins=130, color=COLOR)
ax[0].set_yscale("log")
ax[0].set_title("All readings (log count; clipped to −100…300 m)")
ax[0].set_xlabel("Depth to groundwater (m below ground)")
ax[0].set_ylabel("Number of readings (log)")
ax[1].hist(v[(v >= 0) & (v <= 40)], bins=80, color=COLOR)
ax[1].set_title("Plausible range 0…40 m (linear count)")
ax[1].set_xlabel("Depth to groundwater (m below ground)")
ax[1].set_ylabel("Number of readings")
plt.tight_layout(); plt.show()
n_bad = int(((v < 0) | (v > 100)).sum())
display(Markdown(f"**{ds.name}**: {len(ds.values):,} wells, {v.size:,} readings; "
                 f"range {v.min():g} to {v.max():g} m; **{int((v < 0).sum()):,} negative** and "
                 f"**{int((v > 100).sum()):,} readings above 100 m** (physically implausible)."))
early = np.nanmean(ds.values[:, :em.TEST_FIRST_COL]); late = np.nanmean(ds.values[:, em.TEST_FIRST_COL:])
display(Markdown(f"Mean depth in the earlier years (2000–2017): **{early:.2f} m**; in the test period (2018–2022): **{late:.2f} m** "
                 f"({late - early:+.2f} m). Rules that learn from earlier years inherit this shift as bias."))
"""

RULES_SECTION = """
preds = em.predictions(ds.values)
wid = np.flatnonzero(ds.well_ids == "__EXAMPLE__")
i = int(wid[0]) if len(wid) else int(np.argmax(np.isfinite(ds.values[:, em.TEST_FIRST_COL:]).sum(axis=1)))
cols = np.arange(em.TEST_FIRST_COL, em.N_ROUNDS)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(cols, ds.values[i, cols], color="black", lw=2, marker="o", ms=4, label="actual")
for (rule, p), mk, ls in zip(preds.items(), ["s", "^", "v"], ["-", "--", ":"]):
    ax.plot(cols, p[i, cols], color=COLOR, lw=1.6, ls=ls, marker=mk, ms=4, label=rule)
ax.set_xticks(cols[::2], [ROUND_LABELS[c] for c in cols[::2]], rotation=60)
ax.set_ylabel("Depth to groundwater (m)")
ax.invert_yaxis()
ax.set_title(f"Worked example, well {ds.well_ids[i]} (test period 2018–2022; deeper is lower)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45))
plt.tight_layout(); plt.show()
"""

TABLE = """
tbl = em.metric_table(ds)
show = tbl.set_index("rule")[["n", "RMSE", "MSE", "MAE", "MAPE", "R2", "NRMSE", "MedAE", "Bias"]]
show = show.rename(columns={"RMSE": "RMSE (m)", "MSE": "MSE (m²)", "MAE": "MAE (m)", "MAPE": "MAPE (%)",
                            "NRMSE": "NRMSE (range)", "MedAE": "MedAE (m)", "Bias": "Bias / ME (m)"})
display(Markdown(f"### All eight metrics: {ds.name}, test period 2018–2022"))
display(show.round(3))
display(Markdown("Extra scores: NRMSE relative to the mean, sMAPE, and how many readings MAPE could use:"))
display(tbl.set_index("rule")[["NRMSE_mean", "sMAPE", "MAPE_n"]].round(3))
"""

PANELS = """
fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
for ax, (m, unit) in zip(axes.ravel(), METRICS):
    vals = tbl[m].to_numpy()
    bars = ax.bar(range(3), vals, color=COLOR, width=0.6)
    ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
    ax.set_xticks(range(3), [SHORT[r] for r in tbl.rule], fontsize=7.5)
    ax.set_title(f"{m} ({unit})" if unit else m)
    if m == "Bias":
        ax.axhline(0, color=INK2, lw=0.8)
    ax.margins(y=0.15)
fig.suptitle(f"{ds.name}: every metric for each forecasting rule (test period 2018–2022)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""

SCATTER = """
rule = em.PRIMARY_RULE
y, q, pcols, prows = em.test_pairs(ds, rule)
e = q - y
lo, hi = np.percentile(y, [0.2, 99.8])
cmap = LinearSegmentedColormap.from_list("c", ["#ffffff", COLOR])
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
hb = ax[0].hexbin(y, q, gridsize=70, bins="log", cmap=cmap, mincnt=1, extent=(lo, hi, lo, hi))
ax[0].plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
ax[0].set_xlabel("Actual depth (m)"); ax[0].set_ylabel("Predicted depth (m)")
ax[0].set_title(f"Actual vs predicted ({rule})\\ndashed = perfect forecast")
fig.colorbar(hb, ax=ax[0], label="readings per cell (log)")
ax[1].hist(np.clip(e, -15, 15), bins=121, color=COLOR)
ax[1].axvline(0, color="black", lw=1); ax[1].axvline(e.mean(), color=INK2, lw=1.4, ls="--")
ax[1].text(e.mean(), ax[1].get_ylim()[1] * 0.92, f" bias {e.mean():+.2f} m", color=INK2, fontsize=9)
ax[1].set_xlabel("Error = predicted − actual (m), clipped to ±15"); ax[1].set_ylabel("Readings")
ax[1].set_title(f"Error distribution\\n{(np.abs(e) > 15).mean() * 100:.2f}% of errors beyond ±15 m")
season = np.array(em.SEASONS)[pcols % 4]
ax[2].boxplot([np.abs(e)[season == s] for s in em.SEASONS], showfliers=False, patch_artist=True,
              boxprops=dict(facecolor=COLOR, alpha=0.6), medianprops=dict(color="black"))
ax[2].set_xticks(range(1, 5), em.SEASONS)
ax[2].set_ylabel("Absolute error (m)"); ax[2].set_xlabel("Season of the round")
ax[2].set_title("Absolute error by season\\n(box = middle 50%, outliers hidden)")
plt.tight_layout(); plt.show()
"""

BY_SEASON = """
bs = em.breakdown(ds, em.PRIMARY_RULE, "season")
display(Markdown(f"### By season ({em.PRIMARY_RULE})"))
display(bs[["season", "wells", "n", "RMSE", "MSE", "MAE", "MAPE", "R2", "NRMSE", "MedAE", "Bias"]].round(3).set_index("season"))
piv = pd.DataFrame({r: em.breakdown(ds, r, "season").set_index("season")["RMSE"] for r in em.RULES})
display(Markdown("RMSE (m) by season for every rule:"))
display(piv.round(3))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, m in zip(ax, ["RMSE", "MAE", "Bias"]):
    b = a.bar(bs.season.astype(str), bs[m], color=COLOR, width=0.6)
    a.bar_label(b, fmt="%.2f", fontsize=8.5, padding=2)
    a.set_title(f"{m} by season (m)")
    if m == "Bias":
        a.axhline(0, color=INK2, lw=0.8)
    a.margins(y=0.15)
plt.tight_layout(); plt.show()
"""

BY_YEAR = """
by = em.breakdown(ds, em.PRIMARY_RULE, "year")
display(Markdown(f"### By year ({em.PRIMARY_RULE})"))
display(by[["year", "wells", "n", "RMSE", "MAE", "R2", "MedAE", "Bias"]].round(3).set_index("year"))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(by.year, by.RMSE, color=COLOR, lw=2, marker="o", label="RMSE")
ax.plot(by.year, by.MAE, color=COLOR, lw=2, ls="--", marker="s", label="MAE")
ax.set_xticks(by.year); ax.set_ylabel("Error (m)"); ax.set_xlabel("Year of the scored round")
ax.set_title("Forecast error by year"); ax.legend()
plt.tight_layout(); plt.show()
"""

BY_STATE = """
st = em.breakdown(ds, em.PRIMARY_RULE, "state").sort_values("RMSE", ascending=False)
display(Markdown(f"### By state ({em.PRIMARY_RULE}), sorted by RMSE"))
display(st[["state", "wells", "n", "RMSE", "MAE", "R2", "MedAE", "Bias"]].round(3).set_index("state"))
fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * len(st))))
b = ax.barh(st.state[::-1], st.RMSE[::-1], color=COLOR)
ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
ax.set_xlabel("RMSE (m)"); ax.set_title("RMSE by state (larger = harder to forecast)")
ax.margins(x=0.12)
plt.tight_layout(); plt.show()
"""

# ---------------------------------------------------------------------------
BEFORE_ONLY_SAME = """
raw_same = em.load_raw_same_wells()
tbl_same = em.metric_table(raw_same)
show = tbl_same.set_index("rule")[["n", "RMSE", "MSE", "MAE", "MAPE", "R2", "NRMSE", "MedAE", "Bias"]]
display(Markdown("### Like-for-like baseline: the raw values of exactly the 2,759 wells that survive cleaning"))
display(Markdown("This is the fair 'before' for the 'after' notebook: same wells, same rounds; only the values differ."))
display(show.round(3))
display(raw_same.describe().rename(columns={"value": raw_same.name}).round(3))
"""

BEFORE_SAVE = """
results = pd.concat([tbl, tbl_same], ignore_index=True)
results.to_csv(OUT / "error_metrics_before.csv", index=False)
by_season = pd.concat([em.breakdown(d, r, "season").assign(data=d.name, rule=r)
                       for d in (ds, raw_same) for r in em.RULES], ignore_index=True)
by_season.to_csv(OUT / "error_metrics_before_by_season.csv", index=False)
print("saved:", OUT / "error_metrics_before.csv")
print("saved:", OUT / "error_metrics_before_by_season.csv")
"""

BEFORE_TAKEAWAYS = """
p = tbl.set_index("rule").loc[em.PRIMARY_RULE]
s = tbl_same.set_index("rule").loc[em.PRIMARY_RULE]
display(Markdown(f'''
### Before-cleaning summary (rule: {em.PRIMARY_RULE}, test period 2018–2022)

- **Whole raw table:** RMSE **{p.RMSE:.2f} m**, MAE **{p.MAE:.2f} m**, R² **{p.R2:.3f}**, MedAE {p.MedAE:.2f} m, bias {p.Bias:+.2f} m, on {int(p.n):,} readings.
- **Raw values of the same 2,759 wells:** RMSE **{s.RMSE:.2f} m**, MAE **{s.MAE:.2f} m**, R² **{s.R2:.3f}**, on {int(s.n):,} readings.
- RMSE / MAE is **{p.RMSE / p.MAE:.2f}** for the whole raw table and **{s.RMSE / s.MAE:.2f}** for the same-wells raw values. A larger ratio means
  heavier-tailed errors: a few very large misses dominate the squared-error metrics (RMSE, MSE) more than the absolute ones (MAE, MedAE).
- Next: **error_metrics_after.ipynb** repeats this on the cleaned data and compares.
'''))
"""

# ---------------------------------------------------------------------------
AFTER_COMPARE_LOAD = """
raw_same = em.load_raw_same_wells()
raw_all = em.load_raw_all()
t_after, t_same, t_all = tbl, em.metric_table(raw_same), em.metric_table(raw_all)
blanked = int((np.isfinite(raw_same.values) & ~np.isfinite(ds.values)).sum())
changed = int((np.isfinite(raw_same.values) & np.isfinite(ds.values) & (np.abs(raw_same.values - ds.values) > 1e-9)).sum())
display(Markdown(f"Cleaning blanked **{blanked:,}** raw readings of these wells (the dataset authors' per-well 3-sigma outlier rule) "
                 f"and changed **{changed}** other values. Everything else is identical."))

saved = OUT / "error_metrics_before.csv"
if saved.exists():
    prev = pd.read_csv(saved)
    chk = prev.merge(pd.concat([t_all, t_same]), on=["data", "rule"], suffixes=("_saved", "_now"))
    assert len(chk) == len(prev) and all(np.allclose(chk[f"{c}_saved"], chk[f"{c}_now"], equal_nan=True)
                                         for c in ["n", "RMSE", "MAE", "MAPE", "R2", "MedAE", "Bias"])
    print("Cross-check passed: the 'before' numbers equal those saved by error_metrics_before.ipynb")
else:
    print("(error_metrics_before.csv not found; run the before notebook first to enable the cross-check)")
"""

AFTER_COMPARE_TABLES = """
BETTER = {"R2": "higher", "Bias": "closer to 0"}

def side_by_side(before, after, rule):
    b = before.set_index("rule").loc[rule]; a = after.set_index("rule").loc[rule]
    rows = []
    for m, unit in METRICS:
        verdict = (a[m] > b[m]) if m == "R2" else (abs(a[m]) < abs(b[m])) if m == "Bias" else (a[m] < b[m])
        rows.append({"metric": m + (f" ({unit})" if unit else ""), "before": b[m], "after": a[m], "change": a[m] - b[m],
                     "improvement": em.improvement(b[m], a[m], m), "better when": BETTER.get(m, "lower"),
                     "verdict": "better" if verdict else "worse"})
    out = pd.DataFrame(rows).set_index("metric")
    out["improvement"] = out["improvement"].round(2)
    return out

display(Markdown("## A. Like-for-like: the same 2,759 wells, raw values (before) vs cleaned values (after)"))
display(Markdown("`improvement` is the percentage reduction in error (positive = after is better); for R² it is the absolute change; for Bias it is the reduction in |bias|."))
for rule in em.RULES:
    display(Markdown(f"**Rule: {rule}**   (before n = {int(t_same.set_index('rule').loc[rule, 'n']):,}, after n = {int(t_after.set_index('rule').loc[rule, 'n']):,})"))
    display(side_by_side(t_same, t_after, rule).round(4))
"""

AFTER_COMPARE_PANELS = """
def compare_panels(tb, ta, title, lb="before", la="after"):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8))
    x = np.arange(3); w = 0.38
    for ax, (m, unit) in zip(axes.ravel(), METRICS):
        b1 = ax.bar(x - w / 2, tb[m], w, color=BEFORE, label=lb)
        b2 = ax.bar(x + w / 2, ta[m], w, color=AFTER, label=la)
        ax.bar_label(b1, fmt="%.3g", fontsize=7, padding=1); ax.bar_label(b2, fmt="%.3g", fontsize=7, padding=1)
        ax.set_xticks(x, [SHORT[r] for r in tb.rule], fontsize=7.5)
        ax.set_title(f"{m} ({unit})" if unit else m)
        if m == "Bias":
            ax.axhline(0, color=INK2, lw=0.8)
        ax.margins(y=0.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

compare_panels(t_same, t_after, "A. Same 2,759 wells: raw values (orange) vs cleaned values (blue); lower is better except R²", "raw values", "cleaned")
"""

AFTER_COMPARE_BREAKDOWN = """
rule = em.PRIMARY_RULE
bsb = em.breakdown(raw_same, rule, "season"); bsa = em.breakdown(ds, rule, "season")
display(Markdown(f"### Season-by-season, same 2,759 wells ({rule})"))
sea = pd.DataFrame({("RMSE", "before"): bsb.RMSE.values, ("RMSE", "after"): bsa.RMSE.values,
                    ("MAE", "before"): bsb.MAE.values, ("MAE", "after"): bsa.MAE.values,
                    ("R2", "before"): bsb.R2.values, ("R2", "after"): bsa.R2.values,
                    ("Bias", "before"): bsb.Bias.values, ("Bias", "after"): bsa.Bias.values}, index=bsa.season.astype(str))
display(sea.round(3))
fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.8))
x = np.arange(4); w = 0.38
for a, m in zip(ax, ["RMSE", "MAE", "R2"]):
    b1 = a.bar(x - w / 2, bsb[m], w, color=BEFORE, label="raw values"); b2 = a.bar(x + w / 2, bsa[m], w, color=AFTER, label="cleaned")
    a.bar_label(b1, fmt="%.2f", fontsize=7.5, padding=1); a.bar_label(b2, fmt="%.2f", fontsize=7.5, padding=1)
    a.set_xticks(x, em.SEASONS); a.set_title(f"{m} by season"); a.margins(y=0.15)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.show()

yb = em.breakdown(raw_same, rule, "year"); ya = em.breakdown(ds, rule, "year")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, m in zip(ax, ["RMSE", "MAE"]):
    a.plot(yb.year, yb[m], color=BEFORE, lw=2, marker="o", label="raw values")
    a.plot(ya.year, ya[m], color=AFTER, lw=2, marker="o", label="cleaned")
    a.set_xticks(ya.year); a.set_title(f"{m} by year (m)"); a.set_xlabel("Year of the scored round")
ax[0].legend()
plt.tight_layout(); plt.show()

yb_, qb_, _, _ = em.test_pairs(raw_same, rule); ya_, qa_, _, _ = em.test_pairs(ds, rule)
eb, ea = qb_ - yb_, qa_ - ya_
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
bins = np.linspace(-12, 12, 97)
ax[0].hist(np.clip(eb, -12, 12), bins=bins, histtype="step", lw=2, color=BEFORE, density=True, label="raw values")
ax[0].hist(np.clip(ea, -12, 12), bins=bins, histtype="step", lw=2, color=AFTER, density=True, label="cleaned")
ax[0].set_title("Error distribution (density, clipped to ±12 m)"); ax[0].set_xlabel("Error = predicted − actual (m)"); ax[0].legend()
ax[1].boxplot([np.abs(eb), np.abs(ea)], showfliers=False, patch_artist=True,
              boxprops=dict(alpha=0.6), medianprops=dict(color="black"))
ax[1].set_xticks([1, 2], ["raw values", "cleaned"])
for patch, c in zip(ax[1].patches, [BEFORE, AFTER]):
    patch.set_facecolor(c)
ax[1].set_ylabel("Absolute error (m)"); ax[1].set_title("Absolute error (outliers hidden)")
plt.tight_layout(); plt.show()
"""

AFTER_COMPARE_B = """
display(Markdown("## B. Whole raw table (all well rows) vs the cleaned 2,759 wells"))
display(Markdown("Different wells and a very different spread of values, so R², NRMSE and MAPE are **not** comparable here (see the reading notes). "
                 "Errors fall sharply mainly because of which wells were kept (section C splits the gain into well selection and value cleaning)."))
for rule in em.RULES:
    display(Markdown(f"**Rule: {rule}**   (before n = {int(t_all.set_index('rule').loc[rule, 'n']):,}, after n = {int(t_after.set_index('rule').loc[rule, 'n']):,})"))
    display(side_by_side(t_all, t_after, rule).round(4))
compare_panels(t_all, t_after, "B. Whole raw table (orange) vs cleaned 2,759 wells (blue); R², NRMSE, MAPE are not comparable here", "raw, all wells", "cleaned")
"""

AFTER_DECOMPOSE = """
rule = em.PRIMARY_RULE
stages = [("1. raw table, all wells", raw_all, t_all), ("2. raw values, same 2,759 wells", raw_same, t_same),
          ("3. cleaned, 2,759 wells", ds, t_after)]
rows = []
for name, d, t in stages:
    x = t.set_index("rule").loc[rule]
    rows.append({"stage": name, "wells": len(d.values), "n": int(x.n), "RMSE": x.RMSE, "MAE": x.MAE, "MedAE": x.MedAE, "R2": x.R2})
dec = pd.DataFrame(rows).set_index("stage")
display(Markdown(f"### Rule: {rule}"))
display(Markdown("Step 1 → 2 changes **which wells** are scored: the authors kept only wells without negative values, with at least 2 readings in every year "
                 "and no stuck repeated values. Step 2 → 3 changes **values** only: outlier readings were blanked."))
display(dec.round(3))
sel = 100 * (dec.RMSE.iloc[0] - dec.RMSE.iloc[1]) / (dec.RMSE.iloc[0] - dec.RMSE.iloc[2])
selm = 100 * (dec.MAE.iloc[0] - dec.MAE.iloc[1]) / (dec.MAE.iloc[0] - dec.MAE.iloc[2])
display(Markdown(f"Share of the total reduction going from step 1 to step 3: RMSE **{sel:.1f}% from well selection** and {100 - sel:.1f}% from blanking readings; "
                 f"MAE {selm:.1f}% / {100 - selm:.1f}%. (R² is not comparable between stage 1 and the others; see the reading notes.)"))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for a, m in zip(ax, ["RMSE", "MAE", "MedAE"]):
    b = a.bar(range(3), dec[m], color=[BEFORE, BEFORE, AFTER], width=0.6)
    b[1].set_hatch("//"); b[1].set_edgecolor("white")
    a.bar_label(b, fmt="%.2f", padding=2)
    a.set_xticks(range(3), ["1. raw,\\nall wells", "2. raw values,\\nsame wells", "3. cleaned"])
    a.set_title(f"{m} (m)"); a.margins(y=0.15)
plt.tight_layout(); plt.show()
"""

AFTER_SAVE = """
results = t_after.copy()
results.to_csv(OUT / "error_metrics_after.csv", index=False)
pd.concat([em.breakdown(ds, r, "season").assign(data=ds.name, rule=r) for r in em.RULES], ignore_index=True) \\
  .to_csv(OUT / "error_metrics_after_by_season.csv", index=False)
cmp_rows = []
for tag, tb in [("A: same 2,759 wells", t_same), ("B: all raw wells", t_all)]:
    for rule in em.RULES:
        sbs = side_by_side(tb, t_after, rule).reset_index().assign(comparison=tag, rule=rule)
        cmp_rows.append(sbs)
pd.concat(cmp_rows, ignore_index=True).to_csv(OUT / "error_metrics_comparison.csv", index=False)
print("saved:", OUT / "error_metrics_after.csv"); print("saved:", OUT / "error_metrics_after_by_season.csv")
print("saved:", OUT / "error_metrics_comparison.csv")
"""

AFTER_TAKEAWAYS = """
shift = np.nanmean(ds.values[:, em.TEST_FIRST_COL:]) - np.nanmean(ds.values[:, :em.TEST_FIRST_COL])
cmp = side_by_side(t_same, t_after, em.PRIMARY_RULE)
better = int((cmp.verdict == "better").sum())
display(Markdown(f'''
## Scorecard: did preprocessing help? (same 2,759 wells, rule: {em.PRIMARY_RULE}, test period 2018–2022)

**{better} of {len(cmp)} metrics improved** after cleaning.

- RMSE {cmp.loc["RMSE (m)", "before"]:.3f} → **{cmp.loc["RMSE (m)", "after"]:.3f} m** ({cmp.loc["RMSE (m)", "improvement"]:+.1f}%); MSE {cmp.loc["MSE (m²)", "improvement"]:+.1f}%
- MAE {cmp.loc["MAE (m)", "before"]:.3f} → **{cmp.loc["MAE (m)", "after"]:.3f} m** ({cmp.loc["MAE (m)", "improvement"]:+.1f}%); MedAE {cmp.loc["MedAE (m)", "improvement"]:+.1f}%
- R² {cmp.loc["R2", "before"]:.3f} → **{cmp.loc["R2", "after"]:.3f}** ({cmp.loc["R2", "change"]:+.3f})
- MAPE {cmp.loc["MAPE (%)", "before"]:.1f}% → **{cmp.loc["MAPE (%)", "after"]:.1f}%**; NRMSE {cmp.loc["NRMSE (fraction of range)", "before"]:.3f} → **{cmp.loc["NRMSE (fraction of range)", "after"]:.3f}**
- |Bias| {abs(cmp.loc["Bias (m)", "before"]):.3f} → {abs(cmp.loc["Bias (m)", "after"]):.3f} m (the test period is {abs(shift):.2f} m {"shallower" if shift < 0 else "deeper"} on average than the earlier years the rules learn from, consistent with the {"positive" if cmp.loc["Bias (m)", "after"] > 0 else "negative"} bias)

**Why the improvement is modest:** the cleaning only blanked {blanked:,} extreme readings ({100 * blanked / np.isfinite(raw_same.values).sum():.2f}% of the readings) and changed no other value.
Across the whole raw table the bigger gain ({sel:.0f}% of the RMSE reduction) comes from *which wells were kept* (the authors' selection rules), not from fixing values.
This pipeline's own steps add flags and features and change no reading, so on their own they would score identically to the raw values.
'''))
"""


def build(stage: str) -> dict:
    after = stage == "after"
    cells = [md(f"""
# Error metrics {'AFTER' if after else 'BEFORE'} data preprocessing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhruvvvgg/bde-team7/blob/main/notebooks/error_metrics_{stage}.ipynb)

{'The cleaned groundwater dataset the pipeline uses (2,759 quality-controlled wells). Ends with a before/after comparison.' if after else 'The raw CGWB groundwater table, before any quality control. Run this notebook first; the after notebook cross-checks against the numbers saved here.'}

Eight error metrics: RMSE, MSE, MAE, MAPE, R², NRMSE, MedAE and Bias (mean error), plus sMAPE.
"""), md("## The metrics\n\n" + metric_definitions()), md(INTRO), md(NOTES),
        md("## 0. Setup"), code(SETUP)]
    loader, color = ("load_cleaned", "AFTER") if after else ("load_raw_all", "BEFORE")
    cells += [md(f"## 1. The data ({'after' if after else 'before'} cleaning)"),
              code(LOAD.replace("__LOADER__", loader).replace("__COLOR__", color)), code(DISTRIBUTION),
              md("## 2. The forecasting rules in action"), code(RULES_SECTION.replace("__EXAMPLE__", EXAMPLE_WELL)),
              md("## 3. All eight metrics"), code(TABLE), code(PANELS),
              md("## 4. Where the errors are"), code(SCATTER),
              md("## 5. Breakdown by season, year and state"), code(BY_SEASON), code(BY_YEAR), code(BY_STATE)]
    if not after:
        cells += [md("## 6. Like-for-like baseline"), code(BEFORE_ONLY_SAME),
                  md("## 7. Save the results"), code(BEFORE_SAVE), code(BEFORE_TAKEAWAYS)]
    else:
        cells += [md("## 6. Before vs after comparison"), code(AFTER_COMPARE_LOAD), code(AFTER_COMPARE_TABLES),
                  code(AFTER_COMPARE_PANELS), code(AFTER_COMPARE_BREAKDOWN), code(AFTER_COMPARE_B),
                  md("## C. Where does the improvement come from?"), code(AFTER_DECOMPOSE),
                  md("## 7. Save the results and summarise"), code(AFTER_SAVE), code(AFTER_TAKEAWAYS)]
    for i, c in enumerate(cells):
        c["id"] = f"{stage}-{i:02d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for stage in ("before", "after"):
        p = out / f"error_metrics_{stage}.ipynb"
        p.write_text(json.dumps(build(stage), indent=1, ensure_ascii=False) + "\n")
        print("wrote", p)


if __name__ == "__main__":
    main()
