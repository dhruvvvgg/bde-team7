"""Data-quality error scores before vs after cleaning.

Preprocessing makes no predictions, so RMSE/MAE/R2 are computed for two
simple, fixed forecasting rules applied identically to the data before and
after cleaning. If cleaning improves the data, the same rule scores better.

  Rule 1 "same season last year": predict a round's depth with the reading
         of the same season one year earlier.
  Rule 2 "seasonal mean of earlier years": predict it with the mean of all
         earlier same-season readings (at least 5), i.e. the pipeline's
         expanding baseline.

Both rules use only earlier rounds. Scored on 2018-2022 readings (the test
period). Metrics: RMSE, MAE, R2, median absolute error, bias (mean of
prediction - actual) and sMAPE.

Comparison A (main result): the SAME 2,759 wells, raw values (authors'
Input/ table) vs the cleaned refined table. Only the values differ.
Comparison B: the whole raw table (32,299 well rows) vs the cleaned 2,759
wells. Different wells and very different spread, so R2 is not comparable
here (R2 is relative to the data's own variance).

Writes reports/preprocessing_error_scores.md and
reports/tables/table17_error_scores.{csv,md}.

Run: python scripts/15_error_scores.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, gw_io  # noqa: E402

RAW_FILE = config.GW_DIR / "Input" / "1_India_GWLs_2000_2024_wells_within_India.csv"
TEST_FIRST_COL = 72  # 92 rounds 2000-2022 in Jan/May/Aug/Nov order; column 72 = Jan-2018
MIN_PRIOR = 5


def obs_2000_2022(df: pd.DataFrame) -> list[str]:
    _, obs, _ = gw_io.split_columns(df.drop(columns="well_id"))
    return [c for c in obs if int(c[-2:]) <= 22]


def predictions(v: np.ndarray) -> dict:
    """Both rules use only earlier columns of the same row (past-only)."""
    lag = np.full_like(v, np.nan)
    lag[:, 4:] = v[:, :-4]
    clim = np.full_like(v, np.nan)
    for s in range(4):
        cols = list(range(s, v.shape[1], 4))
        x = v[:, cols]
        ok = ~np.isnan(x)
        cs, cn = np.cumsum(np.where(ok, x, 0), 1), np.cumsum(ok, 1)
        n = np.concatenate([np.zeros((len(x), 1)), cn[:, :-1]], 1)
        sm = np.concatenate([np.zeros((len(x), 1)), cs[:, :-1]], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            clim[:, cols] = np.where(n >= MIN_PRIOR, sm / n, np.nan)
    return {"same season last year": lag, "seasonal mean of earlier years": clim}


def score(v: np.ndarray, p: np.ndarray) -> dict:
    y = v.copy()
    y[:, :TEST_FIRST_COL] = np.nan
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    e = p - y
    return {"n": int(len(y)), "RMSE_m": np.sqrt((e ** 2).mean()), "MAE_m": np.abs(e).mean(),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "MedAE_m": np.median(np.abs(e)),
            "Bias_m": e.mean(), "sMAPE_pct": 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-9))}


def main() -> None:
    ref = gw_io.load_wide(config.GW_REFINED_FILE)
    raw = gw_io.load_wide(RAW_FILE, strict=False)
    obs = obs_2000_2022(ref)
    raw_vals = raw[obs].apply(pd.to_numeric, errors="coerce")

    # Same wells: the raw row with the most readings per well_id (one well has a sparse duplicate).
    cand = raw[raw.well_id.isin(ref.well_id)].copy()
    cand["_n"] = raw_vals.notna().sum(axis=1)
    cand = cand.sort_values("_n").drop_duplicates("well_id", keep="last").set_index("well_id").loc[ref.well_id]
    A = cand[obs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    B = ref.set_index("well_id")[obs].to_numpy(float)
    R = raw_vals.to_numpy(float)
    assert len(A) == len(B) == 2759
    blanked = int((np.isfinite(A) & ~np.isfinite(B)).sum())
    changed = int((np.isfinite(A) & np.isfinite(B) & (np.abs(A - B) > 1e-9)).sum())

    rows = []
    for comp, label, v in [("A: same 2,759 wells", "before (raw values)", A),
                           ("A: same 2,759 wells", "after (cleaned)", B),
                           ("B: all raw wells vs cleaned", "before (raw, 32,299 rows)", R),
                           ("B: all raw wells vs cleaned", "after (cleaned, 2,759 wells)", B)]:
        for rule, p in predictions(v).items():
            rows.append({"comparison": comp, "rule": rule, "data": label, **score(v, p)})
    t = pd.DataFrame(rows)
    out = t.copy()
    for c in ["RMSE_m", "MAE_m", "R2", "MedAE_m", "Bias_m", "sMAPE_pct"]:
        out[c] = out[c].round(3)
    tab = config.REPORTS_DIR / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    out.to_csv(tab / "table17_error_scores.csv", index=False)
    (tab / "table17_error_scores.md").write_text(out.to_markdown(index=False) + "\n")

    def pct(comp, rule, col):
        s = t[(t.comparison == comp) & (t.rule == rule)].set_index("data")[col]
        return 100 * (s.iloc[1] - s.iloc[0]) / abs(s.iloc[0])

    ra, rb = R[np.isfinite(R)], B[np.isfinite(B)]
    L = ["# Error scores before vs after cleaning\n",
         "_Generated by `scripts/15_error_scores.py`. Table: `reports/tables/table17_error_scores.csv`._\n",
         "Preprocessing makes no predictions, so these are **data-quality scores**: two fixed, past-only "
         "forecasting rules are applied identically before and after cleaning and scored on the 2018-2022 "
         "readings (depth in metres; lower is better except R2). They are not model results.\n",
         "- Rule 1, *same season last year*: the reading of the same season one year earlier.",
         f"- Rule 2, *seasonal mean of earlier years*: mean of all earlier same-season readings (at least "
         f"{MIN_PRIOR}), the pipeline's expanding baseline.\n",
         "## A. Same 2,759 wells: raw values vs cleaned (main result)\n",
         f"Only the values differ: cleaning blanked **{blanked:,}** raw readings (the dataset authors' "
         f"per-well 3-sigma rule) and changed **{changed}** others. Our own pipeline changes no values "
         "(it only adds flags), so it adds nothing to these scores.\n",
         out[out.comparison.str.startswith("A")].drop(columns="comparison").to_markdown(index=False)]
    for rule in ["same season last year", "seasonal mean of earlier years"]:
        L.append(f"\n- {rule}: RMSE {pct('A: same 2,759 wells', rule, 'RMSE_m'):+.1f}%, "
                 f"MAE {pct('A: same 2,759 wells', rule, 'MAE_m'):+.1f}%, R2 "
                 f"{t[(t.comparison.str.startswith('A')) & (t.rule == rule)].R2.diff().iloc[-1]:+.3f}.")
    L += ["\n## B. Whole raw table vs cleaned table (different wells)\n",
          out[out.comparison.str.startswith("B")].drop(columns="comparison").to_markdown(index=False),
          f"\nRaw values range {ra.min():g} to {ra.max():g} m ({int((ra < 0).sum()):,} negative readings, "
          f"std {ra.std():.2f} m); cleaned values range {rb.min():g} to {rb.max():g} m (std {rb.std():.2f} m). "
          "Errors fall sharply because wells with impossible values were removed. **R2 is not comparable "
          "here**: it is measured against each dataset's own variance, and the raw table's extreme values "
          "inflate that variance, so the raw R2 looks higher without the forecasts being better. Use "
          "comparison A for R2."]
    (config.REPORTS_DIR / "preprocessing_error_scores.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
