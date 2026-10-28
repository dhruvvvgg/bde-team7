"""Error metrics for the before/after-preprocessing comparison.

Preprocessing makes no predictions, so error metrics need something to
score. We apply the SAME fixed, past-only forecasting rules to the data
before and after cleaning and score them on the 2018-2022 test period. If
cleaning makes the data more reliable and more regular, the same rule scores
better. This measures data quality, not model skill.

Forecasting rules (each uses only EARLIER rounds of the same well):
  same season last year          the reading 4 rounds back (same season, Y-1)
  seasonal mean of earlier years mean of all earlier same-season readings,
                                 needing at least MIN_PRIOR of them (this is
                                 the pipeline's expanding baseline)
  previous round                 the reading of the immediately previous round
                                 (a deliberately weak rule: it ignores the
                                 strong seasonal cycle)

Metrics (e = prediction - actual, y = actual, so Bias > 0 means the rule
over-estimates depth):
  MSE    mean(e^2)
  RMSE   sqrt(MSE)
  MAE    mean(|e|)
  MAPE   100 * mean(|e| / |y|), only over readings with |y| >= MAPE_MIN_ACTUAL_M
         (percentage error explodes as y -> 0; shallow water at 0-1 m is
         common, so those readings are excluded and counted in MAPE_n)
  R2     1 - sum(e^2) / sum((y - mean(y))^2)
  NRMSE  RMSE / (max(y) - min(y)); also reported relative to mean(y)
  MedAE  median(|e|)
  Bias   mean(e), also called Mean Error (ME)
  sMAPE  100 * mean(2|e| / (|y| + |p|)); a bounded alternative to MAPE
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import clean, config, gw_io

RAW_FILE = config.GW_DIR / "Input" / "1_India_GWLs_2000_2024_wells_within_India.csv"
N_ROUNDS = 92                 # Jan/May/Aug/Nov x 2000..2022
TEST_FIRST_COL = 72           # column 72 = Jan-2018; test period = 2018-2022
MIN_PRIOR = 5                 # earlier same-season readings needed for rule 2
MAPE_MIN_ACTUAL_M = 1.0       # MAPE only where |actual| >= 1 m
SEASONS = ["Jan", "May", "Aug", "Nov"]
RULES = ["same season last year", "seasonal mean of earlier years", "previous round"]
PRIMARY_RULE = "same season last year"

METRIC_INFO = pd.DataFrame([
    ("RMSE", "Root Mean Square Error", "lower", "Square root of the average squared error; big misses weigh more."),
    ("MSE", "Mean Squared Error", "lower", "Average squared difference between actual and predicted values."),
    ("MAE", "Mean Absolute Error", "lower", "Average absolute difference between actual and predicted values."),
    ("MAPE", "Mean Absolute Percentage Error", "lower", "Prediction error as a percentage of the actual value."),
    ("R2", "Coefficient of Determination", "higher", "Share of the variation in actual values that the rule explains."),
    ("NRMSE", "Normalized Root Mean Square Error", "lower", "RMSE relative to the scale (range) of the target data."),
    ("MedAE", "Median Absolute Error", "lower", "Median absolute error; insensitive to extreme errors."),
    ("Bias", "Mean Error (ME)", "closer to 0", "Whether predictions systematically over- or under-estimate."),
], columns=["Metric", "Full form", "Better when", "Purpose"])

# Columns of the metric tables, in the order of the reference table.
METRIC_COLS = ["n", "RMSE", "MSE", "MAE", "MAPE", "R2", "NRMSE", "MedAE", "Bias"]
EXTRA_COLS = ["NRMSE_mean", "sMAPE", "MAPE_n"]


@dataclass
class Dataset:
    """Depth readings as a (wells x 92 rounds) matrix, columns Jan-00 .. Nov-22."""
    name: str
    well_ids: np.ndarray
    states: np.ndarray
    values: np.ndarray

    def describe(self) -> pd.DataFrame:
        v = self.values[np.isfinite(self.values)]
        return pd.DataFrame({"value": {
            "wells": len(self.values), "readings": int(v.size),
            "missing % of cells": round(100 * (1 - v.size / self.values.size), 2),
            "min (m)": v.min(), "max (m)": v.max(), "mean (m)": v.mean(), "median (m)": np.median(v),
            "std (m)": v.std(), "negative readings": int((v < 0).sum()), "zero readings": int((v == 0).sum()),
            "readings > 100 m": int((v > 100).sum())}})


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _obs_columns(df: pd.DataFrame) -> list[str]:
    _, obs, _ = gw_io.split_columns(df.drop(columns="well_id"))
    obs = [c for c in obs if int(c[-2:]) <= 22]
    assert len(obs) == N_ROUNDS and obs[0] == "Jan-00" and obs[-1] == "Nov-22", "unexpected round layout"
    assert obs[TEST_FIRST_COL] == "Jan-18"
    return obs


@functools.lru_cache(maxsize=1)
def _raw_frame() -> tuple[pd.DataFrame, list[str]]:
    raw = gw_io.load_wide(RAW_FILE, strict=False)
    return raw, _obs_columns(raw)


def _states(series: pd.Series) -> np.ndarray:
    return clean.standardise_name(series).to_numpy()


def load_raw_all() -> Dataset:
    """BEFORE: the whole raw CGWB table (every well row, no quality control)."""
    raw, obs = _raw_frame()
    v = raw[obs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return Dataset("raw, all wells", raw.well_id.to_numpy(), _states(raw["State"]), v)


def load_cleaned() -> Dataset:
    """AFTER: the authors' refined table that the pipeline uses (2,759 wells)."""
    ref = gw_io.load_wide(config.GW_REFINED_FILE)
    obs = _obs_columns(ref)
    v = ref[obs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return Dataset("cleaned, 2,759 wells", ref.well_id.to_numpy(), _states(ref["State"]), v)


def load_raw_same_wells() -> Dataset:
    """BEFORE, like-for-like: RAW values of exactly the 2,759 wells kept after cleaning.

    Only the values differ from the cleaned set, so this isolates the effect of
    value-level cleaning. One well appears twice in the raw table (a sparse
    twin); the row with more readings is used.
    """
    raw, obs = _raw_frame()
    keep = gw_io.load_wide(config.GW_REFINED_FILE).well_id
    cand = raw[raw.well_id.isin(keep)].copy()
    cand["_n"] = cand[obs].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    cand = cand.sort_values("_n").drop_duplicates("well_id", keep="last").set_index("well_id").loc[keep]
    v = cand[obs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return Dataset("raw values, same 2,759 wells", keep.to_numpy(), _states(cand["State"]), v)


# --------------------------------------------------------------------------
# Forecasting rules (past-only)
# --------------------------------------------------------------------------
def predictions(v: np.ndarray, min_prior: int = MIN_PRIOR) -> dict[str, np.ndarray]:
    """Return one prediction matrix per rule, same shape as `v`.

    LEAKAGE: column j of every matrix is built from columns < j only.
    """
    same = np.full_like(v, np.nan)
    same[:, 4:] = v[:, :-4]
    prev = np.full_like(v, np.nan)
    prev[:, 1:] = v[:, :-1]
    mean = np.full_like(v, np.nan)
    for s in range(4):
        cols = list(range(s, v.shape[1], 4))
        x = v[:, cols]
        ok = np.isfinite(x)
        total, count = np.cumsum(np.where(ok, x, 0.0), 1), np.cumsum(ok, 1)
        n_before = np.concatenate([np.zeros((len(x), 1)), count[:, :-1]], 1)   # shifted: excludes the current round
        s_before = np.concatenate([np.zeros((len(x), 1)), total[:, :-1]], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[:, cols] = np.where(n_before >= min_prior, s_before / n_before, np.nan)
    return {"same season last year": same, "seasonal mean of earlier years": mean, "previous round": prev}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray, mape_min: float = MAPE_MIN_ACTUAL_M) -> dict:
    """All metrics over the pairs where both actual and prediction exist."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {k: np.nan for k in METRIC_COLS + EXTRA_COLS} | {"n": 0}
    e = p - y
    mse = float((e ** 2).mean())
    rmse = float(np.sqrt(mse))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    big = np.abs(y) >= mape_min
    rng = float(y.max() - y.min())
    mean_y = float(y.mean())
    return {
        "n": int(len(y)), "RMSE": rmse, "MSE": mse, "MAE": float(np.abs(e).mean()),
        "MAPE": float(100 * (np.abs(e[big]) / np.abs(y[big])).mean()) if big.any() else np.nan,
        "R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        "NRMSE": rmse / rng if rng > 0 else np.nan,
        "MedAE": float(np.median(np.abs(e))), "Bias": float(e.mean()),
        "NRMSE_mean": rmse / mean_y if mean_y != 0 else np.nan,
        "sMAPE": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-12))),
        "MAPE_n": int(big.sum()),
    }


def test_pairs(ds: Dataset, rule: str, first_col: int = TEST_FIRST_COL):
    """(actual, predicted, column index) arrays for the test period."""
    p = predictions(ds.values)[rule]
    cols = np.arange(first_col, ds.values.shape[1])
    y, q = ds.values[:, first_col:], p[:, first_col:]
    ok = np.isfinite(y) & np.isfinite(q)
    rows, cc = np.nonzero(ok)
    return y[rows, cc], q[rows, cc], cols[cc], rows


def metric_table(ds: Dataset, rules=RULES) -> pd.DataFrame:
    """One row per rule, all metrics, test period 2018-2022."""
    out = []
    for r in rules:
        y, q, _, _ = test_pairs(ds, r)
        out.append({"data": ds.name, "rule": r, **metrics(y, q)})
    return pd.DataFrame(out)


def breakdown(ds: Dataset, rule: str, by: str) -> pd.DataFrame:
    """Metrics split by 'season', 'year' or 'state' for one rule."""
    y, q, cols, rows = test_pairs(ds, rule)
    if by == "season":
        keys = np.array(SEASONS)[cols % 4]
    elif by == "year":
        keys = 2000 + cols // 4
    elif by == "state":
        keys = ds.states[rows]
    else:
        raise ValueError(by)
    res = []
    for k in pd.unique(keys):
        m = keys == k
        res.append({by: k, "wells": len(np.unique(rows[m])), **metrics(y[m], q[m])})
    out = pd.DataFrame(res)
    if by == "season":
        out[by] = pd.Categorical(out[by], SEASONS, ordered=True)
    return out.sort_values(by).reset_index(drop=True)


def improvement(before: pd.Series, after: pd.Series, metric: str) -> float:
    """Signed improvement in percent (positive = after is better).

    For R2 the absolute change is returned instead (a ratio of a ratio misleads).
    For Bias, improvement is the reduction in |bias|.
    """
    if metric == "R2":
        return float(after - before)
    b, a = (abs(before), abs(after)) if metric == "Bias" else (before, after)
    return float(100 * (b - a) / abs(b)) if b else np.nan
