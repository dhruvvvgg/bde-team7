"""Stage 5: time-aware feature engineering.

Every feature for the row (well w, round t) is computed ONLY from rows of
well w strictly before t (plus the reading at t itself where the feature is a
transformation of the current reading, e.g. the anomaly). The general
guarantee is tested in tests/test_features.py by perturbing future readings
and checking that no earlier feature changes.

Seasons are handled separately throughout. WHY: Indian groundwater has a
strong monsoon cycle (May is typically deepest, Nov shallowest); comparing a
May reading with a mixed-season baseline would label every May as "stress".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_MAIN = 5                 # checkpoint-1 decision: main minimum-history rule
K_SENSITIVITY = (3, 5, 8)  # reported in data_quality.md

# Std guard. CGWB manual tape readings are recorded to 0.01 m; a prior std
# below 0.10 m means a nearly constant record, where z = diff / tiny_std would
# explode (e.g. 0.3 m change / 0.01 m std = 30 SD). We floor the std at 0.10 m
# rather than dropping the row: the reading is real, only its scale is unknown.
STD_FLOOR_M = 0.10

TREND_YEARS = 5            # look-back for Theil-Sen trend (same season)
TREND_MIN_POINTS = 4       # need >= 4 of the 5 prior readings

RAIN_HIST_MIN_YEARS = 5    # prior same-season years needed for a rainfall normal
RAIN_PCT_MIN_MM = 5.0      # below this normal, % anomaly is unstable -> NaN

GROUP = ["well_id", "season"]


def _prior_expanding_stats(x: pd.Series) -> pd.DataFrame:
    """Expanding count/mean/std of PRIOR non-null values (current row excluded).

    LEAKAGE GUARD: the cumulative sums are shifted by one row, so row i only
    sees rows 0..i-1. Rows must be in chronological order within the group.
    """
    v = x.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    n = np.concatenate([[0], np.cumsum(ok)[:-1]])
    s1 = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))[:-1]])
    s2 = np.concatenate([[0.0], np.cumsum(np.where(ok, v * v, 0.0))[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        var = np.where(n > 1, (s2 - n * mean ** 2) / (n - 1), np.nan)
    std = np.sqrt(np.clip(var, 0, None))  # clip tiny negative float error
    return pd.DataFrame({"n": n, "mean": mean, "std": std}, index=x.index)


def seasonal_baseline(df: pd.DataFrame, k: int = K_MAIN) -> pd.DataFrame:
    """Add n_prior_same_season, baseline_mean/std, anomaly_z (K-gated).

    Wells with no May series (444: all Kerala and West Bengal, most of
    Odisha and Assam) simply never reach K in that season, so their May rows
    get NaN baseline/anomaly ("Insufficient history" in stage 6) while their
    other three seasons are unaffected. No well is dropped.
    """
    df = df.sort_values(GROUP + ["year"]).copy()
    st = df.groupby(GROUP, observed=True)["gwl_m_bgl"].apply(_prior_expanding_stats)
    st = st.reset_index(level=[0, 1], drop=True).reindex(df.index)
    df["n_prior_same_season"] = st["n"].astype(int)
    enough = st["n"] >= k  # MINIMUM HISTORY RULE: never a z-score from < k readings
    df["baseline_mean_m"] = st["mean"].where(enough)
    df["baseline_std_m"] = st["std"].where(enough)
    df["flag_std_floored"] = enough & (st["std"] < STD_FLOOR_M)
    std_eff = df["baseline_std_m"].clip(lower=STD_FLOOR_M)
    # Positive anomaly = deeper than usual for this season = more stressed.
    df["anomaly_z"] = (df["gwl_m_bgl"] - df["baseline_mean_m"]) / std_eff
    return df


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    i, j = np.triu_indices(len(x), 1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def seasonal_trend(df: pd.DataFrame, years: int = TREND_YEARS,
                   min_points: int = TREND_MIN_POINTS) -> pd.Series:
    """Theil-Sen slope (m/yr) of same-season readings in years [Y-years, Y-1].

    WHY same season rather than the last N rounds: consecutive rounds mix
    May (deep) and Nov (shallow), so a slope over them mostly measures the
    seasonal cycle, not the multi-year trend. WHY Theil-Sen: robust to the
    single bad readings that survive QC (see flag_extreme). LEAKAGE GUARD: the
    window ends at Y-1; the current reading is excluded.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(GROUP, observed=True):
        yrs, vals, idx = g["year"].to_numpy(), g["gwl_m_bgl"].to_numpy(float), g.index
        for pos, y in enumerate(yrs):
            m = (yrs >= y - years) & (yrs < y) & ~np.isnan(vals)
            if m.sum() >= min_points:
                out[idx[pos]] = _theil_sen(yrs[m].astype(float), vals[m])
    return out


def rainfall_anomalies(df: pd.DataFrame, windows=(1, 3, 6)) -> pd.DataFrame:
    """Current window total vs the same well's normal for that window+season,
    using PAST YEARS ONLY (expanding mean over prior years, shifted by one).

    Computed on every round, including rounds with no groundwater reading,
    because rainfall exists regardless and the normal should use all years.
    """
    df = df.sort_values(GROUP + ["year"]).copy()
    for w in windows:
        col = f"rain_{w}m_mm"
        st = df.groupby(GROUP, observed=True)[col].apply(_prior_expanding_stats)
        st = st.reset_index(level=[0, 1], drop=True).reindex(df.index)
        normal = st["mean"].where(st["n"] >= RAIN_HIST_MIN_YEARS)
        df[f"rain_{w}m_normal_mm"] = normal
        df[f"rain_{w}m_dev_mm"] = df[col] - normal
        df[f"rain_{w}m_dev_pct"] = (100 * (df[col] - normal) / normal).where(normal >= RAIN_PCT_MIN_MM)
    return df


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Two completeness indicators.

    completeness_to_date: share of the well's rounds BEFORE this one that have
        a reading. Time-aware: safe as a model input.
    well_completeness_full_period: share over all 92 rounds (2000-2022).
        LEAKAGE WARNING: uses future rounds. Descriptive / filtering use only;
        do NOT feed it to a model predicting a historical round.
    """
    df = df.sort_values(["well_id", "date"]).copy()
    has = df["gwl_m_bgl"].notna().astype(int)
    g = has.groupby(df["well_id"])
    prior_obs = g.cumsum() - has
    prior_rounds = g.cumcount()
    df["completeness_to_date"] = (prior_obs / prior_rounds.replace(0, np.nan))
    df["well_completeness_full_period"] = g.transform("mean")
    return df


def build_features(df: pd.DataFrame, k: int = K_MAIN) -> pd.DataFrame:
    df = seasonal_baseline(df, k)
    df["trend_5y_m_per_yr"] = seasonal_trend(df)
    df = rainfall_anomalies(df)
    df = completeness(df)
    return df.sort_values(["well_id", "date"]).reset_index(drop=True)


def k_sensitivity(df: pd.DataFrame, ks=K_SENSITIVITY) -> pd.DataFrame:
    """Readings that would get a baseline under each K (computed on n_prior)."""
    r = df[df["gwl_m_bgl"].notna()]
    rows = []
    for k in ks:
        ok = r["n_prior_same_season"] >= k
        rows.append({"K": k, "readings": len(r), "with_baseline": int(ok.sum()),
                     "excluded": int((~ok).sum()), "excluded_pct": round(100 * (~ok).mean(), 2),
                     "wells_with_any_baseline": r[ok].well_id.nunique()})
    return pd.DataFrame(rows)
