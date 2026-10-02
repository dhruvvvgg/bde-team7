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

# Models should use anomaly_z_capped. The raw z has a heavy tail (|z| > 5 in
# ~1% of rows, range about -56..+39) driven by level shifts and single bad
# readings; a few such rows would dominate a squared-error loss. Capping at
# +/-5 keeps the sign and "very extreme" information. Stress categories are
# unaffected because every threshold is below 5.
Z_CAP = 5.0

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
    df["anomaly_z_capped"] = df["anomaly_z"].clip(-Z_CAP, Z_CAP)
    # Grouping helper for the 444 wells with no May series. LEAKAGE WARNING:
    # it looks at the whole 2000-2022 record, so it is descriptive only
    # (for grouping / reporting), never a model input.
    has_any = df.groupby(GROUP, observed=True)["gwl_m_bgl"].transform(lambda s: s.notna().any())
    df["well_has_no_series_for_season"] = ~has_any.astype(bool)
    return df


ROLL_WINDOW = 10   # Part B: rolling baseline uses the last 10 earlier same-season readings
ROLL_MIN = 5       # ... and needs at least 5 of them (same K as the expanding baseline)


def rolling_baseline(df: pd.DataFrame, window: int = ROLL_WINDOW,
                     min_n: int = ROLL_MIN) -> pd.DataFrame:
    """Part B: stationary variant of the seasonal baseline.

    For round t of a well-season, the baseline is the mean and std (ddof=1) of
    the last `window` NON-MISSING readings of that well-season strictly before
    t. That is the most recent 10 readings, which can span more than 10 years
    if rounds were missed. It needs at least `min_n` of them, otherwise NaN.
    The std floor (STD_FLOOR_M) and the cap (Z_CAP) are the same as for the
    expanding baseline.

    WHY: the expanding mean averages over all years since 2000. In a well that
    is steadily declining it lags further and further behind, so the late
    years look anomalous just because of the trend (High Stress is 6.4% of
    classified rows, against 2.3% for a standard normal). A 10-reading window
    compares each reading with the recent decade instead.
    Uses only readings of the same well and season from rounds before t.
    """
    df = df.sort_values(GROUP + ["year"]).copy()
    n = np.zeros(len(df), dtype=int)
    mean = np.full(len(df), np.nan)
    std = np.full(len(df), np.nan)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby(GROUP, observed=True, sort=False):
        v = g["gwl_m_bgl"].to_numpy(float)
        hist: list[float] = []
        for ix, x in zip(g.index, v):
            prior = hist[-window:]            # LEAKAGE GUARD: built before x is appended
            k = pos[ix]
            n[k] = len(prior)
            if len(prior) >= min_n:
                mean[k] = np.mean(prior)
                std[k] = np.std(prior, ddof=1)
            if not np.isnan(x):
                hist.append(x)
    df["n_roll10"] = n
    df["baseline_roll10_mean_m"] = mean
    df["baseline_roll10_std_m"] = std
    df["flag_roll10_std_floored"] = (~np.isnan(std)) & (std < STD_FLOOR_M)
    eff = df["baseline_roll10_std_m"].clip(lower=STD_FLOOR_M)
    df["anomaly_roll10"] = (df["gwl_m_bgl"] - df["baseline_roll10_mean_m"]) / eff
    df["anomaly_roll10_capped"] = df["anomaly_roll10"].clip(-Z_CAP, Z_CAP)
    return df


def tadj_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Part B2: small-sample-adjusted anomaly (NEW columns; anomaly_z is unchanged).

    For a reading x with n >= K earlier same-season readings (the same expanding
    window, the same K = 5 gate and the same 0.10 m floor on s as anomaly_z):
        t     = (x - mean_earlier) / (s_eff * sqrt(1 + 1/n))
        z_adj = Phi^-1( F_t(t; df = n - 1) )
    where s is the sample std with ddof = 1 and s_eff = max(s, STD_FLOOR_M).

    How this differs from anomaly_z: anomaly_z ALSO uses s with ddof = 1
    (_prior_expanding_stats divides by n - 1), but it treats
    (x - mean) / s as standard normal. That ignores two small-sample effects:
      1. the earlier mean is itself estimated, so a NEW reading's prediction
         error has variance sigma^2 (1 + 1/n): hence the sqrt(1 + 1/n);
      2. s is estimated from n readings, so the standardised value follows a
         Student t with n - 1 degrees of freedom, whose tails are much heavier
         than normal for n = 5..10.
    Under i.i.d. Gaussian readings, t is exactly t_{n-1}. Mapping it through
    F_t then Phi^-1 gives a score with the nominal normal tail rates at every
    n, so thresholds such as z >= 2 mean the same thing at n = 5 and n = 22.

    Leakage: uses the same past-only baseline columns as anomaly_z plus the
    current reading; n_prior_same_season counts earlier readings only.
    Numerics: the upper tail goes through sf/isf (and the lower tail through
    cdf/ppf), so extreme t do not round to +/-inf. Values are finite.
    """
    from scipy import stats

    df = df.copy()
    n = df["n_prior_same_season"].to_numpy(float)
    s_eff = df["baseline_std_m"].clip(lower=STD_FLOOR_M).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):   # n = 0 rows are NaN anyway (no baseline)
        t = (df["gwl_m_bgl"].to_numpy(float) - df["baseline_mean_m"].to_numpy(float)) / (s_eff * np.sqrt(1 + 1 / n))
    dof = n - 1
    z = np.full(len(df), np.nan)
    ok = ~np.isnan(t)
    up = ok & (t >= 0)
    lo = ok & (t < 0)
    z[up] = stats.norm.isf(stats.t.sf(t[up], dof[up]))
    z[lo] = stats.norm.ppf(stats.t.cdf(t[lo], dof[lo]))
    # Underflow at astronomically large |t| would give inf; such values are
    # beyond any threshold anyway, so clip to a finite bound before capping.
    z = np.clip(z, -38.0, 38.0)
    df["anomaly_tadj"] = z
    df["anomaly_tadj_capped"] = np.clip(z, -Z_CAP, Z_CAP)
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


def build_features(df: pd.DataFrame, k: int = K_MAIN, rain: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage-5 features, plus the Part-A extras when the monthly CHIRPS table `rain` is given."""
    df = seasonal_baseline(df, k)
    df = rolling_baseline(df)
    df = tadj_anomaly(df)
    df["trend_5y_m_per_yr"] = seasonal_trend(df)
    df = rainfall_anomalies(df)
    df = completeness(df)
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)
    if rain is not None:
        from . import features_extra
        df = features_extra.add_all(df, rain)
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
