"""Part A: extra leakage-safe features, added to the main table.

The input is the stage-5 table: one row per (well, round), with ALL 92 rounds
present, including rounds with no reading. Because the grid is complete and
sorted, "previous round" means the previous row of the well, and "same season
one year earlier" means the previous row of the (well, season) group.

LEAKAGE RULE used everywhere in this module: a feature for round t (month M of
year Y) may use
  * groundwater readings from rounds strictly before t, and
  * rainfall from calendar months <= M-1.
The only exception is `gw_delta_from_prev_round_m`, which uses the CURRENT
reading. It is a target candidate (class C in the handoff), never an input.

Rainfall features take the monthly CHIRPS table (well_id, year, month,
precip_mm) and go through `_month_sum()`. That function asserts the last month
summed is before the observation month. The assertion runs on every row, not
just in tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rainfall import LeakageError, month_index

MONSOON_START_MONTH = 6   # June: start of the Indian SW-monsoon and hydrological year
MONSOON_END_MONTH = 9     # September: end of the SW-monsoon (Jun-Sep, IMD convention)
RAIN_Z_MIN_YEARS = 5      # prior years needed for a rainfall z-score / monsoon normal
RAIN_Z_MIN_STD_MM = 1.0   # below this the prior spread is ~0 (dry months): z -> NaN
WY_BASE = 1999            # water_year_index 0 = water year Jun-1999..May-2000


# --------------------------------------------------------------------------
# Groundwater history
# --------------------------------------------------------------------------
def groundwater_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-only groundwater history columns.

    gw_prev_round_m             reading at round t-1 (the immediately previous
                                CGWB round, about 3-4 months earlier). NaN if that
                                round was not measured. Uses round t-1 only.
    gw_last_obs_m               most recent non-missing reading before t.
                                Uses rounds < t.
    rounds_since_last_obs       how many rounds back that reading is (1 = previous
                                round). Uses rounds < t.
    gw_same_season_lag1y_m      same season, year Y-1 (round t-4).
    gw_same_season_lag2y_m      same season, year Y-2 (round t-8).
    gw_prev_change_m            reading(t-1) - reading(t-2): the most recent
                                round-to-round change KNOWN at t. Uses t-1, t-2.
    gw_delta_from_prev_round_m  reading(t) - reading(t-1). USES THE CURRENT
                                READING, so it is a target candidate, NOT an input.
    gw_max_to_date_m            deepest reading in rounds < t (expanding).
    gw_min_to_date_m            shallowest reading in rounds < t (expanding).
    n_readings_to_date          number of non-missing readings in rounds < t.

    WHY these are safe: every lag is a shift by >= 1 row within a well whose
    rows are a complete, date-sorted grid. Expanding statistics are shifted by
    one row, so they never see the current reading.
    """
    df = df.sort_values(["well_id", "date"]).copy()
    g = df.groupby("well_id", sort=False)["gwl_m_bgl"]
    df["gw_prev_round_m"] = g.shift(1)
    df["gw_prev_change_m"] = g.shift(1) - g.shift(2)
    df["gw_delta_from_prev_round_m"] = df["gwl_m_bgl"] - df["gw_prev_round_m"]  # class C
    # Same season = 4 rounds back on the complete grid (asserted below).
    df["gw_same_season_lag1y_m"] = g.shift(4)
    df["gw_same_season_lag2y_m"] = g.shift(8)
    prev = g.shift(1)
    pg = prev.groupby(df["well_id"], sort=False)
    df["gw_max_to_date_m"] = pg.cummax()
    df["gw_min_to_date_m"] = pg.cummin()
    df["n_readings_to_date"] = prev.notna().groupby(df["well_id"], sort=False).cumsum().astype(int)
    # Last observation carried forward from the PREVIOUS row (never the current one).
    df["gw_last_obs_m"] = pg.ffill()
    pos = df.groupby("well_id", sort=False).cumcount()
    last_pos = pd.Series(np.where(df["gwl_m_bgl"].notna(), pos, np.nan), index=df.index)
    last_pos_prev = last_pos.groupby(df["well_id"], sort=False).shift(1)
    last_pos_prev = last_pos_prev.groupby(df["well_id"], sort=False).ffill()
    df["rounds_since_last_obs"] = pos - last_pos_prev

    # Check the grid assumption behind the "4 rounds back = same season" rule.
    s4 = df.groupby("well_id", sort=False)["season"].shift(4)
    ok = s4.isna() | (s4.astype(str) == df["season"].astype(str))
    assert ok.all(), "round grid is not complete; same-season lag by shift(4) is invalid"
    return df


def prior_year_fluctuation(df: pd.DataFrame) -> pd.Series:
    """May-to-November depth change in the most recent COMPLETED monsoon year.

    For a round in year Y, that year is Y-1 in every season:
      Jan/May/Aug of Y: the Nov reading of Y has not happened yet.
      Nov of Y: the Nov reading of Y is the current round, so it is excluded.
    Value = gw(Nov, Y-1) - gw(May, Y-1). Negative means the water table rose
    over the monsoon (recharge). NaN if either reading is missing. This is
    always NaN for the 444 wells with no May series.
    Uses rounds May(Y-1) and Nov(Y-1), which are always strictly before t.
    """
    piv = df.pivot_table(index=["well_id", "year"], columns="month", values="gwl_m_bgl",
                         aggfunc="first", dropna=False)
    fluct = (piv.get(11) - piv.get(5)).rename("fluct")
    key = pd.MultiIndex.from_arrays([df["well_id"], df["year"] - 1])
    return pd.Series(fluct.reindex(key).to_numpy(), index=df.index)


# --------------------------------------------------------------------------
# Rainfall
# --------------------------------------------------------------------------
def _monthly_cumsum(rain: pd.DataFrame) -> dict:
    """Per-well cumulative rainfall and cumulative count over a month index."""
    r = rain.assign(m=month_index(rain["year"], rain["month"])).sort_values(["well_id", "m"])
    out = {}
    for wid, g in r.groupby("well_id", sort=False):
        m = g["m"].to_numpy()
        assert (np.diff(m) == 1).all(), f"gap in CHIRPS months for {wid}"
        v = g["precip_mm"].to_numpy(float)
        out[wid] = (int(m[0]), np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))]),
                    np.concatenate([[0], np.cumsum(~np.isnan(v))]))
    return out


def _month_sum(df: pd.DataFrame, cums: dict, start_m: np.ndarray, end_m: np.ndarray) -> np.ndarray:
    """Sum of monthly rainfall over the inclusive month-index range [start_m, end_m].

    NaN if any month in the range is missing or outside the CHIRPS record.
    LEAKAGE GUARD: raises LeakageError if any end_m >= the observation month.
    """
    obs_m = month_index(df["year"], df["month"])
    if (end_m >= obs_m).any():
        raise LeakageError(f"{int((end_m >= obs_m).sum())} rainfall sums reach the observation month")
    out = np.full(len(df), np.nan)
    wids = df["well_id"].to_numpy()
    for i in range(len(df)):
        first, cs, cn = cums[wids[i]]
        a, b = int(start_m[i]) - first, int(end_m[i]) - first   # 0-based positions
        n_exp = b - a + 1
        if n_exp <= 0 or a < 0 or b >= len(cs) - 1:
            continue
        if cn[b + 1] - cn[a] == n_exp:
            out[i] = cs[b + 1] - cs[a]
    return out


def _prior_z(df: pd.DataFrame, col: str, by: list[str], min_n: int, min_std: float):
    """Expanding mean/std of `col` over EARLIER years in the same group (shift 1)."""
    from .features import _prior_expanding_stats
    d = df.sort_values(by + ["year"])
    st = d.groupby(by, observed=True)[col].apply(_prior_expanding_stats)
    st = st.reset_index(level=list(range(len(by))), drop=True).reindex(df.index)
    mean = st["mean"].where(st["n"] >= min_n)
    std = st["std"].where(st["n"] >= min_n)
    z = ((df[col] - mean) / std).where(std >= min_std)
    return mean, z


def rainfall_extra(df: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    """Add extra past-only rainfall features.

    rain_12m_mm                 total over months M-12 .. M-1. NaN before
                                Jul-2000, because CHIRPS here starts in Jul-1999.
    rain_last_monsoon_mm        total over Jun-Sep of the most recent monsoon that
                                ended before month M: year Y if M >= 10, else Y-1.
                                So it is Y for Nov rounds, and Y-1 for Jan, May
                                and Aug rounds. The Aug round sits mid-monsoon, so
                                the current monsoon is not complete.
    rain_last_monsoon_dev_mm    that total minus the well's mean monsoon total over
    rain_last_monsoon_dev_pct   EARLIER monsoons only (at least 5).
    rain_3m_z, rain_6m_z        (window total - mean) / std over the same well,
                                season and window in EARLIER years only (at least 5).
                                NaN if the prior std is below 1 mm (dry-season
                                windows where every year is about 0).
    rain_3m_z_capped, rain_6m_z_capped
                                the same z-scores clipped to +/-5. Use these in
                                models; the raw z is heavy-tailed.
    rain_monsoon_ytd_mm         rainfall since the start of the current monsoon
                                (hydrological) year, up to the month before the
                                round. The monsoon year starts on 1 June. For a
                                round in month M of year Y, the start is June of
                                Y if M >= 6, else June of Y-1, and the end is M-1:
                                  Jan Y: Jun..Dec of Y-1 (7 months)
                                  May Y: Jun Y-1 .. Apr Y (11 months)
                                  Aug Y: Jun..Jul Y (2 months)
                                  Nov Y: Jun..Oct Y (5 months)
    """
    df = df.copy()
    cums = _monthly_cumsum(rain)
    obs_m = month_index(df["year"], df["month"])
    y, m = df["year"].to_numpy(), df["month"].to_numpy()

    df["rain_12m_mm"] = _month_sum(df, cums, obs_m - 12, obs_m - 1)

    mon_year = np.where(m >= MONSOON_END_MONTH + 1, y, y - 1)
    df["last_monsoon_year"] = mon_year
    df["rain_last_monsoon_mm"] = _month_sum(df, cums, month_index(mon_year, MONSOON_START_MONTH),
                                            month_index(mon_year, MONSOON_END_MONTH))
    # The monsoon normal is per well over earlier MONSOONS (not earlier rounds).
    # Build one value per (well, monsoon year), take an expanding prior mean,
    # then map it back to the rows.
    mt = (df[["well_id", "last_monsoon_year", "rain_last_monsoon_mm"]]
          .drop_duplicates(["well_id", "last_monsoon_year"])
          .rename(columns={"last_monsoon_year": "year"}).reset_index(drop=True))
    mt["season"] = "x"
    normal, _ = _prior_z(mt, "rain_last_monsoon_mm", ["well_id"], RAIN_Z_MIN_YEARS, 0.0)
    mt["normal"] = normal.to_numpy()
    key = pd.MultiIndex.from_arrays([mt["well_id"], mt["year"]])
    nmap = pd.Series(mt["normal"].to_numpy(), index=key)
    norm_rows = nmap.reindex(pd.MultiIndex.from_arrays([df["well_id"], df["last_monsoon_year"]])).to_numpy()
    df["rain_last_monsoon_dev_mm"] = df["rain_last_monsoon_mm"] - norm_rows
    df["rain_last_monsoon_dev_pct"] = 100 * df["rain_last_monsoon_dev_mm"] / norm_rows

    for w in (3, 6):
        _, z = _prior_z(df, f"rain_{w}m_mm", ["well_id", "season"], RAIN_Z_MIN_YEARS, RAIN_Z_MIN_STD_MM)
        df[f"rain_{w}m_z"] = z
        # Rainfall is right-skewed: a single wet pre-monsoon month in a dry
        # region gives z up to ~100. The capped version (+/-5) is the one to
        # use in models, matching anomaly_z_capped.
        df[f"rain_{w}m_z_capped"] = z.clip(-5, 5)

    df = northeast_monsoon(df, cums)

    start_year = np.where(m >= MONSOON_START_MONTH, y, y - 1)
    df["rain_monsoon_ytd_mm"] = _month_sum(df, cums, month_index(start_year, MONSOON_START_MONTH), obs_m - 1)
    return df


NE_START_MONTH = 10  # October: start of the north-east (retreating) monsoon
NE_END_MONTH = 12    # December


def northeast_monsoon(df: pd.DataFrame, cums: dict) -> pd.DataFrame:
    """Part A2: north-east monsoon (Oct-Dec) rainfall, past-only.

    rain_last_ne_mm          total over October, November and December of year
                             Y-1, for every round in year Y. WHY Y-1 in every
                             season: Oct-Dec of year Y is only complete after
                             December of Y, which is after every round of Y
                             (Jan, May, Aug, Nov). For Nov Y, October of Y is
                             already in the past, but November and December are
                             not, so Y-1 is the last COMPLETE season. Months
                             used: Oct(Y-1), Nov(Y-1), Dec(Y-1); the last of
                             these, Dec(Y-1), is always before the round.
    rain_last_ne_dev_mm      that total minus the well's mean Oct-Dec total over
                             EARLIER years only (ne years < Y-1, at least 5).
    rain_last_ne_z_capped    (total - earlier mean) / earlier std, clipped to
                             +/-5; NaN if fewer than 5 earlier years or the
                             earlier std is < 1 mm.
    WHY: Tamil Nadu and coastal Andhra Pradesh get much of their rain in
    Oct-Dec, which the Jun-Sep features miss.
    Leakage guard: the same _month_sum() assertion (last month summed < the
    observation month) runs on every row.
    """
    df = df.copy()
    ne_year = df["year"].to_numpy() - 1
    df["last_ne_year"] = ne_year
    df["rain_last_ne_mm"] = _month_sum(df, cums, month_index(ne_year, NE_START_MONTH),
                                       month_index(ne_year, NE_END_MONTH))
    # Earlier-years normal: one value per (well, ne_year), expanding over prior ne years.
    nt = (df[["well_id", "last_ne_year", "rain_last_ne_mm"]]
          .drop_duplicates(["well_id", "last_ne_year"])
          .rename(columns={"last_ne_year": "year"}).reset_index(drop=True))
    mean, z = _prior_z(nt, "rain_last_ne_mm", ["well_id"], RAIN_Z_MIN_YEARS, RAIN_Z_MIN_STD_MM)
    nt["mean"], nt["z"] = mean.to_numpy(), z.to_numpy()
    key = pd.MultiIndex.from_arrays([df["well_id"], df["last_ne_year"]])
    idx = pd.MultiIndex.from_arrays([nt["well_id"], nt["year"]])
    df["rain_last_ne_dev_mm"] = df["rain_last_ne_mm"] - pd.Series(nt["mean"].to_numpy(), idx).reindex(key).to_numpy()
    df["rain_last_ne_z_capped"] = np.clip(pd.Series(nt["z"].to_numpy(), idx).reindex(key).to_numpy(), -5, 5)
    return df


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
def calendar(df: pd.DataFrame) -> pd.DataFrame:
    """water_year: start year of the Jun-May hydrological year containing the round.
    water_year_index: water_year - 1999 (0 = Jun-1999..May-2000).
    season_order: position of the round within its water year (Aug=0, Nov=1,
    Jan=2, May=3), so the order follows the hydrological cycle.
    These are known from the date alone, so there is no leakage.
    """
    df = df.copy()
    df["water_year"] = np.where(df["month"] >= MONSOON_START_MONTH, df["year"], df["year"] - 1)
    df["water_year_index"] = df["water_year"] - WY_BASE
    df["season_order"] = df["month"].map({8: 0, 11: 1, 1: 2, 5: 3}).astype(int)
    return df


def add_all(df: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    df = groundwater_history(df)
    df["gw_fluct_prev_wy_m"] = prior_year_fluctuation(df)
    df = rainfall_extra(df, rain)
    return calendar(df)
