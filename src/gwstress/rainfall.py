"""Stage 4b: antecedent rainfall windows for each groundwater observation.

Rule: an observation in month M of year Y gets rainfall totals over the 1, 3
and 6 COMPLETE calendar months ending at M-1. Month M itself is NEVER used:
CGWB does not publish the day of the reading, so rain later in month M may
have fallen after the measurement (that would be future information).

How leakage is prevented, and proven:
1. Windows are built from an explicit "contributions" table: one row per
   (observation, source month) pair. It is the only path by which rainfall
   reaches a feature, so it can be audited.
2. assert_no_leakage() checks every contribution satisfies
       obs_month - window <= source_month <= obs_month - 1
   and is called inside rainfall_windows() before any sum is returned.
3. tests/test_rainfall.py feeds it deliberately leaky inputs (lag 0, i.e.
   month M) and checks that it raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (1, 3, 6)  # months; 6 is why CHIRPS starts at 1999-07


class LeakageError(AssertionError):
    """Raised when a rainfall month at/after the observation month is used."""


def month_index(year, month):
    """Months since year 0: lets 'M-1' cross year boundaries with plain integers."""
    return np.asarray(year) * 12 + np.asarray(month) - 1


def contributions(obs: pd.DataFrame, max_window: int = max(WINDOWS),
                  lags: range | None = None) -> pd.DataFrame:
    """Expand each observation to the source months that may feed its windows.

    `lags` is exposed ONLY so tests can inject a leaky configuration (e.g.
    lags=range(0, 6), which includes month M). Production code never passes it;
    the default is lags 1..max_window, i.e. months M-1 .. M-max_window.
    """
    lags = range(1, max_window + 1) if lags is None else lags
    t = month_index(obs["year"], obs["month"])
    parts = [pd.DataFrame({"obs_row": obs.index, "well_id": obs["well_id"].to_numpy(),
                           "obs_m": t, "lag": k, "src_m": t - k}) for k in lags]
    return pd.concat(parts, ignore_index=True)


def assert_no_leakage(contrib: pd.DataFrame, window: int) -> None:
    """Every source month must lie in [obs_m - window, obs_m - 1]."""
    late = contrib["src_m"] >= contrib["obs_m"]
    if late.any():
        raise LeakageError(f"{int(late.sum())} contributions use rainfall from the "
                           f"observation month or later")
    early = contrib["src_m"] < contrib["obs_m"] - window
    if early.any():
        raise LeakageError(f"{int(early.sum())} contributions fall outside the {window}-month window")


def rainfall_windows(obs: pd.DataFrame, rain: pd.DataFrame, windows=WINDOWS,
                     _lags: range | None = None) -> pd.DataFrame:
    """Return `obs` with rain_{w}m_mm columns.

    obs  : needs well_id, year, month (index is preserved).
    rain : CHIRPS table with well_id, year, month, precip_mm.
    A window total is NaN if ANY of its months is missing (NaN or absent);
    a partial sum would silently under-count rainfall.
    """
    r = rain.assign(src_m=month_index(rain["year"], rain["month"]))[["well_id", "src_m", "precip_mm"]]
    base = contributions(obs, max(windows), lags=_lags).merge(r, on=["well_id", "src_m"], how="left")
    out = obs.copy()
    lag_list = list(range(1, max(windows) + 1) if _lags is None else _lags)
    for w in windows:
        # The window uses the first w lags. In production these are 1..w, i.e.
        # months M-1 .. M-w. A buggy lag list (e.g. starting at 0 = month M)
        # flows through unchanged here and is caught by the assertion below,
        # which re-checks exactly the rows that are summed.
        c = base[base["lag"].isin(lag_list[:w])]
        assert_no_leakage(c, w)
        g = c.groupby("obs_row")["precip_mm"]
        total = g.sum(min_count=1)
        complete = g.count().eq(w) & g.size().eq(w)
        out[f"rain_{w}m_mm"] = total.where(complete).reindex(out.index)
    return out
