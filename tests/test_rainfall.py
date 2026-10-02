"""Leakage tests for antecedent rainfall windows (stage 4).

The synthetic rainfall is chosen so every month has a unique, recognisable
amount: precip = month index (months since year 0). Window sums are then
exactly predictable, and any use of month M changes the result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import rainfall as rf  # noqa: E402


@pytest.fixture
def rain():
    p = pd.period_range("1999-01", "2001-12", freq="M")
    df = pd.DataFrame({"year": p.year, "month": p.month})
    df["precip_mm"] = rf.month_index(df.year, df.month).astype(float)
    return pd.concat([df.assign(well_id="W1"), df.assign(well_id="W2", precip_mm=df.precip_mm * 10)],
                     ignore_index=True)


@pytest.fixture
def obs():
    # Jan-2000 crosses the year boundary: its windows must come from 1999.
    return pd.DataFrame({"well_id": ["W1", "W1", "W2"], "year": [2000, 2000, 2001],
                         "month": [1, 5, 8]}, index=[10, 11, 12])


def test_window_sums_exclude_observation_month(rain, obs):
    out = rf.rainfall_windows(obs, rain)
    t = rf.month_index(2000, 1)                      # Jan-2000, well W1
    assert out.loc[10, "rain_1m_mm"] == t - 1        # Dec-1999 only
    assert out.loc[10, "rain_3m_mm"] == (t - 1) + (t - 2) + (t - 3)
    assert out.loc[10, "rain_6m_mm"] == sum(t - k for k in range(1, 7))
    t2 = rf.month_index(2001, 8)                     # W2 uses its own series
    assert out.loc[12, "rain_1m_mm"] == 10 * (t2 - 1)


def test_assertion_catches_leaky_lags(rain, obs):
    """Injecting lag 0 (= month M) must raise, proving the guard works."""
    with pytest.raises(rf.LeakageError, match="observation month or later"):
        rf.rainfall_windows(obs, rain, _lags=range(0, 6))


def test_assertion_catches_future_month_directly():
    c = pd.DataFrame({"obs_m": [100, 100], "src_m": [99, 101]})
    with pytest.raises(rf.LeakageError):
        rf.assert_no_leakage(c, window=3)
    rf.assert_no_leakage(c.iloc[[0]], window=3)      # the clean row passes


def test_assertion_catches_too_long_window():
    with pytest.raises(rf.LeakageError, match="outside"):
        rf.assert_no_leakage(pd.DataFrame({"obs_m": [100], "src_m": [96]}), window=3)


def test_missing_month_gives_nan_not_partial_sum(rain, obs):
    t = rf.month_index(2000, 1)
    holed = rain[~((rain.well_id == "W1") & (rf.month_index(rain.year, rain.month) == t - 2))]
    out = rf.rainfall_windows(obs, holed)
    assert out.loc[10, "rain_1m_mm"] == t - 1        # unaffected
    assert np.isnan(out.loc[10, "rain_3m_mm"]) and np.isnan(out.loc[10, "rain_6m_mm"])
