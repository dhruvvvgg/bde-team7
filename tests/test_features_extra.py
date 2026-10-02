"""Part-A feature tests: exact values on small hand-checkable cases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import features_extra as fx  # noqa: E402
from gwstress.rainfall import LeakageError, month_index  # noqa: E402

SEASONS = {1: "post_monsoon_rabi", 5: "pre_monsoon", 8: "monsoon", 11: "post_monsoon_kharif"}


def _grid(vals, start_year=2000):
    rows = []
    i = 0
    y = start_year
    while i < len(vals):
        for m, s in SEASONS.items():
            if i < len(vals):
                rows.append({"well_id": "W", "year": y, "month": m, "season": s,
                             "date": pd.Timestamp(y, m, 1), "gwl_m_bgl": vals[i]})
                i += 1
        y += 1
    return pd.DataFrame(rows)


def test_groundwater_history_values():
    v = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    d = fx.groundwater_history(_grid(v)).reset_index(drop=True)
    assert np.isnan(d.gw_prev_round_m[0]) and d.gw_prev_round_m[1] == 1.0
    assert np.isnan(d.gw_prev_round_m[3])                 # round 2 was missing
    assert d.gw_last_obs_m[3] == 2.0 and d.rounds_since_last_obs[3] == 2
    assert d.gw_same_season_lag1y_m[4] == 1.0 and d.gw_same_season_lag2y_m[8] == 1.0
    assert d.gw_prev_change_m[5] == 5.0 - 4.0             # reading(t-1) - reading(t-2)
    assert d.gw_delta_from_prev_round_m[5] == 6.0 - 5.0   # uses current reading (class C)
    assert d.gw_max_to_date_m[4] == 4.0 and d.gw_min_to_date_m[4] == 1.0   # excludes 5.0
    assert d.n_readings_to_date.tolist()[:5] == [0, 1, 2, 2, 3]


def test_prior_year_fluctuation_uses_previous_year_only():
    # 2000: Jan 1, May 5, Aug 3, Nov 2 -> fluct(2000) = 2 - 5 = -3
    d = _grid([1.0, 5.0, 3.0, 2.0, 9.0, 9.0, 9.0, 100.0])
    f = fx.prior_year_fluctuation(d).to_numpy()
    assert np.isnan(f[:4]).all()          # no 1999 data
    assert (f[4:] == -3.0).all()          # Nov-2001 (100) is the current round, not used


def _rain(start="1999-01", end="2003-12", val=None):
    p = pd.period_range(start, end, freq="M")
    df = pd.DataFrame({"well_id": "W", "year": p.year, "month": p.month})
    df["precip_mm"] = month_index(df.year, df.month).astype(float) if val is None else val
    return df


def test_rain_12m_monsoon_and_ytd_exact():
    d = _grid([1.0] * 12, start_year=2001)
    d[["rain_3m_mm", "rain_6m_mm"]] = 0.0
    out = fx.rainfall_extra(d, _rain()).reset_index(drop=True)
    t = month_index(2001, 8)                                   # Aug-2001 round (row 2)
    assert out.rain_12m_mm[2] == sum(t - k for k in range(1, 13))
    assert out.last_monsoon_year[2] == 2000                    # Aug: current monsoon incomplete
    assert out.rain_last_monsoon_mm[2] == sum(month_index(2000, m) for m in range(6, 10))
    assert out.last_monsoon_year[3] == 2001                    # Nov: Jun-Sep 2001 complete
    assert out.rain_monsoon_ytd_mm[2] == month_index(2001, 6) + month_index(2001, 7)
    assert out.rain_monsoon_ytd_mm[3] == sum(month_index(2001, m) for m in range(6, 11))
    assert out.rain_monsoon_ytd_mm[0] == sum(month_index(2000, m) for m in range(6, 13))  # Jan
    assert out.rain_monsoon_ytd_mm[1] == (sum(month_index(2000, m) for m in range(6, 13))
                                          + sum(month_index(2001, m) for m in range(1, 5)))  # May


def test_rain_sum_nan_when_record_too_short():
    d = _grid([1.0, 1.0], start_year=2000)
    d[["rain_3m_mm", "rain_6m_mm"]] = 0.0
    out = fx.rainfall_extra(d, _rain(start="1999-07"))
    assert out.rain_12m_mm.isna().all()          # needs 1999-01 / 1999-05


def test_month_sum_guard_raises_on_leak():
    d = _grid([1.0, 1.0], start_year=2001)
    cums = fx._monthly_cumsum(_rain())
    t = month_index(d.year, d.month)
    with pytest.raises(LeakageError):
        fx._month_sum(d, cums, t - 3, t)          # end at month M: leak


def test_calendar():
    d = fx.calendar(_grid([1.0] * 4, start_year=2000))
    assert d.water_year.tolist() == [1999, 1999, 2000, 2000]
    assert d.water_year_index.tolist() == [0, 0, 1, 1]
    assert d.season_order.tolist() == [2, 3, 0, 1]


def test_northeast_monsoon_uses_previous_years_oct_dec():
    d = _grid([1.0] * 4, start_year=2001)                   # Jan, May, Aug, Nov 2001
    d[["rain_3m_mm", "rain_6m_mm"]] = 0.0
    out = fx.rainfall_extra(d, _rain()).reset_index(drop=True)
    exp = sum(month_index(2000, m) for m in (10, 11, 12))
    assert (out.last_ne_year == 2000).all()
    assert (out.rain_last_ne_mm == exp).all()               # Nov-2001 must NOT use Oct-2001
    assert out.rain_last_ne_dev_mm.isna().all()             # < 5 earlier NE seasons


def test_northeast_monsoon_normal_uses_earlier_years_only():
    years = range(2000, 2009)
    p = pd.period_range("1999-01", "2008-12", freq="M")
    r = pd.DataFrame({"well_id": "W", "year": p.year, "month": p.month})
    # Oct-Dec totals: 1999..2006 -> alternating 100/200 per month; 2007 -> 1000 per month
    r["precip_mm"] = np.where(r.month >= 10, np.where(r.year == 2007, 1000.0,
                                                      np.where(r.year % 2 == 0, 100.0, 200.0)), 0.0)
    d = pd.concat([_grid([1.0] * 4, start_year=y) for y in years], ignore_index=True)
    d[["rain_3m_mm", "rain_6m_mm"]] = 0.0
    out = fx.rainfall_extra(d, r)
    row = out[(out.year == 2008) & (out.month == 1)].iloc[0]      # uses NE 2007
    prior = [300.0 if yy % 2 == 0 else 600.0 for yy in range(1999, 2007)]
    assert row.rain_last_ne_mm == 3000.0
    assert np.isclose(row.rain_last_ne_dev_mm, 3000.0 - np.mean(prior))
    assert row.rain_last_ne_z_capped == 5.0
