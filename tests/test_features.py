"""Stage 5 feature tests: correctness on tiny cases + a no-future-information
perturbation test on random data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import features as ft  # noqa: E402

SEASONS = {1: "post_monsoon_rabi", 5: "pre_monsoon", 8: "monsoon", 11: "post_monsoon_kharif"}


def _panel(n_years=12, wells=("W1", "W2"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in wells:
        for y in range(2000, 2000 + n_years):
            for m, s in SEASONS.items():
                rows.append({"well_id": w, "year": y, "month": m, "season": s,
                             "date": pd.Timestamp(y, m, 1), "gwl_m_bgl": rng.uniform(2, 10),
                             **{f"rain_{k}m_mm": rng.uniform(0, 300) for k in (1, 3, 6)}})
    return pd.DataFrame(rows)


def test_baseline_uses_only_earlier_same_season_and_respects_k():
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2007),
                      "gwl_m_bgl": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 100.0]})
    out = ft.seasonal_baseline(d, k=3).set_index("year")
    assert out.loc[2002, "n_prior_same_season"] == 2 and np.isnan(out.loc[2002, "baseline_mean_m"])
    assert out.loc[2003, "baseline_mean_m"] == 2.0          # mean(1,2,3); NaN reading skipped
    assert out.loc[2004, "baseline_mean_m"] == 2.0          # 2003 was NaN, adds nothing
    assert out.loc[2006, "baseline_mean_m"] == 3.0          # mean(1..5); 100 is current -> excluded
    assert np.isclose(out.loc[2006, "anomaly_z"], (100 - 3) / np.std([1, 2, 3, 4, 5], ddof=1))


def test_std_floor_prevents_explosion():
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2007),
                      "gwl_m_bgl": [5.0] * 6 + [5.3]})
    out = ft.seasonal_baseline(d, k=5).set_index("year")
    assert out.loc[2006, "flag_std_floored"]
    assert np.isclose(out.loc[2006, "anomaly_z"], 0.3 / ft.STD_FLOOR_M)


def test_missing_season_series_gives_nan_but_keeps_other_seasons():
    p = _panel(n_years=10, wells=("W1",))
    p.loc[p.season == "pre_monsoon", "gwl_m_bgl"] = np.nan
    out = ft.build_features(p, k=5)
    assert out[out.season == "pre_monsoon"].anomaly_z.isna().all()
    assert out[(out.season == "monsoon") & (out.year >= 2005)].anomaly_z.notna().all()


def test_theil_sen_trend_excludes_current_year():
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2007),
                      "gwl_m_bgl": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0]})
    tr = ft.seasonal_trend(d).to_numpy()
    assert np.isnan(tr[:4]).all()            # < 4 prior points
    assert tr[4] == 1.0 and tr[6] == 1.0     # 1000 in 2006 does not affect 2006's own trend


# Columns that use the whole record by design. They are documented as
# descriptive only (class B) and excluded from the no-future check.
LEAKY_BY_DESIGN = {"well_completeness_full_period", "well_has_no_series_for_season"}


def _monthly_rain(wells=("W1", "W2"), start="1999-01", end="2011-12", seed=1):
    rng = np.random.default_rng(seed)
    p = pd.period_range(start, end, freq="M")
    return pd.concat([pd.DataFrame({"well_id": w, "year": p.year, "month": p.month,
                                    "precip_mm": rng.gamma(1.5, 60, len(p))}) for w in wells],
                     ignore_index=True)


def _full_build(panel, rain):
    """Same path as the real pipeline: rain windows from monthly CHIRPS, then
    the stage-5 features and the Part-A extras."""
    from gwstress import rainfall as rf
    p = rf.rainfall_windows(panel.drop(columns=[c for c in panel if c.startswith("rain_")]), rain)
    return ft.build_features(p, rain=rain).set_index(["well_id", "date"])


def _perturb_future(panel, rain, cutoff):
    """Replace every reading at or after `cutoff` and every rainfall month at or
    after cutoff's month with wildly different values."""
    rng = np.random.default_rng(99)
    p, r = panel.copy(), rain.copy()
    fut = p.date >= cutoff
    p.loc[fut, "gwl_m_bgl"] = rng.uniform(50, 60, fut.sum())
    rfut = (r.year * 12 + r.month) >= (cutoff.year * 12 + cutoff.month)
    r.loc[rfut, "precip_mm"] = rng.uniform(1000, 2000, rfut.sum())
    return p, r


@pytest.mark.parametrize("cutoff", ["2006-01-01", "2006-05-01", "2006-08-01", "2006-11-01"])
def test_no_feature_depends_on_future_readings_or_rain(cutoff):
    """For several cutoffs, including mid-year ones, perturb all future readings
    and rainfall months. Every feature of every earlier row must be identical.
    This covers ALL columns produced by the pipeline, including Part-A extras."""
    cutoff = pd.Timestamp(cutoff)
    panel, rain = _panel(), _monthly_rain()
    a = _full_build(panel, rain)
    b = _full_build(*_perturb_future(panel, rain, cutoff))
    past = a.index.get_level_values("date") < cutoff
    cols = [c for c in a.columns if c not in LEAKY_BY_DESIGN]
    assert {"gw_prev_round_m", "rain_12m_mm", "rain_monsoon_ytd_mm", "rain_last_monsoon_dev_mm",
            "rain_3m_z", "gw_fluct_prev_wy_m", "gw_max_to_date_m"} <= set(cols)
    pd.testing.assert_frame_equal(a.loc[past, cols], b.loc[past, cols])


def test_perturbation_check_detects_a_leaky_feature():
    """Meta-test: a deliberately leaky column (next round's reading) must fail
    the same comparison, proving the test above has teeth."""
    cutoff = pd.Timestamp("2006-05-01")
    panel, rain = _panel(), _monthly_rain()
    a, b = _full_build(panel, rain), _full_build(*_perturb_future(panel, rain, cutoff))
    for d in (a, b):
        d["leaky_next"] = d.groupby(level="well_id")["gwl_m_bgl"].shift(-1)
    past = a.index.get_level_values("date") < cutoff
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(a.loc[past, ["leaky_next"]], b.loc[past, ["leaky_next"]])


def test_rain_normal_uses_prior_years_only():
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2008),
                      **{f"rain_{k}m_mm": [100.0] * 7 + [300.0] for k in (1, 3, 6)}})
    out = ft.rainfall_anomalies(d).set_index("year")
    assert np.isnan(out.loc[2004, "rain_3m_normal_mm"])     # only 4 prior years
    assert out.loc[2005, "rain_3m_normal_mm"] == 100.0
    assert out.loc[2007, "rain_3m_dev_pct"] == 200.0         # 300 vs normal 100


def test_k_sensitivity_counts_monotone():
    out = ft.build_features(_panel(), k=5)
    s = ft.k_sensitivity(out)
    assert s.excluded.is_monotonic_increasing


def test_rolling_baseline_uses_last_10_prior_readings():
    vals = [float(i) for i in range(1, 16)] + [np.nan, 100.0]
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2000 + len(vals)),
                      "gwl_m_bgl": vals})
    out = ft.rolling_baseline(d).set_index("year")
    assert np.isnan(out.loc[2004, "baseline_roll10_mean_m"])          # 4 prior readings < 5
    assert out.loc[2005, "baseline_roll10_mean_m"] == 3.0              # mean(1..5)
    # values: 2000..2014 -> 1..15, 2015 -> NaN, 2016 -> 100
    assert out.loc[2014, "n_roll10"] == 10
    assert out.loc[2014, "baseline_roll10_mean_m"] == np.mean(range(5, 15))  # last 10 of 1..14
    assert out.loc[2015, "baseline_roll10_mean_m"] == np.mean(range(6, 16))  # last 10 of 1..15
    # the NaN in 2015 adds nothing; 2016's window is still 6..15 and excludes the current 100
    assert out.loc[2016, "baseline_roll10_mean_m"] == np.mean(range(6, 16))
    exp = (100 - np.mean(range(6, 16))) / np.std(range(6, 16), ddof=1)
    assert np.isclose(out.loc[2016, "anomaly_roll10"], exp)
    assert out.loc[2016, "anomaly_roll10_capped"] == 5.0


def test_rolling_baseline_forgets_an_old_level_shift():
    """A well that dropped by 5 m in 2005 and then stayed at the new level.
    By 2022 the rolling window holds only post-shift readings, so z is about 0.
    The expanding baseline still averages the pre-shift years in, so z stays
    high. (For a purely LINEAR decline both z values settle at a similar
    constant, around sqrt(3), because the expanding std grows with the lag.
    The rolling baseline helps with shifts and accelerating declines, not
    steady ones.)"""
    rng = np.random.default_rng(0)
    vals = [2.0 + rng.normal(0, 0.2) if y < 2005 else 7.0 + rng.normal(0, 0.2) for y in range(2000, 2023)]
    d = pd.DataFrame({"well_id": "W", "season": "monsoon", "year": range(2000, 2023), "gwl_m_bgl": vals})
    e = ft.seasonal_baseline(d, k=5).set_index("year")
    r = ft.rolling_baseline(d).set_index("year")
    # The rolling mean has forgotten the old level; the expanding mean has not.
    assert abs(r.loc[2022, "baseline_roll10_mean_m"] - 7.0) < 0.2
    assert e.loc[2022, "baseline_mean_m"] < 6.2
    # The shift also inflates the expanding std (~2 m vs ~0.2 m), which is why
    # the expanding z does not necessarily look extreme here.
    assert e.loc[2022, "baseline_std_m"] > 5 * r.loc[2022, "baseline_roll10_std_m"]
