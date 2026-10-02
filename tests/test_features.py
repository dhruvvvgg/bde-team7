"""Stage 5 feature tests: correctness on tiny cases + a no-future-information
perturbation test on random data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def test_no_feature_depends_on_future_readings_or_rain():
    """Perturb everything from 2006 on; every row before 2006 must be identical."""
    base = _panel()
    pert = base.copy()
    fut = pert.year >= 2006
    rng = np.random.default_rng(99)
    pert.loc[fut, "gwl_m_bgl"] = rng.uniform(50, 60, fut.sum())
    for k in (1, 3, 6):
        pert.loc[fut, f"rain_{k}m_mm"] = rng.uniform(1000, 2000, fut.sum())
    a = ft.build_features(base).set_index(["well_id", "date"])
    b = ft.build_features(pert).set_index(["well_id", "date"])
    past = a.index.get_level_values("date") < pd.Timestamp(2006, 1, 1)
    leaky_ok = {"well_completeness_full_period"}   # documented descriptive-only column
    cols = [c for c in a.columns if c not in leaky_ok]
    pd.testing.assert_frame_equal(a.loc[past, cols], b.loc[past, cols])


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
