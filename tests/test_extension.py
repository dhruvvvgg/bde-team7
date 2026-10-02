"""Extension tests: excluded (placeholder) readings never feed any other row."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import extension  # noqa: E402

SEASONS = {1: "post_monsoon_rabi", 5: "pre_monsoon", 8: "monsoon", 11: "post_monsoon_kharif"}


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in ("W1", "W2"):
        for y in range(2000, 2012):
            for m, s in SEASONS.items():
                rows.append({"well_id": w, "year": y, "month": m, "season": s, "date": pd.Timestamp(y, m, 1),
                             "period_label": f"x{y}{m}", "gwl_m_bgl": rng.uniform(2, 10)})
    d = pd.DataFrame(rows)
    d["season"] = pd.Categorical(d.season, categories=list(SEASONS.values()))
    return d


def _rain():
    rng = np.random.default_rng(1)
    p = pd.period_range("1999-01", "2011-12", freq="M")
    return pd.concat([pd.DataFrame({"well_id": w, "year": p.year, "month": p.month,
                                    "precip_mm": rng.gamma(1.5, 60, len(p))}) for w in ("W1", "W2")],
                     ignore_index=True)


def test_excluded_values_do_not_affect_any_other_row():
    panel, rain = _panel(), _rain()
    # Exclude three readings of W1 (like extension zeros), including May.
    ex = (panel.well_id == "W1") & panel.date.isin(pd.to_datetime(["2008-05-01", "2009-05-01", "2010-01-01"]))
    a_in, b_in = panel.copy(), panel.copy()
    a_in.loc[ex, "gwl_m_bgl"] = 0.0
    b_in.loc[ex, "gwl_m_bgl"] = 55.0                      # wildly different excluded value
    a = extension.build_features_excluding(a_in, rain, ex).set_index(["well_id", "date"])
    b = extension.build_features_excluding(b_in, rain, ex).set_index(["well_id", "date"])
    exi = pd.MultiIndex.from_frame(panel.loc[ex, ["well_id", "date"]])
    others = ~a.index.isin(exi)
    skip = {"well_completeness_full_period", "well_has_no_series_for_season", "_excluded"}
    cols = [c for c in a.columns if c not in skip]
    pd.testing.assert_frame_equal(a.loc[others, cols], b.loc[others, cols])
    # The excluded rows keep their stored value and get their own anomaly from it.
    assert (a.loc[exi, "gwl_m_bgl"] == 0.0).all() and (b.loc[exi, "gwl_m_bgl"] == 55.0).all()
    assert (a.loc[exi, "anomaly_z"] < b.loc[exi, "anomaly_z"]).all()
    # The round after an excluded reading does not see it as the previous reading.
    nxt = a.loc[("W1", pd.Timestamp("2008-08-01"))]
    assert np.isnan(nxt.gw_prev_round_m)
