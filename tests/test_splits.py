"""Part F split-construction tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import splits  # noqa: E402

CFG = {"K_min_history": 1, "chrono": {"train": [2000, 2001], "test": [2002, 2002]},
       "sparse_round_min_coverage_pct": 60.0, "ext_sparse_round_min_coverage_pct": 10.0}


def test_eligibility_reasons_and_folds():
    rows = []
    for w, st, vals in (("A", "Kerala", [1, 2, 3]), ("B", "Bihar", [1, np.nan, 3])):
        for i, y in enumerate((2000, 2001, 2002)):
            for m, s, lab in ((5, "pre_monsoon", "May"), (8, "monsoon", "Aug")):
                v = vals[i] if (m == 8 or w == "B") else np.nan
                rows.append({"well_id": w, "date": pd.Timestamp(y, m, 1), "period_label": f"{lab}-{y % 100:02d}",
                             "year": y, "season": s, "state": st, "gwl_m_bgl": v, "n_prior_same_season": i,
                             "suspect_reason": None})
    s, cov = splits.build(pd.DataFrame(rows), None, CFG)
    r = s.set_index(["well_id", "period_label"])
    assert r.loc[("A", "May-02"), "exclusion_reason"] == "no_reading"
    assert r.loc[("A", "Aug-00"), "exclusion_reason"] == "insufficient_history"
    assert r.loc[("B", "Aug-01"), "exclusion_reason"] == "no_reading"
    assert r.loc[("A", "Aug-01"), "exclusion_reason"] == "sparse_round"     # 1 of 2 wells = 50% < 60%
    assert r.loc[("A", "Aug-02"), "eval_eligible"]
    assert r.loc[("A", "Aug-02"), "split_chrono"] == "test"
    assert r.loc[("A", "May-00"), "no_may_state"] and not r.loc[("B", "May-00"), "no_may_state"]
    assert r.loc[("B", "Aug-00"), "state_fold"] == 0 and r.loc[("A", "Aug-00"), "state_fold"] == 1


def test_extension_zero_placeholder_excluded_but_main_zero_kept():
    base = {"date": pd.Timestamp(2023, 1, 1), "year": 2023, "season": "post_monsoon_rabi", "state": "Bihar",
            "n_prior_same_season": 9}
    main = pd.DataFrame([{**base, "well_id": "M", "period_label": "Jan-23", "gwl_m_bgl": 0.0,
                          "suspect_reason": None}])
    ext = pd.DataFrame([{**base, "well_id": "E", "period_label": "Jan-23", "gwl_m_bgl": 0.0,
                         "suspect_reason": "zero_extension"},
                        {**base, "well_id": "F", "period_label": "Jan-23", "gwl_m_bgl": 3.0,
                         "suspect_reason": None}])
    s, _ = splits.build(main, ext, CFG | {"chrono": {"ext": [2023, 2023]}})
    r = s.set_index(["well_id", "source"])
    assert r.loc[("E", "extension"), "exclusion_reason"] == "extension_zero_placeholder"
    assert r.loc[("F", "extension"), "eval_eligible"]
    assert r.loc[("M", "main"), "eval_eligible"]          # main-table zero: not excluded
