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
                             "year": y, "season": s, "state": st, "gwl_m_bgl": v, "n_prior_same_season": i})
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
