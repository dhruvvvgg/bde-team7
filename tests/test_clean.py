"""Stage 3 cleaning tests on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import clean  # noqa: E402


def _long(vals, well="W1", season="monsoon"):
    n = len(vals)
    return pd.DataFrame({
        "well_id": well, "date": pd.date_range("2000-08-01", periods=n, freq="12MS"),
        "season": pd.Categorical([season] * n), "gwl_m_bgl": vals,
        "state": "andhra  pradesh", "district": "Y & Z", "aquifer_type": "-",
        "well_depth_m": 10.0,
    })


def test_standardise_name():
    s = clean.standardise_name(pd.Series(["andhra  pradesh ", "Y & Z", np.nan]))
    assert s.iloc[0] == "Andhra Pradesh" and s.iloc[1] == "Y and Z" and pd.isna(s.iloc[2])


def test_placeholders_become_nan_and_zero_is_flagged_not_removed():
    df, log = clean.clean(_long([-999.0, 0.0, 3.0, 4.0, 3.5]))
    assert log["placeholders_to_nan"] == 1 and pd.isna(df.gwl_m_bgl.iloc[0])
    assert df.gwl_m_bgl.iloc[1] == 0.0 and df.flag_zero.iloc[1]
    assert pd.isna(df.aquifer_type.iloc[0])


def test_exact_duplicates_removed_conflicting_duplicates_raise():
    d = _long([1.0, 2.0])
    df, log = clean.clean(pd.concat([d, d.iloc[[0]]]))
    assert log["exact_duplicate_rows_removed"] == 1 and len(df) == 2
    bad = pd.concat([d, d.iloc[[0]].assign(gwl_m_bgl=9.0)])
    with pytest.raises(ValueError):
        clean.clean(bad)


def test_extreme_is_flagged_not_removed_and_direction_flags():
    df, _ = clean.clean(_long([3.0, 3.1, 2.9, 3.0, 3.2, 2.8, 30.0, -1.0]))
    assert df.gwl_m_bgl.iloc[6] == 30.0 and df.flag_extreme.iloc[6]
    assert df.flag_deeper_than_well.iloc[6] and df.flag_negative.iloc[7]
    assert df.flag_extreme.sum() <= 2
