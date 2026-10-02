"""Stage 6 classification tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import stress  # noqa: E402


def test_categories_boundaries_and_precedence():
    gwl = pd.Series([np.nan, np.nan, 5, 5, 5, 5, 5, 5, 5, 5])
    z = pd.Series([3.0, np.nan, np.nan, -4.0, 0.99, 1.0, 1.49, 1.5, 2.0, 40.0])
    got = stress.classify(gwl, z).astype(str).tolist()
    assert got == ["No reading", "No reading", "Insufficient history", "Normal", "Normal",
                   "Watch", "Watch", "Moderate Stress", "High Stress", "High Stress"]


def test_negative_anomaly_is_normal_not_stress():
    assert stress.classify(pd.Series([3.0]), pd.Series([-10.0])).iloc[0] == "Normal"


def test_thresholds_are_parameters_and_validated():
    z, g = pd.Series([1.2]), pd.Series([1.0])
    assert stress.classify(g, z, (1, 1.5, 2)).iloc[0] == "Watch"
    assert stress.classify(g, z, (0.5, 0.8, 1.1)).iloc[0] == "High Stress"
    with pytest.raises(ValueError):
        stress.classify(g, z, (2, 1.5, 1))


def test_capped_and_raw_z_give_same_categories():
    z = pd.Series(np.linspace(-60, 60, 241))
    g = pd.Series(1.0, index=z.index)
    pd.testing.assert_series_equal(stress.classify(g, z), stress.classify(g, z.clip(-5, 5)))


def test_sensitivity_counts_sum_to_rows():
    df = pd.DataFrame({"gwl_m_bgl": [1.0, np.nan, 2, 3, 4], "anomaly_z": [0.7, 1, np.nan, 1.7, 2.6]})
    counts, changes = stress.sensitivity(df)
    assert (counts.sum() == len(df)).all()
    assert changes.set_index("threshold set").loc["default (1/1.5/2)", "rows changed vs default"] == 0
