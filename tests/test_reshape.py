"""Stage 2 reshape tests on a synthetic two-well wide table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import reshape  # noqa: E402


def _wide():
    base = {"well_id": ["Wa", "Wb"], "Station Code": ["1.4E+14", "1.4E+14"],
            "State": ["Kerala", "Bihar"], "District": ["X", "Y"], "Station Name": ["a", "b"],
            "Latitude": [10.0, 25.0], "Longitude": [76.0, 85.0], "Type of Well": ["Dug well"] * 2,
            "Aquifer Type": ["Unconfined", "-"], "Well Depth": [10.0, 12.0]}
    obs = {"Jan-00": [1.0, 2.0], "May-00": [np.nan, 3.0], "Aug-00": [0.5, 1.5], "Nov-00": [0.8, np.nan]}
    return pd.DataFrame({**base, **obs, "Reference_Sy": [0.13, 0.02]})


def test_to_long_shape_dates_and_values():
    long = reshape.to_long(_wide())
    assert len(long) == 8                       # 2 wells x 4 rounds, empties kept
    assert long.gwl_m_bgl.isna().sum() == 2
    r = long[(long.well_id == "Wb") & (long.period_label == "May-00")].iloc[0]
    assert r.date == pd.Timestamp("2000-05-01") and r.season == "pre_monsoon"
    assert r.gwl_m_bgl == 3.0 and r.specific_yield == 0.02 and r.state == "Bihar"


def test_crosswalk_keeps_corrupted_code():
    cw = reshape.crosswalk(_wide())
    assert list(cw.columns) == ["well_id", "state", "district", "station_name", "lat", "lon",
                                "station_code_corrupted"]
    assert cw.station_code_corrupted.tolist() == ["1.4E+14", "1.4E+14"]
