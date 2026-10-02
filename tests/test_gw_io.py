"""Tests for groundwater loading and well-ID construction."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, gw_io  # noqa: E402


def test_parse_obs_label():
    assert gw_io.parse_obs_label("May-07") == (2007, 5, "pre_monsoon")
    assert gw_io.parse_obs_label("Nov-22") == (2022, 11, "post_monsoon_kharif")
    with pytest.raises(ValueError):
        gw_io.parse_obs_label("Reference_Sy")


def test_split_columns_keeps_trailing_static_column_out_of_observations():
    df = pd.DataFrame(columns=["State", "Latitude", "Jan-00", "May-00", "Reference_Sy"])
    meta, obs, extra = gw_io.split_columns(df)
    assert meta == ["State", "Latitude"]
    assert obs == ["Jan-00", "May-00"]
    assert extra == ["Reference_Sy"]


def test_well_id_is_deterministic_and_detects_collisions():
    row = {"State": "Kerala", "District": "Ernakulam", "Station Name": "Kanjur",
           "Latitude": 10.1278, "Longitude": 76.4264}
    df = pd.DataFrame([row, {**row, "Station Name": "Other"}])
    ids = gw_io.make_well_id(df)
    assert ids.iloc[0] == gw_io.make_well_id(df.iloc[[0]]).iloc[0]
    assert ids.nunique() == 2
    with pytest.raises(ValueError):
        gw_io.make_well_id(pd.DataFrame([row, row]))


@pytest.mark.skipif(not config.GW_REFINED_FILE.exists(), reason="dataset not present")
def test_refined_table_matches_filtered_2022_and_ids_unique():
    sy = gw_io.load_wide(config.GW_REFINED_FILE)
    base = gw_io.load_wide(config.GW_FILTERED_DIR / "CGWB_filtered_wells_2000_2022.csv")
    assert len(sy) == 2759 and sy.well_id.is_unique
    assert (sy.well_id.values == base.well_id.values).all()
    _, obs, _ = gw_io.split_columns(base.drop(columns="well_id"))
    pd.testing.assert_frame_equal(sy[obs], base[obs])
