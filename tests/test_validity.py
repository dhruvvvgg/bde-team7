"""Part D validity-flag tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, validity  # noqa: E402


def test_suspect_flags_reasons_and_no_value_changes():
    zs = config.SUSPICIOUS_ZEROS
    df = pd.DataFrame({
        "well_id": [w for w, _ in zs] + ["X", "Y", "Z"],
        "period_label": [p for _, p in zs] + ["May-10", "May-10", "May-10"],
        "gwl_m_bgl": [0.0, 0.0, 0.0, 30.0, 3.0, 3.0],
        "flag_deeper_than_well": [False, True, False, True, False, False],
        "anomaly_z": [np.nan, 1.0, 0.0, 6.0, -7.0, 2.0],
    })
    out = validity.suspect_flags(df)
    assert out.suspect_reading.tolist() == [True, True, True, True, True, False]
    assert out.suspect_reason.tolist()[:5] == ["zero_suspicious", "zero_suspicious;deeper_than_well",
                                                "zero_suspicious", "deeper_than_well;extreme_anomaly",
                                                "extreme_anomaly"]
    assert out.suspect_reason.isna().iloc[5]
    pd.testing.assert_series_equal(out.gwl_m_bgl, df.gwl_m_bgl)


def test_coordinate_checks():
    w = pd.DataFrame({"well_id": ["a", "b", "c", "d"], "lat": [20.0, 20.0, 50.0, 10.0],
                      "lon": [78.0, 78.0, 78.0, 60.0]})
    r = validity.coordinate_checks(w)
    assert set(r["outside_bbox"].well_id) == {"c", "d"}
    assert set(r["colocated"].well_id) == {"a", "b"}
