"""Part D: data-validity flags. Nothing here deletes or changes a reading.

suspect_reading / suspect_reason combine three documented reasons:
  zero_suspicious   one of the 3 zero readings judged implausible
                    (config.SUSPICIOUS_ZEROS); the other 8 zeros are consistent
                    with a full well in the monsoon and are not included.
  deeper_than_well  reading deeper than the recorded well depth
                    (= flag_deeper_than_well, stage 3).
  extreme_anomaly   |anomaly_z| > 5 (the cap used for anomaly_z_capped).
                    anomaly_z uses only earlier readings, so this reason is
                    past-only. Rows without a baseline cannot get it.
The flag is DESCRIPTIVE (class B): it is for QC and sensitivity analyses
(e.g. "drop suspect rows and re-evaluate"), not a model input. The
deeper_than_well reason compares against well depth, a static attribute that
CGWB may have recorded after the reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EXTREME_Z = 5.0


def suspect_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_key = set(config.SUSPICIOUS_ZEROS)
    r_zero = pd.Series([(w, p) in zero_key for w, p in zip(df.well_id, df.period_label)], index=df.index)
    found = int(r_zero.sum())
    assert found == len(zero_key), f"expected {len(zero_key)} suspicious zeros, found {found}"
    r_deep = df["flag_deeper_than_well"].astype(bool)
    r_ext = df["anomaly_z"].abs().gt(EXTREME_Z).fillna(False)
    reasons = np.select([r_zero & r_deep, r_zero, r_deep & r_ext, r_deep, r_ext],
                        ["zero_suspicious;deeper_than_well", "zero_suspicious",
                         "deeper_than_well;extreme_anomaly", "deeper_than_well", "extreme_anomaly"],
                        default="")
    df["suspect_reading"] = r_zero | r_deep | r_ext
    df["suspect_reason"] = pd.Series(reasons, index=df.index).replace("", None)
    return df


def coordinate_checks(wells: pd.DataFrame) -> dict:
    b = config.INDIA_BBOX
    out = ~(wells.lat.between(b["lat_min"], b["lat_max"]) & wells.lon.between(b["lon_min"], b["lon_max"]))
    co = wells.groupby(["lat", "lon"]).well_id.transform("size")
    return {"outside_bbox": wells[out], "colocated": wells[co > 1].sort_values(["lat", "lon"])}
