"""Stage 2: wide -> long reshape and the well_id crosswalk.

Input layout (verified in stage 1): one row per well, 20 metadata columns,
92 "Mon-YY" reading columns (Jan/May/Aug/Nov, 2000-2022), then Reference_Sy.
Output: one row per (well_id, observation round), INCLUDING rounds with no
reading (gwl = NaN). WHY keep empty rounds: missingness is itself a reported
quantity, and later feature code needs to know a round existed but was not
measured (vs. never scheduled).
"""
from __future__ import annotations

import pandas as pd

from . import config, gw_io

# Static covariates carried onto every long row. Reference_Sy is the
# aquifer-class covariate chosen at checkpoint 1 (hydrogeological map skipped).
STATIC_COLS = ["State", "District", "Station Name", "Latitude", "Longitude",
               "Type of Well", "Aquifer Type", "Well Depth", "Reference_Sy"]

CROSSWALK_COLS = ["well_id", "State", "District", "Station Name", "Latitude",
                  "Longitude", "Station Code"]

RENAME = {"State": "state", "District": "district", "Station Name": "station_name",
          "Latitude": "lat", "Longitude": "lon", "Type of Well": "well_type",
          "Aquifer Type": "aquifer_type", "Well Depth": "well_depth_m",
          "Reference_Sy": "specific_yield", "Station Code": "station_code_corrupted"}


def crosswalk(wide: pd.DataFrame) -> pd.DataFrame:
    """well_id -> identifying attributes, including the corrupted Station Code.

    WHY keep the corrupted code: it is the only link back to CGWB's own
    registry; a user with the original registry can match on (code prefix,
    name, lat/lon). Raw strings, exactly as in the source file.
    """
    return wide[CROSSWALK_COLS].rename(columns=RENAME)


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt observation columns to rows and parse the period label.

    `date` is the first day of the monitoring month. CGWB does not publish the
    day of each reading, so the month is the finest defensible resolution;
    this is why rainfall windows in stage 4 end the month BEFORE the reading.
    """
    _, obs, _ = gw_io.split_columns(wide.drop(columns="well_id"))
    long = wide.melt(id_vars=["well_id"] + STATIC_COLS, value_vars=obs,
                     var_name="period_label", value_name="gwl_m_bgl")
    parsed = pd.DataFrame(long["period_label"].map(gw_io.parse_obs_label).tolist(),
                          columns=["year", "month", "season"], index=long.index)
    long = pd.concat([long, parsed], axis=1).rename(columns=RENAME)
    long["date"] = pd.to_datetime(dict(year=long.year, month=long.month, day=1))
    long["season"] = pd.Categorical(long["season"],
                                    categories=list(config.SEASON_BY_MONTH.values()))
    first = ["well_id", "date", "year", "month", "season", "period_label", "gwl_m_bgl"]
    return (long[first + [c for c in long.columns if c not in first]]
            .sort_values(["well_id", "date"]).reset_index(drop=True))
