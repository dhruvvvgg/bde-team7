"""Part E: 2023-2024 extension holdout for the same 2,759 wells.

The source is the authors' stage-4 file (after 3-sigma) which, unlike the
refined table, still holds the 2023-24 rounds. The refined 2,759 wells were
SELECTED for completeness through 2022 (criterion 4: >= 2 readings in every
year 2000-2022; criterion 5: no value repeated > 2 times in a row, checked on
2000-2022 only). The 2023-24 readings therefore did NOT pass the repeated-value
filter, and are not required to be complete. That is why they are kept
separate from the main table.

Matching rule (proved in reports/extension_2023_2024.md):
  1. Compute the same surrogate well_id (gw_io.make_well_id) for every
     stage-4 row. It hashes state, district, station name, lat and lon, so a
     match implies identical coordinates by construction.
  2. Keep the rows whose well_id is one of the 2,759.
  3. If a well_id has more than one stage-4 row, keep the row whose 2000-2022
     readings equal the main table exactly. Exactly one must qualify.
  4. Assert that every kept row's 2000-2022 readings equal the main table,
     which proves it is the same series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, gw_io


def match_stage4(main_wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s4 = gw_io.load_wide(config.GW_STAGE4_FILE, strict=False)
    _, obs_main, _ = gw_io.split_columns(main_wide.drop(columns="well_id"))
    ids = set(main_wide.well_id)
    cand = s4[s4.well_id.isin(ids)].copy()
    ref = main_wide.set_index("well_id")[obs_main]
    same = np.isclose(cand[obs_main].to_numpy(float), ref.loc[cand.well_id, obs_main].to_numpy(float),
                      equal_nan=True).all(axis=1)
    cand["_same_2000_2022"] = same
    multi = cand.well_id.duplicated(keep=False)
    log = {"stage4_rows": len(s4), "candidate_rows": len(cand), "main_wells": len(ids),
           "unmatched_main_wells": sorted(ids - set(cand.well_id)),
           "ids_with_multiple_rows": sorted(cand[multi].well_id.unique()),
           "multi_rows_resolved": {}}
    keep = cand[~multi | cand._same_2000_2022]
    for wid, g in cand[multi].groupby("well_id"):
        log["multi_rows_resolved"][wid] = {"rows": len(g), "rows_matching_2000_2022": int(g._same_2000_2022.sum())}
    assert keep.well_id.is_unique, "ambiguous match: more than one row equals the main series"
    assert keep._same_2000_2022.all(), "a matched row's 2000-2022 readings differ from the main table"
    assert set(keep.well_id) == ids, "not every main well was matched"
    # Coordinate check: separate from the hash. Compare raw lat/lon.
    cc = keep.set_index("well_id")[["Latitude", "Longitude"]].join(
        main_wide.set_index("well_id")[["Latitude", "Longitude"]], rsuffix="_main")
    log["coordinate_mismatches"] = int(((cc.Latitude - cc.Latitude_main).abs() > 1e-9).sum()
                                       + ((cc.Longitude - cc.Longitude_main).abs() > 1e-9).sum())
    # Same name + state + district but a different location in stage 4 (a
    # relocated or re-surveyed well we are NOT matching). Reported only.
    key = ["State", "District", "Station Name"]
    nm = s4.merge(main_wide[key + ["well_id"]], on=key, suffixes=("", "_main"))
    log["same_name_other_coords_rows"] = int((nm.well_id != nm.well_id_main).sum())
    return keep.drop(columns="_same_2000_2022"), log


def extension_columns(df: pd.DataFrame) -> list[str]:
    _, obs, _ = gw_io.split_columns(df.drop(columns="well_id"))
    return [c for c in obs if config.EXT_FIRST_YEAR <= 2000 + int(c[-2:]) <= config.EXT_LAST_YEAR]
