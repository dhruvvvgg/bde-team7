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


def build_features_excluding(combined: pd.DataFrame, rain: pd.DataFrame,
                             exclude: pd.Series) -> pd.DataFrame:
    """Build all features, treating the readings in `exclude` as missing HISTORY
    while keeping their stored values.

    Used for the 87 extension zero readings judged to be placeholders
    (`zero_extension`). How the exclusion is enforced:
      1. Every feature is computed on a copy where those readings are NaN. So
         no baseline (expanding, rolling, t-adjusted), lag, trend, maximum or
         minimum to date, count, fluctuation or completeness for ANY row can
         use them.
      2. `gwl_m_bgl` is then restored to the stored value (0.0), and only the
         excluded row's OWN current-reading columns (anomalies, their capped
         versions, the three stress labels and gw_delta_from_prev_round_m) are
         recomputed from it, against the baseline built without the excluded
         readings.
    tests/test_extension.py proves that changing an excluded value changes no
    feature of any other row.
    """
    from . import features as ft, rainfall, stress

    exclude = exclude.reindex(combined.index).fillna(False).astype(bool)
    masked = combined.copy()
    masked.loc[exclude, "gwl_m_bgl"] = np.nan
    masked["_excluded"] = exclude
    out = rainfall.rainfall_windows(masked, rain)
    out = ft.build_features(out, k=ft.K_MAIN, rain=rain)
    out["stress_category"] = stress.classify(out["gwl_m_bgl"], out["anomaly_z"])
    out["stress_roll10"] = stress.classify(out["gwl_m_bgl"], out["anomaly_roll10"])
    out["stress_tadj"] = stress.classify(out["gwl_m_bgl"], out["anomaly_tadj"])

    ex = out["_excluded"].to_numpy()
    if ex.any():
        orig = combined.set_index(["well_id", "date"])["gwl_m_bgl"]
        rows = out.loc[ex]
        x = orig.reindex(pd.MultiIndex.from_frame(rows[["well_id", "date"]])).to_numpy(float)
        out.loc[ex, "gwl_m_bgl"] = x
        sub = out.loc[ex].copy()
        std_e = sub["baseline_std_m"].clip(lower=ft.STD_FLOOR_M)
        sub["anomaly_z"] = (sub["gwl_m_bgl"] - sub["baseline_mean_m"]) / std_e
        sub["anomaly_z_capped"] = sub["anomaly_z"].clip(-ft.Z_CAP, ft.Z_CAP)
        std_r = sub["baseline_roll10_std_m"].clip(lower=ft.STD_FLOOR_M)
        sub["anomaly_roll10"] = (sub["gwl_m_bgl"] - sub["baseline_roll10_mean_m"]) / std_r
        sub["anomaly_roll10_capped"] = sub["anomaly_roll10"].clip(-ft.Z_CAP, ft.Z_CAP)
        sub = ft.tadj_anomaly(sub)
        sub["gw_delta_from_prev_round_m"] = sub["gwl_m_bgl"] - sub["gw_prev_round_m"]
        for col in ["stress_category", "stress_roll10", "stress_tadj"]:
            src = {"stress_category": "anomaly_z", "stress_roll10": "anomaly_roll10",
                   "stress_tadj": "anomaly_tadj"}[col]
            sub[col] = stress.classify(sub["gwl_m_bgl"], sub[src])
        for c in ["anomaly_z", "anomaly_z_capped", "anomaly_roll10", "anomaly_roll10_capped",
                  "anomaly_tadj", "anomaly_tadj_capped", "gw_delta_from_prev_round_m",
                  "stress_category", "stress_roll10", "stress_tadj"]:
            out.loc[ex, c] = sub[c].to_numpy()
    return out
