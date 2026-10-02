"""Stage 3: cleaning of the long groundwater table.

WHAT THE AUTHORS ALREADY DID (Code/1_filteration_code.ipynb, ReadMe), in order:
  C1 "nonzero": dropped wells whose 100 readings (2000-2024) were ALL empty.
     Despite the file name, this did NOT remove readings equal to 0.0 m.
  C2 "non_negative": dropped every WELL with any negative reading (a whole
     well, not just the value).
  C3 "3sigma": per well, blanked readings with |x - mean| > 3*std, where mean
     and std are computed over that well's entire 2000-2024 series (so the
     upstream QC itself used the full record; this is QC, not a model feature).
  C4: kept wells with >=2 of the 4 rounds non-empty in EVERY year 2000-2022.
  C5: dropped wells where a value repeats >2 times consecutively (stuck gauge
     or copy-forward).
  Then Reference_Sy was attached by k-means on hydrogeological-map colours.

WHAT WE DO HERE (and deliberately do not do):
  - No outlier REMOVAL: C3 already did it, and doing it again would remove
    genuine extremes, which are exactly what a stress study must keep.
    Extra extremes are only FLAGGED (flag_extreme).
  - Placeholders (-999-like, blanks, non-finite) -> NaN. Stage 1 found none,
    but the rule is kept so the pipeline is safe on other table vintages.
  - Exact zeros are kept and flagged (flag_zero), per checkpoint-1 decision:
    a water table at the surface is physically plausible (waterlogged areas).
  - Exact duplicate (well_id, date) rows removed and counted.
  - State/district names standardised.
  - Direction check: values are depth below ground level (m bgl), so larger =
    deeper = more stressed. Rows inconsistent with that convention are flagged.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PLACEHOLDER_MAX = -900.0  # any value <= -900 is treated as a sentinel (-999, -9999)

# Robust extreme flag threshold. 3.5 is the conventional modified-z cut-off
# (Iglewicz & Hoaglin 1993). Stricter than C3's 3-sigma on purpose: it only
# needs to point at values worth eyeballing, not remove anything.
EXTREME_MODZ = 3.5


def standardise_name(s: pd.Series) -> pd.Series:
    """Title-case, trim, collapse whitespace, unify '&'/'and'.

    WHY: CGWB names are free text ('Andhra pradesh'); later joins to shapefiles
    and dashboards need one canonical spelling per place.
    """
    def _one(x):
        if pd.isna(x):
            return x
        x = re.sub(r"\s+", " ", str(x)).strip()
        x = re.sub(r"\s*&\s*", " and ", x)
        return " ".join(w.capitalize() if w.lower() != "and" else "and" for w in x.split(" "))
    return s.map(_one)


def placeholders_to_nan(gwl: pd.Series) -> tuple[pd.Series, int]:
    bad = gwl.notna() & ((gwl <= PLACEHOLDER_MAX) | ~np.isfinite(gwl.astype(float)))
    return gwl.mask(bad), int(bad.sum())


def flag_extreme(df: pd.DataFrame) -> pd.Series:
    """Per-well modified z-score > EXTREME_MODZ, computed on the same season.

    LEAKAGE NOTE: this flag uses the whole series of the well (past and
    future). That is acceptable ONLY because it is a descriptive QC flag that
    no model feature reads. Do not use flag_extreme as a model input.
    """
    g = df.groupby(["well_id", "season"], observed=True)["gwl_m_bgl"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    modz = 0.6745 * (df["gwl_m_bgl"] - med) / mad.replace(0, np.nan)
    return modz.abs().gt(EXTREME_MODZ).fillna(False)


def clean(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all stage-3 rules. Returns (cleaned table, log of counts)."""
    log: dict = {"rows_in": len(long), "readings_in": int(long.gwl_m_bgl.notna().sum())}
    df = long.copy()

    df["gwl_m_bgl"], log["placeholders_to_nan"] = placeholders_to_nan(df["gwl_m_bgl"])

    before = len(df)
    df = df.drop_duplicates()                       # exact duplicate rows
    log["exact_duplicate_rows_removed"] = before - len(df)
    dup_key = df.duplicated(["well_id", "date"], keep=False)
    if dup_key.any():  # same key, different values: ambiguous, must not be silently resolved
        raise ValueError(f"{dup_key.sum()} rows share (well_id, date) with different values")

    for col in ["state", "district"]:
        new = standardise_name(df[col])
        log[f"{col}_names_changed"] = sorted(set(zip(df[col][new != df[col]], new[new != df[col]])))
        df[col] = new
    # Aquifer Type '-' is CGWB's "not recorded"; make it explicit missing.
    df["aquifer_type"] = df["aquifer_type"].replace("-", pd.NA)

    df["flag_zero"] = df["gwl_m_bgl"].eq(0)
    df["flag_extreme"] = flag_extreme(df)
    # Direction checks. Depth bgl must be >= 0 (negative = above ground /
    # artesian or an elevation convention); a level deeper than the well's own
    # depth means the well was dry or the value is in another datum.
    df["flag_negative"] = df["gwl_m_bgl"].lt(0)
    df["flag_deeper_than_well"] = df["gwl_m_bgl"].gt(df["well_depth_m"])
    for f in ["flag_zero", "flag_extreme", "flag_negative", "flag_deeper_than_well"]:
        log[f] = int(df[f].sum())
    log["rows_out"] = len(df)
    log["readings_out"] = int(df.gwl_m_bgl.notna().sum())
    return df.reset_index(drop=True), log
