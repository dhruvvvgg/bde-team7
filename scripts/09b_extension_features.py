"""Part E step 2: features for the 2023-24 extension rounds.

Every feature is computed exactly as for the main table, on the combined
2000-2024 series (main rows + extension rows) with the combined CHIRPS record
(1999-07..2024-12), using the same functions. Only the 2023-24 rows are
written, to processed/extension_2023_2024/.

Leakage proof: because every feature is past-only, adding 2023-24 data must
not change any 2000-2022 row. This script recomputes the 2000-2022 rows from
the combined data and asserts they equal the committed main table for every
column except the documented whole-record descriptive columns.

Run: python scripts/09b_extension_features.py   (after 09a and 09_chirps_extension)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import clean, config, extension, features as ft, rainfall, stress, validity  # noqa: E402

# Descriptive columns that use the whole record by design (class B). Adding
# 2023-24 data legitimately changes them, so they are excluded from the proof
# and recomputed for the extension on the combined 2000-2024 record.
WHOLE_RECORD = {"well_completeness_full_period", "well_has_no_series_for_season", "flag_extreme"}


def main() -> None:
    main_clean = pd.read_parquet(config.GW_CLEAN)
    ext = pd.read_parquet(config.EXT_LONG)
    combined = pd.concat([main_clean, ext[main_clean.columns]], ignore_index=True)
    assert not combined.duplicated(["well_id", "date"]).any()
    combined["season"] = pd.Categorical(combined["season"].astype(str),
                                        categories=list(config.SEASON_BY_MONTH.values()))
    # flag_extreme is a whole-record QC flag; recompute it on 2000-2024 for the extension rows.
    combined["flag_extreme"] = clean.flag_extreme(combined)

    rain = pd.concat([pd.read_parquet(config.CHIRPS_TABLE), pd.read_parquet(config.CHIRPS_EXT_TABLE)],
                     ignore_index=True)
    assert not rain.duplicated(["well_id", "year", "month"]).any()
    # The 87 extension zeros (likely placeholders) are excluded from every
    # history statistic but keep their stored value (see
    # extension.build_features_excluding).
    ext_zero = combined["flag_zero"] & (combined["year"] >= config.EXT_FIRST_YEAR)
    out = extension.build_features_excluding(combined, rain, ext_zero)
    out = validity.suspect_flags(out)
    # Extension zeros: flagged with their own reason (likely placeholders, see report).
    ez = out["flag_zero"] & (out["year"] >= config.EXT_FIRST_YEAR)
    out.loc[ez, "suspect_reading"] = True
    out.loc[ez, "suspect_reason"] = np.where(out.loc[ez, "suspect_reason"].isna(), "zero_extension",
                                             out.loc[ez, "suspect_reason"].astype(str) + ";zero_extension")

    # ---- Leakage proof against the committed main table ----
    final = pd.read_parquet(config.FINAL_DIR).drop(columns="state_slug")
    key = ["well_id", "date"]
    a = final.sort_values(key).reset_index(drop=True)
    b = out[out.year <= 2022].sort_values(key).reset_index(drop=True)[a.columns]
    assert (a[key].values == b[key].values).all()
    bad = {}
    for c in a.columns:
        if c in WHOLE_RECORD:
            continue
        x, y = a[c], b[c]
        if x.dtype.kind == "f":
            n = int((~np.isclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True)).sum())
        else:
            n = int((x.astype(object).where(x.notna(), "<NA>").astype(str)
                     != y.astype(object).where(y.notna(), "<NA>").astype(str)).sum())
        if n:
            bad[c] = n
    assert not bad, f"2000-2022 rows changed when 2023-24 data was added: {bad}"
    print(f"leakage proof OK: {len(a):,} rows x {len(a.columns) - len(WHOLE_RECORD)} columns identical")

    # Real-data check of the exclusion: a round right after an excluded zero
    # must not see it as the previous reading.
    o = out.sort_values(["well_id", "date"])
    prev_excl = o.groupby("well_id")["_excluded"].shift(1).fillna(False).astype(bool)
    assert o.loc[prev_excl, "gw_prev_round_m"].isna().all(), "excluded zero leaked into gw_prev_round_m"
    assert int(out["_excluded"].sum()) == int(ext_zero.sum())
    n_excl = int(out["_excluded"].sum())

    ext_out = out[out.year >= config.EXT_FIRST_YEAR].copy()
    ext_out["is_extension"] = True
    ext_out = ext_out[list(a.columns) + ["is_extension"]].sort_values(["state", "well_id", "date"])
    if config.EXT_DIR.exists():
        shutil.rmtree(config.EXT_DIR)
    config.EXT_DIR.mkdir(parents=True)
    path = config.EXT_DIR / "extension_features.parquet"
    ext_out.to_parquet(path, index=False)

    r = ext_out[ext_out.gwl_m_bgl.notna()]
    L = ["\n## 4. Extension features\n",
         f"- Written to `{path.relative_to(config.REPO_ROOT)}`: **{len(ext_out):,} rows** "
         f"({ext_out.well_id.nunique()} wells x 8 rounds), {ext_out.shape[1]} columns (same columns as "
         f"the main table plus `is_extension`), {path.stat().st_size / 1e6:.1f} MB.",
         "- Computed with the same functions on the combined 2000-2024 series and the CHIRPS record "
         "1999-07..2024-12 (`chirps_monthly_by_well_2023_2024.parquet`, nearest cell, 66,216 rows, 0 NaN).",
         f"- **Leakage proof:** recomputing the 2000-2022 rows with the 2023-24 data present reproduces "
         f"the committed main table exactly ({len(a):,} rows, all columns except the whole-record "
         f"descriptive {sorted(WHOLE_RECORD)}).",
         f"- **{n_excl} extension zero readings (`zero_extension`) are excluded from all history "
         "statistics** (expanding, rolling and t-adjusted baselines, lags, trends, maxima and minima to "
         "date, counts, completeness) for later rounds, while their stored value (0.0) is unchanged. "
         "Enforced by building features with those readings hidden, then recomputing only their own "
         "row's anomaly and label columns. `tests/test_extension.py` proves that changing them changes "
         "no other row, and this script asserts it on the real data for `gw_prev_round_m`. They are "
         "`eval_eligible = False` (`extension_zero_placeholder`) in `processed/splits.parquet`.",
         "- Whole-record columns for the extension (`well_completeness_full_period`, "
         "`well_has_no_series_for_season`, `flag_extreme`) use 2000-2024.\n",
         "Stress categories (readings only):\n",
         pd.DataFrame({c: r[c].value_counts() for c in ["stress_category", "stress_tadj", "stress_roll10"]})
         .reindex(stress.CATEGORIES[1:]).fillna(0).astype(int).to_markdown(),
         f"\n- Suspect readings: {int(r.suspect_reading.sum())} "
         f"({', '.join(f'{k}: {v}' for k, v in r.suspect_reason.str.split(';').explode().value_counts().items())})."]
    rep = config.REPORTS_DIR / "extension_2023_2024.md"
    txt = rep.read_text().split("\n## 4. Extension features")[0].rstrip("\n")
    rep.write_text(txt + "\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
