"""Compare the outputs of a clean run (directory A) with the committed outputs
(directory B): Parquet contents with a float tolerance (row order and
partition file names ignored), CSV and Markdown text exactly.

Run: python scripts/verify_reproduction.py <clean_run_repo> <reference_repo>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = ["processed/gw_features_by_state", "processed/extension_2023_2024/extension_features.parquet",
           "processed/splits.parquet", "dataset/chirps/chirps_monthly_by_well.parquet",
           "dataset/chirps/chirps_monthly_by_well_3x3.parquet",
           "dataset/chirps/chirps_monthly_by_well_2023_2024.parquet"]
TEXT = ["dataset/well_crosswalk.csv", "reports/chirps_sampling_flagged_wells.csv"] + \
       [f"reports/{n}" for n in ["stage1_inspection.md", "stage3_cleaning.md", "stage4_rainfall.md",
                                 "stage5_features.md", "stage6_stress.md", "data_quality.md",
                                 "data_validity.md", "chirps_sampling_check.md", "extension_2023_2024.md",
                                 "splits_summary.md"]]
RTOL, ATOL = 1e-9, 1e-9


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    if set(a.columns) != set(b.columns):
        return [f"columns differ: {set(a.columns) ^ set(b.columns)}"]
    key = [c for c in ["well_id", "date", "year", "month", "period_label"] if c in a.columns]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)[a.columns]
    if len(a) != len(b):
        return [f"rows {len(a)} vs {len(b)}"]
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind == "f":
            ok = np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            ok = (x.astype(object).where(x.notna(), "<NA>").astype(str).to_numpy()
                  == y.astype(object).where(y.notna(), "<NA>").astype(str).to_numpy())
        if not ok.all():
            bad.append(f"{c}: {int((~ok).sum())} cells differ")
    return bad


def main() -> None:
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    failures = 0
    for p in PARQUET:
        bad = frames_equal(pd.read_parquet(A / p), pd.read_parquet(B / p))
        print(("OK  " if not bad else "DIFF") + f" {p}" + ("" if not bad else f": {bad}"))
        failures += bool(bad)
    for p in TEXT:
        same = (A / p).read_text() == (B / p).read_text()
        print(("OK  " if same else "DIFF") + f" {p}")
        failures += not same
    print(f"\n{failures} difference(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
