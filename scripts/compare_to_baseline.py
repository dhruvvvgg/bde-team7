"""Cell-by-cell comparison of the current final table against the rollback
baseline snapshot (processed/interim/baseline_v1, copied from commit dab7477).
Used to fill the before/after numbers in docs/CHANGELOG_preprocessing.md.

Run: python scripts/compare_to_baseline.py [baseline_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "processed/interim/baseline_v1"
    key = ["well_id", "date"]
    a = pd.read_parquet(base_dir).sort_values(key).reset_index(drop=True)
    b = pd.read_parquet(ROOT / "processed/gw_features_by_state").sort_values(key).reset_index(drop=True)
    assert (a[key].values == b[key].values).all(), "row keys differ"
    print(f"rows {len(a)} -> {len(b)}; columns {a.shape[1]} -> {b.shape[1]}")
    changed = {}
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind == "f":
            n = int((~np.isclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True)).sum())
        else:
            n = int((x.astype(object).where(x.notna(), "<NA>").astype(str)
                     != y.astype(object).where(y.notna(), "<NA>").astype(str)).sum())
        if n:
            changed[c] = n
    print("existing columns with changed values:", changed or "none")
    print("new columns:", [c for c in b.columns if c not in a.columns])


if __name__ == "__main__":
    main()
