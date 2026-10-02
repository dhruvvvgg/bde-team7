"""Part C: rainfall sampling check. Builds dataset/chirps/chirps_monthly_by_well_3x3.parquet
(nearest cell AND 3x3 mean) without touching the existing CHIRPS table.

Run: python scripts/04c_chirps_3x3.py   (resumable; HTTPS only)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config  # noqa: E402


def main() -> None:
    wells = pd.read_csv(config.CROSSWALK)[["well_id", "lat", "lon"]]
    tab = chirps.build_table_3x3(wells)
    print(f"rows={len(tab)} expected={len(wells) * len(chirps.months())}")


if __name__ == "__main__":
    main()
