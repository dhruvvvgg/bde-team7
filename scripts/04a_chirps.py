"""Stage 4a: build dataset/chirps/chirps_monthly_by_well.parquet.

Run: python scripts/04a_chirps.py   (resumable; safe to re-run after failure)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config  # noqa: E402


def main() -> None:
    wells = pd.read_csv(config.CROSSWALK)[["well_id", "lat", "lon"]]
    tab = chirps.build_table(wells)
    n_exp = len(wells) * len(chirps.months())
    print(f"rows={len(tab)} expected={n_exp} nan={tab.precip_mm.isna().sum()}")


if __name__ == "__main__":
    main()
