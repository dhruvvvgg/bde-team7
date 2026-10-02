"""Part E: nearest-cell CHIRPS for 2023-01..2024-12 into a SEPARATE file
(dataset/chirps/chirps_monthly_by_well_2023_2024.parquet). The main rainfall
table is not modified. HTTPS only, sequential, rasters deleted after sampling.

Run: python scripts/09_chirps_extension.py   (resumable)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config  # noqa: E402


def main() -> None:
    wells = pd.read_csv(config.CROSSWALK)[["well_id", "lat", "lon"]]
    t = chirps.build_table(wells, out=config.CHIRPS_EXT_TABLE,
                           first=pd.Period("2023-01", "M"), last=pd.Period("2024-12", "M"))
    print(f"rows={len(t)} expected={len(wells) * 24} nan={int(t.precip_mm.isna().sum())}")


if __name__ == "__main__":
    main()
