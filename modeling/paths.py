"""Where the modeling code reads its input table and writes its results.

Defaults are the original locations (processed/ = NumPy/pandas reference table, results/). Two environment variables
redirect them without touching any code path or protocol:
  BDE_PROCESSED  folder holding gw_features_by_state/ and splits.parquet   (e.g. processed/spark)
  BDE_RESULTS    folder that receives chrono/, geo/, robust/, tables/      (e.g. results_spark)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path(os.environ.get("BDE_PROCESSED", ROOT / "processed"))
RESULTS = Path(os.environ.get("BDE_RESULTS", ROOT / "results"))
