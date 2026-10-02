"""Central configuration: paths and dataset-format constants.

WHY a single module: every later stage (modeling, evaluation, dashboard)
imports from here, so a path or season definition is changed in exactly one
place and the reproducibility section of the report can cite one file.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GW_DIR = REPO_ROOT / "dataset" / "groundwater_data"
GW_OUTPUT_DIR = GW_DIR / "Output"
GW_FILTERED_DIR = GW_OUTPUT_DIR / "Filtered_GWLs_2000_2024"

# Candidate refined table (pending confirmation at checkpoint 1).
# CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv is row-for-row identical to
# Filtered_GWLs_2000_2024/CGWB_filtered_wells_2000_2022.csv (verified in
# stage 1) plus one extra column, Reference_Sy. It is the file the paper's
# ReadMe calls the "final output" (section B).
GW_REFINED_FILE = GW_OUTPUT_DIR / "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv"

CHIRPS_DIR = REPO_ROOT / "dataset" / "chirps"
CHIRPS_RAW_DIR = CHIRPS_DIR / "raw"  # gitignored; files deleted after sampling
CHIRPS_TABLE = CHIRPS_DIR / "chirps_monthly_by_well.parquet"

REPORTS_DIR = REPO_ROOT / "reports"
PROCESSED_DIR = REPO_ROOT / "processed"

# The authors' notebook treats the first 20 columns as well metadata and every
# later column as a "Mon-YY" observation (see Code/1_filteration_code.ipynb,
# `info_cols = columns[:20]`). We do not rely on the position: metadata is
# whatever does not parse as an observation label (see gw_io.split_columns).
N_AUTHOR_METADATA_COLS = 20

# CGWB monitors four times a year. The month -> season mapping follows CGWB's
# own naming of its monitoring rounds (pre-monsoon = May, etc.).
SEASON_BY_MONTH = {
    "Jan": "post_monsoon_rabi",   # January round (winter / rabi)
    "May": "pre_monsoon",         # May round, driest point of the year
    "Aug": "monsoon",             # August round, mid-monsoon
    "Nov": "post_monsoon_kharif", # November round, after the SW monsoon
}
MONTH_NUM = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
