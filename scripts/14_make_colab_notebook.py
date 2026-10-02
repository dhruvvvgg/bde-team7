"""Part M: generate notebooks/preprocessing_colab.ipynb.

The notebook runs the repo's own scripts (no duplicated logic) on a fresh
Colab runtime: clone, install pinned requirements, run each stage with a
visible check, show figures, run pytest, and an OPTIONAL PySpark section.

Run: python scripts/14_make_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER, REPO = "dhruvvvgg", "bde-team7"
BADGE = (f"https://colab.research.google.com/github/{USER}/{REPO}/blob/main/notebooks/"
         "preprocessing_colab.ipynb")

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(f"""
# Seasonal Groundwater Stress Assessment: preprocessing pipeline (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({BADGE})

This notebook reproduces the complete preprocessing pipeline of the Big Data Essentials project, end to end.
It does not reimplement anything: every stage calls the repository's own scripts and modules
(`scripts/*.py`, `src/gwstress/`), so the notebook and the repository cannot drift apart.

**Runtime:** about 5 minutes on a standard Colab CPU runtime. No GPU is needed.
**Data:** the groundwater dataset and the extracted CHIRPS rainfall tables are committed to the repository.
Raw CHIRPS rasters are **not** downloaded unless you switch `RUN_CHIRPS_DOWNLOAD` on below.
""")

md("""
## 0. Settings

- `RUN_CHIRPS_DOWNLOAD`: **off by default.** When on, it re-downloads the CHIRPS v3.0 rasters (282 + 282 + 24 files,
  about 13 GB of transfer, more than an hour) and rebuilds the rainfall tables. When off, the committed tables are used.
- `RUN_PYSPARK`: the optional PySpark demonstration at the end. It is off by default because it installs Java
  dependencies and Spark.
- `REPO_URL`: the GitHub repository. **If the repository is private**, either make it public, or use a
  personal access token, e.g. `https://<TOKEN>@github.com/dhruvvvgg/bde-team7.git`. Never commit a
  notebook that contains a token.
""")
code("""
import os
RUN_CHIRPS_DOWNLOAD = False   # True = re-download CHIRPS rasters (slow, about 13 GB); False = use committed tables
RUN_PYSPARK = os.environ.get("RUN_PYSPARK", "0") == "1"   # set to True to run the optional PySpark section
REPO_URL = "https://github.com/dhruvvvgg/bde-team7.git"
BRANCH = "main"
""")

md("## 1. Get the code and data, install pinned requirements")
code("""
import subprocess, sys
from pathlib import Path

def sh(cmd, check=True):
    \"\"\"Run a shell command and print its output (works the same in Colab and in Jupyter).\"\"\"
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-4000:], r.stderr[-4000:] if r.returncode else "")
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r

# Use the current directory if it already is the repository; otherwise clone into ./bde-team7.
if Path("src/gwstress").exists():
    REPO_DIR = Path.cwd()
elif Path("../src/gwstress").exists():          # opened from notebooks/
    REPO_DIR = Path("..").resolve()
else:
    if not Path("bde-team7").exists():
        sh(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} bde-team7")
    REPO_DIR = Path("bde-team7").resolve()
os.chdir(REPO_DIR)
print("working directory:", Path.cwd())
sh("git log --oneline -1")
""")
code("""
# Pinned versions used to produce the committed outputs (see requirements.txt).
# In Colab this may print dependency-resolver warnings about preinstalled packages; they are harmless here.
sh(f"{sys.executable} -m pip install -q -r requirements.txt")
import pandas as pd, numpy as np
print("pandas", pd.__version__, "| numpy", np.__version__)
""")
code("""
def run(script):
    \"\"\"Run one pipeline script with the notebook's Python interpreter.\"\"\"
    r = subprocess.run([sys.executable, script], text=True, capture_output=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"{script} failed")
    print(f"✓ {script}")
    return r.stdout

pd.set_option("display.width", 160, "display.max_columns", 20)
""")

md("""
## 2. Stage 1: inspection

The script profiles every candidate table, splits the metadata columns from the observation columns,
and quantifies missingness. Output: `reports/stage1_inspection.md`.
""")
code("""
out = run("scripts/01_inspect.py")
print("\\n".join(l for l in out.splitlines() if l.startswith("Shape") or "Overall cell missingness" in l))
""")

md("""
## 3. Stage 2: ingestion and reshape (wide → long)

The script assigns a surrogate `well_id` (the original Station Code is corrupted upstream) and turns the
wide `Mon-YY` columns into one row per well × round. Empty rounds are kept.
""")
code("""
run("scripts/02_reshape.py")
long = pd.read_parquet("processed/interim/gw_long.parquet")
print(long.shape, "| wells:", long.well_id.nunique(), "| rounds:", long.period_label.nunique(),
      "| readings:", int(long.gwl_m_bgl.notna().sum()))
long.head(4)
""")

md("""
## 4. Stage 3: cleaning and flagging

Nothing is removed. Placeholders become NaN, exact duplicates are dropped, names are standardised,
and QC flags are added.
""")
code("""
run("scripts/03_clean.py")
clean = pd.read_parquet("processed/interim/gw_clean.parquet")
clean[[c for c in clean.columns if c.startswith("flag_")]].sum().to_frame("rows flagged")
""")

md("""
## 5. Stage 4: CHIRPS rainfall integration

The script samples monthly CHIRPS v3.0 at every well (nearest 0.05° cell) and builds rainfall windows from months
**up to M−1 only**. A `LeakageError` assertion checks this on every row. By default the committed
rainfall tables are used.
""")
code("""
if RUN_CHIRPS_DOWNLOAD:
    for s in ["scripts/04a_chirps.py", "scripts/04c_chirps_3x3.py", "scripts/09_chirps_extension.py"]:
        run(s)   # resumable; HTTPS only; each raster is deleted after sampling
else:
    print("RUN_CHIRPS_DOWNLOAD = False: using the committed rainfall tables")
rain = pd.read_parquet("dataset/chirps/chirps_monthly_by_well.parquet")
print("CHIRPS table:", rain.shape, "| NaN:", int(rain.precip_mm.isna().sum()),
      "| months:", rain[["year", "month"]].drop_duplicates().shape[0])
run("scripts/04b_rain_windows.py")
gr = pd.read_parquet("processed/interim/gw_rain.parquet")
gr[["rain_1m_mm", "rain_3m_mm", "rain_6m_mm"]].describe().round(1)
""")

md("""
## 6. Stage 5: time-aware feature engineering

These are past-only features: seasonal baselines (expanding, rolling-10, small-sample adjusted), anomalies,
groundwater history, trend, rainfall deviations, and the Oct-Dec monsoon features.
""")
code("""
run("scripts/05_features.py")
feat = pd.read_parquet("processed/interim/gw_features.parquet")
print(feat.shape)
feat[["anomaly_z", "anomaly_z_capped", "anomaly_tadj", "trend_5y_m_per_yr", "rain_12m_mm"]].describe().round(2)
""")

md("""
## 7. Stage 6: stress classification

The same `classify()` function and thresholds (1 / 1.5 / 2) are applied to three anomaly scores.
`stress_category` (expanding baseline) is the primary label.
""")
code("""
run("scripts/06_stress.py")
st = pd.read_parquet("processed/interim/gw_stress.parquet")
pd.DataFrame({c: st[c].astype(str).value_counts() for c in ["stress_category", "stress_roll10", "stress_tadj"]})
""")

md("## 8. Stage 7: storage (partitioned Parquet) and the data-quality report")
code("""
run("scripts/07_output.py")
final = pd.read_parquet("processed/gw_features_by_state")
print("final table:", final.shape, "| partitions:", len(list(Path("processed/gw_features_by_state").iterdir())))
final.groupby("state").well_id.nunique().sort_values(ascending=False).head(8)
""")

md("## 9. Validation: rainfall sampling check and data validity")
code("""
run("scripts/04d_chirps_sampling_check.py")
run("scripts/08_validity.py")
print(open("reports/chirps_sampling_check.md").read().split("## Per state")[0])
""")

md("""
## 10. Extension holdout 2023-2024 (kept separate)

The 2023-24 rounds of the same 2,759 wells are matched, with a proof of the match, and their features are computed with the same code.
The script asserts that adding these data changes no 2000-2022 row (an end-to-end leakage proof).
""")
code("""
run("scripts/09a_extension_match.py")
print(run("scripts/09b_extension_features.py").splitlines()[0])
ext = pd.read_parquet("processed/extension_2023_2024/extension_features.parquet")
print("extension:", ext.shape, "| coverage:", round(100 * ext.gwl_m_bgl.notna().mean(), 1), "%")
""")

md("## 11. Shared splits file")
code("""
run("scripts/10_splits.py")
spl = pd.read_parquet("processed/splits.parquet")
pd.crosstab(spl[spl.eval_eligible].split_chrono, spl[spl.eval_eligible].season, margins=True)
""")

md("## 12. Documentation, figures and architecture diagram")
code("""
for s in ["scripts/11_handoff_dictionary.py", "scripts/12_figures.py", "scripts/13_architecture.py"]:
    run(s)
from IPython.display import Image, display
for f in ["docs/architecture_preprocessing.png", "reports/figures/fig01_missingness_heatmap.png",
          "reports/figures/fig09_excess_high_stress.png", "reports/figures/fig13_leakage_timeline.png",
          "reports/figures/fig16_extension_coverage.png"]:
    print(f)
    display(Image(filename=f, width=900))
""")

md("""
## 13. Tests, including the leakage tests

This runs the full test suite: leakage assertions, perturb-the-future tests at four cutoffs, the Monte Carlo
check of the small-sample adjustment, and the data-contract tests.
""")
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], text=True, capture_output=True)
print(r.stdout[-2500:])
assert r.returncode == 0, "tests failed"
""")
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_rainfall.py", "tests/test_features.py", "tests/test_extension.py", "-v"],
                   text=True, capture_output=True)
print("\\n".join(l for l in r.stdout.splitlines() if "leak" in l.lower() or "future" in l.lower()
                or "excluded" in l.lower() or "passed" in l))
""")

md("""
---
## OPTIONAL: PySpark demonstration of big-data tooling

**Off by default.** Set `RUN_PYSPARK = True` in the settings cell to run it. It reads the partitioned Parquet
with `spark.read.parquet` and reproduces three aggregates computed with pandas: rows per state, mean depth by
season, and stress category counts. It then asserts that the two engines agree. Colab provides Java; the cell
installs `pyspark`.
""")
code("""
if RUN_PYSPARK:
    sh(f"{sys.executable} -m pip install -q pyspark==3.5.3")
    from pyspark.sql import SparkSession, functions as F
    spark = (SparkSession.builder.master("local[*]").appName("gw-preprocessing")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    sdf = spark.read.parquet("processed/gw_features_by_state")
    print("Spark rows:", sdf.count(), "| partitions read:", sdf.select("state_slug").distinct().count())
else:
    print("RUN_PYSPARK = False: skipping the optional PySpark section")
""")
code("""
if RUN_PYSPARK:
    pdf = pd.read_parquet("processed/gw_features_by_state")
    # 1. rows per state
    s1 = {r["state"]: r["count"] for r in sdf.groupBy("state").count().collect()}
    p1 = pdf.groupby("state").size().to_dict()
    assert s1 == p1, "rows per state differ"
    # 2. mean depth by season
    s2 = {r["season"]: r["m"] for r in sdf.groupBy("season").agg(F.avg("gwl_m_bgl").alias("m")).collect()}
    p2 = pdf.groupby(pdf.season.astype(str)).gwl_m_bgl.mean().to_dict()
    assert all(abs(s2[k] - p2[k]) < 1e-9 for k in p2), "mean depth differs"
    # 3. stress category counts
    s3 = {r["stress_category"]: r["count"] for r in sdf.groupBy("stress_category").count().collect()}
    p3 = pdf.stress_category.astype(str).value_counts().to_dict()
    assert s3 == p3, "stress counts differ"
    print("Spark and pandas agree on all three aggregates")
    display(pd.DataFrame({"spark": pd.Series(s2), "pandas": pd.Series(p2)}).round(4))
    display(pd.DataFrame({"spark": pd.Series(s3), "pandas": pd.Series(p3)}))
    spark.stop()
""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = ROOT / "notebooks" / "preprocessing_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out, len(cells), "cells")
