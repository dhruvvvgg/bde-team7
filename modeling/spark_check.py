"""Spark re-computation of the past-only baseline and anomaly (z) from the committed processed table, compared with the pipeline columns.

Window frame = same well and season, ordered by date, from UNBOUNDED PRECEDING to 1 PRECEDING (earlier readings only, K >= 5).
Writes results/tables/spark_check.json. Needs Java and pyspark (Linux: Kaggle, Colab, WSL). On Windows Spark's Python worker and Hadoop
winutils fail, so the script exits with code 2 and the notebook shows the saved JSON from a Linux run instead.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("BDE_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results" / "tables" / "spark_check.json"

if os.name == "nt":
    print("Spark is not run on Windows here (needs Linux); saved result of a Linux run is used.")
    sys.exit(2)

from pyspark.sql import SparkSession, Window, functions as F  # noqa: E402

t0 = time.time()
spark = (SparkSession.builder.master("local[2]").appName("gw-stress-check").config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "8").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sdf = spark.read.parquet(str(ROOT / "processed" / "gw_features_by_state")).select(
    "well_id", "season", "date", "gwl_m_bgl", "baseline_mean_m", "anomaly_z_capped", "stress_category")
n_rows, n_part = sdf.count(), sdf.rdd.getNumPartitions()

w = Window.partitionBy("well_id", "season").orderBy("date").rowsBetween(Window.unboundedPreceding, -1)
x = (sdf.withColumn("n", F.count("gwl_m_bgl").over(w)).withColumn("mu", F.avg("gwl_m_bgl").over(w))
     .withColumn("sd", F.stddev_samp("gwl_m_bgl").over(w)))
x = x.withColumn("z", F.when(F.col("n") >= 5, F.least(F.lit(5.0), F.greatest(F.lit(-5.0), (F.col("gwl_m_bgl") - F.col("mu")) / F.greatest(F.col("sd"), F.lit(0.10))))))
r = (x.filter((F.col("n") >= 5) & F.col("gwl_m_bgl").isNotNull())
     .agg(F.count("*").alias("rows"), F.max(F.abs(F.col("mu") - F.col("baseline_mean_m"))).alias("dmean"),
          F.max(F.abs(F.col("z") - F.col("anomaly_z_capped"))).alias("dz")).collect()[0])
counts = {row["stress_category"]: row["count"] for row in x.groupBy("stress_category").count().collect()}
res = {"spark_version": spark.version, "rows": n_rows, "partitions": n_part, "rows_compared": int(r["rows"]),
       "max_abs_diff_baseline_mean_m": float(r["dmean"]), "max_abs_diff_z": float(r["dz"]), "stress_counts": counts,
       "seconds": round(time.time() - t0, 1), "platform": sys.platform}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
spark.stop()
