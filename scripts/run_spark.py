"""Run the complete Spark pipeline, with visible stages and execution evidence."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pyspark.sql import SparkSession, functions as F
from gwstress import config
from gwstress import spark_pipeline as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="local[2]", help="Spark master; local[2] suits a laptop/Colab")
    parser.add_argument("--output", type=Path, default=config.PROCESSED_DIR / "spark")
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    # Independent output tree preserves the pandas baseline for parity checks.
    protected = [config.FINAL_DIR.resolve(), config.EXT_DIR.resolve(), config.REPO_ROOT.resolve(), config.PROCESSED_DIR.resolve()]
    if args.output in protected or config.GW_DIR.resolve() == args.output or config.CHIRPS_DIR.resolve() == args.output:
        parser.error("--output must be a dedicated Spark-output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spark = (SparkSession.builder.master(args.master).appName("team7-groundwater-preprocessing")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
             .config("spark.sql.adaptive.enabled", True).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    evidence = {"engine": "pyspark", "spark_version": spark.version,
                "master": spark.sparkContext.master, "stages": []}
    try:
        with tempfile.TemporaryDirectory(prefix="gw-spark-checkpoint-") as checkpoint:
            spark.sparkContext.setCheckpointDir(checkpoint)
            print("[Spark 1/5] CSV ingestion, well IDs and wide-to-long reshape", flush=True)
            wide = sp.read_wide(spark, config.GW_REFINED_FILE).checkpoint(eager=True)
            sp.assert_unique(wide, ["well_id"], "main wells")
            long = sp.reshape(wide).checkpoint(eager=True)
            rain = spark.read.parquet(str(config.CHIRPS_TABLE))
            print("[Spark 2/5] QC, rainfall joins, temporal features and stress labels", flush=True)
            main = sp.final_columns(sp.build(long, rain)).checkpoint(eager=True)
            main.write.mode("overwrite").partitionBy("state_slug").parquet(str(args.output / "gw_features_by_state"))
            evidence["main_rows"] = main.count()
            evidence["main_columns"] = len(main.columns)
            evidence["main_wells"] = main.select("well_id").distinct().count()
            evidence["stages"].extend(["spark_csv_ingestion", "spark_explode_reshape", "spark_qc", "spark_rainfall_join",
                                       "spark_history_windows", "spark_stress_labels", "spark_partitioned_write"])
            ext = None
            if not args.main_only:
                print("[Spark 3/5] Match and process 2023-2024 extension", flush=True)
                source = sp.read_wide(spark, config.GW_STAGE4_FILE)
                matched = sp.match_extension(wide, source).checkpoint(eager=True)
                extension_long = sp.reshape(matched, 2023, 2024)
                # Specific yield is static and inherited from the main cohort.
                extension_long = extension_long.drop("specific_yield").join(
                    wide.select("well_id", F.col("Reference_Sy").alias("specific_yield")), "well_id")
                combined = long.unionByName(extension_long)
                combined_rain = rain.unionByName(spark.read.parquet(str(config.CHIRPS_EXT_TABLE)))
                ext = sp.final_columns(sp.build(combined, combined_rain).filter(F.col("year") >= 2023), extension=True).checkpoint(eager=True)
                ext.write.mode("overwrite").parquet(str(args.output / "extension_2023_2024" / "extension_features.parquet"))
                evidence["extension_rows"] = ext.count()
                evidence["stages"].extend(["spark_extension_match", "spark_extension_history"])
            print("[Spark 4/5] Shared chronological and state evaluation splits", flush=True)
            cfg = json.loads((config.REPO_ROOT / "config/splits.json").read_text())
            splits = sp.build_splits(main, ext, cfg).checkpoint(eager=True)
            sp.assert_unique(splits, ["well_id", "period_label"], "splits")
            splits.write.mode("overwrite").parquet(str(args.output / "splits.parquet"))
            evidence["stages"].append("spark_evaluation_splits")
            evidence["eligible_by_split"] = {r.split_chrono: r["count"] for r in splits.filter("eval_eligible").groupBy("split_chrono").count().collect()}
            evidence["stress_counts"] = {r.stress_category: r["count"] for r in main.groupBy("stress_category").count().collect()}
            evidence["output"] = str(args.output)
            evidence["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            # A readable physical plan confirms native execution, including
            # joins/windows and limited specialised BatchEvalPython nodes.
            plan = sp.groundwater_features(sp.rainfall_features(
                sp.clean(long).withColumn("_excluded", F.lit(False)), rain))._jdf.queryExecution().executedPlan().toString()
            (args.output / "execution_plan.txt").write_text(plan)
            (args.output / "run_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
            report = ["# Spark preprocessing run", "", f"Spark {spark.version}; master `{spark.sparkContext.master}`.",
                      "", "Core preprocessing was executed through Spark DataFrames and windows.",
                      "The specialized median, rolling-statistics, Theil-Sen and Student-t UDFs run on executors.",
                      "No main preprocessing table is converted to pandas or collected to the driver.",
                      "", "```json", json.dumps(evidence, indent=2), "```", "",
                      "The upstream curated-cohort/future-QC limitations remain; changing the engine does not remove them."]
            (args.output / "spark_run.md").write_text("\n".join(report) + "\n")
            print("[Spark 5/5] Complete\n" + json.dumps(evidence, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
