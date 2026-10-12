"""Independent timing and scientific-rule checks for the Spark implementation."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import spark_pipeline as sp


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    s = (SparkSession.builder.master("local[2]").appName("gw-spark-tests")
         .config("spark.sql.shuffle.partitions", 2).config("spark.sql.session.timeZone", "UTC").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    s.sparkContext.setCheckpointDir(str(tmp_path_factory.mktemp("spark-checkpoints")))
    yield s
    s.stop()


def panel(spark):
    rows = [("W1", y, m, sp.SEASONS[m], float(y - 1999 + m), False)
            for y in range(2000, 2012) for m in (1, 5, 8, 11)]
    return spark.createDataFrame(rows, "well_id string, year int, month int, season string, gwl_m_bgl double, _excluded boolean").withColumn(
        "date", F.make_date("year", "month", F.lit(1)))


def values(df, cols):
    return {(r.well_id, str(r.date)): tuple(r[c] for c in cols) for r in df.select(*sp.KEY, *cols).collect()}


def test_seasonal_baseline_k_and_current_exclusion(spark):
    d = panel(spark)
    a = sp.groundwater_features(d).filter("month=8 AND year IN (2004,2005)").orderBy("year").collect()
    assert a[0].baseline_mean_m is None and a[0].n_prior_same_season == 4
    assert a[1].baseline_mean_m == 11.0 and a[1].n_prior_same_season == 5
    b = sp.groundwater_features(d.withColumn("gwl_m_bgl", F.when((F.col("year") == 2005) & (F.col("month") == 8), 1000.0).otherwise(F.col("gwl_m_bgl"))))
    row = b.filter("month=8 AND year=2005").first()
    assert row.baseline_mean_m == 11.0
    assert row.anomaly_z > a[1].anomaly_z


def test_future_perturbation_and_extrema_missing_round(spark):
    d = panel(spark).withColumn("gwl_m_bgl", F.when((F.col("year") == 2006) & (F.col("month") == 5), F.lit(None).cast("double")).otherwise(F.col("gwl_m_bgl")))
    a = sp.groundwater_features(d).checkpoint(eager=True)
    changed = d.withColumn("gwl_m_bgl", F.when(F.col("year") >= 2008, 999.0).otherwise(F.col("gwl_m_bgl")))
    b = sp.groundwater_features(changed).checkpoint(eager=True)
    cols = ["baseline_mean_m", "baseline_std_m", "gw_prev_round_m", "gw_last_obs_m", "gw_max_to_date_m", "gw_min_to_date_m", "trend_5y_m_per_yr", "anomaly_tadj"]
    av, bv = values(a.filter("year<2008"), cols), values(b.filter("year<2008"), cols)
    for key in av:
        for x, y in zip(av[key], bv[key]):
            assert x == pytest.approx(y) if x is not None else y is None
    gap = a.filter("year=2006 AND month=8").first()
    assert gap.gw_prev_round_m is None
    assert gap.gw_max_to_date_m is not None and gap.gw_min_to_date_m is not None


def test_excluded_extension_value_never_enters_later_history(spark):
    d = panel(spark).withColumn("_excluded", (F.col("year") == 2007) & (F.col("month") == 5))
    a = sp.groundwater_features(d).checkpoint(eager=True)
    b = sp.groundwater_features(d.withColumn("gwl_m_bgl", F.when(F.col("_excluded"), 10000.0).otherwise(F.col("gwl_m_bgl")))).checkpoint(eager=True)
    cols = ["baseline_mean_m", "gw_prev_round_m", "gw_last_obs_m", "gw_max_to_date_m", "gw_min_to_date_m", "n_readings_to_date", "gw_fluct_prev_wy_m"]
    assert values(a.filter("NOT _excluded"), cols) == values(b.filter("NOT _excluded"), cols)


def test_rainfall_uses_only_complete_earlier_months(spark):
    obs = spark.createDataFrame([("W", 2022, 5, "pre_monsoon"), ("W", 2022, 11, "post_monsoon_kharif")],
                                "well_id string, year int, month int, season string").withColumn("date", F.make_date("year", "month", F.lit(1)))
    rain = spark.createDataFrame([("W", y, m, float(m)) for y in range(2020, 2023) for m in range(1, 13)],
                                 "well_id string, year int, month int, precip_mm double")
    row = sp.rainfall_features(obs, rain).filter("month=5").first()
    assert row.rain_1m_mm == 4 and row.rain_3m_mm == 9 and row.rain_6m_mm == 33
    missing = sp.rainfall_features(obs, rain.filter("NOT(year=2022 AND month=3)")).filter("month=5").first()
    assert missing.rain_1m_mm == 4 and missing.rain_3m_mm is None and missing.rain_6m_mm is None
    changed = rain.withColumn("precip_mm", F.when((F.col("year") == 2022) & (F.col("month") >= 5), 9999.0).otherwise(F.col("precip_mm")))
    b = sp.rainfall_features(obs, changed).filter("month=5").first()
    assert b.rain_6m_mm == row.rain_6m_mm
    # A real input where ordinary double sum followed by float32 rounding
    # differs from the established newest-first compensated float32 sum.
    precision_rain = spark.createDataFrame([
        ("W", 2022, 8, 161.0933837890625),
        ("W", 2022, 9, 96.04691314697266),
        ("W", 2022, 10, 1.7736263275146484),
    ], "well_id string, year int, month int, precip_mm double")
    precision = sp.rainfall_features(obs, precision_rain).filter("month=11").first()
    assert precision.rain_3m_mm == 258.9139404296875


def test_stress_missing_precedence_and_threshold_boundaries(spark):
    d = spark.createDataFrame([(None, None), (2.0, None), (2.0, -2.0), (2.0, 1.0), (2.0, 1.5), (2.0, 2.0)],
                              "gwl_m_bgl double, z double")
    rows = d.withColumn("label", sp.stress_label(F.col("z"))).collect()
    assert [r.label for r in rows] == ["No reading", "Insufficient history", "Normal", "Watch", "Moderate Stress", "High Stress"]
