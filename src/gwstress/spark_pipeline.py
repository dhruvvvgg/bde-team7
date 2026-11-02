"""Spark-native preprocessing; pandas is only an independent validation oracle.

CSV ingestion, reshape, QC, rainfall alignment, historical windows, labels,
extension matching and evaluation splits execute as Spark DataFrame operations.
Small scalar/array Python UDFs run on executors for exact MAD, Theil-Sen and
Student-t mathematics. They never receive or collect a complete pandas table.
"""
from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F, types as T

from . import config

SEASONS = {1: "post_monsoon_rabi", 5: "pre_monsoon", 8: "monsoon", 11: "post_monsoon_kharif"}
MONTHS = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
KEY = ["well_id", "date"]
STATIC = {"State": "state", "District": "district", "Station Name": "station_name",
          "Latitude": "lat", "Longitude": "lon", "Type of Well": "well_type",
          "Aquifer Type": "aquifer_type", "Well Depth": "well_depth_m", "Reference_Sy": "specific_yield"}


def assert_unique(df: DataFrame, keys: list[str], name: str) -> None:
    if df.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError(f"{name}: duplicate keys {keys}")


def read_wide(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read metadata explicitly as text, avoiding further station-code rounding."""
    reader = spark.read.option("header", True).option("mode", "FAILFAST")
    # Header discovery reads no numeric data using inference. CSV headers are
    # available from Spark; all fields are initially strings, then cast below.
    raw = reader.csv(str(path))
    numeric = {"Latitude", "Longitude", "Well Depth", "Reference_Sy"}
    obs = [c for c in raw.columns if re.fullmatch(r"(Jan|May|Aug|Nov)-\d{2}", c)]
    if not obs:
        raise ValueError(f"no monitoring rounds in {path}")
    for c in numeric | set(obs):
        if c in raw.columns:
            raw = raw.withColumn(c, F.col(f"`{c}`").cast("double"))
    parts = [F.lower(F.trim(F.col(c))) for c in ["State", "District", "Station Name"]]
    parts += [F.format_string("%.6f", F.col(c)) for c in ["Latitude", "Longitude"]]
    return raw.withColumn("well_id", F.concat(F.lit("W"), F.substring(F.sha1(F.concat_ws("|", *parts)), 1, 10)))


def reshape(wide: DataFrame, first_year: int = 2000, last_year: int = 2022) -> DataFrame:
    obs = [c for c in wide.columns if re.fullmatch(r"(Jan|May|Aug|Nov)-\d{2}", c)
           and first_year <= 2000 + int(c[-2:]) <= last_year]
    if not obs:
        raise ValueError("no rounds in requested year interval")
    rounds = [F.struct(F.lit(c).alias("period_label"), F.lit(2000 + int(c[-2:])).alias("year"),
                       F.lit(MONTHS[c[:3]]).alias("month"), F.col(f"`{c}`").alias("gwl_m_bgl")) for c in obs]
    static = [F.col(f"`{old}`").alias(new) if old in wide.columns else F.lit(None).cast("double").alias(new)
              for old, new in STATIC.items()]
    d = wide.select("well_id", *static, F.explode(F.array(*rounds)).alias("round")).select("well_id", *STATIC.values(), "round.*")
    mapping = F.create_map(*[F.lit(x) for pair in SEASONS.items() for x in pair])
    return d.withColumn("season", mapping[F.col("month")]).withColumn("date", F.make_date("year", "month", F.lit(1)).cast("timestamp"))


def _normalise_name(c):
    value = F.regexp_replace(F.trim(c), r"\s+", " ")
    value = F.regexp_replace(value, r"\s*&\s*", " and ")
    return F.regexp_replace(F.initcap(value), r"\bAnd\b", "and")


def clean(d: DataFrame) -> DataFrame:
    d = d.dropDuplicates()
    assert_unique(d, KEY, "groundwater")
    x = F.col("gwl_m_bgl")
    valid = ~(F.isnan(x) | (x <= -900) | (F.abs(x) == float("inf")))
    return (d.withColumn("gwl_m_bgl", F.when(valid, x))
            .withColumn("state", _normalise_name(F.col("state")))
            .withColumn("district", _normalise_name(F.col("district")))
            .withColumn("aquifer_type", F.when(F.col("aquifer_type") != "-", F.col("aquifer_type")))
            .withColumn("flag_zero", F.coalesce(F.col("gwl_m_bgl") == 0, F.lit(False)))
            .withColumn("flag_negative", F.coalesce(F.col("gwl_m_bgl") < 0, F.lit(False)))
            .withColumn("flag_deeper_than_well", F.coalesce(F.col("gwl_m_bgl") > F.col("well_depth_m"), F.lit(False))))


@F.udf(T.DoubleType())
def _median(values):
    values = [float(v) for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def robust_flag(d: DataFrame) -> DataFrame:
    """Full-record QC only; never used as an input or by the feature windows."""
    g = ["well_id", "season"]
    med = d.groupBy(*g).agg(_median(F.collect_list("gwl_m_bgl")).alias("_median"))
    d = d.join(med, g)
    mad = d.groupBy(*g).agg(_median(F.collect_list(F.abs(F.col("gwl_m_bgl") - F.col("_median")))).alias("_mad"))
    return (d.join(mad, g).withColumn("flag_extreme", F.coalesce(
        (F.col("_mad") > 0) & (F.abs(0.6745 * (F.col("gwl_m_bgl") - F.col("_median")) / F.col("_mad")) > 3.5), F.lit(False)))
        .drop("_median", "_mad"))


def rainfall_features(d: DataFrame, rain: DataFrame) -> DataFrame:
    """Join explicit earlier source months and aggregate complete windows in Spark."""
    rain = rain.select("well_id", (F.col("year") * 12 + F.col("month") - 1).alias("_src_m"),
                       F.when(~F.isnan("precip_mm") & (F.col("precip_mm") >= 0), F.col("precip_mm").cast("double")).alias("_rain"))
    assert_unique(rain, ["well_id", "_src_m"], "rainfall")
    d = (d.withColumn("_obs_m", F.col("year") * 12 + F.col("month") - 1)
         .withColumn("last_monsoon_year", F.when(F.col("month") >= 10, F.col("year")).otherwise(F.col("year") - 1))
         .withColumn("last_ne_year", F.col("year") - 1)
         .withColumn("water_year", F.when(F.col("month") >= 6, F.col("year")).otherwise(F.col("year") - 1)))
    contrib = d.select(*KEY, "_obs_m", "last_monsoon_year", "last_ne_year", "water_year").withColumn(
        "_src_m", F.explode(F.sequence(F.col("_obs_m") - 23, F.col("_obs_m") - 1))).join(rain, ["well_id", "_src_m"], "left")
    definitions = {f"rain_{w}m_mm": (F.col("_obs_m") - w, F.col("_obs_m") - 1) for w in (1, 3, 6, 12)}
    definitions.update({"rain_last_monsoon_mm": (F.col("last_monsoon_year") * 12 + 5, F.col("last_monsoon_year") * 12 + 8),
                        "rain_last_ne_mm": (F.col("last_ne_year") * 12 + 9, F.col("last_ne_year") * 12 + 11),
                        "rain_monsoon_ytd_mm": (F.col("water_year") * 12 + 5, F.col("_obs_m") - 1)})
    sums = []
    for name, (start, end) in definitions.items():
        inside = (F.col("_src_m") >= start) & (F.col("_src_m") <= end)
        # This expression never contains M or a later month. Missing any
        # expected source value yields NULL, not an underestimated partial sum.
        total = F.sum(F.when(inside, F.col("_rain")))
        # Reproduce the reference's float32 Kahan sum in newest-first order.
        # All operations below are native Spark higher-order expressions;
        # sorting makes arithmetic deterministic after a distributed shuffle.
        if name in {"rain_1m_mm", "rain_3m_mm", "rain_6m_mm"}:
            entries = F.sort_array(F.collect_list(F.when(inside, F.struct(
                F.col("_src_m").alias("month"), F.col("_rain").cast("float").alias("value")))), asc=False)
            initial = F.struct(F.lit(0.0).cast("float").alias("total"),
                               F.lit(0.0).cast("float").alias("compensation"))
            def add(acc, entry):
                y = (entry["value"] - acc["compensation"]).cast("float")
                t = (acc["total"] + y).cast("float")
                correction = ((t - acc["total"]).cast("float") - y).cast("float")
                return F.struct(t.alias("total"), correction.alias("compensation"))
            total = F.aggregate(entries, initial, add)["total"]
        sums.append(F.when(F.count(F.when(inside, F.col("_rain"))) == F.first(end - start + 1), total).alias(name))
    windows = contrib.groupBy(*KEY).agg(*sums)
    d = d.join(windows, KEY)
    prior = Window.partitionBy("well_id", "season").orderBy("year").rowsBetween(Window.unboundedPreceding, -1)
    additions = {}
    for w in (1, 3, 6):
        c = F.col(f"rain_{w}m_mm")
        normal = F.when(F.count(c).over(prior) >= 5, F.avg(c).over(prior))
        additions[f"rain_{w}m_normal_mm"] = normal
        additions[f"rain_{w}m_dev_mm"] = c - normal
        additions[f"rain_{w}m_dev_pct"] = F.when(normal >= 5, 100 * (c - normal) / normal)
        if w in (3, 6):
            std = F.stddev_samp(c).over(prior)
            z = F.when((F.count(c).over(prior) >= 5) & (std >= 1), (c - normal) / std)
            additions[f"rain_{w}m_z"] = z
            additions[f"rain_{w}m_z_capped"] = _cap(z)
    d = d.withColumns(additions)
    for kind, yearcol, raincol in [("monsoon", "last_monsoon_year", "rain_last_monsoon_mm"),
                                   ("ne", "last_ne_year", "rain_last_ne_mm")]:
        yearly = d.select("well_id", yearcol, raincol).dropDuplicates(["well_id", yearcol])
        p = Window.partitionBy("well_id").orderBy(yearcol).rowsBetween(Window.unboundedPreceding, -1)
        yearly = yearly.withColumns({"_normal": F.when(F.count(raincol).over(p) >= 5, F.avg(raincol).over(p)),
                                     "_std": F.stddev_samp(raincol).over(p)})
        d = d.join(yearly.select("well_id", yearcol, "_normal", "_std"), ["well_id", yearcol])
        prefix = f"rain_last_{kind}"
        d = d.withColumn(prefix + "_dev_mm", F.col(raincol) - F.col("_normal"))
        if kind == "monsoon":
            d = d.withColumn(prefix + "_dev_pct", 100 * F.col(prefix + "_dev_mm") / F.col("_normal"))
        else:
            d = d.withColumn(prefix + "_z_capped", _cap(F.when(F.col("_std") >= 1,
                                                             F.col(prefix + "_dev_mm") / F.col("_std"))))
        d = d.drop("_normal", "_std")
    order = F.create_map(F.lit(8), F.lit(0), F.lit(11), F.lit(1), F.lit(1), F.lit(2), F.lit(5), F.lit(3))
    return d.withColumns({"water_year_index": F.col("water_year") - 1999, "season_order": order[F.col("month")]}).drop("_obs_m")


def _cap(c):
    # Spark least/greatest ignore NULL, unlike pandas clip: preserve it explicitly.
    return F.when(c.isNotNull(), F.greatest(F.lit(-5.0), F.least(F.lit(5.0), c)))


@F.udf(T.StructType([T.StructField("n", T.IntegerType()), T.StructField("mean", T.DoubleType()), T.StructField("std", T.DoubleType())]))
def _rolling_stats(values):
    v = [float(x) for x in values][-10:]
    return (len(v), statistics.mean(v) if len(v) >= 5 else None, statistics.stdev(v) if len(v) >= 5 else None)


@F.udf(T.DoubleType())
def _theil_sen(points):
    p = [(x["year"], x["value"]) for x in points if x["value"] is not None]
    if len(p) < 4:
        return None
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for i, a in enumerate(p) for b in p[i + 1:] if b[0] != a[0]]
    return float(statistics.median(slopes)) if slopes else None


@F.udf(T.DoubleType())
def _t_adjusted(x, mean, std, n):
    if x is None or mean is None or std is None or n < 5:
        return None
    from scipy import stats
    t = (x - mean) / (max(std, 0.10) * math.sqrt(1 + 1 / n))
    z = stats.norm.isf(stats.t.sf(t, n - 1)) if t >= 0 else stats.norm.ppf(stats.t.cdf(t, n - 1))
    return float(max(-38, min(38, z)))


def groundwater_features(d: DataFrame) -> DataFrame:
    """Native ordered windows exclude current rows; extension zeros are masked."""
    d = d.withColumn("_history", F.when(~F.col("_excluded"), F.col("gwl_m_bgl")))
    season = Window.partitionBy("well_id", "season").orderBy("year")
    prior = season.rowsBetween(Window.unboundedPreceding, -1)
    whole = Window.partitionBy("well_id", "season")
    well = Window.partitionBy("well_id").orderBy("date")
    wp = well.rowsBetween(Window.unboundedPreceding, -1)
    n = F.count("_history").over(prior)
    mean = F.when(n >= 5, F.avg("_history").over(prior))
    std = F.when(n >= 5, F.stddev_samp("_history").over(prior))
    d = d.withColumns({"n_prior_same_season": n, "baseline_mean_m": mean, "baseline_std_m": std,
                       "flag_std_floored": F.coalesce((n >= 5) & (std < 0.10), F.lit(False)),
                       "well_has_no_series_for_season": F.count("_history").over(whole) == 0,
                       "_rolling": _rolling_stats(F.collect_list("_history").over(prior)),
                       "trend_5y_m_per_yr": _theil_sen(F.collect_list(F.struct("year", F.col("_history").alias("value"))).over(season.rangeBetween(-5, -1))),
                       "gw_prev_round_m": F.lag("_history", 1).over(well),
                       "gw_last_obs_m": F.last("_history", ignorenulls=True).over(wp),
                       "gw_same_season_lag1y_m": F.lag("_history", 1).over(season),
                       "gw_same_season_lag2y_m": F.lag("_history", 2).over(season),
                       "gw_prev_change_m": F.lag("_history", 1).over(well) - F.lag("_history", 2).over(well),
                       "gw_max_to_date_m": F.max("_history").over(wp),
                       "gw_min_to_date_m": F.min("_history").over(wp),
                       "n_readings_to_date": F.count("_history").over(wp),
                       "_round": F.row_number().over(well),
                       "_last_obs_date": F.max(F.when(F.col("_history").isNotNull(), F.col("date"))).over(wp),
                       "well_completeness_full_period": F.avg(F.col("_history").isNotNull().cast("double")).over(Window.partitionBy("well_id"))})
    d = d.withColumns({"n_roll10": F.col("_rolling.n"), "baseline_roll10_mean_m": F.col("_rolling.mean"),
                       "baseline_roll10_std_m": F.col("_rolling.std"),
                       "flag_roll10_std_floored": F.coalesce(F.col("_rolling.std") < 0.10, F.lit(False)),
                       "completeness_to_date": F.when(F.col("_round") > 1, F.col("n_readings_to_date") / (F.col("_round") - 1))})
    # Monitoring-round index, not month differences: Jan->May is one round.
    mi = F.create_map(*[F.lit(x) for pair in {1: 0, 5: 1, 8: 2, 11: 3}.items() for x in pair])
    d = d.withColumn("rounds_since_last_obs", F.when(F.col("_last_obs_date").isNotNull(),
        (F.col("year") - F.year("_last_obs_date")) * 4 + mi[F.col("month")] - mi[F.month("_last_obs_date")]).cast("double"))
    yearly = d.groupBy("well_id", "year").agg(F.max(F.when(F.col("month") == 5, F.col("_history"))).alias("_may"),
                                                F.max(F.when(F.col("month") == 11, F.col("_history"))).alias("_nov"))
    yearly = yearly.select("well_id", (F.col("year") + 1).alias("year"), (F.col("_nov") - F.col("_may")).alias("gw_fluct_prev_wy_m"))
    d = d.join(yearly, ["well_id", "year"], "left")
    d = d.withColumns({"anomaly_z": F.when(F.col("baseline_std_m").isNotNull(),
        (F.col("gwl_m_bgl") - F.col("baseline_mean_m")) / F.greatest(F.col("baseline_std_m"), F.lit(0.10))),
        "anomaly_roll10": F.when(F.col("baseline_roll10_std_m").isNotNull(),
        (F.col("gwl_m_bgl") - F.col("baseline_roll10_mean_m")) / F.greatest(F.col("baseline_roll10_std_m"), F.lit(0.10))),
        "anomaly_tadj": _t_adjusted("gwl_m_bgl", "baseline_mean_m", "baseline_std_m", "n_prior_same_season"),
        "gw_delta_from_prev_round_m": F.col("gwl_m_bgl") - F.col("gw_prev_round_m")})
    return d.withColumns({c + "_capped": _cap(F.col(c)) for c in ["anomaly_z", "anomaly_roll10", "anomaly_tadj"]}).drop(
        "_history", "_rolling", "_round", "_last_obs_date")


def stress_label(z):
    return (F.when(F.col("gwl_m_bgl").isNull(), "No reading").when(z.isNull(), "Insufficient history")
            .when(z >= 2, "High Stress").when(z >= 1.5, "Moderate Stress").when(z >= 1, "Watch").otherwise("Normal"))


def labels(d: DataFrame) -> DataFrame:
    d = d.withColumns({"stress_category": stress_label(F.col("anomaly_z")),
                       "stress_roll10": stress_label(F.col("anomaly_roll10")), "stress_tadj": stress_label(F.col("anomaly_tadj"))})
    zero_suspicious = F.lit(False)
    for wid, period in config.SUSPICIOUS_ZEROS:
        zero_suspicious = zero_suspicious | ((F.col("well_id") == wid) & (F.col("period_label") == period))
    reason = F.concat_ws(";", F.when(zero_suspicious, "zero_suspicious"),
                         F.when(F.col("flag_deeper_than_well"), "deeper_than_well"),
                         F.when(F.abs(F.col("anomaly_z")) > 5, "extreme_anomaly"), F.when(F.col("_excluded"), "zero_extension"))
    return d.withColumn("suspect_reason", F.when(reason != "", reason)).withColumn("suspect_reading", F.col("suspect_reason").isNotNull())


def match_extension(main: DataFrame, source: DataFrame) -> DataFrame:
    """Resolve duplicate candidate wells against all 92 main-series readings."""
    obs = [c for c in main.columns if re.fullmatch(r"(Jan|May|Aug|Nov)-\d{2}", c)]
    ref = main.select("well_id", *[F.col(f"`{c}`").alias("ref_" + c) for c in obs])
    candidates = source.join(ref, "well_id")
    same = F.lit(True)
    for c in obs:
        a, b = F.col(f"`{c}`"), F.col(f"`ref_{c}`")
        # Same tolerance as numpy.isclose in the existing matching code.
        same = same & (a.eqNullSafe(b) | (F.abs(a - b) <= 1e-8 + 1e-5 * F.abs(b)))
    matched = candidates.filter(same).select(*[F.col(f"`{c}`") for c in source.columns])
    assert_unique(matched, ["well_id"], "extension matches")
    if main.select("well_id").join(matched.select("well_id"), "well_id", "left_anti").limit(1).count():
        raise ValueError("unmatched main well in extension")
    return matched


def build_splits(main: DataFrame, ext: DataFrame | None, cfg: dict) -> DataFrame:
    cols = ["well_id", "date", "period_label", "year", "season", "state", "gwl_m_bgl", "n_prior_same_season", "suspect_reason"]
    d = main.select(*cols).withColumn("source", F.lit("main"))
    if ext is not None:
        d = d.unionByName(ext.select(*cols).withColumn("source", F.lit("extension")))
    coverage = d.groupBy("source", "period_label").agg((100 * F.avg(F.col("gwl_m_bgl").isNotNull().cast("double"))).alias("_coverage"))
    d = d.join(coverage, ["source", "period_label"])
    threshold = F.when(F.col("source") == "main", cfg["sparse_round_min_coverage_pct"]).otherwise(cfg["ext_sparse_round_min_coverage_pct"])
    reason = (F.when(F.col("gwl_m_bgl").isNull(), "no_reading")
              .when((F.col("source") == "extension") & F.col("suspect_reason").contains("zero_extension"), "extension_zero_placeholder")
              .when(F.col("n_prior_same_season") < cfg["K_min_history"], "insufficient_history")
              .when(F.col("_coverage") < threshold, "sparse_round"))
    d = d.withColumn("exclusion_reason", reason).withColumn("eval_eligible", F.col("exclusion_reason").isNull())
    split = F.lit(None).cast("string")
    for name, (lo, hi) in cfg["chrono"].items():
        split = F.when(F.col("year").between(lo, hi), name).otherwise(split)
    d = d.withColumn("split_chrono", split)
    if d.filter(F.col("split_chrono").isNull()).limit(1).count():
        raise ValueError("a year is not covered by chronological splits")
    # Only 19 state names are collected; no well-round data is brought to Python.
    states = sorted(r.state for r in d.select("state").distinct().collect())
    mapping = F.create_map(*[F.lit(x) for i, s in enumerate(states) for x in (s, i)])
    no_may = main.filter(F.col("month") == 5).groupBy("well_id").agg((F.count("gwl_m_bgl") == 0).alias("no_may_state"))
    return d.join(no_may, "well_id").withColumn("state_fold", mapping[F.col("state")].cast("byte")).select(
        "well_id", "period_label", "date", "year", "season", "source", "state", "state_fold", "split_chrono",
        "eval_eligible", "exclusion_reason", "no_may_state")


def build(d: DataFrame, rain: DataFrame) -> DataFrame:
    d = robust_flag(clean(d))
    d = d.withColumn("_excluded", (F.col("year") >= 2023) & F.col("flag_zero"))
    d = rainfall_features(d, rain).checkpoint(eager=True)
    return labels(groundwater_features(d)).drop("_excluded")


def final_columns(d: DataFrame, extension: bool = False) -> DataFrame:
    from .contract import COLUMNS
    if extension:
        d = d.withColumn("is_extension", F.lit(True))
        names = [c[0] for c in COLUMNS if c[0] != "state_slug"] + ["is_extension"]
    else:
        d = d.withColumn("state_slug", F.regexp_replace(F.lower("state"), "[^a-z0-9]+", "_"))
        names = [c[0] for c in COLUMNS]
    return d.select(*names)
