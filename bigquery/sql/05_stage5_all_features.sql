-- Stage 5: Time-aware feature engineering (Theil-Sen, Student-t adjustment, history, rainfall extras, calendar)
-- Leakage guard: past-only rows (ends at 1 PRECEDING or M-1)

CREATE OR REPLACE FUNCTION `river-data-511316-j4.gw_stress.theil_sen_slope`(years ARRAY<INT64>, vals ARRAY<FLOAT64>)
RETURNS FLOAT64
LANGUAGE js AS r"""
  if (!years || !vals || years.length === 0 || vals.length === 0) return null;
  const pts = [];
  for (let i = 0; i < years.length; i++) {
    const y = years[i];
    const v = vals[i];
    if (y !== null && v !== null && !isNaN(y) && !isNaN(v)) {
      pts.push({ x: Number(y), y: Number(v) });
    }
  }
  if (pts.length < 4) return null;
  pts.sort((a, b) => a.x - b.x);
  const slopes = [];
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      slopes.push((pts[j].y - pts[i].y) / (pts[j].x - pts[i].x));
    }
  }
  slopes.sort((a, b) => a - b);
  const n = slopes.length;
  if (n % 2 === 1) {
    return slopes[Math.floor(n / 2)];
  } else {
    return (slopes[n / 2 - 1] + slopes[n / 2]) / 2.0;
  }
""";

CREATE OR REPLACE FUNCTION `river-data-511316-j4.gw_stress.student_t_adj_anomaly`(gwl FLOAT64, mean FLOAT64, std_val FLOAT64, n INT64)
RETURNS FLOAT64
LANGUAGE js AS r"""
  if (gwl === null || mean === null || std_val === null || n === null || n < 5) return null;
  const s_eff = Math.max(std_val, 0.10);
  const t = (gwl - mean) / (s_eff * Math.sqrt(1.0 + 1.0 / n));
  if (isNaN(t)) return null;

  function logGamma(z) {
    const c = [
      0.99999999999980993,
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109586547999,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ];
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    let base = z + 7.5;
    let sum = c[0];
    for (let i = 1; i < c.length; i++) {
      sum += c[i] / (z + i);
    }
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(base) - base + Math.log(sum);
  }

  function betacf(a, b, x) {
    const MAXIT = 200;
    const EPS = 3.0e-12;
    const FPMIN = 1.0e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let c = 1.0;
    let d = 1.0 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      let m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      let del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) <= EPS) break;
    }
    return h;
  }

  function ibeta(x, a, b) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    let bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
      return bt * betacf(a, b, x) / a;
    } else {
      return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }
  }

  function wichura_as241(p) {
    let q = p - 0.5;
    let r, num, den, x;
    if (Math.abs(q) <= 0.425) {
      r = 0.180625 - q * q;
      num = (((((((2.5090809287301226727e+3 * r +
                   3.3430575583588128105e+4) * r +
                   6.7265770927008700853e+4) * r +
                   4.5921953931549871457e+4) * r +
                   1.3731693765509461125e+4) * r +
                   1.9715909503065514427e+3) * r +
                   1.3314166789178437745e+2) * r +
                   3.3871328727963666080e+0) * q;
      den = (((((((5.2264952788528545610e+3 * r +
                   2.8729085735721942674e+4) * r +
                   3.9307895800092710610e+4) * r +
                   2.1213794301586595867e+4) * r +
                   5.3941960214247511077e+3) * r +
                   6.8718700749205790830e+2) * r +
                   4.2313330701600911252e+1) * r +
                   1.0);
      return num / den;
    }
    r = q <= 0.0 ? p : 1.0 - p;
    r = Math.sqrt(-Math.log(r));
    if (r <= 5.0) {
      r = r - 1.6;
      num = (((((((7.74545014278341407640e-4 * r +
                   2.27238449892691845833e-2) * r +
                   2.41780725177450611770e-1) * r +
                   1.27045825245236838258e+0) * r +
                   3.64784832476320460504e+0) * r +
                   5.76949722146069140550e+0) * r +
                   4.63033784615654529590e+0) * r +
                   1.42343711074968357734e+0);
      den = (((((((1.05075007164441684324e-9 * r +
                   5.47593808499534494600e-4) * r +
                   1.51986665636164571966e-2) * r +
                   1.48103976427480074590e-1) * r +
                   6.89767334985100004550e-1) * r +
                   1.67638483018380384940e+0) * r +
                   2.05319162663775882187e+0) * r +
                   1.0);
    } else {
      r = r - 5.0;
      num = (((((((2.01033439929228813265e-7 * r +
                   2.71155556874348757815e-5) * r +
                   1.24266094738807843860e-3) * r +
                   2.65321895265761230930e-2) * r +
                   2.96560571828504891230e-1) * r +
                   1.78482653991729133580e+0) * r +
                   5.46378491116411436990e+0) * r +
                   6.65790464350110377720e+0);
      den = (((((((2.04426310338993978564e-15 * r +
                   1.42151175831644588870e-7) * r +
                   1.84631831751005468180e-5) * r +
                   7.86869131145613259100e-4) * r +
                   1.48753612908506148525e-2) * r +
                   1.36929880922735805310e-1) * r +
                   5.99832206555887937690e-1) * r +
                   1.0);
    }
    x = num / den;
    return q < 0.0 ? -x : x;
  }

  const dof = n - 1;
  const x = dof / (dof + t * t);
  const p = 0.5 * ibeta(x, 0.5 * dof, 0.5);
  let z;
  if (t === 0.0) {
    z = 0.0;
  } else if (t > 0.0) {
    z = -wichura_as241(p);
  } else {
    z = wichura_as241(p);
  }
  if (z > 38.0) z = 38.0;
  if (z < -38.0) z = -38.0;
  return z;
""";

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage5_all_features` AS
WITH
-- 1. Groundwater history lags and expanding bounds
gw_raw_pos AS (
  SELECT
    well_id,
    period_label,
    date,
    gwl_m_bgl,
    ROW_NUMBER() OVER (PARTITION BY well_id ORDER BY date) - 1 AS pos
  FROM `river-data-511316-j4.gw_stress.stage2_long`
),
gw_history AS (
  SELECT
    well_id,
    period_label,
    date,
    gwl_m_bgl,
    LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) AS gw_prev_round_m,
    LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) - LAG(gwl_m_bgl, 2) OVER (PARTITION BY well_id ORDER BY date) AS gw_prev_change_m,
    gwl_m_bgl - LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) AS gw_delta_from_prev_round_m,
    LAG(gwl_m_bgl, 4) OVER (PARTITION BY well_id ORDER BY date) AS gw_same_season_lag1y_m,
    LAG(gwl_m_bgl, 8) OVER (PARTITION BY well_id ORDER BY date) AS gw_same_season_lag2y_m,
    IF(LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) IS NOT NULL,
       MAX(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), NULL) AS gw_max_to_date_m,
    IF(LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) IS NOT NULL,
       MIN(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), NULL) AS gw_min_to_date_m,
    COUNT(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_readings_to_date,
    LAST_VALUE(gwl_m_bgl IGNORE NULLS) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS gw_last_obs_m,
    pos - LAST_VALUE(CASE WHEN gwl_m_bgl IS NOT NULL THEN pos END IGNORE NULLS)
      OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS rounds_since_last_obs
  FROM gw_raw_pos
),

-- 2. Fluctuation of previous water year: gw(Nov, Y-1) - gw(May, Y-1)
gw_fluct AS (
  SELECT
    well_id,
    year + 1 AS for_year,
    MAX(CASE WHEN season = 'post_monsoon_kharif' THEN gwl_m_bgl END) -
    MAX(CASE WHEN season = 'pre_monsoon' THEN gwl_m_bgl END) AS gw_fluct_prev_wy_m
  FROM `river-data-511316-j4.gw_stress.stage2_long`
  GROUP BY well_id, year
),

-- 3. Theil-Sen trend slope arrays
theil_sen_prep AS (
  SELECT
    well_id,
    period_label,
    ARRAY_AGG(year) OVER w AS yrs,
    ARRAY_AGG(gwl_m_bgl) OVER w AS vals
  FROM `river-data-511316-j4.gw_stress.stage2_long`
  WINDOW w AS (
    PARTITION BY well_id, season
    ORDER BY year
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
  )
),

-- 4. CHIRPS monthly rolling sums ending at M-1
chirps_prep AS (
  SELECT
    well_id,
    year,
    month,
    -- 12m sum ending at month
    SUM(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS sum_12m,
    COUNT(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS cnt_12m,
    -- 2m sum ending at month (Aug round)
    SUM(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS sum_2m,
    COUNT(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS cnt_2m,
    -- 5m sum ending at month (Nov round)
    SUM(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS sum_5m,
    COUNT(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS cnt_5m,
    -- 7m sum ending at month (Jan round)
    SUM(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS sum_7m,
    COUNT(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS cnt_7m,
    -- 11m sum ending at month (May round)
    SUM(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS sum_11m,
    COUNT(precip_mm) OVER (PARTITION BY well_id ORDER BY year * 12 + month ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS cnt_11m
  FROM `river-data-511316-j4.gw_stress.chirps_monthly`
),

-- 5. Monsoon normal and deviation
chirps_monsoon AS (
  SELECT
    well_id,
    year AS mon_year,
    IF(COUNT(precip_mm) = 4, SUM(precip_mm), NULL) AS mon_rain
  FROM `river-data-511316-j4.gw_stress.chirps_monthly`
  WHERE month IN (6, 7, 8, 9)
  GROUP BY well_id, year
),
monsoon_stats AS (
  SELECT
    well_id,
    mon_year,
    mon_rain,
    AVG(mon_rain) OVER (
      PARTITION BY well_id ORDER BY mon_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS mon_mean,
    COUNT(mon_rain) OVER (
      PARTITION BY well_id ORDER BY mon_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS mon_n
  FROM chirps_monsoon
),

-- 6. North-East monsoon normal and deviation
chirps_ne AS (
  SELECT
    well_id,
    year AS ne_year,
    IF(COUNT(precip_mm) = 3, SUM(precip_mm), NULL) AS ne_rain
  FROM `river-data-511316-j4.gw_stress.chirps_monthly`
  WHERE month IN (10, 11, 12)
  GROUP BY well_id, year
),
ne_stats AS (
  SELECT
    well_id,
    ne_year,
    ne_rain,
    AVG(ne_rain) OVER (
      PARTITION BY well_id ORDER BY ne_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS ne_mean,
    STDDEV_SAMP(ne_rain) OVER (
      PARTITION BY well_id ORDER BY ne_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS ne_std,
    COUNT(ne_rain) OVER (
      PARTITION BY well_id ORDER BY ne_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS ne_n
  FROM chirps_ne
),

-- 7. Rain 1m, 3m and 6m expanding normals and z-stats
rain_z_stats AS (
  SELECT
    well_id,
    period_label,
    -- 1m
    COUNT(rain_1m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r1_n,
    AVG(rain_1m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r1_mean,
    -- 3m
    COUNT(rain_3m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r3_n,
    AVG(rain_3m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r3_mean,
    STDDEV_SAMP(rain_3m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r3_std,
    -- 6m
    COUNT(rain_6m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r6_n,
    AVG(rain_6m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r6_mean,
    STDDEV_SAMP(rain_6m_mm) OVER (PARTITION BY well_id, season ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS r6_std
  FROM `river-data-511316-j4.gw_stress.stage5_baselines`
)

SELECT
  b.well_id,
  b.date,
  b.year,
  b.month,
  b.season,
  b.period_label,
  b.gwl_m_bgl,
  b.state,
  b.district,
  b.station_name,
  b.lat,
  b.lon,
  b.well_type,
  b.aquifer_type,
  b.well_depth_m,
  b.specific_yield,
  b.flag_zero,
  b.flag_extreme,
  b.flag_negative,
  b.flag_deeper_than_well,
  b.rain_1m_mm,
  b.rain_3m_mm,
  b.rain_6m_mm,
  b.n_prior_same_season,
  b.baseline_mean_m,
  b.baseline_std_m,
  b.flag_std_floored,
  b.anomaly_z,
  b.anomaly_z_capped,
  b.well_has_no_series_for_season,
  b.n_roll10,
  b.baseline_roll10_mean_m,
  b.baseline_roll10_std_m,
  b.flag_roll10_std_floored,
  b.anomaly_roll10,
  b.anomaly_roll10_capped,
  
  -- Student t adjustment
  `river-data-511316-j4.gw_stress.student_t_adj_anomaly`(b.gwl_m_bgl, b.baseline_mean_m, b.baseline_std_m, b.n_prior_same_season) AS anomaly_tadj,
  LEAST(5.0, GREATEST(-5.0, `river-data-511316-j4.gw_stress.student_t_adj_anomaly`(b.gwl_m_bgl, b.baseline_mean_m, b.baseline_std_m, b.n_prior_same_season))) AS anomaly_tadj_capped,
  
  -- Theil-Sen trend
  `river-data-511316-j4.gw_stress.theil_sen_slope`(ts.yrs, ts.vals) AS trend_5y_m_per_yr,
  
  -- Rainfall expanding normals & deviations
  IF(rz.r1_n >= 5, rz.r1_mean, NULL) AS rain_1m_normal_mm,
  IF(rz.r1_n >= 5, b.rain_1m_mm - rz.r1_mean, NULL) AS rain_1m_dev_mm,
  IF(rz.r1_n >= 5 AND rz.r1_mean >= 5.0, 100.0 * (b.rain_1m_mm - rz.r1_mean) / rz.r1_mean, NULL) AS rain_1m_dev_pct,
  IF(rz.r3_n >= 5, rz.r3_mean, NULL) AS rain_3m_normal_mm,
  IF(rz.r3_n >= 5, b.rain_3m_mm - rz.r3_mean, NULL) AS rain_3m_dev_mm,
  IF(rz.r3_n >= 5 AND rz.r3_mean >= 5.0, 100.0 * (b.rain_3m_mm - rz.r3_mean) / rz.r3_mean, NULL) AS rain_3m_dev_pct,
  IF(rz.r6_n >= 5, rz.r6_mean, NULL) AS rain_6m_normal_mm,
  IF(rz.r6_n >= 5, b.rain_6m_mm - rz.r6_mean, NULL) AS rain_6m_dev_mm,
  IF(rz.r6_n >= 5 AND rz.r6_mean >= 5.0, 100.0 * (b.rain_6m_mm - rz.r6_mean) / rz.r6_mean, NULL) AS rain_6m_dev_pct,
  
  -- Completeness
  b.completeness_to_date,
  b.well_completeness_full_period,
  
  -- Groundwater history
  gwh.gw_prev_round_m,
  gwh.gw_prev_change_m,
  gwh.gw_delta_from_prev_round_m,
  gwh.gw_same_season_lag1y_m,
  gwh.gw_same_season_lag2y_m,
  gwh.gw_max_to_date_m,
  gwh.gw_min_to_date_m,
  gwh.n_readings_to_date,
  gwh.gw_last_obs_m,
  CAST(gwh.rounds_since_last_obs AS FLOAT64) AS rounds_since_last_obs,
  
  -- Monsoon recharge fluctuation
  gwf.gw_fluct_prev_wy_m,
  
  -- 12m rainfall
  IF(cp.cnt_12m = 12, cp.sum_12m, NULL) AS rain_12m_mm,
  
  -- Monsoon rainfall features
  last_mon_y.last_monsoon_year,
  ms.mon_rain AS rain_last_monsoon_mm,
  IF(ms.mon_n >= 5, ms.mon_rain - ms.mon_mean, NULL) AS rain_last_monsoon_dev_mm,
  IF(ms.mon_n >= 5, 100.0 * (ms.mon_rain - ms.mon_mean) / ms.mon_mean, NULL) AS rain_last_monsoon_dev_pct,
  
  -- Rain 3m and 6m z
  IF(rz.r3_n >= 5 AND rz.r3_std >= 1.0, (b.rain_3m_mm - rz.r3_mean) / rz.r3_std, NULL) AS rain_3m_z,
  IF(rz.r3_n >= 5 AND rz.r3_std >= 1.0, LEAST(5.0, GREATEST(-5.0, (b.rain_3m_mm - rz.r3_mean) / rz.r3_std)), NULL) AS rain_3m_z_capped,
  IF(rz.r6_n >= 5 AND rz.r6_std >= 1.0, (b.rain_6m_mm - rz.r6_mean) / rz.r6_std, NULL) AS rain_6m_z,
  IF(rz.r6_n >= 5 AND rz.r6_std >= 1.0, LEAST(5.0, GREATEST(-5.0, (b.rain_6m_mm - rz.r6_mean) / rz.r6_std)), NULL) AS rain_6m_z_capped,
  
  -- North-East monsoon features
  b.year - 1 AS last_ne_year,
  nes.ne_rain AS rain_last_ne_mm,
  IF(nes.ne_n >= 5, nes.ne_rain - nes.ne_mean, NULL) AS rain_last_ne_dev_mm,
  IF(nes.ne_n >= 5 AND nes.ne_std >= 1.0, LEAST(5.0, GREATEST(-5.0, (nes.ne_rain - nes.ne_mean) / nes.ne_std)), NULL) AS rain_last_ne_z_capped,
  
  -- Rain monsoon YTD
  CASE b.month
    WHEN 8 THEN IF(cp.cnt_2m = 2, cp.sum_2m, NULL)
    WHEN 11 THEN IF(cp.cnt_5m = 5, cp.sum_5m, NULL)
    WHEN 1 THEN IF(cp.cnt_7m = 7, cp.sum_7m, NULL)
    WHEN 5 THEN IF(cp.cnt_11m = 11, cp.sum_11m, NULL)
  END AS rain_monsoon_ytd_mm,
  
  -- Calendar
  IF(b.month >= 6, b.year, b.year - 1) AS water_year,
  (IF(b.month >= 6, b.year, b.year - 1) - 1999) AS water_year_index,
  CASE b.month WHEN 8 THEN 0 WHEN 11 THEN 1 WHEN 1 THEN 2 WHEN 5 THEN 3 END AS season_order

FROM `river-data-511316-j4.gw_stress.stage5_baselines` b
JOIN gw_history gwh ON b.well_id = gwh.well_id AND b.period_label = gwh.period_label
LEFT JOIN gw_fluct gwf ON b.well_id = gwf.well_id AND b.year = gwf.for_year
JOIN theil_sen_prep ts ON b.well_id = ts.well_id AND b.period_label = ts.period_label
CROSS JOIN UNNEST([STRUCT(IF(b.month >= 10, b.year, b.year - 1) AS last_monsoon_year)]) AS last_mon_y
LEFT JOIN monsoon_stats ms ON b.well_id = ms.well_id AND last_mon_y.last_monsoon_year = ms.mon_year
LEFT JOIN ne_stats nes ON b.well_id = nes.well_id AND (b.year - 1) = nes.ne_year
JOIN rain_z_stats rz ON b.well_id = rz.well_id AND b.period_label = rz.period_label
CROSS JOIN UNNEST([STRUCT(
  CASE WHEN b.month = 1 THEN b.year - 1 ELSE b.year END AS prev_m_year,
  CASE WHEN b.month = 1 THEN 12 ELSE b.month - 1 END AS prev_m_month
)]) AS prev_m
LEFT JOIN chirps_prep cp ON b.well_id = cp.well_id AND prev_m.prev_m_year = cp.year AND prev_m.prev_m_month = cp.month;

-- Leakage and integrity assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage5_all_features`) = 253828
  AS "stage5_all_features must have exactly 253,828 rows";

ASSERT (SELECT COUNT(DISTINCT CONCAT(well_id, ':', period_label)) FROM `river-data-511316-j4.gw_stress.stage5_all_features`) = 253828
  AS "stage5_all_features must have unique (well_id, period_label)";
