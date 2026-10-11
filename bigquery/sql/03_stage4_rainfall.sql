-- ============================================================================
-- Stage 4: Antecedent Rainfall Windows (1m, 3m, 6m ending strictly at M-1)
-- Target: gw_stress.stage4_rain
-- Leakage Rule: month M is NEVER included; any incomplete window yields NULL
-- ============================================================================

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage4_rain` AS
WITH obs AS (
  SELECT
    *,
    year * 12 + month - 1 AS obs_m
  FROM `river-data-511316-j4.gw_stress.stage3_clean`
),
rain AS (
  SELECT
    well_id,
    year * 12 + month - 1 AS src_m,
    precip_mm
  FROM `river-data-511316-j4.gw_stress.chirps_monthly`
),
windows AS (
  SELECT
    o.well_id,
    o.obs_m,
    -- 1-month: lag 1 (month M-1)
    SUM(CASE WHEN r.src_m = o.obs_m - 1 THEN r.precip_mm END) AS rain_1m_mm,
    COUNT(CASE WHEN r.src_m = o.obs_m - 1 AND r.precip_mm IS NOT NULL THEN 1 END) AS count_1m,
    -- 3-month: lags 1..3 (months M-3..M-1)
    SUM(CASE WHEN r.src_m BETWEEN o.obs_m - 3 AND o.obs_m - 1 THEN r.precip_mm END) AS rain_3m_mm,
    COUNT(CASE WHEN r.src_m BETWEEN o.obs_m - 3 AND o.obs_m - 1 AND r.precip_mm IS NOT NULL THEN 1 END) AS count_3m,
    -- 6-month: lags 1..6 (months M-6..M-1)
    SUM(CASE WHEN r.src_m BETWEEN o.obs_m - 6 AND o.obs_m - 1 THEN r.precip_mm END) AS rain_6m_mm,
    COUNT(CASE WHEN r.src_m BETWEEN o.obs_m - 6 AND o.obs_m - 1 AND r.precip_mm IS NOT NULL THEN 1 END) AS count_6m
  FROM obs o
  JOIN rain r
    ON o.well_id = r.well_id
   AND r.src_m BETWEEN o.obs_m - 6 AND o.obs_m - 1
  GROUP BY o.well_id, o.obs_m
)
SELECT
  o.well_id,
  o.date,
  o.year,
  o.month,
  o.season,
  o.period_label,
  o.gwl_m_bgl,
  o.state,
  o.district,
  o.station_name,
  o.lat,
  o.lon,
  o.well_type,
  o.aquifer_type,
  o.well_depth_m,
  o.specific_yield,
  o.flag_zero,
  o.flag_extreme,
  o.flag_negative,
  o.flag_deeper_than_well,
  CASE WHEN w.count_1m = 1 THEN w.rain_1m_mm ELSE NULL END AS rain_1m_mm,
  CASE WHEN w.count_3m = 3 THEN w.rain_3m_mm ELSE NULL END AS rain_3m_mm,
  CASE WHEN w.count_6m = 6 THEN w.rain_6m_mm ELSE NULL END AS rain_6m_mm
FROM obs o
LEFT JOIN windows w
  ON o.well_id = w.well_id AND o.obs_m = w.obs_m;

-- Assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage4_rain`) = 253828
  AS 'stage4_rain row count must match 253,828';
ASSERT NOT EXISTS (
  SELECT 1 FROM `river-data-511316-j4.gw_stress.stage4_rain`
  WHERE rain_1m_mm IS NULL OR rain_3m_mm IS NULL OR rain_6m_mm IS NULL
) AS 'All 1m, 3m, 6m rainfall windows must be non-null in main table';
