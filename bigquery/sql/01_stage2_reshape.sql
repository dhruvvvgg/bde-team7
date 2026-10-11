-- ============================================================================
-- Stage 2: Wide-to-Long Reshape, Surrogate Key Generation, and Period Parsing
-- Target: gw_stress.stage2_long
-- Rows expected: 253,828 (2,759 wells x 92 rounds)
-- ============================================================================

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage2_long` AS
WITH unpivoted AS (
  SELECT
    `Station Code` AS station_code_corrupted,
    `Station Name` AS station_name,
    `State` AS state,
    `District` AS district,
    CAST(`Latitude` AS FLOAT64) AS lat,
    CAST(`Longitude` AS FLOAT64) AS lon,
    `Type of Well` AS well_type,
    `Aquifer Type` AS aquifer_type,
    CAST(`Well Depth` AS FLOAT64) AS well_depth_m,
    CAST(`Reference_Sy` AS FLOAT64) AS specific_yield,
    period_label,
    CAST(gwl_val AS FLOAT64) AS gwl_m_bgl
  FROM `river-data-511316-j4.gw_stress.raw_gwl_wide`
  UNPIVOT INCLUDE NULLS (
    gwl_val FOR period_label IN (
      `Jan-00`, `May-00`, `Aug-00`, `Nov-00`,
      `Jan-01`, `May-01`, `Aug-01`, `Nov-01`,
      `Jan-02`, `May-02`, `Aug-02`, `Nov-02`,
      `Jan-03`, `May-03`, `Aug-03`, `Nov-03`,
      `Jan-04`, `May-04`, `Aug-04`, `Nov-04`,
      `Jan-05`, `May-05`, `Aug-05`, `Nov-05`,
      `Jan-06`, `May-06`, `Aug-06`, `Nov-06`,
      `Jan-07`, `May-07`, `Aug-07`, `Nov-07`,
      `Jan-08`, `May-08`, `Aug-08`, `Nov-08`,
      `Jan-09`, `May-09`, `Aug-09`, `Nov-09`,
      `Jan-10`, `May-10`, `Aug-10`, `Nov-10`,
      `Jan-11`, `May-11`, `Aug-11`, `Nov-11`,
      `Jan-12`, `May-12`, `Aug-12`, `Nov-12`,
      `Jan-13`, `May-13`, `Aug-13`, `Nov-13`,
      `Jan-14`, `May-14`, `Aug-14`, `Nov-14`,
      `Jan-15`, `May-15`, `Aug-15`, `Nov-15`,
      `Jan-16`, `May-16`, `Aug-16`, `Nov-16`,
      `Jan-17`, `May-17`, `Aug-17`, `Nov-17`,
      `Jan-18`, `May-18`, `Aug-18`, `Nov-18`,
      `Jan-19`, `May-19`, `Aug-19`, `Nov-19`,
      `Jan-20`, `May-20`, `Aug-20`, `Nov-20`,
      `Jan-21`, `May-21`, `Aug-21`, `Nov-21`,
      `Jan-22`, `May-22`, `Aug-22`, `Nov-22`
    )
  )
)
SELECT
  CONCAT('W', SUBSTR(TO_HEX(SHA1(CONCAT(
    TRIM(LOWER(state)), '|',
    TRIM(LOWER(district)), '|',
    TRIM(LOWER(station_name)), '|',
    FORMAT('%.6f', lat), '|',
    FORMAT('%.6f', lon)
  ))), 1, 10)) AS well_id,
  TIMESTAMP(DATE(
    2000 + CAST(SPLIT(period_label, '-')[OFFSET(1)] AS INT64),
    CASE SPLIT(period_label, '-')[OFFSET(0)]
      WHEN 'Jan' THEN 1
      WHEN 'May' THEN 5
      WHEN 'Aug' THEN 8
      WHEN 'Nov' THEN 11
    END,
    1
  )) AS date,
  2000 + CAST(SPLIT(period_label, '-')[OFFSET(1)] AS INT64) AS year,
  CASE SPLIT(period_label, '-')[OFFSET(0)]
    WHEN 'Jan' THEN 1
    WHEN 'May' THEN 5
    WHEN 'Aug' THEN 8
    WHEN 'Nov' THEN 11
  END AS month,
  CASE SPLIT(period_label, '-')[OFFSET(0)]
    WHEN 'Jan' THEN 'post_monsoon_rabi'
    WHEN 'May' THEN 'pre_monsoon'
    WHEN 'Aug' THEN 'monsoon'
    WHEN 'Nov' THEN 'post_monsoon_kharif'
  END AS season,
  period_label,
  gwl_m_bgl,
  state,
  district,
  station_name,
  lat,
  lon,
  well_type,
  aquifer_type,
  well_depth_m,
  specific_yield
FROM unpivoted;

-- Verify row count
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage2_long`) = 253828
  AS 'stage2_long row count must be exactly 253,828';
