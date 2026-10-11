-- Stage 8: Evaluation splits matching config/splits.json and ref_splits
-- Rules:
-- 1. coverage_pct per round; is_sparse = coverage_pct < 70.0% (main)
-- 2. exclusion_reason: no_reading > insufficient_history > sparse_round
-- 3. eval_eligible: exclusion_reason IS NULL
-- 4. split_chrono: train (2000-2015), val (2016-2017), test (2018-2022)
-- 5. state_fold: 0..18 alphabetical
-- 6. no_may_state: well has 0 May readings across 2000-2022 (444 wells)

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.splits` AS
WITH
-- Coverage per round
round_cov AS (
  SELECT
    period_label,
    100.0 * COUNT(gwl_m_bgl) / 2759.0 AS coverage_pct,
    (100.0 * COUNT(gwl_m_bgl) / 2759.0 < 70.0) AS is_sparse
  FROM `river-data-511316-j4.gw_stress.final_features`
  GROUP BY period_label
),
-- Well no-may flag across 2000-2022
well_may AS (
  SELECT
    well_id,
    (COUNTIF(season = 'pre_monsoon' AND gwl_m_bgl IS NOT NULL) = 0) AS no_may_state
  FROM `river-data-511316-j4.gw_stress.final_features`
  GROUP BY well_id
),
-- Alphabetical state fold
state_ranks AS (
  SELECT
    state,
    DENSE_RANK() OVER (ORDER BY state) - 1 AS state_fold
  FROM (SELECT DISTINCT state FROM `river-data-511316-j4.gw_stress.final_features`)
),
base AS (
  SELECT
    f.well_id,
    f.period_label,
    f.date,
    f.year,
    f.season,
    'main' AS source,
    f.state,
    sr.state_fold,
    CASE
      WHEN f.year BETWEEN 2000 AND 2015 THEN 'train'
      WHEN f.year BETWEEN 2016 AND 2017 THEN 'val'
      WHEN f.year BETWEEN 2018 AND 2022 THEN 'test'
      ELSE 'ext'
    END AS split_chrono,
    CASE
      WHEN f.gwl_m_bgl IS NULL THEN 'no_reading'
      WHEN f.n_prior_same_season < 5 THEN 'insufficient_history'
      WHEN rc.is_sparse THEN 'sparse_round'
      ELSE NULL
    END AS exclusion_reason,
    wm.no_may_state
  FROM `river-data-511316-j4.gw_stress.final_features` f
  JOIN round_cov rc ON f.period_label = rc.period_label
  JOIN well_may wm ON f.well_id = wm.well_id
  JOIN state_ranks sr ON f.state = sr.state
)
SELECT
  well_id,
  period_label,
  date,
  year,
  season,
  source,
  state,
  state_fold,
  split_chrono,
  (exclusion_reason IS NULL) AS eval_eligible,
  exclusion_reason,
  no_may_state
FROM base
ORDER BY well_id, date;

-- Assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.splits`) = 253828
  AS "splits table must have exactly 253,828 rows";

ASSERT (SELECT COUNT(DISTINCT state_fold) FROM `river-data-511316-j4.gw_stress.splits`) = 19
  AS "splits must have 19 distinct state folds";

ASSERT (SELECT COUNT(DISTINCT well_id) FROM `river-data-511316-j4.gw_stress.splits` WHERE no_may_state) = 444
  AS "exactly 444 wells must have no_may_state = true";
