CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR_trimmed_variable_distribution_summary` AS

WITH mapping AS (
  SELECT DISTINCT
    TRIM(CAST(eicu_source AS STRING)) AS source,
    TRIM(CAST(eicu_label AS STRING)) AS label,
    LOWER(TRIM(CAST(eicu_type AS STRING))) AS eicu_type
  FROM `mimic-iv-484300.eICU_processed.eicu_mimic_unit_mapping_rules`
  WHERE LOWER(TRIM(CAST(conversion_status AS STRING))) = 'convertible'
),

base AS (
  SELECT
    e.patientunitstayid,
    e.source,
    TRIM(CAST(e.label AS STRING)) AS label,
    NULLIF(TRIM(CAST(e.value AS STRING)), '') AS value_clean,
    m.eicu_type,

    CASE
      WHEN REGEXP_CONTAINS(m.eicu_type, r'numeric') THEN 'numeric'
      WHEN REGEXP_CONTAINS(m.eicu_type, r'categorical') THEN 'categorical'
      ELSE 'unknown'
    END AS variable_type,

    SAFE_CAST(
      REGEXP_EXTRACT(
        NULLIF(TRIM(CAST(e.value AS STRING)), ''),
        r'^[<>~= ]*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)'
      ) AS FLOAT64
    ) AS numeric_value

  FROM `mimic-iv-484300.eICU_processed.EHR_trimmed` e
  LEFT JOIN mapping m
    ON e.source = m.source
   AND TRIM(CAST(e.label AS STRING)) = m.label
),

variable_base_stats AS (
  SELECT
    source,
    label,
    ANY_VALUE(eicu_type) AS eicu_type,
    ANY_VALUE(variable_type) AS variable_type,

    COUNT(*) AS n_events,
    COUNT(DISTINCT patientunitstayid) AS n_icu_stays,
    COUNTIF(value_clean IS NOT NULL) AS n_non_null_values,
    COUNT(DISTINCT value_clean) AS n_distinct_raw_values,

    COUNTIF(variable_type = 'numeric' AND numeric_value IS NOT NULL) AS n_numeric_parsed_values,
    SAFE_DIVIDE(
      COUNTIF(variable_type = 'numeric' AND numeric_value IS NOT NULL),
      COUNTIF(variable_type = 'numeric' AND value_clean IS NOT NULL)
    ) AS numeric_parse_fraction

  FROM base
  GROUP BY source, label
),

numeric_stats AS (
  SELECT
    source,
    label,

    MIN(numeric_value) AS numeric_min,
    AVG(numeric_value) AS numeric_mean,
    STDDEV(numeric_value) AS numeric_sd,

    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(1)] AS numeric_p001,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(5)] AS numeric_p005,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(10)] AS numeric_p01,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(25)] AS numeric_p025,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(50)] AS numeric_p05,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(100)] AS numeric_p10,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(250)] AS numeric_p25,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(500)] AS numeric_median,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(750)] AS numeric_p75,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(900)] AS numeric_p90,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(950)] AS numeric_p95,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(975)] AS numeric_p975,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(990)] AS numeric_p99,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(995)] AS numeric_p995,
    APPROX_QUANTILES(numeric_value, 1000)[OFFSET(999)] AS numeric_p999,

    MAX(numeric_value) AS numeric_max

  FROM base
  WHERE variable_type = 'numeric'
    AND numeric_value IS NOT NULL
  GROUP BY source, label
),

categorical_stats AS (
  SELECT
    source,
    label,
    COUNT(DISTINCT value_clean) AS categorical_n_unique_levels
  FROM base
  WHERE variable_type = 'categorical'
    AND value_clean IS NOT NULL
  GROUP BY source, label
)

SELECT
  v.source,
  v.label,
  v.eicu_type,
  v.variable_type,

  v.n_events,
  v.n_icu_stays,
  v.n_non_null_values,
  v.n_distinct_raw_values,

  v.n_numeric_parsed_values,
  v.numeric_parse_fraction,

  n.numeric_min,
  n.numeric_mean,
  n.numeric_sd,
  n.numeric_p001,
  n.numeric_p005,
  n.numeric_p01,
  n.numeric_p025,
  n.numeric_p05,
  n.numeric_p10,
  n.numeric_p25,
  n.numeric_median,
  n.numeric_p75,
  n.numeric_p90,
  n.numeric_p95,
  n.numeric_p975,
  n.numeric_p99,
  n.numeric_p995,
  n.numeric_p999,
  n.numeric_max,

  c.categorical_n_unique_levels

FROM variable_base_stats v
LEFT JOIN numeric_stats n
  ON v.source = n.source
 AND v.label = n.label
LEFT JOIN categorical_stats c
  ON v.source = c.source
 AND v.label = c.label

ORDER BY
  v.variable_type,
  v.source,
  v.label;