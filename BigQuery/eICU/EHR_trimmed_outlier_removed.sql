CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR_trimmed_outlier_removed`
CLUSTER BY patientunitstayid, source, label AS

WITH rules AS (
  SELECT
    TRIM(CAST(source AS STRING)) AS source,
    TRIM(CAST(label AS STRING)) AS label,
    LOWER(TRIM(CAST(variable_type AS STRING))) AS variable_type,

    SAFE_CAST(clip_lower AS FLOAT64) AS clip_lower,
    SAFE_CAST(clip_upper AS FLOAT64) AS clip_upper

  FROM `mimic-iv-484300.eICU_processed.eicu_outlier_trim_rules`
),

base AS (
  SELECT
    e.*,
    r.variable_type,
    r.clip_lower,
    r.clip_upper,

    SAFE_CAST(
      REGEXP_EXTRACT(
        NULLIF(TRIM(CAST(e.value AS STRING)), ''),
        r'^[<>~= ]*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)'
      ) AS FLOAT64
    ) AS numeric_value

  FROM `mimic-iv-484300.eICU_processed.EHR_trimmed` e
  JOIN rules r
    ON e.source = r.source
   AND TRIM(CAST(e.label AS STRING)) = r.label
)

SELECT
  admission_id,
  uniquepid,
  patientunitstayid,
  icu_order_within_admission,
  timestamp,
  source,
  label,
  value,
  event_order_in_icu_stay,
  event_order_within_timestamp

FROM base

WHERE
  -- categorical: keep all records
  variable_type = 'categorical'

  OR

  -- numeric: value must parse, then must satisfy all non-null bounds
  (
    variable_type = 'numeric'
    AND numeric_value IS NOT NULL
    AND (clip_lower IS NULL OR numeric_value >= clip_lower)
    AND (clip_upper IS NULL OR numeric_value <= clip_upper)
  );