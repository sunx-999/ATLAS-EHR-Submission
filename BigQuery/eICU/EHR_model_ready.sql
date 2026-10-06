CREATE TEMP FUNCTION resolve_categorical_vid(
  mapping_text STRING,
  raw_value STRING,
  default_vid INT64
)
RETURNS INT64
LANGUAGE js AS """
  if (raw_value === null || raw_value === undefined) {
    return null;
  }

  const raw = String(raw_value).trim().toLowerCase();

  if (raw.length === 0) {
    return null;
  }

  if (mapping_text === null || mapping_text === undefined || String(mapping_text).trim() === '') {
    return default_vid;
  }

  const txt = String(mapping_text).trim().toLowerCase();

  // Split prose mappings like:
  // "present -> vid 1698 (...); decreased -> vid 1697 (...); absent -> vid 1696 (...)"
  const parts = txt.split(';');

  for (let part of parts) {
    part = part.trim();

    if (part.length === 0) {
      continue;
    }

    // Require the clause to mention the raw level.
    // Example: raw = "decreased"; clause = "decreased -> vid 1697 (...)"
    if (!part.includes(raw)) {
      continue;
    }

    // Example: "absent -> do not emit vid 5510"
    if (part.includes('do not emit')) {
      return null;
    }

    const m = part.match(/vid\\s+(\\d+)/);
    if (m) {
      return parseInt(m[1], 10);
    }
  }

  // One-hot / presence-style mapping:
  // "presence of this eICU value -> vid 585 (...)"
  if (txt.includes('presence of this eicu value')) {
    const m = txt.match(/vid\\s+(\\d+)/);
    if (m) {
      return parseInt(m[1], 10);
    }
  }

  // Single-clause mappings:
  // "absent -> map to vid 583 (...)"
  if (txt.includes(raw)) {
    if (txt.includes('do not emit')) {
      return null;
    }

    const m = txt.match(/vid\\s+(\\d+)/);
    if (m) {
      return parseInt(m[1], 10);
    }
  }

  // Fallback for ordinary binary/presence-style categorical variables.
  return default_vid;
""";

CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR_model_ready`
CLUSTER BY patientunitstayid, timestamp, event_order_within_timestamp AS

WITH schema AS (
  SELECT
    CAST(vid AS INT64) AS mimic_vid,
    TRIM(CAST(normalized_concept_label AS STRING)) AS normalized_concept_label,
    LOWER(TRIM(CAST(model_type AS STRING))) AS model_type,
    SAFE_CAST(mean AS FLOAT64) AS mean,
    SAFE_CAST(std AS FLOAT64) AS std,
    SAFE_CAST(p5 AS FLOAT64) AS p5,
    SAFE_CAST(p95 AS FLOAT64) AS p95,
    SAFE_CAST(use_clipping AS BOOL) AS use_clipping
  FROM `mimic-iv-484300.eICU_processed.mimic_schema`
),

mapping AS (
  SELECT DISTINCT
    TRIM(CAST(eicu_source AS STRING)) AS source,
    TRIM(CAST(eicu_label AS STRING)) AS label,

    LOWER(TRIM(CAST(eicu_type AS STRING))) AS eicu_type,
    LOWER(TRIM(CAST(mimic_type AS STRING))) AS mimic_type,

    SAFE_CAST(mimic_vid AS INT64) AS default_mimic_vid,

    COALESCE(SAFE_CAST(multiplier AS FLOAT64), 1.0) AS multiplier,
    COALESCE(SAFE_CAST(offset AS FLOAT64), 0.0) AS offset,

    NULLIF(TRIM(CAST(categorical_level_mapping AS STRING)), '') AS categorical_level_mapping

  FROM `mimic-iv-484300.eICU_processed.eicu_mimic_unit_mapping_rules`
  WHERE LOWER(TRIM(CAST(conversion_status AS STRING))) = 'convertible'
),

labels_clean AS (
  SELECT
    CAST(patientunitstayid AS INT64) AS patientunitstayid,
    TRIM(CAST(gender AS STRING)) AS gender,
    SAFE_CAST(age AS FLOAT64) AS age
  FROM `mimic-iv-484300.eICU_processed.labels`
),

ehr_base AS (
  SELECT
    e.admission_id,
    e.uniquepid,
    CAST(e.patientunitstayid AS INT64) AS patientunitstayid,
    e.icu_order_within_admission,
    CAST(e.timestamp AS INT64) AS timestamp,

    e.source,
    TRIM(CAST(e.label AS STRING)) AS label,
    CAST(e.value AS STRING) AS value,

    e.event_order_in_icu_stay,
    CAST(e.event_order_within_timestamp AS INT64) AS original_event_order_within_timestamp,

    m.eicu_type,
    m.mimic_type,
    m.default_mimic_vid,
    m.multiplier,
    m.offset,
    m.categorical_level_mapping,

    CASE
      WHEN REGEXP_CONTAINS(COALESCE(m.mimic_type, m.eicu_type), r'categorical') THEN
        resolve_categorical_vid(
          m.categorical_level_mapping,
          TRIM(CAST(e.value AS STRING)),
          m.default_mimic_vid
        )
      ELSE
        m.default_mimic_vid
    END AS resolved_mimic_vid,

    SAFE_CAST(
      REGEXP_EXTRACT(
        NULLIF(TRIM(CAST(e.value AS STRING)), ''),
        r'^[<>~= ]*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)'
      ) AS FLOAT64
    ) AS numeric_raw_value

  FROM `mimic-iv-484300.eICU_processed.EHR_trimmed_outlier_removed` e

  JOIN mapping m
    ON e.source = m.source
   AND TRIM(CAST(e.label AS STRING)) = m.label
),

ehr_with_schema AS (
  SELECT
    b.*,

    s.normalized_concept_label,
    s.model_type AS schema_model_type,
    s.mean,
    s.std,
    s.p5,
    s.p95,
    s.use_clipping

  FROM ehr_base b
  LEFT JOIN schema s
    ON b.resolved_mimic_vid = s.mimic_vid
),

ehr_converted AS (
  SELECT
    admission_id,
    uniquepid,
    patientunitstayid,
    icu_order_within_admission,
    timestamp,

    source,
    label,
    value,

    eicu_type,
    mimic_type,

    resolved_mimic_vid AS mimic_vid,
    normalized_concept_label,

    CASE
      WHEN REGEXP_CONTAINS(COALESCE(mimic_type, eicu_type), r'numeric') THEN
        CASE
          WHEN numeric_raw_value IS NULL THEN NULL
          WHEN mean IS NULL OR std IS NULL OR std = 0 THEN NULL
          ELSE
            (
              CASE
                WHEN use_clipping IS TRUE
                 AND p5 IS NOT NULL
                 AND p95 IS NOT NULL
                THEN
                  LEAST(
                    GREATEST(numeric_raw_value * multiplier + offset, p5),
                    p95
                  )
                ELSE
                  numeric_raw_value * multiplier + offset
              END
              - mean
            ) / std
        END

      WHEN REGEXP_CONTAINS(COALESCE(mimic_type, eicu_type), r'categorical') THEN
        1.0

      ELSE NULL
    END AS converted_value,

    event_order_in_icu_stay,

    original_event_order_within_timestamp + 2 AS event_order_within_timestamp

  FROM ehr_with_schema
),

unique_timestamps AS (
  SELECT DISTINCT
    e.admission_id,
    e.uniquepid,
    e.patientunitstayid,
    e.icu_order_within_admission,
    e.timestamp,
    l.gender,
    l.age
  FROM ehr_converted e
  JOIN labels_clean l
    ON e.patientunitstayid = l.patientunitstayid
),

gender_schema AS (
  SELECT
    mimic_vid,
    normalized_concept_label,
    LOWER(normalized_concept_label) AS normalized_concept_label_lower
  FROM schema
  WHERE normalized_concept_label IN (
    'gender_other_f',
    'gender_other_m'
  )
),

gender_broadcast AS (
  SELECT
    u.admission_id,
    u.uniquepid,
    u.patientunitstayid,
    u.icu_order_within_admission,
    u.timestamp,

    'demographics' AS source,
    'gender' AS label,
    u.gender AS value,

    'categorical' AS eicu_type,
    'categorical' AS mimic_type,

    s.mimic_vid,
    s.normalized_concept_label,

    1.0 AS converted_value,

    0 AS event_order_in_icu_stay,
    1 AS event_order_within_timestamp

  FROM unique_timestamps u
  JOIN gender_schema s
    ON (
      LOWER(u.gender) = 'female'
      AND s.normalized_concept_label_lower = 'gender_other_f'
    )
    OR (
      LOWER(u.gender) = 'male'
      AND s.normalized_concept_label_lower = 'gender_other_m'
    )
  WHERE u.gender IS NOT NULL
),

age_schema AS (
  SELECT
    mimic_vid,
    normalized_concept_label,
    mean,
    std,
    p5,
    p95,
    use_clipping
  FROM schema
  WHERE normalized_concept_label = 'age_other_years'
),

age_broadcast AS (
  SELECT
    u.admission_id,
    u.uniquepid,
    u.patientunitstayid,
    u.icu_order_within_admission,
    u.timestamp,

    'demographics' AS source,
    'age' AS label,
    CAST(u.age AS STRING) AS value,

    'numeric' AS eicu_type,
    'numeric' AS mimic_type,

    s.mimic_vid,
    s.normalized_concept_label,

    CASE
      WHEN u.age IS NULL THEN NULL
      WHEN s.mean IS NULL OR s.std IS NULL OR s.std = 0 THEN NULL
      ELSE
        (
          CASE
            WHEN s.use_clipping IS TRUE
             AND s.p5 IS NOT NULL
             AND s.p95 IS NOT NULL
            THEN LEAST(GREATEST(u.age, s.p5), s.p95)
            ELSE u.age
          END
          - s.mean
        ) / s.std
    END AS converted_value,

    0 AS event_order_in_icu_stay,
    2 AS event_order_within_timestamp

  FROM unique_timestamps u
  CROSS JOIN age_schema s
  WHERE u.age IS NOT NULL
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

  eicu_type,
  mimic_type,
  mimic_vid,
  normalized_concept_label,

  converted_value,

  event_order_in_icu_stay,
  event_order_within_timestamp

FROM gender_broadcast
WHERE mimic_vid IS NOT NULL
  AND converted_value IS NOT NULL

UNION ALL

SELECT
  admission_id,
  uniquepid,
  patientunitstayid,
  icu_order_within_admission,
  timestamp,

  source,
  label,
  value,

  eicu_type,
  mimic_type,
  mimic_vid,
  normalized_concept_label,

  converted_value,

  event_order_in_icu_stay,
  event_order_within_timestamp

FROM age_broadcast
WHERE mimic_vid IS NOT NULL
  AND converted_value IS NOT NULL

UNION ALL

SELECT
  admission_id,
  uniquepid,
  patientunitstayid,
  icu_order_within_admission,
  timestamp,

  source,
  label,
  value,

  eicu_type,
  mimic_type,
  mimic_vid,
  normalized_concept_label,

  converted_value,

  event_order_in_icu_stay,
  event_order_within_timestamp

FROM ehr_converted
WHERE mimic_vid IS NOT NULL
  AND converted_value IS NOT NULL;
