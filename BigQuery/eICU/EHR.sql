CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR`
CLUSTER BY patientunitstayid, source, label AS

WITH convertible_map AS (
  SELECT DISTINCT
    TRIM(CAST(eicu_source AS STRING)) AS source,
    TRIM(CAST(eicu_label AS STRING)) AS label
  FROM `mimic-iv-484300.eICU_processed.eicu_mimic_unit_mapping_rules`
  WHERE LOWER(TRIM(CAST(conversion_status AS STRING))) = 'convertible'
),

retained_icu_stays AS (
  SELECT DISTINCT
    CAST(patienthealthsystemstayid AS STRING) AS admission_id,
    CAST(uniquepid AS STRING) AS uniquepid,
    patientunitstayid,

    hospitalid,
    wardid,

    CAST(hospitaladmitoffset AS INT64) AS hospitaladmitoffset,
    CAST(unitdischargeoffset AS INT64) AS unitdischargeoffset,

    SAFE_CAST(
      CASE
        WHEN LOWER(TRIM(CAST(age AS STRING))) = '> 89' THEN '90'
        ELSE REGEXP_EXTRACT(CAST(age AS STRING), r'[0-9]+')
      END AS INT64
    ) AS age,

    TRIM(CAST(gender AS STRING)) AS gender,

    ROW_NUMBER() OVER (
      PARTITION BY patienthealthsystemstayid
      ORDER BY
        hospitaladmitoffset,
        patientunitstayid
    ) AS icu_order_within_admission

  FROM `physionet-data.eicu_crd.patient`

  WHERE patientunitstayid IS NOT NULL
    AND patienthealthsystemstayid IS NOT NULL
    AND uniquepid IS NOT NULL

    -- adult cohort
    AND SAFE_CAST(
      CASE
        WHEN LOWER(TRIM(CAST(age AS STRING))) = '> 89' THEN '90'
        ELSE REGEXP_EXTRACT(CAST(age AS STRING), r'[0-9]+')
      END AS INT64
    ) >= 18

    AND TRIM(CAST(gender AS STRING)) IN ('Male', 'Female')

    -- required for strict ICU-window filter
    AND unitdischargeoffset IS NOT NULL
    AND unitdischargeoffset > 0
),

unified_events AS (
  SELECT
    'customlab_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.customlab_state_core`

  UNION ALL

  SELECT
    'intakeoutput_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.intakeoutput_state_core`

  UNION ALL

  SELECT
    'lab_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.lab_state_core`

  UNION ALL

  SELECT
    'nurseassessment_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.nurseassessment_state_core`

  UNION ALL

  SELECT
    'nursecharting_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.nursecharting_state_core`

  UNION ALL

  SELECT
    'physicalexam_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.physicalexam_state_core`

  UNION ALL

  SELECT
    'respiratorycharting_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.respiratorycharting_state_core`

  UNION ALL

  SELECT
    'vitalaperiodic_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.vitalaperiodic_state_core`

  UNION ALL

  SELECT
    'vitalperiodic_state_core' AS source,
    patientunitstayid,
    timestamp,
    CAST(label AS STRING) AS label,
    CAST(value AS STRING) AS value
  FROM `mimic-iv-484300.eICU_processed.vitalperiodic_state_core`
),

filtered AS (
  SELECT
    r.admission_id,
    r.uniquepid,
    e.patientunitstayid,
    r.icu_order_within_admission,

    CAST(e.timestamp AS INT64) AS timestamp,

    e.source,
    TRIM(e.label) AS label,
    NULLIF(TRIM(e.value), '') AS value

  FROM unified_events e

  JOIN convertible_map m
    ON e.source = m.source
   AND TRIM(e.label) = m.label

  JOIN retained_icu_stays r
    ON e.patientunitstayid = r.patientunitstayid

  WHERE e.patientunitstayid IS NOT NULL
    AND e.timestamp IS NOT NULL
    AND e.label IS NOT NULL
    AND TRIM(e.label) != ''

    -- ICU-admission-level trajectory only.
    AND e.timestamp >= 0
    AND e.timestamp <= r.unitdischargeoffset
),

ranked AS (
  SELECT
    admission_id,
    uniquepid,
    patientunitstayid,
    icu_order_within_admission,

    timestamp,
    source,
    label,
    value,

    ROW_NUMBER() OVER (
      PARTITION BY patientunitstayid
      ORDER BY
        timestamp,
        source,
        label,
        value
    ) AS event_order_in_icu_stay,

    ROW_NUMBER() OVER (
      PARTITION BY patientunitstayid, timestamp
      ORDER BY
        source,
        label,
        value
    ) AS event_order_within_timestamp

  FROM filtered
)

SELECT
  *
FROM ranked;