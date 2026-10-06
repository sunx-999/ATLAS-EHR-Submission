CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.nursecharting_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    nursingchartoffset,
    nursingchartentryoffset,
    nursingchartcelltypecat,
    nursingchartcelltypevallabel,
    nursingchartcelltypevalname,
    nursingchartvalue
  FROM `physionet-data.eicu_crd.nursecharting`
),

cleaned AS (
  SELECT *
  FROM raw
  WHERE patientunitstayid IS NOT NULL
    AND COALESCE(nursingchartoffset, nursingchartentryoffset) IS NOT NULL
    AND nursingchartvalue IS NOT NULL
    AND TRIM(nursingchartvalue) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(nursingchartvalue)),
      r'^(not assessed|unable to assess|unable|not performed|refused|deferred|n/a|na|unknown|comment|see note|null|missing|documentation undone|unable to score due to medication)$'
    )
    AND NOT (
      nursingchartcelltypevallabel = 'Best Verbal Response'
      AND nursingchartcelltypevalname = 'Value'
      AND LOWER(TRIM(nursingchartvalue)) = 'trached or intubated'
    )
),

state_core AS (
  SELECT *
  FROM cleaned
  WHERE
    (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'Heart Rate'
      AND nursingchartcelltypevalname = 'Heart Rate'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'Pulse'
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'Respiratory Rate'
      AND nursingchartcelltypevalname = 'Respiratory Rate'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'O2 Saturation'
      AND nursingchartcelltypevalname = 'O2 Saturation'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'SpO2'
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'Temperature'
      AND nursingchartcelltypevalname IN ('Temperature (C)', 'Temperature (F)')
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'Non-Invasive BP'
      AND nursingchartcelltypevalname IN (
        'Non-Invasive BP Systolic',
        'Non-Invasive BP Diastolic',
        'Non-Invasive BP Mean'
      )
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'Invasive BP'
      AND nursingchartcelltypevalname IN (
        'Invasive BP Systolic',
        'Invasive BP Diastolic',
        'Invasive BP Mean'
      )
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel IN (
        'MAP (mmHg)',
        'Arterial Line MAP (mmHg)'
      )
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'Bedside Glucose'
      AND nursingchartcelltypevalname = 'Bedside Glucose'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'End Tidal CO2'
      AND nursingchartcelltypevalname = 'End Tidal CO2'
    )

    OR (
      nursingchartcelltypecat IN ('Vital Signs', 'SVO2')
      AND nursingchartcelltypevallabel = 'SVO2'
      AND nursingchartcelltypevalname = 'SVO2'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'CVP'
      AND nursingchartcelltypevalname = 'CVP'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'CVP (mmHg)'
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'ICP'
      AND nursingchartcelltypevalname = 'ICP'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'CPP'
      AND nursingchartcelltypevalname = 'CPP'
    )

    OR (
      nursingchartcelltypecat = 'Vital Signs'
      AND nursingchartcelltypevallabel = 'IAP'
      AND nursingchartcelltypevalname = 'IAP'
    )

    OR (
      nursingchartcelltypecat IN ('Vital Signs', 'Invasive')
      AND nursingchartcelltypevallabel = 'PA'
      AND nursingchartcelltypevalname IN (
        'PA Systolic',
        'PA Diastolic',
        'PA Mean'
      )
    )

    OR (
      nursingchartcelltypecat = 'Scores'
      AND nursingchartcelltypevallabel = 'Glasgow coma score'
      AND nursingchartcelltypevalname IN (
        'Eyes',
        'Motor',
        'Verbal',
        'GCS Total'
      )
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel IN (
        'Best Eye Response',
        'Best Motor Response',
        'Best Verbal Response',
        'Score (Glasgow Coma Scale)',
        'Eye Opening',
        'Motor Response',
        'Verbal Response'
      )
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'RASS'
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Scores'
      AND nursingchartcelltypevallabel = 'Pain Score/Goal'
      AND nursingchartcelltypevalname = 'Pain Score'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'Pain Present'
      AND nursingchartcelltypevalname = 'Value'
    )

    OR (
      nursingchartcelltypecat = 'Other Vital Signs and Infusions'
      AND nursingchartcelltypevallabel = 'Symptoms of Delirium Present'
      AND nursingchartcelltypevalname = 'Value'
    )
),

timestamp_resolved AS (
  SELECT
    patientunitstayid,
    CASE
      WHEN nursingchartoffset IS NOT NULL THEN nursingchartoffset
      WHEN nursingchartoffset IS NULL AND nursingchartentryoffset IS NOT NULL THEN nursingchartentryoffset
      ELSE NULL
    END AS timestamp,
    CONCAT(
      TRIM(nursingchartcelltypevallabel),
      ' | ',
      TRIM(nursingchartcelltypevalname)
    ) AS label,
    TRIM(nursingchartvalue) AS value
  FROM state_core
),

final_filtered AS (
  SELECT *
  FROM timestamp_resolved
  WHERE patientunitstayid IS NOT NULL
    AND timestamp IS NOT NULL
    AND label IS NOT NULL
    AND TRIM(label) != ''
    AND value IS NOT NULL
    AND TRIM(value) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(value)),
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note)$'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM final_filtered;