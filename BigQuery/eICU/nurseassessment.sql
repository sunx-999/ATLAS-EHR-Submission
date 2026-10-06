CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.nurseassessment_state_core` AS

WITH state_core AS (
  SELECT *
  FROM `physionet-data.eicu_crd.nurseassessment`
  WHERE
    (
      REGEXP_CONTAINS(cellattributepath, r'Nursing Assessment\|Neurologic\|')
      AND NOT REGEXP_CONTAINS(cellattributepath, r'Train of Four\|')
    )
    OR REGEXP_CONTAINS(
      cellattributepath,
      r'Nursing Assessment\|Respiratory\|'
    )
    OR REGEXP_CONTAINS(
      cellattributepath,
      r'Nursing Assessment\|Cardiovascular\|(Skin Color/Temp|Capillary Refill|Auscultation|Edema|Pulses)\|'
    )
),

filtered_values AS (
  SELECT *
  FROM state_core
  WHERE cellattributevalue IS NOT NULL
    AND TRIM(cellattributevalue) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(cellattributevalue)),
      r'^(not assessed|unable to assess|unable|not performed|refused|deferred|n/a|na|unknown|comment|see note)$'
    )
    AND NOT (
      cellattributepath = 'flowsheet|Flowsheet Cell Labels|Nursing Assessment|Cardiovascular|Capillary Refill|Capillary Refill'
      AND LOWER(TRIM(cellattributevalue)) IN ('hands', 'feet')
    )
    AND NOT (
      cellattributepath = 'flowsheet|Flowsheet Cell Labels|Nursing Assessment|Neurologic|Neurologic|Speech'
      AND LOWER(TRIM(cellattributevalue)) = 'et/trach'
    )
),

timestamp_resolved AS (
  SELECT
    patientunitstayid,

    CASE
      WHEN nurseassessoffset IS NOT NULL THEN nurseassessoffset
      WHEN nurseassessoffset IS NULL AND nurseassessentryoffset IS NOT NULL THEN nurseassessentryoffset
      ELSE NULL
    END AS timestamp,

    CONCAT(TRIM(celllabel), ' | ', TRIM(cellattribute)) AS label,
    TRIM(cellattributevalue) AS value

  FROM filtered_values
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
      r'^(n/a|na|null|none missing|missing|unknown)$'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM final_filtered;