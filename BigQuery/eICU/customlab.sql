CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.customlab_state_core` AS

WITH prepared AS (
  SELECT
    patientunitstayid,
    labotheroffset AS timestamp,

    TRIM(CAST(labothername AS STRING)) AS raw_label,
    LOWER(TRIM(CAST(labothername AS STRING))) AS label_lower,

    NULLIF(TRIM(CAST(labothervaluetext AS STRING)), '') AS raw_value

  FROM `physionet-data.eicu_crd.customlab`
),

value_cleaned AS (
  SELECT
    patientunitstayid,
    timestamp,
    raw_label,
    label_lower,

    TRIM(
      REGEXP_REPLACE(
        raw_value,
        r'\s*(mg/dl|gm/dl|g/dl|mmol/l|meq/l|ng/ml|pg/ml|mcg/ml|units/l|u/l|iu/l|mm hg|mmhg|sec|seconds|%)\.?\s*$',
        ''
      )
    ) AS value,

    raw_value

  FROM prepared
  WHERE patientunitstayid IS NOT NULL
    AND timestamp IS NOT NULL
    AND raw_label IS NOT NULL
    AND raw_label != ''
    AND raw_value IS NOT NULL
    AND raw_value != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(raw_value),
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note)$'
    )
),

state_candidate AS (
  SELECT *
  FROM value_cleaned
  WHERE
    (
      SAFE_CAST(REPLACE(value, ',', '') AS FLOAT64) IS NOT NULL
      OR REGEXP_CONTAINS(value, r'^[<>]=?\s*-?[0-9,]+(\.[0-9]+)?$')
      OR REGEXP_CONTAINS(
        LOWER(value),
        r'^(negative|positive|neg|pos|trace|small|moderate|large|none|few|many|rare|occasional|detected|not detected|present|absent|normal|abnormal|clear|cloudy|yellow|colorless)$'
      )
      OR REGEXP_CONTAINS(
        LOWER(value),
        r'^[0-9]+-[0-9]+(\s*[hl])?$'
      )
      OR REGEXP_CONTAINS(
        LOWER(value),
        r'^[0-4]\+[hl]?$'
      )
    )

    AND NOT REGEXP_CONTAINS(
      label_lower,
      r'(^|[^a-z])(ve|v\.e\.|mv|mve|vt|rr|psv|pip|peep|spont|vent|mode|device|sample type|samply typle|type|note|notes)([^a-z]|$)'
    )

    AND NOT REGEXP_CONTAINS(
      label_lower,
      r'influenza|mrsa|clostridium|mycoplasma|hiv|legionella|occult blood|clo test'
    )

    AND NOT REGEXP_CONTAINS(
      label_lower,
      r'amphetamine|methamphetamine|benzodiazep|benzo|barbit|methadone|opiate|oxycodone|pcp|thc|tricyclic|acetaminophen|salicylate|ethyl alcohol|alcohol'
    )
),

first_label AS (
  SELECT
    patientunitstayid,
    timestamp,
    raw_label,
    label_lower,
    value,

    CASE
      WHEN REGEXP_CONTAINS(label_lower, r'^(gfr|egfr|gfr calculation|crt.*gfr|creatinine.*gfr|creatinene.*gfr|creat.*gfr)')
        THEN 'GFR'

      WHEN REGEXP_CONTAINS(label_lower, r'^(globulin|globulin)$')
        THEN 'Globulin'

      WHEN REGEXP_CONTAINS(label_lower, r'^(abs neut|abs neuts|absol neut|absol neuts|absolute neut|absolute neutrophil|absolute neutrophils|abso neuts)')
        THEN 'Absolute neutrophils'

      WHEN REGEXP_CONTAINS(label_lower, r'^(neut|neuts|neutrophil|neutrophils|neut %|neuts %)')
        THEN 'Neutrophils | %'

      WHEN REGEXP_CONTAINS(label_lower, r'(bnp|natriuretic peptide|nat peptide|brain nat|b-natriure)')
        THEN 'BNP'

      WHEN REGEXP_CONTAINS(label_lower, r'(hgb a1c|a1c|glycated hemoglobin)')
        THEN 'Hemoglobin A1c | %'

      WHEN REGEXP_CONTAINS(label_lower, r'(ionized|ioniz|ion calcium|ionized ca|ionized calcium|whole bld ioniz calcium|whole blood ionized|wh bl ion calcium)')
           AND REGEXP_CONTAINS(label_lower, r'(ph)')
        THEN 'Ionized calcium pH'

      WHEN REGEXP_CONTAINS(label_lower, r'(ionized ca|ionized calcium|whole bld ioniz calcium|whole blood ionized|wh bl ion calcium|whole blood ionized ca|whole bld ionized)')
        THEN 'Ionized calcium'

      WHEN REGEXP_CONTAINS(label_lower, r'(lactate|lactic acid)')
        THEN 'Lactate'

      WHEN REGEXP_CONTAINS(label_lower, r'(serum osmo)')
        THEN 'Serum osmolality'

      WHEN REGEXP_CONTAINS(label_lower, r'(d-?dimer|d-dimer|ddimer)')
        THEN 'D-Dimer'

      WHEN REGEXP_CONTAINS(label_lower, r'(^tsh$)')
        THEN 'TSH'

      WHEN REGEXP_CONTAINS(label_lower, r'(vitamin b ?-?12|b12)')
        THEN 'Vitamin B12'

      WHEN REGEXP_CONTAINS(label_lower, r'(folate)')
        THEN 'Folate'

      WHEN REGEXP_CONTAINS(label_lower, r'(^thb$)')
        THEN 'THB'

      WHEN REGEXP_CONTAINS(label_lower, r'(bun/creat|bun/creat ratio)')
        THEN 'BUN/Creatinine ratio'

      WHEN REGEXP_CONTAINS(label_lower, r'(a/g ratio|alb/glob ratio|albumin/globulin)')
        THEN 'Albumin/Globulin ratio'

      WHEN REGEXP_CONTAINS(label_lower, r'(heparin anti|anti-xa|hep anti|heparin level)')
        THEN 'Heparin anti-Xa'

      WHEN REGEXP_CONTAINS(label_lower, r'(^myoglobin$)')
        THEN 'Myoglobin'

      WHEN REGEXP_CONTAINS(label_lower, r'(^esr$)')
        THEN 'ESR'

      WHEN REGEXP_CONTAINS(label_lower, r'(alk phos|alkaline phosphatase)')
        THEN 'Alkaline phosphatase'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine glucose|glucose, urine|^urine glucose$)')
        THEN 'Urine glucose'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine protein|protein, urine|urine protien)')
        THEN 'Urine protein'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine rbc|rbc urine|^rbc$)')
        THEN 'Urine RBC'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine wbc|wbc urine|^wbc$)')
        THEN 'Urine WBC'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine ph|ph urine)')
        THEN 'Urine pH'

      WHEN REGEXP_CONTAINS(label_lower, r'(specific gravity|ur specific gravity|ur spicific gravity)')
        THEN 'Urine specific gravity'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine nitrate|nitrite|nitrate)')
        THEN 'Urine nitrite'

      WHEN REGEXP_CONTAINS(label_lower, r'(leukocyte esterase|leuk est|leukocye esterase)')
        THEN 'Urine leukocyte esterase'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine ketones)')
        THEN 'Urine ketones'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine bilirubin|bilirubin, urine|urine bile)')
        THEN 'Urine bilirubin'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine blood|urine occult blood)')
        THEN 'Urine blood'

      WHEN REGEXP_CONTAINS(label_lower, r'(urobilinogen)')
        THEN 'Urine urobilinogen'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine bacteria|bacteria urine|^bacteria$)')
        THEN 'Urine bacteria'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine color|ua, color)')
        THEN 'Urine color'

      WHEN REGEXP_CONTAINS(label_lower, r'(urine clarity|clarity|urine appearance)')
        THEN 'Urine clarity'

      ELSE raw_label
    END AS preliminary_label

  FROM state_candidate
),

range_corrected AS (
  SELECT
    patientunitstayid,
    timestamp,

    CASE
      WHEN preliminary_label = 'Ionized calcium'
           AND SAFE_CAST(REPLACE(value, ',', '') AS FLOAT64) BETWEEN 6.8 AND 7.8
        THEN 'Ionized calcium pH'

      WHEN preliminary_label = 'Ionized calcium pH'
           AND SAFE_CAST(REPLACE(value, ',', '') AS FLOAT64) < 6.8
        THEN 'Ionized calcium'

      ELSE preliminary_label
    END AS label,

    value

  FROM first_label
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM range_corrected
WHERE label IS NOT NULL
  AND TRIM(label) != ''
  AND value IS NOT NULL
  AND TRIM(value) != '';