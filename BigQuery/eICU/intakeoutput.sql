CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.intakeoutput_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    intakeoutputoffset,
    intakeoutputentryoffset,
    cellpath,
    celllabel,
    cellvaluenumeric,
    cellvaluetext,
    COALESCE(
      NULLIF(TRIM(CAST(cellvaluenumeric AS STRING)), ''),
      NULLIF(TRIM(CAST(cellvaluetext AS STRING)), '')
    ) AS resolved_value
  FROM `physionet-data.eicu_crd.intakeoutput`
),

cleaned AS (
  SELECT *
  FROM raw
  WHERE patientunitstayid IS NOT NULL
    AND COALESCE(intakeoutputoffset, intakeoutputentryoffset) IS NOT NULL
    AND resolved_value IS NOT NULL
    AND TRIM(resolved_value) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(resolved_value)),
      r'^(not assessed|unable to assess|unable|not performed|refused|deferred|n/a|na|unknown|comment|see note|null|missing|documentation undone)$'
    )
),

output_scope AS (
  SELECT *
  FROM cleaned
  WHERE
    REGEXP_CONTAINS(
      cellpath,
      r'^flowsheet\|Flowsheet Cell Labels\|I&O\|Output \(ml\)\|'
    )
    OR cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Dialysis (ml)|Out'
),

state_output AS (
  SELECT *
  FROM output_scope
  WHERE
    REGEXP_CONTAINS(
      LOWER(CONCAT(COALESCE(cellpath, ''), ' ', COALESCE(celllabel, ''))),
      r'urine|urinary|foley|void|voided|uop|catheter output|indwelling catheter|nephrostomy|ileal conduit|stool|fecal|bowel|ostomy|ileostomy|colostomy|emesis|vomit|gastric|ng|og|nasogastric|orogastric|chest tube|pleural|mediastinal|drain|jp|jackson|hemovac|blake|bulb|wound vac|t-tube|biliary|evd|external ventricular|ventric|csf|lumbar|blood loss|ebl|dialysis|fluid removal|ultrafiltration|effluent'
    )
    AND NOT REGEXP_CONTAINS(
      LOWER(CONCAT(COALESCE(cellpath, ''), ' ', COALESCE(celllabel, ''))),
      r'intake|input|ivf|ivpb|i\.v\.|p\.o\.|oral intake|tube feeding|enteral|parenteral|nutrition|tpn|formula|feeding tube|flush|water bolus|medication|meds|infusion|administered|volume given|blood product|prbc|ffp|platelet|albumin|crystalloid|propofol|fentanyl|heparin|insulin|norepinephrine|dopamine|amiodarone|dexmedetomidine|midazolam'
    )
    AND NOT REGEXP_CONTAINS(
      LOWER(CONCAT(COALESCE(cellpath, ''), ' ', COALESCE(celllabel, ''))),
      r'total|net|balance|cumulative|variance|shift|24 hr|24hr|24-hour|daily total|running|calculated'
    )
    AND NOT REGEXP_CONTAINS(
      LOWER(CONCAT(COALESCE(cellpath, ''), ' ', COALESCE(celllabel, ''))),
      r'generic|other|misc|unknown|not specified|residual discarded'
    )
),

timestamp_resolved AS (
  SELECT
    patientunitstayid,

    CASE
      WHEN intakeoutputoffset IS NOT NULL THEN intakeoutputoffset
      WHEN intakeoutputoffset IS NULL AND intakeoutputentryoffset IS NOT NULL THEN intakeoutputentryoffset
      ELSE NULL
    END AS timestamp,

    CASE
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'urine|urinary|foley|void|voided|uop|catheter output|indwelling catheter')
        THEN 'Urine output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'nephrostomy')
        THEN 'Nephrostomy output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'ileal conduit')
        THEN 'Ileal conduit output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'stool|fecal|bowel|ostomy|ileostomy|colostomy')
        THEN 'Stool output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'emesis|vomit')
        THEN 'Emesis output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'gastric|ng|og|nasogastric|orogastric')
        THEN 'Gastric output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'chest tube|pleural|mediastinal')
        THEN 'Chest tube output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'evd|external ventricular|ventric|csf|lumbar')
        THEN 'CSF drain output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'wound vac')
        THEN 'Wound vac output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'drain|jp|jackson|hemovac|blake|bulb|t-tube|biliary')
        THEN 'Surgical drain output | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'blood loss|ebl')
        THEN 'Blood loss | Volume'
      WHEN REGEXP_CONTAINS(LOWER(CONCAT(cellpath, ' ', celllabel)), r'dialysis|fluid removal|ultrafiltration|effluent')
        THEN 'Dialysis output | Volume'
      ELSE CONCAT(
        REGEXP_REPLACE(
          cellpath,
          r'^flowsheet\|Flowsheet Cell Labels\|I&O\|',
          ''
        ),
        ' | ',
        TRIM(celllabel)
      )
    END AS label,

    TRIM(resolved_value) AS value

  FROM state_output
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
    AND label NOT IN (
      'Urine output | Count',
      'Stool output | Count'
    )
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(value)),
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note|documentation undone)$'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM final_filtered;