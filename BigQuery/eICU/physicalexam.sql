CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.physicalexam_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    physicalexamoffset,
    physicalexampath,
    physicalexamvalue,
    physicalexamtext,
    COALESCE(
      NULLIF(TRIM(physicalexamtext), ''),
      NULLIF(TRIM(physicalexamvalue), '')
    ) AS resolved_value
  FROM `physionet-data.eicu_crd.physicalexam`
),

cleaned AS (
  SELECT *
  FROM raw
  WHERE patientunitstayid IS NOT NULL
    AND physicalexamoffset IS NOT NULL
    AND resolved_value IS NOT NULL
    AND TRIM(resolved_value) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(resolved_value)),
      r'^(not assessed|unable to assess|unable|not performed|refused|deferred|n/a|na|unknown|comment|see note|null|missing|performed - structured|scored)$'
    )
),

state_core AS (
  SELECT *
  FROM cleaned
  WHERE
    physicalexampath IN (
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Admission',
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Current',

      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/BP (systolic)/BP (systolic) Current',
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/BP (diastolic)/BP (diastolic) Current',
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/HR/HR Current',
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/O2 Sat%/O2 Sat% Current',
      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/Resp Rate/Resp Current',

      'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Hemodynamic Data/CVP/CVP'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/[0-9]+$'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Eyes Score/[1-4]$'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Motor Score/[1-6]$'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Verbal Score/[1-5]$'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/(Level of Consciousness|Orientation|Affect)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/(Motor|Sensation|Reflexes|Cranial Nerves)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Eyes/(Pupils|Conjunctivae)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Neck/JVD/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Cardiovascular/(Auscultation|Pulses)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/HR Rhythm/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Extremities/(Perfusion|Edema)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Pulmonary/Auscultation/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Pulmonary/Percussion/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/Resp Description/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Gastrointestinal/(Bowel Sounds|Palpation)/'
    )

    OR REGEXP_CONTAINS(
      physicalexampath,
      r'^notes/Progress Notes/Physical Exam/Physical Exam/Skin/(Rash|Color|Temperature|Turgor|Lesions|Bruising|Cyanosis|Jaundice)/'
    )
),

context_removed AS (
  SELECT *
  FROM state_core
  WHERE NOT REGEXP_CONTAINS(
    LOWER(physicalexampath),
    r'physical exam obtain options|performed - structured|not performed|fi[o0]2|peep|vent rate|resp mode|airway|intubated|tracheostomy|tube|drain|catheter|foley|hardware|gastrostomy|ng tube|p?rbc|intake total|output total|dialysis net|total net|delta|appearance|nutritional status|health status|acuity|well developed|obese|cachectic|ill-appearing|healthy appearing|acute distress|scored|unable to score|estimated due to meds|unable to assess'
  )
),

timestamp_resolved AS (
  SELECT
    patientunitstayid,
    physicalexamoffset AS timestamp,

    CASE
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Admission'
        THEN 'Weight (kg) | Admission'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Current'
        THEN 'Weight (kg) | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/BP (systolic)/BP (systolic) Current'
        THEN 'BP systolic | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/BP (diastolic)/BP (diastolic) Current'
        THEN 'BP diastolic | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/HR/HR Current'
        THEN 'HR | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/O2 Sat%/O2 Sat% Current'
        THEN 'O2 Sat% | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/Resp Rate/Resp Current'
        THEN 'Resp Rate | Current'
      WHEN physicalexampath = 'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Hemodynamic Data/CVP/CVP'
        THEN 'CVP | CVP'
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/[0-9]+$')
        THEN 'GCS | Total'
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Eyes Score/[1-4]$')
        THEN 'GCS | Eyes Score'
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Motor Score/[1-6]$')
        THEN 'GCS | Motor Score'
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Verbal Score/[1-5]$')
        THEN 'GCS | Verbal Score'
      ELSE REGEXP_REPLACE(
        physicalexampath,
        r'^notes/Progress Notes/Physical Exam/Physical Exam/',
        ''
      )
    END AS label,

    CASE
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/[0-9]+$')
        THEN REGEXP_EXTRACT(physicalexampath, r'/Neurologic/GCS/([0-9]+)$')
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Eyes Score/[1-4]$')
        THEN REGEXP_EXTRACT(physicalexampath, r'/Neurologic/GCS/Eyes Score/([1-4])$')
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Motor Score/[1-6]$')
        THEN REGEXP_EXTRACT(physicalexampath, r'/Neurologic/GCS/Motor Score/([1-6])$')
      WHEN REGEXP_CONTAINS(physicalexampath, r'/Neurologic/GCS/Verbal Score/[1-5]$')
        THEN REGEXP_EXTRACT(physicalexampath, r'/Neurologic/GCS/Verbal Score/([1-5])$')
      ELSE TRIM(resolved_value)
    END AS value

  FROM context_removed
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
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note|not performed|performed - structured|scored|unable to score due to meds|unable to score due to medication|estimated due to meds|unable to assess orientation)$'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM final_filtered;