-- ATLAS-EHR: Clinical State Extraction
-- Objective: Capture physiological observations across hospital and ICU sources.

WITH patient_info AS (
    SELECT subject_id, gender, anchor_age, anchor_year
    FROM `physionet-data.mimiciv_3_1_hosp.patients`
),

admission_info AS (
    SELECT hadm_id, subject_id, admittime
    FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),

demographics AS (
    -- 1. Demographics as States: Gender and Age
    SELECT
        p.subject_id, a.hadm_id, 
        CAST(NULL AS STRING) AS concept_id,
        'gender' AS concept_label,
        p.gender AS value,
        CAST(NULL AS STRING) AS unit,
        a.admittime AS charttime,
        'hosp.patients' AS source_table
    FROM patient_info p
    INNER JOIN admission_info a ON p.subject_id = a.subject_id

    UNION ALL

    SELECT
        p.subject_id, a.hadm_id, 
        CAST(NULL AS STRING) AS concept_id,
        'age' AS concept_label,
        CAST((p.anchor_age + DATETIME_DIFF(a.admittime, DATETIME(p.anchor_year, 1, 1, 0, 0, 0), YEAR)) AS STRING) AS value,
        'years' AS unit,
        a.admittime AS charttime,
        'hosp.patients' AS source_table
    FROM patient_info p
    INNER JOIN admission_info a ON p.subject_id = a.subject_id
),

raw_states AS (
    -- 2. Lab Events (hosp.labevents)
    SELECT 
        l.subject_id, l.hadm_id, CAST(l.itemid AS STRING) AS concept_id,
        dl.label AS concept_label, 
        COALESCE(CAST(l.valuenum AS STRING), l.value) AS value, 
        l.valueuom AS unit, 
        l.charttime, 'hosp.labevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` dl ON l.itemid = dl.itemid
    WHERE (l.valuenum IS NOT NULL OR l.value IS NOT NULL)

    UNION ALL

    -- 3. ICU Chart Events (icu.chartevents)
    SELECT 
        c.subject_id, c.hadm_id, CAST(c.itemid AS STRING) AS concept_id,
        di.label AS concept_label, 
        COALESCE(CAST(c.valuenum AS STRING), c.value) AS value, 
        c.valueuom AS unit, 
        c.charttime, 'icu.chartevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.d_items` di ON c.itemid = di.itemid
    WHERE (c.valuenum IS NOT NULL OR c.value IS NOT NULL)

    UNION ALL

    -- 4. Physiological Outputs (icu.outputevents)
    SELECT 
        o.subject_id, o.hadm_id, CAST(o.itemid AS STRING) AS concept_id,
        di.label AS concept_label, 
        CAST(o.value AS STRING) AS value, 
        o.valueuom AS unit, 
        o.charttime, 'icu.outputevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_icu.outputevents` o
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.d_items` di ON o.itemid = di.itemid
    WHERE o.value IS NOT NULL

    UNION ALL

    -- 5. Dynamic Patient Weight (icu.inputevents)
    SELECT DISTINCT
        i.subject_id, i.hadm_id, CAST(NULL AS STRING) AS concept_id,
        'weight' AS concept_label, 
        CAST(i.patientweight AS STRING) AS value, 
        'kg' AS unit, 
        i.starttime AS charttime, 'icu.inputevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_icu.inputevents` i
    WHERE i.patientweight IS NOT NULL
)

SELECT * FROM demographics
UNION ALL
SELECT * FROM raw_states;