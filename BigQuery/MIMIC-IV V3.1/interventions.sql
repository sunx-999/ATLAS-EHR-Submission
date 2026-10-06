-- ATLAS-EHR: Clinical Intervention Extraction
-- Objective: Capture medical actions, infusions, and procedures.

WITH raw_inputevents AS (
    -- 1. Continuous Infusions (icu.inputevents)
    SELECT
        i.subject_id, i.hadm_id, CAST(i.itemid AS STRING) AS concept_id,
        di.label AS concept_label,
        CAST(i.amount AS STRING) AS value,
        i.amountuom AS unit,
        i.starttime, i.endtime,
        'icu.inputevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_icu.inputevents` i
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.d_items` di ON i.itemid = di.itemid
    WHERE i.amount IS NOT NULL
),

raw_icu_proc AS (
    -- 2. ICU Procedures (icu.procedureevents)
    SELECT
        p.subject_id, p.hadm_id, CAST(p.itemid AS STRING) AS concept_id,
        di.label AS concept_label,
        CAST(p.value AS STRING) AS value,
        p.valueuom AS unit,
        p.starttime, p.endtime,
        'icu.procedureevents' AS source_table
    FROM `physionet-data.mimiciv_3_1_icu.procedureevents` p
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.d_items` di ON p.itemid = di.itemid
    WHERE p.value IS NOT NULL
),

raw_emar AS (
    -- 3. Medication Administration (hosp.emar)
    SELECT
        e.subject_id, e.hadm_id, CAST(NULL AS STRING) AS concept_id,
        e.medication AS concept_label,
        '1.0' AS value, -- Standard indicator for eMAR events
        'dose' AS unit,
        e.charttime AS starttime,
        DATETIME_ADD(e.charttime, INTERVAL 1 MINUTE) AS endtime,
        'hosp.emar' AS source_table
    FROM `physionet-data.mimiciv_3_1_hosp.emar` e
    WHERE e.medication IS NOT NULL
),

raw_hosp_proc AS (
    -- 4. Hospital Procedures (hosp.procedures_icd)
    SELECT
        pi.subject_id, pi.hadm_id, pi.icd_code AS concept_id,
        dp.long_title AS concept_label,
        '1.0' AS value,
        'event' AS unit,
        DATETIME(pi.chartdate, TIME '23:59:59') AS starttime,
        DATETIME(pi.chartdate, TIME '23:59:59') AS endtime,
        'hosp.procedures_icd' AS source_table
    FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` pi
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_procedures` dp 
      ON pi.icd_code = dp.icd_code AND pi.icd_version = dp.icd_version
    WHERE pi.chartdate IS NOT NULL
)

SELECT * FROM raw_inputevents
UNION ALL
SELECT * FROM raw_icu_proc
UNION ALL
SELECT * FROM raw_emar
UNION ALL
SELECT * FROM raw_hosp_proc;