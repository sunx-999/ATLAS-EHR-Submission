-- ATLAS-EHR: Admission Labels & Cohort Splits

WITH global_ref_anchor AS (
    -- Step 1: Establish clinical t=0
    SELECT 
        hadm_id, 
        MIN(ref_ts) AS t_zero
    FROM (
        SELECT hadm_id, charttime AS ref_ts FROM `mimic-iv-484300.mimic_iv_3_1_processed.states`
        UNION ALL
        SELECT hadm_id, starttime AS ref_ts FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions`
    )
    GROUP BY hadm_id
),

-- New Step: Aggregate ICD codes into a comma-separated string
aggregated_icd AS (
    SELECT 
        hadm_id,
        STRING_AGG(icd_code, ', ' ORDER BY seq_num) AS icd_codes
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    GROUP BY hadm_id
),

-- New Step: Aggregate DRG codes into a comma-separated string
aggregated_drg AS (
    SELECT 
        hadm_id,
        STRING_AGG(drg_code, ', ' ORDER BY drg_code) AS drg_codes
    FROM `physionet-data.mimiciv_3_1_hosp.drgcodes`
    GROUP BY hadm_id
),

encounter_lifecycle AS (
    -- Step 2: Basic cohort info and age refinement
    SELECT 
        adm.subject_id, adm.hadm_id, ref.t_zero,
        COALESCE(LEAST(adm.admittime, adm.edregtime), adm.admittime) AS clinical_start,
        COALESCE(LEAST(adm.dischtime, adm.deathtime), adm.dischtime) AS clinical_end,
        adm.deathtime, adm.dischtime, adm.hospital_expire_flag, adm.admission_type,
        (p.anchor_age + DATETIME_DIFF(adm.admittime, DATETIME(p.anchor_year, 1, 1, 0, 0, 0), YEAR)) AS real_age
    FROM `physionet-data.mimiciv_3_1_hosp.admissions` adm
    INNER JOIN global_ref_anchor ref ON adm.hadm_id = ref.hadm_id
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON adm.subject_id = p.subject_id
),

patient_stratification AS (
    -- Step 3: Identify the "Global Label" for each patient to enable stratification.
    -- We define a patient as 'Positive' if they died in ANY of their admissions.
    SELECT 
        subject_id,
        MAX(hospital_expire_flag) AS patient_ever_died
    FROM encounter_lifecycle
    GROUP BY subject_id
),

split_assignment AS (
    -- Step 4: Deterministically assign split using hash of subject_id.
    -- We use ABS(MOD(hash, 100)) to create 100 buckets.
    SELECT 
        subject_id,
        CASE 
            WHEN ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 100)) < 80 THEN 'train'
            WHEN ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 100)) < 90 THEN 'val'
            ELSE 'test'
        END AS split
    FROM patient_stratification
),

readmission_logic AS (
    -- Step 5: Identify subsequent visits for readmission labeling.
    SELECT 
        e.*,
        LEAD(e.clinical_start) OVER (PARTITION BY e.subject_id ORDER BY e.clinical_start) AS next_admit_time
    FROM encounter_lifecycle e
)

-- Final SELECT joining labels with the split assignment and aggregated clinical codes
SELECT
    rl.subject_id,
    rl.hadm_id,
    sa.split, -- The 80/10/10 split 
    
    -- 1. RELATIVE TEMPORAL ANCHORS
    DATETIME_DIFF(rl.clinical_start, rl.t_zero, MINUTE) AS study_start_rel_min,
    DATETIME_DIFF(rl.clinical_end, rl.t_zero, MINUTE) AS study_end_rel_min,

    -- 2. MORTALITY & DECOMPENSATION LABELS
    rl.hospital_expire_flag AS label_mortality,
    CASE WHEN rl.deathtime IS NOT NULL THEN DATETIME_DIFF(rl.deathtime, rl.t_zero, MINUTE) ELSE NULL END AS death_time_rel_min,

    -- 3. 30-DAY READMISSION LABELS
    CASE 
        WHEN rl.hospital_expire_flag = 0 
             AND rl.admission_type NOT IN ('NEWBORN', 'AMBULATORY')
             AND DATETIME_DIFF(rl.next_admit_time, rl.dischtime, DAY) <= 30 
        THEN 1 ELSE 0 
    END AS label_readmission_30d,

    -- 4. LENGTH OF STAY (LOS)
    ROUND(DATETIME_DIFF(rl.clinical_end, rl.clinical_start, HOUR) / 24.0, 2) AS label_los_days,

    -- 5. CLINICAL METADATA (Aggregated Codes)
    icd.icd_codes,
    drg.drg_codes,

    -- 6. INDEPENDENT COHORT FLAGS
    CASE WHEN rl.real_age < 18 THEN 1 ELSE 0 END AS flag_is_pediatric,
    CASE WHEN DATETIME_DIFF(rl.clinical_end, rl.clinical_start, HOUR) < 24 THEN 1 ELSE 0 END AS flag_is_short_stay,
    CASE 
        WHEN DATETIME_DIFF(rl.clinical_end, rl.clinical_start, HOUR) < 0 THEN 1 
        ELSE 0 
    END AS flag_is_logical_error,
    CASE 
        WHEN real_age >= 18 
             AND DATETIME_DIFF(rl.clinical_end, rl.clinical_start, HOUR) > 0
        THEN 1 ELSE 0 
    END AS is_standard_cohort

FROM readmission_logic rl
INNER JOIN split_assignment sa ON rl.subject_id = sa.subject_id
LEFT JOIN aggregated_icd icd ON rl.hadm_id = icd.hadm_id
LEFT JOIN aggregated_drg drg ON rl.hadm_id = drg.hadm_id
ORDER BY rl.subject_id, rl.clinical_start;