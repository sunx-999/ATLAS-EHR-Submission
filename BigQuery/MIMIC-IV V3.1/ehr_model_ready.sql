-- ATLAS-EHR: Final Production Data Preparation
-- Hard Requirements:
-- 1. Broadcast Age and Gender ONLY from the first timestamp of each hadm_id.
-- 2. Target Labels: 'age_other_years', 'gender_other_f', 'gender_other_m'.
-- 3. Broadcast ONLY to timestamps containing at least one 'state' event.
-- 4. Intervention-only timestamps remain but receive NO broadcasted demographics.

CREATE OR REPLACE TABLE `mimic-iv-484300.mimic_iv_3_1_processed.ehr_model_ready` AS
WITH aligned_events AS (
    -- STEP 1: Feature Alignment with Frozen Vocabulary
    SELECT 
        t.subject_id, t.hadm_id, l.split,
        MIN(event.timestamp) OVER(PARTITION BY t.hadm_id) AS t_zero,
        event.timestamp,
        event.event_category AS true_category,
        sch.vid, 
        sch.normalized_concept_label, -- Used for precise demographic filtering
        sch.model_type, sch.mean, sch.std, sch.p5, sch.p95,
        event.scaled_value
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.ehr` AS t
    CROSS JOIN UNNEST(t.trajectory) AS event
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` AS l ON t.hadm_id = l.hadm_id
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.schema_processed` AS sch
        ON event.normalized_concept_label = sch.normalized_concept_label
),

demographic_source AS (
    -- STEP 2: Extract Age/Gender only from the FIRST timestamp of each admission
    -- Uses exact labels from schema_processed.csv as requested.
    SELECT *
    FROM aligned_events
    WHERE normalized_concept_label IN ('age_other_years', 'gender_other_f', 'gender_other_m')
    QUALIFY ROW_NUMBER() OVER(PARTITION BY hadm_id, vid ORDER BY timestamp ASC) = 1
),

state_backbone_ts AS (
    -- STEP 3: Identify timestamps that contain at least one 'state' observation
    -- These are the only timestamps eligible for demographic broadcasting.
    SELECT DISTINCT hadm_id, timestamp
    FROM aligned_events
    WHERE true_category = 'state'
    GROUP BY hadm_id, timestamp
    HAVING COUNTIF(true_category = 'state') > 0
),

broadcast_demographics AS (
    -- STEP 4: Perform the broadcast
    -- Join static demographics with valid state-containing timestamps.
    SELECT 
        s.hadm_id,
        t.timestamp,
        s.vid, s.model_type, s.mean, s.std, s.p5, s.p95, s.scaled_value, s.true_category
    FROM state_backbone_ts t
    INNER JOIN demographic_source s ON t.hadm_id = s.hadm_id
),

clinical_dynamic_events AS (
    -- STEP 5: Capture all other events
    -- Exclude original demographics to prevent double-counting.
    SELECT 
        hadm_id, timestamp,
        vid, model_type, mean, std, p5, p95, scaled_value, true_category
    FROM aligned_events
    WHERE normalized_concept_label NOT IN ('age_other_years', 'gender_other_f', 'gender_other_m')
),

combined_pool AS (
    -- STEP 6: Merge broadcasted rows with original dynamic observations
    -- Intervention-only timestamps will come from 'clinical_dynamic_events' 
    -- and will NOT have corresponding rows in 'broadcast_demographics'.
    SELECT * FROM broadcast_demographics
    UNION ALL
    SELECT * FROM clinical_dynamic_events
),

final_trajectories AS (
    -- STEP 7: Sequence Aggregation and Normalization
    SELECT 
        subject_id, hadm_id, split,
        ARRAY_AGG(
            STRUCT(
                vid,
                -- Relative minute from admission start
                TIMESTAMP_DIFF(timestamp, t_zero, MINUTE) AS offset_min,
                -- Normalization logic
                CASE 
                    WHEN model_type = 'numeric' THEN
                        SAFE_DIVIDE((LEAST(GREATEST(scaled_value, p5), p95) - mean), (std + 1e-8))
                    ELSE 1.0 
                END AS val_norm,
                true_category AS cat 
            ) 
            ORDER BY 
                timestamp ASC, 
                -- TIE-BREAKING: Age/Gender (state) before Interventions
                (CASE WHEN true_category = 'state' THEN 0 ELSE 1 END) ASC,
                vid ASC
        ) AS trajectory
    FROM (
        -- Re-join with t_zero and subject_id for aggregation
        SELECT c.*, a.t_zero, a.subject_id, a.split 
        FROM combined_pool c
        JOIN (SELECT DISTINCT hadm_id, t_zero, subject_id, split FROM aligned_events) a 
          ON c.hadm_id = a.hadm_id
    )
    GROUP BY subject_id, hadm_id, split
)

-- Final sharding and quality check
SELECT 
    *,
    MOD(ABS(FARM_FINGERPRINT(CAST(subject_id AS STRING))), 100) AS bucket_id
FROM final_trajectories
WHERE ARRAY_LENGTH(trajectory) > 0;