-- ATLAS-EHR: State-Only Frequency Calculation (Strict Training Logic)
-- Logic:
-- 1. Use only the 'train' split from ehr_model_ready.
-- 2. Map integer 'vid' back to semantic labels using schema_processed.
-- 3. Exclude timestamps that do not contain at least one 'state' variable.
-- 4. Output frequency per vid and label for guided masking.

WITH exploded_train_ehr AS (
    -- Step 1: Flatten trajectories and resolve semantic info
    SELECT 
        t.hadm_id,
        e.offset_min,
        sch.vid, -- Added vid for reference
        sch.normalized_concept_label,
        e.cat AS event_category
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.ehr_model_ready` t
    CROSS JOIN UNNEST(t.trajectory) AS e
    -- Resolve label and metadata via vocab ID
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.schema_processed` sch
        ON e.vid = sch.vid
    WHERE t.split = 'train'
),

valid_timestamps AS (
    -- Step 2: Define valid backbone timestamps (must have >= 1 state)
    SELECT 
        hadm_id, 
        offset_min
    FROM exploded_train_ehr
    GROUP BY hadm_id, offset_min
    HAVING COUNTIF(event_category = 'state') > 0
),

denominator AS (
    -- Step 3: Global valid timestamp count (Denominator)
    SELECT COUNT(*) AS total_valid_ts FROM valid_timestamps
),

variable_counts AS (
    -- Step 4: Count coverage per variable (Numerator)
    SELECT 
        te.vid, -- Grouping by vid to preserve it in final output
        te.normalized_concept_label,
        COUNT(*) AS variable_ts_count
    FROM exploded_train_ehr te
    INNER JOIN valid_timestamps vt 
        ON te.hadm_id = vt.hadm_id 
        AND te.offset_min = vt.offset_min
    WHERE te.event_category = 'state'
    GROUP BY 1, 2 -- Group by both ID and Label
)

-- Step 5: Final Frequency Computation
SELECT 
    vc.vid, -- Included in final selection as requested
    vc.normalized_concept_label,
    vc.variable_ts_count,
    d.total_valid_ts,
    -- Age/Gender result in 1.0 due to broadcasting in ehr_model_ready.sql
    SAFE_DIVIDE(vc.variable_ts_count, d.total_valid_ts) AS frequency
FROM variable_counts vc
CROSS JOIN denominator d
ORDER BY frequency DESC, vid ASC;