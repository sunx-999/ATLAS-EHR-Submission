-- Objective: Comprehensive split-level audit for publication.
-- Metrics: 1/99, 2/98, 5/95, 10/90, 15/85, 20/80 percentiles.

WITH cohort_with_metrics AS (
    -- Step 1: Join pruned trajectories with labels
    SELECT 
        l.split,
        l.label_mortality,
        l.label_readmission_30d,
        l.label_los_days,
        t.subject_id,
        t.hadm_id,
        t.num_records,
        t.num_timestamps
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.ehr` t
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` l 
        ON t.hadm_id = l.hadm_id
),

distribution_quantiles AS (
    -- Step 2: Compute percentiles
    SELECT 
        split,
        -- Record Distribution (Total Tokens/Events)
        MIN(num_records) as rec_min,
        APPROX_QUANTILES(num_records, 100)[OFFSET(1)] as rec_p1,
        APPROX_QUANTILES(num_records, 100)[OFFSET(2)] as rec_p2,
        APPROX_QUANTILES(num_records, 100)[OFFSET(5)] as rec_p5,
        APPROX_QUANTILES(num_records, 100)[OFFSET(10)] as rec_p10,
        APPROX_QUANTILES(num_records, 100)[OFFSET(15)] as rec_p15,
        APPROX_QUANTILES(num_records, 100)[OFFSET(20)] as rec_p20,
        APPROX_QUANTILES(num_records, 100)[OFFSET(50)] as rec_p50,
        APPROX_QUANTILES(num_records, 100)[OFFSET(80)] as rec_p80,
        APPROX_QUANTILES(num_records, 100)[OFFSET(85)] as rec_p85,
        APPROX_QUANTILES(num_records, 100)[OFFSET(90)] as rec_p90,
        APPROX_QUANTILES(num_records, 100)[OFFSET(95)] as rec_p95,
        APPROX_QUANTILES(num_records, 100)[OFFSET(98)] as rec_p98,
        APPROX_QUANTILES(num_records, 100)[OFFSET(99)] as rec_p99,
        MAX(num_records) as rec_max,
        ROUND(AVG(num_records), 2) as rec_mean,
        
        -- Timestamp Distribution (Temporal Steps)
        MIN(num_timestamps) as ts_min,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(1)] as ts_p1,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(2)] as ts_p2,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(5)] as ts_p5,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(10)] as ts_p10,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(15)] as ts_p15,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(20)] as ts_p20,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(50)] as ts_p50,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(80)] as ts_p80,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(85)] as ts_p85,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(90)] as ts_p90,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(95)] as ts_p95,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(98)] as ts_p98,
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(99)] as ts_p99,
        MAX(num_timestamps) as ts_max,
        ROUND(AVG(num_timestamps), 2) as ts_mean
    FROM cohort_with_metrics
    GROUP BY split
),

population_outcomes AS (
    -- Step 3: Compute population metrics per split
    SELECT 
        split,
        COUNT(DISTINCT subject_id) AS n_subjects,
        COUNT(DISTINCT hadm_id) AS n_admissions,
        ROUND(AVG(label_mortality), 4) AS mortality_rate,
        ROUND(AVG(label_readmission_30d), 4) AS readmission_rate,
        ROUND(AVG(label_los_days), 2) AS los_mean,
        APPROX_QUANTILES(label_los_days, 100)[OFFSET(50)] AS los_median
    FROM cohort_with_metrics
    GROUP BY split
)

-- FINAL STEP: Consolidate Population, Outcome, and Detailed Distribution
SELECT 
    p.split,
    p.n_subjects,
    p.n_admissions,
    p.mortality_rate,
    p.readmission_rate,
    p.los_mean,
    p.los_median,
    
    -- Timestamp Full-Spectrum Distribution
    d.ts_mean, d.ts_p50 as ts_median, 
    d.ts_p1, d.ts_p2, d.ts_p5, d.ts_p10, d.ts_p15, d.ts_p20,
    d.ts_p80, d.ts_p85, d.ts_p90, d.ts_p95, d.ts_p98, d.ts_p99,
    
    -- Record Full-Spectrum Distribution
    d.rec_mean, d.rec_p50 as rec_median, 
    d.rec_p1, d.rec_p2, d.rec_p5, d.rec_p10, d.rec_p15, d.rec_p20,
    d.rec_p80, d.rec_p85, d.rec_p90, d.rec_p95, d.rec_p98, d.rec_p99
FROM population_outcomes p
LEFT JOIN distribution_quantiles d ON p.split = d.split
ORDER BY CASE p.split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END;
