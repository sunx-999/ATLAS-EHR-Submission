-- ATLAS-EHR: Universal ID-Label Mapping Audit

-- 1. States
WITH state_mapping AS (
    SELECT concept_id, concept_label, 'state' AS type
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.states`
    GROUP BY 1, 2
),

-- 2. Interventions 
interv_mapping AS (
    SELECT concept_id, concept_label, 'intervention' AS type
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions`
    GROUP BY 1, 2
),

combined_mapping AS (
    SELECT * FROM state_mapping
    UNION ALL
    SELECT * FROM interv_mapping
)

-- Check A: One ID, Multiple Labels?
SELECT 
    type, 
    'ID_TO_MANY_LABELS' AS conflict_type, 
    concept_id AS key_id, 
    COUNT(DISTINCT concept_label) AS variation_count,
    ARRAY_AGG(DISTINCT concept_label) AS label_variations
FROM combined_mapping
WHERE concept_id IS NOT NULL
GROUP BY 1, 2, 3
HAVING variation_count > 1

UNION ALL

-- Check B: One Label, Multiple IDs?
SELECT 
    type, 
    'LABEL_TO_MANY_IDS' AS conflict_type, 
    concept_label AS key_label, 
    COUNT(DISTINCT concept_id) AS variation_count,
    ARRAY_AGG(DISTINCT concept_id) AS id_variations
FROM combined_mapping
GROUP BY 1, 2, 3
HAVING variation_count > 1
ORDER BY type, conflict_type;