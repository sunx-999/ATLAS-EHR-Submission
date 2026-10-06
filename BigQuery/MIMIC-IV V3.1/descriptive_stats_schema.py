"""
ATLAS-EHR: Schema Statistical Audit
Analyzes registration metadata and embedding distributions.
"""
import json
import pandas as pd

def audit_schema(file_path='BigQuery/schema_processed.json'):
    """
    Performs a stratified audit of the clinical manifold, 
    calculating category distributions and inferred type frequencies.
    """
    # Load schema into a DataFrame for analysis
    with open(file_path, 'r') as f:
        schema_data = json.load(f)
    df = pd.DataFrame(schema_data)
    
    # --- Part 1: Semantic Separation Analysis ---
    # Stratify by clinical category: States (Observations) vs Interventions (Actions)
    states_df = df[df['category'] == 'state']
    interventions_df = df[df['category'] == 'intervention']
    
    # Calculate unique label sets to check for semantic overlap
    unique_states = set(states_df['item'].unique())
    unique_interventions = set(interventions_df['item'].unique())
    overlap = unique_states.intersection(unique_interventions)
    
    # --- Part 2: Type Distribution Function ---
    def calculate_type_stats(sub_df):
        """Calculates raw counts and percentages for inferred feature types."""
        counts = sub_df['type'].value_counts()
        percentages = sub_df['type'].value_counts(normalize=True) * 100
        return pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentages.round(2)
        })

    # --- Part 3: Disclosure Report Generation ---
    print("="*60)
    print(" CLINICAL DATA MANIFOLD AUDIT ")
    print("="*60)
    print(f"Total entries in Schema:      {len(df)}")
    print(f"Unique State Labels:          {len(unique_states)}")
    print(f"Unique Intervention Labels:   {len(unique_interventions)}")
    print(f"Labels shared across types:   {len(overlap)} (Ideal: 0 if units differ)")
    
    print("\n[Feature Type Distribution: STATES]")
    print("-" * 40)
    print(calculate_type_stats(states_df))

    print("\n[Feature Type Distribution: INTERVENTIONS]")
    print("-" * 40)
    print(calculate_type_stats(interventions_df))
    print("="*60)

if __name__ == "__main__":
    audit_schema()