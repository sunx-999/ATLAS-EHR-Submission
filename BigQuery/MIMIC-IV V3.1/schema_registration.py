import pandas as pd
import json
import torch
import os


class SchemaRegistrar:
    """
    ATLAS-EHR Final Registration Infrastructure.
    - Encapsulates Trimmed Statistics into model protocol.
    - Preserves feature_category for state/intervention distinction.
    - Zero-Variance Collapse for model safety.
    """
    def __init__(self, raw_csv_path, output_dir="BigQuery"):
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Input file not found: {raw_csv_path}")
            
        self.df = pd.read_csv(raw_csv_path)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)


    def run(self):
        print("Finalizing Protocol...")
        
        # 1. Frequency Gate: Remove sparse variables based on training set counts
        df = self.df[self.df['sample_count'] >= 100].copy()
        
        # 2. Normalize model types to lowercase
        df['model_type'] = df['model_type'].str.lower()
        
        # 3. Zero-variance check: Handle cases where std is 0 in the P5-P95 range
        zero_var_mask = (df['model_type'] == 'numeric') & (df['std'] == 0)
        df.loc[zero_var_mask, 'model_type'] = 'categorical'
        
        # 4. Assign Vocab IDs (ID 0 reserved for Padding)
        df = df.sort_values(['model_type', 'normalized_concept_label']).reset_index(drop=True)
        df['vid'] = range(1, len(df) + 1)

        # 5. Construct Final Protocol Records
        schema_records = []
        
        for _, row in df.iterrows():
            label = row['normalized_concept_label']
            # Protocol entry including the required feature_category column
            entry = {
                "normalized_concept_label": label,
                "model_type": row['model_type'], 
                "feature_category": row['feature_category'], # Preserved from SQL output
                "vid": int(row['vid']), 
                "sample_count": int(row['sample_count'])
            }
            if row['model_type'] == 'numeric':
                entry.update({
                    "mean": float(row['mean']), 
                    "std": float(row['std']), 
                    "p5": float(row['p5']), 
                    "p95": float(row['p95']), 
                    "use_clipping": True
                })
            schema_records.append(entry)
        # 6. Save Final Artifacts
        schema_path = os.path.join(self.output_dir, 'schema_processed.csv')
        pd.DataFrame(schema_records).to_csv(schema_path, index=False)
        print(f"Schema (CSV) saved to {schema_path}")


if __name__ == "__main__":
    registrar = SchemaRegistrar("BigQuery/schema_raw.csv")
    registrar.run()