import pandas as pd
import numpy as np
import re
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class E2EMapper:
    """
    ATLAS-EHR End-to-End Mapping Script.
    Implements Triple-Axis Semantic Isolation: (Canonical, Source, Unit, Polarity).
    """
    def __init__(self, meta_path, similarity_threshold=0.98):
        self.df = pd.read_csv(meta_path)
        self.threshold = similarity_threshold
        
        # Core Unit Physics (Unit-Aware Consolidation)
        self.UNIT_REGISTRY = {
            '°f': ('temp', '°C', 5/9, -32.0),
            'fahrenheit': ('temp', '°C', 5/9, -32.0),
            '°c': ('temp', '°C', 1.0, 0.0),
            'celsius': ('temp', '°C', 1.0, 0.0),
            'k/ul': ('conc', 'k/ul', 1.0, 0.0),
            '#/hpf': ('micro', 'hpf', 1.0, 0.0),
            'mg/dl': ('conc', 'mg/dl', 1.0, 0.0),
            'unspecified': ('misc', 'none', 1.0, 0.0)
        }

    def get_source_group(self, table):
        """Classifies complex table names into clinical contexts."""
        table = str(table).lower()
        if 'icu' in table: return 'icu'
        if 'labevents' in table: return 'lab'
        if 'emar' in table or 'inputevents' in table: return 'med'
        return 'other'

    def get_polarity(self, text):
        """Extracts polarity to prevent High/Low alarm collisions."""
        text = str(text).lower()
        if 'high' in text or 'alarm - h' in text: return 'hi'
        if 'low' in text or 'alarm - l' in text: return 'lo'
        return 'std'

    def clean_text(self, text):
        if not isinstance(text, str): return "unspecified"
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def run_mapping(self):
        print("Step 1: Bio_ClinicalBERT Semantic Clustering...")
        raw_labels = self.df['concept_label'].unique().tolist()
        clean_to_raw = {self.clean_text(lbl): lbl for lbl in raw_labels}
        unique_clean = list(clean_to_raw.keys())

        # Load Bio_ClinicalBERT (Industry Standard for EHR)
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        batch_size = 32
        embeddings = []
        model.eval()
        for i in tqdm(range(0, len(unique_clean), batch_size), desc="BERT Embedding Generation"):
            batch = unique_clean[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling proved more robust for medication specificity than CLS token alone
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)

        sim_matrix = cosine_similarity(embeddings)
        label_to_canonical = {}
        visited = set()

        for i in range(len(unique_clean)):
            if i in visited: continue
            sim_indices = np.where(sim_matrix[i] >= self.threshold)[0]
            # Use the shortest label in cluster as the canonical representative
            canonical = sorted([unique_clean[idx] for idx in sim_indices], key=len)[0]
            for idx in sim_indices:
                label_to_canonical[unique_clean[idx]] = canonical
                visited.add(idx)

        print("Step 2: Triple-Axis Semantic Isolation...")
        mapping_data = []
        for _, row in self.df.iterrows():
            raw_label = row['concept_label']
            clean_lbl = self.clean_text(raw_label)
            
            # Axis 1: Semantic Canonical Label
            canonical = label_to_canonical[clean_lbl]
            
            # Axis 2: Source Group (Clinical Context)
            source_grp = self.get_source_group(row['source_table'])
            
            # Axis 3: Unit Dimension (Standardized Suffix)
            raw_u = str(row['raw_unit']).lower()
            u_info = self.UNIT_REGISTRY.get(raw_u, ('misc', str(row['raw_unit']), 1.0, 0.0))
            target_unit = u_info[1]
            u_suffix = re.sub(r'[^a-z0-9]', '', target_unit.lower()) if target_unit != 'none' else ''
            
            # Axis 4: Polarity Guard (Hi/Lo)
            polarity = self.get_polarity(raw_label)

            # --- Target Identifier Construction ---
            # Composite ID ensures no collisions between distinct modes of observation
            parts = [canonical, source_grp]
            if u_suffix: parts.append(u_suffix)
            if polarity != 'std': parts.append(polarity)
            
            final_target_label = "_".join(parts)

            mapping_data.append({
                'raw_label': raw_label,
                'raw_unit': row['raw_unit'],
                'source_table': row['source_table'],
                'target_label': final_target_label,
                'target_unit': target_unit,
                'multiplier': u_info[2],
                'offset': u_info[3],
                'suggested_type': row['suggested_type_hint']
            })

        return pd.DataFrame(mapping_data)

if __name__ == "__main__":
    # Input Standard: BigQuery/meta_train.csv
    mapper = E2EMapper("BigQuery/meta_train.csv")
    mapping_results = mapper.run_mapping()
    mapping_results.to_csv("BigQuery/e2e_mapping.csv", index=False)
    print("\nMapping Infrastructure Successfully Applied.")