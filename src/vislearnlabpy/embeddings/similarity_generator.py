import numpy as np
from vislearnlabpy.embeddings.similarity_utils import *
from itertools import combinations
import pandas as pd

class SimilarityGenerator():
    def __init__(self, similarity_type, model):
        self.similarity_type = similarity_type
        self.model = model

    def _save_csv(self, similarities, output_csv):
        similarities_df = pd.DataFrame(similarities)
        if output_csv is not None:
            output_csv_path = Path(output_csv)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            similarities_df.to_csv(output_csv, index=False)
        return similarities_df
    
    def all_sims(self, embeddings, texts, output_csv=None):
        embeddings = np.stack(embeddings)
        if self.similarity_type == "cosine":
            sim_matrix = cosine_matrix(embeddings)
        # pearsons
        elif self.similarity_type == "cor":
            sim_matrix = correlation_matrix(embeddings)
        elif self.similarity_type == "gaussian":
            sim_matrix = gaussian_kernel(embeddings)
        elif self.similarity_type == "squared_dist":
            sim_matrix = squared_dists(embeddings)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        similarities = [
            {
                f"{self.similarity_type}_similarity": round(float(sim_matrix[i, j]), 4),
                "text1": f"{texts[i]}",
                "text2": f"{texts[j]}"
            }
            for i, j in combinations(range(len(texts)), 2)
        ]
        return(self._save_csv(similarities, output_csv))
        
    def specific_sims(self, embeddings, text_pairs, output_csv=None):
        df = embeddings.to_dataframe()
        # Process all combinations of embeddings and texts
        similarities = []
        if "url" in df and pd.notna(df["url"].iloc[0]):
            key = "url"
        else:
            key = "text"
        existing_texts = df[key].values
        for (text1, text2) in text_pairs:
            if text1 in existing_texts and text2 in existing_texts:
                # Get text embeddings if available
                text1_embedding = df[df[key] == text1]["embedding"].iloc[0]
                text2_embedding = df[df[key] == text2]["embedding"].iloc[0]
                entry = {}
                if self.similarity_type == "cosine":
                    entry[f"{self.similarity_type}_similarity"] = cosine_sim(text1_embedding, text2_embedding)
                    entry["text1"] = text1
                    entry["text2"] = text2
                similarities.append(entry)
            else:
                print(
                    f"Skipping missing pair of {text1} and {text2}")
        # Save to CSV
        return(self._save_csv(similarities, output_csv))
    
    