import numpy as np
from vislearnlabpy.embeddings.similarity_utils import *
from itertools import combinations
import pandas as pd

class SimilarityGenerator():
    def __init__(self, similarity_type, model):
        self.set_sim_matrix_fn(similarity_type)
        self.model = model

    def set_sim_matrix_fn(self, similarity_type):
        self.similarity_type = similarity_type
        if self.similarity_type == "cosine":
            self.sim_matrix_fn = cosine_matrix
        # pearsons
        elif self.similarity_type == "cor":
            self.sim_matrix_fn = correlation_matrix
        elif self.similarity_type == "gaussian":
            self.sim_matrix_fn = gaussian_kernel
        elif self.similarity_type == "squared_dist":
            self.sim_matrix_fn = squared_dists
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
    
    def _save_csv(self, similarities, output_csv):
        similarities_df = pd.DataFrame(similarities)
        if output_csv is not None:
            output_csv_path = Path(output_csv)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            similarities_df.to_csv(output_csv, index=False)
        return similarities_df
    
    def all_sims(self, embeddings, texts=None, output_csv=None):
        """All-pairs similarity.

        ``embeddings`` may be:
        - a ``{text: ndarray}`` dict  (``texts`` is ignored)
        - a list/array of ndarrays   (``texts`` must be provided)
        """
        if isinstance(embeddings, dict):
            texts = list(embeddings.keys())
            embeddings = np.stack(list(embeddings.values()))
        else:
            embeddings = np.stack(embeddings)
        sim_matrix = self.sim_matrix_fn(embeddings)
        similarities = [
            {
                f"{self.similarity_type}_similarity": round(float(sim_matrix[i, j]), 4),
                "text1": f"{texts[i]}",
                "text2": f"{texts[j]}"
            }
            for i, j in combinations(range(len(texts)), 2)
        ]
        return self._save_csv(similarities, output_csv)
    
    def _sim_key(embedding_store_df):
        if "url" in embedding_store_df and pd.notna(embedding_store_df["url"].iloc[0]):
            key = "url"
        else:
            key = "text"
        return key
    
    def cross_sims(self, list1, list2, output_csv=None):
        """
        Compute all pairwise similarities between the two embedding lists.

        Args:
            list1, list2: EmbeddingList docarray objects to compute similarities between
            output_csv: Path to save the similarity matrix to. If None, the similarity matrix is not saved.

        Returns:
            A DataFrame with columns for the text pairs and their respective similarity scores.
        """
        emb1, emb2 = np.stack(list1.embedding), np.stack(list2.embedding)
        df1, df2 = list1.to_dataframe(), list2.to_dataframe()
        key1, key2 = SimilarityGenerator._sim_key(df1), SimilarityGenerator._sim_key(df2)
        texts1, texts2 = df1[key1].tolist(), df2[key2].tolist()
        sim_matrix = self.sim_matrix_fn(emb1, emb2)

        sims = [
            {f"{self.similarity_type}_similarity": round(sim_matrix[i, j], 4), "text1": texts1[i], "text2": texts2[j]}
            for i in range(len(texts1)) for j in range(len(texts2))
        ]
        return self._save_csv(sims, output_csv)
        
    def specific_sims(self, embeddings, text_pairs: List[Tuple[str, str]], output_csv=None,
                      use_urls=False):
        """Compute similarity for specific pairs.

        ``embeddings`` may be:
        - a ``{key: ndarray}`` dict (keyed by text or URL)
        - a DocArray EmbeddingList (keyed by text by default, or URL if use_urls=True)

        use_urls : bool
            If True, look up embeddings by URL instead of text label. Pairs should
            then be (url1, url2) tuples.
        """
        if isinstance(embeddings, dict):
            embeddings_dict = embeddings
        else:
            df = embeddings.to_dataframe()
            key_col = "url" if use_urls else "text"
            embeddings_dict = {
                row[key_col]: row["embedding"]
                for _, row in df.iterrows()
                if row.get(key_col) is not None
            }
        similarities = []
        for (text1, text2) in text_pairs:
            if text1 in embeddings_dict and text2 in embeddings_dict:
                emb1, emb2 = embeddings_dict[text1], embeddings_dict[text2]
                if self.similarity_type == "cosine":
                    sim = cosine_sim(emb1, emb2)
                else:
                    sim = float(self.sim_matrix_fn(np.stack([emb1, emb2]))[0, 1])
                similarities.append({
                    f"{self.similarity_type}_similarity": sim,
                    "text1": text1,
                    "text2": text2,
                })
            else:
                print(f"Skipping missing pair of {text1} and {text2}")
        return self._save_csv(similarities, output_csv)
    