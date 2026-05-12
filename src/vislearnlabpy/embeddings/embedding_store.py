from typing import Optional
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import ImageUrl, ImageNdArray, NdArray
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.utils import cleaned_doc_path, normalize_embeddings, indexed_embeddings
from vislearnlabpy.embeddings.similarity_generator import SimilarityGenerator
from docarray.index import InMemoryExactNNIndex
from docarray.utils.filter import filter_docs
from itertools import zip_longest
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ── Dynamic schema factories ──────────────────────────────────────────────────

def _image_embedding_type(dim: int):
    """Return a BaseDoc subclass for image embeddings with the given dimension."""
    return type(
        f"ImageEmbedding{dim}",
        (BaseDoc,),
        {
            "__annotations__": {
                "embedding": ImageNdArray[dim],
                "url": Optional[ImageUrl],
                "text": Optional[str],
                "normed_embedding": Optional[ImageNdArray[dim]],
            }
        },
    )

def _text_embedding_type(dim: int):
    """Return a TextDoc subclass for text embeddings with the given dimension."""
    return type(
        f"TextEmbedding{dim}",
        (TextDoc,),
        {
            "__annotations__": {
                "embedding": NdArray[dim],
                "normed_embedding": Optional[NdArray[dim]],
            }
        },
    )

# ── Backward-compatible aliases (CLIP = 512) ─────────────────────────────────
CLIPImageEmbedding = _image_embedding_type(512)
CLIPTextEmbedding  = _text_embedding_type(512)


class EmbeddingStore():
    def __init__(self, EmbeddingList=None, FeatureGenerator=None, EmbeddingType=None, dim: int = None):
        # Resolve dim: explicit > inferred from generator > default 512
        if dim is None:
            dim = getattr(FeatureGenerator, "embedding_dim", 512) if FeatureGenerator is not None else 512

        if EmbeddingType is None:
            self.EmbeddingType = _image_embedding_type(dim)
        else:
            self.EmbeddingType = EmbeddingType

        if EmbeddingList is None:
            self.EmbeddingList = DocList[self.EmbeddingType]()
        else:
            self.EmbeddingList = EmbeddingList

        if FeatureGenerator is None:
            self.FeatureGenerator = CLIPGenerator()
        elif hasattr(FeatureGenerator, "model") and hasattr(FeatureGenerator.model, "text_embeddings"):
            # Unwrap EmbeddingGenerator wrappers — store only needs the raw model
            self.FeatureGenerator = FeatureGenerator.model
        else:
            self.FeatureGenerator = FeatureGenerator

    @property
    def embeddings(self):
        return self.EmbeddingList.embedding

    def from_csv(csv_path, feature_generator=None):
        if not os.path.isabs(csv_path):
            csv_path = Path(os.getcwd()) / csv_path
        df = pd.read_csv(csv_path)
        # Detect format: npy paths vs inline numeric columns
        inline_cols = [c for c in df.columns if c.isdigit()]
        if inline_cols:
            # Inline CSV: columns "0", "1", ... hold the embedding values
            dim = len(inline_cols)
            EmbType = _image_embedding_type(dim)
            embedding_doc = DocList[EmbType]()
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    embedding = np.array([row[c] for c in inline_cols], dtype=np.float32)
                    embedding_doc.append(EmbType(
                        url=row.get("row_id"),
                        embedding=embedding,
                        text=row.get("text"),
                        normed_embedding=None,
                    ))
                except Exception as e:
                    print(f"Failed to load row {row.get('row_id')}: {e}")
        else:
            # npy-path CSV: infer dim from first loadable file
            dim = 512
            for _, row in df.iterrows():
                try:
                    dim = np.load(row["embedding_path"]).shape[-1]
                    break
                except Exception:
                    continue
            EmbType = _image_embedding_type(dim)
            embedding_doc = DocList[EmbType]()
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    embedding = np.load(row["embedding_path"])
                    embedding_doc.append(EmbType(
                        url=row["row_id"],
                        embedding=embedding,
                        text=row.get("text"),
                        normed_embedding=None,
                    ))
                except Exception as e:
                    print(f"Failed to load embedding for {row['row_id']}: {e}")
        return EmbeddingStore(embedding_doc, feature_generator, EmbeddingType=EmbType, dim=dim)

    def to_base_csv(self, csv_output_path):
        row_data = [
            {"row_id": url, **indexed_embeddings(embedding)}
            for url, embedding in zip(self.EmbeddingList.url, self.EmbeddingList.embedding)
        ]
        pd.DataFrame(row_data).to_csv(csv_output_path, index=False)

    def to_doc(self, doc_output_path):
        self.output_path = cleaned_doc_path(doc_output_path)
        self.EmbeddingList.push(self.output_path)
        return self.output_path

    def from_doc(doc_input_path, dim: int = None, feature_generator=None):
        """Load a DocArray store from disk.

        dim: embedding dimension.  If None, it is inferred from the first document.
             Pass explicitly (e.g. dim=1024) when loading non-CLIP models.
        """
        from docarray.typing import AnyTensor
        if feature_generator is not None and hasattr(feature_generator, "embedding_dim"):
            dim = feature_generator.embedding_dim

        doc_input_path = cleaned_doc_path(doc_input_path)

        if dim is None:
            # Load with a flexible schema to peek at the actual dimension
            FlexDoc = type("FlexDoc", (BaseDoc,), {
                "__annotations__": {
                    "embedding": AnyTensor,
                    "url": Optional[ImageUrl],
                    "text": Optional[str],
                    "normed_embedding": Optional[AnyTensor],
                }
            })
            raw_list = DocList[FlexDoc]().pull(doc_input_path, show_progress=False)
            dim = int(raw_list[0].embedding.shape[-1]) if len(raw_list) > 0 else 512

        EmbType = _image_embedding_type(dim)
        typed_list = DocList[EmbType]().pull(doc_input_path, show_progress=True)
        return EmbeddingStore(typed_list, feature_generator, EmbeddingType=EmbType, dim=dim)

    # TODO: add binary save
    def add_embedding(self, embedding, url=None, text=None):
        self.EmbeddingList.append(self.EmbeddingType(
            text=text,
            embedding=embedding,
            url=url,
            normed_embedding=None,
        ))

    def add_embeddings(self, embeddings, urls, texts):
        seen_ids = set()
        temp_list = DocList[self.EmbeddingType](
            [
                self.EmbeddingType(
                    embedding=embedding,
                    url=url if isinstance(url, str) else None,
                    text=text,
                    normed_embedding=None,
                )
                for embedding, url, text in tqdm(zip_longest(embeddings, urls, texts, fillvalue=None))
                if (url if isinstance(url, str) else text) not in seen_ids
                and not seen_ids.add(url if isinstance(url, str) else text)
            ]
        )
        self.EmbeddingList.extend(temp_list)

    def search_store(self, text_query, limit=10, categories=None):
        query = self.FeatureGenerator.text_embeddings([text_query], normalize_embeddings=True)[0].cpu().numpy()
        doc_index = InMemoryExactNNIndex[self.EmbeddingType]()
        if categories is not None:
            filtered_docs = DocList[self.EmbeddingType](filter_docs(self.EmbeddingList, {"text": {"$in": categories}}))
        else:
            filtered_docs = self.EmbeddingList
        filtered_docs.normed_embedding = normalize_embeddings(filtered_docs.embedding)
        doc_index.index(filtered_docs)
        retrieved_docs, scores = doc_index.find(query, search_field="normed_embedding", limit=limit)
        return retrieved_docs, scores

    def retrieve_cross_similarity(self, embedding_list, sim_type="cosine"):
        sim_generator = SimilarityGenerator(similarity_type=sim_type, model=self.FeatureGenerator.model)
        return sim_generator.cross_sims(self.EmbeddingList, embedding_list)

    def retrieve_similarities(self, sim_type="cosine", output_path=None, text_pairs=None):
        sim_generator = SimilarityGenerator(similarity_type=sim_type, model=self.FeatureGenerator.model)
        if text_pairs is None:
            # again assuming url as id, followed by text. Wondering now if we do need an explicit id column.
            texts = self.EmbeddingList.text
            urls = self.EmbeddingList.url
            if urls and urls[0] is not None:
                keys = urls
            elif texts:
                keys = texts
            else:
                keys = None
            return sim_generator.all_sims(self.EmbeddingList.embedding, keys, output_path)
        else:
            return sim_generator.specific_sims(self.EmbeddingList, text_pairs, output_path)

    def compute_text_rdm(self, sim_type="cosine", output_path=None, order=None):
        from vislearnlabpy.embeddings.similarity_utils import compute_rdm, plot_rdm
        texts = self.EmbeddingList.text
        if texts is None:
            raise ValueError("No text column in embeddings")
        # Unique texts
        unique_texts = sorted(set(texts))
        
        if order is not None:
            missing = set(unique_texts) - set(order)
            extra = set(order) - set(unique_texts)
            if extra:
                raise ValueError(f"order contains labels not in embeddings: {extra}")
            if missing:
                print(f"Skipping labels in embeddings not in passed in list: {missing}")
            unique_texts = list(order) 
        
        # Compute mean embedding for each unique text
        text_means = []
        for text in unique_texts:
            embeddings_for_text = [emb.embedding for emb in self.EmbeddingList if emb.text == text]
            if not embeddings_for_text:
                continue
            text_means.append(np.mean(embeddings_for_text, axis=0))
        text_means = np.stack(text_means)
        # Compute RDM
        rdm = compute_rdm(text_means, method=sim_type)
        # Plot RDM if output_path is given
        if output_path:
            plot_rdm(
                out_path=output_path,
                X=text_means,
                method=sim_type,
                x_labels=unique_texts,
                y_labels=unique_texts,
            )
        return rdm
    
