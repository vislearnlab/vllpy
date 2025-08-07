from typing import Optional
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import ImageUrl,  ImageNdArray, NdArray
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

class CLIPImageEmbedding(BaseDoc):
    embedding: ImageNdArray[512]
    url: Optional[ImageUrl]
    text: Optional[str]
    normed_embedding: Optional[ImageNdArray[512]]

class CLIPTextEmbedding(TextDoc):
    embedding: NdArray[512]
    normed_embedding: Optional[NdArray[512]]

class EmbeddingStore():
    def __init__(self, EmbeddingList=None, FeatureGenerator=None, EmbeddingType=None):
        if EmbeddingType is None:
            self.EmbeddingType = CLIPImageEmbedding
        else:
            self.EmbeddingType = EmbeddingType
        if EmbeddingList is None:
            self.EmbeddingList = DocList[self.EmbeddingType]()
        else:
            self.EmbeddingList = EmbeddingList
        if FeatureGenerator is None:
            self.FeatureGenerator = CLIPGenerator()
        else:
            self.FeatureGenerator = FeatureGenerator
    
    @property
    def embeddings(self):
        return self.EmbeddingList.embedding

    def from_csv(csv_path): #"/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_mask/embeddings/image_embeddings/clip_image_embeddings_npy.csv", CLIPImageEmbedding
        embedding_doc = DocList[CLIPImageEmbedding]()
        if not os.path.isabs(csv_path):
            csv_path = Path(os.getcwd()) / csv_path
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                embedding = np.load(row['embedding_path'])
                doc = CLIPImageEmbedding(
                    url=row['row_id'],
                    embedding=embedding,
                    text=row['text'],
                    normed_embedding=None
                )
                embedding_doc.append(doc)
            except Exception as e:
                print(f"Failed to load embedding for {row['row_id']}: {e}")
        return EmbeddingStore(embedding_doc, CLIPGenerator())
    
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
    
    def from_doc(doc_input_path):
        doc_input_path = cleaned_doc_path(doc_input_path)
        embedding_list = DocList[CLIPImageEmbedding]().pull(doc_input_path, show_progress=True)
        return EmbeddingStore(embedding_list, CLIPGenerator())
    
    # TODO: add binary save
    def add_embedding(self, embedding, url=None, text=None):
        self.EmbeddingList.append(self.EmbeddingType(
            text=text,
            embedding=embedding,
            url=url,
            normed_embedding=None
        ))
    
    def add_embeddings(self, embeddings, urls, texts):
        seen_ids = set()  # To track unique ids (either url or text)
        temp_list = DocList[self.EmbeddingType](
            [
                self.EmbeddingType(
                    embedding=embedding,
                    url=url if isinstance(url, str) else None,
                    text=text,
                    normed_embedding=None,
                )
                for embedding, url, text in tqdm(zip_longest(embeddings, urls, texts, fillvalue=None))
                # for url to be unique id when saving embeddings it needs to be a string
                if (url if isinstance(url, str) else text) not in seen_ids and not seen_ids.add(url if isinstance(url, str) else text)
            ]
        )
        self.EmbeddingList.extend(temp_list)
    
    def search_store(self, text_query, limit=10, categories=None):
        query = self.FeatureGenerator.text_embeddings([text_query], normalize_embeddings=True)[0][0].cpu().numpy()
        doc_index = InMemoryExactNNIndex[CLIPImageEmbedding]()
        if categories is not None:
            filter_query = {
                'text': {'$in': categories}
            }
            filtered_docs = filter_docs(self.EmbeddingList, filter_query)
        else:
            filtered_docs = self.EmbeddingList
        filtered_docs.normed_embedding = normalize_embeddings(filtered_docs.embedding)
        doc_index.index(filtered_docs)
        # TODO: Use find_batched
        retrieved_docs, scores = doc_index.find(query, search_field='normed_embedding', limit=limit)
        return retrieved_docs, scores

    def retrieve_cross_similarity(self, embedding_list, sim_type="cosine"):
        sim_generator = SimilarityGenerator(similarity_type=sim_type, model=self.FeatureGenerator.model)
        return(sim_generator.cross_sims(self.EmbeddingList, embedding_list))
    
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
            return(sim_generator.all_sims(self.EmbeddingList.embedding, keys, output_path))
        else:
            return(sim_generator.specific_sims(self.EmbeddingList, text_pairs, output_path))

    def compute_text_rdm(self, sim_type="cosine", output_path=None):
        from vislearnlabpy.embeddings.similarity_utils import compute_rdm, plot_rdm
        texts = self.EmbeddingList.text
        if texts is None:
            raise ValueError("No text column in embeddings")
        # Unique texts
        unique_texts = sorted(set(texts))
        # Compute mean embedding for each unique text
        text_means = []
        for text in unique_texts:
            embeddings_for_text = [emb.embedding for emb in self.EmbeddingList if emb.text == text]
            if not embeddings_for_text:
                continue
            mean_embedding = np.mean(embeddings_for_text, axis=0)
            text_means.append(mean_embedding)
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
                y_labels=unique_texts
            )
        return rdm
