from typing import Optional
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import ImageUrl,  ImageNdArray, NdArray
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.utils import cleaned_doc_path, normalize_embeddings
from docarray.index import InMemoryExactNNIndex
from docarray.utils.filter import filter_docs
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
        temp_list = DocList[self.EmbeddingType](
            [
            self.EmbeddingType(
                embedding=embedding,
                url=url,
                text=text,
                normed_embedding=None,
            )
        for embedding, url, text in tqdm(zip(embeddings, urls, texts))
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
        retrieved_docs, scores = doc_index.find(query, search_field='normed_embedding', limit=limit)
        return retrieved_docs, scores
