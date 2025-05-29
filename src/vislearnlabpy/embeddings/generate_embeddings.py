from typing import Optional
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.stimuli_loader import StimuliLoader
from vislearnlabpy.embeddings.utils import save_df, indexed_embeddings
from vislearnlabpy.embeddings.embedding_store import CLIPImageEmbedding, CLIPTextEmbedding, EmbeddingStore
import torch
import os
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class EmbeddingGenerator():
    def __init__(self, device=None, model_type="clip", model=None, output_type="csv", normalize_embeddings=False):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_type = model_type
        self.model = model or CLIPGenerator()
        self.output_type = output_type
        self.normalize_embeddings = normalize_embeddings

    def save_embedding(self, embedding, curr_id, save_path, text=None):
        # if curr_id represents the full path of the image, portion off the last part and add 
        if os.path.exists(curr_id):
            sub_save_path = str(Path(curr_id).name)
            if text is not None:
                sub_save_path = f"{text}/{sub_save_path}"
        else:
            sub_save_path = curr_id
        embedding_output_path = Path(f"{save_path}/{sub_save_path}").with_suffix('.npy')
        # if save path is a relative path, assume that we want to save files in the current directory
        if not save_path.startswith("/"):
            embedding_output_path = Path(f"{os.getcwd()}/{embedding_output_path}")
        os.makedirs(embedding_output_path.parent, exist_ok=True)
        np.save(str(embedding_output_path), embedding)
        return str(embedding_output_path)
    
    # single file processing
    def process_embedding_row(self, embedding, id, save_path=None, text=None):
        curr_row_data = {'row_id': id}
        # if id is image path we're assuming
        # url = id if id.startswith("/") else None
        if text is not None:
            curr_row_data['text'] = text
        if self.output_type == "csv":
            embedding_data = indexed_embeddings(embedding)
            curr_row_data = curr_row_data | embedding_data
        elif self.output_type == "npy":
            # save embedding as npy in output save directory
            curr_row_data["embedding_path"] = self.save_embedding(embedding, id, save_path)
        return curr_row_data
    
    def create_files(self, save_path=None, type="image"):
        filename = f"{self.model_type}_{type}_embeddings_{self.output_type}.csv"
        save_path = os.path.join(os.getcwd(), "output") if save_path is None else str(save_path)
        full_save_path = os.path.join(save_path, f"{type}_embeddings")
        os.makedirs(full_save_path, exist_ok=True)
        filepath = os.path.join(full_save_path, filename)
        return filepath, full_save_path
    
    @staticmethod
    def _get_existing_row_ids(filepath):
        existing_row_ids = []
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            existing_row_ids = existing_df['row_id'].values
        return existing_row_ids
    
    def save_embedding_paths(self, row_data, store, filepath, full_save_path, overwrite):
        if self.output_type != "doc" and len(row_data) > 0:
            save_df(pd.DataFrame(row_data), Path(filepath).name, full_save_path, overwrite=overwrite)
        else:
            store.to_doc(filepath.removesuffix(".csv"))
    
    def save_text_embeddings(self, texts, save_path, overwrite):
        store = EmbeddingStore(EmbeddingType=CLIPTextEmbedding)
        filepath, full_save_path = self.create_files(type="text", save_path=save_path)
        existing_row_ids = EmbeddingGenerator._get_existing_row_ids(filepath)
        with torch.no_grad():
            row_data = []
            # TODO: add batching, maybe just switch to dataloader
            for text in tqdm(texts, desc="Calculating text embeddings", position=tqdm._get_free_pos()):
                if text not in existing_row_ids or overwrite:
                    curr_text_embeddings = self.model.text_embeddings([text], self.normalize_embeddings)[0][0].cpu().numpy()
                    if self.output_type == "doc":
                        store.add_embedding(curr_text_embeddings, url=None, text=text)
                    else:
                        curr_row_data = self.process_embedding_row(curr_text_embeddings, id=text, save_path=full_save_path)    
                        row_data.append(curr_row_data)
        self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        
    def save_image_embeddings(self, save_path=None, overwrite=False, save_every_batch=False):
        store = EmbeddingStore(EmbeddingType=CLIPImageEmbedding)
        filepath, full_save_path = self.create_files(save_path)
        existing_row_ids = EmbeddingGenerator._get_existing_row_ids(filepath)
        all_text = set()
        row_data = []
        with torch.no_grad():
            for d in tqdm(self.model.dataloader, desc=f"Calculating {self.model_type} embeddings", position=tqdm._get_free_pos()):
                if save_every_batch:
                    row_data = []
                curr_image_embeddings = self.model.image_embeddings(d['images'], self.normalize_embeddings)
                count = 0
                curr_image_embeddings = curr_image_embeddings.cpu().numpy()
                if d['text'] is not None:
                    all_text.update(item for item in d['text'] if item is not None)
                if self.output_type == "doc":
                    store.add_embeddings(curr_image_embeddings, d['item_id'], d['text'] if d['text'] else itertools.repeat(None, len(d['item_id'])))
                else:
                    for (curr_id, text) in tqdm(zip(d['item_id'], d['text'] if d['text'] else itertools.repeat(None, len(d['images']))), total=len(d['images']), desc="Current batch", position=tqdm._get_free_pos()):
                        if curr_id not in existing_row_ids or overwrite:
                            curr_row_data = self.process_embedding_row(embedding=curr_image_embeddings[count], id=curr_id, save_path=full_save_path, text=text)
                            row_data.append(curr_row_data)
                        count = count + 1
                if save_every_batch:
                    self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        if not save_every_batch:
            self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        self.save_text_embeddings(all_text, save_path, overwrite)
    
    def generate_image_embeddings(self, output_path=None, overwrite=False, 
                                  input_csv=None, input_dir=None, batch_size=1, save_every_batch=False, 
                                  id_column="image1"):
        if input_csv is None:
            if input_dir is None:
                raise Exception("Either input CSV or input image directory needs to be provided")
            images_dataloader = StimuliLoader(
                image_folder=input_dir,
                batch_size=batch_size,
                stimuli_type="images"
            ).dataloader()
        else:
            images_dataloader = StimuliLoader(
                image_folder=input_dir,
                dataset_file=input_csv,
                batch_size=batch_size,
                stimuli_type="images",
                id_column=id_column
            ).dataloader()
        self.model = CLIPGenerator(device=self.device, dataloader=images_dataloader)
        self.save_image_embeddings(save_path=output_path,overwrite=overwrite, save_every_batch=save_every_batch)
