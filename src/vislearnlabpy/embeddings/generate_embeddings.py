from typing import Optional
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.stimuli_loader import StimuliLoader
from vislearnlabpy.embeddings.utils import save_df
from vislearnlabpy.embeddings.embedding_store import CLIPImageEmbedding, CLIPTextEmbedding, EmbeddingStore
import torch
import os
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

torch.set_num_threads(32)

class EmbeddingGenerator():
    def __init__(self, device=None, model_type="clip", model=None, output_type="csv", normalize_embeddings=True):
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
    def process_embedding_row(self, embedding, id, save_path=None, store:Optional[EmbeddingStore]=None, text=None):
        curr_row_data = {'row_id': id}
        # if id is image path we're assuming
        url = id if id.startswith("/") else None
        if text is not None:
            curr_row_data['text'] = text
        if self.output_type == "csv":
            curr_embeddings = embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding
            # new row with a separate column for each number in the 512 dimensions and one for the image_path as the row_id
            for i, value in enumerate(curr_embeddings):
                curr_row_data[f"{i}"] = value.item() if isinstance(value, torch.Tensor) else value
        elif self.output_type == "npy":
            # save embedding as npy in output save directory
            curr_row_data["embedding_path"] = self.save_embedding(embedding, id, save_path)
        elif self.output_type == "doc":
            # using docarray
            store.add_embedding(embedding, url=url, text=text)
        return curr_row_data, store
    
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
        if self.output_type != "doc":
            save_df(pd.DataFrame(row_data), Path(filepath).name, full_save_path, overwrite=overwrite)
        else:
            store.to_doc(filepath.removesuffix(".csv"))
    
    def save_text_embeddings(self, texts, save_path, overwrite, normalize_embeddings):
        store = EmbeddingStore(EmbeddingType=CLIPTextEmbedding)
        filepath, full_save_path = self.create_files(type="text", save_path=save_path)
        existing_row_ids = EmbeddingGenerator._get_existing_row_ids(filepath)
        with torch.no_grad():
            row_data = []
            # TODO: add batching, maybe just switch to dataloader
            for text in tqdm(texts, desc="Calculating text embeddings"):
                if text not in existing_row_ids or overwrite:
                    curr_text_embeddings = self.model.text_embeddings([text], normalize_embeddings)[0][0].cpu().numpy()
                    curr_row_data, store = self.process_embedding_row(curr_text_embeddings, id=text, save_path=full_save_path, store=store)    
                    row_data.append(curr_row_data)
        if len(row_data) > 0:
            self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        
    def save_image_embeddings(self, save_path=None, normalize_embeddings=True, overwrite=False, save_every_batch=False):
        store = EmbeddingStore(EmbeddingType=CLIPImageEmbedding)
        filepath, full_save_path = self.create_files(save_path)
        existing_row_ids = EmbeddingGenerator._get_existing_row_ids(filepath)
        all_text = set()
        row_data = []
        with torch.no_grad():
            for d in tqdm(self.model.dataloader, desc=f"Calculating {self.model_type} embeddings", position=tqdm._get_free_pos()):
                if save_every_batch:
                    row_data = []
                curr_image_embeddings = self.model.image_embeddings(d['images'], normalize_embeddings)
                count = 0
                for (image, curr_id, text) in tqdm(zip(d['images'], d['id'], d['text'] if d['text'] else itertools.repeat(None, len(d['images']))), total=len(d['images']), desc="Current batch", position=tqdm._get_free_pos()):
                    if curr_id not in existing_row_ids or overwrite:
                        curr_row_data, store = self.process_embedding_row(embedding=curr_image_embeddings[count][0].cpu().numpy(), id=curr_id, save_path=full_save_path, store=store, text=text)
                        row_data.append(curr_row_data)
                    if text is not None:
                        all_text.add(text)
                    count = count + 1
                if len(row_data) > 0 and save_every_batch:
                    self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        if not save_every_batch and len(row_data) > 0:
            self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)
        self.save_text_embeddings(all_text, save_path, overwrite, normalize_embeddings)
    
    def generate_image_embeddings(self, output_path=None, overwrite=False, 
                                  normalize=True, input_csv=None, 
                                  input_dir=None, batch_size=1, save_every_batch=False):
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
                id_column="image1"
            ).dataloader()
        self.model = CLIPGenerator(device=self.device, dataloader=images_dataloader)
        self.save_image_embeddings(save_path=output_path,normalize_embeddings=normalize, overwrite=overwrite, save_every_batch=save_every_batch)
