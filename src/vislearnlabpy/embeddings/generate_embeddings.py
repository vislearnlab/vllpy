import os
from pathlib import Path
from glob import glob
from typing import Iterable, List
import numpy as np
import pandas as pd
import torch
import itertools
from tqdm import tqdm
import ray
from torch.utils.data import Subset, DataLoader

# keep your original imports (adjust module paths as needed)
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.embeddings.stimuli_loader import StimuliLoader
from vislearnlabpy.embeddings.utils import save_df, indexed_embeddings
from vislearnlabpy.embeddings.embedding_store import (
    CLIPImageEmbedding,
    CLIPTextEmbedding,
    EmbeddingStore,
)

@ray.remote(num_gpus=0.3)
class EmbeddingGenerator:
    def __init__(
        self,
        input_dir: str,
        device=None,
        model_type="clip",
        model=None,
        output_type="csv",
        normalize_embeddings=False,
        transform=None,
        subdirs=False,
    ):
        # store input_dir so each actor can recreate DataLoader locally
        self.input_dir = input_dir
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_type = model_type
        # Model will be created lazily per-actor if not provided
        self.model = model or CLIPGenerator(device=self.device)
        self.output_type = output_type
        self.transform = transform
        self.normalize_embeddings = normalize_embeddings
        self.subdirs = subdirs

    def save_embedding(self, embedding, curr_id, save_path, text=None):
        if os.path.exists(curr_id):
            sub_save_path = str(Path(curr_id).name)
            if self.subdirs:
                sub_save_path = str(Path(curr_id).parent.name) + "/" + str(Path(curr_id).name)
            elif text is not None:
                sub_save_path = f"{text}/{sub_save_path}"
        else:
            sub_save_path = curr_id
        embedding_output_path = Path(f"{save_path}/{sub_save_path}").with_suffix(".npy")
        if not str(save_path).startswith("/"):
            embedding_output_path = Path(f"{os.getcwd()}/{embedding_output_path}")
        os.makedirs(embedding_output_path.parent, exist_ok=True)
        np.save(str(embedding_output_path), embedding)
        return str(embedding_output_path)

    def process_embedding_row(self, embedding, id, save_path=None, text=None):
        print("save path")
        print(save_path)
        curr_row_data = {"row_id": id}
        if text is not None:
            curr_row_data["text"] = text
        if self.output_type == "csv":
            embedding_data = indexed_embeddings(embedding)
            curr_row_data = curr_row_data | embedding_data
        elif self.output_type == "npy":
            curr_row_data["embedding_path"] = self.save_embedding(embedding, id, save_path, text=text)
        return curr_row_data

    def create_files(self, save_path=None, type="image"):
        filename = f"{self.model_type}_{type}_embeddings_{self.output_type}.csv"
        save_path = os.path.join(os.getcwd(), "output") if save_path is None else str(save_path)
        full_save_path = os.path.join(save_path, f"{type}_embeddings")
        os.makedirs(full_save_path, exist_ok=True)
        filepath = os.path.join(full_save_path, filename)
        return filepath, full_save_path

    def _get_existing_row_ids(self, filepath, full_save_path):
        existing_row_ids = []
        if os.path.exists(filepath) and self.output_type == "csv":
            existing_df = pd.read_csv(filepath)
            existing_row_ids = existing_df["row_id"].astype(str).values.tolist()
        if self.output_type == "npy":
            for filename in glob(os.path.join(full_save_path, "**", "*.npy"), recursive=True):
                # store relative id strings (same convention as save_embedding() uses)
                existing_row_ids.append(str(Path(filename).with_suffix("").relative_to(full_save_path)))
        return set(existing_row_ids)

    def save_embedding_paths(self, row_data: Iterable[dict], store, filepath, full_save_path, overwrite):
        # row_data is list of dict rows
        if self.output_type == "csv" and len(row_data) > 0:
            save_df(pd.DataFrame(row_data), Path(filepath).name, full_save_path, overwrite=overwrite)
        elif self.output_type == "doc":
            store.to_doc(filepath.removesuffix(".csv"))

    def save_text_embeddings(self, texts: Iterable[str], save_path, overwrite):
        store = EmbeddingStore(EmbeddingType=CLIPTextEmbedding, FeatureGenerator=self.model)
        filepath, full_save_path = self.create_files(type="text", save_path=save_path)
        existing_row_ids = self._get_existing_row_ids(filepath, full_save_path)
        with torch.no_grad():
            row_data = []
            for text in tqdm(texts, desc="Calculating text embeddings", position=tqdm._get_free_pos()):
                if str(text) not in existing_row_ids or overwrite:
                    curr_text_embeddings = self.model.text_embeddings([text], self.normalize_embeddings)[0][0].cpu().numpy()
                    if self.output_type == "doc":
                        store.add_embedding(curr_text_embeddings, url=None, text=text)
                    else:
                        curr_row_data = self.process_embedding_row(curr_text_embeddings, id=text, save_path=full_save_path)
                        row_data.append(curr_row_data)
        self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)

    # NEW: this method accepts index list (not a full DataLoader)
    def save_image_embeddings(self, indices: List[int], save_path=None, overwrite=False, save_every_batch=False, batch_size=32, num_workers=4):
        """
        indices: list of dataset indices to process in this actor
        The actor will create its own DataLoader from StimuliLoader -> dataset -> Subset.
        """
        # Recreate dataset and DataLoader locally
        base_loader = StimuliLoader(
            image_folder=self.input_dir,
            batch_size=batch_size,
            stimuli_type="images",
            transform=self.transform,
        )
        # base_loader.dataloader() returns a DataLoader with a .dataset attribute
        dataset = base_loader.dataloader().dataset
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, collate_fn=base_loader.collator)

        # prepare store & file paths
        store = EmbeddingStore(EmbeddingType=CLIPImageEmbedding, FeatureGenerator=self.model)
        filepath, full_save_path = self.create_files(save_path)
        existing_row_ids = self._get_existing_row_ids(filepath, full_save_path)

        all_text = set()
        row_data = []

        with torch.no_grad():
            for d in tqdm(dataloader, desc=f"Actor processing {len(indices)} images", position=tqdm._get_free_pos()):
                if save_every_batch:
                    row_data = []

                # Filter and build lists of items to compute
                images_to_process = []
                ids_to_process = []
                texts_to_process = []

                # assume d is like {'images': tensor_batch, 'text': [...], 'item_id': [...]}
                batch_images = d.get("images")
                batch_texts = d.get("text", None)
                batch_ids = d.get("item_id")

                for img, txt, item_id in zip(batch_images, batch_texts if batch_texts else itertools.repeat(None, len(batch_images)), batch_ids):
                    item_id_str = str(item_id)
                    if item_id_str not in existing_row_ids or overwrite:
                        images_to_process.append(img)
                        texts_to_process.append(txt)
                        ids_to_process.append(item_id_str)

                if len(images_to_process) == 0:
                    continue
                # get embeddings from model (assumed to return tensor)
                curr_image_embeddings = self.model.image_embeddings(images_to_process, self.normalize_embeddings)
                curr_image_embeddings = curr_image_embeddings.cpu().numpy()

                # collect text set for later text embeddings
                if batch_texts is not None:
                    for t in batch_texts:
                        if t is not None:
                            all_text.add(t)
                print(self.output_type)
                if self.output_type == "doc":
                    store.add_embeddings(curr_image_embeddings, ids_to_process, texts_to_process)
                # Save rows
                else:
                    for emb, item_id, txt in zip(curr_image_embeddings, ids_to_process, texts_to_process):
                        curr_row_data = self.process_embedding_row(embedding=emb, id=item_id, save_path=full_save_path, text=txt)
                        row_data.append(curr_row_data)

                if save_every_batch:
                    self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)

        if not save_every_batch and row_data:
            self.save_embedding_paths(row_data, store, filepath, full_save_path, overwrite)

        # process text embeddings collected across this actor (safe duplicate checks inside)
        if len(all_text) > 0:
            self.save_text_embeddings(all_text, save_path, overwrite)

def generate_image_embeddings(
    input_dir: str = None,
    input_csv: str = None,
    output_path: str = None,
    overwrite: bool = False,
    batch_size: int = 1,
    save_every_batch: bool = False,
    id_column: str = "image1",
    actor_gpu_fraction: float = 0.3,
    transform=None,
    output_type="csv",
    normalize_embeddings=False,
    subdirs=False,
    input_device=None,
):
    """
    Launch parallel embedding generation using Ray actors.
    Supports both directory-based and CSV-based image datasets.
    """

    if input_csv is None and input_dir is None:
        raise ValueError("Either input_dir or input_csv must be provided.")

    # --- Step 1: Create local StimuliLoader to inspect dataset ---
    stimuli_loader_kwargs = dict(
        image_folder=input_dir,
        batch_size=batch_size,
        stimuli_type="images",
        transform=transform,
    )
    if input_csv is not None:
        stimuli_loader_kwargs["dataset_file"] = input_csv
        stimuli_loader_kwargs["id_column"] = id_column

    dataloader = StimuliLoader(**stimuli_loader_kwargs).dataloader()
    dataset = dataloader.dataset
    total_items = len(dataset)

    if total_items == 0:
        print("No images found. Exiting.")
        return

    ray.init(ignore_reinit_error=True)
    try:
        num_gpus = max(1, torch.cuda.device_count())
        actor_count = num_gpus * 2  # heuristic: 2 actors per GPU
        chunk_size = max(1, total_items // actor_count)

        chunks = [
            list(range(i, min(i + chunk_size, total_items)))
            for i in range(0, total_items, chunk_size)
        ]
        print(
            f"Spawning {len(chunks)} Ray actors ({actor_gpu_fraction} GPU each) "
            f"for {total_items} images, chunk size={chunk_size}"
        )

        actors = [
            EmbeddingGenerator.options(num_gpus=actor_gpu_fraction).remote(
                input_dir=input_dir,
                device=input_device,
                model_type="clip",
                model=None,
                output_type=output_type,
                normalize_embeddings=normalize_embeddings,
                transform=transform,
                subdirs=subdirs,
            )
            for _ in chunks
        ]

        # --- Step 4: Dispatch embedding generation tasks ---
        tasks = []
        for actor, chunk in zip(actors, chunks):
            tasks.append(
                actor.save_image_embeddings.remote(
                    indices=chunk,
                    save_path=output_path,
                    overwrite=overwrite,
                    save_every_batch=save_every_batch,
                    batch_size=batch_size,
                    num_workers=4,
                )
            )

        ray.get(tasks)
        print("All embedding generation tasks completed successfully.")

    finally:
        ray.shutdown()

