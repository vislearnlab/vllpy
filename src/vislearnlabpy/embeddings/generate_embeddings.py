from dataclasses import dataclass, replace as dataclass_replace
import math
from typing import Any, Iterable, Optional
from vislearnlabpy.models.clip_model import CLIPGenerator
from vislearnlabpy.models.hf_model import HuggingFaceVisionGenerator, HuggingFaceCLIPGenerator, MODEL_PRESETS
from vislearnlabpy.embeddings.stimuli_loader import StimuliLoader
from vislearnlabpy.embeddings.utils import save_df, indexed_embeddings, is_url
from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
import torch
import os
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm


@dataclass
class EmbeddingConfig:
    """Configuration for EmbeddingGenerator (model and output settings).

    model_source options:
      "openai_clip"   – default, uses the openai/CLIP package (ViT-B/32 etc.)
      "huggingface"   – any HuggingFace vision model via AutoModel
                        (DINOv2, DINOv3, HF CLIP, …); set model_name to the HF repo id.
    """
    model_type: str = "clip"               # human-readable label used in output filenames
    model_source: str = "openai_clip"      # "openai_clip" | "huggingface"
    model_name: str = "ViT-B/32"          # variant for openai_clip, or HF repo id
    hf_token: Optional[str] = None        # HuggingFace token for private/gated repos
    output_type: str = "csv"              # "csv", "npy", or "doc"
    device: Optional[str] = None          # None → auto-detect CUDA/CPU
    text_prompt: str = "a photo of a "    # prepended to every text label (CLIP only)
    normalize_embeddings: bool = False
    transform: Optional[Any] = None       # torchvision transform pipeline
    num_actors: Optional[int] = None      # for parallel npy generation (Ray)
    gpu_per_actor: float = 0.3            # for parallel npy generation (Ray)
    save_every_batch: bool = False        # save after every batch instead of all at end

try:
    import ray
    from torch.utils.data import Subset, DataLoader

    @ray.remote(num_gpus=0.3)
    class _EmbeddingActor:
        """Internal Ray actor for parallel npy embedding generation."""

        def __init__(self, input_dir, input_csv, id_column, config: "EmbeddingConfig", subdirs):
            self.input_dir = input_dir
            self.input_csv = input_csv
            self.id_column = id_column
            self.config = config
            self.device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            if config.model_source == "huggingface":
                self.model = HuggingFaceVisionGenerator(
                    model_name=config.model_name, device=self.device, token=config.hf_token
                )
            elif config.model_source == "huggingface_clip":
                self.model = HuggingFaceCLIPGenerator(
                    model_name=config.model_name, text_prompt=config.text_prompt,
                    device=self.device, token=config.hf_token
                )
            else:
                self.model = CLIPGenerator(device=self.device, text_prompt=config.text_prompt)
            self.subdirs = subdirs

        def _save_embedding(self, embedding, curr_id, save_path, text=None):
            return _save_embedding(embedding, curr_id, save_path, text=text, subdirs=self.subdirs)

        def process_chunk(self, indices, save_path, overwrite, batch_size, num_workers=4):
            loader_kwargs = dict(image_folder=self.input_dir, batch_size=batch_size,
                                 stimuli_type="images", transform=self.config.transform)
            if self.input_csv is not None:
                loader_kwargs.update(dataset_file=self.input_csv, id_column=self.id_column)
            base_loader = StimuliLoader(**loader_kwargs)
            dataset = base_loader.dataloader().dataset
            subset = Subset(dataset, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                                    collate_fn=base_loader.collator)

            full_save_path = _image_save_path(save_path, self.config.model_type)
            existing_ids = _get_existing_npy_ids(full_save_path)

            with torch.no_grad():
                for d in dataloader:
                    batch_images = d.get("images")
                    batch_texts = d.get("text", None)
                    batch_ids = d.get("item_id")

                    to_process = [
                        (img, txt, str(iid))
                        for img, txt, iid in zip(
                            batch_images,
                            batch_texts if batch_texts else itertools.repeat(None, len(batch_images)),
                            batch_ids,
                        )
                        if img is not None and (str(iid) not in existing_ids or overwrite)
                    ]
                    if not to_process:
                        continue

                    imgs, txts, ids = zip(*to_process)
                    embeddings = self.model.image_embeddings(list(imgs), self.config.normalize_embeddings).cpu().numpy()
                    for emb, iid, txt in zip(embeddings, ids, txts):
                        self._save_embedding(emb, iid, full_save_path, text=txt)

except ImportError:
    _EmbeddingActor = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_embedding(embedding, curr_id, save_path, text=None, subdirs=False):
    if is_url(curr_id):
        sub = Path(curr_id.split("?")[0]).name
    elif os.path.exists(curr_id):
        sub = str(Path(curr_id).name)
        if subdirs:
            sub = str(Path(curr_id).parent.name) + "/" + str(Path(curr_id).name)
        elif text is not None:
            sub = f"{text}/{sub}"
    else:
        sub = curr_id
    out_path = Path(f"{save_path}/{sub}").with_suffix(".npy")
    if not str(save_path).startswith("/"):
        out_path = Path(f"{os.getcwd()}/{out_path}")
    os.makedirs(out_path.parent, exist_ok=True)
    np.save(str(out_path), embedding)
    return str(out_path)


def _image_save_path(save_path, model_type):
    base = os.path.join(os.getcwd(), "output") if save_path is None else str(save_path)
    full = os.path.join(base, "image_embeddings")
    os.makedirs(full, exist_ok=True)
    return full


def _get_existing_npy_ids(full_save_path):
    ids = set()
    for f in glob(os.path.join(full_save_path, "**", "*.npy"), recursive=True):
        ids.add(str(Path(f).with_suffix("").relative_to(full_save_path)))
    return ids


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    def __init__(self, config: Optional[EmbeddingConfig] = None, model=None):
        self.config = config or EmbeddingConfig()
        self.device = self.config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = self.config.model_type
        self.output_type = self.config.output_type
        self.transform = self.config.transform
        self.normalize_embeddings = self.config.normalize_embeddings
        self.text_prompt = self.config.text_prompt
        # todo: additional config values are not being passed as direct instance variables
        self.model = model or self._build_model()

    @classmethod
    def from_model(cls, name: str, **config_kwargs) -> "EmbeddingGenerator":
        """Instantiate from a named preset, e.g. EmbeddingGenerator.from_model('dinov3-babyview').

        Any extra keyword arguments override the preset's EmbeddingConfig fields.
        Available presets: """ + ", ".join(f'"{k}"' for k in MODEL_PRESETS) + """
        """
        if name not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset '{name}'. Available: {list(MODEL_PRESETS)}")
        cfg = EmbeddingConfig(**{**MODEL_PRESETS[name], **config_kwargs})
        return cls(config=cfg)

    def _build_model(self, dataloader=None):
        if self.config.model_source == "huggingface":
            return HuggingFaceVisionGenerator(
                model_name=self.config.model_name,
                dataloader=dataloader,
                device=self.device,
                token=self.config.hf_token,
            )
        if self.config.model_source == "huggingface_clip":
            return HuggingFaceCLIPGenerator(
                model_name=self.config.model_name,
                text_prompt=self.config.text_prompt,
                dataloader=dataloader,
                device=self.device,
                token=self.config.hf_token,
            )
        # default: openai_clip
        return CLIPGenerator(
            device=self.device,
            text_prompt=self.config.text_prompt,
            dataloader=dataloader,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_files(self, save_path=None, type="image"):
        filename = f"{self.model_type}_{type}_embeddings_{self.output_type}.csv"
        base = os.path.join(os.getcwd(), "output") if save_path is None else str(save_path)
        full_save_path = os.path.join(base, f"{type}_embeddings")
        os.makedirs(full_save_path, exist_ok=True)
        filepath = os.path.join(full_save_path, filename)
        return filepath, full_save_path

    def _get_existing_row_ids(self, filepath, full_save_path):
        existing = []
        if self.output_type == "csv" and os.path.exists(filepath):
            existing = pd.read_csv(filepath)["row_id"].astype(str).tolist()
        elif self.output_type == "npy":
            existing = list(_get_existing_npy_ids(full_save_path))
        return set(existing)

    def _save_embedding(self, embedding, curr_id, save_path, text=None, subdirs=False):
        return _save_embedding(embedding, curr_id, save_path, text=text, subdirs=subdirs)

    def _process_row(self, embedding, id, save_path=None, text=None, subdirs=False):
        row = {"row_id": id}
        if text is not None:
            row["text"] = text
        if self.output_type == "csv":
            row = row | indexed_embeddings(embedding)
        elif self.output_type == "npy":
            row["embedding_path"] = self._save_embedding(embedding, id, save_path,
                                                          text=text, subdirs=subdirs)
        return row

    def _flush(self, row_data, store, filepath, full_save_path, overwrite):
        if self.output_type != "doc" and row_data:
            save_df(pd.DataFrame(row_data), Path(filepath).name, full_save_path, overwrite=overwrite)
        elif self.output_type == "doc":
            store.to_doc(filepath.removesuffix(".csv"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text_embeddings(self, texts: Iterable[str], output_path=None, overwrite=False):
        """Generate and save text embeddings for an explicit list of texts."""
        store = EmbeddingStore(FeatureGenerator=self.model)
        filepath, full_save_path = self._create_files(type="text", save_path=output_path)
        existing = self._get_existing_row_ids(filepath, full_save_path)
        row_data = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating text embeddings"):
                if str(text) not in existing or overwrite:
                    emb = self.model.text_embeddings([text], self.normalize_embeddings)[0].cpu().numpy()
                    if self.output_type == "doc":
                        store.add_embedding(emb, url=None, text=text)
                    else:
                        row_data.append(self._process_row(emb, id=text, save_path=full_save_path))
        self._flush(row_data, store, filepath, full_save_path, overwrite)

    def generate_image_embeddings(self, output_path=None, overwrite=False,
                                  input_csv=None, input_dir=None, batch_size=1,
                                  id_column="image1",
                                  parallel=None, subdirs=False):
        """Generate and save image embeddings.

        parallel: None (auto — uses Ray for npy, sequential for csv/doc),
                  True (force parallel, requires output_type='npy' and ray installed),
                  False (force sequential).
        """
        if input_csv is None and input_dir is None:
            raise ValueError("Either input_csv or input_dir must be provided.")

        use_parallel = parallel if parallel is not None else (self.output_type == "npy")

        if use_parallel and self.output_type != "npy":
            raise ValueError(
                "Parallel mode only works with output_type='npy' (each image is its own file). "
                "Use output_type='npy' or set parallel=False."
            )

        if use_parallel:
            self._generate_parallel(input_dir=input_dir, input_csv=input_csv,
                                    output_path=output_path, overwrite=overwrite,
                                    batch_size=batch_size, id_column=id_column, subdirs=subdirs)
        else:
            loader_kwargs = dict(image_folder=input_dir, batch_size=batch_size,
                                 stimuli_type="images", transform=self.transform)
            if input_csv is not None:
                loader_kwargs.update(dataset_file=input_csv, id_column=id_column)
            dataloader = StimuliLoader(**loader_kwargs).dataloader()
            self.model = self._build_model(dataloader=dataloader)
            self._generate_sequential(output_path=output_path, overwrite=overwrite, subdirs=subdirs)

    # ------------------------------------------------------------------
    # Sequential implementation
    # ------------------------------------------------------------------

    def _generate_sequential(self, output_path=None, overwrite=False, subdirs=False):
        store = EmbeddingStore(FeatureGenerator=self.model)
        filepath, full_save_path = self._create_files(output_path)
        existing = self._get_existing_row_ids(filepath, full_save_path)
        all_text = set()
        row_data = []
        with torch.no_grad():
            for d in tqdm(self.model.dataloader, desc=f"Calculating {self.model_type} embeddings"):
                if self.config.save_every_batch:
                    row_data = []
                if d["text"]:
                    all_text.update(t for t in d["text"] if t is not None)
                texts_iter = d["text"] if d["text"] else itertools.repeat(None, len(d["images"]))
                valid = [(img, iid, txt) for img, iid, txt in zip(d["images"], d["item_id"], texts_iter)
                         if img is not None]
                if not valid:
                    continue
                valid_imgs, valid_ids, valid_texts = zip(*valid)
                embeddings = self.model.image_embeddings(list(valid_imgs), self.normalize_embeddings).cpu().numpy()
                if self.output_type == "doc":
                    store.add_embeddings(embeddings, valid_ids, valid_texts)
                else:
                    for embedding, curr_id, text in zip(embeddings, valid_ids, valid_texts):
                        if str(curr_id) not in existing or overwrite:
                            row_data.append(self._process_row(
                                embedding=embedding, id=curr_id,
                                save_path=full_save_path, text=text, subdirs=subdirs
                            ))
                if self.config.save_every_batch:
                    self._flush(row_data, store, filepath, full_save_path, overwrite)
        if not self.config.save_every_batch:
            self._flush(row_data, store, filepath, full_save_path, overwrite)
        if all_text and self.model.supports_text:
            self.generate_text_embeddings(all_text, output_path=output_path, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Parallel implementation (npy only)
    # ------------------------------------------------------------------

    def _generate_parallel(self, input_dir, input_csv, output_path, overwrite,
                            batch_size, id_column, subdirs):
        if _EmbeddingActor is None:
            raise ImportError(
                "Ray is required for parallel embedding generation. "
                "Install with: pip install ray[default]"
            )
        loader_kwargs = dict(image_folder=input_dir, batch_size=batch_size,
                             stimuli_type="images", transform=self.transform)
        if input_csv is not None:
            loader_kwargs.update(dataset_file=input_csv, id_column=id_column)
        dataset = StimuliLoader(**loader_kwargs).dataloader().dataset
        total = len(dataset)
        if total == 0:
            print("No images found.")
            return

        ray.init(ignore_reinit_error=True)
        try:
            num_gpus = max(0, torch.cuda.device_count())
            if num_gpus == 0:
                print("No GPUs detected. Running with CPU only with 2 actors by default.")
                num_gpus = 1 
                self.config.gpu_per_actor = 0
            if self.config.num_actors == 0:
                print("num_actors set to 0, running sequentially on main process.")
                self._generate_sequential(output_path=output_path, subdirs=subdirs, overwrite=overwrite)
            max_actors_per_gpu= int(1 / self.config.gpu_per_actor + 0.1) if self.config.gpu_per_actor > 0 else 2
            actor_count = min(self.config.num_actors or num_gpus * 2, num_gpus * max_actors_per_gpu) 
            chunk_size = math.ceil(total / actor_count)
            chunks = [list(range(i, min(i + chunk_size, total))) for i in range(0, total, chunk_size)]
            print(f"Spawning {len(chunks)} Ray actors for {total} images (chunk size={chunk_size})")

            actor_config = dataclass_replace(self.config, transform=None)
            actors = [
                _EmbeddingActor.options(num_gpus=self.config.gpu_per_actor).remote(
                    input_dir=input_dir,
                    input_csv=input_csv,
                    id_column=id_column,
                    config=actor_config,
                    subdirs=subdirs,
                )
                for _ in chunks
            ]
            ray.get([
                actor.process_chunk.remote(
                    indices=chunk, save_path=output_path,
                    overwrite=overwrite, batch_size=batch_size,
                )
                for actor, chunk in zip(actors, chunks)
            ])
            print("All embedding generation tasks completed.")
        finally:
            ray.shutdown()

        # Text embeddings are fast — generate sequentially on the main process
        if input_csv is not None:
            df = pd.read_csv(input_csv)
            text_cols = [c for c in df.columns if c.startswith("text")]
            all_text = set(df[text_cols].values.flatten().tolist()) - {None, float("nan"), ""}
            if all_text and self.model.supports_text:
                self.generate_text_embeddings(all_text, output_path=output_path, overwrite=overwrite)
