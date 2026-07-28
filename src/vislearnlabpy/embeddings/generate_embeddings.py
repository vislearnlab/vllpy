from dataclasses import dataclass, replace as dataclass_replace
import time
from typing import Any, Iterable, Optional
from vislearnlabpy.models.clip_model import CLIPGenerator, OpenCLIPGenerator
from vislearnlabpy.models.hf_model import HuggingFaceVisionGenerator, HuggingFaceCLIPGenerator, MODEL_PRESETS, SiliconMenagerieGenerator
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
from tqdm.auto import tqdm
import logging
logger = logging.getLogger(__name__)

def _ts():
    """Current timestamp string for logs."""
    return time.strftime("%H:%M:%S")


def _elapsed(t0):
    return f"{time.time() - t0:.2f}s"


@dataclass
class EmbeddingConfig:
    """Configuration for EmbeddingGenerator (model and output settings).

    model_source options:
      "openai_clip"   - default, uses the openai/CLIP package (ViT-B/32 etc.)
      "huggingface"   - any HuggingFace vision model via AutoModel
                        (DINOv2, DINOv3, HF CLIP, ...); set model_name to the HF repo id.
      "openclip"      - uses the open_clip package; set model_name to the architecture
                        (e.g. "ViT-B-32") and pretrained to the checkpoint tag, or
                        checkpoint_path to a local/downloaded .pt file (takes precedence).
    """
    model_type: str = "clip"               # human-readable label used in output filenames
    model_source: str = "openai_clip"      # "openai_clip" | "huggingface" | "huggingface_clip" | "silicon_menagerie" | "openclip"
    model_name: str = "ViT-B/32"          # variant for openai_clip, or HF repo id
    hf_token: Optional[str] = None        # HuggingFace token for private/gated repos
    pretrained: Optional[str] = None      # openclip pretrained checkpoint tag (e.g. "laion2b_s34b_b79k")
    checkpoint_path: Optional[str] = None  # openclip: path to a .pt checkpoint file (overrides pretrained)
    epoch: Optional[int] = None           # openclip: training epoch, appended to the generator name
    output_type: str = "csv"              # "csv", "npy", or "doc"
    device: Optional[str] = None          # None -> auto-detect CUDA/CPU
    text_prompt: str = "a photo of a "    # prepended to every text label (CLIP only)
    normalize_embeddings: bool = False
    transform: Optional[Any] = None       # torchvision transform pipeline
    layer: Optional[int] = None           # extract embeddings from this hidden-state index (HF vision models only)
    mean_pool: bool = False               # mean-pool over all tokens instead of using CLS token (HF models only)
    num_actors: Optional[int] = None      # for parallel npy generation (Ray)
    gpu_per_actor: float = 0.4            # for parallel npy generation (Ray)
    save_every_batch: bool = False        # save after every batch instead of all at end
    ray_temp_dir: Optional[str] = None   # override Ray's temp/spill directory (e.g. /Scratch/tmp/ray)

try:
    import ray
    from torch.utils.data import Subset, DataLoader
    from torchvision import transforms as T

    @ray.remote(num_gpus=0.3)
    class _EmbeddingActor:
        """Internal Ray actor for parallel npy embedding generation."""

        def __init__(self, input_dir, input_csv, id_column, config: "EmbeddingConfig", subdirs):
            t_init = time.time()
            self.input_dir = input_dir
            self.input_csv = input_csv
            self.id_column = id_column
            self.config = config
            self.subdirs = subdirs
            self._chunk_count = 0

            # Use Ray's GPU assignment -- within each actor the assigned GPU
            # is always remapped to cuda:0 via CUDA_VISIBLE_DEVICES
            gpu_ids = ray.get_gpu_ids()
            self.device = "cuda:0" if gpu_ids else "cpu"
            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} device={self.device}")

            t_model = time.time()
            if config.model_source == "huggingface":
                logger.debug("HuggingFaceVisionGenerator")
                self.model = HuggingFaceVisionGenerator(
                    model_name=config.model_name, layer=config.layer,
                    mean_pool=config.mean_pool, device=self.device, token=config.hf_token
                )
            elif config.model_source == "huggingface_clip":
                logger.debug("HuggingFaceCLIPGenerator")
                self.model = HuggingFaceCLIPGenerator(
                    model_name=config.model_name, layer=config.layer,
                    mean_pool=config.mean_pool, text_prompt=config.text_prompt,
                    device=self.device, token=config.hf_token
                )
            elif config.model_source == "silicon_menagerie":
                logger.debug("SiliconMenagerieGenerator")
                self.model = SiliconMenagerieGenerator(model_name=config.model_name, device=self.device)
            elif config.model_source == "openclip":
                logger.debug("OpenCLIPGenerator")
                self.model = OpenCLIPGenerator(
                    model_name=config.model_name, pretrained=config.pretrained,
                    checkpoint_path=config.checkpoint_path, epoch=config.epoch,
                    text_prompt=config.text_prompt, device=self.device
                )
            else:
                logger.debug("CLIPGenerator")
                self.model = CLIPGenerator(device=self.device, text_prompt=config.text_prompt)
            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] model loaded in {_elapsed(t_model)}")

            # Build a transform that includes the model's own preprocessor so that
            # image decoding + resizing + normalization all happen in DataLoader workers
            # (parallel CPU) rather than serially inside image_embeddings().
            t_transform = time.time()
            if hasattr(self.model, "make_processor_transform"):
                processor_transform = self.model.make_processor_transform()
                if config.transform is not None:
                    dataset_transform = T.Compose([config.transform, processor_transform])
                else:
                    dataset_transform = processor_transform
                logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] processor transform built in {_elapsed(t_transform)} — preprocessing will run in DataLoader workers")
            else:
                dataset_transform = config.transform
                print(f"[{_ts()}] [actor pid={os.getpid()}] no make_processor_transform — using model's internal preprocessing")

            # Build dataset once -- avoids re-scanning the image directory on every chunk
            t_dataset = time.time()
            loader_kwargs = dict(image_folder=input_dir, batch_size=1,
                                 stimuli_type="images", transform=dataset_transform)
            if input_csv is not None:
                loader_kwargs.update(dataset_file=input_csv, id_column=id_column)
            self.base_loader = StimuliLoader(**loader_kwargs)
            self.dataset = self.base_loader.dataloader().dataset
            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] dataset built ({len(self.dataset)} items) in {_elapsed(t_dataset)}")

            # Cache existing npy ids once -- updated incrementally as embeddings are saved
            self.full_save_path = None
            self.existing_ids = None

            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] __init__ complete in {_elapsed(t_init)}")

        def _init_save_path(self, save_path):
            """Lazily initialize save path and existing id cache."""
            if self.full_save_path is None:
                t0 = time.time()
                self.full_save_path = _image_save_path(save_path, self.config.model_type)
                self.existing_ids = _get_existing_npy_ids(self.full_save_path)
                logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] save path init: {len(self.existing_ids)} existing ids found in {_elapsed(t0)}")

        def _save_embedding(self, embedding, curr_id, text=None):
            path = _save_embedding(embedding, curr_id, self.full_save_path, text=text, subdirs=self.subdirs)
            rel = str(Path(path).with_suffix("").relative_to(self.full_save_path))
            self.existing_ids.add(rel)
            return path

        def process_chunk(self, indices, save_path, overwrite, batch_size, num_workers=4):
            self._chunk_count += 1
            chunk_id = self._chunk_count
            t_chunk = time.time()
            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] chunk={chunk_id} start: {len(indices)} images, batch_size={batch_size}, num_workers={num_workers}")

            self._init_save_path(save_path)

            t_loader = time.time()
            subset = Subset(self.dataset, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                                    collate_fn=self.base_loader.collator)
            logger.debug(f"[{_ts()}] [actor pid={os.getpid()}] chunk={chunk_id} dataloader built in {_elapsed(t_loader)}")

            n_batches = 0
            n_images = 0
            n_skipped = 0
            t_gpu_total = 0.0
            t_io_total = 0.0
            t_load_total = 0.0

            with torch.no_grad():
                for d in dataloader:
                    t_batch = time.time()
                    batch_images = d.get("images")
                    batch_texts = d.get("text", None)
                    batch_ids = d.get("item_id")
                    t_load_total += time.time() - t_batch

                    to_process = [
                        (img, txt, str(iid))
                        for img, txt, iid in zip(
                            batch_images,
                            batch_texts if batch_texts else itertools.repeat(None, len(batch_images)),
                            batch_ids,
                        )
                        if img is not None and (str(iid) not in self.existing_ids or overwrite)
                    ]
                    n_skipped += len(batch_images) - len(to_process)

                    if not to_process:
                        continue

                    imgs, txts, ids = zip(*to_process)
                    t_gpu = time.time()
                    embeddings = self.model.image_embeddings(list(imgs), self.config.normalize_embeddings).cpu().numpy()
                    t_gpu_total += time.time() - t_gpu

                    t_io = time.time()
                    for emb, iid, txt in zip(embeddings, ids, txts):
                        self._save_embedding(emb, iid, text=txt)
                    t_io_total += time.time() - t_io

                    n_batches += 1
                    n_images += len(imgs)

            t_total = time.time() - t_chunk
            throughput = n_images / t_total if t_total > 0 else 0
            logger.debug(
                f"[{_ts()}] [actor pid={os.getpid()}] chunk={chunk_id} done: "
                f"{n_images} images in {_elapsed(t_chunk)} "
                f"({throughput:.1f} img/s) | "
                f"dataload={t_load_total:.2f}s  gpu={t_gpu_total:.2f}s  io={t_io_total:.2f}s  "
                f"skipped={n_skipped}  batches={n_batches}"
            )

            return len(indices)

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
    full = os.path.join(base, f"{model_type}_image_embeddings")
    os.makedirs(full, exist_ok=True)
    return str(Path(full).resolve())


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
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self.config.model_type
        self.output_type = self.config.output_type
        self.transform = self.config.transform
        self.normalize_embeddings = self.config.normalize_embeddings
        self.text_prompt = self.config.text_prompt
        # todo: additional config values are not being passed as direct instance variables
        self.model = model or self._build_model()

    @classmethod
    def from_model(cls, name: str, **config_kwargs) -> "EmbeddingGenerator":
        """Instantiate from a named preset, e.g. EmbeddingGenerator.from_model('clip-hf', layer=5).

        Preset values are used as defaults; any keyword argument overrides them.
        When ``layer`` is specified and ``model_type`` is not explicitly overridden,
        ``model_type`` is auto-suffixed with ``_layer{N}`` so embedding stores for
        different layers don't share the same output directory.

        Presets may also include a ``layer`` key directly, e.g.:
            MODEL_PRESETS["clip-hf-layer5"] = {**MODEL_PRESETS["clip-hf"], "layer": 5}

        Available presets: """ + ", ".join(f'"{k}"' for k in MODEL_PRESETS) + """
        """
        if name not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset '{name}'. Available: {list(MODEL_PRESETS)}")
        merged = {**MODEL_PRESETS[name], **config_kwargs}
        if merged.get("layer") is not None and "model_type" not in config_kwargs:
            merged["model_type"] = f"{merged['model_type']}_layer{merged['layer']}"
        # Strip preset-only metadata keys that are not EmbeddingConfig fields
        cfg_fields = {f for f in EmbeddingConfig.__dataclass_fields__}
        cfg = EmbeddingConfig(**{k: v for k, v in merged.items() if k in cfg_fields})
        return cls(config=cfg)

    def _build_model(self, dataloader=None):
        if self.config.model_source == "huggingface":
            return HuggingFaceVisionGenerator(
                model_name=self.config.model_name,
                layer=self.config.layer,
                mean_pool=self.config.mean_pool,
                dataloader=dataloader,
                device=self.device,
                token=self.config.hf_token,
            )
        elif self.config.model_source == "huggingface_clip":
            return HuggingFaceCLIPGenerator(
                model_name=self.config.model_name,
                layer=self.config.layer,
                mean_pool=self.config.mean_pool,
                text_prompt=self.config.text_prompt,
                dataloader=dataloader,
                device=self.device,
                token=self.config.hf_token,
            )
        elif self.config.model_source == "silicon_menagerie":
            return SiliconMenagerieGenerator(model_name=self.config.model_name, dataloader=dataloader, device=self.device)
        elif self.config.model_source == "openclip":
            return OpenCLIPGenerator(
                model_name=self.config.model_name,
                pretrained=self.config.pretrained,
                checkpoint_path=self.config.checkpoint_path,
                epoch=self.config.epoch,
                text_prompt=self.config.text_prompt,
                dataloader=dataloader,
                device=self.device,
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
        text_dim = getattr(self.model, "text_embedding_dim", self.model.embedding_dim)
        store = EmbeddingStore(FeatureGenerator=self.model, dim=text_dim)
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

        parallel: None (auto -- uses Ray for npy, sequential for csv/doc),
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

        t_start = time.time()
        logger.debug(f"[{_ts()}] [main] _generate_parallel start")

        loader_kwargs = dict(image_folder=input_dir, batch_size=batch_size,
                            stimuli_type="images", transform=self.transform)
        if input_csv is not None:
            loader_kwargs.update(dataset_file=input_csv, id_column=id_column)

        t_scan = time.time()
        dataset = StimuliLoader(**loader_kwargs).dataloader().dataset
        total = len(dataset)
        logger.debug(f"[{_ts()}] [main] dataset scan complete: {total} images in {_elapsed(t_scan)}")

        if total == 0:
            print("No images found.")
            return

        ray_kwargs = {"ignore_reinit_error": True}
        if self.config.ray_temp_dir:
            ray_kwargs["_temp_dir"] = self.config.ray_temp_dir

        t_ray = time.time()
        ray.init(**ray_kwargs)
        logger.debug(f"[{_ts()}] [main] ray.init complete in {_elapsed(t_ray)}")
        logger.debug(f"[{_ts()}] [main] ray resources: {ray.available_resources()}")

        actors = []
        try:
            num_gpus = max(0, torch.cuda.device_count())
            logger.debug(f"[{_ts()}] [main] torch.cuda.device_count()={num_gpus}")

            if num_gpus == 0:
                print(f"[{_ts()}] [main] no GPUs detected, falling back to CPU with 2 actors")
                num_gpus = 1
                self.config = dataclass_replace(self.config, gpu_per_actor=0)

            if self.config.num_actors == 0:
                print(f"[{_ts()}] [main] num_actors=0, running sequentially")
                self._generate_sequential(output_path=output_path, subdirs=subdirs, overwrite=overwrite)
                return

            max_actors_per_gpu = int(1 / self.config.gpu_per_actor + 0.1) if self.config.gpu_per_actor > 0 else 2
            actor_count = min(self.config.num_actors or num_gpus * 2, num_gpus * max_actors_per_gpu)
            chunk_size = max(batch_size * 4, total // (actor_count * 10))
            chunks = [list(range(i, min(i + chunk_size, total))) for i in range(0, total, chunk_size)]

            print(f"[{_ts()}] [main] gpu_per_actor={self.config.gpu_per_actor}  max_actors_per_gpu={max_actors_per_gpu}  actor_count={actor_count}")
            print(f"[{_ts()}] [main] chunk_size={chunk_size}  num_chunks={len(chunks)}  batch_size={batch_size}")
            print(f"[{_ts()}] [main] spawning {actor_count} actors...")

            actor_config = dataclass_replace(self.config, transform=None)
            t_spawn = time.time()
            actors = [
                _EmbeddingActor.options(num_gpus=self.config.gpu_per_actor).remote(
                    input_dir=input_dir,
                    input_csv=input_csv,
                    id_column=id_column,
                    config=actor_config,
                    subdirs=subdirs,
                )
                for _ in range(actor_count)
            ]

            # Seed each actor with its first chunk
            futures = {}
            next_chunk = 0
            for actor in actors:
                if next_chunk < len(chunks):
                    f = actor.process_chunk.remote(
                        indices=chunks[next_chunk], save_path=output_path,
                        overwrite=overwrite, batch_size=batch_size,
                    )
                    futures[f] = actor
                    next_chunk += 1
            logger.debug(f"[{_ts()}] [main] {len(futures)} initial chunks dispatched in {_elapsed(t_spawn)}")

            t_loop = time.time()
            chunks_done = 0
            with tqdm(total=total, desc="Generating embeddings", unit="img") as pbar:
                while futures:
                    t_wait = time.time()
                    done, _ = ray.wait(list(futures.keys()), num_returns=1)
                    f = done[0]
                    actor = futures.pop(f)
                    n_done = ray.get(f)  # surfaces exceptions
                    chunks_done += 1
                    elapsed_wait = time.time() - t_wait
                    pbar.update(n_done)

                    overall_throughput = pbar.n / (time.time() - t_loop) if (time.time() - t_loop) > 0 else 0
                    logger.debug(
                        f"[{_ts()}] [main] chunk {chunks_done}/{len(chunks)} returned "
                        f"({n_done} imgs, wait={elapsed_wait:.2f}s) | "
                        f"overall={overall_throughput:.1f} img/s  "
                        f"pending={len(futures)}  remaining_chunks={len(chunks)-next_chunk}"
                    )

                    if next_chunk < len(chunks):
                        new_f = actor.process_chunk.remote(
                            indices=chunks[next_chunk], save_path=output_path,
                            overwrite=overwrite, batch_size=batch_size,
                        )
                        futures[new_f] = actor
                        next_chunk += 1

            print(f"[{_ts()}] [main] all chunks complete in {_elapsed(t_loop)} | total wall time {_elapsed(t_start)}")

        finally:
            print(f"[{_ts()}] [main] shutting down actors...")
            for actor in actors:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
            print(f"[{_ts()}] [main] ray shutdown")

        # Text embeddings are fast -- generate sequentially on the main process
        if input_csv is not None:
            df = pd.read_csv(input_csv)
            text_cols = [c for c in df.columns if c.startswith("text")]
            all_text = {t for t in df[text_cols].values.flatten() if isinstance(t, str) and t != ""}
            if all_text and self.model.supports_text:
                self.generate_text_embeddings(all_text, output_path=output_path, overwrite=overwrite)
