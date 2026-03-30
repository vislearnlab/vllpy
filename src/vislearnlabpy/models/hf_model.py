import torch
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from vislearnlabpy.models.feature_generator import FeatureGenerator
from vislearnlabpy.embeddings import utils

# Named presets — pass the key to EmbeddingGenerator.from_model()
MODEL_PRESETS = {
    # Vision-language via openai/clip package
    "clip":              {"model_source": "openai_clip",     "model_name": "ViT-B/32",                                "model_type": "clip"},
    "clip-large":        {"model_source": "openai_clip",     "model_name": "ViT-L/14",                                "model_type": "clip-large"},
    # Vision-language via HuggingFace CLIP
    "clip-hf":           {"model_source": "huggingface_clip","model_name": "openai/clip-vit-base-patch32",             "model_type": "clip"},
    "clip-hf-large":     {"model_source": "huggingface_clip","model_name": "openai/clip-vit-large-patch14",            "model_type": "clip-large"},
    # Vision-only HuggingFace models
    "dinov3-big":            {"model_source": "huggingface",     "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", "model_type": "dinov3"},
    "dinov3":      {"model_source": "huggingface",     "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "model_type": "dinov3-vitl16"},
    "dinov3-babyview":   {"model_source": "huggingface",     "model_name": "awwkl/dinov3-vitl-babyview",              "model_type": "dinov3-bv"},
    "dinov3-laion":      {"model_source": "huggingface",     "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m", "model_type": "dinov3-laion"},
}


class HuggingFaceGenerator(FeatureGenerator):
    """Base class for all HuggingFace-backed generators.

    Subclasses must implement ``_encode_image(pixel_values)`` and, if the model
    supports text, override ``text_embeddings`` and set ``supports_text = True``.

    Args:
        model_name:   HuggingFace repo ID.
        model_cls:    Model class (e.g. AutoModel, CLIPModel).
        processor_cls: Processor class (e.g. AutoImageProcessor, CLIPProcessor).
        text_prompt:  Prefix prepended to text labels when generating text embeddings.
        dataloader:   Optional StimuliLoader dataloader.
        device:       "cuda:0", "cpu", etc.  None → auto-detect.
        token:        HuggingFace access token for private/gated repos.
    """
    supports_text: bool = False

    def __init__(self, model_name, model_cls, processor_cls,
                 text_prompt="a photo of a ", dataloader=None, device=None, token=None):
        model = model_cls.from_pretrained(model_name, token=token)
        processor = processor_cls.from_pretrained(model_name, token=token)
        super().__init__(model, processor, dataloader, device)
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self.text_prompt = text_prompt

    @property
    def embedding_dim(self) -> int:
        cfg = self.model.config
        # CLIP-like models expose projection_dim; vision-only transformers expose hidden_size
        for attr in ("projection_dim", "hidden_size"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        return 512

    def image_embeddings(self, images, normalize_embeddings=False):
        """Returns an (N, D) tensor. Caller must ensure no None images."""
        if not isinstance(images, list):
            return self.image_embeddings([images], normalize_embeddings)
        if not images:
            return torch.empty(0)
        inputs = self.preprocess(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            embeddings = self._encode_image(pixel_values)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def _encode_image(self, _pixel_values):
        """Extract embeddings from a pre-processed pixel_values tensor. Override in subclass."""
        raise NotImplementedError

    def text_embeddings(self, words, normalize_embeddings=False):
        raise NotImplementedError(
            f"{self.model_name} is vision-only and does not support text embeddings."
        )

    def similarities(self, *_):
        raise NotImplementedError("Use image_embeddings / text_embeddings directly.")


class HuggingFaceVisionGenerator(HuggingFaceGenerator):
    supports_text: bool = False
    """Vision-only generator (DINOv2, DINOv3, …). Extracts the CLS token."""

    def __init__(self, model_name, dataloader=None, device=None, token=None):
        super().__init__(model_name, AutoModel, AutoImageProcessor,
                         dataloader=dataloader, device=device, token=token)

    def _encode_image(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # CLS token


class HuggingFaceCLIPGenerator(HuggingFaceGenerator):
    """Vision-language generator for HuggingFace CLIP models."""
    supports_text: bool = True

    def __init__(self, model_name, text_prompt="a photo of a ",
                 dataloader=None, device=None, token=None):
        super().__init__(model_name, CLIPModel, CLIPProcessor,
                         text_prompt=text_prompt, dataloader=dataloader,
                         device=device, token=token)

    def _encode_image(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)

    def text_embeddings(self, words, normalize_embeddings=False):
        prompted = [f"{self.text_prompt}{w}" for w in words]
        inputs = self.preprocess(text=prompted, return_tensors="pt",
                                 padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings
