import torch
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from vislearnlabpy.models.feature_generator import FeatureGenerator
from vislearnlabpy.embeddings import utils
from vislearnlabpy.models import silicon_menagerie_utils

# Named presets — pass the key to EmbeddingGenerator.from_model()
MODEL_PRESETS = {
    # Vision-language via openai/clip package
    "clip":              {"model_source": "openai_clip",     "model_name": "ViT-B/32",                                "model_type": "clip"},
    "clip-large":        {"model_source": "openai_clip",     "model_name": "ViT-L/14",                                "model_type": "clip-large"},
    # Vision-language via HuggingFace CLIP
    "clip-hf":           {"model_source": "huggingface_clip","model_name": "openai/clip-vit-base-patch32",             "model_type": "clip"},
    "clip-hf-large":     {"model_source": "huggingface_clip","model_name": "openai/clip-vit-large-patch14",            "model_type": "clip-large"},
    # Vision-only HuggingFace models
    "dinov3-base":            {"model_source": "huggingface",     "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", "model_type": "dinov3-vitb16"},
    "dinov3":      {"model_source": "huggingface",     "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "model_type": "dinov3-vitl16"},
    "dinov3-babyview":   {"model_source": "huggingface",     "model_name": "awwkl/dinov3-vitl-babyview",              "model_type": "dinov3-bv"},
    "dinov3-small":      {"model_source": "huggingface",     "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m", "model_type": "dinov3-vits16"},
    "dinov2": {"model_source": "huggingface", "model_name": "facebook/dinov2-large", "model_type": "dinov2-l"},
    "dinov2-base": {"model_source": "huggingface",   "model_name": "facebook/dinov2-base", "model_type": "dinov2-b"}
}

for model_name in silicon_menagerie_utils.get_available_models():
    MODEL_PRESETS[model_name] = {"model_source": "silicon_menagerie", "model_name": model_name, "model_type": model_name}

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
    
class SiliconMenagerieGenerator(FeatureGenerator):
    supports_text: bool = False
    class _TorchvisionProcessor:
        """Thin wrapper so torchvision transforms plug into FeatureGenerator's preprocess interface."""
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, images=None, return_tensors=None, **kwargs):
            # images can be a single PIL image or a list
            if not isinstance(images, list):
                images = [images]
            tensors = torch.stack([self.transform(img) for img in images])
            return {"pixel_values": tensors}

    def __init__(self, model_name, image_size=224, dataloader=None, device=None):
        from torchvision import transforms as pth_transforms

        model = silicon_menagerie_utils.load_model(model_name)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        processor = self._TorchvisionProcessor(transform)
        super().__init__(model, processor, dataloader, device, name=model_name)
    
    @property
    def embedding_dim(self) -> int:
        return self.model.embed_dim

    def image_embeddings(self, images, normalize_embeddings=False):
        if not isinstance(images, list):
            images = [images]
        inputs = self.preprocess(images=images)
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            embeddings = self.model(pixel_values)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def similarities(self, stimulus1, stimulus2, dataloader_row):
        raise NotImplementedError("Silicon Menagerie models are vision-only; use image_embeddings directly.")


class HuggingFaceVisionGenerator(HuggingFaceGenerator):
    supports_text: bool = False
    """Vision-only generator (DINOv2, DINOv3, …). Extracts the CLS token."""

    def __init__(self, model_name, dataloader=None, device=None, token=None):
        super().__init__(model_name, AutoModel, AutoImageProcessor,
                         dataloader=dataloader, device=device, token=token)

    def _encode_image(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]

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
