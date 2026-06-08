import torch
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from vislearnlabpy.models.feature_generator import FeatureGenerator
from vislearnlabpy.embeddings import utils
from vislearnlabpy.models import silicon_menagerie_utils

# Named presets — pass the key to EmbeddingGenerator.from_model()
MODEL_PRESETS = {
    # Vision-language via openai/clip package (no layer support)
    "clip":          {"model_source": "openai_clip",      "model_name": "ViT-B/32",                                 "model_type": "clip"},
    "clip-large":    {"model_source": "openai_clip",      "model_name": "ViT-L/14",                                 "model_type": "clip-large"},
    # Vision-language via HuggingFace CLIP — num_layers = num_hidden_layers + 1 (embedding layer)
    "clip-hf":       {"model_source": "huggingface_clip", "model_name": "openai/clip-vit-base-patch32",             "model_type": "clip",       "num_layers": 13},
    "clip-hf-large": {"model_source": "huggingface_clip", "model_name": "openai/clip-vit-large-patch14",            "model_type": "clip-large", "num_layers": 25},
    # Vision-only HuggingFace models
    "dinov3-base":   {"model_source": "huggingface",      "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", "model_type": "dinov3-vitb16",  "num_layers": 13},
    "dinov3":        {"model_source": "huggingface",      "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "model_type": "dinov3-vitl16",  "num_layers": 25},
    "dinov3-babyview":{"model_source": "huggingface",     "model_name": "awwkl/dinov3-vitl-babyview",               "model_type": "dinov3-bv",      "num_layers": 25},
    "dinov3-small":  {"model_source": "huggingface",      "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m", "model_type": "dinov3-vits16",  "num_layers": 13},
    "dinov2":        {"model_source": "huggingface",      "model_name": "facebook/dinov2-large",                   "model_type": "dinov2-l",       "num_layers": 25},
    "dinov2-base":   {"model_source": "huggingface",      "model_name": "facebook/dinov2-base",                    "model_type": "dinov2-b",       "num_layers": 13},
}

for model_name in silicon_menagerie_utils.get_available_models():
    MODEL_PRESETS[model_name] = {"model_source": "silicon_menagerie", "model_name": model_name, "model_type": model_name}

class HuggingFaceTransform:
    def __init__(self, model_name: str, processor_cls):
        self.model_name = model_name
        self.processor_cls = processor_cls
        self._processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self.processor_cls.from_pretrained(self.model_name)
        return self._processor

    def __call__(self, img):
        return self.processor(images=[img], return_tensors="pt")["pixel_values"].squeeze(0)
        
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
        device:       "cuda:0", "cpu", etc.  None -> auto-detect.
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
        """Returns an (N, D) tensor. Caller must ensure no None images.

        If images are already torch.Tensors (pre-processed by DataLoader workers),
        they are stacked and sent to device directly, skipping the processor.
        This allows preprocessing to be parallelized in DataLoader workers.
        """
        if not isinstance(images, list):
            return self.image_embeddings([images], normalize_embeddings)
        if not images:
            return torch.empty(0)

        # Fast path: tensors already preprocessed in DataLoader workers
        if isinstance(images[0], torch.Tensor):
            pixel_values = torch.stack(images).to(self.device)
        else:
            # Slow path: preprocess on single CPU core here (avoid if possible)
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

    def make_processor_transform(self):
        """Return a callable that preprocesses a single PIL image into a tensor.

        This is intended to be used as a DataLoader transform so that preprocessing
        runs in parallel across worker processes rather than serially on the main thread.
        """
        return HuggingFaceTransform(self.model_name, self.preprocess.__class__)

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

        # Fast path: tensors already preprocessed in DataLoader workers
        if isinstance(images[0], torch.Tensor):
            pixel_values = torch.stack(images).to(self.device)
        else:
            inputs = self.preprocess(images=images)
            pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            embeddings = self.model(pixel_values)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def make_processor_transform(self):
        """Return the underlying torchvision transform for use in DataLoader workers."""
        inner_transform = self.preprocess.transform
        return lambda img: inner_transform(img)


class HuggingFaceVisionGenerator(HuggingFaceGenerator):
    supports_text: bool = False
    """Vision-only generator (DINOv2, DINOv3, …). Extracts the CLS token.

    Args:
        layer: If given, extract the CLS token from this hidden-state index instead
               of the final pooler output. 0 = patch-embedding output, 1 = first
               transformer block, -1 = last transformer block, etc.
    """

    def __init__(self, model_name, layer=None, mean_pool=False, dataloader=None, device=None, token=None):
        super().__init__(model_name, AutoModel, AutoImageProcessor,
                         dataloader=dataloader, device=device, token=token)
        self.layer = layer
        self.mean_pool = mean_pool
        if layer is not None:
            self.name = f"{self.name}_layer{layer}"

    @property
    def embedding_dim(self) -> int:
        if self.layer is not None or self.mean_pool:
            return int(self.model.config.hidden_size)
        return super().embedding_dim

    def _encode_image(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=self.layer is not None)
        if self.layer is not None:
            return outputs.hidden_states[self.layer].mean(dim=1)
        if self.mean_pool:
            return outputs.last_hidden_state.mean(dim=1)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]

class HuggingFaceCLIPGenerator(HuggingFaceGenerator):
    """Vision-language generator for HuggingFace CLIP models."""
    supports_text: bool = True

    def __init__(self, model_name, layer=None, mean_pool=False, text_prompt="a photo of a ",
                 dataloader=None, device=None, token=None):
        super().__init__(model_name, CLIPModel, CLIPProcessor,
                         text_prompt=text_prompt, dataloader=dataloader,
                         device=device, token=token)
        self.layer = layer
        self.mean_pool = mean_pool
        if layer is not None:
            self.name = f"{self.name}_layer{layer}"

    @property
    def embedding_dim(self) -> int:
        if self.layer is not None or self.mean_pool:
            return int(self.model.config.vision_config.hidden_size)
        return super().embedding_dim

    @property
    def text_embedding_dim(self) -> int:
        if self.layer is not None or self.mean_pool:
            return int(self.model.config.text_config.hidden_size)
        return int(self.model.config.projection_dim)

    def _encode_image(self, pixel_values):
        if self.layer is not None or self.mean_pool:
            vision_out = self.model.vision_model(pixel_values=pixel_values, output_hidden_states=self.layer is not None)
            hidden = vision_out.hidden_states[self.layer] if self.layer is not None else vision_out.last_hidden_state
            return hidden.mean(dim=1)
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        return image_features.pooler_output if hasattr(image_features, "pooler_output") else image_features.last_hidden_state[:, 0, :]

    def text_embeddings(self, words, normalize_embeddings=False):
        prompted = [f"{self.text_prompt}{w}" for w in words]
        inputs = self.preprocess(text=prompted, return_tensors="pt",
                                 padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.layer is not None or self.mean_pool:
                text_out = self.model.text_model(**inputs, output_hidden_states=self.layer is not None)
                hidden = text_out.hidden_states[self.layer] if self.layer is not None else text_out.last_hidden_state
                embeddings = hidden.mean(dim=1)
            else:
                embeddings = self.model.get_text_features(**inputs)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings
