from vislearnlabpy.models.feature_generator import FeatureGenerator
from vislearnlabpy.embeddings import utils
from torchvision import transforms
import torch

class MultimodalModel(FeatureGenerator):
    """Abstract base class for multimodal models like CLIP and CVCL that extends FeatureGenerator"""
    supports_text: bool = True
    
    @property
    def embedding_dim(self) -> int:
        # openai/clip models expose output_dim on the visual tower
        return int(self.model.visual.output_dim)

    def __init__(self, model, preprocess, dataloader=None, device=None):
        super().__init__(model, preprocess, dataloader, device)
        self.image_word_alignment = lambda **x: self.model(**x).logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # Load and preprocess images
    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):  
            transform = transforms.ToPILImage()
            image = transform(image)
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def preprocess_text(self, text):
        return self.model.tokenize(text).to(self.device)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def image_embeddings(self, images, normalize_embeddings=False):
        """Get image embeddings. Returns (N, D) tensor. Caller must ensure no None images."""
        if not isinstance(images, list):
            return self.image_embeddings([images], normalize_embeddings)
        if not images:
            return torch.empty(0)

        # Fast path: already tensors from DataLoader workers
        if isinstance(images[0], torch.Tensor):
            preprocessed = [img.squeeze(0) if img.dim() == 4 else img for img in images]
        else:
            preprocessed = [self.preprocess_image(img).squeeze(0) for img in images]

        image_batch = torch.stack(preprocessed).to(self.device)
        with torch.no_grad():
            embeddings = self.encode_image(image_batch)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def text_embeddings(self, words, normalize_embeddings=False):
        """Get text embeddings. Returns (N, D) tensor."""
        tokens = torch.cat([self.preprocess_text(w) for w in words]).to(self.device)
        with torch.no_grad():
            embeddings = self.encode_text(tokens)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def multimodal_embeddings(self, image_embeddings, text_embeddings):
        """Get multimodal embeddings: by default, averages image and text embeddings"""
        return [(a + b) / 2 for a, b in zip(image_embeddings, text_embeddings)]
    
    def _resolve_logit_scale(self, logit_scale=None):
        if logit_scale is not None:
            return logit_scale
        if hasattr(self.model, "logit_scale"):
            return self.model.logit_scale.exp().item()
        return 100

    def text_to_images_logits(self, image_embeddings, text_embeddings, logit_scale=None):
        """Get logits of text to image embedding dot products"""
        scale = self._resolve_logit_scale(logit_scale)
        return scale * image_embeddings @ text_embeddings.t()

    def text_to_images_similarity(self, image_embeddings, text_embedding, logit_scale=None):
        if isinstance(image_embeddings, list):
            image_embeddings = torch.stack(image_embeddings)
        logits = self.text_to_images_logits(image_embeddings, text_embedding, logit_scale).to(self.device)
        softmaxes = torch.nn.functional.softmax(logits, dim=0)
        return softmaxes[1][0].item()
    
    def multimodal_luce(self, image_embeddings, text_embedding):
        target_similarity = self.similarity(image_embeddings[0], text_embedding)
        distractor_similarity = self.similarity(image_embeddings[1], text_embedding)
        luce = distractor_similarity / (distractor_similarity + target_similarity)
        return luce
    
    def make_processor_transform(self):
        """Return the CLIP preprocess transform for use in DataLoader workers."""
        return self.preprocess

