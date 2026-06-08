from abc import ABC
import torch
from torch.nn.functional import cosine_similarity


class FeatureGenerator(ABC):
    """Abstract base class for vision and vision-language models.

    Subclasses implement ``image_embeddings()`` and, optionally, ``text_embeddings()``.
    Use ``EmbeddingGenerator`` + ``EmbeddingStore`` to generate and retrieve similarities.
    """
    supports_text: bool = False

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings produced by this model. Override in subclasses."""
        return 512

    @property
    def text_embedding_dim(self) -> int:
        """Dimensionality of text embeddings. Defaults to embedding_dim; override when they differ."""
        return self.embedding_dim

    def __init__(self, model, preprocess, dataloader=None, device=None, name="feature_generator"):
        torch.set_num_threads(64)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.name = name
        self.dataloader = dataloader

    def similarity(self, embeddings1, embeddings2):
        """Cosine similarity between two embedding tensors (scalar)."""
        return cosine_similarity(embeddings1, embeddings2).item()

