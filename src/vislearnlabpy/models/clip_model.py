import clip
import torch
from vislearnlabpy.models.multimodal_model import MultimodalModel
from vislearnlabpy.embeddings import utils

class CLIPGenerator(MultimodalModel):
    def __init__(self, dataloader=None, device=None, text_prompt="a photo of a "):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        super().__init__(self.model, self.preprocess, dataloader, device)
        self.name = "clip"
        self.text_prompt = text_prompt

    def preprocess_text(self, text):
        return clip.tokenize(f"{self.text_prompt}{text}").to(self.device)


class OpenCLIPGenerator(MultimodalModel):
    """OpenCLIP wrapper supporting pre-loaded models or loading by name/checkpoint.

    Args:
        model:           Pre-loaded open_clip model. If None, one is created.
        preprocess:      Preprocessing transform matching the model. Required if model is given.
        model_name:      Architecture name (e.g. 'ViT-H-14'). Used when model is None
                         or to retrieve the correct tokenizer.
        pretrained:      OpenCLIP pretrained tag (e.g. 'laion2b_s32b_b79k'). Used when
                         model is None and checkpoint_path is None.
        checkpoint_path: Path to a .pt checkpoint file. Takes precedence over pretrained.
        text_prompt:     Prefix prepended to labels when generating text embeddings.
        dataloader:      Optional StimuliLoader dataloader.
        device:          Device string. None -> auto-detect.
        epoch:           Optional epoch number appended to the generator name (useful
                         when iterating over training checkpoints).
    """

    def __init__(self, model=None, preprocess=None, model_name='ViT-B-32',
                 pretrained='laion2b_s34b_b79k', checkpoint_path=None,
                 text_prompt="a photo of a ", dataloader=None, device=None, epoch=None):
        import open_clip

        if model is None:
            if checkpoint_path is not None:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=checkpoint_path, load_weights_only=False)
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained)

        self._tokenizer = open_clip.get_tokenizer(model_name)
        super().__init__(model, preprocess, dataloader, device)
        self.text_prompt = text_prompt
        self._model_name = model_name
        name = f"openclip_{model_name.replace('/', '_').replace('-', '_')}"
        if epoch is not None:
            name = f"{name}_epoch{epoch}"
        self.name = name

    def preprocess_text(self, text):
        # Returns (1, seq_len) to match the shape expected by MultimodalModel.text_embeddings
        return self._tokenizer([f"{self.text_prompt}{text}"]).to(self.device)

    def text_embeddings(self, words, normalize_embeddings=False):
        prompted = [f"{self.text_prompt}{w}" for w in words]
        tokens = self._tokenizer(prompted).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def make_processor_transform(self):
        return self.preprocess
