import clip
from vislearnlabpy.models.multimodal_model import MultimodalModel

class CLIPGenerator(MultimodalModel):
    def __init__(self, dataloader=None, device=None, text_prompt="a photo of a "):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        super().__init__(self.model, self.preprocess, dataloader, device)
        self.name = "clip"
        self.text_prompt = text_prompt
    
    def preprocess_text(self, text):
        return clip.tokenize(f"{self.text_prompt}{text}").to(self.device)
    