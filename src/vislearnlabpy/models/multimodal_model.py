import itertools
from vislearnlabpy.models.feature_generator import FeatureGenerator
from vislearnlabpy.embeddings import utils
from torchvision import transforms
import torch

class MultimodalModel(FeatureGenerator):
    """Abstract base class for multimodal models like CLIP and CVCL that extends FeatureGenerator"""
    
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
        """Get image embeddings (batched for speed)"""
        # Handle single image case
        if not isinstance(images, list):
            return self.image_embeddings([images], normalize_embeddings)[0]
        # Preprocess all images → batch
        preprocessed_images = [self.preprocess_image(image) for image in images]
        preprocessed_images = [image.squeeze(0) if image.dim() == 4 else image for image in preprocessed_images]
        # Stack into a single tensor batch (assuming tensors are returned)
        image_batch = torch.stack(preprocessed_images).to(self.device)
        with torch.no_grad():
            embeddings = self.encode_image(image_batch)  # model handles batch
        if normalize_embeddings:
            embeddings = utils.normalize_embeddings(embeddings)
        return embeddings

    def text_embeddings(self, words, normalize_embeddings=False):
        """Get text embeddings"""
        all_text_features = [self.preprocess_text(word) for word in words]
        with torch.no_grad():
            embeddings = [self.encode_text(text_features) for text_features in all_text_features]
        if normalize_embeddings:
            return utils.normalize_embeddings(embeddings)
        else:
            return embeddings

    def multimodal_embeddings(self, image_embeddings, text_embeddings):
        """Get multimodal embeddings: by default, averages image and text embeddings"""
        return [(a + b) / 2 for a, b in zip(image_embeddings, text_embeddings)]
    
    def text_to_images_logits(self, image_embeddings, text_embeddings, logit_scale=100):
        """Get logits of text to image embedding dot products"""
        return logit_scale * image_embeddings @ text_embeddings.t()
    
    def text_to_images_similarity(self, image_embeddings, text_embedding, logit_scale=100):
        # Convert image_embeddings list to tensor if needed
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

    def similarities(self, word1, word2, images):
        valid_images = [img for img in images if img is not None]
        similarity_scores = []
        # TODO: this only returns the similarity scores for the first pair of images: need to separate out, indexing is weird
        for image1, image2 in itertools.combinations(valid_images, 2):
            curr_image_embeddings = self.image_embeddings([image1, image2])
            curr_text_embeddings = self.text_embeddings([word1, word2])
            # TODO: need to fix how each row is labeled in lookit_similarities, 
            similarity_scores.append({
                'image_similarity': self.similarity(curr_image_embeddings[0], curr_image_embeddings[1]),
                'text_similarity': self.similarity(curr_text_embeddings[0], curr_text_embeddings[1]),
                # finding distractor image to target word similarity
                'multimodal_similarity': self.text_to_images_similarity(curr_image_embeddings, curr_text_embeddings[0], logit_scale=10),
            })
        if similarity_scores == []:
            print(f"skipping {word1} and {word2} since they do not have valid images")
            return [{
                'image_similarity': None,
                'text_similarity': None,
                'multimodal_similarity': None
            }]
        else:
            return similarity_scores
        
    # TODO: probably move this to the dataloader row level instead of to a pair of words within a dataloader row
    # TODO: words or texts? what is my parameter
    def embeddings(self, word1, word2, dataloader_row):
        valid_images = [img for img in dataloader_row['images'] if img is not None]
        output_embeddings = []
        for image1, image2 in itertools.combinations(valid_images, 2):
            curr_image_embeddings = self.image_embeddings([image1, image2])
            curr_text_embeddings = self.text_embeddings([word1, word2])
            output_embeddings.append({
                'image_embeddings': curr_image_embeddings,
                'text_embeddings': curr_text_embeddings,
                'multimodal_embeddings': self.multimodal_embeddings(curr_image_embeddings, curr_text_embeddings)
            })
        return output_embeddings
    
