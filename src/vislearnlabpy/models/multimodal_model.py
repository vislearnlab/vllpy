import itertools
from vislearnlabpy.models.feature_generator import FeatureGenerator
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

class MultimodalModel(FeatureGenerator):
    """Abstract base class for multimodal models like CLIP and CVCL that extends FeatureGenerator"""
    
    def __init__(self, model, preprocess, dataloader=None, device=None):
        super().__init__(model, preprocess, dataloader, device)
        print(self.dataloader)
        self.image_word_alignment = lambda **x: self.model(**x).logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # Load and preprocess images
    def preprocess_image(self, image):
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def preprocess_text(self, text):
        return self.model.tokenize(text).to(self.device)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def image_embeddings(self, images, normalize_embeddings=True):
        """Get image embeddings"""
        images = [self.preprocess_image(image) for image in images]
        with torch.no_grad():
            embeddings = [self.encode_image(image) for image in images]
        if normalize_embeddings:
            return self.normalize_embeddings(embeddings)
        else:
            return embeddings

    def text_embeddings(self, words, normalize_embeddings=True):
        """Get text embeddings"""
        all_text_features = [self.preprocess_text(word) for word in words]
        with torch.no_grad():
            embeddings = [self.encode_text(text_features) for text_features in all_text_features]
        if normalize_embeddings:
            return self.normalize_embeddings(embeddings)
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
    
    def save_embedding(self, embedding, curr_id, text, save_path):
        # if curr_id represents the full path of the image, portion off only the last part
        if os.path.exists(curr_id):
            sub_save_path = f"{text}/{Path(curr_id).stem}"
        else:
            sub_save_path = curr_id
        embedding_output_path = Path(f"{os.getcwd()}/{save_path}/{sub_save_path}").with_suffix('.npy')
        os.makedirs(embedding_output_path.parent, exist_ok=True)
        np_embedding = embedding.cpu().numpy()
        np.save(str(embedding_output_path), np_embedding)
        return str(embedding_output_path)
    
    def save_text_embeddings(self, texts, save_path, normalize_embeddings=True, output_type="csv", overwrite=False):
        filename = f"{self.name}_text_embeddings_{output_type}.csv"
        text_save_path = os.path.join(str(save_path), "text_embeddings")
        os.makedirs(text_save_path, exist_ok=True)
        filepath = os.path.join(text_save_path, filename)
        existing_row_ids = []
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            existing_row_ids = existing_df['row_id'].values
        # TODO: repeated code -- move embedding code to embeddings folder
        with torch.no_grad():
            row_data = []
            for text in tqdm(texts, desc="Calculating text embeddings"):
                if text not in existing_row_ids or overwrite:
                    curr_text_embeddings = self.text_embeddings([text], normalize_embeddings)[0]
                    curr_row_data = {'row_id': text}
                    if output_type == "csv":
                        # new row with a separate column for each number in the 512 dimensions and one for the image_path as the row_id
                        curr_image_embeddings = curr_image_embeddings.squeeze(0).tolist() if isinstance(curr_image_embeddings, torch.Tensor) else curr_image_embeddings
                        for i, value in enumerate(curr_image_embeddings):
                            curr_row_data[f"{i}"] = value.item() if isinstance(value, torch.Tensor) else value
                    elif output_type == "npy":
                        curr_row_data["embedding_path"] = self.save_embedding(curr_text_embeddings, text, text, text_save_path)
                    row_data.append(curr_row_data)
        if len(row_data) > 0:
            self.save_df(pd.DataFrame(row_data), filename, text_save_path, overwrite=overwrite)

    def save_image_embeddings(self, save_path=None, normalize_embeddings=True, output_type="csv", overwrite=False):
        filename = f"{self.name}_image_embeddings_{output_type}.csv"
        save_path = str(save_path) or os.path.join(os.getcwd(), "output")
        img_save_path = os.path.join(save_path, "image_embeddings")
        os.makedirs(img_save_path, exist_ok=True)
        filepath = os.path.join(img_save_path, filename)
        existing_row_ids = []
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            existing_row_ids = existing_df['row_id'].values
        all_text = set()
        with torch.no_grad():
            for d in tqdm(self.dataloader, desc=f"Calculating {self.name} embeddings", position=tqdm._get_free_pos()):
                print(d)
                # TODO: only works for single images in a row
                row_data = []
                for count, (image, curr_id, text) in tqdm(enumerate(zip(d['images'], d['id'], d['text'] if d['text'] else itertools.repeat(None, len(d['images'])))), total=len(d['images']), desc="Image embedding progress in current batch", position=tqdm._get_free_pos()):
                    if curr_id not in existing_row_ids or overwrite:
                        curr_image_embeddings = self.image_embeddings([image], normalize_embeddings)[0]
                        curr_row_data = {'row_id': curr_id}
                        if text is not None:
                            curr_row_data['text'] = text
                        if output_type == "csv":
                            # new row with a separate column for each number in the 512 dimensions and one for the image_path as the row_id
                            curr_image_embeddings = curr_image_embeddings.squeeze(0).tolist() if isinstance(curr_image_embeddings, torch.Tensor) else curr_image_embeddings
                            for i, value in enumerate(curr_image_embeddings):
                                curr_row_data[f"{i}"] = value.item() if isinstance(value, torch.Tensor) else value
                        elif output_type == "npy":
                            curr_row_data["embedding_path"] = self.save_embedding(curr_image_embeddings, curr_id, text, img_save_path)
                        row_data.append(curr_row_data)
                    if text is not None:
                        all_text.add(text)
                if len(row_data) > 0:
                    self.save_df(pd.DataFrame(row_data), filename, img_save_path, overwrite=overwrite)
            self.save_text_embeddings(all_text, save_path, normalize_embeddings, output_type, overwrite)
        
    def normalize_embeddings(self, embeddings):
        """Normalize embeddings to unit L2 norm"""
        return [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in embeddings]
    
