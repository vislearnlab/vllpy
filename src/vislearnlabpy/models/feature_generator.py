from abc import ABC, abstractmethod
import itertools
import pandas as pd
import os
import torch
import random

from tqdm import tqdm
from torch.nn.functional import cosine_similarity
random.seed(4)

class FeatureGenerator(ABC):
    """Abstract base class for generating similarity scores between pairs of concepts."""
    def __init__(self, model, preprocess, dataloader, device=None, name="feature_generator"):
        # working with tversky CPU capacity
        torch.set_num_threads(1)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.name = name
        self.dataloader = dataloader

    def similarity(self, embeddings1, embeddings2):
        # Compute cosine similarity
        similarity_score = cosine_similarity(embeddings1, embeddings2)
        return similarity_score.item()

    # Normalize each embedding to have a unit L2 norm
    def normalize_embeddings(self, embeddings):
        return [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in embeddings]

    def save_similarities(self, sim_df: pd.DataFrame, save_path=None):
        """Save similarity scores to CSV."""
        self.save_df(sim_df, f'similarities-{self.name}_data.csv', save_path)

    @abstractmethod
    def similarities(self, stimulus1, stimulus2, dataloader_row):
        """Compute similarity scores between two stimuli"""
        pass

    def format_similarity_row(self, word1, word2, similarity_score):
        return {'target': word1, 'distractor': word2, **similarity_score}

    def lookit_similarities(self, save_path=None):
        """Calculate cosine similarities between all word pairs in the Lookit dataset"""
        full_similarity_df = pd.DataFrame()
        with torch.no_grad():
            for d in tqdm(self.dataloader, desc=f"Calculating {self.name} similarities"):
                similarity_data = []
                # need to fix placing based on the stimuli set
                text_set = set()
                for count, (text1, text2) in enumerate(itertools.permutations(d['text'], 2)):
                    images = [d['images'][d['text'].index(text1)], d['images'][d['text'].index(text2)]]
                    pair = tuple(([text1, text2]))
                    if pair not in text_set:
                        curr_similarities = self.similarities(text1, text2, images)[0]
                        curr_similarities["stimuli_id"] = d['id'][0]
                        curr_similarities["row_id"] = f"{curr_similarities["stimuli_id"]}_{count}"
                        similarity_data.append(self.format_similarity_row(text1, text2, curr_similarities))
                        text_set.add(pair)
                if len(similarity_data) > 0:
                    similarity_df = pd.DataFrame(similarity_data)
                    # Fix concatenation to handle empty DataFrames properly
                    if full_similarity_df.empty:
                        full_similarity_df = similarity_df
                    else:
                        full_similarity_df = pd.concat([full_similarity_df, similarity_df], ignore_index=True)
                    self.save_similarities(similarity_df, save_path)
        return full_similarity_df

    def image_word_alignment(self, images, words):
        """Compute alignment between a set of images and a list of words"""
        inputs = self.preprocess(images=images, text=words, return_tensors="pt", padding=True)
        return self.image_word_alignment(**inputs)
    
    def save_df(self, df, filename, save_path=None, overwrite=False):
        """
        Save dataframe to CSV, appending to existing file if it exists.
        Avoids duplicate row_ids and handles new directory creation.
        """
        filepath = os.path.join(save_path, filename)
        # create directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
            return
        try:
            existing_df = pd.read_csv(filepath)
            row_ids = df['row_id'].values
            if overwrite:
                # Remove rows from existing data that would be overwritten by new data
                existing_df = existing_df[~existing_df['row_id'].isin(row_ids)]
                existing_df.to_csv(filepath, index=False)             
            else:
                df = df[~df['row_id'].isin(existing_df['row_id'].values)]   
            df.to_csv(filepath, mode='a', header=False, index=False)      
        except pd.errors.EmptyDataError:
            # Handle case where existing file is empty
            df.to_csv(filepath, index=False)
