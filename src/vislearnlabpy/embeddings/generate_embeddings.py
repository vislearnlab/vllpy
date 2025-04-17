from vislearnlabpy.models.clip_model import CLIPGenerator
import torch
import os
from vislearnlabpy.embeddings.stimuli_loader import StimuliLoader
import pandas as pd

torch.set_num_threads(32)

class EmbeddingGenerator():
    def __init__(self, device=None, model_type="clip"):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_type = model_type
    
    def generate_image_embeddings(self, output_path=os.getcwd(), overwrite=False, 
                                  normalize=True, output_type="csv", input_csv=None, 
                                  input_dir=None, batch_size=1):
        if input_csv is None:
            if input_dir is None:
                raise Exception("Either input CSV or input image path needs to be provided")
            images_dataloader = StimuliLoader(
                image_folder=input_dir,
                batch_size=batch_size,
                stimuli_type="images"
            ).dataloader()
        else:
            images_dataloader = StimuliLoader(
                image_folder=input_dir,
                dataset_file=input_csv,
                batch_size=batch_size,
                stimuli_type="images",
                id_column="image1"
            ).dataloader()
        CLIPGenerator(device=self.device, dataloader=images_dataloader).save_image_embeddings(save_path=output_path,
                                                                                       output_type=output_type, 
                                                                                       normalize_embeddings=normalize, 
                                                                                       overwrite=overwrite,
                                                                                       )
