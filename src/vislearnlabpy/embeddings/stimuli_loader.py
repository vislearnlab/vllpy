import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from vislearnlabpy.embeddings.utils import process_csv
import random

random.seed(2)

class StimuliDataset(Dataset):
  def __init__(self, manifest, images_folder=None, id_column=None):
    self.manifest = manifest
    self.images_folder = images_folder
    self.num_text_cols = sum(1 for c in self.manifest.columns if re.compile("text[0-9]").match(c))
    self.num_image_cols = len([c for c in self.manifest.columns if re.compile("image[0-9]").match(c)])
    self.id_column = id_column

  def __len__(self):
     return len(self.manifest)

  def __getitem__(self, idx):
    row = self.manifest.iloc[idx]
    texts = [str(row[f"text{i}"]) for i in range(1, self.num_text_cols + 1)]
    images = []
    image_paths = self._get_image_paths(row)
    full_image_paths = []
    for image_path in image_paths:
      if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)
      if image_path is None or not os.path.exists(image_path):
        full_image_paths.append(None)
        images.append(None)
      else:
        with Image.open(image_path).convert('RGB') as img:
          full_image_paths.append(image_path)
          images.append(img.copy())  # Copy the image data to memory
    # just assuming the image path as an item id for now if there is more than one item per row.
    # TODO: allow for the provision of both item ids and row ids
    row_id = [row[self.id_column] if self.id_column is not None else random.randint(0, 1000000)]
    return {"images": images, "text": texts, "item_id": full_image_paths if len(images) > 0 else row_id, "row_id": row_id}

  def _get_image_paths(self, row):
    # If this is a text only dataset
    if self.images_folder is None and self.num_image_cols == 0:
      return [None] * self.num_text_cols
    # If images are not in the manifest
    elif self.num_image_cols == 0:
      return [os.path.join(self.images_folder, row[f"text{i}"] + ".jpg") for i in range(1, self.num_text_cols + 1)]
    else:
      # allowing for a secondary image path to be provided with different images in the same stimuli sets stored in different subpaths
      return [
    os.path.join(*(self.images_folder,) if self.images_folder is not None else (), *(row["image_path"],) if "image_path" in row else (), row[f"image{i}"])
    for i in range(1, self.num_image_cols + 1)]

class StimuliLoader():
  def __init__(self, dataset_file=None, batch_size=1, image_folder=None, id_column=None, stimuli_type='lookit', pairwise=False):
    self.id_column = id_column
    if dataset_file is None:
      # If only an image directory is provided
      if image_folder is not None:
        image_files = [
          f for f in os.listdir(image_folder)
          if (
          os.path.isfile(os.path.join(image_folder, f)) and
          not f.startswith("._") and
          f.lower().endswith((".png", ".jpg", ".jpeg"))
          )
        ]
        self.manifest = pd.DataFrame({'image1': image_files})
        # assuming that the image path is a unique identifier if a specific ID column is not provided
        self.id_column = "image1" if id_column is None else id_column
      else: 
        raise ValueError("Either image folder or dataset file needs to be provided")
    else:
      self.manifest = process_csv(dataset_file)
    self.stimuli_type = stimuli_type
    self.image_folder = image_folder
    self.batch_size = batch_size
    self.pairwise = pairwise
  
  def collator(self, batch):
    return {key: [item for ex in batch for item in ex[key]] for key in batch[0]}
     
  def dataloader(self):
    dataset = StimuliDataset(self.manifest, self.image_folder, self.id_column)
    return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collator)
# TODO: allow for calculating all possible pairwise similarities in a space to do RSA etc.
