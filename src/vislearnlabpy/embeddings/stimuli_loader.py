import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from vislearnlabpy.embeddings.utils import process_csv
import random
from typing import Callable, Any
import warnings
from torchvision import transforms
import numpy as np

random.seed(2)

class ImageExtractor:
    def __init__(self):
        pass
    
    @staticmethod
    def RGBA2RGB(img, background_color=(255, 255, 255)):
        """Alpha composite an RGBA Image with a specified color.
        Source: http://stackoverflow.com/a/9459208/284318

        Keyword Arguments:
        img -- PIL RGBA Image object
        background_color -- Tuple r, g, b (default 255, 255, 255)
        """
        if img.mode in ('RGBA', 'LA'):  # Image has transparency
            # Create white background
            white_bg = Image.new('RGB', img.size, background_color)
            # Paste the image onto white background
            # The image itself acts as the alpha mask
            white_bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = white_bg
        elif img.mode == 'P':  # Palette mode, might have transparency
            img = img.convert('RGBA')
            white_bg = Image.new('RGB', img.size, background_color)
            white_bg.paste(img, mask=img.split()[-1])
            img = white_bg
        else:
            # Image is already RGB/L, convert to RGB if needed
            img = img.convert('RGB')
        return img

    @staticmethod
    def crop_to_content(img, apply_content_crop):
        """Crop image to remove white space around content."""
        if not apply_content_crop:
            return img
            
        arr = np.asarray(img)
        rows, cols, channels = np.where(arr < 255)
        
        try:
            xlb = min(cols)  # left bound
            xub = max(cols)  # right bound  
            ylb = min(rows)  # top bound
            yub = max(rows)  # bottom bound
            
            # Make it square by using the same bounds
            lb = min([xlb, ylb])
            ub = max([xub, yub])
            
            img = img.crop((lb, lb, ub, ub))
        except ValueError:
            print('Blank image - skipping crop')
        return img
    
    @staticmethod
    def get_transformations(resize_dim=256, crop_dim=224, apply_content_crop=True, apply_center_crop=False, use_thumbnail=False):
        """Load image transformations for dataloader.
        
        Args:
            resize_dim: Dimension for resizing (default 256)
            crop_dim: Dimension for center crop (default 224)
            apply_content_crop: Whether to crop to content (default True)
            apply_center_crop: Whether to apply center crop (default True)
            use_thumbnail: Whether to use thumbnail resizing instead of regular resize (default False)
        """
        
        def combined_transform(image):
            # Step 1: Apply thumbnail resizing if requested
            if use_thumbnail:
                image.thumbnail((resize_dim, resize_dim), Image.Resampling.LANCZOS)

            # Step 2: Convert RGBA to RGB
            img_rgb = ImageExtractor.RGBA2RGB(image)
            
            # Step 3: Crop to content (remove whitespace) if enabled
            img_cropped = ImageExtractor.crop_to_content(img_rgb, apply_content_crop)
            return img_cropped
        
        # Build transformation pipeline
        transform_list = []
        
        # Only add regular resize if not using thumbnail
        if not use_thumbnail:
            # resize first
            transform_list.append(transforms.Resize(resize_dim))

        transform_list.append(transforms.Lambda(combined_transform))
        
        if apply_center_crop:
            transform_list.append(transforms.CenterCrop(crop_dim))
        
        # Add any additional transforms you might need
        #transform_list.extend([
            #transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        #])
        
        return transforms.Compose(transform_list)

class StimuliDataset(Dataset):
    def __init__(self, manifest, images_folder=None, id_column=None, transform=None):
        self.manifest = manifest
        self.images_folder = images_folder
        self.num_text_cols = sum(1 for c in self.manifest.columns if re.compile("text[0-9]").match(c))
        self.num_image_cols = len([c for c in self.manifest.columns if re.compile("image[0-9]").match(c)])
        self.id_column = id_column
        self.transform = transform

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
                with Image.open(image_path) as img:
                    img_copy = img.copy()  # Copy the image data to memory
                    
                    if self.transform is not None:
                        # Apply the transform pipeline
                        img_copy = self.transform(img_copy)
                    else:
                        # Fallback to just converting to RGB
                        img_copy = img_copy.convert('RGB')
                    
                    full_image_paths.append(image_path)
                    images.append(img_copy)
        
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
    def __init__(self, dataset_file=None, batch_size=1, image_folder=None, id_column=None, 
                 stimuli_type='lookit', pairwise=False, transform=None):
        self.id_column = id_column
        self.transform = transform
        
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
        dataset = StimuliDataset(self.manifest, self.image_folder, self.id_column, self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collator)
