from dataclasses import dataclass
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
from scipy import ndimage
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage import img_as_bool

random.seed(2)


@dataclass
class ImgExtractionSettings:
    resize_dim: int = 256
    crop_dim: int = 224
    apply_content_crop: bool = True
    apply_center_crop: bool = False
    use_thumbnail: bool = False
    change_stroke_color: bool = False
    stroke_color: tuple = (0, 0, 0)
    stroke_threshold: int = 200
    bg_threshold: int = 200  # threshold to consider a pixel as background (0-255)
    bg_component_size: int = 10  # minimum size of connected components to keep (in pixels)
    filter_edge_artifacts: bool = False  # whether to filter out edge artifacts during cropping
    normalize_stroke_thickness: bool = False
    stroke_target_thickness: int = 2  # target thickness for stroke normalization
    double_resize: bool = False  # whether to resize twice (once with thumbnail, once after with resize func)

class ImageExtractor:
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
    def change_stroke_color(img, target_color=(0, 0, 0), threshold=200):
        """
        Change the color of strokes/dark pixels in an image.
        
        Args:
            img -- PIL Image object
            target_color -- Tuple (r, g, b) for the new stroke color
            threshold -- Brightness threshold to identify strokes (0-255)
                        Pixels darker than this are considered strokes
        
        Returns:
            PIL Image with recolored strokes
        """
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Calculate brightness (average of RGB channels)
        brightness = img_array[:, :, :3].mean(axis=2)
        
        # Create mask for stroke pixels (darker than threshold)
        stroke_mask = brightness < threshold
        
        # Apply new color to stroke pixels while preserving alpha
        result = img_array.copy()
        result[stroke_mask, 0] = target_color[0]  # R
        result[stroke_mask, 1] = target_color[1]  # G
        result[stroke_mask, 2] = target_color[2]  # B
        
        return Image.fromarray(result)
    
    def normalize_stroke_thickness(img, target_thickness=2):
        # Binarize
        arr = np.array(img.convert('L'))
        binary = arr < 128  # assuming dark strokes on light background
        
        # Skeletonize to 1-pixel lines
        skeleton = skeletonize(binary)
        
        # Dilate to consistent thickness
        if target_thickness > 1:
            skeleton = binary_dilation(skeleton, disk(target_thickness//2))
        
        # Convert back to image (white background, black strokes)
        result = (~skeleton * 255).astype(np.uint8)
        return Image.fromarray(result)

    @staticmethod
    def _filter_edge_artifacts(large_components, labeled, component_sizes, img_width, img_height):
        for comp_id in range(1, len(component_sizes)):
            if large_components[comp_id]:
                comp_mask = (labeled == comp_id)
                rows, cols = np.where(comp_mask)
                
                if len(rows) > 0 or len(cols) > 0:
                    ylb, yub = min(rows), max(rows)
                    xlb, xub = min(cols), max(cols)
                    width = xub - xlb + 1
                    height = yub - ylb + 1
                    x_tolerance = 10
                    y_tolerance = 100
                    touches_top = ylb <= y_tolerance
                    touches_bottom = yub >= img_height - 1 - y_tolerance
                    touches_left = xlb <= x_tolerance
                    touches_right = xub >= img_width - 1 - x_tolerance
                    # Check if it's a thin strip spanning nearly the entire edge
                    # vertical strip is more likely to not be image artifact
                    is_thin_vertical_strip = (touches_top or touches_bottom and width < img_width * 0.2)
                    is_thin_horizontal_strip = (touches_left and touches_right and height < img_height * 0.2)
                    if is_thin_vertical_strip or is_thin_horizontal_strip or (width / height >= 10) or (height / width >= 10):
                        large_components[comp_id] = False
        return large_components

    @staticmethod
    def crop_to_content(img, apply_content_crop, stroke_threshold=200, min_component_size=10, filter_edge_artifacts=False):
        """Crop image to remove white space around content, ignoring small artifacts."""
        if not apply_content_crop:
            return img
            
        arr = np.asarray(img)
        img_width, img_height = img.size
        # Create binary mask of non-white pixels
        mask = np.all(arr < stroke_threshold, axis=2) if arr.ndim == 3 else arr < stroke_threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(mask)
        
        # Find component sizes
        component_sizes = np.bincount(labeled.ravel())
        # Component 0 is background, so start from 1
        large_components = component_sizes >= min_component_size
        large_components[0] = False  # Don't include background
        # Filter out edge artifacts (components that span entire width or height)
        if filter_edge_artifacts:
            large_components = ImageExtractor._filter_edge_artifacts(large_components, labeled, component_sizes, img_width, img_height)

        # Create cleaned mask with only large components
        cleaned_mask = large_components[labeled]
        if arr.ndim == 3:
            white_img = np.ones_like(arr) * 255
        else:
            white_img = np.ones_like(arr) * 255
        
        # Copy only the content pixels (keep background white)
        white_img[cleaned_mask] = arr[cleaned_mask]
        img = Image.fromarray(white_img.astype(np.uint8)) 
        try:
            rows, cols = np.where(cleaned_mask)
            ylb = min(rows)  # top bound
            yub = max(rows)  # bottom bound
            xlb = min(cols)  # left bound
            xub = max(cols)  # right bound
            width = xub - xlb  
            height = yub - ylb  
            max_dim = max(width, height) 

            x_center = (xlb + xub) // 2 
            y_center = (ylb + yub) // 2  

            lb_x = x_center - max_dim // 2
            ub_x = x_center + max_dim // 2
            lb_y = y_center - max_dim // 2
            ub_y = y_center + max_dim // 2

            # Shift if out of bounds (to maintain square)
            if lb_x < 0:
                ub_x -= lb_x
                lb_x = 0
            if lb_y < 0:
                ub_y -= lb_y
                lb_y = 0
            if ub_x > img.size[0]:
                lb_x -= (ub_x - img.size[0])
                ub_x = img.size[0]
            if ub_y > img.size[1]:
                lb_y -= (ub_y - img.size[1])
                ub_y = img.size[1]

            img = img.crop((lb_x, lb_y, ub_x, ub_y))
        except ValueError:
            print('Blank image - skipping crop')
        return img
    
    @staticmethod
    def get_transformations(settings: ImgExtractionSettings=ImgExtractionSettings()):
        """Load image transformations for dataloader.
        
        Args:
            settings: ImgExtractionSettings dataclass instance
        """
        
        def combined_transform(image):
            # Step 1: Apply thumbnail resizing if requested
            if settings.use_thumbnail:
                image.thumbnail((settings.resize_dim, settings.resize_dim), Image.Resampling.LANCZOS)
                image.thumbnail((settings.crop_dim, settings.crop_dim), Image.Resampling.LANCZOS)
            # Step 2: Convert RGBA to RGB
            img_rgb = ImageExtractor.RGBA2RGB(image)

            # Step 3: Normalize stroke thickness if enabled
            if settings.normalize_stroke_thickness:
                img_rgb = ImageExtractor.normalize_stroke_thickness(
                    img_rgb, 
                    settings.stroke_target_thickness
                )
            # change stroke color if enabled
            if settings.change_stroke_color:
                img_rgb = ImageExtractor.change_stroke_color(
                    img_rgb, 
                    settings.stroke_color, 
                    settings.stroke_threshold
                )
            # Step 4: Crop to content (remove whitespace) if enabled
            img_cropped = ImageExtractor.crop_to_content(img_rgb, settings.apply_content_crop, settings.bg_threshold, settings.bg_component_size, settings.filter_edge_artifacts)
            return img_cropped
        
        # Build transformation pipeline
        transform_list = []
        if settings.apply_center_crop:
            transform_list.append(transforms.CenterCrop(settings.crop_dim))
        transform_list.append(transforms.Lambda(combined_transform))

        # Only add regular resize if not using thumbnail
        if not settings.use_thumbnail or settings.double_resize:
            # resize last to ensure highest res
            transform_list.append(transforms.Resize(settings.resize_dim))

        # Add any additional transforms you might need
        #transform_list.extend([
            #transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        #])
        
        return transforms.Compose(transform_list)

    @staticmethod
    def save_transformed(original_file, new_file, settings: ImgExtractionSettings = None, transform=None):
        """
        Load, transform, and save an image.
        
        Args:
            original_file: Path to input image
            new_file: Path to save output image
            settings: ImgExtractionSettings dataclass instance (if None, uses default)
            transform: Custom transform (if provided, overrides settings)
        """
        if settings is None:
            settings = ImgExtractionSettings()
            
        if transform is None:
            transform = ImageExtractor.get_transformations(settings)
        
        image_data = Image.open(original_file)
        img = transform(image_data)
        img.save(new_file)

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
                    try:
                        if self.transform is not None:
                            img_copy = self.transform(img_copy)
                        else:
                            img_copy = img_copy.convert('RGB')
                    except Exception as e:
                        print(f"Warning: failed to process image {idx} ({e}); skipping transform.")
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
