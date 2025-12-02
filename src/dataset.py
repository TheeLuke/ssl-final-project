import os
import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder, ConcatDataset

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    Crucial for SSL to prevent the model from learning high-frequency artifacts.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.prob:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    Inverts pixel values above a threshold. This forces the model to ignore 
    simple color statistics and focus on shape.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale=(0.5, 1.0), local_crops_scale=(0.15, 0.5), local_crops_number=6):
        """
        Args:
            global_crops_scale: Scale range of the image for global views.
            local_crops_scale: Scale range of the image for local views.
            local_crops_number: Number of local crops to generate.
        """
        # -------------------------------------------------------------------------
        # TODO: RUN compute_stats.py AND UPDATE THESE VALUES
        # These are standard ImageNet means. Your mixed dataset (Nature+Scenes+Objects)
        # will likely have different statistics.
        # -------------------------------------------------------------------------
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        
        # Base augmentations (flip, jitter, gray)
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # -------------------------------------------------------------------------
        # Transformation 1: First Global Crop (Simple)
        # -------------------------------------------------------------------------
        self.global_transfo1 = transforms.Compose([
            # We resize to 96x96 because that is the strict project constraint
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.0), # Always blur to avoid trivial texture matching
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        # -------------------------------------------------------------------------
        # Transformation 2: Second Global Crop (Harder)
        # -------------------------------------------------------------------------
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1), # Less blur
            Solarization(p=0.2), # Apply solarization here
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        # -------------------------------------------------------------------------
        # Transformation 3: Local Crops (Zoomed In)
        # -------------------------------------------------------------------------
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        # Generate Global Crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        
        # Generate Local Crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
            
        return crops

def load_combined_dataset(data_path_provided, data_path_external):
    """
    Loads and concatenates the provided pretraining data with the external data.
    
    Args:
        data_path_provided (str): Path to the provided 'pretrain/' folder.
        data_path_external (str): Path to the 'external_data/' folder.
        
    Returns:
        dataset: A PyTorch Dataset object returning (crops, label).
    """
    transform = DataAugmentationDINO()
    
    datasets = []
    
    # 1. Load Provided Data
    if os.path.exists(data_path_provided):
        print(f"Loading provided data from: {data_path_provided}")
        ds_provided = ImageFolder(data_path_provided, transform=transform)
        datasets.append(ds_provided)
    else:
        print(f"WARNING: Provided data path not found: {data_path_provided}")

    # 2. Load External Data (Birds, Scenes, Objects)
    if os.path.exists(data_path_external):
        print(f"Loading external data from: {data_path_external}")
        # ImageFolder will automatically find subfolders (nature_birds, scenes, objects)
        # and assign them dummy class IDs (which we ignore in SSL)
        ds_external = ImageFolder(data_path_external, transform=transform)
        datasets.append(ds_external)
    else:
        print(f"WARNING: External data path not found: {data_path_external}")

    if not datasets:
        raise RuntimeError("No datasets found! Check your paths.")

    # Combine them
    full_dataset = ConcatDataset(datasets)
    print(f"Total training images: {len(full_dataset)}")
    
    return full_dataset