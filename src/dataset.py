import os
import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset

# --- Augmentations (Same as before) ---
class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.prob:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        return img

class Solarization(object):
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale=(0.5, 1.0), local_crops_scale=(0.15, 0.5), local_crops_number=6):
        # UPDATE THESE WITH YOUR compute_stats.py RESULTS
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
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
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

# --- NEW: Lazy Dataset (Loads only file paths) ---
class LazyImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.file_paths = []
        
        print(f"Scanning files in {os.path.basename(root)}...", flush=True)
        # Fast walk to just get file paths (Low Memory)
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.file_paths.append(os.path.join(dirpath, f))
        
        print(f"Found {len(self.file_paths)} images. (Lazy Loading Enabled)", flush=True)

    def __getitem__(self, index):
        # Load image from disk ON DEMAND
        path = self.file_paths[index]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception:
            # Simple fallback for corrupt images: return a black image
            img = Image.new('RGB', (96, 96))
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, 0 

    def __len__(self):
        return len(self.file_paths)

def load_combined_dataset(data_path_provided, data_path_external):
    transform = DataAugmentationDINO()
    datasets = []
    
    if os.path.exists(data_path_provided):
        datasets.append(LazyImageDataset(data_path_provided, transform=transform))
    
    if os.path.exists(data_path_external):
        datasets.append(LazyImageDataset(data_path_external, transform=transform))

    if not datasets:
        raise RuntimeError("No datasets found!")

    full_dataset = ConcatDataset(datasets)
    print(f"Total training images: {len(full_dataset)}", flush=True)
    
    return full_dataset