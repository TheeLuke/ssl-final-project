#imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class SSLTransform:
    """
    Generates two augmented views of the same image.
    Required for Contrastive Learning (SimCLR, MoCo, etc.) and DINO.
    """
    def __init__(self, size=96):
        # Base strong augmentation
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # Normalize using ImageNet stats (standard practice)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        # Return two different views of the image
        q = self.transform(x)
        k = self.transform(x)
        return [q, k]

def get_eval_transform(size=96):
    """
    Standard transform for k-NN evaluation (No random augmentation).
    """
    return transforms.Compose([
        transforms.Resize(size), # Resize to 96px
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(dataset_name='cifar10', batch_size=256, num_workers=4):
    """
    Returns:
        train_loader: For SSL Pretraining (returns list of 2 views)
        memory_loader: For k-NN feature bank (returns single image + label)
        test_loader: For k-NN inference (returns single image + label)
    """
    
    print(f"Loading dataset: {dataset_name}...")
    
    # --- A. Sanity Check Mode (CIFAR-10) ---
    if dataset_name == 'cifar10':
        # Note: We UPSAMPLE CIFAR from 32->96 to match competition specs
        
        # 1. Pretraining Set (Unlabeled / Labels ignored)
        train_dataset = datasets.CIFAR10(
            root='data', train=True, download=True, 
            transform=SSLTransform(size=96)
        )
        
        # 2. Memory Bank Set (Train images, no augmentation)
        memory_dataset = datasets.CIFAR10(
            root='data', train=True, download=True, 
            transform=get_eval_transform(size=96)
        )
        
        # 3. Test Set (Test images)
        test_dataset = datasets.CIFAR10(
            root='data', train=False, download=True, 
            transform=get_eval_transform(size=96)
        )

    # --- B. Competition Mode (Hugging Face) ---
    elif dataset_name == 'project_data':
        import os
        
        memory_dataset = None
        test_dataset = None
        
        # Path to where you unzipped the images
        # Ensure this path matches where you ran the unzip command
        traindir = '../data/cc3m_all' 
        
        if not os.path.exists(traindir):
            raise FileNotFoundError(f"Cannot find training data at {traindir}")

        # ImageFolder expects structure like: root/class_x/image.png
        # If your unzip dumped all images into one folder without subfolders,
        # we might need a custom dataset. 
        # Assuming standard ImageFolder structure (root/images/...):
        
        # If the unzip created a flat folder of images, we need a custom class:
        from glob import glob
        from PIL import Image

        class FlatFolderDataset(Dataset):
            def __init__(self, root, transform=None, cache_file="data_cache.pth"):
                self.root = root
                self.transform = transform
                
                # Absolute path for the cache file so it saves in the root
                cache_path = os.path.abspath(cache_file)
                
                # 1. Check if cache exists
                if os.path.exists(cache_path):
                    print(f"Loading cached file list from {cache_path}...")
                    # This takes 0.5 seconds instead of 10 minutes
                    self.files = torch.load(cache_path)
                    
                # 2. If not, scan and save
                else:
                    print(f"Scanning {root}... (This will take time, but only once!)")
                    self.files = []
                    # Scan for images
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        self.files.extend(glob(os.path.join(root, "**", ext), recursive=True))
                    
                    self.files.sort()
                    
                    if len(self.files) > 0:
                        print(f"Saving cache to {cache_path}...")
                        torch.save(self.files, cache_path)
                    else:
                        raise FileNotFoundError(f"No images found in {root}")

                print(f"Dataset size: {len(self.files)}")

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                try:
                    img = Image.open(self.files[idx]).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    return img, 0 
                except Exception as e:
                    # Skip corrupted images instead of crashing
                    print(f"Error loading {self.files[idx]}: {e}")
                    return torch.zeros(3, 96, 96), 0

        train_dataset = FlatFolderDataset(traindir, transform=SSLTransform(size=96))
        
        # (Eval logic remains to be implemented once public test is released)
        memory_loader = None
        test_loader = None

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    
    # Memory/Test loaders are only needed if memory_dataset is defined
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers) if memory_dataset else None
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers) if test_dataset else None

    return train_loader, memory_loader, test_loader

if __name__ == "__main__":
    train_dl, _, _ = get_dataloaders(dataset_name='cifar10', batch_size=4)
    
    images, _ = next(iter(train_dl))
    
    # images is a list [view1, view2]
    print(f"View 1 Shape: {images[0].shape}") # Should be [4, 3, 96, 96]
    print(f"View 2 Shape: {images[1].shape}") # Should be [4, 3, 96, 96]
    
    if images[0].shape[-1] == 96:
        print("SUCCESS: Images are correctly sized to 96x96.")
    else:
        print("FAIL: Image resolution incorrect.")