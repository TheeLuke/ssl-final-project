"""
Compute Dataset Statistics
==========================
Calculates the specific Mean and Std.
"""

import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

def compute_mean_std(loader):
    """
    Computes mean and std in a single pass using the Welford's online algorithm 
    or simple accumulation for stability.
    """
    mean = 0.
    std = 0.
    total_images_count = 0
    
    print("Computing Mean and Std... (this may take a few minutes)")
    
    for images, _ in tqdm(loader):
        # Shape: [Batch, Channels, Height, Width]
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        
        # Accumulate sum of pixel values per channel
        mean += images.mean(2).sum(0)
        
        # Accumulate sum of squared pixel values per channel
        std += images.std(2).sum(0)
        
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def main():
    parser = argparse.ArgumentParser(description='Calculate Mean/Std for custom SSL dataset')
    parser.add_argument('--provided_dir', type=str, default='./pretrain',
                        help='Path to the provided unlabeled dataset')
    parser.add_argument('--external_dir', type=str, default='./external_data',
                        help='Path to the downloaded external dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for calculation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()

    # 1. Define a simple transform
    # We only resize to 96x96 and convert to tensor.
    # NO augmentation (blur/jitter) should be used for stats calculation.
    basic_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    datasets_list = []

    # 2. Load Provided Data
    if os.path.exists(args.provided_dir):
        print(f"Adding provided data from: {args.provided_dir}")
        try:
            ds_provided = datasets.ImageFolder(args.provided_dir, transform=basic_transform)
            datasets_list.append(ds_provided)
        except Exception as e:
            print(f"  Note: Could not load provided dir (might be empty or wrong path): {e}")

    # 3. Load External Data
    if os.path.exists(args.external_dir):
        print(f"Adding external data from: {args.external_dir}")
        try:
            ds_external = datasets.ImageFolder(args.external_dir, transform=basic_transform)
            datasets_list.append(ds_external)
        except Exception as e:
            print(f"  Note: Could not load external dir: {e}")

    if not datasets_list:
        print("Error: No datasets found. Check your paths.")
        print(f"Provided: {args.provided_dir}")
        print(f"External: {args.external_dir}")
        return

    # 4. Combine
    full_dataset = ConcatDataset(datasets_list)
    loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Total images to process: {len(full_dataset)}")

    # 5. Compute
    if len(full_dataset) > 0:
        mean, std = compute_mean_std(loader)
        
        print("\n" + "="*40)
        print("COMPUTED STATISTICS")
        print("="*40)
        print(f"Mean: {tuple(mean.tolist())}")
        print(f"Std:  {tuple(std.tolist())}")
        print("="*40)
        print("\nACTION REQUIRED:")
        print("Copy these values into 'dataset.py' inside the 'DataAugmentationDINO' class.")
        print(f"self.mean = {tuple(mean.tolist())}")
        print(f"self.std = {tuple(std.tolist())}")
    else:
        print("No images found.")

if __name__ == "__main__":
    main()