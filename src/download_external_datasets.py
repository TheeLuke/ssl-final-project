"""
External Dataset Downloader
================================================
Sources unlabeled data to match the test set domains:
1. iNaturalist (Nature/Birds)
2. Places365 (Scenes)
3. COCO (Objects)

All images are resized to 96x96 on the fly to save disk space.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Please run: pip install datasets")
    exit(1)

def process_and_save(image, save_path, resolution=96):
    """Resizes and saves a single image."""
    try:
        # Convert to RGB (handle Grayscale or RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize using BICUBIC for quality
        image = image.resize((resolution, resolution), Image.BICUBIC)
        
        # Save
        image.save(save_path, quality=90, optimize=True)
        return True
    except Exception as e:
        return False

def download_domain(dataset_name, subset, split, output_dir, target_count, domain_label):
    """Streams data from Hugging Face and saves it locally."""
    print(f"\n[START] Downloading {domain_label} from {dataset_name}...")
    
    save_dir = Path(output_dir) / domain_label
    save_dir.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(save_dir.glob("*.jpg")))
    if existing >= target_count:
        print(f"  - Found {existing} images. Skipping download.")
        return

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  - Error loading dataset config: {e}")
        return

    count = 0
    pbar = tqdm(total=target_count, desc=f"  - Saving {domain_label}")
    
    # Iterate through the stream
    for sample in ds:
        if count >= target_count:
            break
            
        # Handle different dataset structures
        # Most HF vision datasets have an 'image' key
        if 'image' in sample:
            img = sample['image']
        elif 'file_name' in sample and isinstance(sample['file_name'], str):
            continue
        else:
            continue

        # Generate filename
        filename = f"{domain_label}_{count:06d}.jpg"
        save_path = save_dir / filename
        
        # Save
        success = process_and_save(img, save_path)
        
        if success:
            count += 1
            pbar.update(1)
            
    pbar.close()
    print(f"[DONE] Saved {count} images for {domain_label}")

def main():
    parser = argparse.ArgumentParser(description="Download external data for SSL Project")
    parser.add_argument('--output_dir', type=str, default='./external_data', 
                        help='Where to save the images')
    parser.add_argument('--n_birds', type=int, default=100000, 
                        help='Number of nature/bird images (iNaturalist)')
    parser.add_argument('--n_scenes', type=int, default=100000, 
                        help='Number of scene images (Places365)')
    parser.add_argument('--n_objects', type=int, default=100000, 
                        help='Number of object images (COCO)')
    args = parser.parse_args()

    print("="*60)
    print(" EXTERNAL DATA PREPARATION STRATEGY")
    print("="*60)
    print("Targeting 3 domains to match Leaderboard Test Sets:")
    print(" 1. Nature/Birds (for CUB-200)")
    print(" 2. Scenes/Interiors (for SUN397)")
    print(" 3. General Objects (for Mini-ImageNet)")
    print("-" * 60)

    # 1. DOWNLOAD NATURE/BIRDS (Source: iNaturalist Mini)
    # iNaturalist matches the domain of CUB-200 perfectly without being the same dataset.
    download_domain(
        dataset_name="pcuenq/inaturalist-2021-mini", 
        subset=None, 
        split="train", 
        output_dir=args.output_dir, 
        target_count=args.n_birds, 
        domain_label="nature_birds"
    )

    # 2. DOWNLOAD SCENES (Source: Places365)
    # Places365 matches the domain of SUN397.
    # We use a small optimized HF mirror for speed.
    download_domain(
        dataset_name="timm/places365", # Reliable source via timm
        subset=None, 
        split="train", 
        output_dir=args.output_dir, 
        target_count=args.n_scenes, 
        domain_label="scenes"
    )

    # 3. DOWNLOAD OBJECTS (Source: COCO 2017 Unlabeled)
    # COCO provides general objects similar to Mini-ImageNet but is NOT ImageNet.
    download_domain(
        dataset_name="merve/coco2017", 
        subset="unlabeled", # Use the unlabeled set (larger)
        split="train", 
        output_dir=args.output_dir, 
        target_count=args.n_objects, 
        domain_label="objects"
    )

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print(f"Data saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()